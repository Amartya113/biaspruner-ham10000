"""Sequential BiasPruner training and configurable random task-order experiments."""
import argparse,json,random
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from config import Config
from dataset import HAM_CLASSES,load_ham_frame,split_lesions,make_loaders
from models import BiasPrunerNet
from losses import generalized_cross_entropy,WeightedCrossEntropy
from bias_pruner import (score_channels,prune_channels,freeze_previous_subnetworks,
                         protected_snapshot,assert_protected_unchanged,max_output_inference)
from metrics import evaluate_continual_predictions
from utils import set_seed,ensure_dirs,append_jsonl,save_checkpoint

def _local_targets(y,global_classes):
    mapping={g:i for i,g in enumerate(global_classes)}
    unknown=set(y.tolist())-set(mapping)
    if unknown: raise ValueError(f"labels not in this task mapping: {sorted(unknown)}")
    return torch.tensor([mapping[int(v)] for v in y.tolist()],device=y.device)

@torch.no_grad()
def _gce_validation_accuracy(model,loader,task,classes,device,q):
    model.eval(); ys=[]; ps=[]
    for x,y,_,_,_ in loader:
        logits=model(x.to(device),task)
        local=logits.argmax(1).cpu()
        ys.extend(y.tolist()); ps.extend([classes[i] for i in local.tolist()])
    return float(accuracy_score(ys,ps))

@torch.no_grad()
def _validation_metrics(model,loader,task_masks,device,task_classes):
    model.eval(); ys=[]; ps=[]; ages=[]
    for images,y,a,_,_ in loader:
        pred,_,_=max_output_inference(model,images.to(device),task_masks,device)
        ys.extend(y.tolist()); ps.extend(pred.cpu().tolist()); ages.extend(a.tolist())
    return evaluate_continual_predictions(ys,ps,ages,task_classes)

def _run_epoch(model,loader,task,classes,device,optimizer,loss_fn,scaler,amp,mask=None,gce_table=None):
    model.train(); total=count=0
    if getattr(model,"freeze_bn_stats",False):
        for module in model.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
    for x,y,_,sample_indices,_ in loader:
        x=x.to(device,non_blocking=True); y=_local_targets(y,classes)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=amp and device.type=="cuda"):
            logits=model(x,task,mask)
            if gce_table is None: loss=loss_fn(logits,y)
            else: loss=loss_fn(logits,y,gce_table[sample_indices.to(device)])
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total+=float(loss.detach())*len(y); count+=len(y)
    return total/max(count,1)

def _selection_key(metrics,priority):
    acc=metrics["balanced_accuracy"]; eod=metrics["eod"]
    eod_score=-eod if np.isfinite(eod) else -float("inf")
    return (acc,eod_score) if priority=="balanced_accuracy_then_eod" else (eod_score,acc)

@torch.no_grad()
def _cache_training_gce(model,loader,task,classes,device,q):
    """Cache first-stage per-image GCE values, as used by the authors' WCE class."""
    model.eval(); values=torch.zeros(len(loader.dataset),device=device)
    for x,y,_,indices,_ in loader:
        x=x.to(device,non_blocking=True); y=_local_targets(y,classes)
        per_image=generalized_cross_entropy(model(x,task),y,q,reduction="none")
        values[indices.to(device)]=per_image
    return values

def _run_single(cfg,output_dir):
    cfg=replace(cfg,output_dir=str(output_dir)); cfg.validate(); set_seed(cfg.seed)
    root=ensure_dirs(cfg.output_dir); cfg.device=str(torch.device(cfg.device if cfg.device=="cpu" or torch.cuda.is_available() else "cpu"))
    device=torch.device(cfg.device); cfg.save(root/"logs"/"config.json")
    frame=load_ham_frame(cfg.data_root,cfg.metadata_csv,cfg.image_dirs)
    splits=split_lesions(frame,cfg.seed,cfg.train_fraction,cfg.val_fraction)
    loaders=make_loaders(splits,cfg.task_order,cfg.image_size,cfg.batch_size,cfg.workers)
    groups=[tuple(cfg.task_order[:2]),tuple(cfg.task_order[2:4]),tuple(cfg.task_order[4:])]
    global_classes=[[HAM_CLASSES.index(c) for c in group] for group in groups]
    assert [len(group) for group in global_classes]==[2,2,3]
    model=BiasPrunerNet(global_classes,cfg.pretrained).to(device)
    masks=[]; log=root/"logs"/"training.jsonl"
    for task,classes in enumerate(global_classes):
        model.freeze_bn_stats=task>0
        free=freeze_previous_subnetworks(model,masks)
        snapshot=protected_snapshot(model,masks)
        for p in model.heads[str(task)].parameters(): p.requires_grad_(True)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg.learning_rate)
        scaler=torch.amp.GradScaler("cuda",enabled=cfg.amp and device.type=="cuda")
        best_acc=-1.; wait=0; best_state=None
        for epoch in range(cfg.gce_epochs):
            train_loss=_run_epoch(model,loaders["train"][task],task,classes,device,optimizer,
                lambda z,y:generalized_cross_entropy(z,y,cfg.q),scaler,cfg.amp)
            val_acc=_gce_validation_accuracy(model,loaders["val"][task],task,classes,device,cfg.q)
            append_jsonl(log,{"task":task,"stage":"gce","epoch":epoch+1,"train_loss":train_loss,"val_accuracy":val_acc})
            if val_acc>best_acc:
                best_acc=val_acc; wait=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                wait+=1
                if wait>=cfg.patience: break
        if best_state: model.load_state_dict(best_state)
        assert_protected_unchanged(model,snapshot,masks)

        gce_table=_cache_training_gce(model,loaders["train"]["eval"][task],task,classes,device,cfg.q)
        # Score every residual stage's post-ReLU channel maps and globally remove
        # the highest gamma fraction while leaving at least one channel per stage.
        class_map={g:i for i,g in enumerate(classes)}
        scores=score_channels(model,loaders["train"][task],task,device,len(classes),cfg.tau,class_map=class_map)
        current,removed=prune_channels(scores,cfg.gamma)
        assert all(current[s].numel()==model.unit_channels[s] for s in model.unit_channels)
        criterion=WeightedCrossEntropy(cfg.q,cfg.alpha)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg.finetune_lr)
        scaler=torch.amp.GradScaler("cuda",enabled=cfg.amp and device.type=="cuda")
        best_key=None; best_state=None
        masks_for_validation=masks+[current]
        for epoch in range(cfg.finetune_epochs):
            train_loss=_run_epoch(model,loaders["train"][task],task,classes,device,optimizer,criterion,scaler,cfg.amp,current,gce_table)
            val_metrics=_validation_metrics(model,loaders["val"][task],masks_for_validation,device,global_classes)
            key=_selection_key(val_metrics,cfg.checkpoint_priority)
            append_jsonl(log,{"task":task,"stage":"wce","epoch":epoch+1,"train_loss":train_loss,"val_metrics":val_metrics,"alpha":cfg.alpha})
            if best_key is None or key>best_key:
                best_key=key; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best_state: model.load_state_dict(best_state)
        assert_protected_unchanged(model,snapshot,masks)
        masks.append({unit:current[unit].detach().cpu() for unit in model.unit_channels})
        save_checkpoint(root/"checkpoints"/f"task_{task}.pt",model=model.state_dict(),
            task_masks=[{unit:m[unit].tolist() for unit in model.unit_channels} for m in masks],config=cfg.__dict__,task=task)

    save_checkpoint(root/"checkpoints"/"final.pt",model=model.state_dict(),
        task_masks=[{unit:m[unit].tolist() for unit in model.unit_channels} for m in masks],config=cfg.__dict__)
    final_metrics=_validation_metrics(model,loaders["test"]["all"],masks,device,global_classes)
    (root/"logs"/"evaluation.json").write_text(json.dumps(final_metrics,indent=2,allow_nan=True),encoding="utf-8")
    return final_metrics

def run(cfg):
    """Run one explicit task order, preserving compatibility for notebook callers."""
    return _run_single(cfg,cfg.output_dir)

def run_experiments(cfg,num_orders=None):
    """Run 1 debug order or the paper's 3 random orders and summarize mean/std."""
    n=cfg.num_task_orders if num_orders is None else num_orders
    if n<1: raise ValueError("num_orders must be >=1")
    base=Path(cfg.output_dir); base.mkdir(parents=True,exist_ok=True)
    results=[]
    for i in range(n):
        classes=list(cfg.task_order)
        if n>1: random.Random(cfg.seed+i).shuffle(classes)
        order_cfg=replace(cfg,seed=cfg.seed+i,task_order=tuple(classes))
        metrics=_run_single(order_cfg,base/f"order_{i+1:02d}")
        results.append({"order":i+1,"seed":order_cfg.seed,"task_order":classes,"metrics":metrics})
    keys=("accuracy","f1","balanced_accuracy","dpr","eod")
    summary={k:{"mean":float(np.nanmean([r["metrics"][k] for r in results])),
                "std":float(np.nanstd([r["metrics"][k] for r in results],ddof=1 if n>1 else 0))} for k in keys}
    payload={"orders":results,"summary":summary}
    (base/"summary.json").write_text(json.dumps(payload,indent=2,allow_nan=True),encoding="utf-8")
    return payload

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root"); parser.add_argument("--metadata-csv"); parser.add_argument("--image-dirs",nargs="+")
    parser.add_argument("--output-dir"); parser.add_argument("--seed",type=int); parser.add_argument("--gce-epochs",type=int)
    parser.add_argument("--finetune-epochs",type=int); parser.add_argument("--batch-size",type=int)
    parser.add_argument("--orders",type=int); parser.add_argument("--task-order",nargs=7)
    parser.add_argument("--image-size",type=int); parser.add_argument("--workers",type=int)
    parser.add_argument("--alpha",type=float); parser.add_argument("--q",type=float); parser.add_argument("--tau",type=float); parser.add_argument("--gamma",type=float)
    parser.add_argument("--checkpoint-priority",choices=("balanced_accuracy_then_eod","eod_then_balanced_accuracy"))
    parser.add_argument("--no-pretrained",action="store_true"); parser.add_argument("--pretrained",action="store_true")
    args=parser.parse_args(); cfg=Config()
    for key,value in vars(args).items():
        if key=="no_pretrained" and value: cfg.pretrained=False
        elif key=="pretrained" and value: cfg.pretrained=True
        elif key=="task_order" and value is not None: cfg.task_order=tuple(value)
        elif key=="image_dirs" and value is not None: cfg.image_dirs=tuple(value)
        elif key=="orders" and value is not None: cfg.num_task_orders=value
        elif key=="checkpoint_priority" and value is not None: cfg.checkpoint_priority=value
        elif value is not None: setattr(cfg,key.replace("-","_"),value)
    print(json.dumps(run_experiments(cfg),indent=2,allow_nan=True))
if __name__=="__main__": main()
