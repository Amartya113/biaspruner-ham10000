"""Quick synthetic pipeline checks; optionally validates an attached HAM dataset."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from config import Config
from dataset import HAM_CLASSES,load_ham_frame,split_lesions,task_class_groups,make_loaders
from models import BiasPrunerNet
from losses import generalized_cross_entropy,WeightedCrossEntropy
from bias_pruner import (score_channels,prune_channels,freeze_previous_subnetworks,
                         protected_snapshot,assert_protected_unchanged,max_output_inference)
from metrics import evaluate_predictions

class _ScoreToy(nn.Module):
    unit_channels={"toy.conv1":4,"toy.conv2":3,"toy.out":5}
    def __init__(self):
        super().__init__()
    def forward(self,x,task,masks=None):
        ids=x[:,0,0,0].long(); y=(ids//2).clamp(max=2); easy=(ids%2)==0
        target=torch.where(easy,y,(y+1)%3)
        logits=torch.full((len(ids),3),-8.,device=x.device)
        logits.scatter_(1,target[:,None],8.)
        activations=(x.flatten(1).mean(1)[:,None,None,None]+torch.arange(1,6,device=x.device)[None,:,None,None])
        self.last_activations={"toy.conv1":activations[:,:4].expand(-1,-1,2,2),
                               "toy.conv2":activations[:,:3].expand(-1,-1,2,2),
                               "toy.out":activations.expand(-1,-1,2,2)}
        return logits

class _InferenceToy:
    task_classes={"0":(0,1),"1":(2,3)}
    def __call__(self,x,task,mask):
        first=x[:,0].long()==0
        logits=torch.zeros(len(x),2)
        if task==0:
            logits[first]=torch.tensor([5.,0.]); logits[~first]=torch.tensor([0.,3.])
        else:
            logits[first]=torch.tensor([0.,4.]); logits[~first]=torch.tensor([7.,0.])
        return logits

def run_sanity(data_root=None,metadata_csv=None,image_dirs=()):
    cfg=Config(pretrained=False,batch_size=2,workers=0)
    cfg.validate()
    groups=task_class_groups(cfg.task_order)
    assert [len(g) for g in groups]==[2,2,3]
    if data_root:
        frame=load_ham_frame(data_root,metadata_csv,image_dirs)
        assert frame.label.between(0,6).all()
        assert set(frame.age_group.unique()).issubset({-1,0,1})
        splits=split_lesions(frame,cfg.seed,cfg.train_fraction,cfg.val_fraction)
        assert not (set(splits["train"].lesion_id)&set(splits["val"].lesion_id))
        assert not (set(splits["train"].lesion_id)&set(splits["test"].lesion_id))
        assert not (set(splits["val"].lesion_id)&set(splits["test"].lesion_id))
        loaders=make_loaders({"train":splits["train"]},cfg.task_order,64,2,0)
        for task in range(3):
            batch=next(iter(loaders["train"][task]))
            assert batch[0].shape[0]>0
            assert set(batch[1].tolist()).issubset(set(HAM_CLASSES.index(c) for c in groups[task]))

    classes=[[HAM_CLASSES.index(c) for c in group] for group in groups]
    model=BiasPrunerNet(classes,pretrained=False).eval()
    masks={key:torch.ones(size) for key,size in model.unit_channels.items()}
    with torch.no_grad():
        for task in range(3):
            logits=model(torch.randn(1,3,64,64),task,masks)
            assert logits.shape==(1,len(classes[task]))
    logits=torch.tensor([[2.,0.],[0.,2.]])
    y=torch.tensor([0,0])
    gce=generalized_cross_entropy(logits,y,.7)
    assert torch.isfinite(gce)
    criterion=WeightedCrossEntropy(.7,.5)
    assert not list(criterion.parameters()),"released-code alpha is fixed, not a trainable parameter"
    wce=criterion(logits,y,torch.tensor([.1,.9]))
    assert torch.isfinite(wce)

    toy=_ScoreToy()
    ids=torch.arange(6).float().view(6,1,1,1)
    labels=torch.tensor([0,0,1,1,2,2])
    loader=[(ids,labels,torch.tensor([0,1,0,1,0,1]),torch.arange(6),[str(i) for i in range(6)])]
    scores=score_channels(toy,loader,0,"cpu",3,.7)
    assert set(scores)==set(toy.unit_channels) and all(v.numel()==toy.unit_channels[k] for k,v in scores.items())
    task_masks,removed=prune_channels(scores,.6)
    assert all(mask.ndim==1 and mask.any() and torch.all((mask==0)|(mask==1)) for mask in task_masks.values())

    prior={key:torch.zeros(size) for key,size in model.unit_channels.items()}
    for value in prior.values(): value[0]=1
    prior_copy={key:value.clone() for key,value in prior.items()}
    free=freeze_previous_subnetworks(model,[prior])
    assert all(torch.equal(free[k].bool(),~prior[k].bool()) for k in prior)
    snapshot=protected_snapshot(model,[prior])
    model.eval()
    for p in model.heads["1"].parameters(): p.requires_grad_(True)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    before={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    prediction=model(torch.randn(2,3,64,64),1,masks)
    nn.functional.cross_entropy(prediction,torch.tensor([0,1])).backward()
    optimizer.step()
    assert any(not torch.equal(before[k],dict(model.named_parameters())[k].detach()) for k in before)
    assert_protected_unchanged(model,snapshot,[prior])
    assert all(torch.equal(prior[k],prior_copy[k]) for k in prior)

    inf_model=_InferenceToy()
    pred,selected,_=max_output_inference(inf_model,torch.tensor([[0.],[1.]]),[None,None],"cpu")
    assert pred.tolist()==[0,2] and selected.tolist()==[0,1]
    assert set(pred.tolist()).issubset(set(range(7)))
    metric=evaluate_predictions([0,1,0,1],[0,1,1,1],[0,0,1,1],positive_class=1)
    assert np.isfinite([metric[k] for k in ("accuracy","f1","balanced_accuracy","dpr","eod")]).all()
    print("Sanity checks passed (synthetic pipeline)." + (" HAM data checks passed." if data_root else " HAM data checks skipped; pass --data-root in Kaggle."))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data-root"); p.add_argument("--metadata-csv"); p.add_argument("--image-dirs",nargs="+")
    a=p.parse_args(); run_sanity(a.data_root,a.metadata_csv,tuple(a.image_dirs or ()))
