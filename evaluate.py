"""Evaluation helpers and command-line evaluation of a trained checkpoint."""
import argparse,json
from pathlib import Path
import torch
from config import Config
from dataset import load_ham_frame,split_lesions,make_loaders,HAM_CLASSES
from metrics import evaluate_continual_predictions
from models import BiasPrunerNet
from bias_pruner import max_output_inference
from utils import set_seed

@torch.no_grad()
def evaluate_model(model,loader,task_masks,device,task_classes):
    model.eval()
    ys=[];ps=[];ages=[]
    for images,y,a,_,_ in loader:
        pred,_,_=max_output_inference(model,images.to(device),task_masks,device)
        ys.extend(y.tolist()); ps.extend(pred.cpu().tolist()); ages.extend(a.tolist())
    return evaluate_continual_predictions(ys,ps,ages,task_classes)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",required=True); parser.add_argument("--data-root")
    args=parser.parse_args(); ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    cfg=Config(**ckpt["config"])
    if args.data_root: cfg.data_root=args.data_root
    set_seed(cfg.seed); device=torch.device(cfg.device if cfg.device=="cpu" or torch.cuda.is_available() else "cpu")
    frame=load_ham_frame(cfg.data_root,cfg.metadata_csv,cfg.image_dirs); splits=split_lesions(frame,cfg.seed,cfg.train_fraction,cfg.val_fraction)
    loaders=make_loaders(splits,cfg.task_order,cfg.image_size,cfg.batch_size,cfg.workers)
    groups=[tuple(cfg.task_order[:2]),tuple(cfg.task_order[2:4]),tuple(cfg.task_order[4:])]
    classes=[[HAM_CLASSES.index(c) for c in g] for g in groups]
    model=BiasPrunerNet(classes,False); model.load_state_dict(ckpt["model"]); model.to(device)
    masks=[{unit:torch.tensor(values) for unit,values in task.items()} for task in ckpt["task_masks"]]
    result={"overall_task_agnostic":evaluate_model(model,loaders["test"]["all"],masks,device,classes)}
    path=Path(cfg.output_dir)/"logs"/"evaluation.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,allow_nan=True))
    print(json.dumps(result,indent=2,allow_nan=True))
if __name__=="__main__": main()
