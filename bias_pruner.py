"""Easy/hard sets, activation-variance scores, stage masks, transfer, and maxoutput."""
from dataclasses import dataclass
import torch

@dataclass
class Partition:
    easy: torch.Tensor
    hard: torch.Tensor

def identify_easy_hard(logits,targets,tau=.70):
    probs=logits.softmax(1); confidence,pred=probs.max(1)
    p_y=probs.gather(1,targets.long().view(-1,1)).squeeze(1)
    easy=(pred==targets)&(p_y>=tau)
    # The paper's prose calls all misclassified examples hard; its displayed
    # set expression is ambiguous about whether confidence also constrains H.
    hard=pred!=targets
    return Partition(easy,hard)

@torch.no_grad()
def score_channels(model,loader,task,device,num_classes,tau=.70,class_map=None):
    """For every bottleneck conv1/conv2/output unit, average class-wise variance gaps."""
    sums_e={k:torch.zeros(num_classes,n,device=device) for k,n in model.unit_channels.items()}
    sums_h={k:torch.zeros_like(v) for k,v in sums_e.items()}
    counts_e=torch.zeros(num_classes,device=device); counts_h=torch.zeros_like(counts_e)
    model.eval()
    for images,targets,_,_,_ in loader:
        images=images.to(device,non_blocking=True); targets=targets.to(device).long()
        if class_map is not None: targets=torch.tensor([class_map[int(y)] for y in targets.tolist()],device=device)
        model.collect_activations=True
        try:
            logits=model(images,task)
            partition=identify_easy_hard(logits,targets,tau)
            variances={k:v.var(dim=(-2,-1),unbiased=False) for k,v in model.last_activations.items()}
            for c in range(num_classes):
                class_mask=targets==c; em=class_mask&partition.easy; hm=class_mask&partition.hard
                if em.any():
                    counts_e[c]+=em.sum()
                    for k in model.unit_channels: sums_e[k][c]+=variances[k][em].sum(0)
                if hm.any():
                    counts_h[c]+=hm.sum()
                    for k in model.unit_channels: sums_h[k][c]+=variances[k][hm].sum(0)
        finally: model.collect_activations=False
    valid=(counts_e>0)&(counts_h>0)
    if not valid.any(): raise RuntimeError("Bias scores require at least one class with both easy and hard examples")
    return {k:((sums_e[k][valid]/counts_e[valid,None])-(sums_h[k][valid]/counts_h[valid,None])).mean(0) for k in model.unit_channels}

def prune_channels(scores,gamma=.60):
    """Remove top gamma fraction across all bottleneck units, retaining one per tensor."""
    if not 0<=gamma<1: raise ValueError("gamma must be in [0,1)")
    if not scores or any(not torch.isfinite(v).all() for v in scores.values()): raise ValueError("bias scores must be finite and nonempty")
    total=sum(v.numel() for v in scores.values()); target=min(round(gamma*total),sum(v.numel()-1 for v in scores.values()))
    entries=sorted(((float(score),stage,index) for stage,values in scores.items() for index,score in enumerate(values.detach().cpu().tolist())),reverse=True)
    removed={s:torch.zeros(v.numel(),dtype=torch.bool,device=v.device) for s,v in scores.items()}
    used={s:0 for s in scores}
    for _,stage,index in entries:
        if sum(used.values())>=target: break
        if used[stage]<removed[stage].numel()-1:
            removed[stage][index]=True; used[stage]+=1
    masks={s:(~removed[s]).float() for s in scores}
    return masks,removed

def freeze_previous_subnetworks(model,previous_masks):
    """Freeze previous units; later stages may update only genuinely free output filters."""
    for p in model.backbone.parameters(): p.requires_grad_(False)
    for head in model.heads.values():
        for p in head.parameters(): p.requires_grad_(False)
    if not previous_masks:
        for p in model.backbone.parameters(): p.requires_grad_(True)
        model._free_unit_masks={k:torch.ones(n,dtype=torch.bool) for k,n in model.unit_channels.items()}
        return {k:m.float() for k,m in model._free_unit_masks.items()}
    free={k:~torch.stack([m[k].bool() for m in previous_masks]).any(0) for k in model.unit_channels}
    model._free_unit_masks=free
    def hook_for(unit_name):
        def masked(grad):
            active=model._free_unit_masks[unit_name].to(device=grad.device,dtype=grad.dtype)
            return grad*active.reshape((-1,)+(1,)*(grad.ndim-1))
        return masked
    stem_conv=model.backbone.conv1; stem_bn=model.backbone.bn1
    stem_conv.weight.requires_grad_(True); stem_bn.weight.requires_grad_(True); stem_bn.bias.requires_grad_(True)
    if not getattr(stem_conv,"_biaspruner_hook",False):
        stem_conv.weight.register_hook(hook_for("stem")); stem_bn.weight.register_hook(hook_for("stem")); stem_bn.bias.register_hook(hook_for("stem"))
        stem_conv._biaspruner_hook=True
    for key,block in model._unit_modules.items():
        for unit,conv,bn in ((f"{key}.conv1",block.conv1,block.bn1),
                             (f"{key}.conv2",block.conv2,block.bn2),
                             (f"{key}.out",block.conv3,block.bn3)):
            conv.weight.requires_grad_(True); bn.weight.requires_grad_(True); bn.bias.requires_grad_(True)
            if not getattr(conv,"_biaspruner_hook",False):
                conv.weight.register_hook(hook_for(unit)); bn.weight.register_hook(hook_for(unit)); bn.bias.register_hook(hook_for(unit))
                if conv.bias is not None: conv.bias.register_hook(hook_for(unit))
                conv._biaspruner_hook=True
            if unit.endswith(".out") and block.downsample is not None:
                downconv=block.downsample[0]; downbn=block.downsample[1]
                downconv.weight.requires_grad_(True); downbn.weight.requires_grad_(True); downbn.bias.requires_grad_(True)
                if not getattr(downconv,"_biaspruner_hook",False):
                    downconv.weight.register_hook(hook_for(unit)); downbn.weight.register_hook(hook_for(unit)); downbn.bias.register_hook(hook_for(unit))
                    downconv._biaspruner_hook=True
    return {s:m.float() for s,m in free.items()}

def protected_snapshot(model,previous_masks):
    """Copy all protected conv/BN slices and prior heads for invariant assertions."""
    snapshot={}
    if not previous_masks: return snapshot
    occupied={k:torch.stack([m[k].bool() for m in previous_masks]).any(0) for k in model.unit_channels}
    mask=occupied["stem"]; conv=model.backbone.conv1; bn=model.backbone.bn1
    for label,param in (("weight",conv.weight),("bn_weight",bn.weight),("bn_bias",bn.bias),
                        ("running_mean",bn.running_mean),("running_var",bn.running_var)):
        snapshot[f"stem.{label}"]=param.detach()[mask.to(param.device)].clone()
    for block_key,block in model._unit_modules.items():
        for suffix,conv,bn in (("conv1",block.conv1,block.bn1),("conv2",block.conv2,block.bn2),("out",block.conv3,block.bn3)):
            mask=occupied[f"{block_key}.{suffix}"]
            for label,param in (("weight",conv.weight),("bn_weight",bn.weight),("bn_bias",bn.bias),
                                ("running_mean",bn.running_mean),("running_var",bn.running_var)):
                snapshot[f"{block_key}.{suffix}.{label}"]=param.detach()[mask.to(param.device)].clone()
        if block.downsample is not None:
            mask=occupied[f"{block_key}.out"]; conv,bn=block.downsample
            for label,param in (("weight",conv.weight),("bn_weight",bn.weight),("bn_bias",bn.bias),
                                ("running_mean",bn.running_mean),("running_var",bn.running_var)):
                snapshot[f"{block_key}.downsample.{label}"]=param.detach()[mask.to(param.device)].clone()
    for name in range(len(previous_masks)):
        for key,value in model.heads[str(name)].state_dict().items(): snapshot[f"head{name}.{key}"]=value.detach().clone()
    return snapshot

def assert_protected_unchanged(model,snapshot,previous_masks):
    if not snapshot: return
    occupied={k:torch.stack([m[k].bool() for m in previous_masks]).any(0) for k in model.unit_channels}
    mask=occupied["stem"]; conv=model.backbone.conv1; bn=model.backbone.bn1
    for label,param in (("weight",conv.weight),("bn_weight",bn.weight),("bn_bias",bn.bias),
                        ("running_mean",bn.running_mean),("running_var",bn.running_var)):
        assert torch.equal(param.detach()[mask.to(param.device)],snapshot[f"stem.{label}"].to(param.device)),f"protected stem {label} changed"
    for block_key,block in model._unit_modules.items():
        for suffix,conv,bn in (("conv1",block.conv1,block.bn1),("conv2",block.conv2,block.bn2),("out",block.conv3,block.bn3)):
            mask=occupied[f"{block_key}.{suffix}"]
            params=(("weight",conv.weight),("bn_weight",bn.weight),("bn_bias",bn.bias),
                    ("running_mean",bn.running_mean),("running_var",bn.running_var))
            for label,param in params:
                assert torch.equal(param.detach()[mask.to(param.device)],snapshot[f"{block_key}.{suffix}.{label}"].to(param.device)),f"protected {block_key}.{suffix} {label} changed"
        if block.downsample is not None:
            mask=occupied[f"{block_key}.out"]; conv,bn=block.downsample
            for label,param in (("weight",conv.weight),("bn_weight",bn.weight),("bn_bias",bn.bias),
                                ("running_mean",bn.running_mean),("running_var",bn.running_var)):
                assert torch.equal(param.detach()[mask.to(param.device)],snapshot[f"{block_key}.downsample.{label}"].to(param.device)),f"protected {block_key}.downsample {label} changed"
    for name in range(len(previous_masks)):
        for key,value in model.heads[str(name)].state_dict().items():
            assert torch.equal(value.detach(),snapshot[f"head{name}.{key}"].to(value.device)),f"protected head {name} changed"

def max_output_inference(model,images,task_masks,device):
    """Select max-output task independently per image and return global HAM IDs."""
    outputs=[model(images,task,mask) for task,mask in enumerate(task_masks)]
    responses=torch.stack([logits.max(1).values for logits in outputs],dim=0) # [task,batch]
    selected=responses.argmax(0)
    global_pred=torch.empty(images.shape[0],dtype=torch.long,device=device)
    for task,logits in enumerate(outputs):
        rows=torch.where(selected==task)[0]
        if rows.numel():
            local=logits[rows].argmax(1); labels=model.task_classes[str(task)]
            mapping=torch.tensor(labels,dtype=torch.long,device=device)
            global_pred[rows]=mapping[local]
    return global_pred,selected,outputs
