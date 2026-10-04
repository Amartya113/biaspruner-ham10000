"""ResNet-50 with binary masks on every bottleneck's channel units."""
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

STAGES=("layer1","layer2","layer3","layer4")

class BiasPrunerNet(nn.Module):
    """Fixed-shape masked ResNet-50; masks gate conv1/conv2/out units per block."""
    def __init__(self,task_classes,pretrained=False):
        super().__init__()
        flat=[int(label) for group in task_classes for label in group]
        if len(task_classes)!=3 or [len(group) for group in task_classes]!=[2,2,3]:
            raise ValueError("BiasPruner HAM model requires three task heads with class counts [2,2,3]")
        if sorted(flat)!=list(range(7)): raise ValueError("task heads must map each global HAM class exactly once")
        weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone=resnet50(weights=weights)
        self._feature_channels=self.backbone.fc.in_features
        self.backbone.fc=nn.Identity()
        self.heads=nn.ModuleDict({str(i):nn.Linear(self._feature_channels,len(classes)) for i,classes in enumerate(task_classes)})
        self.task_classes={str(i):tuple(classes) for i,classes in enumerate(task_classes)}
        self.unit_channels={"stem":self.backbone.conv1.out_channels}
        self._unit_modules={}
        for stage in STAGES:
            for block_idx,block in enumerate(getattr(self.backbone,stage)):
                key=f"{stage}.{block_idx}"
                self.unit_channels[f"{key}.conv1"]=block.conv1.out_channels
                self.unit_channels[f"{key}.conv2"]=block.conv2.out_channels
                self.unit_channels[f"{key}.out"]=block.conv3.out_channels
                self._unit_modules[key]=block
        self.last_activations={}
        self.collect_activations=False
        self._free_unit_masks=None
    @property
    def feature_channels(self): return self._feature_channels
    def _apply_unit_mask(self,x,masks,key):
        if self.collect_activations: self.last_activations[key]=x
        if masks is not None:
            if key not in masks or masks[key].numel()!=self.unit_channels[key]:
                raise ValueError(f"invalid channel mask for {key}")
            x=x*masks[key].to(device=x.device,dtype=x.dtype).view(1,-1,1,1)
        return x
    def _bottleneck(self,x,block,key,masks):
        identity=x
        out=block.conv1(x); out=block.bn1(out); out=block.relu(out)
        out=self._apply_unit_mask(out,masks,f"{key}.conv1")
        out=block.conv2(out); out=block.bn2(out); out=block.relu(out)
        out=self._apply_unit_mask(out,masks,f"{key}.conv2")
        out=block.conv3(out); out=block.bn3(out)
        if block.downsample is not None: identity=block.downsample(x)
        out=block.relu(out+identity)
        return self._apply_unit_mask(out,masks,f"{key}.out")
    def features(self,x,masks=None):
        if masks is not None:
            assert set(masks)==set(self.unit_channels),"task mask does not cover all ResNet units"
        self.last_activations={}
        b=self.backbone
        x=b.conv1(x); x=b.bn1(x); x=b.relu(x)
        x=self._apply_unit_mask(x,masks,"stem")
        x=b.maxpool(x)
        for stage in STAGES:
            for idx,block in enumerate(getattr(b,stage)):
                x=self._bottleneck(x,block,f"{stage}.{idx}",masks)
        return torch.flatten(b.avgpool(x),1)
    def forward(self,x,task,masks=None):
        return self.heads[str(task)](self.features(x,masks))
