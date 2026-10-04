"""BiasPruner losses."""
import torch
from torch import nn
import torch.nn.functional as F
def generalized_cross_entropy(logits, targets, q=0.7, reduction="mean"):
    if q <= 0 or q > 1: raise ValueError("q must be in (0,1]")
    p_y = F.softmax(logits, dim=1).gather(1, targets.long().view(-1, 1)).squeeze(1)
    loss = (1.0 - p_y.pow(q)) / q
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss
class WeightedCrossEntropy(nn.Module):
    """WCE=exp(alpha*GCE)*CE with fixed alpha, matching authors' released WCE class."""
    def __init__(self, q=0.7, alpha=0.5):
        super().__init__()
        if not 0 < alpha < 1: raise ValueError("alpha must be in (0,1)")
        self.q = q
        self.alpha = float(alpha)
    def forward(self, logits, targets, gce_values=None):
        # Authors' released WCE reads cached per-image GCE values from the first stage.
        gce = generalized_cross_entropy(logits, targets, self.q, "none") if gce_values is None else gce_values
        ce = F.cross_entropy(logits, targets.long(), reduction="none")
        return (torch.exp(self.alpha * gce.detach()) * ce).mean()
