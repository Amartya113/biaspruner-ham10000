"""Classification and authors-code-compatible binary age-group fairness metrics."""
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score

def evaluate_predictions(y_true,y_pred,age_group,positive_class=1):
    y=np.asarray(y_true); p=np.asarray(y_pred); g=np.asarray(age_group)
    if not set(np.unique(g)).issubset({-1,0,1}): raise ValueError("age_group must use -1 (missing), 0 (<60), or 1 (>=60)")
    weighted_f1=float(f1_score(y,p,average="weighted",zero_division=0))
    out={"accuracy":float(accuracy_score(y,p)),"f1":weighted_f1,"f1_weighted":weighted_f1,
         "balanced_accuracy":float(balanced_accuracy_score(y,p))}
    valid=np.isin(g,[0,1]); y,p,g=y[valid],p[valid],g[valid]
    rates={}; tprs={}
    for group in (0,1):
        m=g==group
        if m.any():
            rates[group]=float(np.mean(p[m]==positive_class))
            denom=np.sum(y[m]==positive_class)
            tprs[group]=float(np.sum((p[m]==positive_class)&(y[m]==positive_class))/denom) if denom else float("nan")
    # Authors' released DPR uses predicted-positive rates and min/max ratio for
    # binary class index 1. Paper calls EOD equal-opportunity difference (TPR gap).
    out["dpr"]=(min(rates.values())/max(rates.values()) if len(rates)==2 and max(rates.values())>0 else
                1.0 if len(rates)==2 else float("nan"))
    out["eod"]=(abs(tprs[0]-tprs[1]) if len(tprs)==2 and np.isfinite(list(tprs.values())).all() else float("nan"))
    out["fairness_positive_class"]=int(positive_class)
    out["group_positive_rates"]={str(k):v for k,v in rates.items()}
    out["group_true_positive_rates"]={str(k):v for k,v in tprs.items()}
    return out

def evaluate_continual_predictions(y_true,y_pred,age_group,task_classes):
    """Overall classification metrics plus authors-style class-1 fairness averaged by task."""
    y=np.asarray(y_true); p=np.asarray(y_pred); age=np.asarray(age_group)
    overall=evaluate_predictions(y,p,age,positive_class=1)
    global_classification={k:overall[k] for k in ("accuracy","f1","balanced_accuracy")}
    per_task={}
    task_values={k:[] for k in ("accuracy","f1","balanced_accuracy","dpr","eod")}
    for task,classes in enumerate(task_classes):
        member=np.isin(y,classes)
        if not member.any(): continue
        to_local={int(label):i for i,label in enumerate(classes)}
        local_y=np.asarray([to_local[int(label)] for label in y[member]])
        # A prediction routed to a different task is not a positive class-1 prediction.
        local_p=np.asarray([to_local.get(int(label),-1) for label in p[member]])
        result=evaluate_predictions(local_y,local_p,age[member],positive_class=1)
        per_task[str(task)]={k:result[k] for k in ("accuracy","f1","balanced_accuracy","dpr","eod")}
        per_task[str(task)]["group_positive_rates"]=result["group_positive_rates"]
        per_task[str(task)]["group_true_positive_rates"]=result["group_true_positive_rates"]
        for key in task_values:
            if np.isfinite(result[key]): task_values[key].append(result[key])
    for key,values in task_values.items():
        if values: overall[key]=float(np.mean(values))
    for key,value in global_classification.items(): overall[f"global_{key}"]=value
    overall["global_class1_group_positive_rates"]=overall.pop("group_positive_rates")
    overall["global_class1_group_true_positive_rates"]=overall.pop("group_true_positive_rates")
    overall["dpr"]=float(np.mean(task_values["dpr"])) if task_values["dpr"] else float("nan")
    overall["eod"]=float(np.mean(task_values["eod"])) if task_values["eod"] else float("nan")
    overall["fairness_aggregation"]="mean across observed tasks; each task uses local class index 1 as positive"
    overall["fairness_by_task"]=per_task
    return overall
