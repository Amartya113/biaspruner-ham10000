"""HAM10000 metadata, image lookup, lesion-level splits, and class-incremental tasks."""
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
HAM_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
class HAM10000Dataset(Dataset):
    def __init__(self, frame, transform=None):
        self.frame=frame.reset_index(drop=True); self.transform=transform
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        row=self.frame.iloc[i]
        with Image.open(row.path) as im: image=im.convert("RGB")
        if self.transform: image=self.transform(image)
        return image, int(row.label), int(row.age_group), int(i), str(row.image_id)
def _find_metadata(root, explicit=None):
    if explicit: return Path(explicit)
    root=Path(root)
    for p in (root/"HAM10000_metadata.csv", root/"metadata.csv"):
        if p.is_file(): return p
    found=list(root.rglob("HAM10000_metadata.csv"))
    if found: return found[0]
    raise FileNotFoundError(f"HAM metadata CSV not found under {root}; configure metadata_csv")
def load_ham_frame(data_root, metadata_csv=None, image_dirs=()):
    root=Path(data_root); meta=pd.read_csv(_find_metadata(root,metadata_csv))
    missing={"image_id","dx","age","lesion_id"}-set(meta.columns)
    if missing: raise ValueError(f"HAM metadata lacks columns: {sorted(missing)}")
    dirs=[Path(p) for p in image_dirs] if image_dirs else [root]
    lookup={}
    for d in dirs:
        if d.exists():
            for p in d.rglob("*.jpg"): lookup.setdefault(p.stem,p)
    meta["path"]=meta.image_id.map(lambda x:str(lookup.get(str(x),"")))
    if (meta.path=="").any(): raise FileNotFoundError(f"{int((meta.path=='').sum())} images not found; set image_dirs")
    meta["age"]=pd.to_numeric(meta.age,errors="coerce")
    meta["age_group"]=meta.age.map(lambda a:-1 if pd.isna(a) else int(a>=60)).astype(int)
    meta["label"]=pd.Categorical(meta.dx,categories=HAM_CLASSES).codes
    if (meta.label<0).any(): raise ValueError("Unexpected diagnosis label in metadata")
    if not set(meta.age_group.unique()).issubset({-1,0,1}): raise ValueError("Invalid age group value")
    if not set(meta.label.unique()).issubset(set(range(len(HAM_CLASSES)))): raise ValueError("Invalid diagnosis class index")
    return meta
def split_lesions(frame,seed=42,train_fraction=.65,val_fraction=.12):
    """Disjoint lesion_id splits prevent near-duplicate image leakage."""
    from sklearn.model_selection import train_test_split
    lesion=frame[["lesion_id","label"]].drop_duplicates("lesion_id")
    try: train,rest=train_test_split(lesion.lesion_id,train_size=train_fraction,random_state=seed,stratify=lesion.label)
    except ValueError: train,rest=train_test_split(lesion.lesion_id,train_size=train_fraction,random_state=seed)
    fraction=val_fraction/(1-train_fraction); other=lesion[lesion.lesion_id.isin(rest)]
    try: val,test=train_test_split(other.lesion_id,train_size=fraction,random_state=seed+1,stratify=other.label)
    except ValueError: val,test=train_test_split(other.lesion_id,train_size=fraction,random_state=seed+1)
    return {k:frame[frame.lesion_id.isin(v)].copy() for k,v in {"train":train,"val":val,"test":test}.items()}
def task_class_groups(task_order=HAM_CLASSES):
    """Paper specifies 3 tasks with [2,2,3] classes; class order remains configurable."""
    if len(task_order)!=7 or set(task_order)!=set(HAM_CLASSES): raise ValueError("task_order must be a permutation of all seven HAM classes")
    return [tuple(task_order[:2]),tuple(task_order[2:4]),tuple(task_order[4:7])]
def make_loaders(splits,task_order=HAM_CLASSES,image_size=224,batch_size=32,workers=4):
    norm=transforms.Normalize((.485,.456,.406),(.229,.224,.225))
    train_tf=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(15),transforms.ToTensor(),norm])
    eval_tf=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),norm])
    out={}
    for name,frame in splits.items():
        frame=frame.copy(); frame["task"]=-1
        for i,classes in enumerate(task_class_groups(task_order)): frame.loc[frame.dx.isin(classes),"task"]=i
        out[name]={}
        tf=train_tf if name=="train" else eval_tf
        mixed=HAM10000Dataset(frame,tf)
        out[name]["all"]=DataLoader(mixed,batch_size=batch_size,shuffle=name=="train",num_workers=workers,pin_memory=True,persistent_workers=workers>0)
        for i in range(3):
            ds=HAM10000Dataset(frame[frame.task==i],tf)
            out[name][i]=DataLoader(ds,batch_size=batch_size,shuffle=name=="train",num_workers=workers,pin_memory=True,persistent_workers=workers>0)
            if name=="train":
                eval_ds=HAM10000Dataset(frame[frame.task==i],eval_tf)
                out[name].setdefault("eval",{})[i]=DataLoader(eval_ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    return out
