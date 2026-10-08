"""LYX24 round3 frozen small neural matrix; no adaptive search. CLI --prepare or --run."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["OMP_NUM_THREADS"]="4"
os.environ["MKL_NUM_THREADS"]="4"
from pathlib import Path
import json,hashlib,time,sys,random,traceback,tempfile,subprocess,datetime,argparse,gc
import numpy as np,pandas as pd,torch
from torch import nn
torch.set_num_threads(4);torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
O=Path(__file__).resolve().parent;R=O.parents[3]
V2=O.parent.parent/"hf_scene_classification_v2_lyx24"/"execution_20261006_01"
ACT=["HW","RS","HG","TYP","RUN","JJ","BUR","PCH"]
CH=["AccX(g)","AccY(g)","AccZ(g)","GyroX(dps)","GyroY(dps)","GyroZ(dps)","Ut1(mV)","Ut2(mV)"]
SEEDS=[20261006,20261007,20261008]
ARMS={"MIMU":list(range(6)),"HF":[6,7],"FUS":list(range(8))}
CFG=dict(experiment="LYX24_round3_CNN_CNNLSTM_fixed",activities=ACT,channels=CH,seeds=SEEDS,split_seed=20261006,
 models=[a+"_"+b for a in ["CNN","CNNLSTM"] for b in ARMS],main_jobs=432,main_fit_processes=864,aux_jobs=6,aux_fit_processes=12,
 architecture="Conv1d C16 k9s2p4 ReLU,16to24 k7s2p3 ReLU,24to32 k5s2p2 ReLU. T800to400to200to100. CNN temporal mean. CNNLSTM unidirectional one-layer LSTM32to32 then temporal mean. Dropout .1,Linear32to8. No normalization layers.",
 latent_dim=32,optimizer=dict(name="AdamW",lr=.001,weight_decay=.0001,clip_norm=1.),batch_size=64,dtype="float32",amp=False,
 min_epochs=15,max_epochs=40,patience=12,min_delta=0,selection="minimum weighted validation CE over all completed epochs; earliest strict minimum; best epoch can be below15; refit fresh same-seed initialization for selected epochs, no test selection",
 training_weights="1/(8 * train_records_in_class * windows_in_record), normalized to mean1. Uniform randperm windows; batch mean(w*CE). No balanced sampler/no batch-sum normalization.",
 scaler="Weighted global per-channel mean and population variance over training windows and all800 samples; same class-record-window weights; float64 statistics of float32 cached inputs. All8 channels computed together, subset consistently. Zero variance => scale1; no within-window centering.",
 folds="24 complete-record LOO; fixed class-balanced8-record validation,15 inner-training; refit all23. Shared across6configs3seeds.",
 missing="Exact frozen V2 windows; np.interp per-channel within800 rows, nearest edges, allmissing error, no exclusions.",
 augmentation=False,scheduler=False,deterministic=True,tf32=False,cudnn_benchmark=False,device="cuda:0",
 aux="Independent per-class16train8test; within16:8innertrain8val. One seed20261006 all6configs. Reset and refit16. Never reuse primary epochs/scalers.",
 aux_points="For each fixed auxiliary test record: first frozen window then greedily earliest next start>=previous end; shared all6configs. No outcome selection.",
 embeddings=dict(layer="32-d before Dropout/Linear",scale="training16 latent weighted global mean/std only",pca="training16 fit; display test points",tsne=dict(seed=20261006,perplexities=[5,15],init="pca",learning_rate="auto",max_iter=1000),caveat="test-set joint tSNE descriptive; no held-out tSNE generalization claim; no concatenation of independent model spaces"),
 limits=["single-participant posthoc panel","1941 overlapping windows are not independent","date/rewear confounded","classification not heart-rate improvement","all seeds/folds reported; no result-dependent changes"])
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def jw(p,x):
 p=Path(p);t=p.with_name(p.name+".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");os.replace(t,p)
def tsave(p,x):
 p=Path(p);t=p.with_name(p.name+".tmp");torch.save(x,t);os.replace(t,p)
def log(s):
 print(s,flush=True)
 with (O/"training.log").open("a",encoding="utf-8") as f:f.write(utc()+" "+s+"\n")
def seed(n):
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def rng():return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):
 random.setstate(r["py"]);np.random.set_state(r["np"]);torch.set_rng_state(r["cpu"]);torch.cuda.set_rng_state_all(r["cuda"])
class Net(nn.Module):
 def __init__(self,c,arch):
  super().__init__();self.arch=arch
  self.conv=nn.Sequential(nn.Conv1d(c,16,9,1,4),nn.AvgPool1d(2,2),nn.ReLU())
  if arch=="CNNLSTM":self.lstm=nn.LSTM(16,32,1,batch_first=True)
  self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(32 if arch=="CNNLSTM" else 16,8))
 def encode(self,x):
  z=self.conv(x)
  if self.arch=="CNNLSTM":
   _,(h,_)=self.lstm(z.transpose(1,2));return h[-1]
  return z.mean(2)
 def forward(self,x):return self.head(self.encode(x))

def statehash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def weight(meta,indices,mean_one=False):
 g=meta.iloc[indices];nr=g.groupby("activity").record.nunique();nw=g.groupby("record").size()
 assert len(nr)==8
 w=np.array([1/(8*nr[r.activity]*nw[r.record]) for r in g.itertuples()],np.float64)
 assert np.isclose(w.sum(),1)
 return w*len(w) if mean_one else w
def make_stats(x):
 means=x.mean(2,dtype=np.float64)
 second=np.empty_like(means)
 for j in range(8):second[:,j]=np.mean(np.square(x[:,j,:].astype(np.float64)),axis=1)
 return means,second
def scale_for(meta,ids,mom):
 w=weight(meta,ids);second=w@mom[1][ids];scale=np.ones(8);groups=[]
 for cols in [list(range(3)),list(range(3,6)),[6,7]]:
  value=float(np.sqrt(np.mean(second[cols])));fallback=value<=1e-12
  scale[cols]=1. if fallback else value
  groups.append(dict(channels=cols,rms=value,fallback=fallback))
 return dict(mean=np.zeros(8),variance=second,scale=scale,groups=groups,records=sorted(meta.iloc[ids].record.unique().tolist()),indices=np.array(ids,dtype=int),weights=w)

def indices(meta,records):return np.flatnonzero(meta.record.isin(records).to_numpy())
def scaled(x,ids,cols,scaler):
 return torch.from_numpy(np.ascontiguousarray(((x[ids][:,cols,:].astype(np.float64)-scaler["mean"][cols][None,:,None])/scaler["scale"][cols][None,:,None]).astype(np.float32)))
def evaluate(model,x,y,w):
 was=model.training;model.eval();losses=[];out=[];lat=[]
 with torch.no_grad():
  for start in range(0,len(x),64):
   xx=x[start:start+64].to("cuda");z=model.encode(xx);scores=model.head(z)
   ce=nn.functional.cross_entropy(scores,y[start:start+64].to("cuda"),reduction="none")
   losses.extend(ce.cpu().numpy().tolist());out.append(scores.cpu().numpy());lat.append(z.cpu().numpy())
 if was:model.train()
 return float(np.dot(np.array(losses),w)),np.concatenate(out),np.concatenate(lat)
def train_epoch(model,opt,x,y,w):
 model.train();perm=torch.randperm(len(x));total=0.
 for ix in perm.split(64):
  opt.zero_grad(set_to_none=True);score=model(x[ix].to("cuda"))
  loss=(nn.functional.cross_entropy(score,y[ix].to("cuda"),reduction="none")*w[ix].to("cuda")).mean()
  if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
  loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
  if not torch.isfinite(norm):raise RuntimeError("nonfinite gradient")
  opt.step();total+=float(loss.detach())*len(ix)
 return total/len(x)
def unit_tests():
 assert torch.cuda.is_available()
 tests=[]
 for arch in ["CNN","CNNLSTM"]:
  seed(2468);model=Net(8,arch).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  x=torch.randn(128,8,800);y=torch.arange(128)%8;w=torch.ones(128)
  initial=statehash(model);train_epoch(model,opt,x,y,w);assert statehash(model)!=initial
  model.eval()
  with torch.no_grad():
   a=model(x[:4].cuda());b=model(x[:4].cuda());assert torch.equal(a,b)
  model.train();evaluate(model,x[:8],y[:8],np.ones(8)/8);assert model.training
  snapshot=dict(model=model.state_dict(),opt=opt.state_dict(),rng=rng(),identity="synthetic")
  with tempfile.TemporaryDirectory(prefix="lyx24-v3-unit-") as td:
   cp=Path(td)/"checkpoint.pt";tsave(cp,snapshot)
   train_epoch(model,opt,x,y,w);target=statehash(model)
   loaded=torch.load(cp,map_location="cpu",weights_only=False)
   other=Net(8,arch).cuda();op=torch.optim.AdamW(other.parameters(),lr=.001,weight_decay=.0001);other.load_state_dict(loaded["model"]);op.load_state_dict(loaded["opt"]);restore_rng(loaded["rng"])
   train_epoch(other,op,x,y,w);assert statehash(other)==target
  tests.append(arch+": finite gradient/update, eval dropout off, train restoration, exact optimizer/RNG checkpoint continuation")
 tests.extend(preprocess_tests())
 benchmark=[];counts={}
 for arch in ["CNN","CNNLSTM"]:
  for arm,cols in ARMS.items():
   seed(SEEDS[0]);m=Net(len(cols),arch).cuda();op=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
   xx=torch.randn(128,len(cols),800);yy=torch.arange(128)%8;ww=torch.ones(128)
   train_epoch(m,op,xx,yy,ww);torch.cuda.synchronize();t=time.perf_counter()
   for _ in range(3):train_epoch(m,op,xx,yy,ww)
   torch.cuda.synchronize();seconds=(time.perf_counter()-t)/6
   benchmark.append(dict(architecture=arch,arm=arm,seconds_per_batch64=seconds,synthetic=True))
   counts[arch+"_"+arm]=sum(v.numel() for v in m.parameters())
   del m,op
 for arm,cols in ARMS.items():
  seed(123);a=Net(len(cols),"CNN");seed(123);b=Net(len(cols),"CNNLSTM")
  assert all(torch.equal(a.conv.state_dict()[k],v) for k,v in b.conv.state_dict().items())
 return dict(status="PASS",utc=utc(),formal_fits=0,tests=tests,benchmark=benchmark,parameters=counts,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),device="cuda:0")

def prepare():
 if (O/"freeze_receipt.json").exists():raise RuntimeError("Already frozen; do not overwrite")
 assert not (O/"jobs").exists()
 meta=pd.read_csv(V2/"window_metadata.csv");rec=pd.read_csv(V2/"record_manifest_LOCAL.csv")
 assert len(meta)==1941 and len(rec)==24 and rec.groupby("activity").size().eq(3).all()
 assert meta.record.nunique()==24 and (meta.end_row_exclusive-meta.start_row).eq(800).all()
 raw_receipts=[]
 x=np.empty((1941,8,800),np.float64)
 for rr in rec.itertuples():
  p=Path(rr.input_path);assert sha(p)==rr.input_sha256
  raw_receipts.append(dict(record=rr.record,path=str(p),sha256=rr.input_sha256,bytes=p.stat().st_size))
  df=pd.read_csv(p);a=df[CH].apply(pd.to_numeric,errors="coerce").to_numpy()
  for i,row in meta[meta.record==rr.record].iterrows():
   z=a[int(row.start_row):int(row.end_row_exclusive)].copy()
   assert z.shape==(800,8)
   for j in range(8):
    valid=np.isfinite(z[:,j]);assert valid.any()
    z[:,j]=np.interp(np.arange(800),np.flatnonzero(valid),z[valid,j])
   x[i]=physical(z.T)
 assert np.isfinite(x).all()
 np.save(O/"windows_physical_float64.npy",x)
 # Retain original row order, exact source labels.
 meta.to_csv(O/"windows.csv",index=False);rec.to_csv(O/"records_LOCAL.csv",index=False)
 rngsplit=np.random.default_rng(CFG["split_seed"]);perms={a:rngsplit.permutation(sorted(rec[rec.activity==a].record)).tolist() for a in ACT}
 use={r:0 for r in rec.record};splits=[];rows=[];allrecs=sorted(rec.record)
 for i,held in enumerate(allrecs):
  val=[]
  for a in ACT:
   order=perms[a][i%3:]+perms[a][:i%3];options=[r for r in order if r!=held]
   chosen=min(options,key=lambda r:(use[r],order.index(r)));val.append(chosen);use[chosen]+=1
  outer=[r for r in allrecs if r!=held];inner=[r for r in outer if r not in val]
  assert len(inner)==15 and len(val)==8 and not set(inner)&set(val)
  assert set(meta[meta.record.isin(inner)].activity)==set(ACT)
  assert set(meta[meta.record.isin(val)].activity)==set(ACT)
  splits.append(dict(fold=held,inner_train=inner,inner_val=val,outer_train=outer,test=[held]))
  for rid in allrecs:rows.append(dict(fold=held,record=rid,role="test" if rid==held else "inner_val" if rid in val else "inner_train"))
 aux=dict(fold="AUX16_8",inner_train=[perms[a][2] for a in ACT],inner_val=[perms[a][1] for a in ACT],outer_train=[perms[a][i] for a in ACT for i in [1,2]],test=[perms[a][0] for a in ACT])
 selected=[]
 for rid in aux["test"]:
  end=-1
  for i,z in meta[meta.record==rid].sort_values("start_row").iterrows():
   if z.start_row>=end:selected.append(int(i));end=int(z.end_row_exclusive)
 assert len(selected)>16
 aux["point_indices"]=selected
 pd.DataFrame(rows).to_csv(O/"folds.csv",index=False)
 split_object=dict(main=splits,aux=aux,record_permutations=perms,inner_validation_counts=use)
 assert split_object==json.loads((O.parent.parent/"hf_scene_classification_v3_lyx24/splits.json").read_text())
 jw(O/"splits.json",split_object)
 CFG["libraries"]={"scipy":scipy.__version__,"sklearn":sklearn.__version__,"joblib":joblib.__version__}
 jw(O/"config.json",CFG)
 tests=unit_tests();jw(O/"unit_tests.json",tests)
 # No learned real-data scores have been generated at this point.
 baseline=[dict(path=str(p.relative_to(V2)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(V2.rglob("*")) if p.is_file()]
 freeze=dict(status="frozen_before_formal_training",utc=utc(),formal_fits_started=0,formal_outer_scores_observed=0,raw=raw_receipts,v2_baseline=baseline,
  artifacts={n:sha(O/n) for n in ["config.json","splits.json","folds.csv","windows.csv","records_LOCAL.csv","windows_physical_float64.npy","unit_tests.json",Path(__file__).name,"FINAL_PLAN.md"]},
  environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"]),
  git=dict(head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=R,text=True).strip(),status=subprocess.check_output(["git","status","--short"],cwd=R,text=True,encoding="utf-8").strip()),
  resume="This frozen script --run validates identities; completed tasks verified by hashes; incomplete epoch checkpoint resumes optimizer and RNG. No automatic migration.")
 jw(O/"freeze_receipt.json",freeze)
 jw(O/"run_status.json",dict(status="frozen_ready",main_completed=0,aux_completed=0,failed=0,device="cuda:0",utc=utc()))
 log("FROZEN: 24 raw hashes,1941 windows,432 main+6aux jobs,formal fits0; deterministic CUDA tests PASS.")
def identity(freeze,scope,split,arch,arm,n):
 return dict(freeze_sha256=sha(O/"freeze_receipt.json"),scope=scope,fold=split["fold"],architecture=arch,arm=arm,seed=n,config_sha256=freeze["artifacts"]["config.json"],split_sha256=freeze["artifacts"]["splits.json"],script_sha256=freeze["artifacts"][Path(__file__).name])
def run_job(freeze,scope,split,arch,arm,n,meta,x,mom):
 ident=identity(freeze,scope,split,arch,arm,n);key=f"{scope}__{split['fold']}__{arch}_{arm}__s{n}"
 dest=O/"jobs"/key;dest.mkdir(parents=True,exist_ok=True);cp=dest/"checkpoint.pt";done=dest/"done.json"
 if done.exists():
  dd=json.loads(done.read_text());assert dd["identity"]==ident
  assert all(sha(dest/name)==h for name,h in dd["files"].items());return False
 cols=ARMS[arm];trainids=indices(meta,split["inner_train"]);valids=indices(meta,split["inner_val"]);finalids=indices(meta,split["outer_train"]);testids=indices(meta,split["test"])
 assert not set(trainids)&set(valids) and not set(finalids)&set(testids)
 scalers=dict(inner=scale_for(meta,trainids,mom),final=scale_for(meta,finalids,mom))
 y=torch.tensor(meta.activity.map(dict(zip(ACT,range(8)))).to_numpy(),dtype=torch.long)
 started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 st=None
 if cp.exists():
  st=torch.load(cp,map_location="cpu",weights_only=False);assert st["identity"]==ident
 if st is None:
  seed(n);model=Net(len(cols),arch).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  st=dict(identity=ident,phase="inner",epoch=0,best_epoch=0,best_loss=float("inf"),bad_epochs=0,inner_history=[],final_history=[],scalers=scalers,initial_model_hash=statehash(model),elapsed_seconds=0.)
 else:
  model=Net(len(cols),arch).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  model.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"]);restore_rng(st["rng"])
  for phase in ["inner","final"]:
   assert np.array_equal(st["scalers"][phase]["indices"],scalers[phase]["indices"])
   assert np.array_equal(st["scalers"][phase]["mean"],scalers[phase]["mean"])
   assert np.array_equal(st["scalers"][phase]["scale"],scalers[phase]["scale"])
 def save():
  st.update(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),parameters=sum(p.numel() for p in model.parameters()),device=str(next(model.parameters()).device))
  tsave(cp,st)
 if st["phase"]=="inner":
  xt=scaled(x,trainids,cols,scalers["inner"]);yt=y[trainids];wt=torch.tensor(weight(meta,trainids,True),dtype=torch.float32)
  xv=scaled(x,valids,cols,scalers["inner"]);yv=y[valids];wv=weight(meta,valids)
  while st["epoch"]<CFG["max_epochs"]:
   tick=time.perf_counter();tl=train_epoch(model,opt,xt,yt,wt);vl,_,_=evaluate(model,xv,yv,wv);st["epoch"]+=1
   st["inner_history"].append(dict(epoch=st["epoch"],train_CE=tl,validation_CE=vl,seconds=time.perf_counter()-tick))
   if vl<st["best_loss"]:st.update(best_loss=vl,best_epoch=st["epoch"],bad_epochs=0)
   else:st["bad_epochs"]+=1
   stop=st["epoch"]>=CFG["max_epochs"] or (st["epoch"]>=CFG["min_epochs"] and st["bad_epochs"]>=CFG["patience"])
   if stop:st["phase"]="inner_finished"
   save()
   if stop:break
  del xt,yt,wt,xv,yv
 if st["phase"]=="inner_finished":
  seed(n);model=Net(len(cols),arch).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  assert statehash(model)==st["initial_model_hash"]
  st.update(phase="final",epoch=0,refit_initial_model_hash=statehash(model));save()
 if st["phase"]=="final":
  xt=scaled(x,finalids,cols,scalers["final"]);yt=y[finalids];wt=torch.tensor(weight(meta,finalids,True),dtype=torch.float32)
  while st["epoch"]<st["best_epoch"]:
   tick=time.perf_counter();tl=train_epoch(model,opt,xt,yt,wt);st["epoch"]+=1
   st["final_history"].append(dict(epoch=st["epoch"],train_CE=tl,seconds=time.perf_counter()-tick))
   if st["epoch"]==st["best_epoch"]:st["phase"]="final_finished"
   save()
  del xt,yt,wt
 if st["phase"]=="final_finished":
  model.eval();xtest=scaled(x,testids,cols,scalers["final"])
  wt=weight(meta,testids) if scope=="aux" else np.ones(len(testids))/len(testids)
  _,logits,latent=evaluate(model,xtest,y[testids],wt);prob=torch.softmax(torch.tensor(logits),1).numpy();pr=logits.argmax(1)
  g=meta.iloc[testids][["record","activity","window_idx","center_s","missing_rows"]].copy();g["original_row"]=testids;g["scope"]=scope;g["configuration"]=arch+"_"+arm;g["seed"]=n;g["fold"]=split["fold"];g["true_index"]=y[testids].numpy();g["pred_index"]=pr;g["prediction"]=[ACT[i] for i in pr]
  g["eval_weight"]=[1/(8*(3 if scope=="main" else 1)*len(meta[meta.record==rid])) for rid in g.record]
  for j,a in enumerate(ACT):g["logit_"+a]=logits[:,j];g["prob_"+a]=prob[:,j]
  g.to_csv(dest/"predictions.csv",index=False)
  files=["checkpoint.pt","predictions.csv"]
  if scope=="aux":
   # Learned representation normalization/PCA reference only uses16 training records.
   xtrain=scaled(x,finalids,cols,scalers["final"]);_,_,trainlatent=evaluate(model,xtrain,y[finalids],weight(meta,finalids))
   np.savez_compressed(dest/"latents.npz",test_indices=testids,test_latent=latent,train_indices=finalids,train_latent=trainlatent,selected_indices=np.array(split["point_indices"],int));files.append("latents.npz")
  st["phase"]="done";st["elapsed_seconds"]+=time.perf_counter()-started;st["peak_cuda_allocated"]=torch.cuda.max_memory_allocated();st["peak_cuda_reserved"]=torch.cuda.max_memory_reserved();save()
  jw(done,dict(identity=ident,status="complete",selected_epoch=st["best_epoch"],inner_epochs=len(st["inner_history"]),final_epochs=len(st["final_history"]),parameters=st["parameters"],device=st["device"],elapsed_seconds=st["elapsed_seconds"],peak_cuda_allocated=st["peak_cuda_allocated"],files={name:sha(dest/name) for name in files}))
 elif st["phase"]=="done":
  # Completed atomic checkpoint means predictions were fully written before it; reconstruct only the marker.
  files=["checkpoint.pt","predictions.csv"]+(["latents.npz"] if scope=="aux" else [])
  assert all((dest/name).is_file() for name in files)
  jw(done,dict(identity=ident,status="complete",selected_epoch=st["best_epoch"],inner_epochs=len(st["inner_history"]),final_epochs=len(st["final_history"]),parameters=st["parameters"],device=st["device"],elapsed_seconds=st["elapsed_seconds"],peak_cuda_allocated=st["peak_cuda_allocated"],files={name:sha(dest/name) for name in files},reconstructed_marker=True))
 del model,opt;gc.collect();torch.cuda.empty_cache()
 return True
def run():
 freeze=json.loads((O/"freeze_receipt.json").read_text(encoding="utf-8"))
 for name,h in freeze["artifacts"].items():assert sha(O/name)==h,("Frozen artifact changed",name)
 for row in freeze["raw"]:assert sha(row["path"])==row["sha256"]
 assert torch.cuda.is_available() and torch.__version__==freeze["environment"]["torch"]
 lock=O/"RUNNING.lock"
 # Exclusive OS file lock is released if process exits, retained file records task identity.
 import msvcrt
 lf=lock.open("a+b")
 if lf.tell()==0:lf.write(b"0");lf.flush()
 lf.seek(0)
 try:msvcrt.locking(lf.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise RuntimeError("Another third-round runner owns RUNNING.lock")
 meta=pd.read_csv(O/"windows.csv");x=np.load(O/"windows_physical_float64.npy");mom=make_stats(x);sp=json.loads((O/"splits.json").read_text())
 jobs=[("main",s,a,b,n) for s in sp["main"] for a in ["CNN","CNNLSTM"] for b in ARMS for n in SEEDS]+[("aux",sp["aux"],a,b,SEEDS[0]) for a in ["CNN","CNNLSTM"] for b in ARMS]
 start=time.perf_counter();main=aux=0;fresh=0
 try:
  run_classic(freeze,meta,x,mom,sp)
  for scope,s,a,b,n in jobs:
   key=f"{scope}__{s['fold']}__{a}_{b}__s{n}"
   jw(O/"run_status.json",dict(status="running",utc=utc(),pid=os.getpid(),current_job=key,main_completed=main,aux_completed=aux,failed=0,device="cuda:0",session_elapsed_seconds=time.perf_counter()-start))
   ran=run_job(freeze,scope,s,a,b,n,meta,x,mom);fresh+=int(ran)
   if scope=="main":main+=1
   else:aux+=1
   elapsed=time.perf_counter()-start;eta=elapsed/fresh*(438-main-aux) if fresh else None
   jw(O/"run_status.json",dict(status="running",utc=utc(),pid=os.getpid(),current_job=key,main_completed=main,aux_completed=aux,failed=0,device="cuda:0",session_elapsed_seconds=elapsed,eta_seconds=eta))
   log(f"DONE main={main}/432 aux={aux}/6 failed=0 device=cuda:0 job={key} elapsed={elapsed:.1f}s eta={eta}")
  parts=[pd.read_csv(p) for p in sorted((O/"jobs").glob("main*/predictions.csv"))];assert len(parts)==432
  pd.concat(parts,ignore_index=True).to_csv(O/"oof_predictions.csv",index=False)
  parts=[pd.read_csv(p) for p in sorted((O/"jobs").glob("aux*/predictions.csv"))];assert len(parts)==6
  pd.concat(parts,ignore_index=True).to_csv(O/"aux_predictions.csv",index=False)
  assert all(sha(z["path"])==z["sha256"] for z in freeze["raw"])
  assert all(sha(V2/z["path"])==z["sha256"] for z in freeze["v2_baseline"])
  jw(O/"run_status.json",dict(status="training_complete",utc=utc(),main_completed=432,aux_completed=6,failed=0,device="cuda:0",session_elapsed_seconds=time.perf_counter()-start,raw_hashes_unchanged=True,v2_hashes_unchanged=True))
  log("TRAINING COMPLETE; all432 main+6aux; no tuning.")
 except BaseException as e:
  jw(O/"run_status.json",dict(status="paused_on_error",utc=utc(),main_completed=main,aux_completed=aux,failed=1,current_job=key,device="cuda:0",error=repr(e)))
  log(traceback.format_exc());raise
 finally:
  lf.seek(0);msvcrt.locking(lf.fileno(),msvcrt.LK_UNLCK,1);lf.close()

from scipy.signal import butter,sosfiltfilt
import scipy,sklearn,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
SOS=butter(4,20,fs=100,btype="lowpass",output="sos")
CFG.update(experiment="LYX24_round5_unified_Wei_inspired_light",architecture="Conv1d(C,16,k9,s1,p4)->AvgPool1d(2,2)->ReLU. CNN global time mean->Dropout.2->Linear16to8; CNNLSTM sequence400x16->one-layer unidirectional LSTM32 last hidden->Dropout.2->Linear32to8. Actual C=6/2/8. No normalization layers.",latent_dim={"CNN":16,"CNNLSTM":32},
 scaler="No subtraction after physical preprocessing. Three class-record-window weighted training RMS scalars for ACC3/GYRO3/HF2. HF shares denominator. fallback1 if RMS<=1e-12. Refit each inner/outer training split only. Physical cache float64, scaled inputs float32 shared by all algorithms.",
 preprocessing=dict(fs=100,window=800,filter="Butterworth fourth-order single-pass design, sosfiltfilt odd padlen100 per window; effective zero-phase response squared,20Hz approx -6dB",sos=SOS.tolist(),baseline="HF each window/channel mean removed after filtering; ACC/GYRO retain DC. No per-window SD, no highpass/detrend, no initial30 baseline"),
 classical=dict(features="seven per channel: mean,std_ddof0,ptp,mean_abs_diff,dominant_0.5_20,normalized_entropy_0.5_20,power0.5_5_over0.5_20; HF means exact0; float64 computation from common scaledfloat32 tensor",ET=dict(n_estimators=200,max_depth=4,max_leaf_nodes=16,max_features=1.0,bootstrap=False,criterion="gini",min_samples_leaf=1,n_jobs=1,seeds=SEEDS),SVC=dict(C=1.,gamma="1/d",kernel="rbf",scaler="weighted training StandardScaler",seed="deterministic single fit"),main_fits=288),
 total_main_fits=1152,total_all_fits=1164,search_budget=0,synthetic_benchmark="unit_tests.json; no performance tuning")
CFG["embeddings"]["layer"]="before Dropout/Linear: CNN16, CNNLSTM32"
CFG["models"]=[a+"_"+b for a in ["ET","SVC","CNN","CNNLSTM"] for b in ARMS]

def physical(a):
 z=np.asarray(a,dtype=np.float64).copy()
 assert z.shape==(8,800)
 for j in range(8):
  v=np.isfinite(z[j])
  if not v.any():raise ValueError("all missing channel")
  z[j]=np.interp(np.arange(800),np.flatnonzero(v),z[j,v])
 z=sosfiltfilt(SOS,z,axis=1,padtype="odd",padlen=100)
 z[6:]-=z[6:].mean(axis=1,keepdims=True)
 assert np.isfinite(z).all()
 return z

def preprocess_tests():
 t=np.arange(800)/100;z=np.zeros((8,800));z[2]=1;z[3]=.25;z[6]=100+np.sin(2*np.pi*2*t);z[7]=500+2*np.sin(2*np.pi*2*t)
 a=physical(z);b=z.copy();b[6]+=777;b[7]-=246;b=physical(b)
 assert np.allclose(a,b,atol=1e-10) and np.allclose(a[7],2*a[6],atol=1e-10)
 assert np.allclose(a[2],1) and np.allclose(a[3],.25) and np.max(abs(a[6:].mean(1)))<1e-10
 c=z.copy();c[6,100:110]=np.nan;assert np.isfinite(physical(c)).all()
 c=z.copy();c[6]=np.nan
 try:physical(c);raise AssertionError("missing rejection")
 except ValueError:pass
 sm=pd.DataFrame([dict(record=f"{v}-{r}",activity=v) for v in ACT for r in range(2) for _ in range(r+1)])
 xx=np.stack([a*(i+1) for i in range(len(sm))]);ids=np.arange(len(sm));mom=make_stats(xx);ss=scale_for(sm,ids,mom);w=weight(sm,ids)
 for cols in [list(range(3)),list(range(3,6)),[6,7]]:
  direct=np.sqrt(np.sum(w*np.mean(xx[:,cols,:]**2,axis=(1,2))))
  assert np.allclose(ss["scale"][cols],direct)
 assert np.array_equal(scaled(xx,ids,[6,7],ss).numpy(),scaled(xx,ids,list(range(8)),ss).numpy()[:,6:])
 assert np.allclose(scaled(xx,ids,[6,7],ss).numpy()[:,1],2*scaled(xx,ids,[6,7],ss).numpy()[:,0])
 small=np.arange(len(sm)-2);s1=scale_for(sm,small,mom);xx[-2:]*=1000;s2=scale_for(sm,small,make_stats(xx));assert np.array_equal(s1["scale"],s2["scale"])
 for act in ACT:assert np.isclose(w[sm.activity==act].sum(),1/8)
 feat=features(scaled(xx,ids,list(range(8)),ss).numpy());assert feat.shape==(len(sm),8,7) and np.all(feat[:,6:,0]==0)
 # DC constants / edge step remain finite. Verify deep stopband attenuation away from padded endpoints.
 q=np.zeros((8,800));q[0]=np.sin(2*np.pi*40*t);q[1]=np.sin(2*np.pi*2*t);q[2,400:]=1
 f=physical(q);assert np.std(f[0,100:-100])<.01 and np.std(f[1,100:-100])>.6 and np.isfinite(f).all()
 return ["HF baseline shift invariant, shared channel amplitude ratio retained, ACC/GYRO DC retained", "window-local missing interpolation and allmissing rejection", "weighted group RMS analytic agreement, held-out perturbation cannot change training scale, modal tensors identical", "HF mean features exactly zero, lowpass pass/stop bands and step finite", "synthetic GPU checkpoint exact continuation and deterministic eval; CNN/LSTM common conv initialization"]

FREQ=np.fft.rfftfreq(800,.01);BAND=(FREQ>=.5)&(FREQ<=20);LOW=(FREQ>=.5)&(FREQ<=5)
def features(x):
 z=x.astype(np.float64);mean=z.mean(2);power=abs(np.fft.rfft((z-mean[:,:,None])*np.hanning(800),axis=2))**2
 pb=power[:,:,BAND];total=pb.sum(2);q=np.divide(pb,total[:,:,None],out=np.zeros_like(pb),where=total[:,:,None]>0)
 entropy=-(q*np.log(np.maximum(q,np.finfo(float).tiny))).sum(2)/np.log(pb.shape[2]);dom=FREQ[BAND][pb.argmax(2)];dom[total==0]=0
 ratio=np.divide(power[:,:,LOW].sum(2),total,out=np.zeros_like(total),where=total>0)
 out=np.stack([mean,z.std(2),np.ptp(z,axis=2),abs(np.diff(z,axis=2)).mean(2),dom,entropy,ratio],axis=2);out[:,6:,0]=0
 assert np.isfinite(out).all();return out

def run_classic(freeze,meta,x,mom,sp):
 y=meta.activity.map(dict(zip(ACT,range(8)))).to_numpy();ew=weight(meta,np.arange(len(meta)));completed=0
 with threadpool_limits(limits=1):
  for split in sp["main"]:
   ti=indices(meta,split["outer_train"]);vi=indices(meta,split["test"]);sc=scale_for(meta,ti,mom)
   assert not set(meta.iloc[ti].record)&set(meta.iloc[vi].record)
   allids=np.arange(len(meta));ft=features(scaled(x,allids,list(range(8)),sc).numpy());w=weight(meta,ti);sw=weight(meta,ti,True)
   for arch in ["ET","SVC"]:
    for arm,cols in ARMS.items():
     mat=ft[:,cols,:].reshape(len(meta),-1)
     for n in (SEEDS if arch=="ET" else [0]):
      ident=identity(freeze,"classic",split,arch,arm,n);dest=O/"classic_jobs"/f"{split['fold']}__{arch}_{arm}__s{n}";dest.mkdir(parents=True,exist_ok=True)
      done=dest/"done.json"
      if done.exists():
       dd=json.loads(done.read_text());assert dd["identity"]==ident and all(sha(dest/k)==v for k,v in dd["files"].items());completed+=1;continue
      tick=time.perf_counter();fs=None
      if arch=="ET":
       model=ExtraTreesClassifier(n_estimators=200,max_depth=4,max_leaf_nodes=16,min_samples_leaf=1,max_features=1.0,bootstrap=False,criterion="gini",random_state=n,n_jobs=1,class_weight=None)
       model.fit(mat[ti],y[ti],sample_weight=w);pr=model.predict(mat[vi]);scores=model.predict_proba(mat[vi]);kind="probability"
      else:
       fs=StandardScaler().fit(mat[ti],sample_weight=sw)
       model=SVC(C=1.,kernel="rbf",gamma=1./mat.shape[1],class_weight=None,probability=False,decision_function_shape="ovr",tol=.001,shrinking=True,cache_size=200,max_iter=-1)
       model.fit(fs.transform(mat[ti]),y[ti],sample_weight=sw);assert model.fit_status_==0
       pr=model.predict(fs.transform(mat[vi]));scores=model.decision_function(fs.transform(mat[vi]));kind="ovr_decision_not_probability"
      assert np.array_equal(model.classes_,np.arange(8)) and np.isfinite(scores).all()
      g=meta.iloc[vi].copy();g["original_row"]=vi;g["scope"]="main";g["configuration"]=arch+"_"+arm;g["seed"]=n;g["fold"]=split["fold"];g["true_index"]=y[vi];g["pred_index"]=pr;g["prediction"]=[ACT[j] for j in pr];g["eval_weight"]=ew[vi];g["score_kind"]=kind
      for j,act in enumerate(ACT):g["score_"+act]=scores[:,j]
      g.to_csv(dest/"predictions.csv",index=False);joblib.dump(dict(model=model,feature_scaler=fs,signal_scaler=sc,train_indices=ti,test_indices=vi,identity=ident),dest/"model.joblib")
      jw(done,dict(identity=ident,status="complete",seconds=time.perf_counter()-tick,features=mat.shape[1],files={name:sha(dest/name) for name in ["model.joblib","predictions.csv"]}));completed+=1
   jw(O/"classic_status.json",dict(status="running",completed=completed,total=288,utc=utc()));log(f"CLASSIC {completed}/288")
 parts=[pd.read_csv(p) for p in sorted((O/"classic_jobs").glob("*/predictions.csv"))];assert len(parts)==288
 pd.concat(parts,ignore_index=True).to_csv(O/"classic_oof_predictions.csv",index=False)
 jw(O/"classic_status.json",dict(status="complete",completed=288,total=288,utc=utc()))

if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");parser.add_argument("--run",action="store_true");args=parser.parse_args()
 if args.prepare:prepare()
 elif args.run:run()
 else:parser.error("choose --prepare or --run")

