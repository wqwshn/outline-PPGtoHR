"""Frozen 119-record, six-subject CNN study. --prepare validates without real fitting; --run executes144fits."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["OMP_NUM_THREADS"]="4"
os.environ["MKL_NUM_THREADS"]="4"
from pathlib import Path
import sys,json,hashlib,time,datetime,random,argparse,importlib.util,gc,traceback,tempfile,copy
import numpy as np,pandas as pd,torch
O=Path(__file__).resolve().parent;R=O.parents[2]
V3=O.parent/"hf_scene_classification_v3_lyx24";V2=O.parent/"hf_scene_classification_v2_lyx24"/"execution_20261006_01"
P0=O.parent/"hf_scene_classification_v1"/"p0_20261005_02"
spec=importlib.util.spec_from_file_location("frozen_v3_core",V3/"run_neural_matrix.py")
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
ACT=b.ACT;CH=b.CH;SEEDS=b.SEEDS;ARMS=b.ARMS
sha=b.sha;utc=b.utc;jw=b.jw;tsave=b.tsave;seed=b.seed;statehash=b.statehash
CFG=dict(experiment="v4_119_fixed_CNN_144fits",activities=ACT,channels=CH,seeds=SEEDS,inner_seed=SEEDS[0],
 arms=ARMS,architecture="Same v3CNN; uniform8input,16/24/32conv,k9/7/5,s2,p4/3/2;meanT;dropout.1;linear32to8",parameters=8016,
 inactive_channels="exact zero AFTER training-only8-channel standardization; no per-window centering",
 optimizer=dict(name="AdamW",lr=.001,weight_decay=.0001,clip_norm=1.),batch_size=64,dtype="float32",
 min_epochs=15,max_epochs=40,patience=12,selection="earliest strict minimum weighted validationCE over all completed epochs; min15 is executed epochs not minimum selected epoch; median of five integer best epochs (odd count => exact integer), separately per outer-person and arm; fresh5person refit",
 subjects=[f"subject-{i}" for i in range(1,7)],outer_folds=6,inner_per_outer=5,inner_fits=90,refit_fits=54,total_fits=144,
 weights="1/(Nsubjects*8*Nrecords_subject_activity*Nwindows_record); training mean-one, uniform randperm windows, batchmean(w*CE); scaler uses sum-one same weights and all800samples",
 scaler="train-only weighted8-channel mean and population variance in float64 from float32 windows; scale0=>1; refit5persons recomputes independently",
 missing="reuse all9520 P0windows; np.interp withinwindow/channel and nearest edges; allmissing error; no quality exclusions",
 primary="subject-weighted macroF1 per held-out person then equalperson mean; pairedFUSminusMIMU averaged across6persons; all3seeds reported and their mean; overlappingwindows not independent",
 record_vote="per-record unweighted window prediction counts; choose first maximum in frozen ACT order; record count accuracy and subject-activity equalweighted record accuracy both reported; disclose ties and tie-as-wrong sensitivity",
 probability_ties="np.argmax chooses first class in ACT",
 late_fusion=dict(HF_LATE=".75*MIMU_seed+.25*HF_same_seed",MIMU_ENS=".75*MIMU_seed+.25*MIMU_nextseed",next_seed=dict(zip(map(str,SEEDS),SEEDS[1:]+SEEDS[:1]))),
 case_rules="posthoc explanation, not representative inference: rank records by mean3seed(FUSwindowAcc-MIMUwindowAcc), maximumpositive and minimumnegative, ties lexicographicrecord; use chronologically median matching rescue/damage window if any; fixed HG-07 subject1 sentinel (v3HG02 hash matched); same-subject confused-class reference first record lexicographic and median original window; report missingness stratification and counterexamples",
 no_extra_families=True,augmentation=False,scheduler=False,deterministic=True,tf32=False,
 provenance_limits=["21rawrecords overlapLYXv3 subject1","historically developed119panel, not pristine external validation","same-row100Hz alignment not hardware timing proof","HF physical transduction/calibration not fully established","classification is not HR measurement improvement"],
 source="P0 record_bindings_LOCAL.csv119 and motion_windows.csv9520; no redetection")
def log(s):
 print(s,flush=True)
 with (O/"training.log").open("a",encoding="utf-8") as f:f.write(utc()+" "+s+"\n")
def weight(meta,ids,mean_one=False):
 g=meta.iloc[ids];ns=g.subject.nunique();nr=g.groupby(["subject","activity"]).record.nunique();nw=g.groupby("record").size()
 assert len(nr)==8*ns
 w=np.array([1/(ns*8*nr[(r.subject,r.activity)]*nw[r.record]) for r in g.itertuples()],float)
 assert np.isclose(w.sum(),1)
 return w*len(w) if mean_one else w
def scaler(meta,ids,mom):
 w=weight(meta,ids);m=w@mom[0][ids];v=np.maximum(w@mom[1][ids]-m*m,0);s=np.sqrt(v);s[s==0]=1
 return dict(mean=m,variance=v,scale=s,indices=np.asarray(ids),weights=w,subjects=sorted(meta.iloc[ids].subject.unique().tolist()),records=sorted(meta.iloc[ids].record.unique().tolist()))
def scaled(x,ids,arm,sc):
 out=np.empty((len(ids),8,800),np.float32);inactive=[j for j in range(8) if j not in ARMS[arm]]
 for k in range(0,len(ids),128):
  z=x[ids[k:k+128]].astype(np.float64);z-=sc["mean"][None,:,None];z/=sc["scale"][None,:,None];z[:,inactive,:]=0
  out[k:k+128]=z
 return torch.from_numpy(out)
def idx(meta,subjects):return np.flatnonzero(meta.subject.isin(subjects).to_numpy())
def unit_tests():
 assert torch.cuda.is_available()
 sm=pd.DataFrame([dict(subject=f"s{s}",activity=a,record=f"s{s}-{a}-{r}") for s in range(2) for a in ACT for r in range(s+1) for k in range(r+1)])
 xx=np.stack([np.full((8,800),float(i)) for i in range(len(sm))]).astype(np.float32);ii=np.arange(len(sm));ww=weight(sm,ii);sc=scaler(sm,ii,b.make_stats(xx))
 for (s,a),g in sm.groupby(["subject","activity"]):assert np.isclose(ww[g.index].sum(),1/16)
 assert np.isclose(weight(sm,ii,True).mean(),1)
 assert np.allclose(sc["mean"],np.average(xx.mean(2),axis=0,weights=ww))
 fused=scaled(xx,ii,"FUS",sc).numpy()
 for arm,cols in ARMS.items():
  z=scaled(xx,ii,arm,sc).numpy();assert np.array_equal(z[:,cols],fused[:,cols])
  assert np.all(z[:,[j for j in range(8) if j not in cols]]==0)
 assert int(np.median([1,2,7,10,40]))==7
 init=[]
 for arm in ARMS:
  seed(SEEDS[0]);m=b.Net(8,"CNN").cuda();assert sum(p.numel() for p in m.parameters())==8016;init.append(statehash(m))
 assert len(set(init))==1
 seed(2468);m=b.Net(8,"CNN").cuda();op=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);sx=torch.randn(128,8,800);sy=torch.arange(128)%8;sw=torch.ones(128)
 b.train_epoch(m,op,sx,sy,sw)
 snapshot=copy.deepcopy(dict(model=m.state_dict(),optimizer=op.state_dict(),rng=b.rng()))
 b.train_epoch(m,op,sx,sy,sw);expected=statehash(m)
 n=b.Net(8,"CNN").cuda();no=torch.optim.AdamW(n.parameters(),lr=.001,weight_decay=.0001);n.load_state_dict(snapshot["model"]);no.load_state_dict(snapshot["optimizer"]);b.restore_rng(snapshot["rng"])
 b.train_epoch(n,no,sx,sy,sw);assert statehash(n)==expected
 n.eval()
 with torch.no_grad():assert torch.equal(n(sx[:4].cuda()),n(sx[:4].cuda()))
 # Shape/resource benchmark with synthetic labels; no real outer scores.
 seed(7890);bx=torch.randn(6400,8,800);by=torch.arange(6400)%8;bw=torch.ones(6400)
 m=b.Net(8,"CNN").cuda();op=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);torch.cuda.reset_peak_memory_stats()
 times=[]
 for _ in range(3):
  torch.cuda.synchronize();t=time.perf_counter();b.train_epoch(m,op,bx,by,bw);torch.cuda.synchronize();times.append(time.perf_counter()-t)
 peak=torch.cuda.max_memory_allocated();free,total=torch.cuda.mem_get_info()
 return dict(status="PASS",utc=utc(),real_fits=0,tests=["hierarchicalweight","trainonlyscaler","zerochannels","matchedinitialization8016","exactRNGoptimizerresume","evaldeterminism","median5rule","synthetic6400x8x800resource"],synthetic_epoch_seconds=times,peak_cuda_allocated=peak,cuda_free=free,cuda_total=total)
def prepare():
 assert not (O/"freeze_receipt.json").exists(),"Already frozen"
 assert not (O/"jobs").exists()
 rec=pd.read_csv(P0/"record_bindings_LOCAL.csv");meta=pd.read_csv(P0/"motion_windows.csv")
 assert len(rec)==119 and len(meta)==9520 and rec.record.nunique()==119
 assert sorted(rec.subject.unique())==CFG["subjects"];assert rec.groupby("subject").activity.nunique().eq(8).all()
 assert (meta.end_row_exclusive-meta.start_row).eq(800).all() and meta.support_in_bounds.all()
 assert not meta.duplicated(["record","window_idx"]).any()
 assert len(set(rec.input_sha256))==119 and len(set(rec.signal_sha256))==119
 assert (meta.groupby("record").size().sort_index().values==rec.set_index("record").motion_windows.sort_index().values).all()
 lookup=rec.set_index("record")
 assert all(z.subject==lookup.loc[z.record,"subject"] and z.activity==lookup.loc[z.record,"activity"] for z in meta.itertuples())
 x=np.lib.format.open_memmap(O/"windows_float32.npy",mode="w+",dtype=np.float32,shape=(9520,8,800))
 raws=[];missing=[]
 for rr in rec.itertuples():
  p=Path(rr.input_path);assert sha(p)==rr.input_sha256
  df=pd.read_csv(p);assert len(df)==rr.rows
  tt=pd.to_numeric(df["Time(s)"]).to_numpy();assert np.allclose(np.diff(tt),.01,rtol=0,atol=1e-7)
  a=df[CH].apply(pd.to_numeric,errors="coerce").to_numpy(float)
  for i,z in meta[meta.record==rr.record].iterrows():
   aa=a[int(z.start_row):int(z.end_row_exclusive)].copy();assert aa.shape==(800,8)
   bad=~np.isfinite(aa);missing.append(dict(original_row=int(i),imputed_rows=int(bad.any(1).sum()),imputed_values=int(bad.sum())))
   for j in range(8):
    good=np.isfinite(aa[:,j]);assert good.any()
    aa[:,j]=np.interp(np.arange(800),np.flatnonzero(good),aa[good,j])
   x[i]=aa.T.astype(np.float32)
  raws.append(dict(record=rr.record,subject=rr.subject,path=str(p),sha256=rr.input_sha256))
 x.flush();assert np.isfinite(x).all();del x
 meta.to_csv(O/"windows.csv",index=False);rec.to_csv(O/"records_LOCAL.csv",index=False);pd.DataFrame(missing).sort_values("original_row").to_csv(O/"imputation.csv",index=False)
 splits=[]
 for held in CFG["subjects"]:
  train=[s for s in CFG["subjects"] if s!=held]
  inn=[dict(validation=v,train=[s for s in train if s!=v]) for v in train]
  assert all(not(set(z["train"])&{held,z["validation"]}) for z in inn)
  splits.append(dict(test=held,train=train,inner=inn))
 jw(O/"splits.json",splits);jw(O/"config.json",CFG)
 tests=unit_tests();jw(O/"unit_tests.json",tests)
 v3hashes={str(p.relative_to(V3)):sha(p) for p in sorted(V3.rglob("*")) if p.is_file() and p.suffix in [".py",".json",".csv",".pt",".npy",".npz"] and p.name not in ["figure_qa.json"]}
 v2hashes=json.loads((V3/"freeze_receipt.json").read_text())["v2_baseline"]
 assert all(sha(V2/z["path"])==z["sha256"] for z in v2hashes)
 v3raw=json.loads((V3/"freeze_receipt.json").read_text())["raw"];overlap=[z for z in raws if z["sha256"] in {v["sha256"] for v in v3raw}]
 assert len(overlap)==21 and {z["subject"] for z in overlap}=={"subject-1"}
 names=["run_119.py","config.json","splits.json","windows.csv","records_LOCAL.csv","windows_float32.npy","imputation.csv","unit_tests.json"]
 fr=dict(status="frozen_before_real_training",utc=utc(),real_fits=0,new_outer_scores=0,raw=raws,artifacts={n:sha(O/n) for n in names},source_P0={n:sha(P0/n) for n in ["record_bindings_LOCAL.csv","motion_windows.csv"]},v3=v3hashes,v2=v2hashes,overlap_v3=overlap,environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),numpy=np.__version__,pandas=pd.__version__),scope="Only144fixedfits; no result-dependent additions")
 jw(O/"freeze_receipt.json",fr);jw(O/"run_status.json",dict(status="frozen_ready",utc=utc(),completed=0,total=144,inner_completed=0,refit_completed=0,failed=0))
 log("FROZEN119 records/9520windows,90inner+54refit,zero realfits; unit/resource PASS")


def jobkey(phase,held,arm,n,val=None):return f"{phase}__{held}__{arm}__s{n}"+(f"__val{val}" if val else "")
def run_job(fr,phase,held,train,arm,n,meta,x,mom,val=None,epochs=None,selection=None):
 key=jobkey(phase,held,arm,n,val);dest=O/"jobs"/key;dest.mkdir(parents=True,exist_ok=True);cp=dest/"checkpoint.pt";done=dest/"done.json"
 ident=dict(freeze_sha256=sha(O/"freeze_receipt.json"),phase=phase,held=held,train=train,arm=arm,seed=n,validation=val,epochs=epochs,selection=selection)
 if done.exists():
  d=json.loads(done.read_text());assert d["identity"]==ident and all(sha(dest/name)==h for name,h in d["files"].items());return d
 ids=idx(meta,train);testids=idx(meta,[val if phase=="inner" else held])
 assert not set(ids)&set(testids);assert held not in set(meta.iloc[ids].subject)
 sc=scaler(meta,ids,mom);labels=torch.tensor(meta.activity.map(dict(zip(ACT,range(8)))).to_numpy(),dtype=torch.long)
 seed(n);model=b.Net(8,"CNN").cuda();initial=statehash(model);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 if cp.exists():
  st=torch.load(cp,map_location="cpu",weights_only=False);assert st["identity"]==ident
  assert np.array_equal(sc["indices"],st["scaler"]["indices"]) and np.array_equal(sc["mean"],st["scaler"]["mean"])
  model.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"]);b.restore_rng(st["rng"])
 else:
  st=dict(identity=ident,status="training",epoch=0,best_epoch=0,best_loss=float("inf"),bad_epochs=0,history=[],scaler=sc,initial_model_hash=initial,seconds=0.)
 assert st["initial_model_hash"]==initial
 def save():
  st.update(model=model.state_dict(),optimizer=opt.state_dict(),rng=b.rng(),parameters=sum(p.numel() for p in model.parameters()))
  tsave(cp,st)
 torch.cuda.reset_peak_memory_stats()
 if st["status"]=="training":
  xt=scaled(x,ids,arm,sc);yt=labels[ids];wt=torch.tensor(weight(meta,ids,True),dtype=torch.float32)
  if phase=="inner":xv=scaled(x,testids,arm,sc);yv=labels[testids];wv=weight(meta,testids)
  while st["epoch"]<(CFG["max_epochs"] if phase=="inner" else epochs):
   tick=time.perf_counter();tl=b.train_epoch(model,opt,xt,yt,wt);st["epoch"]+=1
   h=dict(epoch=st["epoch"],train_CE=tl)
   if phase=="inner":
    vl,_,_=b.evaluate(model,xv,yv,wv);h["validation_CE"]=vl
    if vl<st["best_loss"]:st.update(best_loss=vl,best_epoch=st["epoch"],bad_epochs=0)
    else:st["bad_epochs"]+=1
    stop=st["epoch"]==CFG["max_epochs"] or (st["epoch"]>=CFG["min_epochs"] and st["bad_epochs"]>=CFG["patience"])
   else:stop=st["epoch"]==epochs
   h["seconds"]=time.perf_counter()-tick;st["history"].append(h);st["seconds"]+=h["seconds"]
   if stop:st["status"]="trained"
   save()
   if stop:break
  del xt,yt,wt
  if phase=="inner":del xv,yv,wv
 if st["status"]=="trained":
  if phase=="refit":
   xx=scaled(x,testids,arm,sc);_,logits,latent=b.evaluate(model,xx,labels[testids],weight(meta,testids));del xx
   prob=torch.softmax(torch.tensor(logits),1).numpy();pr=logits.argmax(1)
   g=meta.iloc[testids].copy();g["original_row"]=testids;g["configuration"]=arm;g["seed"]=n;g["fold"]=held;g["true_index"]=labels[testids].numpy();g["pred_index"]=pr;g["prediction"]=[ACT[j] for j in pr];g["eval_weight"]=weight(meta,testids)
   for j,a in enumerate(ACT):g["logit_"+a]=logits[:,j];g["prob_"+a]=prob[:,j]
   g.to_csv(dest/"predictions.csv",index=False)
   np.savez_compressed(dest/"test_latents.npz",indices=testids,latent=latent)
  st["status"]="done";st["peak_cuda_allocated"]=torch.cuda.max_memory_allocated();save()
 assert st["status"]=="done"
 files=["checkpoint.pt"]+(["predictions.csv","test_latents.npz"] if phase=="refit" else [])
 d=dict(identity=ident,status="complete",best_epoch=st["best_epoch"] if phase=="inner" else epochs,epochs=st["epoch"],seconds=st["seconds"],parameters=8016,initial_model_hash=st["initial_model_hash"],peak_cuda_allocated=st["peak_cuda_allocated"],files={z:sha(dest/z) for z in files})
 jw(done,d);del model,opt;gc.collect();torch.cuda.empty_cache();return d
def run():
 fr=json.loads((O/"freeze_receipt.json").read_text())
 for name,h in fr["artifacts"].items():assert sha(O/name)==h,("frozen changed",name)
 for z in fr["raw"]:assert sha(z["path"])==z["sha256"]
 assert sha(V3/"run_neural_matrix.py")==fr["v3"]["run_neural_matrix.py"]
 import msvcrt
 lf=(O/"RUNNING.lock").open("a+b")
 if lf.tell()==0:lf.write(b"0");lf.flush()
 lf.seek(0);msvcrt.locking(lf.fileno(),msvcrt.LK_NBLCK,1)
 meta=pd.read_csv(O/"windows.csv");x=np.load(O/"windows_float32.npy",mmap_mode="r");mom=b.make_stats(x);splits=json.loads((O/"splits.json").read_text())
 start=time.perf_counter();ni=nf=0;current=None;audit=[];selected=[]
 def status(state,error=None):
  jw(O/"run_status.json",dict(status=state,utc=utc(),pid=os.getpid(),completed=ni+nf,total=144,inner_completed=ni,refit_completed=nf,failed=int(error is not None),current_job=current,error=error,device="cuda:0",elapsed_seconds=time.perf_counter()-start,freeze_sha256=sha(O/"freeze_receipt.json")))
 try:
  # Complete all fixed inner fits before any outer predictions.
  for sp in splits:
   for arm in ARMS:
    best=[]
    for iv in sp["inner"]:
     current=jobkey("inner",sp["test"],arm,SEEDS[0],iv["validation"]);status("running_inner")
     d=run_job(fr,"inner",sp["test"],iv["train"],arm,SEEDS[0],meta,x,mom,val=iv["validation"]);ni+=1;best.append(d["best_epoch"]);audit.append(dict(job=current,**{k:d[k] for k in ["best_epoch","epochs","seconds","parameters","initial_model_hash","peak_cuda_allocated"]}))
     status("running_inner");log(f"DONE {ni+nf}/144 inner={ni}/90 refit={nf}/54 {current} elapsed={time.perf_counter()-start:.1f}s")
    selected.append(dict(subject=sp["test"],arm=arm,inner_best_epochs=best,selected_epoch=int(np.median(best))))
  jw(O/"selected_epochs.json",selected)
  for sp in splits:
   for arm in ARMS:
    sel=next(s for s in selected if s["subject"]==sp["test"] and s["arm"]==arm)
    for n in SEEDS:
     current=jobkey("refit",sp["test"],arm,n);status("running_refit")
     d=run_job(fr,"refit",sp["test"],sp["train"],arm,n,meta,x,mom,epochs=sel["selected_epoch"],selection=sel["inner_best_epochs"]);nf+=1;audit.append(dict(job=current,**{k:d[k] for k in ["best_epoch","epochs","seconds","parameters","initial_model_hash","peak_cuda_allocated"]}))
     status("running_refit");log(f"DONE {ni+nf}/144 inner={ni}/90 refit={nf}/54 {current} elapsed={time.perf_counter()-start:.1f}s")
  parts=[pd.read_csv(p) for p in sorted((O/"jobs").glob("refit*/predictions.csv"))];assert len(parts)==54
  pd.concat(parts,ignore_index=True).to_csv(O/"oof_predictions.csv",index=False);pd.DataFrame(audit).to_csv(O/"fit_audit.csv",index=False)
  assert all(sha(z["path"])==z["sha256"] for z in fr["raw"])
  assert all(sha(V3/name)==h for name,h in fr["v3"].items())
  assert all(sha(V2/z["path"])==z["sha256"] for z in fr["v2"])
  status("training_complete");log("COMPLETE144/144; raw119,v3protected,v2unchanged; no additional fits")
 except BaseException as e:
  status("paused_on_error",repr(e));log(traceback.format_exc());raise
 finally:
  lf.seek(0);msvcrt.locking(lf.fileno(),msvcrt.LK_UNLCK,1);lf.close()
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--run",action="store_true");a=p.parse_args()
 if a.prepare:prepare()
 elif a.run:run()
 else:p.error("choose --prepare or --run")

