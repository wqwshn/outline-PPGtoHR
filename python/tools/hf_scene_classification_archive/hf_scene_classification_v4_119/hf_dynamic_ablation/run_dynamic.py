"""Predeclared two-arm HF dynamic ablation. Uses frozen main engine without changing it."""
from pathlib import Path
import os,sys,json,time,datetime,importlib.util,argparse,traceback
import numpy as np,pandas as pd,torch
O=Path(__file__).resolve().parent;MAIN=O.parent
spec=importlib.util.spec_from_file_location("fixed119engine",MAIN/"run_119.py");v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
raw_scaled=v.scaled;raw_scaler=v.scaler
PLAN=json.loads((O/"plan_freeze.json").read_text())
ARMS={"HF_dynamic":[6,7],"FUS_dynamic":list(range(8))}
CFG=json.loads((MAIN/"config.json").read_text());CFG.update(experiment="119_HF_dynamic_predeclared_ablation",arms=ARMS,inner_fits=60,refit_fits=36,total_fits=96,primary="FUS_dynamic-MIMU from fixed raw main matrix; secondaryFUS_dynamic-FUS_raw",HF_transform=PLAN["transform"],HF_scale=PLAN["scale"],zero_scale_threshold_mV=1e-12,zero_scale_fallback_mV=1.,case_rules="Primary same mainrecord rank scheme for dynamicFUSminusMIMU plus fixedHG07 and originalLYXTYP03 byrawhash; all contrary cases retained")
v.O=O;v.ARMS=ARMS;v.CFG=CFG
def moments(x):
 m,s=v.b.make_stats(x)
 m[:,6:]=0
 for j in [6,7]:
  for k in range(0,len(x),256):
   z=x[k:k+256,j,:].astype(np.float64);z-=z.mean(1,keepdims=True);s[k:k+256,j]=np.mean(z*z,axis=1)
 return m,s
def dynamic_scaler(meta,ids,mom):
 sc=raw_scaler(meta,ids,mom)
 rms=np.sqrt(sc["variance"][6:]);fallback=rms<=CFG["zero_scale_threshold_mV"];sc["scale"][6:]=np.where(fallback,1.,rms);sc["HF_centered_RMS_mV"]=rms;sc["HF_zero_scale_fallback"]=fallback
 assert np.all(sc["mean"][6:]==0)
 return sc
def dynamic_scaled(x,ids,arm,sc):
 out=np.empty((len(ids),8,800),np.float32);inactive=[j for j in range(8) if j not in ARMS[arm]]
 for k in range(0,len(ids),128):
  z=x[ids[k:k+128]].astype(np.float64)
  z[:,6:]-=z[:,6:].mean(axis=2,keepdims=True)
  z-=sc["mean"][None,:,None];z/=sc["scale"][None,:,None];z[:,inactive]=0
  out[k:k+128]=z
 return torch.from_numpy(out)
v.scaler=dynamic_scaler;v.scaled=dynamic_scaled
def prepare():
 assert not (O/"freeze_receipt.json").exists()
 fr=json.loads((MAIN/"freeze_receipt.json").read_text())
 for n,h in fr["artifacts"].items():assert v.sha(MAIN/n)==h
 meta=pd.read_csv(MAIN/"windows.csv");x=np.load(MAIN/"windows_float32.npy",mmap_mode="r")
 mom=moments(x);orig=v.b.make_stats(x);ids=v.idx(meta,CFG["subjects"][1:]);sc=dynamic_scaler(meta,ids,mom);rawsc=raw_scaler(meta,ids,orig)
 assert np.array_equal(sc["mean"][:6],rawsc["mean"][:6]) and np.array_equal(sc["scale"][:6],rawsc["scale"][:6])
 a=dynamic_scaled(x,ids[:64],"FUS_dynamic",sc).numpy()
 # Raw function captured earlier resolves mutatedARMS; explicit fullraw comparison.
 z=x[ids[:64]].astype(np.float64);z=(z-rawsc["mean"][None,:,None])/rawsc["scale"][None,:,None]
 assert np.array_equal(a[:,:6],z[:,:6].astype(np.float32))
 assert abs(a[:,6:].mean(axis=2)).max()<1e-6
 h=dynamic_scaled(x,ids[:64],"HF_dynamic",sc).numpy();assert np.all(h[:,:6]==0) and np.array_equal(h[:,6:],a[:,6:])
 # Constant HF and lineartrend synthetic tests; offset invariance and no detrend.
 xx=np.zeros((2,8,800),np.float32);xx[:,6]=np.arange(800)*.25+1000;xx[:,7]=1400
 simple=dict(mean=np.zeros(8),scale=np.ones(8))
 before=dynamic_scaled(xx,np.array([0,1]),"HF_dynamic",simple).numpy()
 xx[:,6:]+=256;after=dynamic_scaled(xx,np.array([0,1]),"HF_dynamic",simple).numpy()
 assert np.array_equal(before,after) and np.all(before[:,7]==0) and before[:,6].std()>10
 # Zero fallback uses fixedthreshold and never fitted perwindow.
 sm=pd.DataFrame([dict(subject="s",activity=a,record=a) for a in v.ACT])
 zz=np.zeros((8,8,800),np.float32);zs=dynamic_scaler(sm,np.arange(8),moments(zz));assert zs["HF_zero_scale_fallback"].all() and np.all(zs["scale"][6:]==1)
 # Every phase/scaler preserves mainMIMU weightedstats, not just one example.
 splits=json.loads((MAIN/"splits.json").read_text());scalerchecks=0
 for sp in splits:
  for train in [sp["train"]]+[z["train"] for z in sp["inner"]]:
   ids=v.idx(meta,train);ds=dynamic_scaler(meta,ids,mom);rs=raw_scaler(meta,ids,orig)
   assert np.array_equal(ds["mean"][:6],rs["mean"][:6]) and np.array_equal(ds["scale"][:6],rs["scale"][:6]);assert np.array_equal(ds["weights"],rs["weights"])
   scalerchecks+=1
 # TYP03 reference by original hash, never assume same recordID.
 lyx=pd.read_csv(v.V3/"records_LOCAL.csv");typ=lyx[lyx.record=="TYP-03"].input_sha256.iloc[0];rr=pd.read_csv(MAIN/"records_LOCAL.csv");mapped=rr[rr.input_sha256==typ][["record","subject","activity","input_sha256"]].to_dict("records")
 assert len(mapped)<=1
 tests=dict(status="PASS",utc=v.utc(),formal_dynamic_fits=0,checks=["HFoffsetinvariance","slowtrendretained","constantHFfallback","zeroInactive","HFmatchingbetweenarms","all36phaseMIMUscalersidentical","representativeMIMUtensorsbitidentical"],MIMU_scaler_checks=scalerchecks,LYX_TYP03_sha256=typ,LYX_TYP03_mapping=mapped)
 v.jw(O/"unit_tests.json",tests);v.jw(O/"config.json",CFG);v.jw(O/"splits.json",splits)
 freeze=dict(status="frozen_before_dynamic_training",utc=v.utc(),outer_scores_viewed=False,plan_sha256=v.sha(O/"plan_freeze.json"),main_freeze_sha256=v.sha(MAIN/"freeze_receipt.json"),main_artifacts=fr["artifacts"],artifacts={n:v.sha(O/n) for n in ["run_dynamic.py","plan_freeze.json","config.json","splits.json","unit_tests.json"]},raw=fr["raw"],MIMU_reuse="exactmaininputscalersweightsseedsarchitecturebatchpolicyinnerselection; no reselectingMIMU",TYP03=tests["LYX_TYP03_mapping"])
 v.jw(O/"freeze_receipt.json",freeze);v.jw(O/"run_status.json",dict(status="frozen_waiting_main",utc=v.utc(),completed=0,total=96,failed=0))
 print("DYNAMIC_FROZEN",json.dumps(freeze["artifacts"]),"TYP03",mapped,flush=True)
def run():
 assert json.loads((MAIN/"run_status.json").read_text())["status"]=="training_complete","WaitforprimaryGPUrunner"
 fr=json.loads((O/"freeze_receipt.json").read_text())
 for n,h in fr["artifacts"].items():assert v.sha(O/n)==h
 for n,h in fr["main_artifacts"].items():assert v.sha(MAIN/n)==h
 assert v.sha(MAIN/"freeze_receipt.json")==fr["main_freeze_sha256"]
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(PLAN["latest_new_configuration_start_UTC"]),"Too late to start newmatrix"
 import msvcrt
 lf=(O/"RUNNING.lock").open("a+b")
 if lf.tell()==0:lf.write(b"0");lf.flush()
 lf.seek(0);msvcrt.locking(lf.fileno(),msvcrt.LK_NBLCK,1)
 # Protect completed primary models/predictions, not evolving report/QA files.
 mainprotected={str(p.relative_to(MAIN)):v.sha(p) for p in (MAIN/"jobs").glob("*/*") if p.is_file()};mainprotected["oof_predictions.csv"]=v.sha(MAIN/"oof_predictions.csv")
 v.jw(O/"main_completed_artifact_hashes.json",mainprotected)
 meta=pd.read_csv(MAIN/"windows.csv");x=np.load(MAIN/"windows_float32.npy",mmap_mode="r");mom=moments(x);splits=json.loads((O/"splits.json").read_text())
 ni=nf=0;start=time.perf_counter();current=None;selected=[];audit=[]
 def status(state,error=None):
  v.jw(O/"run_status.json",dict(status=state,utc=v.utc(),pid=os.getpid(),completed=ni+nf,total=96,inner_completed=ni,refit_completed=nf,failed=int(error is not None),current_job=current,error=error,elapsed_seconds=time.perf_counter()-start,freeze_sha256=v.sha(O/"freeze_receipt.json")))
 try:
  for sp in splits:
   for arm in ARMS:
    best=[]
    for iv in sp["inner"]:
     current=v.jobkey("inner",sp["test"],arm,v.SEEDS[0],iv["validation"]);status("running_inner")
     d=v.run_job(fr,"inner",sp["test"],iv["train"],arm,v.SEEDS[0],meta,x,mom,val=iv["validation"]);ni+=1;best.append(d["best_epoch"]);audit.append(dict(job=current,epochs=d["epochs"],selected_epoch=d["best_epoch"],seconds=d["seconds"]));status("running_inner");v.log(f"DONE {ni+nf}/96 inner={ni}/60 refit={nf}/36 {current}")
    selected.append(dict(subject=sp["test"],arm=arm,inner_best_epochs=best,selected_epoch=int(np.median(best))))
  v.jw(O/"selected_epochs.json",selected)
  for sp in splits:
   for arm in ARMS:
    sel=next(z for z in selected if z["subject"]==sp["test"] and z["arm"]==arm)
    for n in v.SEEDS:
     current=v.jobkey("refit",sp["test"],arm,n);status("running_refit")
     d=v.run_job(fr,"refit",sp["test"],sp["train"],arm,n,meta,x,mom,epochs=sel["selected_epoch"],selection=sel["inner_best_epochs"]);nf+=1;audit.append(dict(job=current,epochs=d["epochs"],selected_epoch=d["best_epoch"],seconds=d["seconds"]));status("running_refit");v.log(f"DONE {ni+nf}/96 inner={ni}/60 refit={nf}/36 {current}")
  parts=[pd.read_csv(p) for p in sorted((O/"jobs").glob("refit*/predictions.csv"))];assert len(parts)==36
  pd.concat(parts,ignore_index=True).to_csv(O/"oof_predictions.csv",index=False);pd.DataFrame(audit).to_csv(O/"fit_audit.csv",index=False)
  assert all(v.sha(z["path"])==z["sha256"] for z in fr["raw"])
  assert all(v.sha(MAIN/n)==h for n,h in mainprotected.items())
  status("training_complete");v.log("COMPLETE96fits; main144models/predictions and119rawunchanged")
 except BaseException as e:
  status("paused_on_error",repr(e));v.log(traceback.format_exc());raise
 finally:
  lf.seek(0);msvcrt.locking(lf.fileno(),msvcrt.LK_UNLCK,1);lf.close()
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--run",action="store_true");a=p.parse_args()
 if a.prepare:prepare()
 elif a.run:run()
 else:p.error("choose --prepare or --run")

