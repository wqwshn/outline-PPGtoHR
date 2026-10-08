"""Independent frozen119 source/checkpoint/metric verification. No fitting."""
from pathlib import Path
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import json,hashlib,time,sys,math
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import f1_score,balanced_accuracy_score
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
O=Path(__file__).resolve().parent;V3=O.parent/"hf_scene_classification_v3_lyx24";V2=O.parent/"hf_scene_classification_v2_lyx24"/"execution_20261006_01";P0=O.parent/"hf_scene_classification_v1"/"p0_20261005_02"
A=["HW","RS","HG","TYP","RUN","JJ","BUR","PCH"];S=[20261006,20261007,20261008]
CH=["AccX(g)","AccY(g)","AccZ(g)","GyroX(dps)","GyroY(dps)","GyroZ(dps)","Ut1(mV)","Ut2(mV)"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
def ck(x,msg):
 global checks
 assert x,msg;checks+=1
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv1d(8,16,9,2,4),nn.ReLU(),nn.Conv1d(16,24,7,2,3),nn.ReLU(),nn.Conv1d(24,32,5,2,2),nn.ReLU());self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(32,8))
 def forward(self,x):return self.head(self.conv(x).mean(2))
def wts(g):
 ns=g.subject.nunique();nr=g.groupby(["subject","activity"]).record.nunique();nw=g.groupby("record").size()
 return np.array([1/(ns*8*nr[(r.subject,r.activity)]*nw[r.record]) for r in g.itertuples()])
def metrics(g):
 y=g.true_index.to_numpy(int);p=g.pred_index.to_numpy(int);w=wts(g);cm=np.zeros((8,8));np.add.at(cm,(y,p),w)
 support=cm.sum(1);pred=cm.sum(0);diag=cm.diagonal();f=np.divide(2*diag,support+pred,out=np.zeros(8),where=support+pred>0);rec=np.divide(diag,support,out=np.zeros(8),where=support>0)
 ck(np.isclose(f.mean(),f1_score(y,p,labels=range(8),average="macro",sample_weight=w,zero_division=0)),"F1sklearn")
 ck(np.isclose(rec.mean(),balanced_accuracy_score(y,p,sample_weight=w)),"BAsklearn")
 return dict(macro_F1=f.mean(),BA=rec.mean(),unweighted_window_accuracy=np.mean(y==p)),cm,f,rec
def main():
 start=time.perf_counter();fr=json.loads((O/"freeze_receipt.json").read_text());cfg=json.loads((O/"config.json").read_text())
 ck(json.loads((O/"run_status.json").read_text())["status"]=="training_complete","training_complete")
 for n,h in fr["artifacts"].items():ck(sha(O/n)==h,"frozen"+n)
 for n,h in fr["source_P0"].items():ck(sha(P0/n)==h,"P0"+n)
 for n,h in fr["v3"].items():ck(sha(V3/n)==h,"v3"+n)
 for z in fr["v2"]:ck(sha(V2/z["path"])==z["sha256"],"v2"+z["path"])
 meta=pd.read_csv(O/"windows.csv");rec=pd.read_csv(O/"records_LOCAL.csv");x=np.load(O/"windows_float32.npy",mmap_mode="r")
 ck(x.shape==(9520,8,800),"shape");im=pd.read_csv(O/"imputation.csv").set_index("original_row")
 # Independently rebuild original source windows using pandas interpolation.
 for r in rec.itertuples():
  ck(sha(r.input_path)==r.input_sha256,"rawhash"+r.record)
  d=pd.read_csv(r.input_path)[CH].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
  for i,z in meta[meta.record==r.record].iterrows():
   q=d.iloc[int(z.start_row):int(z.end_row_exclusive)].copy()
   ck(q.isna().any(axis=1).sum()==im.loc[i,"imputed_rows"],"missingcount")
   ar=q.interpolate(method="linear",limit_direction="both").to_numpy(dtype=np.float32).T
   ck(np.array_equal(ar,x[i]),"raw reconstruction"+str(i))
 # Independent per-record moments and hierarchical aggregation.
 recordmom={}
 for rid,g in meta.groupby("record"):
  xx=x[g.index];mu=xx.mean(axis=(0,2),dtype=np.float64);ss=np.array([(xx[:,j,:].astype(np.float64)**2).mean() for j in range(8)])
  recordmom[rid]=(mu,ss)
 oo=pd.read_csv(O/"oof_predictions.csv");ck(len(oo)==9520*9,"OOFcount")
 ck(not oo.duplicated(["configuration","seed","original_row"]).any(),"OOFunique")
 for (arm,n),g in oo.groupby(["configuration","seed"]):
  ck(set(g.original_row)==set(range(9520)),"OOFcoverage")
  ck(np.array_equal(g.subject.to_numpy(),meta.iloc[g.original_row].subject.to_numpy()),"subjectids")
  ck(np.array_equal(g.activity.to_numpy(),meta.iloc[g.original_row].activity.to_numpy()),"labels")
  pr=g[[f"prob_{a}" for a in A]].to_numpy();lo=g[[f"logit_{a}" for a in A]].to_numpy(np.float32).astype(float);ex=np.exp(lo-lo.max(1,keepdims=True));ex/=ex.sum(1,keepdims=True)
  ck(np.max(abs(pr-ex))<1e-6,"independentF64softmax");ck(np.max(abs(pr-torch.softmax(torch.tensor(lo.astype(np.float32)),1).numpy()))<5e-8,"originalFP32softmax");ck(np.array_equal(pr.argmax(1),g.pred_index),"argmax")
  for sub,q in g.groupby("subject"):ck(np.allclose(q.eval_weight,wts(q),rtol=0,atol=1e-15),"evalweights")
 selections=json.loads((O/"selected_epochs.json").read_text());hist=[];audit=[];maxerr=0.;maxmean=0.;tie=[];initial={}
 donefiles=sorted((O/"jobs").glob("*/done.json"));ck(len(donefiles)==144,"144jobs")
 for dp in donefiles:
  dest=dp.parent;d=json.loads(dp.read_text());ident=d["identity"];st=torch.load(dest/"checkpoint.pt",map_location="cpu",weights_only=False)
  ck(st["identity"]==ident and ident["freeze_sha256"]==sha(O/"freeze_receipt.json"),"identity")
  for n,h in d["files"].items():ck(sha(dest/n)==h,"jobfile"+n)
  ck(st["status"]=="done" and d["parameters"]==8016,"done8016")
  ids=np.flatnonzero(meta.subject.isin(ident["train"]));sc=st["scaler"]
  ck(np.array_equal(ids,sc["indices"]),"trainids");ck(ident["held"] not in ident["train"],"heldoutperson")
  ck(np.allclose(wts(meta.iloc[ids]),sc["weights"],rtol=0,atol=1e-16),"trainweights")
  if ident["phase"]=="inner":ck(len(ident["train"])==4 and ident["validation"] not in ident["train"] and ident["validation"]!=ident["held"],"innerpersonisolation")
  else:ck(len(ident["train"])==5,"refit5")
  means=[];seconds=[]
  for sub in ident["train"]:
   am=[];ass=[]
   for a in A:
    records=rec[(rec.subject==sub)&(rec.activity==a)].record.tolist();am.append(np.mean([recordmom[r][0] for r in records],axis=0));ass.append(np.mean([recordmom[r][1] for r in records],axis=0))
   means.append(np.mean(am,axis=0));seconds.append(np.mean(ass,axis=0))
  mu=np.mean(means,axis=0);var=np.maximum(np.mean(seconds,axis=0)-mu*mu,0);scale=np.sqrt(var);scale[scale==0]=1
  maxmean=max(maxmean,float(np.max(abs(mu-sc["mean"]))));ck(np.allclose(mu,sc["mean"],rtol=0,atol=1e-8),"scalermean");ck(np.allclose(scale,sc["scale"],rtol=1e-8,atol=1e-8),"scalerscale")
  histories=st["history"]
  ck(len(histories)==d["epochs"],"historyepochs")
  for h in histories:hist.append(dict(job=dest.name,phase=ident["phase"],held=ident["held"],arm=ident["arm"],seed=ident["seed"],**h))
  if ident["phase"]=="inner":
   best=int(np.argmin([h["validation_CE"] for h in histories]))+1;ck(best==d["best_epoch"]==st["best_epoch"],"earliestbestepoch")
   ck(15<=len(histories)<=40,"minexecuted");sel=next(z for z in selections if z["subject"]==ident["held"] and z["arm"]==ident["arm"]);ck(best in sel["inner_best_epochs"],"selectionmember")
  else:
   sel=next(z for z in selections if z["subject"]==ident["held"] and z["arm"]==ident["arm"]);ck(d["epochs"]==sel["selected_epoch"]==int(np.median(sel["inner_best_epochs"])),"medianrefit")
   ck(ident["selection"]==sel["inner_best_epochs"],"selectionidentity")
   g=pd.read_csv(dest/"predictions.csv");ti=g.original_row.to_numpy(int);ck(np.all(meta.iloc[ti].subject==ident["held"]),"testsubject")
   model=Net().cuda();model.load_state_dict(st["model"]);model.eval();vals=[]
   cols={"MIMU":range(6),"HF":[6,7],"FUS":range(8)}[ident["arm"]];inactive=[j for j in range(8) if j not in cols]
   with torch.no_grad():
    for k in range(0,len(ti),64):
     xx=x[ti[k:k+64]].astype(float);xx=(xx-sc["mean"][None,:,None])/sc["scale"][None,:,None];xx[:,inactive]=0
     vals.append(model(torch.from_numpy(xx.astype(np.float32)).cuda()).cpu().numpy())
   logits=np.concatenate(vals);saved=g[[f"logit_{a}" for a in A]].to_numpy(np.float32)
   maxerr=max(maxerr,float(abs(logits-saved).max()));ck(np.allclose(logits,saved,rtol=1e-6,atol=3e-5),"reloadlogits");ck(np.array_equal(logits.argmax(1),g.pred_index),"reloadlabels")
   match=oo[(oo.configuration==ident["arm"])&(oo.seed==ident["seed"])&(oo.subject==ident["held"])].sort_values("original_row")
   ck(np.array_equal(match.pred_index,g.sort_values("original_row").pred_index),"aggregationmatch")
   del model
  initial.setdefault(ident["seed"],set()).add(st["initial_model_hash"])
  audit.append(dict(job=dest.name,phase=ident["phase"],held=ident["held"],arm=ident["arm"],seed=ident["seed"],epochs=d["epochs"],selected_epoch=d["best_epoch"],seconds=d["seconds"]))
 ck(all(len(v)==1 for v in initial.values()),"sameinitializationacrossarms")
 for z in selections:
  actual=[json.loads((O/"jobs"/f"inner__{z['subject']}__{z['arm']}__s{S[0]}__val{s}"/"done.json").read_text())["best_epoch"] for s in cfg["subjects"] if s!=z["subject"]]
  ck(actual==z["inner_best_epochs"],"all5epochorder")
 # No-fit probability controls, paired windows and fixed cyclic seed rule.
 parts=[oo]
 for n in S:
  m=oo[(oo.configuration=="MIMU")&(oo.seed==n)].sort_values("original_row")
  h=oo[(oo.configuration=="HF")&(oo.seed==n)].sort_values("original_row")
  nxt=S[(S.index(n)+1)%3];e=oo[(oo.configuration=="MIMU")&(oo.seed==nxt)].sort_values("original_row")
  for name,other in [("HF_LATE",h),("MIMU_ENS",e)]:
   ck(np.array_equal(m.original_row,other.original_row),"controlalignment");g=m.copy();prob=.75*m[[f"prob_{a}" for a in A]].to_numpy()+.25*other[[f"prob_{a}" for a in A]].to_numpy();g["configuration"]=name;g["pred_index"]=prob.argmax(1);g["prediction"]=[A[j] for j in prob.argmax(1)]
   for j,a in enumerate(A):g["prob_"+a]=prob[:,j];g["logit_"+a]=np.nan
   parts.append(g)
 full=pd.concat(parts,ignore_index=True);full.to_csv(O/"all_predictions_with_controls.csv",index=False)
 sm=[];rm=[];cmrows=[];ac=[]
 for (conf,n,sub),g in full.groupby(["configuration","seed","subject"]):
  met,cm,f,rc=metrics(g);sm.append(dict(configuration=conf,seed=n,subject=sub,**met))
  for j,a in enumerate(A):
   ac.append(dict(configuration=conf,seed=n,subject=sub,activity=a,F1=f[j],recall=rc[j]))
   for k,pa in enumerate(A):cmrows.append(dict(configuration=conf,seed=n,subject=sub,true=a,predicted=pa,weight=cm[j,k]))
  for rid,q in g.groupby("record"):
   counts=np.bincount(q.pred_index,minlength=8);tied=np.flatnonzero(counts==counts.max());vote=int(tied[0]);true=int(q.true_index.iloc[0])
   rm.append(dict(configuration=conf,seed=n,subject=sub,record=rid,activity=A[true],windows=len(q),window_accuracy=np.mean(q.pred_index==true),vote=A[vote],correct=int(vote==true),tie=len(tied)>1,tie_as_wrong_correct=int(vote==true and len(tied)==1)))
   if len(tied)>1:tie.append(dict(configuration=conf,seed=int(n),subject=sub,record=rid,votes={A[j]:int(counts[j]) for j in range(8) if counts[j]},winner=A[vote]))
 sm=pd.DataFrame(sm);rm=pd.DataFrame(rm);sm.to_csv(O/"subject_metrics.csv",index=False);rm.to_csv(O/"record_metrics.csv",index=False);pd.DataFrame(cmrows).to_csv(O/"confusion_matrices.csv",index=False);pd.DataFrame(ac).to_csv(O/"activity_metrics.csv",index=False)
 sd=sm.groupby(["configuration","seed"])[["macro_F1","BA","unweighted_window_accuracy"]].mean().reset_index();sd.to_csv(O/"summary_by_seed.csv",index=False)
 sd.groupby("configuration")[["macro_F1","BA","unweighted_window_accuracy"]].agg(["mean","std","min","max"]).to_csv(O/"summary_across_seeds.csv")
 pair=[]
 for (n,sub),g in sm.groupby(["seed","subject"]):
  v=g.set_index("configuration")
  for conf in ["FUS","HF_LATE","MIMU_ENS"]:
   pair.append(dict(seed=n,subject=sub,configuration=conf,delta_macro_F1=v.loc[conf,"macro_F1"]-v.loc["MIMU","macro_F1"],delta_BA=v.loc[conf,"BA"]-v.loc["MIMU","BA"]))
 pd.DataFrame(pair).to_csv(O/"paired_deltas.csv",index=False);pd.DataFrame(hist).to_csv(O/"learning_curves.csv",index=False);pd.DataFrame(audit).to_csv(O/"verified_fit_audit.csv",index=False)
 comp=[]
 for (n,sub),m in full[full.configuration=="MIMU"].groupby(["seed","subject"]):
  m=m.sort_values("original_row");base=m.pred_index.to_numpy()==m.true_index.to_numpy()
  for conf in ["FUS","HF_LATE","MIMU_ENS"]:
   g=full[(full.configuration==conf)&(full.seed==n)&(full.subject==sub)].sort_values("original_row");ok=g.pred_index.to_numpy()==g.true_index.to_numpy();missing=im.loc[g.original_row,"imputed_rows"].to_numpy()>0
   for label,mask in [("all",np.ones(len(g),bool)),("complete",~missing),("imputed",missing)]:
    comp.append(dict(seed=n,subject=sub,configuration=conf,stratum=label,windows=int(mask.sum()),rescued=int((~base&ok&mask).sum()),harmed=int((base&~ok&mask).sum()),both_wrong=int((~base&~ok&mask).sum()),both_correct=int((base&ok&mask).sum())))
 pd.DataFrame(comp).to_csv(O/"complementarity_missingness.csv",index=False)
 (O/"vote_ties.json").write_text(json.dumps(tie,indent=2),encoding="utf-8")
 qa=dict(status="PASS",checks=checks,raw_reconstructed_windows=9520,raw_files=119,checkpoints_verified=144,refit_checkpoints_reloaded=54,OOF_rows=len(oo),all_predictions_rows=len(full),prediction_labels_identical=True,max_logit_reload_error=maxerr,max_scaler_mean_difference=maxmean,raw_v3_v2_unchanged=True,all_subjects_seeds_arms=True,record_vote_ties=len(tie),seconds=time.perf_counter()-start,verifier_sha256=sha(__file__))
 (O/"independent_verification.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
 print(json.dumps(qa),flush=True);print(sd.to_string(index=False),flush=True)
if __name__=="__main__":main()

