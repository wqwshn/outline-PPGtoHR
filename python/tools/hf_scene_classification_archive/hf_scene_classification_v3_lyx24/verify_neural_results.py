"""Independent verification and summaries of the frozen v3 run; never trains."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from pathlib import Path
import json,hashlib,itertools,time,sys
import numpy as np,pandas as pd,torch
from torch import nn
torch.set_num_threads(4);torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
O=Path(__file__).resolve().parent;V2=O.parent/"hf_scene_classification_v2_lyx24"/"execution_20261006_01"
ACT=["HW","RS","HG","TYP","RUN","JJ","BUR","PCH"];ARMS={"MIMU":list(range(6)),"HF":[6,7],"FUS":list(range(8))}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jw(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
class ReloadNet(nn.Module):
 def __init__(self,c,arch):
  super().__init__();self.arch=arch
  self.conv=nn.Sequential(nn.Conv1d(c,16,9,2,4),nn.ReLU(),nn.Conv1d(16,24,7,2,3),nn.ReLU(),nn.Conv1d(24,32,5,2,2),nn.ReLU())
  if arch=="CNNLSTM":self.lstm=nn.LSTM(32,32,1,batch_first=True)
  self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(32,8))
 def forward(self,x):
  x=self.conv(x)
  if self.arch=="CNNLSTM":x,_=self.lstm(x.transpose(1,2));x=x.mean(1)
  else:x=x.mean(2)
  return self.head(x)
nchecks=0;maxdiff=0.
def check(ok,msg):
 global nchecks
 if not bool(ok):raise AssertionError(msg)
 nchecks+=1
def close(a,b,msg,atol=1e-8,rtol=1e-7):
 global maxdiff
 aa=np.asarray(a);bb=np.asarray(b);maxdiff=max(maxdiff,float(np.max(abs(aa-bb))) if aa.size else 0)
 check(np.allclose(aa,bb,atol=atol,rtol=rtol),msg)
def metrics(g):
 cm=np.zeros((8,8))
 for a,b,w in zip(g.true_index,g.pred_index,g.eval_weight):cm[int(a),int(b)]+=w
 tp=np.diag(cm);rec=np.divide(tp,cm.sum(1),out=np.zeros(8),where=cm.sum(1)>0)
 f=np.divide(2*tp,cm.sum(1)+cm.sum(0),out=np.zeros(8),where=cm.sum(1)+cm.sum(0)>0)
 return cm,f,rec
def source_weights(meta,ids):
 groups=meta.iloc[ids];records=groups.groupby("activity").record.nunique();sizes=groups.groupby("record").size()
 return np.array([1/(8*records[row.activity]*sizes[row.record]) for row in groups.itertuples()])
def main():
 started=time.perf_counter();fr=json.loads((O/"freeze_receipt.json").read_text(encoding="utf-8"));cfg=json.loads((O/"config.json").read_text());sp=json.loads((O/"splits.json").read_text())
 check(json.loads((O/"run_status.json").read_text())["status"]=="training_complete","training not complete")
 check(fr["formal_fits_started"]==0 and fr["formal_outer_scores_observed"]==0,"freeze order")
 for name,h in fr["artifacts"].items():check(sha(O/name)==h,"frozen hash "+name)
 check(len({r["sha256"] for r in fr["raw"]})==24,"distinct record input hashes")
 for r in fr["raw"]:check(sha(r["path"])==r["sha256"],"raw hash "+r["record"])
 for r in fr["v2_baseline"]:check(sha(V2/r["path"])==r["sha256"],"V2 changed "+r["path"])
 meta=pd.read_csv(O/"windows.csv");x=np.load(O/"windows_float32.npy")
 check(x.shape==(1941,8,800) and np.isfinite(x).all(),"cache shape finite")
 # Rebuild every frozen window from original values with independent pandas interpolation.
 ch=cfg["channels"]
 for r in fr["raw"]:
  raw=pd.read_csv(r["path"])[ch]
  for i,row in meta[meta.record==r["record"]].iterrows():
   a=raw.iloc[int(row.start_row):int(row.end_row_exclusive)].apply(pd.to_numeric,errors="coerce").interpolate(method="linear",limit_direction="both").to_numpy().T.astype(np.float32)
   close(a,x[i],"raw/cache "+str(i),atol=0,rtol=0)
 # Scaler reconstructed by equal averaging of per-record window/sample first and second moments.
 record_mom={}
 for rid in meta.record.unique():
  z=x[meta.record==rid].astype(np.float64)
  record_mom[rid]=(z.mean(axis=(0,2)),np.square(z).mean(axis=(0,2)))
 def expected_scale(records):
  means=[];moments=[]
  for a in ACT:
   rr=[rid for rid in records if meta[meta.record==rid].activity.iloc[0]==a]
   check(len(rr)>0,"class coverage")
   means.append(np.mean([record_mom[rid][0] for rid in rr],axis=0));moments.append(np.mean([record_mom[rid][1] for rid in rr],axis=0))
  mu=np.mean(means,axis=0);var=np.maximum(np.mean(moments,axis=0)-mu*mu,0);scale=np.sqrt(var);scale[scale==0]=1
  return mu,var,scale
 p=pd.read_csv(O/"oof_predictions.csv");auxpred=pd.read_csv(O/"aux_predictions.csv")
 check(len(p)==1941*6*3,"OOF row count")
 check(p.groupby(["configuration","seed","original_row"]).size().eq(1).all(),"OOF uniqueness")
 check(set(p.seed)==set(cfg["seeds"]) and set(p.configuration)==set(cfg["models"]),"all seeds/configs")
 for (config,seed),g in p.groupby(["configuration","seed"]):check(set(g.original_row)==set(range(1941)),"all windows")
 check(p.fold.eq(p.record).all(),"complete-record holdout identity")
 expected_w=np.array([1/(24*len(meta[meta.record==rid])) for rid in p.record]);close(p.eval_weight,expected_w,"evaluation weights")
 logits=p[["logit_"+a for a in ACT]].to_numpy(dtype=np.float32).astype(np.float64);pr=logits.argmax(1)
 check(np.array_equal(pr,p.pred_index),"argmax saved logits")
 check(np.isfinite(logits).all(),"finite logits")
 prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
 close(prob,p[["prob_"+a for a in ACT]].to_numpy(),"saved softmax",atol=2e-7)
 summary=[];recrows=[];classrows=[];confrows=[]
 for (config,seed),g in p.groupby(["configuration","seed"],sort=True):
  cm,f,rec=metrics(g);correct_votes=0
  for rid,gg in g.groupby("record"):
   hist=np.bincount(gg.pred_index,minlength=8);vote=int(np.argmax(hist));truth=int(gg.true_index.iloc[0]);right=vote==truth;correct_votes+=right
   recrows.append(dict(configuration=config,seed=int(seed),record=rid,activity=ACT[truth],windows=len(gg),window_accuracy=float((gg.pred_index==truth).mean()),majority_prediction=ACT[vote],majority_correct=bool(right),tie_count=int((hist==hist.max()).sum())))
  summary.append(dict(configuration=config,seed=int(seed),macro_F1=float(f.mean()),balanced_accuracy=float(rec.mean()),weighted_window_accuracy=float(np.trace(cm)/cm.sum()),unweighted_window_accuracy=float((g.true_index==g.pred_index).mean()),correct_majority_records=int(correct_votes),majority_record_accuracy=correct_votes/24))
  for j,a in enumerate(ACT):
   classrows.append(dict(configuration=config,seed=int(seed),activity=a,F1=float(f[j]),recall=float(rec[j])))
   for k,b in enumerate(ACT):confrows.append(dict(configuration=config,seed=int(seed),true_activity=a,predicted_activity=b,weight=float(cm[j,k])))
 sums=pd.DataFrame(summary);sums.to_csv(O/"summary_by_seed.csv",index=False)
 pd.DataFrame(recrows).to_csv(O/"record_metrics.csv",index=False);pd.DataFrame(classrows).to_csv(O/"activity_metrics.csv",index=False);pd.DataFrame(confrows).to_csv(O/"confusion_matrices.csv",index=False)
 agg=sums.groupby("configuration").agg({col:["mean","std","min","max"] for col in ["macro_F1","balanced_accuracy","weighted_window_accuracy","unweighted_window_accuracy","correct_majority_records","majority_record_accuracy"]});agg.columns=["_".join(c) for c in agg.columns];agg.to_csv(O/"summary_across_seeds.csv")
 anchor=pd.read_csv(V2/"summary_metrics.csv").set_index("configuration").loc["SVC_MIMU"]
 paired=[];comp=[]
 for arch in ["CNN","CNNLSTM"]:
  for seed in cfg["seeds"]:
   ss=sums.set_index(["configuration","seed"]);base=ss.loc[(arch+"_MIMU",seed)];fus=ss.loc[(arch+"_FUS",seed)]
   paired.append(dict(architecture=arch,seed=seed,fusion_minus_MIMU_macro_F1=float(fus.macro_F1-base.macro_F1),fusion_minus_MIMU_BA=float(fus.balanced_accuracy-base.balanced_accuracy),MIMU_minus_v2_SVC_macro_F1=float(base.macro_F1-anchor.macro_F1),fusion_minus_v2_SVC_macro_F1=float(fus.macro_F1-anchor.macro_F1)))
   b=p[(p.configuration==arch+"_MIMU")&(p.seed==seed)].sort_values("original_row");ff=p[(p.configuration==arch+"_FUS")&(p.seed==seed)].sort_values("original_row")
   bc=(b.true_index==b.pred_index).to_numpy();fc=(ff.true_index==ff.pred_index).to_numpy()
   for act in ["ALL"]+ACT:
    mask=np.ones(len(b),bool) if act=="ALL" else b.activity.to_numpy()==act;w=b.eval_weight.to_numpy()[mask]
    res=(~bc&fc)[mask];dam=(bc&~fc)[mask]
    comp.append(dict(architecture=arch,seed=seed,activity=act,rescued=int(res.sum()),damaged=int(dam.sum()),weighted_rescued=float(w[res].sum()/w.sum()),weighted_damaged=float(w[dam].sum()/w.sum())))
 pd.DataFrame(paired).to_csv(O/"paired_deltas.csv",index=False);pd.DataFrame(comp).to_csv(O/"complementarity.csv",index=False)
 # Audit every independent training job/checkpoint and reproduce all saved predictions.
 jobrows=[];history=[];reload_max=0.;scale_max=0.
 splitmap={s["fold"]:s for s in sp["main"]};splitmap["AUX16_8"]=sp["aux"]
 alljobs=sorted((O/"jobs").iterdir());check(len(alljobs)==438,"job count")
 for i,dest in enumerate(alljobs):
  dd=json.loads((dest/"done.json").read_text());ident=dd["identity"]
  check(ident["freeze_sha256"]==sha(O/"freeze_receipt.json"),"checkpoint freeze identity")
  for n,h in dd["files"].items():check(sha(dest/n)==h,"job file hash")
  st=torch.load(dest/"checkpoint.pt",map_location="cpu",weights_only=False);check(st["identity"]==ident and st["phase"]=="done","completed checkpoint identity")
  ss=splitmap[ident["fold"]]
  check(set(ss["inner_train"]).isdisjoint(ss["inner_val"]) and set(ss["outer_train"]).isdisjoint(ss["test"]),"record separation")
  check(set(ss["outer_train"])==set(ss["inner_train"])|set(ss["inner_val"]),"refit includes only outertrain")
  check(st["initial_model_hash"]==st["refit_initial_model_hash"],"fresh initialization")
  h=st["inner_history"];vals=[z["validation_CE"] for z in h];best=int(np.argmin(vals))+1
  check(st["best_epoch"]==best==len(st["final_history"])==st["epoch"],"selected epoch")
  check(15<=len(h)<=40,"epoch range")
  if len(h)<40:check(len(h)-best>=12,"patience stop")
  for phase,records in [("inner",ss["inner_train"]),("final",ss["outer_train"])]:
   sc=st["scalers"][phase];ids=np.flatnonzero(meta.record.isin(records))
   check(np.array_equal(sc["indices"],ids) and set(sc["records"])==set(records),"scaler sources")
   mu,var,scale=expected_scale(records);close(sc["mean"],mu,"channel mean",atol=1e-9);close(sc["variance"],var,"channel variance",atol=2e-8,rtol=1e-7);close(sc["scale"],scale,"channel scale",atol=1e-8)
   close(sc["weights"],source_weights(meta,ids),"scaler weights")
   scale_max=max(scale_max,float(np.max(abs(sc["mean"]-mu))))
  g=pd.read_csv(dest/"predictions.csv");ids=g.original_row.to_numpy()
  check(set(g.record)==set(ss["test"]),"prediction test membership")
  sc=st["scalers"]["final"];cols=ARMS[ident["arm"]]
  model=ReloadNet(len(cols),ident["architecture"]).cuda();model.load_state_dict(st["model"]);model.eval()
  check(sum(q.numel() for q in model.parameters())==st["parameters"],"saved parameter count")
  z=((x[ids][:,cols,:].astype(np.float64)-sc["mean"][cols][None,:,None])/sc["scale"][cols][None,:,None]).astype(np.float32)
  outs=[]
  with torch.no_grad():
   for off in range(0,len(z),64):outs.append(model(torch.tensor(z[off:off+64],device="cuda")).cpu().numpy())
  out=np.concatenate(outs);saved=g[["logit_"+a for a in ACT]].to_numpy();diff=float(abs(out-saved).max());reload_max=max(reload_max,diff)
  close(out,saved,"checkpoint full inference",atol=3e-5,rtol=1e-5);check(np.array_equal(out.argmax(1),g.pred_index),"reloaded predictions identical")
  for z in h:history.append(dict(job=dest.name,scope=ident["scope"],configuration=ident["architecture"]+"_"+ident["arm"],seed=ident["seed"],fold=ident["fold"],**z))
  jobrows.append(dict(job=dest.name,scope=ident["scope"],configuration=ident["architecture"]+"_"+ident["arm"],seed=ident["seed"],fold=ident["fold"],inner_epochs=len(h),selected_epoch=best,final_epochs=len(st["final_history"]),parameters=st["parameters"],seconds=dd["elapsed_seconds"],peak_cuda_allocated=st["peak_cuda_allocated"],device=st["device"]))
  del model
  if (i+1)%60==0:print("CHECKPOINT_QA",i+1,"/438",flush=True)
 pd.DataFrame(jobrows).to_csv(O/"fit_audit.csv",index=False);pd.DataFrame(history).to_csv(O/"inner_learning_curves.csv",index=False)
 auxrows=[]
 for config,g in auxpred.groupby("configuration"):
  check(set(g.record)==set(sp["aux"]["test"]),"aux records")
  cm,f,rec=metrics(g);auxrows.append(dict(configuration=config,macro_F1=float(f.mean()),balanced_accuracy=float(rec.mean()),windows=len(g),note="descriptive auxiliary split; not main CV"))
 pd.DataFrame(auxrows).to_csv(O/"aux_metrics.csv",index=False)
 point=sp["aux"]["point_indices"]
 for rid in sp["aux"]["test"]:
  z=meta.iloc[point];z=z[z.record==rid].sort_values("start_row")
  check((z.start_row.to_numpy()[1:]>=z.end_row_exclusive.to_numpy()[:-1]).all(),"aux nonoverlap")
 # Cross-check pooled metrics independently through sklearn sample_weight implementation.
 from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
 for (config,seed),g in p.groupby(["configuration","seed"]):
  row=sums[(sums.configuration==config)&(sums.seed==seed)].iloc[0]
  close(row.macro_F1,f1_score(g.true_index,g.pred_index,labels=range(8),average="macro",sample_weight=g.eval_weight),"sklearn F1")
  close(row.balanced_accuracy,balanced_accuracy_score(g.true_index,g.pred_index,sample_weight=g.eval_weight),"sklearn BA")
  close(row.weighted_window_accuracy,accuracy_score(g.true_index,g.pred_index,sample_weight=g.eval_weight),"sklearn accuracy")
 receipt=dict(status="PASS",checks=nchecks,raw_files=24,raw_windows_reconstructed=1941,main_jobs=432,aux_jobs=6,main_fit_processes=864,aux_fit_processes=12,OOF_rows=len(p),all_checkpoints_reloaded=438,checkpoint_predictions_identical=True,max_logit_reload_error=reload_max,max_scaler_mean_difference=scale_max,max_absolute_comparison_error=maxdiff,all_seeds_and_folds_present=True,raw_and_v2_hashes_unchanged=True,seconds=time.perf_counter()-started,verifier_sha256=sha(__file__),formal_training_script_unchanged=True,verification_note="Initial FP64 interpretation of shortest FP32 CSV decimal logits differed from saved softmax by3.045e-7. Restore original FP32 values before independent FP64 softmax; original predictions/models unchanged. Direct FP32 recomputation max error2.98e-8.",independent_scope="independent source interpolation, per-record weighted scaler, saved checkpoint inference, pooled weighted confusion metrics plus sklearn cross-check; no new fitting")
 jw(O/"independent_verification.json",receipt)
 print("QA_PASS",json.dumps(receipt),flush=True);print(sums.to_string(index=False),flush=True);print("PAIRED",pd.DataFrame(paired).to_string(index=False),flush=True)
if __name__=="__main__":main()

