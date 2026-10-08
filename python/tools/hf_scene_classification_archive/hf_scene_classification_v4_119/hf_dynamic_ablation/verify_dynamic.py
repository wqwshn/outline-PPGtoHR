"""Independent dynamicHF audit using centered raw-cache residuals. No fitting."""
from pathlib import Path
import importlib.util,json,time,hashlib
import numpy as np,pandas as pd,torch
O=Path(__file__).resolve().parent;MAIN=O.parent
sp=importlib.util.spec_from_file_location("independent_mainQA",MAIN/"verify_119.py");q=importlib.util.module_from_spec(sp);sp.loader.exec_module(q)
A=q.A;S=q.S;ck=q.ck;sha=q.sha
def main():
 start=time.perf_counter();fr=json.loads((O/"freeze_receipt.json").read_text());ck(json.loads((O/"run_status.json").read_text())["status"]=="training_complete","complete96")
 for n,h in fr["artifacts"].items():ck(sha(O/n)==h,"frozen"+n)
 for n,h in fr["main_artifacts"].items():ck(sha(MAIN/n)==h,"maininput"+n)
 ck(json.loads((MAIN/"independent_verification.json").read_text())["status"]=="PASS","mainindependentPASS")
 protected=json.loads((O/"main_completed_artifact_hashes.json").read_text())
 for n,h in protected.items():ck(sha(MAIN/n)==h,"mainoutput"+n)
 for r in fr["raw"]:ck(sha(r["path"])==r["sha256"],"raw")
 meta=pd.read_csv(MAIN/"windows.csv");rec=pd.read_csv(MAIN/"records_LOCAL.csv");x=np.load(MAIN/"windows_float32.npy",mmap_mode="r")
 rs={}
 for rid,g in meta.groupby("record"):
  z=x[g.index,6:,:].astype(float);z-=z.mean(2,keepdims=True);rs[rid]=np.mean(z*z,axis=(0,2))
 oo=pd.read_csv(O/"oof_predictions.csv");ck(len(oo)==9520*6,"rows");ck(not oo.duplicated(["configuration","seed","original_row"]).any(),"unique")
 for (arm,n),g in oo.groupby(["configuration","seed"]):
  ck(set(g.original_row)==set(range(9520)),"coverage");ck(np.array_equal(g.activity,meta.iloc[g.original_row].activity),"labels")
 sel=json.loads((O/"selected_epochs.json").read_text());maxerr=0.;hist=[];rmsrows=[];init={}
 donefiles=list((O/"jobs").glob("*/done.json"));ck(len(donefiles)==96,"96jobs")
 for dp in donefiles:
  d=json.loads(dp.read_text());dest=dp.parent;ident=d["identity"];st=torch.load(dest/"checkpoint.pt",map_location="cpu",weights_only=False)
  ck(st["identity"]==ident and ident["freeze_sha256"]==sha(O/"freeze_receipt.json"),"identity")
  for n,h in d["files"].items():ck(sha(dest/n)==h,"jobfile")
  ids=np.flatnonzero(meta.subject.isin(ident["train"]));sc=st["scaler"];ck(np.array_equal(ids,sc["indices"]),"trainindices");ck(ident["held"] not in ident["train"],"heldisolation")
  ck(np.allclose(sc["weights"],q.wts(meta.iloc[ids]),rtol=0,atol=1e-16),"weights")
  primarykey=f"{ident['phase']}__{ident['held']}__MIMU__s{ident['seed']}"+(f"__val{ident['validation']}" if ident["phase"]=="inner" else "")
  primary=torch.load(MAIN/"jobs"/primarykey/"checkpoint.pt",map_location="cpu",weights_only=False)
  ck(np.array_equal(sc["mean"][:6],primary["scaler"]["mean"][:6]) and np.array_equal(sc["scale"][:6],primary["scaler"]["scale"][:6]),"MIMUscaleridentical")
  ck(st["initial_model_hash"]==primary["initial_model_hash"],"sameinit")
  ck(primary["identity"]["train"]==ident["train"],"sametrainpeople")
  mu=[]
  for sub in ident["train"]:
   aa=[]
   for activity in A:
    records=rec[(rec.subject==sub)&(rec.activity==activity)].record;aa.append(np.mean([rs[r] for r in records],axis=0))
   mu.append(np.mean(aa,axis=0))
  rms=np.sqrt(np.mean(mu,axis=0));expected=np.where(rms<=1e-12,1.,rms)
  ck(np.all(sc["mean"][6:]==0),"centerzero");ck(np.allclose(rms,sc["HF_centered_RMS_mV"],rtol=1e-10,atol=1e-10),"pooledRMS");ck(np.allclose(expected,sc["scale"][6:],rtol=1e-10,atol=1e-10),"RMSscale")
  ck(np.array_equal(rms<=1e-12,sc["HF_zero_scale_fallback"]),"fallback")
  for j in range(2):rmsrows.append(dict(job=dest.name,phase=ident["phase"],held=ident["held"],arm=ident["arm"],HF=j+1,pooled_centered_RMS_mV=rms[j],fallback=bool(rms[j]<=1e-12)))
  if ident["phase"]=="inner":
   ck(len(ident["train"])==4 and ident["validation"] not in ident["train"] and ident["validation"]!=ident["held"],"innerisolation")
   best=int(np.argmin([h["validation_CE"] for h in st["history"]]))+1;ck(best==d["best_epoch"],"bestepoch");ck(15<=d["epochs"]<=40,"innerbounds")
  else:
   selection=next(z for z in sel if z["subject"]==ident["held"] and z["arm"]==ident["arm"]);ck(d["epochs"]==int(np.median(selection["inner_best_epochs"]))==selection["selected_epoch"],"median5")
   actual=[json.loads((O/"jobs"/f"inner__{ident['held']}__{ident['arm']}__s{S[0]}__val{s}"/"done.json").read_text())["best_epoch"] for s in sorted(meta.subject.unique()) if s!=ident["held"]];ck(actual==selection["inner_best_epochs"],"owninnerselection")
   g=pd.read_csv(dest/"predictions.csv");ti=g.original_row.to_numpy(int);ck(np.all(meta.iloc[ti].subject==ident["held"]),"testheld")
   ck(np.allclose(g.eval_weight,q.wts(g),rtol=0,atol=1e-15),"evalweights")
   model=q.Net().cuda();model.load_state_dict(st["model"]);model.eval();out=[]
   with torch.no_grad():
    for k in range(0,len(ti),64):
     z=x[ti[k:k+64]].astype(float);z[:,6:]-=z[:,6:].mean(2,keepdims=True);z=(z-sc["mean"][None,:,None])/sc["scale"][None,:,None]
     if ident["arm"]=="HF_dynamic":z[:,:6]=0
     out.append(model(torch.from_numpy(z.astype(np.float32)).cuda()).cpu().numpy())
   logits=np.concatenate(out);saved=g[[f"logit_{a}" for a in A]].to_numpy(np.float32);maxerr=max(maxerr,float(abs(logits-saved).max()));ck(np.allclose(logits,saved,rtol=1e-6,atol=3e-5),"reload");ck(np.array_equal(logits.argmax(1),g.pred_index),"reloadlabels")
   ex=np.exp(saved.astype(float)-saved.max(1,keepdims=True));ex/=ex.sum(1,keepdims=True);ck(np.max(abs(ex-g[[f"prob_{a}" for a in A]].to_numpy()))<1e-6,"independentF64softmax");ck(np.max(abs(torch.softmax(torch.tensor(saved),1).numpy()-g[[f"prob_{a}" for a in A]].to_numpy()))<5e-8,"originalFP32softmax")
   match=oo[(oo.configuration==ident["arm"])&(oo.seed==ident["seed"])&(oo.subject==ident["held"])].sort_values("original_row");ck(np.array_equal(match.pred_index,g.sort_values("original_row").pred_index),"aggregate")
   del model
  for h in st["history"]:hist.append(dict(job=dest.name,arm=ident["arm"],phase=ident["phase"],held=ident["held"],**h))
  del primary,st
 base=pd.read_csv(MAIN/"oof_predictions.csv");full=pd.concat([base[base.configuration.isin(["MIMU","FUS"])],oo],ignore_index=True);full.to_csv(O/"comparison_predictions.csv",index=False)
 sm=[];rm=[];ac=[];ties=[]
 for (arm,n,sub),g in full.groupby(["configuration","seed","subject"]):
  met,cm,f,rc=q.metrics(g);sm.append(dict(configuration=arm,seed=n,subject=sub,**met))
  for j,a in enumerate(A):ac.append(dict(configuration=arm,seed=n,subject=sub,activity=a,F1=f[j],recall=rc[j]))
  for rid,zz in g.groupby("record"):
   counts=np.bincount(zz.pred_index,minlength=8);t=np.flatnonzero(counts==counts.max());vote=int(t[0]);truth=int(zz.true_index.iloc[0]);rm.append(dict(configuration=arm,seed=n,subject=sub,record=rid,activity=A[truth],windows=len(zz),window_accuracy=float(np.mean(zz.pred_index==truth)),vote=A[vote],correct=int(vote==truth),tie=len(t)>1,tie_as_wrong_correct=int(vote==truth and len(t)==1)))
   if len(t)>1:ties.append(dict(configuration=arm,seed=int(n),subject=sub,record=rid,counts=counts.tolist()))
 sm=pd.DataFrame(sm);sm.to_csv(O/"subject_metrics.csv",index=False);pd.DataFrame(rm).to_csv(O/"record_metrics.csv",index=False);pd.DataFrame(ac).to_csv(O/"activity_metrics.csv",index=False)
 sd=sm.groupby(["configuration","seed"])[["macro_F1","BA"]].mean().reset_index();sd.to_csv(O/"summary_by_seed.csv",index=False)
 paired=[]
 for (n,sub),g in sm.groupby(["seed","subject"]):
  g=g.set_index("configuration")
  for baseline in ["MIMU","FUS"]:paired.append(dict(seed=n,subject=sub,baseline=baseline,delta_macro_F1=g.loc["FUS_dynamic","macro_F1"]-g.loc[baseline,"macro_F1"],delta_BA=g.loc["FUS_dynamic","BA"]-g.loc[baseline,"BA"]))
 pd.DataFrame(paired).to_csv(O/"paired_deltas.csv",index=False);pd.DataFrame(rmsrows).to_csv(O/"HF_training_centered_RMS.csv",index=False);pd.DataFrame(hist).to_csv(O/"learning_curves.csv",index=False);(O/"vote_ties.json").write_text(json.dumps(ties,indent=2),encoding="utf-8")
 qa=dict(status="PASS",checks=q.checks,checkpoints=96,refit_models_reloaded=36,max_logit_reload_error=maxerr,prediction_labels_identical=True,main_MIMU_scalers_weights_initialization_same=True,own_dynamic_inner_epochs_verified=True,main_models_predictions_unchanged=True,raw119_unchanged=True,seconds=time.perf_counter()-start,verifier_sha256=sha(__file__))
 (O/"independent_verification.json").write_text(json.dumps(qa,indent=2),encoding="utf-8");print(json.dumps(qa),flush=True);print(sd.to_string(index=False),flush=True)
if __name__=="__main__":main()

