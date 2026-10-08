"""Independent post-run verification. No model fitting and no runner import."""
from pathlib import Path
import hashlib,json,ast
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
ROOT=Path(r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR")
O=ROOT/"data/experiments/hf_scene_classification_v1/p1_20261005_01"
P=O.parent/"p0_20261005_02"
checks=[];errors=[]
def ok(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(name,a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    err=float(np.max(np.abs(a-b))) if a.size else 0.
    errors.append(err);ok(name,np.allclose(a,b,rtol=1e-10,atol=1e-12))
cfg=json.loads((O/"frozen_config.json").read_text())
receipt=json.loads((O/"freeze_receipt.json").read_text())
manifest=json.loads((O/"analysis_manifest.json").read_text())
ok("all numeric manifest hashes unchanged",all(sha(O/r["path"])==r["sha256"] for r in manifest))
ok("frozen config hash",sha(O/"frozen_config.json")==receipt["config_sha256"])
script=ROOT/"python/tools/run_hf_scene_classification_v1.py"
ok("executed source hash",sha(script)==receipt["script_sha256"])
ok("float max_features=1.0",type(cfg["model"]["max_features"]) is float and cfg["model"]["max_features"]==1.0)
ok("seven unique features per eight channels",len(cfg["features"])==len(set(cfg["features"]))==7 and len(cfg["channels"])==8)
tree=ast.parse(script.read_text())
fits=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="fit"]
ok("pilot and LOSO actual fit receive sample_weight",len(fits)==2 and all(any(k.arg=="sample_weight" for k in n.keywords) for n in fits))
ok("LOSO fit uses original X with train mask",any(ast.unparse(n)=="m.fit(X[train][:, cols[arm]], y[train], sample_weight=train_weights)" for n in fits))
m=pd.read_csv(O/"window_metadata.csv");p=pd.read_csv(O/"oof_predictions.csv")
x=np.load(O/"features.npz")["X"];s=pd.read_csv(O/"subject_metrics.csv");a=pd.read_csv(O/"activity_metrics.csv")
rec=pd.read_csv(O/"record_metrics.csv");strata=pd.read_csv(O/"missingness_strata.csv");cms=pd.read_csv(O/"confusion_matrices.csv")
act=cfg["activities"];arms=["MIMU","HF","MIMU_HF"]
ok("finite feature matrix 9520x56",x.shape==(9520,56) and np.isfinite(x).all())
ok("all 28560 OOF rows unique by arm and window",len(p)==28560 and not p.duplicated(["arm","original_row"]).any())
ok("same window membership as P0",m[list(pd.read_csv(P/"motion_windows.csv").columns)].fillna("").equals(pd.read_csv(P/"motion_windows.csv").fillna("")))
probs=p[["p_"+v for v in act]].to_numpy()
ok("finite probabilities in [0,1]",np.isfinite(probs).all() and np.min(probs)>=0 and np.max(probs)<=1)
close("probabilities sum one",probs.sum(1),np.ones(len(p)))
ok("saved prediction is probability argmax",np.array_equal(np.array(act)[probs.argmax(1)],p.prediction.to_numpy()))
recomputed=[];strata_out=[]
def independent_cm(g,w):
    cm=np.zeros((8,8))
    for yt,yp,wi in zip(g.activity,g.prediction,w):cm[act.index(yt),act.index(yp)]+=wi
    den=cm.sum(1)+cm.sum(0)
    f=np.divide(2*cm.diagonal(),den,out=np.zeros(8),where=den!=0)
    r=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(8),where=cm.sum(1)!=0)
    return cm,f,r
for arm in arms:
    q=p[p.arm==arm]
    ok(arm+" covers every input row",set(q.original_row)==set(range(len(m))))
    for sub,g in q.groupby("subject"):
        original=m.iloc[g.original_row]
        ok(arm+sub+" metadata alignment and held-out identity",np.array_equal(g.record,original.record) and np.array_equal(g.activity,original.activity) and (original.subject==sub).all())
        counts=g.groupby("activity").record.nunique();nw=g.groupby("record").size()
        w=np.array([1/(8*counts[r.activity]*nw[r.record]) for r in g.itertuples()])
        close(arm+sub+" independent test weights",w,g.test_weight)
        cm,f,r=independent_cm(g,w)
        saved=s[(s.arm==arm)&(s.subject==sub)].iloc[0]
        close(arm+sub+" macro F1 and BA",[f.mean(),r.mean()],[saved.macro_F1,saved.balanced_accuracy])
        ac=a[(a.arm==arm)&(a.subject==sub)].set_index("activity").loc[act]
        close(arm+sub+" every activity metric",np.column_stack([f,r,cm.sum(1)]),ac[["F1","recall","support_weight"]])
        savedcm=cms[(cms.arm==arm)&(cms.subject==sub)].pivot(index="true_activity",columns="predicted_activity",values="weight").loc[act,act]
        close(arm+sub+" every confusion cell",cm,savedcm)
        for rid,rg in g.groupby("record"):
            rr=rec[(rec.arm==arm)&(rec.record==rid)].iloc[0]
            close(arm+rid+" record accuracy",np.mean(rg.activity==rg.prediction),rr.accuracy)
        for level,gg in g.groupby("missing_bin"):
            sr=strata[(strata.arm==arm)&(strata.subject==sub)&(strata.missing_bin==level)].iloc[0]
            ww=w[g.missing_bin.to_numpy()==level]
            ok(arm+sub+level+" stratum membership",len(gg)==sr.windows and gg.activity.nunique()==sr.activity_count)
            accuracy=np.average(gg.activity==gg.prediction,weights=ww)
            close(arm+sub+level+" stratum accuracy",accuracy,sr.weighted_accuracy)
            _,ff,rr=independent_cm(gg,ww)
            if gg.activity.nunique()==8:close(arm+sub+level+" covered metrics",[ff.mean(),rr.mean()],[sr.macro_F1_eight_classes,sr.balanced_accuracy_eight_classes])
            else:ok(arm+sub+level+" incomplete class metrics suppressed",pd.isna(sr.macro_F1_eight_classes) and pd.isna(sr.balanced_accuracy_eight_classes))
        recomputed.append(dict(subject=sub,arm=arm,macro_F1=float(f.mean()),balanced_accuracy=float(r.mean())))
weights=pd.read_csv(O/"training_weight_audit.csv")
for sub,g in weights.groupby("fold"):
    train=m[m.subject!=sub]
    ok(sub+" train/test subject and record isolation",sub not in set(g.subject) and set(g.record)==set(train.record) and set(g.record).isdisjoint(set(m[m.subject==sub].record)))
    cnt=train.groupby(["subject","activity"]).record.nunique()
    expected=[1/(5*8*cnt[r.subject,r.activity]) for r in g.itertuples()]
    close(sub+" independent actual-fit record totals",expected,g.total_weight)
    close(sub+" subject totals",g.groupby("subject").total_weight.sum(),np.full(5,.2))
    close(sub+" subject/activity totals",g.groupby(["subject","activity"]).total_weight.sum(),np.full(40,.025))
mc=pd.read_csv(O/"model_costs.csv")
ok("18 models obey feature/tree/depth/leaf limits",len(mc)==18 and (mc.tree_count==200).all() and (mc.max_depth_actual<=4).all() and (mc.max_leaves_actual<=16).all() and (mc.features==mc.arm.map(dict(MIMU=42,HF=14,MIMU_HF=56))).all())
bindings=pd.read_csv(P/"record_bindings_LOCAL.csv")
ok("all 119 raw input hashes unchanged",all(sha(r.input_path)==r.input_sha256 for r in bindings.itertuples()))
# Independent feature implementation: pandas interpolation and explicit scalar frequency formulas.
spot=[]
for rid,g in m.groupby("record"):
    chosen=[g.index[0],g.missing_sample_rows.idxmax()]
    raw=pd.read_csv(bindings.set_index("record").loc[rid,"input_path"])[cfg["channels"]].apply(pd.to_numeric,errors="coerce")
    for idx in sorted(set(chosen)):
        row=m.loc[idx];v=raw.iloc[int(row.start_row):int(row.end_row_exclusive)].replace([np.inf,-np.inf],np.nan).interpolate(method="linear",limit_direction="both").to_numpy()
        vals=[]
        for ch in range(8):
            z=v[:,ch];f=np.arange(401)*100/800
            power=np.abs(np.fft.rfft((z-z.mean())*(.5-.5*np.cos(2*np.pi*np.arange(800)/799))))**2
            band=(f>=.5)&(f<=20);lo=(f>=.5)&(f<=5);pb=power[band];energy=pb.sum()
            if energy>0:
                dist=pb/energy;positive=dist>0
                freq=f[band][np.argmax(pb)];entropy=-sum(dist[positive]*np.log(dist[positive]))/np.log(len(pb));ratio=sum(power[lo])/energy
            else:freq=entropy=ratio=0.
            vals.extend([sum(z)/800,np.sqrt(sum((z-z.mean())**2)/800),max(z)-min(z),sum(abs(np.diff(z)))/799,freq,entropy,ratio])
        close("independent feature spot "+str(idx),vals,x[idx]);spot.append(int(idx))
sm=pd.read_csv(O/"exploration_samples.csv");emb=np.load(O/"embeddings.npz")
ok("exploration identical 952 samples and eight per record",len(sm)==952 and sm.groupby("record").size().eq(8).all())
ok("exploration windows nonoverlapping",all(np.diff(g.sort_values("center_s").center_s).min()>=8 for _,g in sm.groupby("record")))
ok("three PCA and twelve fixed tSNE arrays finite",len(emb.files)==15 and all(emb[k].shape==(952,2) and np.isfinite(emb[k]).all() for k in emb.files))
cluster=pd.read_csv(O/"clustering_metrics.csv")
for arm in arms:
    cl=pd.read_csv(O/f"clusters_{arm}.csv")
    ok(arm+" cluster row identities",np.array_equal(sm.original_row,cl.original_row))
    rr=cluster[cluster.arm==arm].iloc[0]
    close(arm+" independently recomputed ARI AMI",[adjusted_rand_score(sm.activity,cl.cluster),adjusted_mutual_info_score(sm.activity,cl.cluster),adjusted_rand_score(sm.subject,cl.cluster),adjusted_mutual_info_score(sm.subject,cl.cluster)],rr[["activity_ARI","activity_AMI","subject_ARI","subject_AMI"]])
summary=json.loads((O/"summary.json").read_text());rs=pd.DataFrame(recomputed)
for arm in arms:
    close(arm+" summary means",rs[rs.arm==arm][["macro_F1","balanced_accuracy"]].mean(),list(summary["arm_means"][arm].values()))
pivot=rs.pivot(index="subject",columns="arm",values="macro_F1")
delta=pivot.MIMU_HF-pivot.MIMU
close("all paired deltas",delta,pd.read_csv(O/"paired_subject_deltas.csv").delta)
close("mean paired delta",delta.mean(),summary["mean_paired_delta"])
activity=a.groupby(["arm","activity"])[["F1","recall"]].mean()
change=activity.loc["MIMU_HF"]-activity.loc["MIMU"]
change.to_csv(O/"verified_activity_changes.csv")
stratum_summary=strata.groupby(["arm","missing_bin"]).agg(windows=("windows","sum"),subjects_with_windows=("windows",lambda v:int((v>0).sum())),subjects_full_coverage=("full_eight_class_coverage","sum"),mean_subject_conditional_accuracy=("weighted_accuracy","mean")).reset_index()
stratum_summary.to_csv(O/"verified_strata_summary.csv",index=False)
result=dict(status="PASS",checks=len(checks),check_names=checks,max_absolute_numeric_error=max(errors),feature_spot_windows=len(spot),feature_spot_indices=spot,source_sha256=sha(__file__),raw_inputs_unchanged=119,numeric_manifest_files=len(manifest),models_refit=0,scope_limits=["Models were not persisted: actual fit is evidenced by hashed executed source and run receipts, not independently inspected serialized estimators.","Spot features independently recalculated; remaining feature rows checked finite and source-hash protected.","No causal interpretation or statistical significance claimed."])
(O/"independent_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k not in ("check_names","feature_spot_indices")}))
print("ACTIVITY DELTAS\n"+change.to_string())
print("STRATA\n"+stratum_summary.to_string(index=False))
print("PAIRS\n"+pd.read_csv(O/"paired_subject_deltas.csv").to_string(index=False))

