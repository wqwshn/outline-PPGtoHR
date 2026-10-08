"""LYX24 approved fixed matrix. Freeze first, then execute once. No tuning."""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[k]="1"
from pathlib import Path
import sys,json,hashlib,time,platform,subprocess,argparse
from datetime import datetime,timezone
import numpy as np,pandas as pd,sklearn,scipy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix,adjusted_rand_score,adjusted_mutual_info_score
from threadpoolctl import threadpool_limits
R=Path(r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR")
O=Path(__file__).resolve().parent;P=O.parent/"preparation_20261006_01"
ACT=["HW","RS","HG","TYP","RUN","JJ","BUR","PCH"]
CH=["AccX(g)","AccY(g)","AccZ(g)","GyroX(dps)","GyroY(dps)","GyroZ(dps)","Ut1(mV)","Ut2(mV)"]
F7=["mean","std_ddof0","peak_to_peak","mean_abs_first_difference","dominant_frequency_0.5_20Hz","normalized_spectral_entropy_0.5_20Hz","power_ratio_0.5_5Hz_to_0.5_20Hz"]
B7=["linear_slope_per_second","first_half_minus_second_half","Q1_minus_Q2","Q3_minus_Q4","slow_1s_std","fast_residual_std","fast_residual_dominant_frequency_0.5_20Hz"]
MATRICES=[dict(id=f"{clf}_{arm}",classifier=clf,arm=arm,features=d) for clf in ["ET","SVC"] for arm,d in [("MIMU",42),("HF_A",14),("FUS_A",56),("HF_B",28),("FUS_B",70)]]
MATRICES += [dict(id="SVC_HF_SHAPE",classifier="SVC",arm="HF_SHAPE",features=38),dict(id="SVC_FUS_SHAPE",classifier="SVC",arm="FUS_SHAPE",features=80)]
CFG=dict(experiment="LYX24_round2_fixed12_LORO",status="frozen_before_features_and_results",activities=ACT,channels=CH,records=24,windows=1941,subject="subject-1",
    input_scope="All frozen MIMU is_motion windows; no reliable/HR-error/used_adaptive conditioning; 100Hz, 800 rows, 1s stride.",
    imputation="Per 800-point window/channel np.interp finite samples; internal linear, nearest endpoints; all missing raises; no dropped rows/windows.",
    original_features=F7,hf_B_added_per_channel=B7,slope="OLS slope with t=arange(800)/100 seconds, centered t; raw mV/s.",
    contrasts="Q1..Q4 are consecutive 200-point means; first_half_minus_second_half=(Q1+Q2-Q3-Q4)/2; signed Q1-Q2 and Q3-Q4.",
    slow_filter="100-point uniform moving average, exactly 1 second; np.pad(x,(49,50),mode='reflect'), convolution valid. Even-width half-sample center convention frozen. All support inside current window, no neighboring windows.",
    spectral="Demean, np.hanning(800), squared rFFT; inclusive 0.5..20Hz band and 0.5..5Hz low band; entropy normalized by log(bin count); zero power=>0; first argmax tie.",
    shape="Per HF channel 16 consecutive 50-point means minus whole-window mean, then raw whole-window mean, std_ddof0, fast_residual_std: 19/channel, 38 HF, 80 fusion. No IQR normalization.",
    matrix=MATRICES,primary="ET_FUS_B minus ET_MIMU in pooled record-balanced weighted eight-class macro-F1",
    extra_trees=dict(n_estimators=200,max_depth=4,max_leaf_nodes=16,min_samples_leaf=1,max_features=1.0,bootstrap=False,criterion="gini",random_state=20261005,n_jobs=1,class_weight=None),
    svc=dict(C=1.0,kernel="rbf",gamma="1/n_features (numeric float)",class_weight=None,probability=False,decision_function_shape="ovr",tol=0.001,shrinking=True,cache_size=200,max_iter=-1),
    folds="24 leave-one-complete-record-out; 23 training; all train classes retained; no cross-record shared wearing-batch claim.",
    train_weights="1/(8 * number_training_records_in_activity * windows_in_record). ET receives sum-one weights. SVC weights multiplied by n_train/sum(weight), hence mean exactly one.",
    scaling="Only SVC: StandardScaler fitted on training feature rows using same hierarchical sample weights; saved mean/var/scale. ET no scaler. No full-data supervised transform.",
    evaluation_weights="OOF window weight=1/(8*3*windows_in_record); pool all 24 records before 8-class macro-F1/BA. Never average single-class fold macro-F1.",
    record_vote="Most frequent predicted window label per complete record; ties lowest fixed ACT index. Each record one equal vote.",
    quality_strata=["0%","(0,10%]","(10,25%]",">25%"],quality_interpretation="Conditional descriptive subsets only; suppress eight-class metrics unless all eight true classes present.",
    exploration=dict(unit="24 records, within-record median feature per column",spaces=["A_MIMU","A_HF","A_FUS","B_MIMU","B_HF","B_FUS"],scaler="Unweighted StandardScaler on 24 record medians only; single arm /sqrt(d); fusion blocks /sqrt(42) and /sqrt(HFdim). Never used in supervised folds.",
        kmeans=dict(n_clusters=8,n_init=20,max_iter=300,algorithm="lloyd",random_state=20261005),pca=dict(n_components=2,svd_solver="full"),
        tsne=dict(n_components=2,perplexities=[5,10],random_state=20261005,init="pca",learning_rate="auto",max_iter=1000,method="barnes_hut",angle=0.5,n_jobs=1),
        labels="Activity ARI/AMI only; each record has unique rewear so no wear ARI. Color activity and verified reference-date categories; RS-02 conflicting and HG-03 unknown separate.",
        caveat="24 points/K8 and PCA/tSNE are descriptive; cannot establish generalization or pick representations."),
    limitations=["Single-person posthoc heart-rate development panel, not new validation.", "21 inputs and 1726 windows overlap round1. Other three same subject.",
        "Date strongly associated with class; high HF classification cannot exclude thermal baseline/date confounding.", "Record rewear confirmed by author, no three shared wear batches, no controlled pressure.",
        "HF voltage-domain features do not constitute force/flow calibration or HR improvement evidence."],
    search_budget=0,stop_after_matrix=True,old_threefold_proposal="NOT USED")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jw(name,v):
    with (O/name).open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def save(name,rows):pd.DataFrame(rows).to_csv(O/name,index=False)
def log(s):
    print(s,flush=True)
    with (O/"progress.log").open("a",encoding="utf-8") as f:f.write(s+"\n")
def freeze():
    assert type(CFG["extra_trees"]["max_features"]) is float
    meta=pd.read_csv(P/"frozen_motion_windows.csv");rec=pd.read_csv(P/"record_manifest_LOCAL.csv")
    assert len(meta)==1941 and len(rec)==24 and rec.groupby("activity").size().eq(3).all()
    for r in rec.itertuples():assert sha(r.input_path)==r.input_sha256
    jw("frozen_config.json",CFG)
    folds=[dict(fold=held,record=r.record,activity=r.activity,role="test" if r.record==held else "train") for held in sorted(rec.record) for r in rec.itertuples()]
    save("frozen_folds.csv",folds)
    meta["missing_fraction"]=meta.missing_rows/800
    meta["missing_bin"]=pd.cut(meta.missing_rows,[-1,0,80,200,800],labels=CFG["quality_strata"]).astype(str)
    meta.to_csv(O/"window_metadata.csv",index=False)
    rec.to_csv(O/"record_manifest_LOCAL.csv",index=False)
    git={key:subprocess.check_output(["git",*cmd],cwd=R,text=True,encoding="utf-8").strip() for key,cmd in dict(HEAD=["rev-parse","HEAD"],branch=["branch","--show-current"],status=["status","--short"],staged=["diff","--cached","--name-status"]).items()}
    assert not git["staged"]
    jw("freeze_receipt.json",dict(utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha(O/"frozen_config.json"),script_sha256=sha(__file__),folds_sha256=sha(O/"frozen_folds.csv"),metadata_sha256=sha(O/"window_metadata.csv"),preparation_manifest_sha256=sha(P/"preparation_manifest_final.json"),git=git,python=sys.version,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,features_extracted=False,models_fitted=0,isolation="new execution directory; original main worktree unchanged; no new git worktree",command=f"{sys.executable} -B {__file__} --run"))
    log("FROZEN: 24 records, 1941 windows, 24 LORO folds, 12 configurations; no models yet.")
def impute(a):
    out=np.array(a,float,copy=True)
    for j in range(out.shape[1]):
        valid=np.isfinite(out[:,j])
        if not valid.any():raise ValueError("All-missing window/channel")
        out[:,j]=np.interp(np.arange(800),np.flatnonzero(valid),out[valid,j])
    return out
FREQ=np.fft.rfftfreq(800,.01);BAND=(FREQ>=.5)&(FREQ<=20);LOW=(FREQ>=.5)&(FREQ<=5)
def spectral(x):
    p=abs(np.fft.rfft((x-x.mean())*np.hanning(800)))**2;pb=p[BAND];total=pb.sum()
    if total==0:return [0.,0.,0.]
    q=pb/total;pos=q>0
    return [float(FREQ[BAND][np.argmax(pb)]),float(-(q[pos]*np.log(q[pos])).sum()/np.log(len(q))),float(p[LOW].sum()/total)]
def original(x):return [float(x.mean()),float(x.std()),float(np.ptp(x)),float(np.abs(np.diff(x)).mean()),*spectral(x)]
def extra(x):
    slow=np.convolve(np.pad(x,(49,50),mode="reflect"),np.ones(100)/100,mode="valid");fast=x-slow
    t=np.arange(800)/100;t=t-t.mean();q=x.reshape(4,200).mean(1)
    return [float(np.dot(t,x-x.mean())/np.dot(t,t)),float((q[0]+q[1]-q[2]-q[3])/2),float(q[0]-q[1]),float(q[2]-q[3]),float(slow.std()),float(fast.std()),spectral(fast)[0]]
def extract(a):
    a=impute(a);base=np.array([original(a[:,j]) for j in range(8)]).reshape(-1)
    hb=[];shape=[]
    for j in [6,7]:
        z=a[:,j];ext=extra(z)
        hb.extend(original(z)+ext)
        shape.extend((z.reshape(16,50).mean(1)-z.mean()).tolist()+[float(z.mean()),float(z.std()),ext[5]])
    return base,np.array(hb),np.array(shape)
def tests():
    z=np.arange(800)/100;ext=extra(z);assert np.isclose(ext[0],1) and np.allclose(ext[1:4],[-4,-2,-2])
    a=np.tile(z[:,None],(1,8));a[200:210]=np.nan;assert np.allclose(impute(a)[200:210,0],z[200:210])
    a[:5]=np.nan;assert np.all(impute(a)[:5,0]==z[5])
    try:impute(np.full((800,8),np.nan));raise AssertionError("all missing not rejected")
    except ValueError:pass
    a,b,c=extract(np.ones((800,8)));assert (a.shape,b.shape,c.shape)==((56,),(28,),(38,))
    assert np.isfinite(np.r_[a,b,c]).all() and np.allclose(c[:16],0) and b[7]==0
def weights(meta,mean_one=False):
    nr=meta.groupby("activity").record.nunique();nw=meta.groupby("record").size()
    w=np.array([1/(8*nr[r.activity]*nw[r.record]) for r in meta.itertuples()])
    assert np.isclose(w.sum(),1)
    for a in ACT:assert np.isclose(w[(meta.activity==a).to_numpy()].sum(),1/8)
    return w*(len(w)/w.sum()) if mean_one else w
def metrics(g):
    cm=confusion_matrix(g.true_index,g.pred_index,labels=range(8),sample_weight=g.eval_weight)
    tp=cm.diagonal();sup=cm.sum(1);pr=cm.sum(0)
    f=np.divide(2*tp,sup+pr,out=np.zeros(8),where=sup+pr>0)
    r=np.divide(tp,sup,out=np.zeros(8),where=sup>0)
    return cm,f,r
def run():
    started=time.perf_counter();frozen=json.loads((O/"freeze_receipt.json").read_text())
    assert sha(__file__)==frozen["script_sha256"] and sha(O/"frozen_config.json")==frozen["config_sha256"]
    assert sha(O/"frozen_folds.csv")==frozen["folds_sha256"] and sha(O/"window_metadata.csv")==frozen["metadata_sha256"]
    assert not (O/"analysis_complete.json").exists()
    tests();jw("feature_tests.json",dict(status="PASS",tests=["linear slope/sign of quarter contrasts","interior interpolation","nearest edges","all-missing rejection","constant signal dimensions/finite/shape centering"],models_fitted=0))
    meta=pd.read_csv(O/"window_metadata.csv");rec=pd.read_csv(O/"record_manifest_LOCAL.csv");n=len(meta);fa=np.empty((n,56));fb=np.empty((n,28));fc=np.empty((n,38))
    tick=time.perf_counter()
    for r in rec.itertuples():
        assert sha(r.input_path)==r.input_sha256
        raw=pd.read_csv(r.input_path)[CH].apply(pd.to_numeric,errors="coerce").to_numpy()
        for i in meta.index[meta.record==r.record]:
            row=meta.loc[i];fa[i],fb[i],fc[i]=extract(raw[int(row.start_row):int(row.end_row_exclusive)])
    X=dict(MIMU=fa[:,:42],HF_A=fa[:,42:],FUS_A=fa,HF_B=fb,FUS_B=np.column_stack([fa[:,:42],fb]),HF_SHAPE=fc,FUS_SHAPE=np.column_stack([fa[:,:42],fc]))
    assert all(np.isfinite(z).all() for z in X.values())
    np.savez_compressed(O/"feature_matrices.npz",**X);feature_time=time.perf_counter()-tick
    log(f"FEATURES complete: {n} windows, no exclusions, {feature_time:.3f}s.")
    y=meta.activity.map(dict(zip(ACT,range(8)))).to_numpy();eval_w=weights(meta)
    preds=[];cost=[];audit=[];scalers={};split_rows=[];fit_count=0;fit_started=time.perf_counter()
    for held in sorted(rec.record):
        train=(meta.record!=held).to_numpy();test=~train;trainmeta=meta[train]
        wt=weights(trainmeta);sw=weights(trainmeta,True)
        assert np.isclose(sw.mean(),1) and set(y[train])==set(range(8))
        assert set(meta.loc[train,"record"]).isdisjoint(set(meta.loc[test,"record"]))
        for rid,g in trainmeta.groupby("record"):
            mask=(trainmeta.record==rid).to_numpy()
            audit.append(dict(fold=held,train_record=rid,activity=g.iloc[0].activity,windows=len(g),et_weight_total=float(wt[mask].sum()),svc_weight_total=float(sw[mask].sum()),svc_total_sum=float(sw.sum()),train_windows=int(train.sum())))
        for cfg in MATRICES:
            tick=time.perf_counter();mat=X[cfg["arm"]];assert mat.shape[1]==cfg["features"]
            if cfg["classifier"]=="ET":
                model=ExtraTreesClassifier(**CFG["extra_trees"]);model.fit(mat[train],y[train],sample_weight=wt)
                pr=model.predict(mat[test]);score=model.predict_proba(mat[test]);kind="probability"
                info=dict(tree_count=len(model.estimators_),max_depth=max(t.get_depth() for t in model.estimators_),max_leaves=max(t.get_n_leaves() for t in model.estimators_),support_vectors=0,fit_status=0)
            else:
                scale=StandardScaler().fit(mat[train],sample_weight=sw)
                ztrain=scale.transform(mat[train]);ztest=scale.transform(mat[test])
                model=SVC(C=1.,kernel="rbf",gamma=1./cfg["features"],class_weight=None,probability=False,decision_function_shape="ovr",tol=.001,shrinking=True,cache_size=200,max_iter=-1)
                model.fit(ztrain,y[train],sample_weight=sw);assert model.fit_status_==0
                pr=model.predict(ztest);score=model.decision_function(ztest);kind="ovr_decision_not_probability"
                key=cfg["id"]+"__"+held
                scalers[key+"__mean"]=scale.mean_;scalers[key+"__var"]=scale.var_;scalers[key+"__scale"]=scale.scale_
                info=dict(tree_count=0,max_depth=0,max_leaves=0,support_vectors=int(model.n_support_.sum()),fit_status=int(model.fit_status_))
            assert np.array_equal(model.classes_,np.arange(8)) and score.shape==(test.sum(),8) and np.isfinite(score).all()
            for pos,i in enumerate(np.flatnonzero(test)):
                row=meta.loc[i];d=dict(configuration=cfg["id"],classifier=cfg["classifier"],arm=cfg["arm"],fold=held,original_row=int(i),record=row.record,activity=row.activity,true_index=int(y[i]),pred_index=int(pr[pos]),prediction=ACT[int(pr[pos])],window_idx=int(row.window_idx),center_s=float(row.center_s),eval_weight=float(eval_w[i]),missing_bin=row.missing_bin,missing_fraction=float(row.missing_fraction),score_kind=kind)
                d.update({f"score_{a}":float(score[pos,k]) for k,a in enumerate(ACT)});preds.append(d)
            cost.append(dict(configuration=cfg["id"],fold=held,features=cfg["features"],train_windows=int(train.sum()),test_windows=int(test.sum()),seconds=time.perf_counter()-tick,**info));fit_count+=1
        log(f"LORO {held}: {fit_count}/288 fits complete (12 configurations).")
    fit_time=time.perf_counter()-fit_started
    save("oof_predictions.csv",preds);save("fit_audit.csv",cost);save("training_weights.csv",audit);np.savez_compressed(O/"fold_scalers.npz",**scalers)
    p=pd.DataFrame(preds);results=[];perclass=[];perrecord=[];conf=[];quality=[]
    for config,g in p.groupby("configuration",sort=False):
        cm,f,r=metrics(g);votes=[]
        for rid,rg in g.groupby("record"):
            hist=np.bincount(rg.pred_index,minlength=8);vote=int(np.argmax(hist));truth=int(rg.true_index.iloc[0]);votes.append(vote==truth)
            perrecord.append(dict(configuration=config,record=rid,activity=ACT[truth],windows=len(rg),accuracy=float((rg.pred_index==truth).mean()),majority_prediction=ACT[vote],majority_correct=bool(vote==truth),majority_tie_count=int((hist==hist.max()).sum())))
        results.append(dict(configuration=config,macro_F1=float(f.mean()),balanced_accuracy=float(r.mean()),weighted_window_accuracy=float(np.average(g.true_index==g.pred_index,weights=g.eval_weight)),majority_record_accuracy=float(np.mean(votes)),correct_majority_records=int(sum(votes))))
        for i,a in enumerate(ACT):
            perclass.append(dict(configuration=config,activity=a,F1=float(f[i]),recall=float(r[i])))
            for j,b in enumerate(ACT):conf.append(dict(configuration=config,true_activity=a,predicted_activity=b,weight=float(cm[i,j])))
        for lev in CFG["quality_strata"]:
            gg=g[g.missing_bin==lev];present=int(gg.activity.nunique())
            d=dict(configuration=config,missing_bin=lev,windows=len(gg),records=int(gg.record.nunique()),activity_count=present,activities=";".join(sorted(gg.activity.unique())),weighted_accuracy=float(np.average(gg.true_index==gg.pred_index,weights=gg.eval_weight)) if len(gg) else "",macro_F1="",balanced_accuracy="")
            if present==8:
                _,ff,rr=metrics(gg);d.update(macro_F1=float(ff.mean()),balanced_accuracy=float(rr.mean()))
            quality.append(d)
    save("summary_metrics.csv",results);save("activity_metrics.csv",perclass);save("record_metrics.csv",perrecord);save("confusion_matrices.csv",conf);save("missingness_strata.csv",quality)
    complements=[]
    for clf in ["ET","SVC"]:
        anchor=p[p.configuration==clf+"_MIMU"].set_index("original_row").sort_index();bc=anchor.true_index==anchor.pred_index
        targets=["FUS_A","FUS_B"]+(["FUS_SHAPE"] if clf=="SVC" else [])
        for target in targets:
            gg=p[p.configuration==clf+"_"+target].set_index("original_row").sort_index();fcorr=gg.true_index==gg.pred_index
            for label,sel in [("ALL",np.ones(n,bool))]+[(a,(anchor.activity==a).to_numpy()) for a in ACT]:
                w=anchor.eval_weight.to_numpy()[sel];b=bc.to_numpy()[sel];f=fcorr.to_numpy()[sel]
                complements.append(dict(comparison=clf+"_"+target+" minus "+clf+"_MIMU",activity=label,windows=int(sel.sum()),baseline_correct=int(b.sum()),baseline_wrong=int((~b).sum()),rescued=int((~b&f).sum()),damaged=int((b&~f).sum()),both_correct=int((b&f).sum()),both_wrong=int((~b&~f).sum()),weighted_rescued=float(w[~b&f].sum()/w.sum()),weighted_damaged=float(w[b&~f].sum()/w.sum()),baseline_error_ceiling=float(w[~b].sum()/w.sum())))
    save("complementarity.csv",complements)
    log("ALL 12 CONFIGURATIONS COMPLETE; no performance-driven iteration.")
    # Finite record-level descriptive exploration, completely separate from models.
    tick=time.perf_counter();labels=rec.sort_values("record").reset_index(drop=True).copy()
    labels["plot_date"]=labels.date_evidence.fillna("")
    labels.loc[labels.record=="HG-03","plot_date"]="Unknown date (HG-03)"
    labels.loc[labels.record=="RS-02","plot_date"]="Conflicting date (RS-02)"
    labels.to_csv(O/"exploration_record_labels_LOCAL.csv",index=False)
    public=labels[["record","subject","activity","plot_date"]];public.to_csv(O/"exploration_record_labels.csv",index=False)
    spaces={"A_MIMU":("MIMU",0),"A_HF":("HF_A",0),"A_FUS":("FUS_A",14),"B_MIMU":("MIMU",0),"B_HF":("HF_B",0),"B_FUS":("FUS_B",28)}
    arrays={};cr=[];er=[];assign=[];medians={}
    for space,(arm,hfdim) in spaces.items():
        med=np.array([np.median(X[arm][(meta.record==rid).to_numpy()],axis=0) for rid in labels.record]);medians[space]=med
        z=StandardScaler().fit_transform(med)
        if hfdim:z[:,:42]/=np.sqrt(42);z[:,42:]/=np.sqrt(hfdim)
        else:z/=np.sqrt(z.shape[1])
        km=KMeans(**CFG["exploration"]["kmeans"]);c=km.fit_predict(z)
        cr.append(dict(space=space,n_records=24,K=8,activity_ARI=float(adjusted_rand_score(labels.activity,c)),activity_AMI=float(adjusted_mutual_info_score(labels.activity,c)),wear_ARI="not computed: each wear record unique"))
        for rid,k in zip(labels.record,c):assign.append(dict(space=space,record=rid,cluster=int(k)))
        pc=PCA(n_components=2,svd_solver="full");arrays[space+"__PCA"]=pc.fit_transform(z)
        er.append(dict(space=space,embedding="PCA",explained_variance_ratio=float(pc.explained_variance_ratio_.sum()),kl=""))
        for per in [5,10]:
            ts=TSNE(n_components=2,perplexity=per,random_state=20261005,init="pca",learning_rate="auto",max_iter=1000,method="barnes_hut",angle=.5,n_jobs=1)
            arrays[space+"__tSNE_p"+str(per)]=ts.fit_transform(z)
            er.append(dict(space=space,embedding="tSNE_p"+str(per),explained_variance_ratio="",kl=float(ts.kl_divergence_)))
    np.savez_compressed(O/"record_medians.npz",**medians);np.savez_compressed(O/"record_embeddings.npz",**arrays)
    save("clustering_metrics.csv",cr);save("cluster_assignments.csv",assign);save("embedding_diagnostics.csv",er)
    for row in rec.itertuples():assert sha(row.input_path)==row.input_sha256
    summary=pd.DataFrame(results).set_index("configuration")
    jw("analysis_complete.json",dict(status="completed",models=fit_count,configurations=12,records=24,windows=n,oof_rows=len(preds),feature_seconds=feature_time,fit_seconds=fit_time,exploration_seconds=time.perf_counter()-tick,total_seconds=time.perf_counter()-started,primary_delta_macro_F1=float(summary.loc["ET_FUS_B","macro_F1"]-summary.loc["ET_MIMU","macro_F1"]),primary_delta_BA=float(summary.loc["ET_FUS_B","balanced_accuracy"]-summary.loc["ET_MIMU","balanced_accuracy"]),raw_hashes_unchanged=24,search_budget=0))
    log("COMPLETE\n"+pd.DataFrame(results).to_string(index=False))
    jw("numeric_manifest.json",[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(O.iterdir()) if p.is_file() and p.name!="numeric_manifest.json"])
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--freeze-only",action="store_true");a.add_argument("--run",action="store_true");args=a.parse_args()
    assert args.freeze_only != args.run
    with threadpool_limits(limits=1):
        if args.freeze_only:freeze()
        else:run()

