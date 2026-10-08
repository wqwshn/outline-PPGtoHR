"""Approved frozen first-round classification. Separate exploration from LOSO."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
import argparse,csv,hashlib,json,platform,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
import scipy,sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,adjusted_rand_score,adjusted_mutual_info_score,silhouette_score
from threadpoolctl import threadpool_limits

ARMS=("MIMU","HF","MIMU_HF")
CHANNELS=["AccX(g)","AccY(g)","AccZ(g)","GyroX(dps)","GyroY(dps)","GyroZ(dps)","Ut1(mV)","Ut2(mV)"]
ACT=["HW","RS","HG","TYP","RUN","JJ","BUR","PCH"]
FEATURES=["mean","std_ddof0","peak_to_peak","mean_abs_first_difference","dominant_frequency_0.5_20Hz","normalized_spectral_entropy_0.5_20Hz","power_ratio_0.5_5Hz_to_0.5_20Hz"]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def jwrite(p,v):
    with p.open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def savecsv(p,rows):pd.DataFrame(rows).to_csv(p,index=False,encoding="utf-8")
def impute(a):
    b=np.array(a,dtype=float,copy=True)
    for k in range(b.shape[1]):
        valid=np.isfinite(b[:,k])
        if not valid.any():raise ValueError("All-missing window channel; no exclusion permitted")
        xp=np.flatnonzero(valid);assert np.all(np.diff(xp)>0)
        b[:,k]=np.interp(np.arange(len(b)),xp,b[valid,k])
    return b
def features(a):
    a=impute(a); out=[]
    f=np.fft.rfftfreq(800,.01);band=(f>=.5)&(f<=20); low=(f>=.5)&(f<=5)
    p=np.abs(np.fft.rfft((a-a.mean(axis=0))*np.hanning(800)[:,None],axis=0))**2
    for k in range(8):
        x=a[:,k]; power=p[band,k]; total=power.sum()
        if total>0:
            dist=power/total; pos=dist>0
            dom=f[band][int(np.argmax(power))]
            entropy=-(dist[pos]*np.log(dist[pos])).sum()/np.log(len(power))
            ratio=p[low,k].sum()/total
        else:dom=entropy=ratio=0.
        out.extend([x.mean(),x.std(ddof=0),np.ptp(x),np.abs(np.diff(x)).mean(),dom,entropy,ratio])
    return np.asarray(out,dtype=float)
def weights(meta):
    nsub=meta.subject.nunique(); counts=meta.groupby(["subject","activity"]).record.nunique()
    nw=meta.groupby("record").size()
    out=np.array([1/(nsub*8*counts.loc[(r.subject,r.activity)]*nw.loc[r.record]) for r in meta.itertuples()])
    assert np.isclose(out.sum(),1)
    for s in meta.subject.unique():
        assert np.isclose(out[(meta.subject==s).to_numpy()].sum(),1/nsub)
        for a in ACT:
            m=((meta.subject==s)&(meta.activity==a)).to_numpy()
            assert np.isclose(out[m].sum(),1/(nsub*8))
    for rec,g in meta.groupby("record",sort=False):
        s=g.iloc[0].subject;a=g.iloc[0].activity
        assert np.isclose(out[(meta.record==rec).to_numpy()].sum(),1/(nsub*8*counts.loc[(s,a)]))
    return out
def metric(y,p,w):
    cm=confusion_matrix(y,p,labels=np.arange(8),sample_weight=w)
    support=cm.sum(axis=1);pred=cm.sum(axis=0);tp=cm.diagonal()
    f1=np.divide(2*tp,support+pred,out=np.zeros(8),where=support+pred>0)
    rec=np.divide(tp,support,out=np.zeros(8),where=support>0)
    return cm,float(f1.mean()),float(rec.mean()),f1,rec
def quality_bin(n):
    return "0%" if n==0 else "(0,10%]" if n<=80 else "(10,25%]" if n<=200 else ">25%"
def self_test():
    a=np.tile(np.arange(800,dtype=float)[:,None],(1,8));a[30:40]=np.nan
    assert np.allclose(impute(a)[30:40,0],np.arange(30,40))
    a[:5]=np.nan;assert (impute(a)[:5,0]==5).all()
    try:impute(np.full((800,8),np.nan));raise AssertionError("Did not reject")
    except ValueError:pass
    assert features(np.ones((800,8))).shape==(56,)
    assert np.isfinite(features(np.ones((800,8)))).all()
    cm,f,b,_,_=metric(np.arange(8),np.arange(8),np.ones(8)/8);assert f==b==1
    _,f,b,_,_=metric(np.arange(8),(np.arange(8)+1)%8,np.ones(8)/8);assert f==b==0
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--run-id",default="p1_20261005_01");args=parser.parse_args()
    root=args.root.resolve();base=root/"data/experiments/hf_scene_classification_v1";p0=base/"p0_20261005_02";out=base/args.run_id
    if not out.resolve().is_relative_to(base.resolve()) or out==base:raise ValueError("Invalid output")
    out.mkdir(exist_ok=False);begin=time.perf_counter();self_test()
    for x in json.loads((p0/"output_manifest.json").read_text(encoding="utf-8")):assert sha(p0/x["path"])==x["sha256"]
    meta=pd.read_csv(p0/"motion_windows.csv");bindings=pd.read_csv(p0/"record_bindings_LOCAL.csv")
    assert len(meta)==9520 and len(bindings)==119
    cols={"MIMU":np.arange(42),"HF":np.arange(42,56),"MIMU_HF":np.arange(56)}
    cfg=dict(status="approved_frozen_before_features",approved="2026-10-05 user approval",activities=ACT,channels=CHANNELS,features=FEATURES,features_per_channel=7,
        model=dict(n_estimators=200,max_depth=4,max_leaf_nodes=16,min_samples_leaf=1,max_features=1.0,bootstrap=False,criterion="gini",random_state=20261005,n_jobs=1,class_weight=None),
        imputation="Per 800-point window/channel: linear internal gaps, nearest edges via np.interp; all-missing fails, no drops; no flag predictors.",
        spectral="Demean + numpy.hanning(800), rFFT squared magnitudes; inclusive bands 0.5-20Hz, low 0.5-5Hz; normalized entropy/log(bin_count), zero-band energy => zeros, first max tie.",
        supervised_preprocessing="No learned scaler/PCA; frozen deterministic per-window features. No feature search.",
        weights="1/(N_subjects * 8 * N_records_subject_activity * N_windows_record). Per-subject test uses N_subjects=1.",
        exploration=dict(sample="Per record sort centers, greedily retain windows separated >=8s; choose exactly 8 evenly spaced entries by floor(linspace). No activity/subject/quality used in selection; record identity groups only.",sample_per_record=8,
            scaling="Fit StandardScaler on exploration sample only. Single arms divide by sqrt(d); fusion divides MIMU block by sqrt(42), HF by sqrt(14), equal expected squared block norm.",
            kmeans=dict(n_clusters=8,n_init=20,random_state=20261005,max_iter=300,algorithm="lloyd",space="standardized block-scaled full feature space; not t-SNE"),
            pca=dict(n_components=2,svd_solver="full"),
            tsne=dict(n_components=2,perplexities=[15,40],seeds=[20261005,20261006],init="pca",learning_rate="auto",max_iter=1000,method="barnes_hut",angle=.5,metric="euclidean",early_exaggeration=12.0,n_jobs=1),
            labels="Activity, anonymous subject and missingness are used only after fitting for ARI/AMI and coloring. K=8 uses known class-count prior."),
        search_budget=0,LOSO="6 global subject folds",all_windows=9520,all_records=119,missing_strata=["0%","(0,10%]","(10,25%]",">25%"],primary="mean of six paired subject weighted macroF1 deltas MIMU_HF minus MIMU",
        figure_contract="Exploratory report, Python matplotlib; 183mm width 600dpi PNG + editable PDF/SVG. Paired subject effects as primary evidence; embedding views descriptive, no between-plot distance/area claims.")
    assert type(cfg["model"]["max_features"]) is float and cfg["model"]["max_features"]==1.0
    jwrite(out/"frozen_config.json",cfg)
    jwrite(out/"freeze_receipt.json",dict(utc=datetime.now(timezone.utc).isoformat(),config_sha256=sha(out/"frozen_config.json"),script_sha256=sha(__file__),p0_manifest_sha256=sha(p0/"output_manifest.json"),git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),command=subprocess.list2cmdline([sys.executable,"-B",*sys.argv]),python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,training_before_freeze=False))
    (out/"progress.log").write_text("Configuration frozen before features and results.\n",encoding="utf-8")
    def progress(msg):
        print(msg,flush=True)
        with (out/"progress.log").open("a",encoding="utf-8") as f:f.write(msg+"\n")
    # Extract each record once, never call the HR pipeline.
    started=time.perf_counter();X=np.empty((len(meta),56));input_fps=[]
    for r in bindings.itertuples():
        p=Path(r.input_path);assert sha(p)==r.input_sha256
        raw=pd.read_csv(p);a=raw[CHANNELS].apply(pd.to_numeric,errors="coerce").to_numpy()
        for i in meta.index[meta.record==r.record]:
            row=meta.loc[i];s=int(row.start_row);e=int(row.end_row_exclusive)
            assert e-s==800 and 0<=s<e<=len(a)
            X[i]=features(a[s:e])
        input_fps.append(dict(record=r.record,subject=r.subject,sha256=r.input_sha256))
    assert np.isfinite(X).all()
    np.savez_compressed(out/"features.npz",X=X)
    meta["missing_fraction"]=meta.missing_sample_rows/800
    meta["missing_bin"]=[quality_bin(n) for n in meta.missing_sample_rows]
    meta.to_csv(out/"window_metadata.csv",index=False)
    savecsv(out/"feature_columns.csv",[dict(index=i,channel=CHANNELS[i//7],feature=FEATURES[i%7]) for i in range(56)])
    jwrite(out/"input_hashes.json",input_fps)
    feature_seconds=time.perf_counter()-started
    # Technical pilot: one predetermined global fold, all three arms, no scores inspected.
    y=meta.activity.map({x:i for i,x in enumerate(ACT)}).to_numpy()
    mask=(meta.subject!="subject-1").to_numpy();wt=weights(meta[mask]);pilot=[]
    for arm in ARMS:
        t=time.perf_counter();m=ExtraTreesClassifier(**cfg["model"])
        m.fit(X[mask][:,cols[arm]],y[mask],sample_weight=wt)
        prob=m.predict_proba(X[~mask][:,cols[arm]])
        assert prob.shape==(int((~mask).sum()),8) and np.allclose(prob.sum(axis=1),1)
        pilot.append(dict(arm=arm,seconds=time.perf_counter()-t,train_windows=int(mask.sum()),test_windows=int((~mask).sum()),scores_computed=False))
    jwrite(out/"pilot_receipt.json",dict(feature_seconds=feature_seconds,arms=pilot,estimated_18_model_seconds=sum(x["seconds"] for x in pilot)*6,basis="three-arm subject-1 technical fold times six; approximate, not a guarantee",self_tests="PASS",weights_assertions="subject/activity/record sums PASS",decision="continue unchanged",no_scores_inspected=True))
    progress("PILOT "+json.dumps(dict(feature_seconds=feature_seconds,arms=pilot),ensure_ascii=False))
    # Exploration uses sampled features only and is never returned to classification.
    chosen=[]
    for rec,g in meta.groupby("record",sort=True):
        candidates=[];last=-np.inf
        for i in g.sort_values("center_s").index:
            if meta.loc[i,"center_s"]>=last+8-1e-9:candidates.append(i);last=meta.loc[i,"center_s"]
        assert len(candidates)>=8
        indices=np.floor(np.linspace(0,len(candidates)-1,8)).astype(int);assert len(set(indices))==8
        chosen.extend(candidates[k] for k in indices)
    sm=meta.loc[chosen].copy();sm.insert(0,"original_row",chosen);sm.to_csv(out/"exploration_samples.csv",index=False)
    assert len(sm)==952 and sm.groupby("record").size().eq(8).all()
    t=time.perf_counter();cluster_rows=[];embedding_rows=[];explore_arrays={}
    for arm in ARMS:
        scaler=StandardScaler();z=scaler.fit_transform(X[chosen][:,cols[arm]])
        if arm=="MIMU_HF":z[:,:42]/=np.sqrt(42);z[:,42:]/=np.sqrt(14)
        else:z/=np.sqrt(z.shape[1])
        km=KMeans(**{k:v for k,v in cfg["exploration"]["kmeans"].items() if k!="space"})
        c=km.fit_predict(z)
        cluster_rows.append(dict(arm=arm,n=len(z),K=8,activity_ARI=adjusted_rand_score(sm.activity,c),activity_AMI=adjusted_mutual_info_score(sm.activity,c),subject_ARI=adjusted_rand_score(sm.subject,c),subject_AMI=adjusted_mutual_info_score(sm.subject,c),silhouette_descriptive=silhouette_score(z,c)))
        savecsv(out/f"clusters_{arm}.csv",[dict(original_row=int(i),cluster=int(k)) for i,k in zip(chosen,c)])
        pca=PCA(n_components=2,svd_solver="full");xy=pca.fit_transform(z);explore_arrays[f"{arm}_PCA"]=xy
        embedding_rows.append(dict(arm=arm,embedding="PCA",variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),kl_divergence=""))
        for perplexity in (15,40):
            for seed in (20261005,20261006):
                ts=TSNE(n_components=2,perplexity=perplexity,random_state=seed,init="pca",learning_rate="auto",max_iter=1000,method="barnes_hut",angle=.5,metric="euclidean",early_exaggeration=12.,n_jobs=1)
                xy=ts.fit_transform(z);key=f"tSNE_p{perplexity}_s{seed}";explore_arrays[f"{arm}_{key}"]=xy
                embedding_rows.append(dict(arm=arm,embedding=key,variance_ratio_sum="",kl_divergence=float(ts.kl_divergence_)))
        # Scaler parameters saved as exploration-only, not classifier dependencies.
        np.savez_compressed(out/f"exploration_transform_{arm}.npz",mean=scaler.mean_,scale=scaler.scale_,pca_components=pca.components_,pca_mean=pca.mean_)
    np.savez_compressed(out/"embeddings.npz",**explore_arrays)
    savecsv(out/"clustering_metrics.csv",cluster_rows);savecsv(out/"embedding_diagnostics.csv",embedding_rows)
    exploration_seconds=time.perf_counter()-t
    progress("EXPLORATION "+json.dumps(cluster_rows,ensure_ascii=False))
    # Full LOSO: independent model objects on original feature matrix, no exploration transforms.
    t=time.perf_counter();pred_rows=[];score_rows=[];class_rows=[];record_rows=[];strata_rows=[];cm_rows=[];weight_rows=[];model_cost=[]
    for sub in sorted(meta.subject.unique()):
        train=(meta.subject!=sub).to_numpy();test=~train;tm=meta[test];train_weights=weights(meta[train]);test_weights=weights(tm)
        for rec,g in meta[train].groupby("record"):
            inds=(meta[train].record==rec).to_numpy()
            weight_rows.append(dict(fold=sub,role="train",record=rec,subject=g.iloc[0].subject,activity=g.iloc[0].activity,windows=len(g),total_weight=float(train_weights[inds].sum())))
        for arm in ARMS:
            tick=time.perf_counter();m=ExtraTreesClassifier(**cfg["model"]);m.fit(X[train][:,cols[arm]],y[train],sample_weight=train_weights)
            probs=m.predict_proba(X[test][:,cols[arm]]);pr=m.classes_[np.argmax(probs,axis=1)]
            assert np.array_equal(m.classes_,np.arange(8))
            cm,f1,ba,per_f1,recall=metric(y[test],pr,test_weights)
            score_rows.append(dict(subject=sub,arm=arm,macro_F1=f1,balanced_accuracy=ba,records=tm.record.nunique(),windows=len(tm)))
            for k,a in enumerate(ACT):
                class_rows.append(dict(subject=sub,arm=arm,activity=a,F1=per_f1[k],recall=recall[k],support_weight=cm[k].sum()))
                for j,b in enumerate(ACT):cm_rows.append(dict(subject=sub,arm=arm,true_activity=a,predicted_activity=b,weight=cm[k,j]))
            for pos,(i,row) in enumerate(tm.iterrows()):
                d=dict(original_row=int(i),record=row.record,subject=sub,activity=row.activity,arm=arm,window_idx=int(row.window_idx),center_s=row.center_s,missing_fraction=row.missing_fraction,missing_bin=row.missing_bin,test_weight=test_weights[pos],prediction=ACT[pr[pos]])
                d.update({f"p_{a}":probs[pos,k] for k,a in enumerate(ACT)});pred_rows.append(d)
            for rec,g in tm.groupby("record"):
                sel=(tm.record==rec).to_numpy()
                record_rows.append(dict(record=rec,subject=sub,activity=g.iloc[0].activity,arm=arm,windows=int(sel.sum()),accuracy=float(np.mean(pr[sel]==y[test][sel])),missing_window_fraction=float(np.mean(g.missing_sample_rows>0))))
            for level in cfg["missing_strata"]:
                sel=(tm.missing_bin==level).to_numpy();present=sorted(set(tm.loc[sel,"activity"]))
                if sel.any():
                    _,sf,sb,_,_=metric(y[test][sel],pr[sel],test_weights[sel])
                    acc=float(np.average(pr[sel]==y[test][sel],weights=test_weights[sel]))
                else:sf=sb=acc=""
                strata_rows.append(dict(subject=sub,arm=arm,missing_bin=level,windows=int(sel.sum()),records=tm.loc[sel,"record"].nunique(),activities=";".join(present),activity_count=len(present),full_eight_class_coverage=len(present)==8,weighted_accuracy=acc,macro_F1_eight_classes=sf if len(present)==8 else "",balanced_accuracy_eight_classes=sb if len(present)==8 else "",coverage_note="descriptive conditional subset; no refit; macroF1/BA suppressed if not all eight classes"))
            model_cost.append(dict(subject=sub,arm=arm,seconds=time.perf_counter()-tick,features=len(cols[arm]),tree_count=len(m.estimators_),max_depth_actual=max(z.get_depth() for z in m.estimators_),max_leaves_actual=max(z.get_n_leaves() for z in m.estimators_)))
        progress(f"LOSO completed {sub}, 3 models")
    for name,data in [("oof_predictions",pred_rows),("subject_metrics",score_rows),("activity_metrics",class_rows),("record_metrics",record_rows),("missingness_strata",strata_rows),("confusion_matrices",cm_rows),("training_weight_audit",weight_rows),("model_costs",model_cost)]:savecsv(out/f"{name}.csv",data)
    scores=pd.DataFrame(score_rows);pivot=scores.pivot(index="subject",columns="arm",values="macro_F1")
    deltas=pd.DataFrame(dict(subject=pivot.index,MIMU=pivot.MIMU,HF=pivot.HF,MIMU_HF=pivot.MIMU_HF,delta=pivot.MIMU_HF-pivot.MIMU)).reset_index(drop=True)
    deltas.to_csv(out/"paired_subject_deltas.csv",index=False)
    classification_seconds=time.perf_counter()-t
    summary=dict(arm_means=scores.groupby("arm")[["macro_F1","balanced_accuracy"]].mean().to_dict("index"),mean_paired_delta=float(deltas.delta.mean()),subject_deltas=deltas[["subject","delta"]].to_dict("records"),positive_subjects=int((deltas.delta>0).sum()),records=119,windows=9520,oof_rows=len(pred_rows),models=18,pilot_models=3,no_tuning=True,no_exclusions=True,feature_seconds=feature_seconds,exploration_seconds=exploration_seconds,classification_seconds=classification_seconds,total_seconds=time.perf_counter()-begin,config_sha256=sha(out/"frozen_config.json"),script_sha256=sha(__file__),imputation_performed_on_derived_arrays_only=True,not_done=["alternative-imputation rerun","causal online deployment","new subjects","HR improvement test"])
    jwrite(out/"summary.json",summary)
    progress("FINAL "+json.dumps(summary,ensure_ascii=False))
    jwrite(out/"analysis_manifest.json",[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()])
if __name__=="__main__":
    with threadpool_limits(limits=1):main()
