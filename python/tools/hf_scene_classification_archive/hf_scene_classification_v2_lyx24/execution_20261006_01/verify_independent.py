"""Independent verifier. Reads persisted predictions/features, no model fitting."""
from pathlib import Path
import ast,json,hashlib,subprocess
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
O=Path(__file__).resolve().parent;R=O.parents[3];checks=[];errors=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ok(name,v):
    if not bool(v):raise AssertionError(name)
    checks.append(name)
def close(name,x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);e=float(np.max(abs(x-y))) if x.size else 0
    errors.append(e);ok(name,np.allclose(x,y,rtol=1e-9,atol=2e-10))
cfg=json.loads((O/"frozen_config.json").read_text());fr=json.loads((O/"freeze_receipt.json").read_text())
manifest=json.loads((O/"numeric_manifest.json").read_text());ok("all frozen numeric hashes",all(sha(O/r["path"])==r["sha256"] for r in manifest))
ok("actual executed source hash",sha(O/"run_fixed_matrix.py")==fr["script_sha256"])
ok("configuration frozen hash",sha(O/"frozen_config.json")==fr["config_sha256"])
ok("max_features float1.0",type(cfg["extra_trees"]["max_features"]) is float and cfg["extra_trees"]["max_features"]==1.)
source=ast.parse((O/"run_fixed_matrix.py").read_text(encoding="utf-8-sig"))
fits=[n for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="fit" and isinstance(n.func.value,ast.Name) and n.func.value.id=="model"]
ok("both ET and SVC actual fit receive sample weights",len(fits)==2 and all(any(k.arg=="sample_weight" for k in x.keywords) for x in fits))
m=pd.read_csv(O/"window_metadata.csv");r=pd.read_csv(O/"record_manifest_LOCAL.csv");p=pd.read_csv(O/"oof_predictions.csv")
folds=pd.read_csv(O/"frozen_folds.csv");wa=pd.read_csv(O/"training_weights.csv");fa=pd.read_csv(O/"fit_audit.csv");X=np.load(O/"feature_matrices.npz");sc=np.load(O/"fold_scalers.npz")
act=cfg["activities"];s=pd.read_csv(O/"summary_metrics.csv").set_index("configuration");ac=pd.read_csv(O/"activity_metrics.csv");rc=pd.read_csv(O/"record_metrics.csv");cm_saved=pd.read_csv(O/"confusion_matrices.csv");qs=pd.read_csv(O/"missingness_strata.csv")
ok("12 configurations x1941 exactly",len(p)==12*1941 and p.configuration.nunique()==12 and not p.duplicated(["configuration","original_row"]).any())
ok("288 successful fits",len(fa)==288 and fa.fit_status.eq(0).all())
ok("ET resource limits",fa[fa.configuration.str.startswith("ET")].tree_count.eq(200).all() and fa.max_depth.max()<=4 and fa.max_leaves.max()<=16)
ok("1941 windows same frozen preparation",m[["record","window_idx","center_s","start_row","end_row_exclusive"]].equals(pd.read_csv(O.parent/"preparation_20261006_01/frozen_motion_windows.csv")[["record","window_idx","center_s","start_row","end_row_exclusive"]]))
ok("24 raw hashes unchanged",all(sha(z.input_path)==z.input_sha256 for z in r.itertuples()))
for spec in cfg["matrix"]:ok(spec["id"]+" feature dimensions",X[spec["arm"]].shape==(1941,spec["features"]) and np.isfinite(X[spec["arm"]]).all())
for held,fg in folds.groupby("fold"):
    train=fg[fg.role=="train"];test=fg[fg.role=="test"]
    ok(held+" whole record split",len(train)==23 and len(test)==1 and test.record.iloc[0]==held and set(train.record).isdisjoint(test.record) and train.activity.nunique()==8)
    tm=m[m.record!=held];nr=tm.groupby("activity").record.nunique();nw=tm.groupby("record").size()
    w=np.array([1/(8*nr[z.activity]*nw[z.record]) for z in tm.itertuples()]);sw=w*len(w)/w.sum()
    close(held+" mean1 SVC weights",sw.mean(),1)
    audit=wa[wa.fold==held].set_index("train_record")
    close(held+" ET record totals",[1/(8*nr[z.activity]) for z in train.itertuples()],audit.loc[train.record,"et_weight_total"])
    close(held+" SVC record totals",[len(tm)/(8*nr[z.activity]) for z in train.itertuples()],audit.loc[train.record,"svc_weight_total"])
    close(held+" SVC saved sum",audit.svc_total_sum,np.full(len(audit),len(tm)))
    for spec in cfg["matrix"]:
        if spec["classifier"]!="SVC":continue
        z=X[spec["arm"]][(m.record!=held).to_numpy()];mean=np.average(z,axis=0,weights=sw);var=np.average((z-mean)**2,axis=0,weights=sw)
        key=spec["id"]+"__"+held
        close(key+" weighted train-only scaler mean",mean,sc[key+"__mean"]);close(key+" weighted train-only scaler var",var,sc[key+"__var"])
        ok(key+" scale positive finite",np.isfinite(sc[key+"__scale"]).all() and (sc[key+"__scale"]>0).all())
def calc(g):
    cm=np.zeros((8,8));np.add.at(cm,(g.true_index.to_numpy(),g.pred_index.to_numpy()),g.eval_weight.to_numpy())
    support=cm.sum(1);prediction=cm.sum(0);tp=cm.diagonal()
    f=np.divide(2*tp,support+prediction,out=np.zeros(8),where=support+prediction>0)
    recall=np.divide(tp,support,out=np.zeros(8),where=support>0)
    return cm,f,recall
for name,g in p.groupby("configuration"):
    g=g.sort_values("original_row");orig=m.iloc[g.original_row]
    ok(name+" OOF identities",np.array_equal(g.record,orig.record) and np.array_equal(g.activity,orig.activity) and (g.fold==g.record).all() and set(g.original_row)==set(range(1941)))
    ok(name+" predicted indices labels",np.array_equal(np.array(act)[g.pred_index],g.prediction))
    expected=np.array([1/(24*int((m.record==z.record).sum())) for z in g.itertuples()])
    close(name+" independent OOF hierarchical weight",g.eval_weight,expected)
    cm,f,recall=calc(g);close(name+" macroF1 BA accuracy",[f.mean(),recall.mean(),np.average(g.true_index==g.pred_index,weights=expected)],s.loc[name,["macro_F1","balanced_accuracy","weighted_window_accuracy"]])
    aa=ac[ac.configuration==name].set_index("activity").loc[act];close(name+" all per-class metrics",np.c_[f,recall],aa[["F1","recall"]])
    saved=cm_saved[cm_saved.configuration==name].pivot(index="true_activity",columns="predicted_activity",values="weight").loc[act,act];close(name+" confusion cells",cm,saved)
    votes=[]
    for rid,gg in g.groupby("record"):
        rr=rc[(rc.configuration==name)&(rc.record==rid)].iloc[0];counts=np.bincount(gg.pred_index,minlength=8);v=int(np.argmax(counts));truth=int(gg.true_index.iloc[0])
        close(name+rid+" record accuracy",np.mean(gg.true_index==gg.pred_index),rr.accuracy)
        ok(name+rid+" record majority/tie",rr.majority_prediction==act[v] and int(rr.majority_tie_count)==int((counts==counts.max()).sum()));votes.append(v==truth)
    close(name+" majority pooled record accuracy",np.mean(votes),s.loc[name,"majority_record_accuracy"])
    for _,q in qs[qs.configuration==name].iterrows():
        gg=g[g.missing_bin==q.missing_bin];ok(name+q.missing_bin+" subset coverage",len(gg)==q.windows and gg.activity.nunique()==q.activity_count)
        if len(gg):close(name+q.missing_bin+" subset accuracy",np.average(gg.true_index==gg.pred_index,weights=gg.eval_weight),q.weighted_accuracy)
        if gg.activity.nunique()==8:
            _,ff,rr=calc(gg);close(name+q.missing_bin+" subset metrics",[ff.mean(),rr.mean()],[q.macro_F1,q.balanced_accuracy])
        else:ok(name+q.missing_bin+" unavailable eight-class metric suppressed",pd.isna(q.macro_F1) and pd.isna(q.balanced_accuracy))
    score=g[["score_"+a for a in act]].to_numpy();ok(name+" finite score fields",np.isfinite(score).all())
    if name.startswith("ET"):
        close(name+" probabilities sum1",score.sum(1),np.ones(len(g)));ok(name+" ET prediction argmax",np.array_equal(score.argmax(1),g.pred_index))
# Independently recompute A/B/shape for first and most-missing window of each record.
spots=[]
def spec(x):
    freq=np.arange(401)/8;power=np.abs(np.fft.rfft((x-np.mean(x))*(.5-.5*np.cos(2*np.pi*np.arange(800)/799))))**2
    mask=(freq>=.5)&(freq<=20);lo=(freq>=.5)&(freq<=5);e=power[mask].sum()
    if e==0:return [0.,0.,0.]
    prob=power[mask]/e;pos=prob>0
    return [freq[mask][np.argmax(power[mask])],-np.dot(prob[pos],np.log(prob[pos]))/np.log(pos.size),power[lo].sum()/e]
offset=np.arange(800)[:,None]+np.arange(-49,51)[None,:];reflect=np.where(offset<0,-offset,offset);reflect=np.where(reflect>799,1598-reflect,reflect)
for rid,group in m.groupby("record"):
    inp=r.set_index("record").loc[rid,"input_path"];raw=pd.read_csv(inp)[cfg["channels"]].apply(pd.to_numeric,errors="coerce")
    for idx in sorted(set([group.index[0],group.missing_rows.idxmax()])):
        row=m.loc[idx];z=raw.iloc[int(row.start_row):int(row.end_row_exclusive)].replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").to_numpy()
        base=[];hb=[];shape=[]
        for j in range(8):
            x=z[:,j];a=[np.mean(x),np.sqrt(np.mean((x-x.mean())**2)),np.max(x)-np.min(x),np.sum(np.abs(np.diff(x)))/799,*spec(x)];base.extend(a)
            if j>=6:
                slow=x[reflect].mean(1);res=x-slow;q=[np.mean(x[k:k+200]) for k in range(0,800,200)]
                b=[np.polyfit(np.arange(800)*.01,x,1)[0],np.mean(x[:400])-np.mean(x[400:]),q[0]-q[1],q[2]-q[3],np.std(slow),np.std(res),spec(res)[0]]
                hb.extend(a+b);shape.extend([np.mean(x[k:k+50])-np.mean(x) for k in range(0,800,50)]+[np.mean(x),np.std(x),np.std(res)])
        close("A independent spot "+str(idx),base,X["FUS_A"][idx]);close("B independent spot "+str(idx),hb,X["HF_B"][idx]);close("shape independent spot "+str(idx),shape,X["HF_SHAPE"][idx]);spots.append(int(idx))
# Exact reused raw windows: first-round A features should agree.
v1=R/"data/experiments/hf_scene_classification_v1/p1_20261005_01";xm=np.load(v1/"features.npz")["X"];mm=pd.read_csv(v1/"window_metadata.csv")
matches=0
for rr in r.fillna("").itertuples():
    if rr.overlap_v1_records:
        aa=m[m.record==rr.record].set_index("center_s");bb=mm[mm.record==rr.overlap_v1_records].set_index("center_s")
        for center in aa.index:
            i=int(m.index[(m.record==rr.record)&(m.center_s==center)][0]);j=int(mm.index[(mm.record==rr.overlap_v1_records)&(mm.center_s==center)][0])
            close("first-round A identity "+str(i),X["FUS_A"][i],xm[j]);matches+=1
com=pd.read_csv(O/"complementarity.csv")
for _,row in com.iterrows():
    lhs,rhs=row.comparison.split(" minus ");a=p[p.configuration==rhs].sort_values("original_row");b=p[p.configuration==lhs].sort_values("original_row")
    mask=np.ones(len(a),bool) if row.activity=="ALL" else (a.activity==row.activity).to_numpy()
    acorr=(a.true_index==a.pred_index).to_numpy()[mask];bcorr=(b.true_index==b.pred_index).to_numpy()[mask];ww=a.eval_weight.to_numpy()[mask]
    close(row.comparison+row.activity+" rescue damage counts",[sum(~acorr&bcorr),sum(acorr&~bcorr),sum(acorr),sum(~acorr)],[row.rescued,row.damaged,row.baseline_correct,row.baseline_wrong])
    close(row.comparison+row.activity+" weighted rescue damage ceiling",[ww[~acorr&bcorr].sum()/ww.sum(),ww[acorr&~bcorr].sum()/ww.sum(),ww[~acorr].sum()/ww.sum()],[row.weighted_rescued,row.weighted_damaged,row.baseline_error_ceiling])
labels=pd.read_csv(O/"exploration_record_labels.csv");cluster=pd.read_csv(O/"cluster_assignments.csv");cm=pd.read_csv(O/"clustering_metrics.csv");med=np.load(O/"record_medians.npz");emb=np.load(O/"record_embeddings.npz")
spaces=dict(A_MIMU="MIMU",A_HF="HF_A",A_FUS="FUS_A",B_MIMU="MIMU",B_HF="HF_B",B_FUS="FUS_B")
for space,arm in spaces.items():
    expected=np.array([np.median(X[arm][(m.record==rid).to_numpy()],axis=0) for rid in labels.record]);close(space+" independent record medians",expected,med[space])
    cc=cluster[cluster.space==space].set_index("record").loc[labels.record,"cluster"];row=cm[cm.space==space].iloc[0]
    close(space+" ARI AMI",[adjusted_rand_score(labels.activity,cc),adjusted_mutual_info_score(labels.activity,cc)],[row.activity_ARI,row.activity_AMI])
    for key in ["PCA","tSNE_p5","tSNE_p10"]:ok(space+key+" finite24x2",emb[space+"__"+key].shape==(24,2) and np.isfinite(emb[space+"__"+key]).all())
ok("date labels retain unknown/conflicting",labels.set_index("record").loc["HG-03","plot_date"].startswith("Unknown") and labels.set_index("record").loc["RS-02","plot_date"].startswith("Conflicting"))
v1man=json.loads((v1/"final_delivery_manifest.json").read_text());ok("first round 81 files unchanged",all(sha(v1/z["path"])==z["sha256"] for z in v1man["files"]))
ok("Git state unchanged",subprocess.check_output(["git","status","--short"],cwd=R,text=True,encoding="utf-8").strip()==fr["git"]["status"])
out=dict(status="PASS",checks=len(checks),check_names=checks,max_absolute_error=max(errors),feature_spot_windows=len(spots),spot_indices=spots,round1_matching_feature_windows=matches,models_refit=0,source_sha256=sha(__file__),limits=["Actual fit provenance audited through hashed executed code/weight tables, not persisted model objects.","SVC ovr decision values are not probabilities and predict can differ from their argmax in voting ties.","Record-level unweighted majority and weighted pooled window metrics are distinct."])
(O/"independent_verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k not in ["check_names","spot_indices"]},ensure_ascii=False))
print("PRIMARY",json.loads((O/"analysis_complete.json").read_text()))
print("COMPLEMENT\n",com[com.activity=="ALL"].to_string(index=False))
print("CLUSTERS\n",cm.to_string(index=False))

