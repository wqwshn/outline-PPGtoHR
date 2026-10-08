"""Independent completed-run verification. Never imports the training runner, never fits classifiers."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,hashlib,time,datetime
import numpy as np,pandas as pd,torch,joblib
from torch import nn
from scipy.signal import butter,sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;ACT=['HW','RS','HG','TYP','RUN','JJ','BUR','PCH'];SEEDS=[20261006,20261007,20261008];ARMS={'MIMU':list(range(6)),'HF':[6,7],'FUS':list(range(8))}
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def weights(meta,ids):
 g=meta.iloc[ids];nw=g.groupby('record').size();nr=g.groupby('activity').record.nunique()
 return np.array([1/(8*nr[r.activity]*nw[r.record]) for r in g.itertuples()])
class CheckNet(nn.Module):
 def __init__(self,c,arch):
  super().__init__();self.arch=arch;self.conv=nn.Sequential(nn.Conv1d(c,16,9,1,4),nn.AvgPool1d(2,2),nn.ReLU())
  if arch=='CNNLSTM':self.lstm=nn.LSTM(16,32,1,batch_first=True)
  self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(32 if arch=='CNNLSTM' else 16,8))
 def encode(self,x):
  z=self.conv(x)
  if self.arch=='CNNLSTM':return self.lstm(z.transpose(1,2))[1][0][-1]
  return z.mean(2)
 def forward(self,x):return self.head(self.encode(x))
def feats(x):
 ans=np.empty((len(x),8,7));freq=np.fft.rfftfreq(800,.01);band=(freq>=.5)&(freq<=20);low=(freq>=.5)&(freq<=5)
 for i,xx in enumerate(x):
  for j,raw in enumerate(xx):
   z=raw.astype(np.float64);q=abs(np.fft.rfft((z-z.mean())*np.hanning(800)))**2;b=q[band];tot=b.sum()
   if tot>0:
    probs=b/tot;positive=probs>0;sp=[freq[band][b.argmax()],-(probs[positive]*np.log(probs[positive])).sum()/np.log(len(b)),q[low].sum()/tot]
   else:sp=[0,0,0]
   ans[i,j]=[0 if j>=6 else z.mean(),z.std(),np.ptp(z),abs(np.diff(z)).mean(),*sp]
 return ans

def main():
 started=time.perf_counter();freeze=json.loads((O/'freeze_receipt.json').read_text());cfg=json.loads((O/'config.json').read_text());sp=json.loads((O/'splits.json').read_text());meta=pd.read_csv(O/'windows.csv');rec=pd.read_csv(O/'records_LOCAL.csv')
 assert json.loads((O/'run_status.json').read_text())['status']=='training_complete'
 assert json.loads((O/'classic_status.json').read_text())['completed']==288
 assert all(sha(O/k)==v for k,v in freeze['artifacts'].items())
 x=np.load(O/'windows_physical_float64.npy');assert x.shape==(1941,8,800)
 rebuilt=np.empty_like(x);sos=butter(4,20,fs=100,btype='lowpass',output='sos')
 for rr in rec.itertuples():
  assert sha(rr.input_path)==rr.input_sha256
  raw=pd.read_csv(rr.input_path)[cfg['channels']].to_numpy(float)
  for i,row in meta[meta.record==rr.record].iterrows():
   z=raw[int(row.start_row):int(row.end_row_exclusive)].T.copy()
   for c in range(8):
    finite=np.isfinite(z[c]);assert finite.any();z[c]=np.interp(np.arange(800),np.flatnonzero(finite),z[c,finite])
   z=sosfiltfilt(sos,z,axis=1,padtype='odd',padlen=100);z[6:]-=z[6:].mean(axis=1,keepdims=True);rebuilt[i]=z
 assert np.array_equal(rebuilt,x);del rebuilt
 moment=np.mean(x*x,axis=2);scale_cache={}
 def checkscale(sc,records):
  ids=np.flatnonzero(meta.record.isin(records));assert np.array_equal(ids,sc['indices']) and sc['records']==sorted(records)
  key=tuple(ids)
  if key not in scale_cache:
   w=weights(meta,ids);s=moment[ids].T@w;scale=np.ones(8)
   for cols in [list(range(3)),list(range(3,6)),[6,7]]:
    v=np.sqrt(s[cols].mean());scale[cols]=1 if v<=1e-12 else v
   scale_cache[key]=(scale,w)
  scale,w=scale_cache[key]
  assert np.allclose(scale,sc['scale'],rtol=2e-14,atol=1e-14) and np.array_equal(sc['mean'],np.zeros(8)) and np.allclose(sc['weights'],w,rtol=0,atol=1e-15)
  assert sc['scale'][6]==sc['scale'][7]
 def inp(ids,cols,sc):return np.ascontiguousarray((x[ids][:,cols]/sc['scale'][cols][None,:,None]).astype(np.float32))
 splitmap={s['fold']:s for s in sp['main']};splitmap['AUX16_8']=sp['aux'];audit=[];allpred=[];maxlogit=0.;maxlatent=0.
 dirs=sorted((O/'jobs').glob('*'));assert len(dirs)==438
 with torch.no_grad():
  for k,d in enumerate(dirs):
   done=json.loads((d/'done.json').read_text());assert all(sha(d/n)==h for n,h in done['files'].items());ident=done['identity'];assert ident['freeze_sha256']==sha(O/'freeze_receipt.json')
   st=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);assert st['identity']==ident and st['phase']=='done';split=splitmap[ident['fold']]
   for phase,rr in [('inner',split['inner_train']),('final',split['outer_train'])]:checkscale(st['scalers'][phase],rr)
   assert not set(split['inner_train'])&set(split['inner_val']) and not set(split['outer_train'])&set(split['test'])
   hist=st['inner_history'];chosen=min(hist,key=lambda z:z['validation_CE'])['epoch'];assert chosen==st['best_epoch']==len(st['final_history'])
   assert st['initial_model_hash']==st['refit_initial_model_hash'] and 15<=len(hist)<=40
   g=pd.read_csv(d/'predictions.csv');ids=g.original_row.to_numpy();expected=np.flatnonzero(meta.record.isin(split['test']));assert np.array_equal(ids,expected)
   model=CheckNet(len(ARMS[ident['arm']]),ident['architecture']).cuda().eval();model.load_state_dict(st['model']);assert sum(v.numel() for v in model.parameters())==done['parameters']
   z=inp(ids,ARMS[ident['arm']],st['scalers']['final']);logits=[];lat=[]
   for a in range(0,len(z),64):
    q=model.encode(torch.from_numpy(z[a:a+64]).cuda());lat.append(q.cpu().numpy());logits.append(model.head(q).cpu().numpy())
   scores=np.concatenate(logits);hidden=np.concatenate(lat);saved=g[['logit_'+a for a in ACT]].to_numpy();err=float(abs(scores-saved).max());maxlogit=max(maxlogit,err)
   assert np.allclose(scores,saved,rtol=2e-5,atol=2e-5) and np.array_equal(scores.argmax(1),g.pred_index)
   if ident['scope']=='aux':
    data=np.load(d/'latents.npz');assert np.array_equal(data['test_indices'],ids) and np.array_equal(data['selected_indices'],sp['aux']['point_indices']);delta=float(abs(hidden-data['test_latent']).max());maxlatent=max(maxlatent,delta);assert delta<2e-5
    trainids=np.flatnonzero(meta.record.isin(split['outer_train']));assert np.array_equal(trainids,data['train_indices']);trainx=inp(trainids,ARMS[ident['arm']],st['scalers']['final']);trainlat=[]
    for a in range(0,len(trainx),64):trainlat.append(model.encode(torch.from_numpy(trainx[a:a+64]).cuda()).cpu().numpy())
    delta=float(abs(np.concatenate(trainlat)-data['train_latent']).max());maxlatent=max(maxlatent,delta);assert delta<2e-5
   else:allpred.append(g)
   audit.append(dict(job=d.name,kind='neural',logit_max_abs_error=err,selected_epoch=chosen,inner_epochs=len(hist),final_epochs=len(st['final_history']),parameters=done['parameters'],elapsed_seconds=done['elapsed_seconds']))
   del model,st
   if (k+1)%60==0:print('CHECK neural',k+1,'/438',flush=True)
 classicdirs=sorted((O/'classic_jobs').glob('*'));assert len(classicdirs)==288
 feature_cache={}
 with threadpool_limits(limits=1):
  for k,d in enumerate(classicdirs):
   done=json.loads((d/'done.json').read_text());assert all(sha(d/n)==h for n,h in done['files'].items());ident=done['identity'];assert ident['freeze_sha256']==sha(O/'freeze_receipt.json')
   data=joblib.load(d/'model.joblib');assert data['identity']==ident;split=splitmap[ident['fold']];checkscale(data['signal_scaler'],split['outer_train']);g=pd.read_csv(d/'predictions.csv');ids=g.original_row.to_numpy();assert np.array_equal(ids,data['test_indices'])
   ft=feats(inp(ids,list(range(8)),data['signal_scaler']))[:,ARMS[ident['arm']],:].reshape(len(ids),-1)
   if data['feature_scaler'] is not None:
    fold=ident['fold'];trainids=data['train_indices'];assert np.array_equal(trainids,np.flatnonzero(meta.record.isin(split['outer_train'])))
    if fold not in feature_cache:feature_cache[fold]=feats(inp(trainids,list(range(8)),data['signal_scaler']))
    rawfeat=feature_cache[fold][:,ARMS[ident['arm']],:].reshape(len(trainids),-1);w=weights(meta,trainids);mu=w@rawfeat;var=w@((rawfeat-mu)**2);fs=data['feature_scaler']
    assert np.allclose(mu,fs.mean_,rtol=1e-10,atol=1e-12) and np.allclose(var,fs.var_,rtol=1e-9,atol=1e-12)
    ft=fs.transform(ft)
   pred=data['model'].predict(ft);assert np.array_equal(pred,g.pred_index)
   score=data['model'].predict_proba(ft) if ident['architecture']=='ET' else data['model'].decision_function(ft)
   err=float(abs(score-g[['score_'+a for a in ACT]].to_numpy()).max());assert err<1e-8
   allpred.append(g);audit.append(dict(job=d.name,kind='classic',logit_max_abs_error=err,elapsed_seconds=done['seconds']))
   if (k+1)%72==0:print('CHECK classic',k+1,'/288',flush=True)
 p=pd.concat(allpred,ignore_index=True);metrics=[];cms=[];classes=[];records=[];quality=[];true=meta.activity.map(dict(zip(ACT,range(8)))).to_numpy();ew=weights(meta,np.arange(len(meta)))
 for (config,seed),g in p.groupby(['configuration','seed']):
  g=g.sort_values('original_row');ids=g.original_row.to_numpy();assert np.array_equal(ids,np.arange(1941));assert np.array_equal(g.true_index,true);assert np.allclose(g.eval_weight,ew,rtol=0,atol=1e-15)
  cm=np.bincount(true*8+g.pred_index.to_numpy(),weights=ew,minlength=64).reshape(8,8);f=np.divide(2*cm.diagonal(),cm.sum(0)+cm.sum(1),out=np.zeros(8),where=cm.sum(0)+cm.sum(1)>0);recall=cm.diagonal()/cm.sum(1)
  metrics.append(dict(configuration=config,seed=int(seed),macro_F1=float(f.mean()),balanced_accuracy=float(recall.mean()),weighted_accuracy=float(cm.diagonal().sum())))
  for a in range(8):
   classes.append(dict(configuration=config,seed=int(seed),activity=ACT[a],F1=float(f[a]),recall=float(recall[a])))
   for b in range(8):cms.append(dict(configuration=config,seed=int(seed),true_activity=ACT[a],predicted_activity=ACT[b],weight=float(cm[a,b])))
  for rid,gg in g.groupby('record'):records.append(dict(configuration=config,seed=int(seed),record=rid,activity=gg.activity.iloc[0],accuracy=float((gg.pred_index==gg.true_index).mean()),windows=len(gg)))
  missing=meta.missing_rows.to_numpy()/800
  for label,sel in [('0%',missing==0),('(0,10%]',(missing>0)&(missing<=.1)),('(10,25%]',(missing>.1)&(missing<=.25)),('>25%',missing>.25)]:
   present=np.unique(true[sel]);quality.append(dict(configuration=config,seed=int(seed),stratum=label,windows=int(sel.sum()),classes=len(present),weighted_accuracy=float(np.average((g.pred_index.to_numpy()==true)[sel],weights=ew[sel])) if sel.any() else None))
 m=pd.DataFrame(metrics);assert len(m)==30 and m.configuration.nunique()==12
 draft=O.parent.parent/'lyx24_group_meeting_20261007/audit/draft_classic_by_seed.csv'
 if draft.exists():
  previous=pd.read_csv(draft).set_index(['configuration','seed']);current=m.set_index(['configuration','seed']).loc[previous.index];assert np.allclose(previous.macro_F1,current.macro_F1,atol=1e-12,rtol=0)
 summary=m.groupby('configuration').agg(macro_F1_mean=('macro_F1','mean'),macro_F1_std=('macro_F1','std'),balanced_accuracy_mean=('balanced_accuracy','mean'),n_seeds=('seed','size')).reset_index()
 for name,rows in [('summary_by_seed.csv',m),('summary_across_seeds.csv',summary),('confusion_matrices.csv',pd.DataFrame(cms)),('activity_metrics.csv',pd.DataFrame(classes)),('record_metrics.csv',pd.DataFrame(records)),('missingness_strata.csv',pd.DataFrame(quality)),('verification_jobs.csv',pd.DataFrame(audit))]:rows.to_csv(O/name,index=False)
 p.to_csv(O/'all_oof_predictions.csv',index=False)
 # Fixed auxiliary, training-only representation scale and PCA; t-SNE descriptive joint layout of 82 held-out points.
 coords={};diagnostics=[];points=np.array(sp['aux']['point_indices']);assert len(points)==82
 with threadpool_limits(limits=1):
  for d in sorted((O/'jobs').glob('aux*')):
   ident=json.loads((d/'done.json').read_text())['identity'];data=np.load(d/'latents.npz');tr=data['train_latent'].astype(float);te=data['test_latent'].astype(float);w=weights(meta,data['train_indices']);mu=w@tr;sd=np.sqrt(np.maximum(w@(tr*tr)-mu*mu,0));sd[sd==0]=1;train=(tr-mu)/sd;test=(te-mu)/sd
   loc={int(v):i for i,v in enumerate(data['test_indices'])};picked=test[[loc[int(i)] for i in points]];key=ident['architecture']+'_'+ident['arm']
   pc=PCA(n_components=2,svd_solver='full').fit(train);coords[key+'__PCA']=pc.transform(picked)
   for per in [5,15]:
    ts=TSNE(n_components=2,perplexity=per,random_state=20261006,init='pca',learning_rate='auto',max_iter=1000);coords[key+'__tSNE_p'+str(per)]=ts.fit_transform(picked);diagnostics.append(dict(configuration=key,perplexity=per,kl=float(ts.kl_divergence_),latent_dimension=tr.shape[1]))
 np.savez_compressed(O/'aux_embedding_coordinates.npz',**coords);pd.DataFrame(diagnostics).to_csv(O/'embedding_diagnostics.csv',index=False)
 v2=O.parent.parent/'hf_scene_classification_v2_lyx24/execution_20261006_01';assert all(sha(v2/z['path'])==z['sha256'] for z in freeze['v2_baseline']);assert all(sha(row['path'])==row['sha256'] for row in freeze['raw'])
 receipt=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-started,raw_hashes_verified=24,physical_windows_rebuilt_exact=1941,neural_checkpoints_reloaded=438,classic_models_reloaded=288,all_oof_rows=len(p),seed_metrics=30,configurations=12,max_neural_logit_abs_error=maxlogit,max_aux_latent_abs_error=maxlatent,train_only_scale_checks='all inner/final and classic',frozen_artifacts_unchanged=True,old_v2_files_unchanged=True,script_sha256=sha(__file__))
 write(O/'independent_verification.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True);print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
