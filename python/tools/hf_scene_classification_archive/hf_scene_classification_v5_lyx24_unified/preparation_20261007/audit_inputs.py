"""Preparation only: raw integrity, missingness and descriptive signal statistics; no model imports."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;V=R/'hf_scene_classification_v3_lyx24'
rec=pd.read_csv(V/'records_LOCAL.csv');meta=pd.read_csv(V/'windows.csv');cfg=json.loads((V/'config.json').read_text());ch=cfg['channels'];split=json.loads((V/'splits.json').read_text());train16=set(split['aux']['outer_train'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fill(z):
 a=z.copy()
 for j in range(a.shape[1]):
  good=np.isfinite(a[:,j]);assert good.any();a[:,j]=np.interp(np.arange(len(a)),np.flatnonzero(good),a[good,j])
 return a
rows=[];spectral=[];initial=[];freq=np.fft.rfftfreq(800,.01);band=(freq>=.5)&(freq<=50)
for rr in rec.itertuples():
 assert sha(rr.input_path)==rr.input_sha256
 d=pd.read_csv(rr.input_path);time=pd.to_numeric(d['Time(s)']).to_numpy();a=d[ch].apply(pd.to_numeric,errors='coerce').to_numpy(float);bad=~np.isfinite(a);g=meta[meta.record==rr.record]
 assert np.allclose(np.diff(time),.01,atol=1e-7,rtol=0);assert ((d.SampleIndex.diff().dropna())==1).all()
 expected_bad=bad.any(1);vf=pd.to_numeric(d.ValidFlag).to_numpy();it=pd.to_numeric(d.InterpFlag).to_numpy()
 union=np.zeros(len(d),bool);complete=0
 for i,w in g.iterrows():
  z=a[int(w.start_row):int(w.end_row_exclusive)];b=~np.isfinite(z);assert int(b.any(1).sum())==w.missing_rows;union[int(w.start_row):int(w.end_row_exclusive)]=True
  if b.any():continue
  complete+=1
  if rr.record not in train16:continue
  power=abs(np.fft.rfft((z-z.mean(0))*np.hanning(800)[:,None],axis=0))**2
  for channel in range(8):
   p=power[:,channel];den=p[band].sum();total=p[1:].sum()
   spectral.append(dict(record=rr.record,activity=rr.activity,original_row=int(i),channel=ch[channel],power_over10_frac=float(p[freq>10].sum()/den) if den else 0.,power_over20_frac=float(p[freq>20].sum()/den) if den else 0.,power_05to5_frac=float(p[(freq>=.5)&(freq<=5)].sum()/den) if den else 0.,power_below05_frac=float(p[(freq>0)&(freq<.5)].sum()/total) if total else 0.,peak_05to20_Hz=float(freq[(freq>=.5)&(freq<=20)][np.argmax(p[(freq>=.5)&(freq<=20)])])))
 first=(time>=0)&(time<30);z=fill(a[first]);tf=time[first];tc=tf-tf.mean();slope=(tc[:,None]*(z-z.mean(0))).sum(0)/(tc@tc)
 accdev=np.sqrt(np.mean(np.sum((z[:,:3]-z[:,:3].mean(0))**2,axis=1)));gyro=np.sqrt(np.mean(np.sum(z[:,3:6]**2,axis=1)))
 for j in range(8):initial.append(dict(record=rr.record,activity=rr.activity,channel=ch[j],initial30_mean=float(z[:,j].mean()),initial30_sd=float(z[:,j].std()),initial30_slope_per_s=float(slope[j]),first10mean=float(z[:1000,j].mean()),last10mean=float(z[-1000:,j].mean())))
 rows.append(dict(record=rr.record,activity=rr.activity,rows=len(d),raw_duration_s=float(time[-1]-time[0]),fs=100,time_grid_ok=True,all_eight_missing_together=bool(np.all(bad==bad[:,0,None])),raw_missing_rows=int(expected_bad.sum()),finite_but_invalid=int(((vf<=0)&~expected_bad).sum()),finite_but_interp=int(((it>0)&~expected_bad).sum()),motion_windows=len(g),complete_motion_windows=complete,motion_missing_windows=int((g.missing_rows>0).sum()),max_window_missing_rows=int(g.missing_rows.max()),max_window_gap_samples=int(g.max_gap_samples.max()),motion_union_rows=int(union.sum()),motion_union_missing_rows=int((union&expected_bad).sum()),first_motion_start_s=float(time[int(g.start_row.min())]),initial30_missing_rows=int(expected_bad[first].sum()),initial30_acc_dynamic_RMS_g=float(accdev),initial30_gyro_RMS_dps=float(gyro),source_sha256=rr.input_sha256,spectral_audit_train16=rr.record in train16,rest_verified=False))
pd.DataFrame(rows).to_csv(O/'record_audit.csv',index=False);pd.DataFrame(initial).to_csv(O/'initial30_descriptive.csv',index=False)
spec=pd.DataFrame(spectral);med=spec.groupby(['record','activity','channel']).median(numeric_only=True).reset_index();med.to_csv(O/'training16_complete_window_spectrum.csv',index=False)
summary=[]
for channel,g in med.groupby('channel'):
 summary.append(dict(channel=channel,records=len(g),over10_median=float(g.power_over10_frac.median()),over10_max=float(g.power_over10_frac.max()),over20_median=float(g.power_over20_frac.median()),below05_median=float(g.power_below05_frac.median()),peak_median_Hz=float(g.peak_05to20_Hz.median())))
rawdf=pd.DataFrame(rows);ini=pd.DataFrame(initial)
# Estimate upper training effort from actual past seconds per executed epoch, without GPU work.
audit=pd.read_csv(V/'fit_audit.csv');a=audit[audit.scope=='main'].copy();a['executed_epochs']=a.inner_epochs+a.final_epochs
cost=a.groupby(a.configuration.str.startswith('CNNLSTM')).apply(lambda g:float(g.seconds.sum()/g.executed_epochs.sum()),include_groups=False).to_dict()
summarydict=dict(status='PREPARATION_ONLY_NO_FITS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=24,classes=8,windows=len(meta),missing_windows=int(rawdf.motion_missing_windows.sum()),missing_records=int((rawdf.motion_missing_windows>0).sum()),max_missing_per_window=int(rawdf.max_window_missing_rows.max()),max_gap_samples=int(rawdf.max_window_gap_samples.max()),motion_union_rows=int(rawdf.motion_union_rows.sum()),motion_union_missing_rows=int(rawdf.motion_union_missing_rows.sum()),all_raw_hashes_match=True,finite_but_invalid=int(rawdf.finite_but_invalid.sum()),finite_but_interp=int(rawdf.finite_but_interp.sum()),initial30_acc_dynamic_RMS_g_range=[float(rawdf.initial30_acc_dynamic_RMS_g.min()),float(rawdf.initial30_acc_dynamic_RMS_g.max())],initial30_gyro_RMS_dps_range=[float(rawdf.initial30_gyro_RMS_dps.min()),float(rawdf.initial30_gyro_RMS_dps.max())],spectral_subset='Existing fixed auxiliary16 training records, complete frozen windows only. No held8 data or predictions used for spectrum design audit. Historical development data; not pristine validation.',spectral_complete_windows=int(spec.original_row.nunique()),spectrum_record_median_summary=summary,initial30_not_rest_evidence=True,seconds_per_epoch_CNN=cost[False],seconds_per_epoch_CNNLSTM=cost[True],historical_main_training_seconds=float(audit[audit.scope=='main'].seconds.sum()),expected_max_main_neural_minutes=float((cost[False]+cost[True])*216*80/60),source_files={str(p):sha(p) for p in [V/'records_LOCAL.csv',V/'windows.csv',V/'config.json',V/'splits.json',V/'fit_audit.csv']})
(O/'audit_summary.json').write_text(json.dumps(summarydict,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summarydict,ensure_ascii=False));print('INITIAL_HF_SLOPE',ini[ini.channel.isin(ch[6:])].groupby('channel').initial30_slope_per_s.agg(['min','max']).to_dict());print(rawdf[['record','first_motion_start_s','initial30_acc_dynamic_RMS_g','initial30_gyro_RMS_dps']].to_string(index=False))
