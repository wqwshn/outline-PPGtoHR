"""Preparation only: hashes, metadata and frozen motion membership. No model imports."""
from pathlib import Path
import json,hashlib,subprocess,sys
import pandas as pd,numpy as np
R=Path(r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR")
O=Path(__file__).resolve().parent
REL=R/"data/experiments/paper_release_20260906"; AUD=R/"data/experiments/paper_fft_acc_audit_20260907"
P0=R/"data/experiments/hf_scene_classification_v1/p0_20261005_02"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jw(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")
rows=pd.read_csv(REL/"lyx24/results.csv");old=pd.read_csv(P0/"record_bindings_LOCAL.csv")
manifest=json.loads((REL/"artifact_manifest.json").read_text())["files"];mf={r["path"].replace("\\","/"):r for r in manifest}
assert sha(REL/"lyx24/results.csv")==mf["lyx24/results.csv"]["sha256"]
assert len(rows)==24 and rows.groupby("scene").size().eq(3).all()
act=dict(xiezi="HW",tiaosheng="RS",woli="HG",jianpan="TYP",run="RUN",kaihe="JJ",bobi="BUR",quanji="PCH")
chn=["AccX(g)","AccY(g)","AccZ(g)","GyroX(dps)","GyroY(dps)","GyroZ(dps)","Ut1(mV)","Ut2(mV)"]
wm=pd.read_csv(AUD/"window_route_metrics.csv");wm=wm[wm.cohort=="lyx24"]
records=[];windows=[];issues=[];inputhash=[];freezechecks=[]
def gap(mask):
    z=np.r_[False,mask,False].astype(int);return int(max(np.where(np.diff(z)==-1)[0]-np.where(np.diff(z)==1)[0],default=0))
for scene,g in rows.groupby("scene"):
  for ordinal,rid in enumerate(sorted(g.record_id),1):
    label=f"{act[scene]}-{ordinal:02d}";p=REL/"raw"/(rid+".csv");ref=REL/"raw"/(rid+"_HR_ref.csv")
    digest=sha(p);assert digest==mf["raw/"+rid+".csv"]["sha256"];assert sha(ref)==mf["raw/"+rid+"_HR_ref.csv"]["sha256"]
    raw=pd.read_csv(p);data=raw[chn].apply(pd.to_numeric,errors="coerce").to_numpy();bad=~np.isfinite(data)
    grid=np.allclose(raw["Time(s)"],np.arange(len(raw))/100,atol=1e-8,rtol=0)
    assert grid and len(set(raw.columns))==len(raw.columns)
    hr=pd.read_csv(ref);ts=pd.to_datetime(hr.timestamp_local,errors="coerce")
    dates=sorted(set(ts.dropna().dt.strftime("%Y-%m-%d")));date=dates[0] if len(dates)==1 else ";".join(dates)
    tr=json.loads((REL/"lyx24/traces"/(rid+"_hf.json")).read_text())
    assert sha(REL/"lyx24/traces"/(rid+"_hf.json"))==mf["lyx24/traces/"+rid+"_hf.json"]["sha256"]
    tw=pd.DataFrame(tr["window_table"]);motion=tw[tw.is_motion.astype(bool)]
    support=set((int(v.window_idx),float(v.center_s)) for v in motion.itertuples())
    for route in ["HF","ACC","FFT"]:
      q=wm[(wm.record_id==rid)&(wm.route_id==route)&wm.is_motion]
      actual=set(zip(q.window_idx.astype(int),q.center_s.astype(float)))
      freezechecks.append(dict(record=label,route=route,motion_equal=actual==support,windows=len(actual)))
      if actual!=support:issues.append(dict(record=label,issue="motion membership differs "+route))
    assert tr["motion_detection"]["source"]=="raw_imu_acc_gyro"
    union=np.zeros(len(raw),bool);local=[]
    for row in motion.itertuples():
      start=round((row.center_s-4)*100);end=round((row.center_s+4)*100);assert 0<=start<end<=len(raw) and end-start==800
      b=bad[start:end];bm=b.any(axis=1);union[start:end]=True
      d=dict(record=label,subject="subject-1",activity=act[scene],window_idx=int(row.window_idx),center_s=float(row.center_s),start_row=start,end_row_exclusive=end,sample_count=800,missing_rows=int(bm.sum()),missing_values=int(b.sum()),max_gap_samples=gap(bm),all_missing_channels=int(b.all(axis=0).sum()),invalid_flag_samples=int((raw.iloc[start:end].ValidFlag==0).sum()),interp_flag_samples=int((raw.iloc[start:end].InterpFlag!=0).sum()))
      local.append(d);windows.append(d)
    ov=old[old.input_sha256==digest]
    desc=dict(record=label,subject="subject-1",activity=act[scene],scene=scene,algorithm_record_id=rid,date_evidence=date,date_token=rid.rsplit("_",1)[-1],date_token_matches=len(dates)==1 and dates[0][5:].replace("-","")==rid.rsplit("_",1)[-1],ref_first_timestamp=str(ts.min()),ref_last_timestamp=str(ts.max()),raw_rows=len(raw),raw_duration_s=float(raw["Time(s)"].iloc[-1]),channels=len(chn),raw_fs_hz=100,time_grid_ok=bool(grid),motion_windows=len(local),missing_motion_windows=sum(d["missing_rows"]>0 for d in local),max_window_missing_rows=max(d["missing_rows"] for d in local),max_window_gap_samples=max(d["max_gap_samples"] for d in local),motion_union_rows=int(union.sum()),motion_union_missing_rows=int(bad[union].any(axis=1).sum()),raw_missing_rows=int(bad.any(axis=1).sum()),synchronous_missing_all_channels=bool(np.array_equal(bad.any(axis=1),bad.all(axis=1))),overlap_v1_records=";".join(ov.record),overlap_v1_subjects=";".join(ov.subject),input_path=str(p),input_sha256=digest,wear_session_evidence="author-confirmed record-level rewear; no shared cross-activity wear batch ID",trace_source=tr["motion_detection"]["source"])
    records.append(desc);inputhash.append(dict(path=str(p),sha256=digest))
rec=pd.DataFrame(records);win=pd.DataFrame(windows);f=pd.DataFrame(freezechecks)
rec.to_csv(O/"record_manifest_LOCAL.csv",index=False);win.to_csv(O/"frozen_motion_windows.csv",index=False);f.to_csv(O/"motion_membership_checks.csv",index=False)
counts=pd.crosstab(rec.activity,rec.date_evidence).reindex(list(act.values())).fillna(0).astype(int);counts.to_csv(O/"activity_by_acquisition_date.csv")
coverage=[]
for d in counts.columns:
    held=rec[rec.date_evidence==d];train=rec[rec.date_evidence!=d]
    coverage.append(dict(date=d,test_records=len(held),test_activities=";".join(sorted(set(held.activity))),train_missing_activities=";".join(sorted(set(rec.activity)-set(train.activity)))))
pd.DataFrame(coverage).to_csv(O/"leave_date_out_coverage.csv",index=False)
publiccols=["record","subject","activity","date_evidence","motion_windows","missing_motion_windows","max_window_missing_rows","max_window_gap_samples","overlap_v1_records"]
rec[publiccols].to_csv(O/"record_summary.csv",index=False)
# Candidate record-level three folds, never label ordinal as an actual common wear batch.
split=[]
for _,r in rec.iterrows():
  hold=int(r.record.rsplit("-",1)[1])
  for fold in [1,2,3]:split.append(dict(candidate_fold=fold,record=r.record,activity=r.activity,role="test" if hold==fold else "train",grouping_basis="anonymous sorted-record ordinal; NOT a shared wear batch",status="proposal_only_not_executed"))
pd.DataFrame(split).to_csv(O/"candidate_record_folds_NOT_BATCHES.csv",index=False)
git={k:subprocess.check_output(["git",*v],cwd=R,text=True,encoding="utf-8").strip() for k,v in dict(root=["rev-parse","--show-toplevel"],HEAD=["rev-parse","HEAD"],branch=["branch","--show-current"],worktrees=["worktree","list","--porcelain"],status=["status","--short"],staged=["diff","--cached","--name-status"]).items()}
jw("git_state.json",git)
v1=R/"data/experiments/hf_scene_classification_v1/p1_20261005_01";man=json.loads((v1/"final_delivery_manifest.json").read_text());v1_ok=all(sha(v1/r["path"])==r["sha256"] for r in man["files"])
snapshot=O/"source_snapshots";snapshot.mkdir()
for name in ["audit_hf_scene_classification_p0.py","run_hf_scene_classification_v1.py","plot_hf_scene_classification_v1.py"]:
    (snapshot/name).write_bytes((R/"python/tools"/name).read_bytes())
facts=dict(records=len(rec),subjects=1,activities=8,motion_windows=len(win),missing_motion_windows=int((win.missing_rows>0).sum()),records_with_missing_motion=int((rec.missing_motion_windows>0).sum()),max_missing_samples=int(win.missing_rows.max()),max_gap_samples=int(win.max_gap_samples.max()),all_missing_window_channels=int(win.all_missing_channels.sum()),motion_union_rows=int(rec.motion_union_rows.sum()),motion_union_missing_rows=int(rec.motion_union_missing_rows.sum()),synchronous_missing_all_records=bool(rec.synchronous_missing_all_channels.all()),overlap_v1_records=int(rec.overlap_v1_records.ne("").sum()),overlap_v1_subjects=sorted(set(rec.overlap_v1_subjects)-{""}),motion_routes_all_match=bool(f.motion_equal.all()),date_tokens_match=bool(rec.date_token_matches.all()),date_groups=len(counts.columns),first_round_all_81_files_unchanged=v1_ok,models_executed=0,features_extracted=False,imputation_executed=False,issues=issues,python=sys.version)
jw("preparation_receipt.json",facts)
jw("input_fingerprints_LOCAL.json",inputhash)
def tab(df):return "| "+" | ".join(map(str,df.columns))+" |\n| "+" | ".join(["---"]*len(df.columns))+" |\n"+"\n".join("| "+" | ".join(str(x) for x in t)+" |" for t in df.itertuples(index=False,name=None))
report=f"""# 第二轮HF场景识别：LYX24数据与执行准备

本阶段仅核验数据和执行边界，模型调用0次，没有提取分类特征、填补原始数据或启动模型比较；等待两篇论文方法摘要。第一轮81项交付文件哈希核验：{v1_ok}。

## 最重要的结论
LYX24确为subject-1的24条记录、8类各3条。它是经过后验筛选的心率开发面板，不能当新独立验证集。本轮可用于方法探索。
佩戴依据来自论文Research State.md第3节及采集事实：作者2026-09-07确认该人的记录间视为摘下重戴；没有统一腕带松紧/压力干预。这支持整记录佩戴会话留出，但没有三个跨场景共同批次的证据。Fig.3匿名编号按冻结record_id排序，不是采集时间顺序。

与第一轮119条的原始文件SHA-256交集为{facts['overlap_v1_records']}条，同属{facts['overlap_v1_subjects']}。剩余记录也来自同一人，不能称独立新受试者验证。名单、窗口、缺失和哈希均已保存。

## 采集日期与场景分布
日期读取对应HR_ref.csv的timestamp_local，并与记录日期token核对；这是参考设备采集日期证据，不是传感器硬件绝对同步证明，也不是受控佩戴批次ID。
{tab(counts.reset_index())}

日期留出覆盖：
{tab(pd.DataFrame(coverage))}
如果某日期留出导致训练缺类，不能用它报告完整八类跨日期能力。日期与类别明显耦合，禁止把日期分类差异解释成佩戴泛化。

## 冻结运动窗口与缺失
只采用冻结HF轨迹window_table.is_motion，来源raw_imu_acc_gyro；同时与现有HF/独立ACC/FFT三路线逐窗表比对，{len(f)}项记录—路线检查一致={facts['motion_routes_all_match']}。不引用reliable、used_adaptive、HR误差或1713共同可靠运动窗进行筛选。
100Hz原始网格，每窗[c−4,c+4)秒/800行，步长1秒；输入为六轴Acc/Gyro与Ut1/Ut2，单位g/dps/mV。HR算法内部25/50/100Hz参数不改变本次原始100Hz输入定义。
全体运动窗{len(win)}，含缺失{facts['missing_motion_windows']}，涉及{facts['records_with_missing_motion']}条；最大单窗缺{facts['max_missing_samples']}/800点，最长连续{facts['max_gap_samples']}点；全缺失通道窗口数{facts['all_missing_window_channels']}。
运动支撑去重后{facts['motion_union_rows']}行，其中{facts['motion_union_missing_rows']}行缺失；同步八通道缺失检查={facts['synchronous_missing_all_records']}。本阶段不执行插值，后续保留缺失审计并在方法冻结时明确处理。

{tab(rec[publiccols])}

## 推荐拆分协议（尚未执行）
优先把原始完整记录作为最小分组。若依作者确认把一条记录视作一次重新佩戴会话，则整记录留出同时隔离该会话；不可臆造跨场景session-1/2/3。
可选A：24折leave-one-record-out，每折只测试整条记录；每折测试仅一个类别，不能计算每折八类macro-F1，须汇总全24条OOF预测、活动/记录等权计算，并展示记录级结果。
可选B：预先冻结三折，每折每类留1条，训练16条/测试8条；现有candidate_record_folds_NOT_BATCHES.csv仅给出排序序号轮换的候选，不声称真实共同佩戴批次。每类3条而无法验证采集批次共享关系是限制。
A/B须在看分类结果前确定；本阶段不选模型、不比较拆分优劣。任何标准化、特征选择、降维及参数拟合只用训练记录；相邻或重叠窗口绝不随机跨训练/测试。探索可单独用全体数据可视化，但变换不能流入评估。
第一轮119含重叠记录，同一人的未交集记录也非跨人独立确认。若第二轮反复开发后回到第一轮，需明确其为已见数据复用，不包装为盲测。

## 来源层级与仍未知事项
权威成员：算法paper_release_20260906/lyx24/results.csv及artifact_manifest.json；与论文Fig.3制图脚本prepare_hr_subject1.py排序匿名规则一致。
作者佩戴事实：D:/Thesis/AegisPulse/Research State.md约454–458、611–613行。后验开发边界：docs/adr/0052-close-lyx-cross-wear-development-stage-on-generalization-branch.md。
HF在项目docs/paper/algorithm-and-evaluation.md被称为双侧热式界面参考信号；论文术语为thermal interface sensing。仍不把电压直接换算为压力，也不在文献摘要前决定慢信号表征。
缺少共同跨场景佩戴批次ID、受控松紧等级/压力、独立新参与者。日期字段来自参考HR设备；未用其替代原始传感器时间网格。标签与采样均为现有资料证据，不代表现场重新确认。

## 工作区与隔离
现有主仓库{git['root']}，分支{git['branch']}，HEAD={git['HEAD']}。暂存区为空={not bool(git['staged'])}。没有创建worktree，没有add/commit/push，没有改动第一轮文件或用户GUI代码。
已存在detached worktree仅从git worktree list核对，没有读取其内容。当前新增内容仅本独立准备目录。
已将第一轮三个未跟踪audit/run/plot脚本按字节复制到source_snapshots，保留原路径文件；正式建隔离工作树前仍须记录完整工作区基线。git worktree从HEAD新建不会自动包含这些未提交脚本，不能误认为历史代码齐全。
建议后续从固定HEAD新建任务专用worktree，显式复制经哈希核对的必要未提交脚本；主仓库冻结数据只读引用，新产物继续独立目录。绝不stash/clean/reset用户已有改动。共享ppg-hr editable安装可能指向其他工作树，运行入口需显式选代码路径，不能为切换实验随意重装。建立工作树及后续方法实施在下一阶段完成，此次没有执行。

## 交付与下一步
record_manifest_LOCAL.csv含内部ID和绝对路径，仅本地溯源；record_summary.csv为匿名摘要。frozen_motion_windows.csv为全部候选成员，未按结果筛选。activity_by_acquisition_date.csv、leave_date_out_coverage.csv、motion_membership_checks.csv及preparation_receipt.json提供可核查证据。
等待文献方法摘要后，先冻结预处理/特征/模型预算、选择记录分组评估协议，再实施；不沿用旧心率MAE/可靠掩码作分类标签。分类改善与心率改善分开论证。
"""
(O/"PREPARATION_REPORT.md").write_text(report,encoding="utf-8")
jw("preparation_manifest.json",[dict(path=str(p.relative_to(O)).replace("\\","/"),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(O.rglob("*")) if p.is_file() and p.name!="preparation_manifest.json"])
print(json.dumps(facts,ensure_ascii=False));print(counts.to_string());print(pd.DataFrame(coverage).to_string(index=False));print(rec[publiccols].to_string(index=False))

