"""P0 only: bind frozen paper records/windows and audit raw inputs; never train."""
from __future__ import annotations
import argparse, csv, hashlib, json, platform, subprocess, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

CHANNELS = ["AccX(g)","AccY(g)","AccZ(g)","GyroX(dps)","GyroY(dps)","GyroZ(dps)","Ut1(mV)","Ut2(mV)"]
SCENES = dict(xiezi="HW",tiaosheng="RS",woli="HG",jianpan="TYP",run="RUN",kaihe="JJ",bobi="BUR",quanji="PCH")
SUBJECTS = [f"subject-{i}" for i in range(1,7)]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows: path.write_text("",encoding="utf-8"); return
    with path.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def write_json(path,value):
    with path.open("x",encoding="utf-8") as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def git(root,*args):
    return subprocess.check_output(["git","-C",str(root),*args],text=True,encoding="utf-8",errors="replace").strip()

def array_sha(a):
    a=np.asarray(a,dtype="<f8").copy(); a[np.isnan(a)]=np.nan
    return hashlib.sha256(a.tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--thesis",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve(); thesis=args.thesis.resolve(); out=args.output.resolve()
    allowed=(root/"data/experiments/hf_scene_classification_v1").resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("Output must be a new run directory under the task experiment.")
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); started=datetime.now(timezone.utc).isoformat()
    audit=root/"data/experiments/paper_fft_acc_audit_20260907"
    release=root/"data/experiments/paper_release_20260906"
    final=thesis/"figures/figure4/0-final/source_data"
    checks=[]; issues=[]
    def check(name,condition,detail=""):
        checks.append(dict(check=name,status="PASS" if condition else "FAIL",detail=str(detail)))
    def issue(rec,sub,kind,detail):
        issues.append(dict(record=rec,subject=sub,kind=kind,detail=str(detail)))
    routes=[r for r in read_csv(audit/"record_route_metrics.csv") if r["cohort"]=="cross_subject119"]
    hf=[r for r in routes if r["route_id"]=="HF"]
    paper=read_csv(final/"record_MAE.csv")
    si=read_csv(final/"SI_record_MAE_by_subject.csv")
    paper_lookup={(r["record"],r["method"]):r for r in paper}
    alias={}
    for scene in SCENES:
        for i,rid in enumerate(sorted({r["record_id"] for r in hf if r["scene"]==scene}),1):
            alias[rid]=f"{SCENES[scene]}-{i:02d}"
    check("119_unique_records",len(hf)==len(alias)==119)
    check("357_unique_routes",len(routes)==len({(r["record_id"],r["route_id"]) for r in routes})==357)
    check("six_subjects",sorted({r["subject_id"] for r in hf})==SUBJECTS)
    check("activity_counts",Counter(r["scene"] for r in hf)=={k:14 if k=="kaihe" else 15 for k in SCENES})
    mismatch=0
    for r in routes:
        p=paper_lookup.get((alias[r["record_id"]],r["route_id"]))
        if not p or p["subject"]!=r["subject_id"] or p["scene"]!=SCENES[r["scene"]] or abs(float(p["mae_bpm"])-float(r["mae_bpm"]))>1e-10: mismatch+=1
    check("paper_357_value_identity_match",mismatch==0,mismatch)
    si_mismatch=0
    for r in si:
        for arm in ("HF","ACC","FFT"):
            p=paper_lookup.get((r["record"],arm))
            if not p or p["subject"]!=r["subject"] or abs(float(p["mae_bpm"])-float(r[arm]))>1e-10: si_mismatch+=1
    check("paper_SI_119_match",len(si)==119 and si_mismatch==0,si_mismatch)
    # Read the authoritative local mapping, retaining no real-world names in reports.
    state=(thesis/"Research State.md").read_text(encoding="utf-8-sig")
    section=state.split("## 2. 受试者身份与匿名编号",1)[1].split("\n## ",1)[0]
    identities={}
    for line in section.splitlines():
        fields=[s.strip() for s in line.split("|")]
        if len(fields)>3 and fields[1] in SUBJECTS:
            import re
            for token in re.findall(r"\b[A-Z]{2,4}\b",fields[2]): identities[token]=fields[1]
    check("author_identity_mapping",all(identities.get(r["record_id"].split("_")[1])==r["subject_id"] for r in hf))
    release_manifest=json.loads((release/"artifact_manifest.json").read_text(encoding="utf-8"))
    entries={r["path"]:r for r in release_manifest["files"]}
    audit_manifest=json.loads((audit/"artifact_manifest.json").read_text(encoding="utf-8"))
    window_path=audit/"window_route_metrics.csv"
    checks.append(dict(check="global_window_csv_historical_hash",status="NOT_RECORDED",detail="Global CSV absent from historical manifest; verify per-record windows.csv and trace members instead."))
    frozen=defaultdict(list)
    for r in read_csv(window_path):
        if r["cohort"]=="cross_subject119" and r["route_id"] in ("HF","ACC"):
            frozen[(r["record_id"],r["route_id"])].append(r)
    records=[]; windows=[]; quality=[]; duplicate_rows=[]
    file_groups=defaultdict(list); signal_groups=defaultdict(list); block_groups={}
    route_bad=0; mask_bad=0; trace_bad=0; input_bad=0; table_bad=0; configs=set(); other_grid_diff=0
    for r in sorted(hf,key=lambda x:alias[x["record_id"]]):
        rid=r["record_id"]; anon=alias[rid]; sub=r["subject_id"]
        raw=release/"raw"/f"{rid}.csv"; entry=entries.get(f"raw/{rid}.csv",{})
        local_key=f"records/cross_subject119/{rid}/windows.csv"
        if sha(audit/local_key)!=audit_manifest.get(local_key): table_bad+=1
        local_rows=read_csv(audit/local_key)
        for arm in ("HF","ACC"):
            if [x for x in local_rows if x["route_id"]==arm]!=frozen[(rid,arm)]: table_bad+=1
        if not raw.is_file():
            issue(anon,sub,"missing_file","Frozen source missing"); input_bad+=1; continue
        raw_hash=sha(raw)
        if raw_hash!=entry.get("sha256"): input_bad+=1; issue(anon,sub,"input_hash","Archive mismatch")
        file_groups[raw_hash].append((anon,sub))
        arms={}
        for arm in ("HF","ACC"):
            rr=next(x for x in routes if x["record_id"]==rid and x["route_id"]==arm)
            trace_path=audit/rr["trace_path"]; trace_hash=sha(trace_path)
            j=json.loads(trace_path.read_text(encoding="utf-8"))
            if trace_hash!=rr["trace_sha256"] or j["identity"]["data_sha256"]!=raw_hash: trace_bad+=1
            cfg=j["identity"]["config"]
            configs.add(tuple(cfg[k] for k in ("fs_origin","window_seconds","window_step_seconds","calib_time","motion_th_scale")))
            hr=j["hr"]; source_rows=frozen[(rid,arm)]
            if len(hr)!=len(source_rows): route_bad+=1
            for wr in source_rows:
                i=int(wr["window_idx"])
                if i>=len(hr) or abs(float(hr[i][0])-float(wr["center_s"]))>1e-9 or bool(hr[i][4])!=(wr["is_motion"]=="True"): route_bad+=1
            arms[arm]=[wr for wr in source_rows if wr["is_motion"]=="True"]
        if [x["center_s"] for x in arms["HF"]]!=[x["center_s"] for x in arms["ACC"]]: mask_bad+=1
        if [x["center_s"] for x in frozen[(rid,"HF")]]!=[x["center_s"] for x in frozen[(rid,"ACC")]]: other_grid_diff+=1
        df=pd.read_csv(raw); missing=[c for c in CHANNELS if c not in df]
        if missing:
            issue(anon,sub,"missing_channels",missing); continue
        arr=df[CHANNELS].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
        signal_hash=array_sha(arr); signal_groups[signal_hash].append((anon,sub))
        time_values=pd.to_numeric(df["Time(s)"],errors="coerce").to_numpy() if "Time(s)" in df else np.full(len(df),np.nan)
        time_ok=bool(np.isfinite(time_values).all() and np.allclose(time_values,np.arange(len(df))/100,rtol=0,atol=1e-7))
        if not time_ok: issue(anon,sub,"time_axis","Time(s) differs from zero-based 100Hz row grid; no correction applied")
        covered=np.zeros(len(df),dtype=bool); bad_windows=0; invalid_windows=0; flat_windows=0; max_window_gap=0; max_window_missing_rows=0
        for wr in arms["HF"]:
            center=float(wr["center_s"]); s=int(round((center-4)*100)); e=int(round((center+4)*100))
            valid_support=(s>=0 and e<=len(df) and e-s==800)
            if not valid_support:
                issue(anon,sub,"support_bounds",f"window={wr['window_idx']},slice={s}:{e}")
                block=arr[0:0]
            else:
                block=arr[s:e]; covered[s:e]=True
            nonfinite=int((~np.isfinite(block)).sum())
            missing_rows=(~np.isfinite(block)).any(axis=1)
            edges=np.diff(np.r_[False,missing_rows,False].astype(int))
            lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
            gap=int(lengths.max()) if lengths.size else 0
            max_window_gap=max(max_window_gap,gap)
            max_window_missing_rows=max(max_window_missing_rows,int(missing_rows.sum()))
            invalid=int((pd.to_numeric(df["ValidFlag"].iloc[s:e],errors="coerce")!=1).sum()) if valid_support and "ValidFlag" in df else -1
            interp=int((pd.to_numeric(df["InterpFlag"].iloc[s:e],errors="coerce")!=0).sum()) if valid_support and "InterpFlag" in df else -1
            flat=[CHANNELS[k] for k in range(8) if len(block) and np.isfinite(block[:,k]).all() and np.ptp(block[:,k])==0]
            bad_windows+=int(nonfinite>0 or not valid_support); invalid_windows+=int(invalid>0); flat_windows+=bool(flat)
            windows.append(dict(record=anon,subject=sub,activity=SCENES[r["scene"]],window_idx=int(wr["window_idx"]),center_s=center,start_s=center-4,end_s_exclusive=center+4,start_row=s,end_row_exclusive=e,sample_count=e-s,support_in_bounds=valid_support,nonfinite_values=nonfinite,missing_sample_rows=int(missing_rows.sum()),max_missing_run_samples=gap,invalid_flag_samples=invalid,interp_flag_samples=interp,constant_channels=";".join(flat),hf_motion=True,acc_motion=True))
        if not arms["HF"]: issue(anon,sub,"empty_motion","No frozen motion members")
        if bad_windows: issue(anon,sub,"nonfinite_or_bounds",f"{bad_windows} retained windows; no fill/drop performed")
        if invalid_windows: issue(anon,sub,"invalid_flags",f"{invalid_windows} retained windows contain ValidFlag != 1")
        if flat_windows: issue(anon,sub,"constant_window_channels",f"{flat_windows} retained windows; descriptive, no exclusion")
        for k,c in enumerate(CHANNELS):
            v=arr[covered,k]; finite=v[np.isfinite(v)]
            quality.append(dict(record=anon,subject=sub,channel=c,motion_union_samples=len(v),nonfinite=int((~np.isfinite(v)).sum()),all_missing=bool(len(v) and not len(finite)),minimum=float(finite.min()) if len(finite) else "",maximum=float(finite.max()) if len(finite) else "",constant_union=bool(len(finite) and np.ptp(finite)==0),time_grid_ok=time_ok))
        # Exact aligned one-second signal blocks detect copied/cropped material at the grid
        # resolution relevant to the one-second window stride. Not a near-duplicate proof.
        for s in range(0,len(arr)-99,100):
            if not covered[s:s+100].any(): continue
            b=arr[s:s+100]
            if not np.isfinite(b).all() or not np.any(np.ptp(b,axis=0)>0): continue
            bh=array_sha(b)
            if bh in block_groups:
                oldrec,oldsub,oldstart=block_groups[bh]
                if oldrec!=anon and oldsub!=sub:
                    duplicate_rows.append(dict(kind="exact_1s_signal_block_cross_subject",record_a=oldrec,subject_a=oldsub,start_row_a=oldstart,record_b=anon,subject_b=sub,start_row_b=s))
            else: block_groups[bh]=(anon,sub,s)
        records.append(dict(record=anon,subject=sub,activity=SCENES[r["scene"]],algorithm_record_id=rid,input_path=str(raw),input_sha256=raw_hash,archive_original_source=entry.get("source",""),signal_sha256=signal_hash,rows=len(df),time_grid_ok=time_ok,motion_windows=len(arms["HF"]),max_window_missing_rows=max_window_missing_rows,max_window_gap_samples=max_window_gap,bad_numeric_windows=bad_windows,invalid_flag_windows=invalid_windows,constant_channel_windows=flat_windows,hf_trace=str(audit/next(x["trace_path"] for x in routes if x["record_id"]==rid and x["route_id"]=="HF")),acc_trace=str(audit/next(x["trace_path"] for x in routes if x["record_id"]==rid and x["route_id"]=="ACC"))))
        print(f"P0 checked {len(records)}/119",flush=True) if len(records)%30==0 else None
    for kind,groups in (("exact_file",file_groups),("exact_signal",signal_groups)):
        for members in groups.values():
            if len({s for _,s in members})>1:
                for rec,sub in members[1:]:
                    duplicate_rows.append(dict(kind=kind+"_cross_subject",record_a=members[0][0],subject_a=members[0][1],start_row_a="",record_b=rec,subject_b=sub,start_row_b=""))
    check("119_local_window_tables_hash_and_global_members",table_bad==0,table_bad)
    check("all_inputs_match_archive",input_bad==0,input_bad)
    check("238_traces_match_hash_and_input",trace_bad==0,trace_bad)
    check("frozen_csv_motion_matches_trace",route_bad==0,route_bad)
    check("119_HF_ACC_motion_members_equal",mask_bad==0,mask_bad)
    check("detector_configuration",configs=={(100,8.0,1.0,30.0,2.5)},sorted(configs))
    check("119_records_retained",len(records)==119,len(records))
    check("9520_windows_retained",len(windows)==9520,len(windows))
    check("all_supports_800_in_bounds",all(w["support_in_bounds"] for w in windows))
    check("time_grid_identity",all(r["time_grid_ok"] for r in records))
    check("all_required_motion_numbers_finite",all(w["nonfinite_values"]==0 for w in windows))
    check("no_all_missing_channel",not any(q["all_missing"] for q in quality))
    check("no_invalid_flag_in_motion_support",all(w["invalid_flag_samples"]==0 for w in windows))
    check("no_detected_exact_cross_subject_duplicates",not duplicate_rows,len(duplicate_rows))
    splits=[]
    for held in SUBJECTS:
        train=[r for r in records if r["subject"]!=held]; test=[r for r in records if r["subject"]==held]
        check(f"{held}_eight_activity_coverage",len({r["activity"] for r in train})==len({r["activity"] for r in test})==8)
        check(f"{held}_group_isolation",not ({r["subject"] for r in train}&{r["subject"] for r in test}) and not ({r["input_sha256"] for r in train}&{r["input_sha256"] for r in test}))
        for r in records:
            splits.append(dict(fold=held,record=r["record"],subject=r["subject"],activity=r["activity"],role="test" if r["subject"]==held else "train",motion_windows=r["motion_windows"]))
    write_csv(out/"record_bindings_LOCAL.csv",records)
    write_csv(out/"motion_windows.csv",windows)
    write_csv(out/"loso_splits.csv",splits)
    write_csv(out/"channel_quality.csv",quality)
    write_csv(out/"issues.csv",issues)
    write_csv(out/"duplicate_findings.csv",duplicate_rows)
    write_csv(out/"checks.csv",checks)
    source_files=[Path(__file__),root/"AGENTS.md",root/"docs/agents/workflow.md",root/"python/src/ppg_hr/v2/solver.py",root/"python/src/ppg_hr/v2/signal_preparation.py",root/"python/src/ppg_hr/preprocess/data_loader.py",root/"python/src/ppg_hr/core/heart_rate_solver.py",audit/"record_route_metrics.csv",window_path,audit/"artifact_manifest.json",release/"artifact_manifest.json",final/"record_MAE.csv",final/"SI_record_MAE_by_subject.csv",thesis/"Research State.md",thesis/"Terminology.md"]
    write_json(out/"source_fingerprints_LOCAL.json",[dict(path=str(p),sha256=sha(p)) for p in source_files])
    peak=None
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL("kernel32",use_last_error=True); kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL("psapi",use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb): peak=counters.PeakWorkingSetSize
    except (ImportError,OSError,AttributeError): pass
    receipt=dict(schema="hf_scene_classification_p0_v1",started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,peak_working_set_bytes=peak,python=sys.executable,python_version=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),command=subprocess.list2cmdline([sys.executable,"-B",*sys.argv]),git_head=git(root,"rev-parse","HEAD"),git_status=git(root,"status","--short"),tracked_diff_sha256=hashlib.sha256(git(root,"diff","--binary").encode()).hexdigest(),records=len(records),windows=len(windows),folds=6,checks_pass=sum(c["status"]=="PASS" for c in checks),checks_fail=sum(c["status"]=="FAIL" for c in checks),nonmotion_grid_difference_records=other_grid_diff,issue_counts=dict(Counter(i["kind"] for i in issues)),training_started=False,input_modified=False,scope="Frozen MIMU-selected motion windows; no HR evaluation/error mask",training_ready=not any(c["status"]=="FAIL" for c in checks),not_done=["classification training","hardware saturation/calibration validation","near-duplicate waveform similarity","arbitrary sub-second shifted crop matching","independent physical motion annotation"],checks=checks)
    write_json(out/"receipt.json",receipt)
    write_json(out/"output_manifest.json",[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()])
    print(json.dumps({k:receipt[k] for k in ("records","windows","checks_pass","checks_fail","issue_counts","elapsed_seconds","peak_working_set_bytes","training_ready")},ensure_ascii=False))
if __name__=="__main__": main()
