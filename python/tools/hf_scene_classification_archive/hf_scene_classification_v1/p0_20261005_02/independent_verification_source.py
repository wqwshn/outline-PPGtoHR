import csv,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
root=Path(r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR")
o=root/"data/experiments/hf_scene_classification_v1/p0_20261005_02"
def rows(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.perf_counter()
manifest=json.loads((o/"output_manifest.json").read_text(encoding="utf-8"))
assert all(h(o/x["path"])==x["sha256"] for x in manifest)
records=rows(o/"record_bindings_LOCAL.csv"); windows=rows(o/"motion_windows.csv"); splits=rows(o/"loso_splits.csv")
assert len(records)==119 and len(windows)==9520 and len(splits)==714
cols=["AccX(g)","AccY(g)","AccZ(g)","GyroX(dps)","GyroY(dps)","GyroZ(dps)","Ut1(mV)","Ut2(mV)"]
affected=0; nbad=0
for r in records:
    p=Path(r["input_path"]);assert h(p)==r["input_sha256"]
    df=pd.read_csv(p);a=df[cols].to_numpy(dtype=float)
    ws=[w for w in windows if w["record"]==r["record"]]
    trace=json.loads(Path(r["hf_trace"]).read_text(encoding="utf-8"))
    assert {(int(w["window_idx"]),float(w["center_s"])) for w in ws}=={(i,float(x[0])) for i,x in enumerate(trace["hr"]) if x[4]}
    bad=0
    for w in ws:
        c=float(w["center_s"]);s=round((c-4)*100);e=round((c+4)*100)
        assert e-s==800 and s==int(w["start_row"]) and e==int(w["end_row_exclusive"]) and 0<=s<e<=len(a)
        n=int(np.count_nonzero(~np.isfinite(a[s:e])))
        assert n==int(w["nonfinite_values"])
        bad+=n>0
    affected+=bad>0;nbad+=bad
for sub in sorted({r["subject"] for r in records}):
    ss=[s for s in splits if s["fold"]==sub]
    assert len(ss)==119 and all((s["role"]=="test")==(s["subject"]==sub) for s in ss)
    assert len({s["activity"] for s in ss if s["role"]=="test"})==8
assert affected==47 and nbad==2052
result={"status":"PASS","checks":"artifact hashes;119 inputs post-run hashes;9520 members directly from HF traces;all 800-point slices;all window nonfinite counts;714 split memberships and eight-class test coverage","records":119,"windows":9520,"affected_records":affected,"affected_windows":nbad,"elapsed_seconds":time.perf_counter()-start,"training_performed":False,"source_script_sha256":h(root/"python/tools/audit_hf_scene_classification_p0.py")}
with (o/"independent_verification.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
print(json.dumps(result))
