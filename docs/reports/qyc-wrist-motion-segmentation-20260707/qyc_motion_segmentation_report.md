# QYC wrist motion segmentation audit (2026-07-07)

Protocol assumption supplied by user: approximately 0-60 s rest, 60-120 s motion, and remaining time rest. Detector timestamps are 8 s window centers, so a few seconds of boundary tolerance is expected.

## Summary

- Samples audited: 9
- Pass: 4
- Fail/review: 5
- CSV: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\qyc_motion_segmentation_summary.csv`

| sample | detected segment (s) | duration | start error | end error | ACC max ratio | Gyro max ratio | tail HF z | tail PPG z | status | signal adjudication |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| jianpan1_QYC_0615 | 60.00-129.00 | 69.00 | 0.00 | 9.00 | 20.000 | 20.000 | 14.160 | 21.957 | pass | protocol_aligned |
| jianpan2_QYC_0615 | 59.00-128.00 | 69.00 | -1.00 | 8.00 | 20.000 | 20.000 | 10.436 | 3.263 | pass | protocol_aligned |
| jianpan3_QYC_0615 | 60.00-131.00 | 71.00 | 0.00 | 11.00 | 20.000 | 20.000 | 4.989 | 7.638 | fail | extended_wrist_motion_signal |
| woli1_QYC_0615 | 59.00-129.00 | 70.00 | -1.00 | 9.00 | 20.000 | 20.000 | 102.136 | 12.733 | pass | protocol_aligned |
| woli2_QYC_0615 | 59.00-130.00 | 71.00 | -1.00 | 10.00 | 20.000 | 20.000 | 49.987 | 16.542 | pass | protocol_aligned |
| woli3_QYC_0615 | 59.00-131.00 | 72.00 | -1.00 | 11.00 | 20.000 | 20.000 | 10.268 | 25.078 | fail | extended_wrist_motion_signal |
| xiezi1_QYC_0615 | 62.00-132.00 | 70.00 | 2.00 | 12.00 | 20.000 | 20.000 | 70.732 | 2.393 | fail | extended_wrist_motion_signal |
| xiezi2_QYC_0615 | 60.00-142.00 | 82.00 | 0.00 | 22.00 | 20.000 | 20.000 | 14.137 | 10.504 | fail | extended_wrist_motion_signal |
| xiezi3_QYC_0615 | 59.00-136.00 | 77.00 | -1.00 | 16.00 | 20.000 | 20.000 | 18.229 | 3.870 | fail | extended_wrist_motion_signal |

## Figures

- `jianpan1_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\jianpan1_QYC_0615-motion-scores.png`
- `jianpan2_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\jianpan2_QYC_0615-motion-scores.png`
- `jianpan3_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\jianpan3_QYC_0615-motion-scores.png`
- `woli1_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\woli1_QYC_0615-motion-scores.png`
- `woli2_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\woli2_QYC_0615-motion-scores.png`
- `woli3_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\woli3_QYC_0615-motion-scores.png`
- `xiezi1_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\xiezi1_QYC_0615-motion-scores.png`
- `xiezi2_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\xiezi2_QYC_0615-motion-scores.png`
- `xiezi3_QYC_0615`: `D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\docs\reports\qyc-wrist-motion-segmentation-20260707\figures\xiezi3_QYC_0615-motion-scores.png`

## Interpretation

Several samples exceed the nominal 120 s motion end, but their 120 s-to-detected-end tails still show strong ACC/Gyro and HF/PPG disturbance. This points to extended wrist motion in the collected signal rather than a detector-only overrun. No detector code change is indicated by this audit.
Extended-signal samples: jianpan3_QYC_0615, woli3_QYC_0615, xiezi1_QYC_0615, xiezi2_QYC_0615, xiezi3_QYC_0615.

## Raw Candidate Runs

### jianpan1_QYC_0615
- raw candidates: `[{"start_s": 60.0, "end_s": 129.0, "windows": 70}]`
- retained: `[{"start_s": 60.0, "end_s": 129.0, "windows": 70}]`

### jianpan2_QYC_0615
- raw candidates: `[{"start_s": 59.0, "end_s": 128.0, "windows": 70}]`
- retained: `[{"start_s": 59.0, "end_s": 128.0, "windows": 70}]`

### jianpan3_QYC_0615
- raw candidates: `[{"start_s": 60.0, "end_s": 131.0, "windows": 72}]`
- retained: `[{"start_s": 60.0, "end_s": 131.0, "windows": 72}]`

### woli1_QYC_0615
- raw candidates: `[{"start_s": 59.0, "end_s": 129.0, "windows": 71}]`
- retained: `[{"start_s": 59.0, "end_s": 129.0, "windows": 71}]`

### woli2_QYC_0615
- raw candidates: `[{"start_s": 59.0, "end_s": 130.0, "windows": 72}]`
- retained: `[{"start_s": 59.0, "end_s": 130.0, "windows": 72}]`

### woli3_QYC_0615
- raw candidates: `[{"start_s": 59.0, "end_s": 131.0, "windows": 73}]`
- retained: `[{"start_s": 59.0, "end_s": 131.0, "windows": 73}]`

### xiezi1_QYC_0615
- raw candidates: `[{"start_s": 62.0, "end_s": 132.0, "windows": 71}]`
- retained: `[{"start_s": 62.0, "end_s": 132.0, "windows": 71}]`

### xiezi2_QYC_0615
- raw candidates: `[{"start_s": 60.0, "end_s": 142.0, "windows": 83}]`
- retained: `[{"start_s": 60.0, "end_s": 142.0, "windows": 83}]`

### xiezi3_QYC_0615
- raw candidates: `[{"start_s": 59.0, "end_s": 136.0, "windows": 78}]`
- retained: `[{"start_s": 59.0, "end_s": 136.0, "windows": 78}]`
