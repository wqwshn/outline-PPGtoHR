## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/14

## What to build

提供 8 条件 LMS/KLMS 机制门控补跑脚本。脚本使用整段求解运动段评估口径，固定 Lite、green、raw_bandpass、full、HF 和既有 BO/追踪/后处理参数组。脚本必须支持 dry-run、单样本、全样本和重复条件过滤，并把每个条件输出到独立子目录。

样本发现应从数据根目录派生，只纳入写字、敲键盘、握力计和拳击场景，排除 `run/`、`run1/run2/run3` 和既有输出目录。正式批量运行前，应能用 `xiezi2_LYX_0708` 做 smoke test，证明 gate-off、low-only、high-only 和 gate-full 的机制状态符合预期。

## Acceptance criteria

- [ ] 脚本可 dry-run 输出将运行的样本、场景、条件和输出目录，不写实验结果。
- [ ] 脚本支持只运行 `xiezi2_LYX_0708` 的 smoke test。
- [ ] 脚本支持 `--all` 批量运行纳入样本。
- [ ] 脚本支持重复指定 `--condition` 以运行 8 条件子集。
- [ ] 输出目录采用独立时间戳根目录，每个条件一个子目录。
- [ ] 样本筛选排除跑步样本和历史输出目录。
- [ ] smoke test 可检查 KLMS gate-full 不再全部 disabled，gate-off 均 disabled，low-only/high-only 各自独立。

## Blocked by

- https://github.com/wqwshn/outline-PPGtoHR/issues/15
