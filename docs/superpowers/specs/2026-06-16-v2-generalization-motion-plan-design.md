# v2 泛化评估运动分类与实验计划设计

## 背景

现有 `python/src/ppg_hr/v2/generalization.py` 已经能从样本文件名推断 `motion_type`，并在每个运动类型内部运行 `all_train` 和 `leave_one_group_out`。但当前推断是开放式的，界面文案仍假设输入目录只有同一种运动，用户在点击“开始泛化评估”前也看不到样本如何被分类、哪些样本会被跳过、以及每类运动会生成多少个 fold。

新采集目录可能同时包含多种运动，例如 `data/20260615` 下同时有 `multi_bobi1_TS.csv`、`multi_fuwo2_TS.csv`、`multi_kaihe1_TS.csv`、`multi_tiaosheng2_TS.csv`、`multi_wanju1_TS.csv` 等文件。由于同一个运动场景共用一套 BO 超参数，泛化评估必须先按文件名里的运动信息把样本分组，再在每个运动组内独立训练和重放。

## 目标

1. 在泛化评估计算前自动扫描输入目录，按固定运动类型库分类样本。
2. 只让已识别运动类型进入 BO 泛化评估，避免未知文件名误混入共享参数组。
3. 在 GUI 中展示“运动分类与实验计划”，让用户启动计算前能确认分组、样本数、fold 数和跳过原因。
4. 让正式计算复用同一套分类和计划逻辑，避免 GUI 预览与实际执行不一致。
5. 保持现有输出结构、`summary_csv` 字段和 all-train / LOGO 语义兼容。

## 运动类型库

固定支持以下运动关键词：

| 关键词 | 含义 |
| --- | --- |
| `bobi` | 波比跳 |
| `fuwo` | 俯卧撑 |
| `kaihe` | 开合跳 |
| `tiaosheng` | 跳绳 |
| `wanju` | 弯举 |
| `run` | 跑步 |
| `rest` | 静息 |
| `yangwo` | 仰卧起坐 |
| `box` | 快速出拳 |
| `gaotai` | 高抬腿 |

匹配时对文件 stem 小写化，并按关键词边界识别。典型命名如 `multi_fuwo2_TS`、`multi_tiaosheng1`、`run_01` 都能识别。参考文件 `_ref.csv` 和 `_HR_ref.csv` 不单独参与分类，只作为传感器文件的配对参考。

若传感器文件和参考文件能够配对，但文件名不包含上述关键词，则该样本归为未识别运动类型，默认不进入计算，并在 GUI 计划表与日志中列出。

## 样本配对与分类

新增一层计划构建 API：

- `KNOWN_MOTION_TYPES`: 固定运动关键词元组。
- `infer_known_motion_type(sample_stem)`: 返回已识别运动类型或 `None`。
- `build_v2_generalization_plan(input_dir, evaluation_modes, motion_types=None)`: 扫描目录、配对 CSV、分类并生成实验计划。

`discover_sample_pairs()` 保留兼容行为，但内部改为使用固定运动库；默认只返回已识别运动类型的 `V2SamplePair`。计划对象额外记录：

- `included_pairs`: 进入计算的已识别样本；
- `unknown_pairs`: 已配对但运动类型未识别的样本；
- `unpaired_data_files`: 没有 `_ref.csv` 或 `_HR_ref.csv` 的传感器 CSV；
- `groups`: 每个运动类型的样本列表和 fold 列表。

## 实验计划语义

每个运动类型单独规划和计算：

- `all_train`: 该运动类型的全部样本作为训练集，同时重放该运动类型全部样本。
- `leave_one_group_out`: 当该运动类型样本数不少于 2 时，每次留出一个样本测试，其余样本训练；样本数小于 2 时不生成 LOGO fold。

对于 `data/20260615` 这类每个运动 2 个样本的目录，如果勾选 `all_train` 和 `leave_one_group_out`，每个运动会生成 1 个 all-train fold 和 2 个 LOGO fold。5 个运动共 15 个 fold，但每个 fold 只在自身运动组内训练共享 BO 参数。

`run_v2_generalization()` 在 setup 阶段先构建计划，再按计划执行。进度事件会继续携带 `motion_type`、`evaluation_mode` 和 `fold_id`，并新增计划摘要字段，供 GUI 或日志展示。

## GUI 设计

`V2GeneralizationPage` 的输入说明改为支持多运动目录。结果侧新增“运动分类与实验计划”表，列包含：

- 运动类型；
- 状态；
- 样本数；
- fold 数；
- 样本 stem 列表；
- 备注。

状态取值：

- `将计算`: 已识别且至少有一个选中评估模式对应的 fold；
- `仅 all_train`: 已识别但样本数不足 2，无法生成 LOGO；
- `未识别`: 文件能配对但不在运动类型库中，默认跳过；
- `未配对`: 传感器 CSV 缺少参考 HR 文件，默认跳过。

`刷新` 按钮只扫描目录并更新计划表与日志，不启动 BO。`开始泛化评估` 会先刷新计划，若没有任何可计算样本则提示错误；否则锁定按钮并启动 worker。计算完成后摘要表继续显示输出目录、汇总 CSV 和记录数。

## 错误处理

- 输入目录无效时保持现有 GUI 错误提示。
- 没有任何已识别且可计算的样本时，`run_v2_generalization()` 抛出清晰的 `ValueError`，GUI 显示“没有可计算的已识别运动样本”。
- 未识别和未配对样本不让整体任务失败，只记录在计划表与日志中。
- `motion_types` 参数若被调用方显式传入，只能进一步筛选已识别运动类型；未知 motion type 不会被隐式创建。

## 测试计划

新增和更新测试集中在两处：

1. `python/tests/test_v2_generalization.py`
   - 验证 `infer_known_motion_type()` 能识别 `bobi/fuwo/kaihe/tiaosheng/wanju/run/rest/yangwo/box/gaotai`。
   - 验证 `custom_jump_rope` 返回 `None`，不会进入计算样本。
   - 验证混合目录会按运动类型生成独立 fold，未知已配对样本记录在计划中。
   - 验证样本数为 1 的运动类型只生成 all-train，不生成 LOGO。

2. `python/tests/test_gui_v2_smoke.py`
   - 验证泛化评估页包含计划表。
   - 验证刷新时调用计划构建函数并展示运动类型、样本数和未识别状态。
   - 验证开始计算前会刷新计划，并把无可计算样本作为 GUI 错误处理。

相关测试命令：

```powershell
conda run -n ppg-hr python -m pytest -q python/tests/test_v2_generalization.py python/tests/test_gui_v2_smoke.py --basetemp D:\tmp\ppg_hr_v2_generalization_plan
```

如果 GUI 环境或临时目录权限导致测试无法运行，需要记录具体原因；优先使用 `--basetemp` 指向工作区或 `D:\tmp` 下的目录，避免默认临时目录权限问题。

## 非目标

- 不引入用户可编辑的运动类型配置文件。
- 不改变 BO 搜索空间、求解器参数或 v2 HR 算法本身。
- 不把未识别运动类型自动作为新组计算。
- 不新增命令行入口；本次只覆盖 Python API 和现有 GUI 泛化评估页。

## 交付结果

实现完成后，用户在 GUI 中选择包含多种运动的目录并点击刷新，可以看到每个运动类型的样本和 fold 规划。点击开始后，系统按计划在每个运动内部独立训练共享 BO 参数并输出现有泛化评估报告，不再要求输入目录只包含同一种运动。
