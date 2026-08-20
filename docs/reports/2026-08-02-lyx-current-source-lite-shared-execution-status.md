# LYX 当前源码 Lite 与共享参数实验执行状态报告

日期：2026-08-02
状态：已完成可审计执行框架；正式 solver 求解未启动
证据等级：`development_reuse_pilot` 的执行治理证据，不是性能结果

## 核心结论

本轮已经把“当前源码 Lite 基线更新与 25 Hz 共享参数实验”的执行入口、任务票据、
proposal、授权窗口、180 组合物理空间、双级联身份、两层缓存、八项安全门和选择器
排序合同落成到可测试 runner 中。由于执行时本机时间已经超过
`2026-08-02T08:30:00+08:00` 的预注册权限窗口，runner 按合同写入受控暂停
completion，未启动 Lite BO、物理面板或任何正式 solver 请求。code review 发现的
合同风险已经收敛到最终 runner：训练对后置约束缺失会失败关闭，solver cache 只按
实际数值配置复用，测试用时钟覆盖需要显式 opt-in，旧授权哈希不匹配时不会被复用。

## 已完成的执行证据

9 个 GitHub tickets 已发布并标记为 `ready-for-agent`：#108-#116。它们覆盖执行基线、
proposal/runner、合同与缓存、Lite 基线、公共可行性、三折选择器、证据包、最终报告
和 code review 收尾。

实际 proposal 位于
`data/experiments/lyx_current_source_lite_shared_20260802/proposal.json`，绑定到：

- 执行起点 HEAD：`90c5aeb886e75209a8ae9a63f9f88ff5f68a28f8`；
- 当前 runner HEAD：`7b7f0b3cc9316af72e90ab0ad6f054ef98ec0851`；
- `python/src/ppg_hr` 树：`df13ba56853b38fe9a51debfa976685ad4d0ae3a`；
- 实验规范 SHA-256：`fb8b081f9bd45e7a045d2cc5ef61e29c0a08ffd1fd43a700a7892e8f595556f3`；
- runner SHA-256：`3dc378b70459d8c247cb3f2a4346fa971f3ec8bc7d5cb8fb091126b5fa5fc65a`；
- 180 组合空间 SHA-256：`14d8c35f4dd039be37cce2428de159676e38dcb76791457e81ecab9636368bf2`；
- proposal SHA-256：`cd90a7e7aa33b11a34f65d307a10d28c23d8e583cf546a3da3aba402bb9c2735`。

刷新后的 proposal 未复用旧授权回执：旧回执绑定的是旧 proposal 哈希，与当前
proposal 不匹配。runner 在 `2026-08-02T09:24:31+08:00` 执行时检测到权限窗口已
过期，按“无有效授权”写入
`paused_authorization_expired_before_start` completion：

- authorization SHA-256：`null`；
- completion SHA-256：`375f480bd90f78bcc0c4950a465a922cdff6ef72e57c7726de9cbce64beb0783`；
- formal solver run count：0；
- logical request count：0；
- next state：`requires_fresh_execution_authorization`。

## 合同覆盖

本轮新增 runner 明确保留旧 `physical_v1` 语义，并新增独立的
`physical_25hz_extended_v1` 空间：固定 `fs_target=25 Hz`，九档物理记忆、五档
`mu_base` 和四档排除半宽，共 180 个确定性组合。统一当前控制
`200 ms / mu=0.010 / 6 BPM` 是该空间中的一个普通坐标，不预设为最优。

双级联身份固定为 `dual_cascade_two_hf_v1`：`reference_groups_order=["HF"]`、
`adaptive_reference_stage_limit=None`，且实际 HF 自适应级数必须等于 2。rank-1 或
级数漂移会失败关闭。

缓存合同分为 solver 与 evaluation 两层。solver 键排除 stage/fold/记录别名、候选
逻辑编号与搜索空间名，只绑定源码、数据、实际数值参数、固定参数和双级联机制身份；
evaluation 键单独绑定 solver 结果、参考心率、指标合同和安全门合同，因此参考或评价
规则变化不会错误复用旧评价。

八项逐记录安全门和近优稳定性优先选择器已有单测覆盖。安全门对缺失、非有限、双级联
漂移、L10/L20、MAE、右删失恢复、真实上升低估和当前控制退化均采用失败关闭。选择器
在训练对内按“最差训练 MAE 的 0.5 BPM 近优池、邻居支持、悬崖、平均 MAE、坐标序”
产生唯一冻结坐标；训练对相对独立 BO 的平均 MAE 退化约束缺失或非有限时，同样失败
关闭。

## 测试与边界

已通过测试：

- `conda run -n ppg-hr python -m pytest -q python/tests/test_lyx_current_source_lite_shared_runner.py --basetemp .codex-tmp\lyx-current-source-lite-shared-fix\pytest-runner-3`
  结果：10 passed；
- 测试前设置：`$env:PYTHONPATH='python/src'`
- 然后运行：`conda run -n ppg-hr python -m pytest -q python/tests/test_v2_bo_space_generalization.py --basetemp D:\codex-tmp\outline-PPGtoHR\lyx-current-source-lite-shared\pytest-bo-space`
  结果：47 passed，14 warnings；
- `git diff --check 90c5aeb886e75209a8ae9a63f9f88ff5f68a28f8..HEAD` 通过。

因此，本报告只支持一个有限结论：本轮执行框架已经能防止在过期授权窗口下误跑正式
solver，并能测试关键实验合同。它不支持任何 Lite 性能、共享参数、平台稳定性或三折
留出结果结论。

## 后续动作

继续正式实验前，需要生成新的执行授权回执，绑定当前 proposal 或重新生成后的 proposal。
获得新授权后，runner 的下一状态应从 `ready_for_lite_baseline_execution` 开始，先执行
24 条 Lite `3 × 50` 当前源码基线更新；只有 Lite 代理审核写出 `proceed` 后，才进入
四场景 180 组合公共可行性和三折选择器审计。
