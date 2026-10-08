# 论文119记录原始信号与频谱重放

Windows 下双击仓库根目录的 `GUI.bat` 即可启动。启动脚本自动定位本机 Conda，使用 `ppg-hr` 环境；失败时保留错误信息。也可在仓库根目录运行：

```powershell
conda run --no-capture-output -n ppg-hr python tools/launch_paper_replay.py
```

启动器使用当前检出的源码，打开原 GUI 的“窗口诊断”页并加载论文索引，不改变共享环境的 editable 安装。常规 `python -m ppg_hr.gui` 也可使用该页，顶部默认选择“论文119冻结结果与原始信号”；旧报告入口保留在另一个模式中，不能作为论文冻结回放。

## 数据接入

依次使用 `PPG_HR_DATA_ROOT`、当前目录下包含论文归档的 `data/`，或本机保存项目的 `D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data`。也可在页面中直接填写 data 目录并加载。worktree 不需要复制实验数据。

只读输入为 `experiments/paper_fft_acc_audit_20260907/` 的记录、窗口路线表与 traces，以及 `experiments/paper_release_20260906/` 的 raw 索引、信号、参考及冻结源码。索引必须恰有119个唯一记录、6个subject、8个场景，每记录具有HF、ACC、FFT三条路线，不生成组合笛卡尔积。只展示存在记录的场景；没有记录的组合不出现在列表。

匿名ID沿用论文映射：按 scene、subject、原始record_id排序，在每个subject×scene中从01编号。HW=Handwriting，HG=Handgrip，TYP=Typing，RS=Rope Skipping，RUN=Running，JJ=Jumping Jacks，BUR=Burpees，PCH=Punching。内部原始身份不用于用户选择信息。

## 手动浏览

1. 选择subject、场景、记录和HF/ACC/独立FFT路线。选择区下方“原始数据文件”同步显示实际CSV文件名，可选中复制，悬停查看完整路径。切换立即清空旧图，并取消前一请求；旧请求完成后不会覆盖新记录。
2. HR概览显示三路线及离线参考，灰色背景为运动范围。点击概览、拖动滑块、前后窗口按钮、输入中心/起点均定位同一实际窗口。时间是原始记录秒数，参考时间另行显示，不能把评价time bias加到原始波形上。
3. 原始区为green PPG、HF1、HF2独立三轨及ACC三轴叠画，共用时间轴。默认PPG=ADC counts/16 nA，也可切回counts；HF为Ut1/Ut2原始mV，ACC为原始g。保留正负极性、基线和重力。无效样本断线并标红；已有插值标橙，不另补造信号。
4. 上下文长度独立于归档算法窗。蓝色高亮始终使用实际配置中的8秒算法窗，既有步长来自该路线配置。静息参照有独立起点与长度，默认5秒；作者自行调整，不自动选峰。左右区域可分别滚动，滚动左侧即可同时观察原始四轨和右侧频谱。
5. 点击“完整历史重放 / 读取已核验缓存”。首次计算从记录起点执行整条记录，缓存后所有窗口直接查看。HF、ACC各用自己的冻结配置与采样率；FFT使用归档中正式独立FFT列，不以最大峰或adaptive内部handoff代替。
6. 诊断分为频谱、滤波级联、完整追踪记录三个标签。谱图显示实际捕获的滤前、滤后、惩罚后幅值、候选峰、剔除峰、历史搜索及保护区。滤前ADC谱、LMS标准化滤后谱、参考谱分轴显示，不能据其幅值比推断能量抑制；滤后惩罚前后共用尺度，显示层不再独立归一化。可关闭滤前谱。参考谱为实际惩罚参考的峰谱，单独坐标不代表绝对能量抑制比。没有捕获的中间量明确标缺失；静息无需自适应级联。
7. “保存选择 JSON”记录匿名ID、subject、场景、序号、路线、起止/中心秒、版本与trace哈希、上下文、静息参照及备注。“恢复选择 JSON”核对版本与trace身份，再恢复记录和窗口。默认存放于本地 `data/experiments/paper_replay_browser/selections/`。

## 回放身份与边界

冻结源码在独立Python进程中导入，当前源码不会代替它。适配器仅观察冻结函数返回时的数组，不替换函数、参数、状态或返回值。完整HR矩阵（含NaN模式）对归档逐点校验，绝对容差1e-9、相对容差0；不一致则不发布为已核验中间量。源代码、原始数据、参考、trace均按归档哈希核验。

缓存键绑定适配器schema、trace哈希（包含完整配置）与冻结源码哈希，读取时再次核验源身份和NPZ哈希。缓存和Numba编译输出写当前检出的 `data/experiments/paper_replay_browser/cache/`；取消会终止独立进程。源数据和冻结源码不写入。

PPG输入候选谱采用冻结求解器的去均值、Hamming窗、8192点FFT与单边2/N幅值定义；它是经过算法预处理的输入谱，不是原始直流波形谱。参考惩罚峰谱采用其冻结矩形窗FFT规则。冻结LMS内部会标准化参考及期望信号，因此滤后输出为无量纲；级联输出图明确标注该处理。原始图区没有这些处理。详细诊断中的连续 `fft` 追踪与运动后独立reset、handoff记录分别保留；归档所选路线HR单独标明，不能把连续追踪峰直接称为最终独立FFT结果。

## 验证与复现

```powershell
$env:PYTHONPATH = "$PWD/python/src"
New-Item -ItemType Directory -Force .codex-tmp/paper-replay | Out-Null
conda run -n ppg-hr python -m pytest -q python/tests/test_paper_replay.py python/tests/test_paper_replay_async.py python/tests/test_paper_replay_native_layout.py python/tests/test_gui_smoke.py --basetemp .codex-tmp/paper-replay/pytest
conda run --no-capture-output -n ppg-hr python tools/validate_paper_replay.py --ui-snapshot
```

本次验收：119记录/357路线全数加载及归档曲线核验；匿名ID与论文既有119映射完全一致。HG、HW、TYP、JJ、RS五类场景各HF/ACC共10次整记录回放，覆盖25/50/100 Hz，完整HR矩阵最大差均为0。新功能与原GUI测试40项通过；旧窗口诊断通过15项，另13项因历史 `testforwindiag` 数据未提供而跳过。旧测试中开合跳fixture只读引用主检出路径，输出留在worktree临时目录。

原生Qt回归测试已复现并防止三行Matplotlib画布在标签页/滚动区中因尺寸协商递归导致的栈溢出；画布沿用项目既有固定高度模式。

验收脚本还检查静息、运动、运动后、首尾窗口导航并生成真实Qt截图。默认输出 `data/experiments/paper_replay_browser/acceptance.json` 与 `ui-*.png`，供本地审阅。实验数据、缓存、机器回执及截图均为Git忽略的本地产物，不暂存、不提交、不推送。

## 异步闪退与交互修复

后台结果原先通过普通Python闭包回调进入界面，在本机PySide环境中会运行在worker线程，导致QTextDocument等Qt对象跨线程创建并可能闪退。现改为绑定页面的Qt Slot，并显式使用QueuedConnection；成功、失败与线程清理都在GUI主线程处理。线程由主线程先移除持有记录、再deleteLater，取消后的过期结果仍按请求编号丢弃。

滑块拖动期间合并连续请求，停顿80 ms或松开时更新最终窗口；输入秒数在确认编辑时定位。仅绘制当前打开的原始/静息及诊断标签页，切换标签时按当前窗口更新；整条HR概览复用曲线，仅移动窗口高亮。算法配置、数值与归档校验不变。

本机同一真实记录的原生Qt交互检查中，20次连续滑块变化的总响应从约8.2秒降至约0.32秒，单次窗口切换约0.3–0.4秒。该数字是本机GUI交互检查，不是算法速度基准。新增线程归属、错误/取消/关闭、拖动合并与隐藏标签页联动回归；相关GUI测试共40项通过。


后续原生闪退修复：阶段波形标签页也固定画布高度，避免缩放窗口或切换标签时触发 Qt 布局递归。原生回归覆盖五组窗口尺寸、三轮全部诊断标签切换；修复前会以 `0xc00000fd` 栈溢出退出。启动器将 Python 异常及原生崩溃堆栈写入 `.codex-tmp/paper-replay-logs/gui-时间-进程号.log`，启动控制台显示具体路径。

## 2026-10-08 版本纳管核验

本次将既有重放实现、线程/布局回归、启动器和验收工具纳入 Git。仅修正导入排序及测试中保留 QApplication 生命周期引用的变量名，没有改变冻结算法配置、数值或缓存身份。

相关测试初次为76项通过、13项因缺少历史 `testforwindiag` 数据跳过；导入与变量名修正后，论文重放及新旧GUI的61项测试全部通过，包含原生Qt布局子进程回归。受影响源码、测试与工具的Ruff检查通过。

实际归档的119记录/357路线全数加载、输入哈希、归档曲线和首尾边界核验通过；五类场景各HF/ACC共10组完整记录回放核验覆盖25/50/100 Hz，完整HR矩阵最大差为0。此次复用已核验的本地缓存，验收仍核对冻结输入、源码、trace及缓存NPZ指纹。新回执与截图独立保存于本地 `data/experiments/paper_replay_browser/version_closeout_20261008/`，没有覆盖旧验收材料。
