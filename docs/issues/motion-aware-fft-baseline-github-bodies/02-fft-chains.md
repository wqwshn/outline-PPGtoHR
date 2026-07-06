## What to build

在独立实验工具中实现三条全程纯 FFT 基线链路。旧 `continuous_fft` 作为对照；`post_guard_reset_fft` 在保护窗结束后的首个重捕获窗无历史直取当前 PPG 主峰；`post_guard_weak_inherit_fft` 在同一边界使用 `previous_fft ± 40 BPM` 宽搜索且禁止 `held_previous` fallback。

## Acceptance criteria

- [ ] 三条链路都能对同一原始样本输出完整窗口曲线。
- [ ] `post_guard_reset_fft` 首窗不继承保护窗末端历史。
- [ ] `post_guard_weak_inherit_fft` 首窗只继承 FFT 自身历史，宽搜索固定为 40 BPM，且无峰时退回当前 PPG 主峰而不是 held previous。
- [ ] 三条链路均不启用运动参考频谱惩罚。
- [ ] 测试覆盖 reset 首窗、weak inherit 首窗和 continuous 对照行为差异。

## Blocked by

Blocked by #2
