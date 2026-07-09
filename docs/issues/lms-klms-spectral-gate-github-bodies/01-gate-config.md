## Parent

https://github.com/wqwshn/outline-PPGtoHR/issues/14

## What to build

建立 LMS/KLMS 运动段机制门控实验的公共配置 seam。默认行为必须保持当前生产语义：KLMS 不启用低频重捕获或高频锁定逃逸。实验运行可以通过 allowlist 显式允许 KLMS 使用这些门控，从而支持 8 条件矩阵。

这个 slice 的完成结果应该能通过 v2 求解公开配置证明：默认 KLMS gate 仍关闭；allowlist 包含 KLMS 时，低频重捕获和高频锁定逃逸可以按各自开关独立生效；输出 payload 记录本次运行的 allowlist 与有效门控状态。

## Acceptance criteria

- [ ] 默认 KLMS 求解不启用低频重捕获或高频锁定逃逸。
- [ ] 实验 allowlist 包含 KLMS 时，KLMS 可按配置启用低频重捕获。
- [ ] 实验 allowlist 包含 KLMS 时，KLMS 可按配置启用高频锁定逃逸。
- [ ] 低频重捕获与高频锁定逃逸可以独立关闭和开启。
- [ ] 结果 payload 记录 active allowlist、低频重捕获有效开关和高频锁定逃逸有效开关。
- [ ] 相关 solver 测试通过，且测试验证公开行为而不是私有实现细节。

## Blocked by

None - can start immediately
