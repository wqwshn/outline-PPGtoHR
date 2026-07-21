# 稳定交汇必须相对实际 Final 判断

Status: superseded by ADR-0030

`stable_crossover` 被定义为已取得资格的交接 reset FFT 与当前实际 Final 的连续可达交汇，并且永远采用非硬切的正常过渡。内部 adaptive 轨迹与交接 reset 接近只能作为辅助诊断，不能单独触发稳定交汇；当实际 Final 与目标仍有较大距离时，系统继续等待或独立评估 `gap_rescue`。

该决定修订 ADR-0005 的正常入口判据。HB 的 kaihe3 表明，后处理前的 adaptive 内部轨迹可能已经在错误低频区与 reset 相交，而后处理后的实际 Final 仍相差近 40 BPM；继续以内部轨迹判断会让名义上的稳定交汇产生硬跳。改用实际 Final 会牺牲部分最快切换时机，但保证正常入口的输出连续性，并把需要快速纠错的大间隙情况留给取得资格后的 `gap_rescue` 硬切。
