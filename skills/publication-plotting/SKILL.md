---
name: publication-plotting
description: Use when creating publication-ready scientific figures, journal or thesis plots, signal-processing visualizations, algorithm performance comparisons, line plots, error bars, heatmaps, multi-panel figures, or exported PDF/SVG/high-DPI PNG figures from research data.
---

# Publication Plotting

## Overview

Create论文级科研图时使用本 Skill。优先生成可复现的 Python 脚本，统一风格、尺寸、字体、颜色、导出格式和基础质量检查。

## Workflow

1. 明确目标期刊或用途：`ieee_single_column`、`nature_single_column`、`thesis_double_column`，或按用户指定尺寸设置。
2. 读取数据时优先使用 `pandas`、`numpy`、`scipy`、`h5py`、`openpyxl` 等结构化接口。
3. 使用 `scripts/plot_style.py` 设置全局风格，不在每个脚本中重复硬编码 rcParams。
4. 根据图型选择清晰编码：折线图用于趋势，误差棒/置信区间用于不确定性，热图用于矩阵或参数扫描，多子图用于对齐比较。
5. 使用色盲友好、可灰度区分的调色板；需要详细规则时读取 `references/color_palette_rules.md`。
6. 使用 `scripts/export_figure.py` 同时导出 PDF、SVG 和 600 dpi PNG，除非用户明确指定其他格式。
7. 使用 `scripts/figure_check.py` 检查文件存在、非空、PNG 尺寸和图像可读性。

## Bundled Resources

Use these files from the skill directory:

- `scripts/plot_style.py`: apply journal/thesis presets and reusable color cycles.
- `scripts/export_figure.py`: export Matplotlib figures to PDF, SVG, and PNG with consistent metadata and bounding boxes.
- `scripts/figure_check.py`: validate exported figure artifacts.
- `assets/*.mplstyle`: Matplotlib style presets for IEEE, Nature-like, and thesis figures.
- `references/journal_figure_rules.md`: sizing, typography, line width, panels, captions, and export rules.
- `references/color_palette_rules.md`: palette selection and accessibility rules.

## Defaults

- 默认输出目录：`figures/`
- 默认数据目录：`data/`
- 默认脚本目录：`scripts/`
- 默认导出：PDF、SVG、600 dpi PNG
- 默认字体：Arial/DejaVu Sans fallback；缺少 LaTeX 时使用 `science` + `no-latex`
- 默认尺寸：
  - IEEE single column: 3.5 in wide
  - Nature single column: 3.54 in wide
  - Thesis double column: 6.8 in wide

## Plotting Rules

- Never rely on default Matplotlib colors for final figures.
- Keep axes labels explicit and include units where available.
- Avoid chart titles in paper figures unless the target venue expects them; put context in captions or panel labels.
- Use vector output for manuscripts and high-DPI PNG for previews or slides.
- Keep line widths, marker sizes, tick sizes, and legend text readable after column scaling.
- For signal-processing figures, show enough context to interpret preprocessing, reference signal, estimates, residuals, or error bands.
- For algorithm comparisons, keep metric definitions and axis ranges consistent across methods.

## Quick Example

```python
from pathlib import Path
import sys
import matplotlib.pyplot as plt

skill_scripts = Path("skills/publication-plotting/scripts").resolve()
sys.path.insert(0, str(skill_scripts))
from export_figure import export_figure
from plot_style import apply_publication_style

apply_publication_style("ieee_single_column")
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 0], label="Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (a.u.)")
ax.legend()
export_figure(fig, Path("figures/example"), formats=("pdf", "svg", "png"), dpi=600)
```

## Common Mistakes

- Too many colors: group methods or use facets instead.
- Tiny legends: move legend outside or use direct labels.
- Raster-only manuscript figure: export PDF/SVG as the canonical artifact.
- Hidden preprocessing choices: annotate filters, windows, confidence intervals, and reference signals in script variables or captions.
