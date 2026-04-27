---
name: publication-plotting
description: Use when creating publication-ready scientific figures, journal or thesis plots, signal-processing visualizations, algorithm performance comparisons, line plots, error bars, heatmaps, multi-panel figures, in-figure tables, or exported PDF/SVG/high-DPI PNG figures from research data. Also use when the user asks about plotting parameters, figure styling, color palettes for papers, Matplotlib publication tricks, or how to make a figure look like a specific journal format. Trigger even for simple plot improvement requests like "make this chart look better for my paper" or "adjust the font/line colors for publication".
---

# Publication Plotting

## Overview

Create publication-ready scientific figures. The skill covers end-to-end workflow: style setup, data visualization patterns, in-figure annotation (error tables, legends), and multi-format export with quality checks.

All guidance is framework-agnostic but uses Matplotlib examples. Adapt patterns to Plotly, Seaborn, or other toolkits as needed.

## Workflow

1. Identify target venue and format: `ieee_single_column`, `nature_single_column`, `thesis_double_column`, or custom dimensions.
2. Load data with structured interfaces (`pandas`, `numpy`, `scipy`, `h5py`).
3. Apply global style via `scripts/plot_style.py` -- never hardcode rcParams in individual scripts.
4. Choose the right visual encoding for each data relationship (see Plotting Patterns below).
5. Use color-blind-safe, grayscale-compatible palettes (see `references/color_palette_rules.md`).
6. Export via `scripts/export_figure.py` and validate with `scripts/figure_check.py`.

## Bundled Resources

- `scripts/plot_style.py` -- Journal/thesis presets with reusable color cycles.
- `scripts/export_figure.py` -- Multi-format export with metadata and consistent bounding.
- `scripts/figure_check.py` -- Validate exported figure artifacts.
- `assets/*.mplstyle` -- Matplotlib style presets for IEEE, Nature-like, and thesis figures.
- `references/journal_figure_rules.md` -- Sizing, typography, line width, panels, captions, and export rules.
- `references/color_palette_rules.md` -- Palette selection, accessibility rules, and signal-processing palettes.

## Defaults

- Default output directory: `figures/`
- Default export: PDF + SVG + 600 dpi PNG (use PNG-only for review workflows, see below)
- Default font: Arial / DejaVu Sans fallback; SciencePlots `science` + `no-latex` when available
- Default sizes: IEEE 3.5" / Nature 3.54" / Thesis 6.8" wide

## Plotting Patterns

These patterns come from real publication figure development and address common pitfalls.

### Algorithm Comparison Line Plots

When plotting multiple algorithms against a reference (e.g., estimated vs. ground truth), create visual hierarchy through line properties rather than relying solely on color:

| Role | Line width | Marker | Color family | Rationale |
|------|-----------|--------|-------------|-----------|
| Reference / ground truth | 1.05 | none | dark gray (#2B2B2B) | Solid, neutral baseline |
| Best algorithm | 1.45 | circle, size 2.0 | warm (orange #E68653) | Thickest line draws the eye |
| Secondary algorithm | 1.05 | dot, size 2.0 | cool (blue #5DA9C9) | Normal weight |
| Baseline / FFT / naive | 0.9 | none | neutral gray (#A8ADB3), dashed | Visually subordinate |

This hierarchy lets readers instantly see which method performs best without reading the legend.

### Marker Density

Dense time-series (hundreds+ points) should not have markers on every data point. Use adaptive spacing:

```python
markevery=max(1, len(x_data) // 18)
```

This produces ~18 markers across the plot range -- enough to distinguish lines when printed in grayscale, but not so many that they form a thick band.

### Event / Condition Background Shading

To highlight time periods (motion segments, experimental conditions, anomalies):

```python
ax.fill_between(
    x, 0, 1, where=condition_mask,
    transform=ax.get_xaxis_transform(),  # 0-1 on y-axis regardless of data range
    color="#D9DDE3",        # low-saturation gray-blue
    alpha=0.24,             # sweet spot: visible but not competing with data
    edgecolor="none",
    zorder=0,               # behind all data lines
)
```

Keep alpha in the 0.16-0.30 range. Above 0.35, the background starts to obscure data lines and makes the figure look washed out when printed.

### In-Figure Error / Summary Tables

Do NOT use `ax.table()` or monospace multi-line `ax.text()` for embedded error tables -- they produce misaligned columns across different renderers and backends.

Instead, render each cell as an individually positioned `ax.text()` call:

```python
# Background box (rendered first, empty text, just for the box)
ax.text(x0, y_top, "", transform=ax.transAxes, fontsize=1,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
              "edgecolor": "#D6D6D6", "linewidth": 0.35, "alpha": 0.84})

# Each cell individually
for row_idx, (label, values) in enumerate(data_rows):
    y = y_top - offset - row_idx * line_height
    for x, text in zip(x_columns, [label, f"{val1:.1f}", f"{val2:.1f}"]):
        ax.text(x, y, text, transform=ax.transAxes, ha="center",
                fontsize=6, family="Arial", color="#333333")
```

Why this works: `ha="center"` on each cell gives true horizontal centering regardless of font metrics. The background box is a separate element that doesn't affect cell positions.

### Legend Placement for Narrow Figures

Single-column figures (3.5" or narrower) have very limited space:

- Prefer `ncol=1` (single vertical column) for narrow figures; `ncol=2` only for wider formats
- Use `frameon=False` for a cleaner look in paper figures
- Position adaptively based on data density:
  - Upper area clear: `loc="upper right"`, no `bbox_to_anchor` needed
  - Upper area has shading/annotations: `loc="lower right"`
  - Fine-tune with `bbox_to_anchor=(x, y)` in axes coordinates

### Y-axis Range for Multi-Panel Comparisons

Do not use auto-scaling when panels should be visually comparable. Use a common range function:

```python
def common_ylim(*series, default_lo=55.0, default_hi=150.0, step=5.0,
                floor=35.0, ceiling=210.0):
    values = np.concatenate([np.asarray(s).ravel() for s in series])
    values = values[np.isfinite(values)]
    lo, hi = default_lo, default_hi
    if values.min() < lo:
        lo = np.floor((values.min() - 3.0) / step) * step
    if values.max() > hi:
        hi = np.ceil((values.max() + 3.0) / step) * step
    return max(floor, lo), min(ceiling, hi)
```

Round to `step` increments so tick labels land cleanly.

### Font Consistency

All text elements in one figure must use the same font family. Common pitfall: axis labels and legend use the style preset, but in-figure tables or annotations use the system default.

When rendering text via `ax.text()` (for tables, annotations, etc.), always explicitly set `family=` to match the style preset. Recommended sizes for Nature-style single column:

| Element | Size (pt) |
|---------|----------|
| Axis labels | 7 |
| Tick labels | 6 |
| Legend | 6 |
| In-figure tables | 5.8-6 |
| Panel labels (A, B, C) | 8-9, bold |

### SciencePlots Graceful Fallback

Always handle the case where SciencePlots is not installed:

```python
try:
    import scienceplots  # noqa: F401
    styles = ["science", "no-latex", your_preset]
except ImportError:
    styles = [your_preset]
plt.style.use(styles)

# Provide a complete fallback rcParams dict
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,    # TrueType embedding
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
```

### Unique Output Paths

When generating figures that may be re-run (review cycles, parameter sweeps), prevent accidental overwrites:

```python
def unique_path(path):
    if not path.exists():
        return path
    for idx in range(2, 10000):
        candidate = path.with_name(f"{path.stem}-{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot allocate unique path for {path}")
```

This is especially important in GUI tools where users click "render" multiple times.

### PNG-Only Review Workflow

For iterative review with collaborators, default to PNG-only to keep file sizes manageable and avoid confusing version mixes:

```python
fig.savefig(path, dpi=600, bbox_inches=None, pad_inches=0.02)
```

Note: `bbox_inches=None` with small `pad_inches` often produces tighter bounds than `bbox_inches="tight"`. Switch to PDF+SVG only for the final camera-ready version.

## Plotting Rules

- Never rely on default Matplotlib colors for final figures.
- Keep axis labels explicit and include units where available.
- Avoid chart titles in paper figures; put context in captions or panel labels.
- Keep line widths, marker sizes, tick sizes, and legend text readable after column scaling.
- For algorithm comparisons, keep metric definitions and axis ranges consistent across panels.
- For signal-processing figures, show enough context to interpret preprocessing, reference signals, estimates, and error bands.

## Quick Example

```python
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

skill_scripts = Path("skills/publication-plotting/scripts").resolve()
sys.path.insert(0, str(skill_scripts))
from export_figure import export_figure
from plot_style import apply_publication_style

apply_publication_style("nature_single_column")
fig, ax = plt.subplots(figsize=(3.54, 2.6))

t = np.linspace(0, 30, 500)
ref = 75 + 10 * np.sin(2 * np.pi * t / 15)
best = ref + np.random.normal(0, 1.5, len(t))
baseline = ref + np.random.normal(0, 3.0, len(t))

ax.plot(t, ref, color="#2B2B2B", linewidth=1.05, label="Reference", zorder=5)
ax.plot(t, best, color="#E68653", linewidth=1.45, label="Proposed", zorder=4,
        marker="o", markersize=2.0, markevery=max(1, len(t) // 18))
ax.plot(t, baseline, color="#A8ADB3", linewidth=0.9, linestyle="--", label="FFT", zorder=2)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Heart rate (BPM)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", ncol=1, frameon=False, fontsize=6)
export_figure(fig, Path("figures/example"), formats=("png",), dpi=600)
```

## Common Mistakes

- Too many colors: group methods or use facets instead.
- Tiny legends: move legend outside or use direct labels on lines.
- Raster-only manuscript figure: export PDF/SVG as the canonical artifact for camera-ready.
- Hidden preprocessing choices: annotate filters, windows, confidence intervals, and reference signals.
- Mixed fonts in one figure: always set `family=` explicitly on every `ax.text()` call.
- Overwriting previous results: use unique output paths in iterative workflows.
- Dense markers on long time-series: use `markevery` to keep markers sparse and readable.
- Auto-scaled y-limits across comparison panels: use a common range function for visual consistency.
