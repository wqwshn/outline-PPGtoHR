# Journal Figure Rules

## Size

- IEEE single-column figures: start at 3.5 in wide.
- Nature-style single-column figures: start at 3.54 in wide.
- Thesis or two-column comparison figures: start at 6.8 in wide.
- Keep height proportional to content; avoid tall unused whitespace.
- Aspect ratio 0.68-0.73 works well for most single-panel figures.

## Typography

- Use consistent sans-serif fonts unless the target journal requires serif.
- Keep axis labels near 7 pt at final print size for Nature-style single column.
- Tick labels at 6 pt; legend at 6 pt.
- Use explicit units in labels: `Time (s)`, `Heart rate (BPM)`, `Frequency (Hz)`.
- Avoid plot titles in manuscript panels unless required.
- All text elements (axis labels, legend, in-figure tables, annotations) must use the same font family. Set `family=` explicitly on every `ax.text()` call.

## Lines, Markers, and Error Bars

- Use 0.9-1.45 pt lines for final figures, with visual hierarchy:
  - Best algorithm: 1.45 pt (thickest)
  - Secondary: 1.05 pt (normal)
  - Baseline/FFT: 0.9 pt (thin, dashed)
  - Reference ground truth: 1.05 pt solid dark gray
- Use markers sparingly on dense time-series. Use adaptive spacing:
  `markevery=max(1, len(x_data) // 18)` produces ~18 visible markers.
- Error bars must describe SD, SEM, CI, or IQR in the caption or legend.
- Use shaded confidence bands when dense error bars would clutter the plot.

## Multi-Panel Figures

- Align axes and shared labels.
- Use bold panel labels: A, B, C at 8-9 pt.
- Keep panel spacing tight but prevent tick label overlap.
- Use identical y-limits for direct method comparisons.
- Do NOT rely on auto-scaling; use a common range function across panels.

## Signal-Processing Figures

- Show sampling rate, filter band, window size, or relevant parameters in script variables.
- Distinguish raw, filtered, reference, and estimated signals by line style AND color.
- For heart-rate estimates, keep BPM axis ranges comparable across methods.
- Mark rejected or low-quality intervals explicitly when they affect interpretation.
- Use `fill_between()` with `transform=ax.get_xaxis_transform()` for event/condition
  background shading. Keep alpha in 0.16-0.30 range. Use low-saturation colors
  (e.g., #D9DDE3) that don't compete with data lines.

## In-Figure Tables

- Do NOT use `ax.table()` or monospace multi-line `ax.text()` -- column alignment breaks
  across different renderers.
- Render each cell as an individually positioned `ax.text()` with `ha="center"`.
- Place a background box as a separate empty `ax.text("")` with `bbox={}`.
- Recommended background: white fill, light gray edge (#D6D6D6), 0.35 pt border, 0.84 alpha.
- Table font should match the rest of the figure (typically Arial 6 pt).

## Legend

- Single-column narrow figures: `ncol=1` (vertical stack); wider formats can use `ncol=2`.
- Use `frameon=False` for cleaner paper figures.
- Position adaptively: `upper right` when the area is clear, `lower right` when upper
  area has background shading or annotations.
- Fine-tune with `bbox_to_anchor=(x, y)` in axes coordinates.

## Export

- For iterative review: default to 600 dpi PNG only. Smaller files, faster iteration.
- For camera-ready final: export vector PDF/SVG as the canonical artifact.
- Use `bbox_inches=None, pad_inches=0.02` for tighter bounds than `bbox_inches="tight"`.
- Use unique output paths (auto-incrementing suffix) to prevent overwriting previous results.
- Run artifact checks after export.
