# Journal Figure Rules

## Size

- IEEE single-column figures: start at 3.5 in wide.
- Nature-style single-column figures: start at 3.54 in wide.
- Thesis or two-column comparison figures: start at 6.8 in wide.
- Keep height proportional to content; avoid tall unused whitespace.

## Typography

- Use consistent sans-serif fonts unless the target journal requires serif.
- Keep axis labels near 7-9 pt at final print size.
- Use explicit units in labels: `Time (s)`, `Heart rate (bpm)`, `Frequency (Hz)`.
- Avoid plot titles in manuscript panels unless required.

## Lines, Markers, and Error Bars

- Use 1.0-1.5 pt lines for final figures.
- Use markers sparingly when many samples are present.
- Error bars must describe SD, SEM, CI, or IQR in the caption or legend.
- Use shaded confidence bands when dense error bars would clutter the plot.

## Multi-Panel Figures

- Align axes and shared labels.
- Use bold panel labels: A, B, C.
- Keep panel spacing tight but prevent tick label overlap.
- Use identical y-limits for direct method comparisons when possible.

## Signal-Processing Figures

- Show sampling rate, filter band, window size, or relevant parameters in script variables.
- Distinguish raw, filtered, reference, and estimated signals by line style and color.
- For heart-rate estimates, keep bpm axis ranges comparable across methods.
- Mark rejected or low-quality intervals explicitly when they affect interpretation.

## Export

- Export vector PDF/SVG for manuscripts.
- Export 600 dpi PNG for previews, reports, and slide decks.
- Use transparent backgrounds only when the destination requires it.
- Run artifact checks after export.
