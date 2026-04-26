# Color Palette Rules

## General

- Use color-blind-safe palettes for categorical comparisons.
- Do not encode important meaning with color alone; combine color with line style, marker, label, or panel layout.
- Avoid low-contrast yellow, pale cyan, and saturated red-green pairs on white backgrounds.
- Check figures in grayscale when the paper may be printed.

## Recommended Categorical Palettes

- Okabe-Ito for small method comparisons.
- ColorBrewer qualitative palettes for 3-8 groups.
- `colorcet`, `cmasher`, or `cmcrameri` for perceptually controlled maps.

## Sequential and Diverging Maps

- Use sequential maps for magnitude-only data.
- Use diverging maps only when a meaningful center exists, such as zero error.
- Avoid rainbow/jet for quantitative heatmaps.
- Label colorbars with units and metric names.

## PPG and Signal Figures

- Use one stable color for each signal type across a paper.
- Recommended defaults:
  - Raw PPG: blue
  - Filtered PPG: green
  - Reference HR: red
  - Estimated HR: purple
  - Rejected or low-quality intervals: neutral gray
