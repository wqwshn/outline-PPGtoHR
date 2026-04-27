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

## Signal Processing Palettes

For algorithm comparison plots (estimated vs. reference), use warm/cool contrast to create visual hierarchy. The "best" algorithm gets a warm color that naturally draws the eye:

### Heart Rate / Time-Series Algorithm Comparison

| Role | Hex | Description | Rationale |
|------|-----|-------------|-----------|
| Reference (ground truth) | `#2B2B2B` | Dark charcoal | Neutral, authoritative, never confused with estimates |
| Best algorithm | `#E68653` | Warm orange | Warm colors naturally attract attention; stands out against cool blue |
| Secondary algorithm | `#5DA9C9` | Cool blue | Pairs well with warm orange; clearly distinct |
| Baseline / FFT / naive | `#A8ADB3` | Neutral gray | Visually subordinate; doesn't compete with primary results |
| Event background | `#D9DDE3` | Very light gray-blue | Low-saturation background for motion/condition shading |

This palette works because:
1. The warm/cool contrast between best (#E68653) and secondary (#5DA9C9) makes the primary result immediately visible.
2. Gray tones (reference, baseline, background) recede, letting the colored algorithms stand out.
3. It passes Deuteranopia and Protanopia simulation checks -- the algorithms remain distinguishable by line width and marker style even without color.
4. It prints well in grayscale: the darkest lines are reference and best algorithm.

### General Signal-Type Palette

For raw vs. processed signal visualization:

| Signal type | Hex | Description |
|-------------|-----|-------------|
| Raw PPG | `#2A6FBB` | Blue |
| Filtered PPG | `#238B45` | Green |
| Reference HR | `#D43F3A` | Red |
| Estimated HR | `#6A51A3` | Purple |
| Rejected / low-quality intervals | `#B0B0B0` | Neutral gray |

## Palette Selection Decision Tree

1. Comparing algorithms against a reference? Use the algorithm comparison palette above.
2. Showing raw/filtered/reference signals? Use the signal-type palette above.
3. General categorical comparison (3-8 groups)? Use Okabe-Ito.
4. Heatmap / matrix data? Use perceptually uniform sequential maps (viridis, cividis, inferno).
5. Diverging data (positive/negative)? Use coolwarm or RdBu with a meaningful center point.
