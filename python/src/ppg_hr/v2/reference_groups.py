"""Reference-group parsing and publication colors for v2."""

from __future__ import annotations

V2_REFERENCE_GROUPS: tuple[str, ...] = ("HF", "CF", "ACC")

_CHANNELS: dict[str, tuple[str, ...]] = {
    "HF": ("hf1", "hf2"),
    "CF": ("cf1", "cf2"),
    "ACC": ("accx", "accy", "accz"),
}

_ALGO_SHORT: dict[str, str] = {
    "lms": "LMS",
    "klms": "K-LMS",
    "volterra": "V-LMS",
    "noncausal_lms": "NC-LMS",
    "rff_lms": "RFF-LMS",
}

_REF_SHORT: dict[str, str] = {
    "HF": "H",
    "CF": "C",
    "ACC": "A",
}


def method_label(adaptive_filter: str, groups: tuple[str, ...]) -> str:
    algo = _ALGO_SHORT.get(str(adaptive_filter).strip().lower(), "LMS")
    if not groups:
        return algo
    suffix = "".join(_REF_SHORT.get(g, g) for g in normalise_reference_order(groups))
    return f"{algo}+{suffix}"


_ORDER_COLORS: dict[str, str] = {
    # Baseline / spectral method
    "FFT": "#A7ADB5",      # soft neutral gray

    # Single-reference cases
    "HF": "#D95F5F",       # warm coral red
    "CF": "#6AAA8B",       # muted sage green / cyan-green
    "ACC": "#5B8FC0",      # soft scientific blue

    # Two-reference combinations
    # HF-dominant: warm family, shifted by CF or ACC
    "HF+CF": "#D9855E",    # coral-orange, HF with CF contribution
    "HF+ACC": "#C96C88",   # muted rose, HF blended with ACC

    # CF-dominant: green family, shifted toward warm or cool
    "CF+HF": "#B2A75D",    # soft olive, CF shifted by HF
    "CF+ACC": "#58AA9B",   # teal-green, CF shifted by ACC

    # ACC-dominant: blue family, shifted toward warm or CF
    "ACC+HF": "#8B7CB8",   # muted blue-violet, ACC shifted by HF
    "ACC+CF": "#5F9DB8",   # blue-teal, ACC shifted by CF

    # Three-reference combinations
    # HF-first: still warm, but less saturated than pure HF
    "HF+CF+ACC": "#D27565",  # warm terracotta
    "HF+ACC+CF": "#C97993",  # muted rose-purple

    # CF-first: green/olive/teal branch
    "CF+HF+ACC": "#A5AD68",  # soft yellow-green / olive
    "CF+ACC+HF": "#6BA996",  # muted sea green

    # ACC-first: blue/violet branch
    "ACC+HF+CF": "#8D83AD",  # muted lavender-blue
    "ACC+CF+HF": "#6B99B2",  # muted slate blue-teal
}

_FALLBACK_COLORS: tuple[str, ...] = (
    "#D95F5F",  # warm HF-like
    "#6AAA8B",  # CF-like green
    "#5B8FC0",  # ACC-like blue
    "#D9855E",  # warm mixed
    "#58AA9B",  # teal mixed
    "#8B7CB8",  # blue-violet mixed
)


def normalise_reference_order(groups: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for raw in groups:
        group = str(raw).strip().upper()
        if group not in V2_REFERENCE_GROUPS:
            raise ValueError(
                f"Unsupported reference group {raw!r}; expected one of {V2_REFERENCE_GROUPS}"
            )
        if group not in seen:
            seen.append(group)
    return tuple(seen)


def reference_order_key(groups: tuple[str, ...]) -> str:
    order = normalise_reference_order(groups)
    return "+".join(order) if order else "FFT"


def channel_names_for_group(group: str) -> tuple[str, ...]:
    key = normalise_reference_order([group])[0]
    return _CHANNELS[key]


def color_for_reference_order(groups: tuple[str, ...]) -> str:
    key = reference_order_key(groups)
    if key in _ORDER_COLORS:
        return _ORDER_COLORS[key]
    idx = sum(ord(ch) for ch in key) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]
