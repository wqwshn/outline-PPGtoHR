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
    "klms": "KLMS",
    "volterra": "VLMS",
    "noncausal_lms": "NLMS",
    "rff_lms": "RFF",
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
    return f"{algo}-{suffix}"


_ORDER_COLORS: dict[str, str] = {
    "FFT": "#A8ADB3",
    "HF": "#D4382C",
    "CF": "#8CCB9B",
    "ACC": "#D9A66A",
    "HF+CF": "#E06930",
    "CF+HF": "#E07B40",
    "HF+ACC": "#D4552A",
    "ACC+HF": "#C84830",
    "CF+ACC": "#A7C98B",
    "ACC+CF": "#D7A4A4",
    "HF+CF+ACC": "#D44A30",
    "HF+ACC+CF": "#E06038",
    "CF+HF+ACC": "#D46038",
    "CF+ACC+HF": "#D06040",
    "ACC+HF+CF": "#C85038",
    "ACC+CF+HF": "#D05840",
}

_FALLBACK_COLORS: tuple[str, ...] = (
    "#D4382C",
    "#E06930",
    "#D4552A",
    "#C84830",
    "#D44A30",
    "#E06038",
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
