"""Reference-group parsing and publication colors for v2."""

from __future__ import annotations

V2_REFERENCE_GROUPS: tuple[str, ...] = ("HF", "CF", "ACC")

_CHANNELS: dict[str, tuple[str, ...]] = {
    "HF": ("hf1", "hf2"),
    "CF": ("cf1", "cf2"),
    "ACC": ("accx", "accy", "accz"),
}

_ORDER_COLORS: dict[str, str] = {
    "FFT": "#A8ADB3",
    "HF": "#6FA8DC",
    "CF": "#8CCB9B",
    "ACC": "#D9A66A",
    "HF+CF": "#7BAF9E",
    "CF+HF": "#9AB7D9",
    "HF+ACC": "#DFAE7B",
    "ACC+HF": "#B7A0D8",
    "CF+ACC": "#A7C98B",
    "ACC+CF": "#D7A4A4",
    "HF+CF+ACC": "#5FA4B8",
    "HF+ACC+CF": "#C7A46B",
    "CF+HF+ACC": "#79B58B",
    "CF+ACC+HF": "#B6B46E",
    "ACC+HF+CF": "#AA9DD6",
    "ACC+CF+HF": "#D69AA6",
}

_FALLBACK_COLORS: tuple[str, ...] = (
    "#6FA8DC",
    "#8CCB9B",
    "#D9A66A",
    "#B7A0D8",
    "#D7A4A4",
    "#5FA4B8",
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
