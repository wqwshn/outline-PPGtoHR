from __future__ import annotations

import pytest

from ppg_hr.v2.reference_groups import (
    V2_REFERENCE_GROUPS,
    channel_names_for_group,
    color_for_reference_order,
    method_label,
    normalise_reference_order,
    reference_order_key,
)


def test_reference_group_constants_are_stable() -> None:
    assert V2_REFERENCE_GROUPS == ("HF", "CF", "ACC")
    assert channel_names_for_group("HF") == ("hf1", "hf2")
    assert channel_names_for_group("CF") == ("cf1", "cf2")
    assert channel_names_for_group("ACC") == ("accx", "accy", "accz")


def test_reference_order_is_normalised_and_deduplicated() -> None:
    assert normalise_reference_order([" hf ", "CF", "HF", "acc"]) == ("HF", "CF", "ACC")
    assert reference_order_key(("HF", "CF", "ACC")) == "HF+CF+ACC"
    assert reference_order_key(()) == "FFT"


def test_invalid_reference_group_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported reference group"):
        normalise_reference_order(["HF", "GYRO"])


def test_ordered_reference_colors_are_distinct_and_stable() -> None:
    hf_acc = color_for_reference_order(("HF", "ACC"))
    acc_hf = color_for_reference_order(("ACC", "HF"))
    assert hf_acc != acc_hf
    assert color_for_reference_order(("HF", "ACC")) == hf_acc
    assert color_for_reference_order(("HF",)) == "#D4382C"


def test_method_label_formats_correctly() -> None:
    assert method_label("lms", ("HF",)) == "LMS-H"
    assert method_label("klms", ("HF", "CF", "ACC")) == "KLMS-HCA"
    assert method_label("noncausal_lms", ("ACC", "HF")) == "NLMS-AH"
    assert method_label("volterra", ("CF",)) == "VLMS-C"
    assert method_label("rff_lms", ()) == "RFF"
    assert method_label("unknown", ("HF", "ACC")) == "LMS-HA"
