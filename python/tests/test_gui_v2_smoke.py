from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


def test_v2_workers_are_exported() -> None:
    from ppg_hr.gui.workers import (
        V2BatchPipelineWorker,
        V2BatchPlotWorker,
        V2GeneralizationWorker,
        V2SpO2Worker,
    )

    assert V2GeneralizationWorker is not None
    assert V2BatchPipelineWorker is not None
    assert V2BatchPlotWorker is not None
    assert V2SpO2Worker is not None


def test_v2_batch_page_exposes_reference_order_controls() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        assert page.selected_reference_order() == ("HF",)
        for i in range(page._ref_list.count()):
            item = page._ref_list.item(i)
            if item is not None and item.text() == "CF":
                item.setCheckState(Qt.CheckState.Checked)
        assert page.selected_reference_order() == ("HF", "CF")
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_plot_page_has_refresh_button_and_curve_defaults() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPlotPage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPlotPage()
    try:
        assert page._refresh_btn.text() == "刷新"
        assert page.selected_plot_curves() == ("reference", "fft", "adaptive")
        page._plot_fft_check.setChecked(False)
        assert page.selected_plot_curves() == ("reference", "adaptive")
    finally:
        page.deleteLater()
        app.processEvents()


def test_main_window_can_switch_between_v1_and_v2() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        assert win.current_version() == "v2"
        assert win.nav_names() == [
            "批量全流程",
            "泛化评估",
            "批量绘图",
            "窗口诊断",
            "血氧计算",
        ]
        win.set_version("v1")
        assert win.current_version() == "v1"
        v1_names = win.nav_names()
        assert "优化" in v1_names
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_v2_batch_page_defaults_to_hf_and_exposes_all_filters() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        filters = [
            str(page._filter_combo.itemData(i))
            for i in range(page._filter_combo.count())
        ]
        assert filters == ["lms", "as_lms", "klms", "volterra", "noncausal_lms", "rff_lms"]
        assert page.selected_reference_order() == ("HF",)
        assert page._num_repeats.value() == 3
        assert page._ppg_input_transform_combo.currentData() == "raw_bandpass"
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_batch_page_exposes_algorithm_preset_selection() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        presets = [
            str(page._algorithm_preset_combo.itemData(i))
            for i in range(page._algorithm_preset_combo.count())
        ]
        assert presets == ["dynamic_rest_bo", "lite", "trace_rescue"]
        assert page.selected_algorithm_preset() == "dynamic_rest_bo"
        page._algorithm_preset_combo.setCurrentIndex(
            page._algorithm_preset_combo.findData("trace_rescue")
        )
        assert page.selected_algorithm_preset() == "trace_rescue"
        assert page.selected_comparison_groups() == (("ACC",),)
        app.processEvents()
        assert page._max_iter.isHidden() is True
        page._filter_combo.setCurrentIndex(page._filter_combo.findData("klms"))
        app.processEvents()
        assert page._max_iter.isHidden() is False
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_generalization_page_exposes_full_all_train_defaults() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2GeneralizationPage

    app = QApplication.instance() or QApplication([])
    page = V2GeneralizationPage()
    try:
        assert page._ppg_input_transform_combo.currentData() == "raw_bandpass"
        assert page._scope_combo.currentData() == "full"
        assert page._eval_mode_combo.currentData() == "all_train"
        assert page.selected_evaluation_modes() == ("all_train",)
        assert page._k_fold_spin.isHidden() is True
        assert page._external_test_dir_pick.isHidden() is True
        page._eval_mode_combo.setCurrentIndex(page._eval_mode_combo.findData("k_fold_holdout"))
        app.processEvents()
        assert page._k_fold_spin.isHidden() is False
        assert page._k_fold_spin.value() == 5
        page._eval_mode_combo.setCurrentIndex(page._eval_mode_combo.findData("cross_person"))
        app.processEvents()
        assert page._external_test_dir_pick.isHidden() is False
        assert page._max_iter.value() == 150
        assert page.selected_reference_order() == ("HF",)
        for i in range(page._ref_list.count()):
            item = page._ref_list.item(i)
            if item is not None and item.text() == "ACC":
                item.setCheckState(Qt.CheckState.Checked)
        assert page.selected_reference_order() == ("HF", "ACC")
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_generalization_page_exposes_algorithm_preset_selection() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2GeneralizationPage

    app = QApplication.instance() or QApplication([])
    page = V2GeneralizationPage()
    try:
        presets = [
            str(page._algorithm_preset_combo.itemData(i))
            for i in range(page._algorithm_preset_combo.count())
        ]
        assert presets == ["dynamic_rest_bo", "lite", "trace_rescue"]
        assert page.selected_algorithm_preset() == "dynamic_rest_bo"
        page._algorithm_preset_combo.setCurrentIndex(
            page._algorithm_preset_combo.findData("trace_rescue")
        )
        assert page.selected_algorithm_preset() == "trace_rescue"
        assert page.selected_comparison_groups() == (("ACC",),)
        app.processEvents()
        assert page._max_iter.isHidden() is True
        page._filter_combo.setCurrentIndex(page._filter_combo.findData("klms"))
        app.processEvents()
        assert page._max_iter.isHidden() is False
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_generalization_page_updates_progress_widgets() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2GeneralizationPage

    app = QApplication.instance() or QApplication([])
    page = V2GeneralizationPage()
    try:
        page._on_progress(
            {
                "overall_percent": 37,
                "stage_percent": 65,
                "title": "训练共享参数",
                "message": "repeat 1/3 | trial 4/75 | sample=multi_tiaosheng4",
            }
        )

        assert page._overall_progress.value() == 37
        assert page._stage_progress.value() == 65
        assert "37%" in page._overall_progress.text()
        assert "65%" in page._stage_progress.text()
        assert page._progress_title.text() == "训练共享参数"
        assert "multi_tiaosheng4" in page._progress_meta.text()
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_generalization_page_refresh_displays_motion_plan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui import v2_pages
    from ppg_hr.gui.v2_pages import V2GeneralizationPage
    from ppg_hr.v2.generalization import (
        V2GeneralizationFoldPlan,
        V2GeneralizationGroupPlan,
        V2GeneralizationPlan,
        V2SamplePair,
    )

    data = tmp_path / "multi_bobi1_TS.csv"
    ref = tmp_path / "multi_bobi1_TS_HR_ref.csv"
    data.write_text("sensor\n", encoding="utf-8")
    ref.write_text("ref\n", encoding="utf-8")
    pair = V2SamplePair(data, ref, "bobi")

    def fake_build_plan(input_dir, *, evaluation_modes, motion_types=None, **_kwargs):
        return V2GeneralizationPlan(
            input_dir=Path(input_dir),
            evaluation_modes=tuple(evaluation_modes),
            groups=(
                V2GeneralizationGroupPlan(
                    "bobi",
                    (pair,),
                    (
                        V2GeneralizationFoldPlan(
                            "all_train",
                            "all_train",
                            (pair,),
                            (pair,),
                        ),
                    ),
                ),
            ),
            included_pairs=(pair,),
            unknown_pairs=(
                V2SamplePair(
                    tmp_path / "custom.csv",
                    tmp_path / "custom_HR_ref.csv",
                    "unknown",
                ),
            ),
            unpaired_data_files=(tmp_path / "multi_run1_TS.csv",),
        )

    monkeypatch.setattr(v2_pages, "build_v2_generalization_plan", fake_build_plan)

    app = QApplication.instance() or QApplication([])
    page = V2GeneralizationPage()
    try:
        page._input_dir_pick.setPath(tmp_path)
        page._refresh()

        assert page._plan_table.rowCount() == 3
        assert page._plan_table.item(0, 0).text() == "all_train"
        assert page._plan_table.item(0, 1).text() == "bobi"
        assert page._plan_table.item(0, 2).text() == "1"
        assert page._plan_table.item(1, 6).text() == "\u672a\u8bc6\u522b"
        assert page._plan_table.item(2, 6).text() == "\u672a\u914d\u5bf9"
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_generalization_page_run_stops_when_plan_has_no_runnable_folds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui import v2_pages
    from ppg_hr.gui.v2_pages import V2GeneralizationPage
    from ppg_hr.v2.generalization import V2GeneralizationPlan

    def fake_build_plan(input_dir, *, evaluation_modes, motion_types=None, **_kwargs):
        return V2GeneralizationPlan(
            input_dir=Path(input_dir),
            evaluation_modes=tuple(evaluation_modes),
            groups=(),
            included_pairs=(),
            unknown_pairs=(),
            unpaired_data_files=(),
        )

    monkeypatch.setattr(v2_pages, "build_v2_generalization_plan", fake_build_plan)
    started = {"value": False}

    class FakeHolder:
        def __init__(self, worker) -> None:
            self.worker = worker

        def start(self) -> None:
            started["value"] = True

    monkeypatch.setattr(v2_pages, "WorkerThread", FakeHolder)

    app = QApplication.instance() or QApplication([])
    page = V2GeneralizationPage()
    try:
        page._input_dir_pick.setPath(tmp_path)
        page._run()

        assert started["value"] is False
        assert "没有可计算" in page._log.toPlainText()
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_batch_page_can_reorder_enabled_references() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2BatchPipelinePage

    app = QApplication.instance() or QApplication([])
    page = V2BatchPipelinePage()
    try:
        for i in range(page._ref_list.count()):
            item = page._ref_list.item(i)
            if item is not None and item.text() == "CF":
                item.setCheckState(Qt.CheckState.Checked)
        assert page.selected_reference_order() == ("HF", "CF")
        hf_item = page._ref_list.takeItem(0)
        page._ref_list.insertItem(1, hf_item)
        assert page.selected_reference_order() == ("CF", "HF")
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_spo2_page_exposes_reference_order_controls() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2SpO2Page

    app = QApplication.instance() or QApplication([])
    page = V2SpO2Page()
    try:
        assert page.selected_reference_order() == ("HF",)
        assert page._delay_samples.value() == 20
        assert page._max_order.value() == 20
        assert page._mu_base.value() == pytest.approx(0.12)
        assert "Ut1、Ut2 两条独立恢复" in page._ref_list.toolTip()
        filters = [
            str(page._filter_combo.itemData(i))
            for i in range(page._filter_combo.count())
        ]
        assert filters == ["lms", "as_lms", "klms", "volterra", "noncausal_lms", "rff_lms"]
        assert page._filter_combo.currentData() == "lms"
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_spo2_page_holdbreath_checkbox_disables_ut_controls() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2SpO2Page

    app = QApplication.instance() or QApplication([])
    page = V2SpO2Page()
    try:
        assert page._holdbreath_check.text() == "屏气实验"
        page._holdbreath_check.setChecked(True)
        app.processEvents()
        assert page._ref_list.isEnabled() is False
        assert page._filter_combo.isEnabled() is False
        assert page._delay_samples.isEnabled() is False
        assert page._max_order.isEnabled() is False
        assert page._mu_base.isEnabled() is False
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_spo2_page_builds_holdbreath_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui import v2_pages
    from ppg_hr.gui.v2_pages import V2SpO2Page

    data = tmp_path / "Spo2_HB1.csv"
    data.write_text("Time(s),PPG_Red,PPG_IR,ValidFlag\n0,1,1,1\n", encoding="utf-8")
    captured = {}

    class FakeSignal:
        def connect(self, _slot):
            return None

    class FakeWorker:
        def __init__(self, cfg, output_prefix):
            captured["cfg"] = cfg
            captured["output_prefix"] = output_prefix
            self.log = FakeSignal()
            self.finished = FakeSignal()
            self.failed = FakeSignal()

    class FakeHolder:
        def __init__(self, worker) -> None:
            self.worker = worker

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(v2_pages, "V2SpO2Worker", FakeWorker)
    monkeypatch.setattr(v2_pages, "WorkerThread", FakeHolder)

    app = QApplication.instance() or QApplication([])
    page = V2SpO2Page()
    try:
        page._data_pick.setPath(data)
        page._holdbreath_check.setChecked(True)
        page._run()

        assert captured["cfg"].extras["holdbreath_enabled"] is True
        assert captured["output_prefix"] == "Spo2_HB1"
        assert captured["started"] is True
    finally:
        page.deleteLater()
        app.processEvents()


def test_main_window_v2_navigation_includes_window_diagnostics() -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.app import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        assert "窗口诊断" in win.nav_names()
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_v2_window_diagnostics_page_exposes_controls() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2WindowDiagnosticsPage

    app = QApplication.instance() or QApplication([])
    page = V2WindowDiagnosticsPage()
    try:
        assert page._report_pick is not None
        assert page._time_spin.suffix() == " s"
        assert page._wave_final_check.isChecked()
        assert page._wave_stage_check.isChecked() is False
        assert page._spectrum_penalized_check.isChecked()
        assert page._spectrum_canvas is not None
        assert page._tracking_canvas is not None
        assert page._spectrum_canvas.axes is not page._tracking_canvas.axes
        assert page._window_ranges_label is not None
        assert page.selected_comparison_groups() == ()
        for i in range(page._comparison_ref_list.count()):
            item = page._comparison_ref_list.item(i)
            if item is not None and item.text() == "ACC":
                item.setCheckState(Qt.CheckState.Checked)
        assert page.selected_comparison_groups() == (("ACC",),)
        assert page._save_vectors_check.isChecked() is False
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_window_diagnostics_page_places_waveform_left_and_details_right() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2WindowDiagnosticsPage

    app = QApplication.instance() or QApplication([])
    page = V2WindowDiagnosticsPage()
    try:
        assert page.body().indexOf(page._wave_card) >= 0
        assert page.body_right().indexOf(page._wave_card) == -1
        assert page.body_right().indexOf(page._spectrum_card) >= 0
        assert page.body_right().indexOf(page._tracking_card) >= 0
        assert page.body_right().indexOf(page._result_card) >= 0
        assert (
            page.body_right().indexOf(page._spectrum_card)
            < page.body_right().indexOf(page._tracking_card)
            < page.body_right().indexOf(page._result_card)
        )
        assert page._wave_canvas.minimumHeight() == 260
        assert page._wave_canvas.maximumHeight() == 260
        assert page._spectrum_canvas.minimumHeight() == 220
        assert page._spectrum_canvas.maximumHeight() == 220
        assert len(page._spectrum_canvas.figure.axes) == 1
        assert page._tracking_canvas.minimumHeight() == 240
        assert page._tracking_canvas.maximumHeight() == 240
        assert page._summary.minimumHeight() >= 96
        assert page._summary.verticalHeader().defaultSectionSize() >= 44

        for i in range(page._comparison_ref_list.count()):
            item = page._comparison_ref_list.item(i)
            if item is not None and item.text() == "ACC":
                item.setCheckState(Qt.CheckState.Checked)
        page._sync_spectrum_canvas_layout()
        assert len(page._spectrum_canvas.figure.axes) == 2
        assert page._spectrum_canvas.minimumHeight() == 440
        assert page._spectrum_canvas.maximumHeight() == 440

        for i in range(page._comparison_ref_list.count()):
            item = page._comparison_ref_list.item(i)
            if item is not None and item.text() == "ACC":
                item.setCheckState(Qt.CheckState.Unchecked)
        page._sync_spectrum_canvas_layout()
        assert len(page._spectrum_canvas.figure.axes) == 1
        assert page._spectrum_canvas.minimumHeight() == 220
        assert page._spectrum_canvas.maximumHeight() == 220
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_window_diagnostics_page_keeps_load_and_render_workers_separate(
    monkeypatch,
) -> None:
    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui import v2_pages
    from ppg_hr.gui.v2_pages import V2WindowDiagnosticsPage
    from ppg_hr.v2.window_diagnostics import DiagnosticWindow

    class FakeSession:
        data_path = "data.csv"
        ref_path = "ref.csv"

        def __init__(self) -> None:
            self.windows = [
                DiagnosticWindow(
                    window_idx=0,
                    start_s=0.0,
                    center_s=4.0,
                    end_s=8.0,
                    aligned_time_s=9.0,
                    ref_hr_bpm=70.0,
                    fft_hr_bpm=72.0,
                    final_hr_bpm=71.0,
                    error_bpm=1.0,
                    is_motion=False,
                    used_adaptive=False,
                    reliable=True,
                    window_kind="rest",
                ),
                DiagnosticWindow(
                    window_idx=1,
                    start_s=1.0,
                    center_s=5.0,
                    end_s=9.0,
                    aligned_time_s=10.0,
                    ref_hr_bpm=90.0,
                    fft_hr_bpm=88.0,
                    final_hr_bpm=89.0,
                    error_bpm=-1.0,
                    is_motion=True,
                    used_adaptive=True,
                    reliable=True,
                    window_kind="motion",
                ),
                DiagnosticWindow(
                    window_idx=2,
                    start_s=2.0,
                    center_s=6.0,
                    end_s=10.0,
                    aligned_time_s=11.0,
                    ref_hr_bpm=85.0,
                    fft_hr_bpm=80.0,
                    final_hr_bpm=84.0,
                    error_bpm=-1.0,
                    is_motion=False,
                    used_adaptive=True,
                    reliable=True,
                    window_kind="recovery",
                )
            ]

        def select_nearest_window(self, _value: float):
            return self.windows[0]

        def window_kind_ranges(self):
            return [
                ("rest", 9.0, 9.0),
                ("motion", 10.0, 10.0),
                ("recovery", 11.0, 11.0),
            ]

    class FakeHolder:
        instances: list[FakeHolder] = []

        def __init__(self, worker) -> None:
            self.worker = worker
            FakeHolder.instances.append(self)

        def start(self) -> None:
            return None

    monkeypatch.setattr(v2_pages, "WorkerThread", FakeHolder)

    app = QApplication.instance() or QApplication([])
    page = V2WindowDiagnosticsPage()
    try:
        page._on_session_loaded(FakeSession())

        assert hasattr(page, "_load_worker_holder")
        assert hasattr(page, "_render_worker_holder")
        assert page._render_worker_holder is FakeHolder.instances[-1]
        assert page._load_worker_holder is not page._render_worker_holder
        ranges = page._window_ranges_label.text()
        assert "静息段：9.0–9.0 s" in ranges
        assert "运动段：10.0–10.0 s" in ranges
        assert "运动恢复段：11.0–11.0 s" in ranges
    finally:
        page.deleteLater()
        app.processEvents()


def test_v2_window_diagnostics_summary_exposes_tracking_fields() -> None:
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication

    from ppg_hr.gui.v2_pages import V2WindowDiagnosticsPage

    app = QApplication.instance() or QApplication([])
    page = V2WindowDiagnosticsPage()
    result = SimpleNamespace(
        summary={
            "aligned_time_s": 99.5,
            "center_s": 94.0,
            "start_s": 90.0,
            "end_s": 98.0,
            "window_kind": "motion",
            "tracking_path": "adaptive",
            "penalty_applied": True,
            "tracking_source": "diagnostic_replay",
            "candidate_peaks_bpm_json": "[54.0, 108.0]",
            "raw_candidate_hr_bpm": 54.0,
            "previous_hr_bpm": 108.0,
            "search_min_bpm": 83.0,
            "search_max_bpm": 133.0,
            "selected_peak_rank": 2,
            "tracked_hr_bpm": 108.0,
            "slew_limited_hr_bpm": 108.0,
            "smoothed_path_hr_bpm": 108.4,
            "ref_hr_bpm": 129.9,
            "fft_hr_bpm": 54.2,
            "final_hr_bpm": 108.4,
            "error_bpm": -21.5,
            "is_motion": True,
            "used_adaptive": True,
            "reliable": True,
        }
    )
    try:
        labels = {label for label, _value in page._summary_rows(result)}
        assert {
            "窗口类别",
            "算法路径",
            "追踪来源",
            "前5候选峰",
            "上一窗口HR",
            "搜索范围",
            "选中峰排名",
            "限幅后HR",
        } <= labels
    finally:
        page.deleteLater()
        app.processEvents()
