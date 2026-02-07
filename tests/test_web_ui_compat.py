from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_shift_web_ui as ui


def test_button_fallback_for_old_streamlit(monkeypatch) -> None:
    calls = []

    def fake_button(label, **kwargs):
        calls.append(kwargs.copy())
        if "type" in kwargs or "use_container_width" in kwargs:
            raise TypeError("unexpected keyword")
        return True

    monkeypatch.setattr(ui, "st", SimpleNamespace(button=fake_button))
    assert ui._button("実行", type="primary", use_container_width=True, key="k1") is True
    assert len(calls) == 2
    assert "type" in calls[0]
    assert "use_container_width" in calls[0]
    assert "type" not in calls[1]
    assert "use_container_width" not in calls[1]


def test_dataframe_fallback_for_old_streamlit(monkeypatch) -> None:
    calls = []

    def fake_dataframe(data, **kwargs):
        calls.append(kwargs.copy())
        if "use_container_width" in kwargs:
            raise TypeError("unexpected keyword")
        return None

    monkeypatch.setattr(ui, "st", SimpleNamespace(dataframe=fake_dataframe))
    ui._dataframe([{"a": 1}], use_container_width=True)
    assert len(calls) == 2
    assert calls[0] == {"use_container_width": True}
    assert calls[1] == {}


def test_data_editor_fallback_to_experimental(monkeypatch) -> None:
    calls = []

    def fake_experimental_data_editor(data, **kwargs):
        calls.append(kwargs.copy())
        if "column_config" in kwargs:
            raise TypeError("unexpected keyword")
        return [{"ok": True}]

    fake_st = SimpleNamespace(
        experimental_data_editor=fake_experimental_data_editor,
        warning=lambda *_args, **_kwargs: None,
        dataframe=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(ui, "st", fake_st)
    got = ui._data_editor([{"x": 1}], column_config={"x": "cfg"}, use_container_width=True)
    assert got == [{"ok": True}]
    assert len(calls) == 2
    assert "column_config" in calls[0]
    assert "column_config" not in calls[1]


def test_status_and_rerun_fallback(monkeypatch) -> None:
    info_calls = []
    rerun_called = {"count": 0}

    class Placeholder:
        def info(self, msg: str) -> None:
            info_calls.append(msg)

    fake_st = SimpleNamespace(
        empty=lambda: Placeholder(),
        experimental_rerun=lambda: rerun_called.__setitem__("count", rerun_called["count"] + 1),
    )
    monkeypatch.setattr(ui, "st", fake_st)

    status = ui._status("起動中", state="running")
    status.update(label="完了", state="complete")
    ui._rerun()

    assert any("起動中" in msg for msg in info_calls)
    assert any("完了" in msg for msg in info_calls)
    assert rerun_called["count"] == 1

