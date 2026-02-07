from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer import load_policy
from optimizer.shift_model import solve_problem
from optimizer.schemas import ProblemDefinition


TEMPLATE_NAMES = [
    "さくら",
    "ひなた",
    "あおい",
    "みお",
    "ゆい",
    "りん",
    "はる",
    "めい",
    "すず",
    "のあ",
]
TEMPLATE_ICU = {"さくら", "あおい", "りん", "はる"}

SHIFT_TYPES = [
    {"id": "day", "label": "日勤"},
    {"id": "evening", "label": "遅番"},
    {"id": "night", "label": "夜勤"},
]
SHIFT_LABELS = {item["id"]: item["label"] for item in SHIFT_TYPES}
SHIFT_LABEL_TO_ID = {label: shift_id for shift_id, label in SHIFT_LABELS.items()}
SHIFT_CLASS = {
    "day": "shift-day",
    "evening": "shift-evening",
    "night": "shift-night",
    "-": "shift-empty",
}


def _toggle(label: str, *, value: bool, key: str, help: str) -> bool:
    """Compatibility wrapper for older Streamlit versions without st.toggle."""
    if hasattr(st, "toggle"):
        return bool(st.toggle(label, value=value, key=key, help=help))
    return bool(st.checkbox(label, value=value, key=key, help=help))


class _CompatStatus:
    def __init__(self, label: str, state: str) -> None:
        self._placeholder = st.empty()
        self.update(label=label, state=state)

    def update(self, *, state: str | None = None, label: str | None = None) -> None:
        state_text = f"[{state}]" if state else ""
        label_text = label or ""
        self._placeholder.info(f"{state_text} {label_text}".strip())


def _status(label: str, *, state: str = "running") -> Any:
    if hasattr(st, "status"):
        return st.status(label, state=state)
    return _CompatStatus(label=label, state=state)


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
        return
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _data_editor(data: Any, **kwargs: Any) -> Any:
    if hasattr(st, "data_editor"):
        return st.data_editor(data, **kwargs)
    if hasattr(st, "experimental_data_editor"):
        try:
            return st.experimental_data_editor(data, **kwargs)
        except TypeError:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("column_config", None)
            return st.experimental_data_editor(data, **fallback_kwargs)
    st.warning("このStreamlitバージョンは編集UIに非対応のため、表を閲覧表示します。")
    _dataframe(data, use_container_width=bool(kwargs.get("use_container_width", True)))
    return data


def _dataframe(data: Any, *, use_container_width: bool = True) -> None:
    try:
        st.dataframe(data, use_container_width=use_container_width)
    except TypeError:
        st.dataframe(data)


def _button(label: str, **kwargs: Any) -> bool:
    try:
        return bool(st.button(label, **kwargs))
    except TypeError:
        fallback = dict(kwargs)
        fallback.pop("type", None)
        fallback.pop("use_container_width", None)
        return bool(st.button(label, **fallback))


def _build_dates(start: date, days: int) -> List[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(days)]


def _build_default_state(start: date, days: int) -> Dict[str, Any]:
    dates = _build_dates(start, days)
    requested = {
        "さくら": [dates[1]] if len(dates) > 1 else [],
        "ひなた": [dates[3]] if len(dates) > 3 else [],
        "あおい": [dates[2]] if len(dates) > 2 else [],
        "みお": [dates[5]] if len(dates) > 5 else [],
        "ゆい": [],
        "りん": [dates[0]] if dates else [],
        "はる": [],
        "めい": [dates[4]] if len(dates) > 4 else [],
        "すず": [],
        "のあ": [dates[6]] if len(dates) > 6 else [],
    }

    nurses = []
    for name in TEMPLATE_NAMES:
        nurses.append(
            {
                "氏名": name,
                "ICU": name in TEMPLATE_ICU,
                "最大勤務回数": 4,
                "希望休": ", ".join(requested[name]),
                "登録看護師": True,
                "新人": False,
                "ベテラン": name in {"さくら", "りん", "はる"},
                "休職中": False,
                "支援要員": False,
                "外部要員": False,
                "所属病棟": "A病棟",
                "許可勤務区分": "day,evening,night",
                "月間最大勤務時間": 160,
                "月間夜勤上限": 8,
                "承認休暇日": "",
                "停止日": "",
                "希望シフト(JSON)": "{}",
            }
        )

    demand_rows = []
    for day in dates:
        demand_rows.append(
            {
                "日付": day,
                "日勤必要人数": 2,
                "遅番必要人数": 1,
                "夜勤必要人数": 1,
                "夜勤ICU必須": True,
                "病棟": "A病棟",
                "リーダー必要": False,
                "支援シフト": False,
                "祝日": False,
            }
        )

    return {
        "nurses": nurses,
        "demands": demand_rows,
    }


def _build_default_demand_rows(start: date, days: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for day in _build_dates(start, days):
        rows.append(
            {
                "日付": day,
                "日勤必要人数": 2,
                "遅番必要人数": 1,
                "夜勤必要人数": 1,
                "夜勤ICU必須": True,
                "病棟": "A病棟",
                "リーダー必要": False,
                "支援シフト": False,
                "祝日": False,
            }
        )
    return rows


def _init_state(start: date, days: int) -> None:
    if "nurses_editor" not in st.session_state or "demands_editor" not in st.session_state:
        defaults = _build_default_state(start, days)
        st.session_state.nurses_editor = defaults["nurses"]
        st.session_state.demands_editor = defaults["demands"]


def _reset_template(start: date, days: int) -> None:
    defaults = _build_default_state(start, days)
    st.session_state.nurses_editor = defaults["nurses"]
    st.session_state.demands_editor = defaults["demands"]


def _sync_demands_editor_with_period(start: date, days: int) -> None:
    """Match demand rows to planning period length and dates.

    - Row count is always equal to planning days.
    - Existing per-day values are kept when date still exists.
    - New dates get default values.
    """
    target_rows = _build_default_demand_rows(start, days)
    existing_rows = st.session_state.get("demands_editor", [])
    existing_by_date: Dict[str, Dict[str, Any]] = {}
    for row in existing_rows:
        day = str(row.get("日付", "")).strip()
        if day and day not in existing_by_date:
            existing_by_date[day] = row

    merged_rows: List[Dict[str, Any]] = []
    for row in target_rows:
        day = row["日付"]
        if day in existing_by_date:
            current = existing_by_date[day]
            merged_rows.append(
                {
                    "日付": day,
                    "日勤必要人数": int(current.get("日勤必要人数", row["日勤必要人数"])),
                    "遅番必要人数": int(current.get("遅番必要人数", row["遅番必要人数"])),
                    "夜勤必要人数": int(current.get("夜勤必要人数", row["夜勤必要人数"])),
                    "夜勤ICU必須": bool(current.get("夜勤ICU必須", row["夜勤ICU必須"])),
                    "病棟": str(current.get("病棟", row["病棟"])),
                    "リーダー必要": bool(current.get("リーダー必要", row["リーダー必要"])),
                    "支援シフト": bool(current.get("支援シフト", row["支援シフト"])),
                    "祝日": bool(current.get("祝日", row["祝日"])),
                }
            )
        else:
            merged_rows.append(row)
    st.session_state.demands_editor = merged_rows


def _apply_infeasible_recommendation(
    code: str, item: Dict[str, Any], nurse_count: int
) -> str:
    """Apply one practical tweak from infeasibility analysis and return summary text."""
    if code == "global_capacity_exceeded":
        current = int(st.session_state.get("ui_max_shifts_per_nurse", 4))
        updated = min(7, current + 1)
        st.session_state.ui_max_shifts_per_nurse = updated
        return f"最大勤務回数を {current} → {updated} に更新しました。"

    if code == "daily_capacity_exceeded":
        day_text = str(item.get("message", ""))
        day = day_text.split(" ")[0] if day_text else ""
        rows = st.session_state.get("demands_editor", [])
        for row in rows:
            if str(row.get("日付", "")).strip() != day:
                continue
            # Reduce demand to nurse count while preserving non-negative values.
            total = (
                int(row.get("日勤必要人数", 0))
                + int(row.get("遅番必要人数", 0))
                + int(row.get("夜勤必要人数", 0))
            )
            overflow = max(0, total - nurse_count)
            for field in ("日勤必要人数", "遅番必要人数", "夜勤必要人数"):
                if overflow <= 0:
                    break
                value = int(row.get(field, 0))
                dec = min(value, overflow)
                row[field] = value - dec
                overflow -= dec
            st.session_state.demands_editor = rows
            return f"{day} の必要人数合計を看護師数以内に調整しました。"
        return "対象日が特定できなかったため、自動調整できませんでした。"

    if code == "skill_capacity_exceeded":
        message = str(item.get("message", ""))
        day = message.split(" ")[0] if message else ""
        rows = st.session_state.get("demands_editor", [])
        for row in rows:
            if str(row.get("日付", "")).strip() == day:
                row["夜勤ICU必須"] = False
                st.session_state.demands_editor = rows
                return f"{day} の夜勤ICU必須を OFF にしました。"
        return "対象日の夜勤設定を見つけられませんでした。"

    if code == "night_rest_capacity_exceeded":
        current = int(st.session_state.get("ui_rest_after_night", 1))
        updated = max(0, current - 1)
        st.session_state.ui_rest_after_night = updated
        return f"夜勤後休日日数を {current} → {updated} に更新しました。"

    # Generic fallback: relax exact-demand requirement.
    st.session_state.ui_enforce_exact_demand = False
    return "必要人数を厳密一致 OFF に変更しました。"


def _parse_nurses(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nurses: List[Dict[str, Any]] = []
    for row in rows:
        name = str(row.get("氏名", "")).strip()
        if not name:
            continue
        requested_raw = str(row.get("希望休", "")).strip()
        requested = [item.strip() for item in requested_raw.split(",") if item.strip()]
        leave_raw = str(row.get("承認休暇日", "")).strip()
        leave_dates = [item.strip() for item in leave_raw.split(",") if item.strip()]
        suspended_raw = str(row.get("停止日", "")).strip()
        suspended_dates = [item.strip() for item in suspended_raw.split(",") if item.strip()]
        allowed_types_raw = str(row.get("許可勤務区分", "")).strip()
        allowed_shift_types = [
            item.strip() for item in allowed_types_raw.split(",") if item.strip()
        ]
        preferences_raw = str(row.get("希望シフト(JSON)", "{}")).strip()
        try:
            preferences = json.loads(preferences_raw) if preferences_raw else {}
            if not isinstance(preferences, dict):
                preferences = {}
        except json.JSONDecodeError:
            preferences = {}
        novice = bool(row.get("新人", False))
        experienced = bool(row.get("ベテラン", not novice))
        nurses.append(
            {
                "id": name,
                "skills": ["icu"] if bool(row.get("ICU", False)) else [],
                "max_shifts": int(row.get("最大勤務回数", 4)),
                "requested_days_off": requested,
                "registered": bool(row.get("登録看護師", True)),
                "novice": novice,
                "experienced": experienced,
                "suspended": bool(row.get("休職中", False)),
                "support_staff": bool(row.get("支援要員", False)),
                "external": bool(row.get("外部要員", False)),
                "ward": str(row.get("所属病棟", "")).strip() or None,
                "allowed_shift_types": allowed_shift_types,
                "max_monthly_hours": float(row.get("月間最大勤務時間", 160)),
                "max_monthly_night_shifts": int(row.get("月間夜勤上限", 8)),
                "approved_leave_dates": leave_dates,
                "suspended_dates": suspended_dates,
                "shift_preferences": preferences,
            }
        )
    return nurses


def _parse_demands(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    demands: List[Dict[str, Any]] = []
    for row in rows:
        day = str(row.get("日付", "")).strip()
        if not day:
            continue

        day_count = int(row.get("日勤必要人数", 0))
        evening_count = int(row.get("遅番必要人数", 0))
        night_count = int(row.get("夜勤必要人数", 0))
        ward = str(row.get("病棟", "")).strip() or None
        requires_experienced = bool(row.get("リーダー必要", False))
        is_support = bool(row.get("支援シフト", False))
        holiday = bool(row.get("祝日", False))

        if day_count > 0:
            demands.append(
                {
                    "date": day,
                    "shift_type": "day",
                    "required_count": day_count,
                    "ward": ward,
                    "requires_experienced": requires_experienced,
                    "is_support": is_support,
                    "holiday": holiday,
                }
            )
        if evening_count > 0:
            demands.append(
                {
                    "date": day,
                    "shift_type": "evening",
                    "required_count": evening_count,
                    "ward": ward,
                    "requires_experienced": requires_experienced,
                    "is_support": is_support,
                    "holiday": holiday,
                }
            )
        if night_count > 0:
            night_row: Dict[str, Any] = {
                "date": day,
                "shift_type": "night",
                "required_count": night_count,
                "ward": ward,
                "requires_experienced": requires_experienced,
                "is_support": is_support,
                "holiday": holiday,
            }
            if bool(row.get("夜勤ICU必須", True)):
                night_row["required_skills"] = ["icu"]
            demands.append(night_row)
    return demands


def _weekday_label(day: str) -> str:
    week = ["月", "火", "水", "木", "金", "土", "日"]
    dt = datetime.strptime(day, "%Y-%m-%d")
    return week[dt.weekday()]


def _build_calendar(assignments: List[Dict[str, Any]], nurses: List[Dict[str, Any]], days: List[str]) -> Dict[str, Dict[str, str]]:
    calendar: Dict[str, Dict[str, str]] = {day: {} for day in days}
    for assignment in assignments:
        day = assignment["date"]
        nurse_id = assignment["nurse_id"]
        shift_type = assignment["shift_type"]
        if day in calendar:
            calendar[day][nurse_id] = shift_type
    for day in calendar:
        for nurse in nurses:
            calendar[day].setdefault(nurse["id"], "-")
    return calendar


def _render_kpi(result: Dict[str, Any], nurses: List[Dict[str, Any]], demands: List[Dict[str, Any]]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    total_assignments = len(result.get("assignments", []))
    total_demand = sum(item.get("required_count", 0) for item in demands)
    violations = len(result.get("soft_violations", []))
    avg = round(total_assignments / max(len(nurses), 1), 2)
    c1.metric("割当件数", total_assignments)
    c2.metric("必要件数", total_demand)
    c3.metric("希望違反件数", violations)
    c4.metric("1人あたり平均勤務", avg)


def _render_calendar(calendar: Dict[str, Dict[str, str]], nurses: List[Dict[str, Any]]) -> None:
    nurse_ids = [nurse["id"] for nurse in nurses]
    header_cells = "".join([f"<th>{nurse_id}</th>" for nurse_id in nurse_ids])
    rows_html = []

    for day in sorted(calendar.keys()):
        weekday = _weekday_label(day)
        date_cell = f"<td class='cell-date'>{day}<br><span class='weekday'>{weekday}</span></td>"
        cells = []
        for nurse_id in nurse_ids:
            shift = calendar[day].get(nurse_id, "-")
            label = SHIFT_LABELS.get(shift, shift)
            css = SHIFT_CLASS.get(shift, "shift-empty")
            cells.append(f"<td><span class='pill {css}'>{label}</span></td>")
        rows_html.append(f"<tr>{date_cell}{''.join(cells)}</tr>")

    table_html = f"""
    <div class="calendar-wrap">
      <table class="calendar-table">
        <thead>
          <tr>
            <th>日付</th>
            {header_cells}
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def _render_legend() -> None:
    st.markdown(
        """
        <div class="legend-wrap">
          <span class="pill shift-day">日勤</span>
          <span class="pill shift-evening">遅番</span>
          <span class="pill shift-night">夜勤</span>
          <span class="pill shift-empty">休み</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_assignment_matrix(
    calendar: Dict[str, Dict[str, str]], nurses: List[Dict[str, Any]], days: List[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    nurse_ids = [nurse["id"] for nurse in nurses]
    for day in days:
        row: Dict[str, Any] = {"日付": day, "曜日": _weekday_label(day)}
        for nurse_id in nurse_ids:
            shift_id = calendar.get(day, {}).get(nurse_id, "-")
            row[nurse_id] = SHIFT_LABELS.get(shift_id, "休み" if shift_id == "-" else shift_id)
        rows.append(row)
    return rows


def _matrix_to_assignments(
    matrix_rows: List[Dict[str, Any]], nurses: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    assignments: List[Dict[str, str]] = []
    nurse_ids = [nurse["id"] for nurse in nurses]
    for row in matrix_rows:
        day = str(row.get("日付", "")).strip()
        if not day:
            continue
        for nurse_id in nurse_ids:
            value = str(row.get(nurse_id, "休み")).strip()
            if value in ("", "休み", "-"):
                continue
            shift_id = SHIFT_LABEL_TO_ID.get(value)
            if shift_id is None:
                continue
            assignments.append(
                {
                    "date": day,
                    "shift_type": shift_id,
                    "nurse_id": nurse_id,
                }
            )
    return assignments


def _build_manual_check(
    assignments: List[Dict[str, str]], demands: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    actual: Dict[tuple[str, str], int] = {}
    required: Dict[tuple[str, str], int] = {}
    for row in demands:
        key = (row["date"], row["shift_type"])
        required[key] = required.get(key, 0) + int(row["required_count"])
    for row in assignments:
        key = (row["date"], row["shift_type"])
        actual[key] = actual.get(key, 0) + 1

    all_keys = sorted(set(required.keys()) | set(actual.keys()))
    checks: List[Dict[str, Any]] = []
    for date_key, shift_type in all_keys:
        req = required.get((date_key, shift_type), 0)
        act = actual.get((date_key, shift_type), 0)
        checks.append(
            {
                "日付": date_key,
                "勤務": SHIFT_LABELS.get(shift_type, shift_type),
                "必要人数": req,
                "割当人数": act,
                "差分": act - req,
            }
        )
    return checks


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
          --bg-main: #f7f3ed;
          --bg-card: #ffffff;
          --line: #e5ddd2;
          --text-main: #111111;
          --text-sub: #3f4650;
          --accent: #2563eb;
          --accent-strong: #1d4ed8;
          --accent-soft: #eef3ff;
          --day: #dff2ff;
          --day-text: #114a72;
          --evening: #fff1dd;
          --evening-text: #7a4a00;
          --night: #eee8ff;
          --night-text: #49308f;
          --empty: #f2f2f2;
          --empty-text: #707782;
          --shadow: 0 10px 30px rgba(31, 35, 40, 0.07);
        }

        .stApp {
          font-family: "Inter", "Avenir Next", "Segoe UI", "Hiragino Kaku Gothic ProN", "Yu Gothic UI", "Yu Gothic", "Meiryo", sans-serif;
          background:
            radial-gradient(circle at 10% 10%, #fff8eb 0%, rgba(255, 248, 235, 0) 38%),
            radial-gradient(circle at 86% 14%, #f3ecff 0%, rgba(243, 236, 255, 0) 34%),
            linear-gradient(180deg, #faf7f2 0%, #f3eee7 100%),
            var(--bg-main);
          color: var(--text-main);
          line-height: 1.6;
        }

        section[data-testid="stMain"] {
          color: var(--text-main);
        }

        section[data-testid="stMain"] p,
        section[data-testid="stMain"] li,
        section[data-testid="stMain"] label,
        section[data-testid="stMain"] .stCaption {
          color: var(--text-main) !important;
        }

        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #182230 0%, #121a25 100%);
          border-right: 1px solid #2c3b4f;
        }

        section[data-testid="stSidebar"] * {
          color: #f3f6fb !important;
        }

        section[data-testid="stSidebar"] .stCaption {
          color: #cdd8e8 !important;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
          color: #1f2937 !important;
          font-weight: 700 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
          color: #111827 !important;
          font-weight: 800 !important;
        }

        .hero {
          background: linear-gradient(160deg, #ffffff 0%, #fbfbfa 100%);
          border: 1px solid var(--line);
          border-radius: 22px;
          padding: 30px 32px;
          box-shadow: var(--shadow);
          margin-bottom: 22px;
          animation: rise-in 320ms ease-out;
        }

        .hero h1 {
          margin: 0 0 6px;
          font-size: 32px;
          line-height: 1.2;
          letter-spacing: 0.01em;
          font-weight: 800;
          color: #111111;
        }

        .hero p {
          margin: 0;
          color: #111111;
          font-size: 15px;
        }

        .section-box {
          background: var(--bg-card);
          border: 1px solid var(--line);
          border-radius: 20px;
          padding: 20px;
          box-shadow: var(--shadow);
          margin-bottom: 22px;
          animation: rise-in 280ms ease-out;
        }

        .calendar-wrap {
          background: var(--bg-card);
          border: 1px solid var(--line);
          border-radius: 20px;
          box-shadow: var(--shadow);
          padding: 14px;
          overflow-x: auto;
          margin-top: 8px;
          margin-bottom: 20px;
        }

        .calendar-table {
          width: 100%;
          border-collapse: collapse;
          min-width: 880px;
        }

        .calendar-table th {
          background: var(--accent-soft);
          text-align: center;
          padding: 12px 10px;
          border-bottom: 1px solid var(--line);
          font-size: 13px;
          color: #111111;
          font-weight: 600;
        }

        .calendar-table td {
          text-align: center;
          padding: 12px 8px;
          border-bottom: 1px solid var(--line);
          font-size: 13px;
        }

        .cell-date {
          font-weight: 700;
          color: #111111;
          background: #fbfbfb;
        }

        .weekday {
          font-size: 11px;
          color: #111111;
        }

        .pill {
          display: inline-block;
          min-width: 56px;
          padding: 5px 10px;
          border-radius: 999px;
          font-weight: 700;
          font-size: 12px;
          letter-spacing: 0.01em;
        }

        .shift-day { background: var(--day); color: var(--day-text); }
        .shift-evening { background: var(--evening); color: var(--evening-text); }
        .shift-night { background: var(--night); color: var(--night-text); }
        .shift-empty { background: var(--empty); color: var(--empty-text); }

        .legend-wrap {
          display: flex;
          gap: 10px;
          margin: 8px 0 16px;
          flex-wrap: wrap;
        }

        .legend-wrap .pill {
          color: #111111 !important;
        }

        .stTextInput input, .stNumberInput input, .stDateInput input {
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #fcfbf8 !important;
          color: var(--text-main) !important;
          -webkit-text-fill-color: var(--text-main) !important;
          font-weight: 500;
        }

        .stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus {
          border-color: var(--accent);
          box-shadow: 0 0 0 2px rgba(11, 94, 215, 0.18);
        }

        div[data-testid="stDateInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
          background: #fffdf8 !important;
          color: var(--text-main) !important;
          -webkit-text-fill-color: var(--text-main) !important;
          border-color: var(--line) !important;
        }

        section[data-testid="stMain"] div[data-testid="stDateInput"] input,
        section[data-testid="stMain"] div[data-testid="stNumberInput"] input {
          background: #111111 !important;
          color: #ffffff !important;
          -webkit-text-fill-color: #ffffff !important;
          border-color: #2f2f2f !important;
        }

        section[data-testid="stMain"] div[data-testid="stDateInput"] label,
        section[data-testid="stMain"] div[data-testid="stNumberInput"] label {
          color: #111111 !important;
          font-weight: 800 !important;
          font-size: 16px !important;
        }

        div[data-testid="stDateInput"] svg,
        div[data-testid="stNumberInput"] svg {
          fill: #3f4650 !important;
        }

        section[data-testid="stMain"] div[data-testid="stDateInput"] svg,
        section[data-testid="stMain"] div[data-testid="stNumberInput"] svg {
          fill: #ffffff !important;
        }

        div[data-testid="stDateInput"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stTextInput"] label {
          color: var(--text-main) !important;
          font-weight: 700;
        }

        .stButton > button {
          background: linear-gradient(120deg, var(--accent), var(--accent-strong));
          color: white;
          border: 0;
          border-radius: 12px;
          padding: 10px 16px;
          font-weight: 600;
          box-shadow: 0 8px 20px rgba(37, 99, 235, 0.22);
        }

        .stButton > button:hover { filter: brightness(1.07); }

        .stDownloadButton > button {
          background: #f8fafc !important;
          color: #111111 !important;
          border: 1px solid #d2dae5 !important;
          border-radius: 12px !important;
          padding: 10px 16px !important;
          font-weight: 600 !important;
          box-shadow: 0 8px 20px rgba(31, 40, 51, 0.10);
        }

        .stDownloadButton > button:hover {
          filter: brightness(0.98);
        }

        div[data-testid="stDataFrame"] * {
          color: #111111 !important;
        }

        div[data-testid="stExpander"] summary {
          background: #f8fafc !important;
          color: #111111 !important;
          border: 1px solid #d8e1eb !important;
          border-radius: 12px !important;
          padding: 8px 12px !important;
          font-size: 20px !important;
          font-weight: 800 !important;
        }

        div[data-testid="stExpander"] summary * {
          color: #111111 !important;
        }

        div[data-testid="stExpander"] .stCaption,
        div[data-testid="stExpander"] .stMarkdown,
        div[data-testid="stExpander"] .stMarkdown *,
        div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] li,
        div[data-testid="stExpander"] label,
        div[data-testid="stExpander"] span,
        div[data-testid="stExpander"] div {
          color: #111111 !important;
          font-size: 15px !important;
        }

        @keyframes rise-in {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 860px) {
          .hero h1 { font-size: 25px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="看護師シフト最適化", layout="wide")
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
          <h1>看護師シフト最適化</h1>
          <p>人員条件を調整し、1週間のシフトを自動作成。結果はカレンダー形式で直感的に確認できます。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("実行設定")
        time_limit = st.number_input(
            "計算時間上限（秒）",
            min_value=1,
            value=20,
            key="ui_time_limit",
            help="最適化の探索を打ち切る時間です。長いほど良い解を見つけやすいですが、待ち時間が増えます。",
        )
        random_seed = st.number_input(
            "乱数シード",
            min_value=0,
            value=11,
            key="ui_random_seed",
            help="同じ入力と同じシードで実行したとき、同じ結果になりやすくするための値です。",
        )
        fairness_penalty = st.slider(
            "公平性重み",
            0.0,
            10.0,
            1.5,
            0.5,
            key="ui_fairness_penalty",
            help="大きいほど勤務回数の偏りを抑えます。0は公平性を考慮しません。",
        )
        day_off_penalty = st.slider(
            "希望休違反ペナルティ",
            0.0,
            20.0,
            12.0,
            1.0,
            key="ui_day_off_penalty",
            help="大きいほど希望休違反を避けます。高すぎると他制約との両立が難しくなる場合があります。",
        )
        skill_penalty = st.slider(
            "スキル優先ペナルティ",
            0.0,
            20.0,
            10.0,
            1.0,
            key="ui_skill_penalty",
            help="大きいほど必要スキル保持者を優先します。0はスキル優先なしです。",
        )
        max_shifts_per_nurse = st.number_input(
            "週あたり最大勤務回数（月〜日）",
            min_value=1,
            max_value=7,
            value=4,
            key="ui_max_shifts_per_nurse",
            help="この上限は月曜〜日曜の各週ごとに適用されます。",
        )
        rest_after_night = st.number_input(
            "夜勤後休日日数", min_value=0, max_value=3, value=1, key="ui_rest_after_night"
        )
        enforce_exact_demand = _toggle(
            "必要人数を厳密一致",
            value=True,
            key="ui_enforce_exact_demand",
            help="ONにすると必要人数ぴったりで割り当てます。OFFにすると必要人数以上でも許容します。",
        )
        enforce_required_skills_hard = _toggle(
            "必要スキル制約をHard適用",
            value=True,
            key="ui_enforce_required_skills_hard",
            help="ONにすると必要スキルを持たない看護師はそのシフトに一切入れません。",
        )

        with st.expander("高度なHard制約", expanded=False):
            max_consecutive_days = st.number_input(
                "最大連続勤務日数",
                min_value=1,
                max_value=14,
                value=5,
                key="ui_max_consecutive_days",
                help="連続勤務できる上限日数です。小さいほど負荷は下がりますが、実行不能になりやすくなります。",
            )
            max_monthly_hours_rule = st.number_input(
                "月間勤務時間上限（共通）",
                min_value=40,
                max_value=220,
                value=160,
                key="ui_max_monthly_hours_rule",
                help="1か月あたりの勤務時間上限です。超えないように自動調整します。",
            )
            max_monthly_night_rule = st.number_input(
                "月間夜勤上限（共通）",
                min_value=0,
                max_value=20,
                value=8,
                key="ui_max_monthly_night_rule",
                help="1か月あたりの夜勤回数上限です。夜勤偏りや過重労働を防ぎます。",
            )
            min_rest_days_per_week = st.number_input(
                "週あたり最低休日日数",
                min_value=0,
                max_value=3,
                value=1,
                key="ui_min_rest_days_per_week",
                help="1週間（月〜日）で最低限確保したい休日日数です。",
            )

        with st.expander("高度なSoft制約", expanded=False):
            preference_penalty = st.slider(
                "希望シフト優先ペナルティ",
                0.0,
                10.0,
                2.0,
                0.5,
                key="ui_preference_penalty",
                help="大きいほど希望に沿わない割当を減らします。",
            )
            consecutive_night_penalty = st.slider(
                "連続夜勤抑制ペナルティ",
                0.0,
                10.0,
                3.0,
                0.5,
                key="ui_consecutive_night_penalty",
                help="大きいほど連続夜勤を避ける方向に調整します。",
            )
            night_fairness_penalty = st.slider(
                "夜勤平準化ペナルティ",
                0.0,
                10.0,
                1.0,
                0.5,
                key="ui_night_fairness_penalty",
                help="大きいほど夜勤回数の偏りを抑えます。",
            )
            weekend_fairness_penalty = st.slider(
                "週末平準化ペナルティ",
                0.0,
                10.0,
                1.0,
                0.5,
                key="ui_weekend_fairness_penalty",
                help="大きいほど土日勤務の偏りを減らします。",
            )
            holiday_fairness_penalty = st.slider(
                "祝日平準化ペナルティ",
                0.0,
                10.0,
                1.0,
                0.5,
                key="ui_holiday_fairness_penalty",
                help="大きいほど祝日勤務の偏りを減らします。",
            )
            external_usage_penalty = st.slider(
                "外部要員抑制ペナルティ",
                0.0,
                10.0,
                2.0,
                0.5,
                key="ui_external_usage_penalty",
                help="大きいほど外部・応援要員への依存を抑えます。",
            )
            novice_with_experienced_penalty = st.slider(
                "新人+ベテラン同席ペナルティ",
                0.0,
                10.0,
                2.0,
                0.5,
                key="ui_novice_with_experienced_penalty",
                help="大きいほど新人とベテランを同じシフトに配置しやすくなります。",
            )
            abrupt_transition_penalty = st.slider(
                "急変遷移抑制ペナルティ",
                0.0,
                10.0,
                1.0,
                0.5,
                key="ui_abrupt_transition_penalty",
                help="大きいほど急なシフト切替（例: 夜勤→日勤）を避ける方向になります。",
            )

    period_col, demand_col = st.columns([1, 1])
    with period_col:
        start_date = st.date_input("開始日", value=date.today())
    with demand_col:
        days = st.number_input("計画日数", min_value=1, max_value=31, value=7)

    _init_state(start_date, int(days))
    _sync_demands_editor_with_period(start_date, int(days))

    st.markdown(
        """
        <div class="section-box">
          <strong>入力ステップ</strong><br>
          1) 看護師設定 → 2) 日別必要人数 → 3) 実行設定確認 → 4) 最適化実行
        </div>
        """,
        unsafe_allow_html=True,
    )

    action_col1, action_col2 = st.columns([1, 2])
    with action_col1:
        if _button("テンプレートを再生成"):
            _reset_template(start_date, int(days))
            _rerun()
    with action_col2:
        st.caption("テンプレートは仮名10名、希望休、日勤2・遅番1・夜勤1、夜勤ICU必須です。")

    with st.expander("1. 看護師設定", expanded=True):
        st.caption("氏名、ICU対応可否、最大勤務回数、希望休を調整します。")
        nurses_editor_kwargs: Dict[str, Any] = {
            "num_rows": "dynamic",
            "use_container_width": True,
        }
        if hasattr(st, "column_config"):
            nurses_editor_kwargs["column_config"] = {
                "氏名": st.column_config.TextColumn(required=True),
                "ICU": st.column_config.CheckboxColumn(help="夜勤ICU必須シフトの候補になります"),
                "最大勤務回数": st.column_config.NumberColumn(min_value=1, max_value=7, step=1),
                "希望休": st.column_config.TextColumn(help="YYYY-MM-DDをカンマ区切りで入力"),
                "登録看護師": st.column_config.CheckboxColumn(),
                "新人": st.column_config.CheckboxColumn(),
                "ベテラン": st.column_config.CheckboxColumn(),
                "休職中": st.column_config.CheckboxColumn(),
                "支援要員": st.column_config.CheckboxColumn(),
                "外部要員": st.column_config.CheckboxColumn(),
                "所属病棟": st.column_config.TextColumn(),
                "許可勤務区分": st.column_config.TextColumn(help="例: day,evening,night"),
                "月間最大勤務時間": st.column_config.NumberColumn(min_value=40, max_value=220),
                "月間夜勤上限": st.column_config.NumberColumn(min_value=0, max_value=20),
                "承認休暇日": st.column_config.TextColumn(help="YYYY-MM-DDをカンマ区切り"),
                "停止日": st.column_config.TextColumn(help="YYYY-MM-DDをカンマ区切り"),
                "希望シフト(JSON)": st.column_config.TextColumn(
                    help='例: {"2026-02-10":["day","evening"]}'
                ),
            }
        nurses_editor = _data_editor(st.session_state.nurses_editor, **nurses_editor_kwargs)

    with st.expander("2. 日別必要人数", expanded=True):
        st.caption("日勤・遅番・夜勤それぞれの必要人数を日ごとに設定します。")
        demands_editor_kwargs: Dict[str, Any] = {
            "num_rows": "dynamic",
            "use_container_width": True,
        }
        if hasattr(st, "column_config"):
            demands_editor_kwargs["column_config"] = {
                "日付": st.column_config.TextColumn(required=True),
                "日勤必要人数": st.column_config.NumberColumn(min_value=0, step=1),
                "遅番必要人数": st.column_config.NumberColumn(min_value=0, step=1),
                "夜勤必要人数": st.column_config.NumberColumn(min_value=0, step=1),
                "夜勤ICU必須": st.column_config.CheckboxColumn(),
                "病棟": st.column_config.TextColumn(),
                "リーダー必要": st.column_config.CheckboxColumn(),
                "支援シフト": st.column_config.CheckboxColumn(),
                "祝日": st.column_config.CheckboxColumn(),
            }
        demands_editor = _data_editor(st.session_state.demands_editor, **demands_editor_kwargs)

    with st.expander("3. 実行前チェック", expanded=False):
        st.caption("左サイドバーの実行設定（公平性重み、希望休ペナルティ、時間上限など）を確認してください。")
        st.markdown(
            "- 計画期間・人数・必要人数の整合性を確認\n"
            "- 必要人数を厳密一致にする場合は、総必要件数と最大勤務回数を確認\n"
            "- 実行不能が出る場合は最大勤務回数または必要人数を調整"
        )

    run_clicked = _button("最適化を実行", type="primary", use_container_width=True)

    if not run_clicked:
        return

    progress = st.progress(0)
    status_box = _status("入力を整形中", state="running")

    try:
        nurses = _parse_nurses(nurses_editor)
        demands = _parse_demands(demands_editor)
    except Exception as exc:
        status_box.update(state="error", label=f"入力の解釈に失敗しました: {exc}")
        st.stop()

    progress.progress(25)
    status_box.update(label="最適化モデルを作成中")

    payload = {
        "planning_period": {
            "start_date": start_date.isoformat(),
            "days": int(days),
        },
        "nurses": nurses,
        "shift_types": SHIFT_TYPES,
        "demands": demands,
        "rules": {
            "max_shifts_per_nurse": int(max_shifts_per_nurse),
            "rest_after_night_days": int(rest_after_night),
            "night_shift_type": "night",
            "enforce_exact_demand": bool(enforce_exact_demand),
            "enforce_required_skills_hard": bool(enforce_required_skills_hard),
            "max_consecutive_days": int(max_consecutive_days),
            "max_monthly_hours": float(max_monthly_hours_rule),
            "max_monthly_night_shifts": int(max_monthly_night_rule),
            "min_rest_days_per_week": int(min_rest_days_per_week),
            "forbidden_after_night_shift_types": ["day"],
            "abrupt_transition_pairs": [
                ["day", "night"],
                ["night", "day"],
                ["evening", "day"],
            ],
        },
        "weights": {
            "day_off_penalty": float(day_off_penalty),
            "fairness_penalty": float(fairness_penalty),
            "skill_priority_penalty": float(skill_penalty),
            "preference_penalty": float(preference_penalty),
            "consecutive_night_penalty": float(consecutive_night_penalty),
            "night_fairness_penalty": float(night_fairness_penalty),
            "weekend_fairness_penalty": float(weekend_fairness_penalty),
            "holiday_fairness_penalty": float(holiday_fairness_penalty),
            "external_usage_penalty": float(external_usage_penalty),
            "novice_with_experienced_penalty": float(novice_with_experienced_penalty),
            "abrupt_transition_penalty": float(abrupt_transition_penalty),
        },
        "time_limit_seconds": int(time_limit),
        "random_seed": int(random_seed) if random_seed else None,
    }

    problem = ProblemDefinition(
        problem_type="nurse_shift",
        sense="min",
        base_dir=PROJECT_ROOT,
        data_paths={},
        parameters={"nurse_shift": payload},
    )

    policy = load_policy(PROJECT_ROOT / "policy.yaml")
    result = solve_problem(problem, policy)

    progress.progress(70)
    status_box.update(label="結果を表示中")

    if result.get("status") == "error":
        status_box.update(state="error", label="入力エラー")
        issues = result.get("errors", [])
        if issues:
            st.error("入力値に問題があります。以下を修正してください。")
            _dataframe(issues, use_container_width=True)
        else:
            st.error("入力エラーが発生しました。")
        st.stop()

    if result.get("status") == "infeasible":
        status_box.update(state="error", label="実行不能")
        st.error("この条件ではシフトを作成できません。原因と調整案を確認してください。")
        analysis = (
            result.get("constraints_summary", {}).get("analysis", [])
            if isinstance(result.get("constraints_summary", {}), dict)
            else []
        )
        if analysis:
            rows = []
            for idx, item in enumerate(analysis, start=1):
                rows.append(
                    {
                        "No": idx,
                        "原因": item.get("message", ""),
                        "推奨対応": item.get("suggestion", ""),
                        "コード": item.get("code", ""),
                    }
                )
            st.subheader("実行不能の分析結果")
            _dataframe(rows, use_container_width=True)
            st.info("下のボタンで推奨パラメータを自動反映できます。")
            nurse_count = len(nurses)
            for idx, item in enumerate(analysis, start=1):
                code = str(item.get("code", "generic_infeasible"))
                label_map = {
                    "daily_capacity_exceeded": "この日の必要人数を自動調整",
                    "global_capacity_exceeded": "最大勤務回数を +1",
                    "skill_capacity_exceeded": "夜勤ICU必須を緩和",
                    "night_rest_capacity_exceeded": "夜勤後休日日数を -1",
                    "generic_infeasible": "必要人数の厳密一致を OFF",
                }
                button_label = label_map.get(code, "推奨パラメータを適用")
                if _button(f"{idx}. {button_label}", key=f"apply_infeasible_fix_{idx}"):
                    message = _apply_infeasible_recommendation(code, item, nurse_count)
                    st.session_state["auto_fix_message"] = message
                    _rerun()
            if st.session_state.get("auto_fix_message"):
                st.success(st.session_state["auto_fix_message"])
        st.stop()

    progress.progress(100)
    status_box.update(state="complete", label="最適化完了")

    st.subheader("3. 結果サマリー")
    _render_kpi(result, nurses, demands)
    _render_legend()

    dates = _build_dates(start_date, int(days))
    calendar = _build_calendar(result.get("assignments", []), nurses, dates)
    _render_calendar(calendar, nurses)

    st.subheader("看護師ごとの勤務回数")
    counts: Dict[str, int] = {nurse["id"]: 0 for nurse in nurses}
    for item in result.get("assignments", []):
        counts[item["nurse_id"]] = counts.get(item["nurse_id"], 0) + 1
    chart_rows = [{"看護師": name, "勤務回数": count} for name, count in counts.items()]
    _dataframe(chart_rows, use_container_width=True)

    st.subheader("手動調整")
    st.caption("各セルを直接編集できます（休み/日勤/遅番/夜勤）。")
    matrix_rows = _build_assignment_matrix(calendar, nurses, dates)
    matrix_editor_kwargs: Dict[str, Any] = {
        "num_rows": "fixed",
        "use_container_width": True,
        "key": "manual_matrix_editor",
    }
    if hasattr(st, "column_config"):
        column_config: Dict[str, Any] = {
            "日付": st.column_config.TextColumn(disabled=True),
            "曜日": st.column_config.TextColumn(disabled=True),
        }
        for nurse in nurses:
            column_config[nurse["id"]] = st.column_config.SelectboxColumn(
                options=["休み", "日勤", "遅番", "夜勤"],
                required=True,
            )
        matrix_editor_kwargs["column_config"] = column_config

    edited_matrix = _data_editor(matrix_rows, **matrix_editor_kwargs)
    edited_assignments = _matrix_to_assignments(edited_matrix, nurses)

    st.subheader("手動調整後の充足チェック")
    _dataframe(_build_manual_check(edited_assignments, demands), use_container_width=True)

    st.subheader("エクスポート")
    st.download_button(
        "JSONをダウンロード",
        data=json.dumps({"assignments": edited_assignments, "metadata": result.get("metadata", {})}, ensure_ascii=False, indent=2),
        file_name="nurse_shift_result.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
