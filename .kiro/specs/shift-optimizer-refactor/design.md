# 設計書

## 概要
本リファクタは `optimizer/shift_model.py` の制約ロジックを分離し、
モデル初期化・制約定義・目的関数集約を分割した構成に変更する。
依存方向を一方向に保ち、既存 I/F と挙動を維持する。

## 設計方針
- モデル初期化は `optimizer/model/` に集約
- ハード制約は `optimizer/constraints/hard/` に集約
- ソフト制約は `optimizer/constraints/soft/` に集約
- 目的関数（ペナルティ集約）は `optimizer/objectives/` に集約
- `optimizer/shift_model.py` はオーケストレーションのみ
- 依存方向は `constraints -> model` のみ許可

## 目標構成
```
optimizer/
  model/
    __init__.py
    base.py           # solver/variable/indices
    assignments.py    # assignment vars + helpers
  constraints/
    hard/
      __init__.py
      staffing.py     # demand/coverage
      workload.py     # max shifts, rest rules
      compliance.py   # license/ward/support/novice
    soft/
      __init__.py
      preferences.py  # requests/violations
      fairness.py     # fairness penalties
      transitions.py  # shift transition penalties
  objectives/
    __init__.py
    aggregate.py      # combine penalty vars into objective
  shift_model.py      # orchestration only
```

## 主要責務

### optimizer/model
- solver 初期化
- 日付・シフト種別・看護師のインデックス管理
- 割当変数と共通補助変数の生成
- モデルの参照用データクラスの提供
  - API: `build_model(data, policy) -> (model, vars)`
  - `vars` は named structure（例: `x[n,d,s]`）を保持

### optimizer/constraints/hard
- 需要充足、夜勤条件、病棟/支援要員などの必須制約
- 週次/連続勤務/休養日など勤務整合性制約
- 法令/上限制約

### optimizer/constraints/soft
- 希望休/希望シフトの違反ペナルティ
- 公平性（夜勤/週末/祝日）ペナルティ
- 新人同席/急な遷移抑制などのペナルティ
  - API: `add_hard_constraints(model, vars, data, policy) -> None`
  - API: `add_soft_constraints(model, vars, data, policy) -> list[PenaltyTerm]`
  - `PenaltyTerm` は `(name, weight, expr or var)` を持つ

### optimizer/objectives
- 各ソフト制約のペナルティ変数を統合
- 目的関数の重み適用
  - API: `build_objective(model, penalty_terms) -> None`（Minimize）

### optimizer/shift_model.py
- 入力正規化
- model/constraints/objectives の組み立て
- solve 実行と結果整形
- I/F は既存と同一

## 依存ルール
- `constraints` は `model` に依存してよい
- `model` は `constraints` を参照しない
- `objectives` は `constraints` から得たペナルティ変数を受け取る
- `solve`/オーケストレーションは model と constraints を呼ぶのみ

## 互換性戦略
- 既存 `solve_problem` のシグネチャ維持
- problem JSON -> dict result のスキーマ変更なし
- CLI 実行（`scripts/run_shift_demo.py`）のI/F維持
- 目的関数値の比較は許容誤差を設ける

## テスト
- 既存 `tests/test_shift_model.py` を維持
- 追加でリファクタ互換性テストを用意（必要に応じて）
