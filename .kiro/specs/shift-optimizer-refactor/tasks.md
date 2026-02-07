# 実装タスク

- [ ] 1. 新規フォルダ・ファイル作成
  - [ ] `optimizer/model/` を作成
  - [ ] `optimizer/constraints/hard/` を作成
  - [ ] `optimizer/constraints/soft/` を作成
  - [ ] `optimizer/objectives/` を作成

- [ ] 2. 変数定義の分離
  - [ ] `shift_model.py` から変数定義・インデックス生成を切り出し
  - [ ] `optimizer/model/` に `build_model(data, policy) -> (model, vars)` を実装
  - [ ] `vars` は named structure（例: `x[n,d,s]`）で保持

- [ ] 3. Hard制約の分離
  - [ ] `shift_model.py` からハード制約式を切り出し
  - [ ] `optimizer/constraints/hard/` に移動
  - [ ] `add_hard_constraints(model, vars, data, policy) -> None` を実装

- [ ] 4. Soft制約の分離
  - [ ] `shift_model.py` からソフト制約式を切り出し
  - [ ] `optimizer/constraints/soft/` に移動
  - [ ] `add_soft_constraints(model, vars, data, policy) -> list[PenaltyTerm]` を実装

- [ ] 5. 目的関数の分離
  - [ ] `shift_model.py` から目的関数集約を切り出し
  - [ ] `optimizer/objectives/` に `build_objective(model, penalty_terms)` を実装

- [ ] 6. オーケストレーションの再構成
  - [ ] `shift_model.py` を orchestration に整理
  - [ ] 呼び出し順序を `build_model -> add_hard_constraints -> add_soft_constraints -> build_objective -> solve` に統一
  - [ ] 既存の入力/出力スキーマと CLI I/F を維持

- [ ] 7. 回帰テスト追加
  - [ ] 同一入力で feasible 判定が変わらないテストを追加
  - [ ] 目的関数値が同一 or 許容差内で一致するテストを追加
  - [ ] 許容差の定義（例: `abs(diff) <= 1e-6`）を明記

- [ ] 8. import 整理と循環参照チェック
  - [ ] `constraints -> model` 依存のみ許可
  - [ ] `model -> constraints` 禁止を確認
  - [ ] import の循環参照がないことを確認
