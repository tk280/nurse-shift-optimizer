# 要件定義

## 概要
本リファクタは `optimizer/shift_model.py` に混在している制約ロジックを分離し、
モデル初期化・制約定義・目的関数集約の責務を明確化することを目的とする。

## 要件

### Requirement 1: モジュール分割
**Objective:** As a 開発者, I want 制約ロジックを分割されたモジュールに移動したい, so that 責務が明確になり保守性が向上する

#### Acceptance Criteria
1. When リファクタ完了後, the system shall `optimizer/model/` に変数定義・モデル初期化ロジックを配置する
2. When リファクタ完了後, the system shall `optimizer/constraints/hard/` にハード制約の実装を配置する
3. When リファクタ完了後, the system shall `optimizer/constraints/soft/` にソフト制約の実装を配置する
4. When リファクタ完了後, the system shall `optimizer/objectives/` にペナルティ集約・目的関数構成を配置する
5. The system shall `optimizer/shift_model.py` をオーケストレーション層として残し、制約式を直接記述しない
6. The system shall model / `optimizer/shift_model.py` に制約を直接記述しない

### Requirement 2: 依存方向の制約
**Objective:** As a 開発者, I want 依存方向を一方向に制御したい, so that 循環依存を防止できる

#### Acceptance Criteria
1. The system shall `constraints -> model` の依存を許可する
2. The system shall `model -> constraints` の依存を禁止する
3. The system shall `shift_model.py` でモデル・制約・目的関数を組み合わせる

### Requirement 3: 振る舞い互換性
**Objective:** As a 運用担当者, I want 既存入力に対する結果を維持したい, so that リファクタ後も業務影響が発生しない

#### Acceptance Criteria
1. Given 同一の入力JSONと `policy.yaml`, the system shall 可否判定を同等に保つ
2. Given 同一の入力JSONと `policy.yaml`, the system shall 目的関数値が許容範囲内で一致する
3. If ソフト制約に差異がある場合, then the system shall 許容誤差を明示した比較で評価できる
4. The system shall 既存入出力（problem JSON -> dict result）を維持する
5. The system shall 既存CLIのI/Fを維持する

### Requirement 4: 既存テストの維持
**Objective:** As a 開発者, I want 既存のテストを壊さずに通したい, so that リグレッションを防げる

#### Acceptance Criteria
1. The system shall 既存の `pytest` テストを全て通過させる
2. The system shall 既存テストが維持される、または同等の回帰テストを追加する
