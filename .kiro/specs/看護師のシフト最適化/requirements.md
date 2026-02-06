# Requirements Document

## Introduction
本仕様は、病棟の看護師シフトを公平かつ安全に自動編成する最適化機能を定義する。対象期間内の勤務区分（早番・日勤・遅番・夜勤）について、法令・院内ルール・希望休・必要スキルを満たしつつ、充足率と公平性を最大化する。利用者は Web UI から入力、実行、結果確認、手動調整を行える。

## Requirements

### Requirement 1: シフト計画データの入力と検証
**Objective:** As a 師長, I want 最適化に必要な入力データを一貫形式で登録できる, so that 計算前に欠損や矛盾を防止できる

#### Acceptance Criteria
1. When 計画期間・看護師一覧・勤務区分定義・必要人数定義が登録されたとき, the Shift Optimizer shall 入力データを内部モデルに変換する
2. If 必須項目が欠損しているとき, then the Shift Optimizer shall 不足項目と対象レコードをエラーとして返す
3. If 同一看護師に対して同一日時の重複勤務が入力されたとき, then the Shift Optimizer shall 重複内容を検証エラーとして返す
4. The Shift Optimizer shall 検証結果を機械可読な構造で返却する

### Requirement 2: ハード制約の充足
**Objective:** As a 病棟管理者, I want 法令と運用上必須の制約を必ず満たしたシフト案を得たい, so that 安全性とコンプライアンスを確保できる

#### Acceptance Criteria
1. When 最適化を実行したとき, the Shift Optimizer shall 各日時・勤務区分の必要人数下限を満たす割当のみを有効解として扱う
2. While 各看護師の月間上限勤務日数が定義されている, the Shift Optimizer shall 上限を超える割当を禁止する
3. While 夜勤後休息ルールが有効である, the Shift Optimizer shall 夜勤直後に禁止された連続勤務を割り当てない
4. If すべてのハード制約を満たす解が存在しないとき, then the Shift Optimizer shall 非実行可能である理由を返す

### Requirement 3: 希望と公平性を考慮した最適化
**Objective:** As a 看護師, I want 希望休や勤務バランスが考慮された割当を受けたい, so that 業務負荷の偏りを軽減できる

#### Acceptance Criteria
1. When 希望休が提出されているとき, the Shift Optimizer shall 希望休違反を最小化する目的関数を適用する
2. When 勤務区分ごとの公平性重みが設定されているとき, the Shift Optimizer shall 看護師間の勤務回数偏差を最小化する
3. Where 優先スキル制約が有効なとき, the Shift Optimizer shall 必要スキル保持者を優先して割り当てる
4. The Shift Optimizer shall ハード制約を優先しつつソフト制約の最適値を算出する

### Requirement 4: 結果出力と説明可能性
**Objective:** As a 師長, I want 作成されたシフト案の根拠を確認できる, so that 現場説明と手動調整を容易にできる

#### Acceptance Criteria
1. When 最適化が完了したとき, the Shift Optimizer shall 日時・看護師・勤務区分を含むシフト表を出力する
2. The Shift Optimizer shall 目的関数値と主要制約の充足状況サマリーを出力する
3. If ソフト制約違反が残るとき, then the Shift Optimizer shall 違反件数と対象を出力する
4. The Shift Optimizer shall 出力をJSON形式で返却する

### Requirement 5: 実行制御と再現性
**Objective:** As a 運用担当者, I want 計算時間と実行条件を制御できる, so that 定時運用で安定して利用できる

#### Acceptance Criteria
1. When 実行時に時間上限が指定されたとき, the Shift Optimizer shall 指定時間内で最良解を返す
2. If 乱数シードが指定されたとき, then the Shift Optimizer shall 同一入力に対して再現可能な結果を返す
3. When 最適化パラメータが設定されたとき, the Shift Optimizer shall 設定値を実行メタデータとして出力する
4. The Shift Optimizer shall 実行ステータスを成功・非実行可能・エラーのいずれかで返す

### Requirement 6: Web UI でのシフト計画操作
**Objective:** As a 師長, I want ブラウザ画面からシフト計画を作成・実行・確認できる, so that 専門知識なしで日次運用できる

#### Acceptance Criteria
1. When 利用者が計画条件を入力して実行を指示したとき, the Shift Optimizer shall Web UI 上で最適化ジョブを開始する
2. While 最適化ジョブが実行中のとき, the Shift Optimizer shall Web UI 上で進行状態を表示する
3. When 最適化が完了したとき, the Shift Optimizer shall Web UI 上にシフト表と制約サマリーを表示する
4. If 入力エラーまたは実行不能が発生したとき, then the Shift Optimizer shall Web UI 上で修正可能なエラー情報を表示する
5. The Shift Optimizer shall Web UI 上で手動調整後のシフトをJSONとしてエクスポートできる

### Requirement 7: 高度なHard制約の充足
**Objective:** As a 病棟管理者, I want 安全性と法令を満たす詳細制約を必須条件として適用したい, so that 現場運用可能なシフトのみを採用できる

#### Acceptance Criteria
1. When シフトを作成するとき, the Shift Optimizer shall 各シフトに必要最小人数を割り当てる
2. Where 夜勤シフトを割り当てるとき, the Shift Optimizer shall 登録看護師を最低1名含める
3. Where 病棟が専門スキルを要求するとき, the Shift Optimizer shall 必要資格を持つ看護師のみを割り当てる
4. If 看護師が新人区分であるとき, then the Shift Optimizer shall not 新人のみで夜勤を構成する
5. While 日次スケジュールを生成しているとき, the Shift Optimizer shall not 看護師1人に1日2シフト以上を割り当てる
6. If 看護師が夜勤に入ったとき, then the Shift Optimizer shall not 翌日に禁止された早番を割り当てる
7. Where 連続勤務日数を計算するとき, the Shift Optimizer shall not 最大連続勤務日数を超える
8. While 月間労働時間を計算しているとき, the Shift Optimizer shall 法定上限を超えないようにする
9. Where 夜勤が割り当てられるとき, the Shift Optimizer shall 月間夜勤回数の上限を超えないようにする
10. When 週間スケジュールを作成するとき, the Shift Optimizer shall 各看護師に週1日以上の完全休養日を確保する
11. If 承認済み希望休がある日付のとき, then the Shift Optimizer shall 当該日にシフトを割り当てない
12. Where 看護師が休職中または停止中のとき, the Shift Optimizer shall シフトを割り当てない
13. If 看護師がパートまたは短時間勤務区分のとき, then the Shift Optimizer shall 許可時間内の勤務のみを割り当てる
14. Where シフトにリーダー要件があるとき, the Shift Optimizer shall 経験者を最低1名割り当てる
15. If 複数名シフトを割り当てるとき, then the Shift Optimizer shall not 新人のみの組み合わせにしない
16. Where 看護師が特定病棟所属であるとき, the Shift Optimizer shall 当該病棟のシフトのみに割り当てる
17. If 看護師が支援要員として登録されていないとき, then the Shift Optimizer shall not 支援シフトに割り当てない
18. When 夜勤を割り当てるとき, the Shift Optimizer shall 事前定義された遷移パターンを適用する

### Requirement 8: 高度なSoft制約の最適化
**Objective:** As a 看護師長, I want 公平性と満足度を高める運用上の望ましい条件を最適化したい, so that 長期的に持続可能な勤務体制を維持できる

#### Acceptance Criteria
1. Where 希望シフトが提出されているとき, the Shift Optimizer should 可能な限り希望を満たす
2. If 連続夜勤が発生する場合, then the Shift Optimizer should その頻度を最小化する
3. Where 可能であるとき, the Shift Optimizer should 連休を確保する方向で割り当てる
4. While 夜勤を割り当てるとき, the Shift Optimizer should 看護師間で均等に分散する
5. Where 月間負荷を計算するとき, the Shift Optimizer should 看護師間の負荷分散のばらつきを最小化する
6. If 新人がシフトに入るとき, then the Shift Optimizer should 同シフトに経験者を配置する
7. Where 同一ペアの組み合わせが繰り返されるとき, the Shift Optimizer should 過度な固定化を避ける
8. If 看護師が夜勤を終えたとき, then the Shift Optimizer should 可能な限り翌日を休みにする
9. Where 日間でシフト種別が変わるとき, the Shift Optimizer should 急激な遷移を避ける
10. While 週末シフトを編成するとき, the Shift Optimizer should 週末勤務を公平に配分する
11. Where 祝日シフトを編成するとき, the Shift Optimizer should 祝日勤務を公平に配分する
12. If 外部応援や支援要員を利用する場合, then the Shift Optimizer should その利用量を最小化する
