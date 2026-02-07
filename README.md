# 看護師シフト最適化

OR-Tools を使って看護師シフトを最適化するプロジェクトです。Streamlit Web UI で条件を入力し、制約を満たすシフトを生成できます。

## 主な機能
- Hard 制約に基づくシフト最適化
- Soft 制約（公平性・希望・スキル優先）の重み調整
- 実行不能時の原因分析と推奨修正
- カレンダー/表形式での結果表示
- JSON エクスポート

## セットアップ
```bash
poetry install
```

## Web UI の起動
```bash
poetry run streamlit run scripts/run_shift_web_ui.py
```

## テスト実行
```bash
poetry run pytest -q
```

## 構成
- `optimizer/`: 最適化ロジック
- `scripts/`: 実行スクリプト（CLI/UI）
- `tests/`: テスト
- `policy.yaml`: 実行ポリシー
- `.kiro/specs/看護師のシフト最適化/`: 要件・設計・タスク
