# MyVoiceger 🎤

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

MyVoicegerは、RVC（Retrieval-based Voice Conversion）とUVR5を使用した高性能音声合成・変換Webアプリケーションです。楽曲からの音声分離、AI分析、カバーアート生成機能を搭載した包括的な音声処理プラットフォームです。

## ✨ 主な機能

- **🎵 音声分離**: UVR5を使用した楽曲からの高精度音声分離
- **🔄 音声変換**: RVCモデルによる高品質音声変換
- **🤖 AI分析**: Google Gemini APIを使用した歌詞分析と理解
- **🎨 カバーアート生成**: AIによる自動カバーアート作成
- **🌐 Web UI**: 直感的なFlaskベースのユーザーインターフェース

## 🚀 クイックスタート

### 要件
- Python 3.8+
- pip
- Git

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/myvoiceger.git
cd myvoiceger

# 依存関係のインストール
pip install -r requirements.txt

# 環境変数の設定（オプション）
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 実行

```bash
# アプリケーションの起動
python app.py

# ブラウザでアクセス
# http://localhost:5556
```

## 📋 使用方法

### 基本ワークフロー

1. **音声分離 Tab**: 楽曲ファイルをアップロードし、音声とInstrumentalを分離
2. **音声変換 Tab**: RVCモデルを選択肢、適切なパラメータを設定して音声変換を実行
3. **後処理 Tab**: エフェクト、音量調整、最終ミックスを適用
4. **AI分析 Tab**: 歌詞分析とカバーアート生成を実行

### 推奨設定

#### 品質向上のためのヒント
- **ターゲット音声**: 15-30秒の高品質なクリーンな音声を使用
- **楽曲ファイル**: 44.1kHz、16bit以上の高品質なファイルを推奨
- **Pitch Shift**: ±3半音以内で調整推奨
- **Formant Shift**: 0.8-1.2の範囲で微調整推奨

## 🛠️ 技術スタック

| 分野 | 技術 |
|------|------|
| **フロントエンド** | Gradio, HTML/CSS, JavaScript |
| **バックエンド** | Python Flask |
| **音声処理** | librosa, pydub, pedalboard |
| **AI分析** | Google Gemini 2.5 Flash |
| **音声変換** | RVC（Retrieval-based Voice Conversion） |
| **音声分離** | UVR5 |

## 📁 プロジェクト構造

```
myvoiceger/
├── app.py                 # メインアプリケーション
├── app_flask.py          # Flask Web サーバー
├── audio_utils.py        # 音声処理ユーティリティ
├── rvc_pipeline.py       # RVCパイプライン
├── gemini_utils.py       # Gemini API ユーティリティ
├── requirements.txt      # 依存関係
├── .gitignore           # Git 除外ファイル
├── README.md            # このファイル
├── templates/           # HTML テンプレート
│   ├── index.html       # メイン UI
│   └── error.html       # エラー ページ
├── static/              # 静的ファイル
│   ├── css/
│   ├── js/
│   └── images/
├── models/              # AI モデル（Large files 用）
│   ├── rvc/            # RVC モデル
│   └── uvr5/           # UVR5 モデル
├── uploads/            # アップロードファイル
├── outputs/            # 出力ファイル
│   ├── audio/         # 生成音声
│   └── images/        # 生成画像
├── docs/               # ドキュメント
└── logs/               # ログファイル
```

## 📜 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

## 🙏 謝辞

- [Google Gemini API](https://ai.google.dev/) - AI 分析機能を提供
- [librosa](https://librosa.org/) - 音声処理ライブラリ
- [Gradio](https://gradio.app/) - Web UI フレームワーク
- [RVC Community](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - 音声変換技術

## ⚠️ 注意

このアプリケーションは実験的な機能を含んでおり、実際のRVCモデルによる音声変換はモデルファイルの追加実装が必要です。

## 📖 詳細ドキュメント

- [UI操作ガイド](docs/UI_Screenshot_Guide.md) - 詳細なスクリーンショット付き手順書
- [APIリファレンス](docs/api.md) - 開発者向け API ドキュメント

## 🤝 コントリビューション

バグ報告、功能リクエスト、Pull Requestを歓迎します！

1. Fork してください
2. Feature branch を作成してください (`git checkout -b feature/AmazingFeature`)
3. 変更をコミットしてください (`git commit -m 'Add some AmazingFeature'`)
4. Branch を Push してください (`git push origin feature/AmazingFeature`)
5. Pull Request を開いてください

## 📞 サポート

質問やサポートが必要な場合は、[GitHub Issues](../../issues) を使用してください。

---

**MyVoiceger** - 高品質音声合成・変換プラットフォーム