# MyVoiceger — UI操作（スクリーンショット付き）ガイド

このドキュメントは、`MyVoiceger` の主要なUI操作をスクリーンショット付きで記録・共有するためのガイドです。
スクリーンショットを撮影し、指定のファイル名で `assets/screenshots/` に保存すると、ドキュメント内の該当箇所に画像を差し込めます。

---

## 準備
- スクリーンショット保存先（プロジェクトルートからの相対パス）: `assets/screenshots/`
- 推奨画像形式: `PNG`（可逆圧縮で品質を保つため）
- 推奨解像度: 1280x720 以上（UIが潰れない程度）

### 推奨ファイル名（順序に沿って配置してください）
- `01_input_tab.png` — 入力・前処理タブ全体のスクリーンショット
- `02_preprocessing_options.png` — 前処理オプション（分離ボタンやパラメータ）のクローズアップ
- `03_conversion_tab.png` — 変換エンジンタブ全体のスクリーンショット
- `04_conversion_params.png` — 変換パラメータ（モデル選択、ピッチ等）のクローズアップ
- `05_postprocessing_tab.png` — 後処理・ミックスタブのスクリーンショット
- `06_analysis_tab.png` — AI分析・視覚化タブのスクリーンショット
- `07_output_files.png` — `outputs/` に生成されたファイル一覧のスクリーンショット

> 注: 上記ファイル名はテンプレートです。任意に増やして問題ありませんが、ドキュメント内の画像参照名を一致させてください。

---

## スクリーンショットの撮り方（簡易）

### 方法A — Windows のショートカット（手早い）
- `Win + Shift + S` でスクリーンショット範囲選択（クリップボードへ保存）。
- その後、`Paint` などに貼り付けて `PNG` で保存。

### 方法B — Snipping Tool を使用
1. スタートメニューで「Snipping Tool」を検索して起動
2. `New` を押して範囲選択 → 保存

### 方法C — PowerShell で全画面を自動保存（例）
以下のコマンドは、メインスクリーンのスクリーンショットを `assets/screenshots/screen_auto.png` に保存します。

```powershell
# PowerShell（管理者不要）
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bmp)
$graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bmp.Size)
if(-not (Test-Path -Path "assets/screenshots")) { New-Item -ItemType Directory -Path "assets/screenshots" -Force }
$bmp.Save("assets/screenshots/screen_auto.png", [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose(); $bmp.Dispose(); Write-Host "Saved: assets/screenshots/screen_auto.png"
```

> 注意: このスクリプトはフルスクリーンを保存します。必要に応じてトリミングしてください。

---

## UI 操作手順（スクリーンショットを撮る位置を明記）

### 1) アプリ起動とトップ画面
1. `python app.py` でアプリを起動する。
2. ブラウザで `http://localhost:5556` を開く。
3. トップ（全体）画面のスクリーンショットを撮り、`01_input_tab.png` として保存。

```markdown
![トップ画面](../assets/screenshots/01_input_tab.png)
```

### 2) 入力・前処理タブ
- 操作: 楽曲（伴奏入り）と変換元音声をアップロード、必要な前処理オプション（ノイズ除去、サンプリングレート指定など）を確認。
- 撮影ポイント: 前処理のオプションが見える状態で `02_preprocessing_options.png` を保存。

```markdown
![前処理オプション](../assets/screenshots/02_preprocessing_options.png)
```

### 3) 変換エンジンタブ
- 操作: RVCモデル選択、ピッチやホルマントの設定、変換ボタンを確認。
- 撮影ポイント: 全体画面を `03_conversion_tab.png`、パラメータ部分を `04_conversion_params.png` として保存。

```markdown
![変換タブ](../assets/screenshots/03_conversion_tab.png)

![変換パラメータ](../assets/screenshots/04_conversion_params.png)
```

### 4) 後処理・ミックスタブ
- 操作: エフェクト選択、音量バランス調整、最終ミックス手順。
- 撮影ポイント: `05_postprocessing_tab.png` を保存。

```markdown
![後処理タブ](../assets/screenshots/05_postprocessing_tab.png)
```

### 5) AI分析・視覚化タブ
- 操作: 歌詞入力、Gemini分析、カバーアート生成の実行。
- 撮影ポイント: `06_analysis_tab.png` を保存。

```markdown
![AI分析タブ](../assets/screenshots/06_analysis_tab.png)
```

### 6) 出力確認
- 操作: `outputs/audio/` や `outputs/images/` に生成されたファイル名を確認。
- 撮影ポイント: ファイル一覧を表示して `07_output_files.png` を保存。

```markdown
![出力ファイル一覧](../assets/screenshots/07_output_files.png)
```

---

## ドキュメント内での画像差し替え方法
1. 実際に撮影して `assets/screenshots/` に保存
2. 上記のMarkdown の `![説明](../assets/screenshots/xxxx.png)` のパスが正しいか確認
3. Gitで管理する場合は `.gitignore` に個人情報など不要なファイルが含まれないよう注意

---

## 追加（任意）: 自動化スニペット
- `selenium` 等を使ってブラウザの特定要素をスクリーンショットする自動化も可能です。必要ならサンプルスクリプトを追加します。

---

作成後: スクリーンショットを追加して頂ければ、このドキュメント内の画像が表示されます。追加作業が必要なら、スクリーンショット取得の自動化（Selenium / Playwright）スクリプトを作成します。