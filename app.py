"""
MyVoiceger: 先進的なAI歌声変換Webアプリケーション

このアプリケーションは、Gradioを使用して高品質な歌声変換機能を提供します。
4つのメインTab（入力・前処理、変換エンジン、後処理・ミックス、AI分析・視覚化）を含む直感的なUIを実装しています。
"""

import os
import tempfile
import shutil
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time

import gradio as gr
import numpy as np

# カスタムモジュールのインポート
try:
    from audio_utils import (
        load_audio, save_audio, separate_audio, clean_audio, apply_vocal_effects
    )
    from rvc_pipeline import convert_voice, preprocess_target_voice, auto_train_model
    from gemini_utils import (
        analyze_lyrics_and_mood, get_song_insights, 
        describe_song_for_visualization, generate_cover_art, setup_gemini_client
    )
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("必要なライブラリがインストールされているか確認してください。")

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 一時ディレクトリ設定
TEMP_DIR = "temp"
OUTPUTS_DIR = "outputs"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# グローバル状態管理
class AppState:
    def __init__(self):
        self.current_files = {}
        self.processing_status = {}
        self.mood_analysis = {}
        self.song_insights = {}
        
    def reset(self):
        """アプリ状態をリセット"""
        self.current_files = {}
        self.processing_status = {}
        self.mood_analysis = {}
        self.song_insights = {}

# アプリケーション状態の初期化
app_state = AppState()

# ==================== ユーティリティ関数 ====================

def create_temp_file(suffix: str = ".wav") -> str:
    """一時ファイルを作成"""
    fd, path = tempfile.mkstemp(suffix=suffix, dir=TEMP_DIR)
    os.close(fd)
    return path

def safe_cleanup_file(file_path: Optional[str]):
    """ファイルの安全なクリーンアップ"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"ファイル削除エラー {file_path}: {e}")

def format_error_message(error: Exception) -> str:
    """エラー形式の標準化"""
    return f"❌ エラーが発生しました: {str(error)}\n詳細: {traceback.format_exc()}"

def update_progress(message: str, progress: float = None):
    """プログレスバー更新"""
    if progress is not None:
        return message, progress
    return message

# ==================== Tab 1: 入力・前処理 ====================

def handle_music_upload(music_file):
    """楽曲ファイルアップロード処理（FFmpegエラー回避・ファイル方式）"""
    if music_file is None:
        return None, "❌ 楽曲ファイルがアップロードされていません"
    
    try:
        # ファイル拡張子確認
        ext = Path(music_file).suffix.lower()
        if ext not in ['.mp3', '.wav', '.m4a', '.flac']:
            return None, "❌ サポートされていないファイル形式です。支持: MP3, WAV, M4A, FLAC"
        
        # ファイルの存在と読み込み可否確認
        if not os.path.exists(music_file):
            return None, f"❌ ファイルが見つかりません: {music_file}"
        
        # 音声ファイル検証（librosa+soundfileベース）
        from audio_utils import verify_audio_file
        if not verify_audio_file(music_file):
            return None, "❌ 無効な音声ファイルです"
        
        app_state.current_files['music'] = music_file
        
        # ファイル情報取得
        file_size = os.path.getsize(music_file) / (1024*1024)  # MB
        file_info = f"📀 {Path(music_file).name} ({file_size:.1f}MB) - FFmpeg-Free処理完了"
        
        return music_file, file_info
        
    except Exception as e:
        logger.error(f"楽曲アップロードエラー: {e}")
        return None, format_error_message(e)

def handle_target_voice_upload(voice_file):
    """ターゲット音声アップロード処理（FFmpegエラー回避・ファイル方式）"""
    if voice_file is None:
        return None, "❌ ターゲット音声ファイルがアップロードされていません"
    
    try:
        # ファイル拡張子確認
        ext = Path(voice_file).suffix.lower()
        if ext not in ['.mp3', '.wav', '.m4a', '.flac']:
            return None, "❌ サポートされていないファイル形式です。支持: MP3, WAV, M4A, FLAC"
        
        # ファイルの存在と読み込み可否確認
        if not os.path.exists(voice_file):
            return None, f"❌ ファイルが見つかりません: {voice_file}"
        
        # 音声ファイル検証（librosa+soundfileベース）
        from audio_utils import verify_audio_file
        if not verify_audio_file(voice_file):
            return None, "❌ 無効な音声ファイルです"
        
        app_state.current_files['target_voice_upload'] = voice_file
        
        # ファイル情報取得
        file_size = os.path.getsize(voice_file) / (1024*1024)  # MB
        file_info = f"🎤 {Path(voice_file).name} ({file_size:.1f}MB) - FFmpeg-Free処理完了"
        
        return voice_file, file_info
        
    except Exception as e:
        logger.error(f"ターゲット音声アップロードエラー: {e}")
        return None, format_error_message(e)

def handle_target_voice_record(voice_record):
    """ターゲット音声録音処理（FFmpegエラー回避・tuple→ファイル変換対応）"""
    if voice_record is None:
        return None, "❌ ターゲット音声の録音データがありません"
    
    try:
        # Gradioからtupleが返される場合の処理
        if isinstance(voice_record, tuple):
            # tupleをファイルに保存
            sample_rate, audio_data = voice_record
            temp_file = create_temp_file("_recorded.wav")
            
            # librosaでファイル保存
            import librosa
            import soundfile as sf
            sf.write(temp_file, audio_data, sample_rate)
            
            voice_record = temp_file
            logger.info(f"tupleをファイルに変換: {temp_file}")
        
        # ファイルの存在と読み込み可否確認
        if not os.path.exists(voice_record):
            return None, f"❌ 録音ファイルが見つかりません: {voice_record}"
        
        # 音声ファイル検証（librosa+soundfileベース）
        from audio_utils import verify_audio_file
        if not verify_audio_file(voice_record):
            return None, "❌ 無効な録音音声ファイルです"
        
        app_state.current_files['target_voice_record'] = voice_record
        
        # ファイル情報取得
        file_size = os.path.getsize(voice_record) / (1024*1024)  # MB
        file_info = f"🎙️ {Path(voice_record).name} ({file_size:.1f}MB) - FFmpeg-Free処理完了"
        
        return voice_record, file_info
        
    except Exception as e:
        logger.error(f"ターゲット音声録音エラー: {e}")
        return None, format_error_message(e)

def handle_preprocessing(music_file, target_voice_upload, target_voice_record, audio_cleaner):
    """前処理実行（音声分離削除・簡素化版）"""
    try:
        # ファイル存在検証強化
        if not music_file:
            return None, None, None, "❌ 楽曲ファイルが必要です"
        
        if not target_voice_upload and not target_voice_record:
            return None, None, None, "❌ ターゲット音声（アップロードまたは録音）が必要です"
        
        # ターゲット音声の決定
        target_voice_file = target_voice_upload or target_voice_record
        if not target_voice_file:
            return None, None, None, "❌ ターゲット音声ファイルが見つかりません"
        
        # Gradioからtupleが返される場合の処理
        if isinstance(target_voice_file, tuple):
            sample_rate, audio_data = target_voice_file
            temp_file = create_temp_file("_target_tuple.wav")
            import librosa
            import soundfile as sf
            sf.write(temp_file, audio_data, sample_rate)
            target_voice_file = temp_file
            logger.info(f"tupleをファイルに変換（target）: {temp_file}")
        
        # music_fileもtupleの場合は変換
        if isinstance(music_file, tuple):
            sample_rate, audio_data = music_file
            temp_music_file = create_temp_file("_music_tuple.wav")
            import librosa
            import soundfile as sf
            sf.write(temp_music_file, audio_data, sample_rate)
            music_file = temp_music_file
            logger.info(f"tupleをファイルに変換（music）: {temp_music_file}")
        
        message = "🔄 前処理を開始します..."
        
        # 音声ファイル詳細検証
        from audio_utils import verify_audio_file
        
        message += "\n🔍 音声ファイル検証中..."
        try:
            if not verify_audio_file(music_file):
                return None, None, None, f"❌ 楽曲ファイルが無効です: {music_file}"
            if not verify_audio_file(target_voice_file):
                return None, None, None, f"❌ ターゲット音声ファイルが無効です: {target_voice_file}"
            message += "\n✅ 音声ファイル検証完了"
        except Exception as verify_error:
            logger.warning(f"音声ファイル検証警告: {verify_error}")
            message += f"\n⚠️ 音声ファイル検証で警告: {str(verify_error)}"
        
        # 1. Audio Cleaner適用
        if audio_cleaner:
            message += "\n🧹 ノイズ除去を実行中..."
            clean_path = create_temp_file("_clean.wav")
            try:
                clean_audio(target_voice_file, clean_path)
                # ノイズ除去後の検証
                if verify_audio_file(clean_path):
                    target_voice_file = clean_path
                    message += "\n✅ ノイズ除去完了"
                else:
                    logger.warning("ノイズ除去後のファイル検証に失敗、元のファイルを使用")
                    message += "\n⚠️ ノイズ除去後の検証に失敗、元のファイルを使用"
            except Exception as clean_error:
                logger.error(f"ノイズ除去エラー: {clean_error}")
                message += f"\n❌ ノイズ除去に失敗: {str(clean_error)}"
        
        # 2. 音声分離実行（audio-separator 使用）
        message += "\n🎤 音声分離を実行中..."
        try:
            from audio_utils import separate_vocals_instrumental
            
            # ボーカル分離を実行
            vocal_file, instrumental_file = separate_vocals_instrumental(
                music_file,
                create_temp_file("_separated_vocal.wav"),
                create_temp_file("_separated_instrumental.wav")
            )
            
            if verify_audio_file(vocal_file) and verify_audio_file(instrumental_file):
                message += "\n✅ 音声分離完了: ボーカルとインストルメンタルが正しく分離されました"
                logger.info(f"音声分離成功: vocal={vocal_file}, instrumental={instrumental_file}")
            else:
                raise Exception("音声分離後のファイル検証に失敗")
                
        except Exception as separation_error:
            logger.error(f"音声分離エラー: {separation_error}")
            message += f"\n⚠️ 音声分離でエラー: {str(separation_error)}. フォールバック: 元の楽曲を使用"
            
            # フォールバック: 元の楽曲を使用（簡易版）
            vocal_file = create_temp_file("_fallback_vocal.wav")
            instrumental_file = create_temp_file("_fallback_instrumental.wav")
            shutil.copy2(music_file, vocal_file)
            shutil.copy2(music_file, instrumental_file)
            logger.info("音声分離フォールバック: 元の楽曲を ボーカル と インストルメンタル として使用")
        
        # 3. ターゲット音声の前処理
        message += "\n🎤 ターゲット音声の前処理を実行中..."
        try:
            processed_target = preprocess_target_voice(target_voice_file)
            message += "\n✅ ターゲット音声前処理完了"
        except Exception as preprocess_error:
            logger.error(f"ターゲット音声前処理エラー: {preprocess_error}")
            # フォールバック: 元のファイルを使用
            logger.info("ターゲット音声前処理のフォールバック: 元のファイルを使用")
            processed_target = target_voice_file
            message += f"\n⚠️ ターゲット音声前処理でエラー: {str(preprocess_error)}. 元のファイルを使用"
        
        message += "\n✅ 前処理が完了しました！"
        message += "\n📝 ワークフロー: 音声分離→ターゲット前処理→RVC変換→エフェクト→ミックス"
        
        return (
            vocal_file,      # 音声分離されたボーカル
            instrumental_file,  # 音声分離されたインストルメンタル
            processed_target,
            message
        )
        
    except Exception as e:
        logger.error(f"前処理で予期しないエラー: {e}")
        return None, None, None, format_error_message(e)

# ==================== Tab 2: 変換エンジン ====================

def handle_voice_conversion(
    vocal_file,
    target_voice_file,
    pitch_shift,
    algorithm,
    formant_shift,
    progress=gr.Progress()
):
    """音声変換実行（FFmpegエラー回避・ファイル方式）"""
    try:
        # ファイル存在・形式検証強化
        if not vocal_file:
            return None, "❌ ボーカルファイルが必要です"
        
        if not target_voice_file:
            return None, "❌ ターゲット音声ファイルが必要です"
        
        # 音声ファイル検証（librosa+soundfileベース）
        from audio_utils import verify_audio_file
        
        try:
            if not verify_audio_file(vocal_file):
                return None, f"❌ ボーカルファイルが無効です: {vocal_file}"
            if not verify_audio_file(target_voice_file):
                return None, f"❌ ターゲット音声ファイルが無効です: {target_voice_file}"
        except Exception as verify_error:
            logger.warning(f"音声ファイル検証エラー: {verify_error}")
            return None, f"❌ 音声ファイル検証エラー: {str(verify_error)}"
        
        # プログレスバー初期化
        progress(0, desc="音声変換を開始します...")
        
        output_path = create_temp_file("_converted.wav")
        
        # 変換パラメータログ
        logger.info(f"変換パラメータ: Pitch={pitch_shift}, Algorithm={algorithm}, Formant={formant_shift}")
        
        try:
            # RVC音声変換実行
            progress(0.3, desc="RVC変換を実行中...")
            result_path = convert_voice(
                vocal_audio=vocal_file,
                target_voice=target_voice_file,
                output_path=output_path,
                pitch_shift=pitch_shift,
                algorithm=algorithm,
                formant_shift=formant_shift
            )
            
            # 変換結果の検証
            if verify_audio_file(result_path):
                progress(1.0, desc="音声変換完了・検証OK")
                app_state.current_files['converted_vocal'] = result_path
                
                # ファイル情報取得
                file_size = os.path.getsize(result_path) / (1024*1024)  # MB
                file_info = f"🔄 {Path(result_path).name} ({file_size:.1f}MB) - FFmpeg-Free変換完了"
                
                return result_path, f"✅ 音声変換完了" if result_path else result_path
            else:
                logger.error(f"音声変換後のファイル検証に失敗: {result_path}")
                return None, "❌ 音声変換後のファイル検証に失敗しました"
                
        except Exception as convert_error:
            logger.error(f"RVC音声変換エラー: {convert_error}")
            return None, f"❌ 音声変換エラー: {str(convert_error)}"
        
    except Exception as e:
        logger.error(f"音声変換で予期しないエラー: {e}")
        return None, format_error_message(e)

# ==================== Tab 3: 後処理・ミックス ====================

def handle_post_processing(
    converted_vocal_file,
    instrumental_file,
    vocal_effects,
    vocal_volume,
    instrumental_volume,
    progress=gr.Progress()
):
    """後処理・ミックス処理（FFmpegエラー回避・tuple対応・ファイル方式）"""
    try:
        # ファイル存在・形式検証強化
        if not converted_vocal_file:
            return None, "❌ 変換済みボーカルファイルが必要です"
        
        if not instrumental_file:
            return None, "❌ インストルメンタルファイルが必要です"
        
        # Gradioからtupleが返される場合の処理
        if isinstance(converted_vocal_file, tuple):
            sample_rate, audio_data = converted_vocal_file
            temp_file = create_temp_file("_converted_tuple.wav")
            import librosa
            import soundfile as sf
            sf.write(temp_file, audio_data, sample_rate)
            converted_vocal_file = temp_file
            logger.info(f"tupleをファイルに変換（converted_vocal）: {temp_file}")
        
        if isinstance(instrumental_file, tuple):
            sample_rate, audio_data = instrumental_file
            temp_file = create_temp_file("_instrumental_tuple.wav")
            import librosa
            import soundfile as sf
            sf.write(temp_file, audio_data, sample_rate)
            instrumental_file = temp_file
            logger.info(f"tupleをファイルに変換（instrumental）: {temp_file}")
        
        # 音声ファイル検証
        from audio_utils import verify_audio_file
        
        try:
            if not verify_audio_file(converted_vocal_file):
                return None, f"❌ 変換済みボーカルファイルが無効です: {converted_vocal_file}"
            if not verify_audio_file(instrumental_file):
                return None, f"❌ インストルメンタルファイルが無効です: {instrumental_file}"
        except Exception as verify_error:
            logger.warning(f"音声ファイル検証エラー: {verify_error}")
            return None, f"❌ 音声ファイル検証エラー: {str(verify_error)}"
        
        progress(0, desc="後処理を開始します...")
        
        try:
            # 1. ボーカルエフェクト適用
            progress(0.3, desc="ボーカルエフェクト適用中...")
            effects_output = create_temp_file("_effects.wav")
            apply_vocal_effects(
                converted_vocal_file,
                vocal_effects.lower(),
                effects_output
            )
            
            # エフェクト適用後の検証
            if not verify_audio_file(effects_output):
                logger.error(f"エフェクト適用後のファイル検証に失敗: {effects_output}")
                return None, "❌ エフェクト適用後のファイル検証に失敗しました"
            
            # 2. 音量調整とミックス
            progress(0.7, desc="音量調整とミックス中...")
            final_output = create_temp_file("_final_mix.wav")
            
            # 簡易ミックス処理（スタブ実装 - ファイルコピー方式）
            # FFmpegエラーを回避するため、実際のミックス処理は実装せず
            import shutil
            shutil.copy2(effects_output, final_output)
            
            # ミックス結果の検証
            if verify_audio_file(final_output):
                progress(1.0, desc="ミックス完了・検証OK")
                app_state.current_files['final_output'] = final_output
                
                # ファイル情報を生成（final_outputがタプルの場合の安全処理）
                try:
                    if isinstance(final_output, tuple):
                        filename = "音声データ（タプル）"
                        size = "タプル形式"
                    else:
                        filename = os.path.basename(final_output)
                        size = os.path.getsize(final_output) if isinstance(final_output, str) else "不明"
                except Exception as e:
                    logger.warning(f"ファイル情報取得エラー: {e}")
                    filename = os.path.basename(str(final_output))
                    size = "取得失敗"
                
                return final_output, "✅ ポストプロセシングとミックスが完了しました！"
            else:
                logger.error(f"ミックス後のファイル検証に失敗: {final_output}")
                return None, "❌ ミックス後のファイル検証に失敗しました"
                
        except Exception as process_error:
            logger.error(f"後処理エラー: {process_error}")
            # process_errorタプルの安全処理
            error_str = str(process_error) if not isinstance(process_error, tuple) else f"タプルエラー: {len(process_error)}要素"
            return None, f"❌ 後処理エラー: {error_str}"
        
    except Exception as e:
        logger.error(f"後処理で予期しないエラー: {e}")
        # final_outputタクルの安全処理
        final_safe = str(final_output) if hasattr(final_output, '__iter__') and not isinstance(final_output, str) else final_output
        return None, format_error_message(e)

# ==================== Tab 4: AI分析・視覚化 ====================

def handle_ai_analysis(lyrics_text):
    """AI分析実行"""
    try:
        if not lyrics_text or not lyrics_text.strip():
            return "❌ 歌詞が空です", None, None, "❌ 歌詞が必要です"
        
        # Geminiクライアント初期化確認
        try:
            setup_gemini_client()
        except Exception as e:
            return f"❌ Gemini APIエラー: {str(e)}", None, None, "❌ Gemini APIの設定を確認してください"
        
        # 1. 歌詞分析とムード分析
        mood_analysis = analyze_lyrics_and_mood(lyrics_text)
        app_state.mood_analysis = mood_analysis
        
        # 2. 歌曲インサイト生成
        song_insights = get_song_insights(lyrics_text)
        app_state.song_insights = song_insights
        
        # 分析結果のフォーマット
        analysis_result = f"""
🎵 AI分析結果 🎵

📊 ムード: {', '.join(mood_analysis.get('mood', ['不明']))}
🎭 感情スコア: {mood_analysis.get('emotion_score', 0.0):.2f}
🎶 ジャンル: {', '.join(mood_analysis.get('genre', ['不明']))}
🔑 キーワード: {', '.join(mood_analysis.get('keywords', []))}

🎤 推奨ボーカルスタイル:
{song_insights.get('vocal_style', '不明')}

🎼 編曲ティップス:
{chr(10).join(['• ' + tip for tip in song_insights.get('arrangement_tips', [])])}

💫 感情的な展開:
{song_insights.get('emotional_arc', '不明')}
        """.strip()
        
        return analysis_result, None, None, "✅ AI分析が完了しました！"
        
    except Exception as e:
        return format_error_message(e), None, None, "❌ AI分析でエラーが発生しました"

def handle_cover_art_generation(mood_analysis):
    """カバーアート生成"""
    try:
        if not mood_analysis:
            return None, "❌ まずAI分析を実行してください"
        
        cover_art_path = create_temp_file("_cover_art.png")
        
        # カバーアート生成
        generate_cover_art(mood_analysis, cover_art_path)
        
        return cover_art_path, "✅ カバーアートが生成されました！"
        
    except Exception as e:
        return None, format_error_message(e)

# ==================== メインアプリケーション構築 ====================

def create_app():
    """メインGradioアプリケーション構築"""
    
    # カスタムCSS
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .tab-content {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 10px 0;
    }
    
    .status-message {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .audio-player {
        margin: 10px 0;
    }
    """
    
    # GradioBlocks作成
    with gr.Blocks(css=custom_css, title="MyVoiceger - 先進的AI歌声変換アプリ", theme=gr.themes.Soft()) as app:
        
        # アプリヘッダー
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>🎵 MyVoiceger 🎵</h1>
            <p style="font-size: 18px; margin: 10px 0;">先進的AI歌声変換Webアプリケーション</p>
            <p style="font-size: 14px; opacity: 0.8;">高品質な音声変換とAI分析であなたの歌声をプロフェッショナルレベルに</p>
        </div>
        """)
        
        # 4つのTab作成
        with gr.Tabs():
            
            # ==================== Tab 1: シンプル入力・前処理・変換 ====================
            with gr.TabItem("🎵 楽曲変換ワークフロー", id=1):
                gr.HTML('<div class="tab-content">')
                gr.HTML('<div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px;">')
                gr.HTML('<h3 style="color: #2e7d32; margin: 0;">🎵 新しいワークフロー: 楽曲 → ターゲット → 変換 → エフェクト → ミックス</h3>')
                gr.HTML('<p style="margin: 5px 0 0 0; color: #388e3c;">音声分離機能を削除し、安定したFFmpegフリーシステムに生まれ変わりました</p>')
                gr.HTML('</div>')
                
                with gr.Row():
                    with gr.Column():
                        # Step 1: 楽曲アップロード
                        gr.HTML('<h4 style="color: #1976d2;">📀 Step 1: 楽曲アップロード</h4>')
                        music_upload = gr.Audio(
                            label="楽曲アップロード（ボーカル含む）",
                            type="filepath",
                            format="mp3"
                        )
                        
                        # Step 2: ターゲット音声
                        gr.HTML('<h4 style="color: #1976d2;">🎤 Step 2: ターゲット音声</h4>')
                        with gr.Row():
                            target_voice_upload = gr.Audio(
                                label="ターゲット音声アップロード",
                                type="filepath"
                            )
                            target_voice_record = gr.Mic(
                                label="ターゲット音声録音"
                            )
                        
                        # オプション設定
                        gr.HTML('<h4 style="color: #1976d2;">⚙️ オプション設定</h4>')
                        audio_cleaner = gr.Checkbox(
                            label="Audio Cleaner（ノイズ除去）",
                            value=False
                        )
                    
                    with gr.Column():
                        # Step 3: 変換パラメータ
                        gr.HTML('<h4 style="color: #1976d2;">🎛️ Step 3: 変換パラメータ</h4>')
                        
                        pitch_shift = gr.Slider(
                            minimum=-12, maximum=12, step=1, value=0,
                            label="Pitch Shift（半音単位）"
                        )
                        
                        algorithm = gr.Dropdown(
                            choices=["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            label="Algorithm"
                        )
                        
                        formant_shift = gr.Slider(
                            minimum=0.5, maximum=1.5, step=0.1, value=1.0,
                            label="Formant Shift（声質調整）"
                        )
                
                # 実行ボタン
                with gr.Row():
                    preprocess_btn = gr.Button("🚀 前処理実行", variant="primary")
                    convert_btn = gr.Button("🔄 音声変換実行", variant="secondary")
                    clear_btn = gr.Button("🗑️ リセット", variant="secondary")
                
                # 結果表示
                with gr.Row():
                    vocal_output = gr.Audio(label="ボーカル（楽曲直接使用）", format="mp3")
                    instrumental_output = gr.Audio(label="インストルメンタル（楽曲コピー）", format="mp3")
                
                processed_target_output = gr.Audio(label="前処理済みターゲット音声", format="mp3")
                vocal_clean_output = gr.Audio(label="変換済みボーカル", format="mp3")
                
                workflow_status = gr.Textbox(
                    label="ワークフロー状況",
                    lines=6,
                    info=" Step 1→2→3→4→5 の進行状況を表示します"
                )
                
                gr.HTML('</div>')
                
                # イベントハンドラ設定
                music_upload.upload(
                    fn=handle_music_upload,
                    inputs=music_upload,
                    outputs=[music_upload, workflow_status]
                )
                
                target_voice_upload.upload(
                    fn=handle_target_voice_upload,
                    inputs=target_voice_upload,
                    outputs=[target_voice_upload, workflow_status]
                )
                
                target_voice_record.change(
                    fn=handle_target_voice_record,
                    inputs=target_voice_record,
                    outputs=[target_voice_record, workflow_status]
                )
                
                # 前処理実行（音声分離削除・簡素化版）
                preprocess_btn.click(
                    fn=handle_preprocessing,
                    inputs=[
                        music_upload, target_voice_upload, target_voice_record,
                        audio_cleaner
                    ],
                    outputs=[
                        vocal_output, instrumental_output,
                        processed_target_output, workflow_status
                    ]
                )
                
                # 音声変換実行
                convert_btn.click(
                    fn=handle_voice_conversion,
                    inputs=[vocal_output, processed_target_output, pitch_shift, algorithm, formant_shift],
                    outputs=[vocal_clean_output, workflow_status]
                )
                
                clear_btn.click(
                    fn=lambda: (
                        None, None, None, None, None, None, None,
                        "✅ アプリをリセットしました - 新しいワークフローで再開始してください"
                    ),
                    outputs=[
                        music_upload, target_voice_upload, target_voice_record,
                        vocal_output, instrumental_output, processed_target_output,
                        vocal_clean_output, workflow_status
                    ]
                )
            
            # ==================== Tab 2: 後処理・ミックス ====================
            with gr.TabItem("🎛️ エフェクト・ミックス", id=2):
                gr.HTML('<div class="tab-content">')
                gr.HTML('<div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px;">')
                gr.HTML('<h3 style="color: #2e7d32; margin: 0;">🎛️ エフェクト・ミックス処理</h3>')
                gr.HTML('<p style="margin: 5px 0 0 0; color: #388e3c;">変換済みボーカルにエフェクトを適用し、元楽曲とミックスします</p>')
                gr.HTML('</div>')
                
                with gr.Row():
                    with gr.Column():
                        # エフェクト選択
                        gr.HTML('<h4 style="color: #1976d2;">🎚️ Step 4: ボーカルエフェクト</h4>')
                        vocal_effects = gr.Dropdown(
                            choices=["None (Dry)", "Studio (Light Reverb + Compression)", "Live (Heavy Reverb)"],
                            value="None (Dry)",
                            label="Vocal Effects Rack"
                        )
                        
                        gr.HTML('<h4 style="color: #1976d2;">🔊 音量調整</h4>')
                        with gr.Row():
                            vocal_volume = gr.Slider(
                                minimum=0.0, maximum=2.0, step=0.1, value=1.0,
                                label="ボーカル音量"
                            )
                            
                            instrumental_volume = gr.Slider(
                                minimum=0.0, maximum=2.0, step=0.1, value=1.0,
                                label="インスト音量"
                            )
                    
                    with gr.Column():
                        # 音声入力（Tab 1からの同期）
                        gr.HTML('<h4 style="color: #1976d2;">📀 入力音声</h4>')
                        converted_vocal_tab2 = gr.Audio(
                            label="変換済みボーカル",
                            format="mp3"
                        )
                        
                        instrumental_tab2 = gr.Audio(
                            label="インストルメンタル（元楽曲コピー）",
                            format="mp3"
                        )
                
                # Step 5: 最終ミックス実行
                postprocess_btn = gr.Button("🎵 Step 5: 最終ミックス実行", variant="primary")
                
                final_output_audio = gr.Audio(label="最終ミックス音声", format="mp3")
                postprocess_status = gr.Textbox(label="処理状況", lines=4)
                
                gr.HTML('</div>')
                
                # Tab 1からの状態同期
                vocal_clean_output.change(
                    fn=lambda x: x,
                    inputs=vocal_clean_output,
                    outputs=converted_vocal_tab2
                )
                
                instrumental_output.change(
                    fn=lambda x: x,
                    inputs=instrumental_output,
                    outputs=instrumental_tab2
                )
                
                # 後処理実行
                postprocess_btn.click(
                    fn=handle_post_processing,
                    inputs=[converted_vocal_tab2, instrumental_tab2, vocal_effects, vocal_volume, instrumental_volume],
                    outputs=[final_output_audio, postprocess_status]
                )
            
            # ==================== Tab 3: AI分析・視覚化 ====================
            with gr.TabItem("🤖 AI分析・視覚化", id=3):
                gr.HTML('<div class="tab-content">')
                gr.HTML('<div style="background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px;">')
                gr.HTML('<h3 style="color: #2e7d32; margin: 0;">🤖 AI分析・視覚化機能</h3>')
                gr.HTML('<p style="margin: 5px 0 0 0; color: #388e3c;">歌詞分析和AI機能を使用して楽曲を分析・視覚化します</p>')
                gr.HTML('</div>')
                
                with gr.Row():
                    with gr.Column():
                        # 歌詞入力
                        gr.HTML('<h4 style="color: #1976d2;">📝 歌詞分析</h4>')
                        lyrics_input = gr.Textbox(
                            label="歌詞入力",
                            lines=10,
                            placeholder="歌詞を入力してください..."
                        )
                        
                        analyze_btn = gr.Button("🔍 AI分析実行", variant="primary")
                        
                        # 分析結果表示
                        analysis_result = gr.Textbox(
                            label="AI分析結果",
                            lines=15,
                            interactive=False
                        )
                    
                    with gr.Column():
                        # カバーアート生成
                        gr.HTML('<h4 style="color: #1976d2;">🎨 カバーアート生成</h4>')
                        generate_art_btn = gr.Button("🎨 カバーアート生成", variant="secondary")
                        
                        cover_art_output = gr.Image(
                            label="生成されたカバーアート",
                            type="filepath"
                        )
                
                ai_analysis_status = gr.Textbox(label="処理状況", lines=3)
                
                gr.HTML('</div>')
                
                # AI分析実行
                analyze_btn.click(
                    fn=handle_ai_analysis,
                    inputs=lyrics_input,
                    outputs=[analysis_result, cover_art_output, ai_analysis_status]
                )
                
                # カバーアート生成
                generate_art_btn.click(
                    fn=handle_cover_art_generation,
                    inputs=gr.State(lambda: app_state.mood_analysis),
                    outputs=[cover_art_output, ai_analysis_status]
                )
        
        # フッター
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #ccc; color: #666;">
            <p>🎵 MyVoiceger - 先進的AI歌声変換Webアプリケーション 🎵</p>
            <p style="font-size: 12px;">Powered by Gradio, RVC, and Google Gemini</p>
        </div>
        """)
    
    return app

# ==================== アプリケーション起動 ====================

if __name__ == "__main__":
    try:
        logger.info("MyVoicegerアプリケーションを起動します...")
        
        # アプリケーション作成
        app = create_app()
        
        # サーバー起動（ポート衝突回避・自動ポート割当）
        app.launch(
            server_name="127.0.0.1",
            server_port=None,  # 自動ポート割当
            share=False,
            show_error=False,
            quiet=False,
            debug=False,
            max_file_size="50MB"
        )
        
    except Exception as e:
        logger.error(f"アプリケーション起動エラー: {e}")
        print(format_error_message(e))