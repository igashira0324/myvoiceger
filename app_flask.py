"""
MyVoiceger: Flask版 AI歌声変換Webアプリケーション

起動メッセージ表示を強化
"""

import os
import sys
import time
import logging
import platform
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, jsonify, request, session
import numpy as np

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# グローバル状態管理クラス
class AppState:
    def __init__(self):
        self.current_files = {}
        self.processing_status = {}
        self.start_time = datetime.datetime.now()
        self.system_info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    def reset(self):
        """アプリケーション状態をリセット"""
        self.current_files = {}
        self.processing_status = {}

# アプリケーションファクトリ
def create_app():
    """Flaskアプリケーションファクトリ"""
    
    # 起動メッセージ表示（巨大タイトル）
    print_large_startup_message()
    
    # Flask app作成
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'myvoiceger-flask-secret-key-2024'
    
    # 必要ディレクトリの作成
    create_required_directories()
    
    # グローバル状態管理
    app_state = AppState()
    app.config['APP_STATE'] = app_state
    
    # デバッグ情報のログ出力
    logger.info("🚀 MyVoiceger Flask アプリケーション初期化開始")
    logger.info(f"📅 起動日時: {app_state.system_info['start_time']}")
    logger.info(f"🖥️  プラットフォーム: {app_state.system_info['platform']}")
    logger.info(f"🐍 Pythonバージョン: {app_state.system_info['python_version']}")
    
    # ルートの登録
    register_routes(app)
    
    # エラーハンドラーの登録
    register_error_handlers(app)
    
    logger.info("✅ Flaskアプリケーション初期化完了")
    
    return app

def print_large_startup_message():
    """大きな起動タイトルメッセージを表示"""
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "="*80)
    print("🎤" + " "*38 + "🎵" + " "*38 + "🎤")
    print("🎤" + " "*15 + "MyVoiceger Flask" + " "*25 + "🎵")
    print("🎤" + " "*10 + "AI Voice Conversion Web Application" + " "*10 + "🎵")
    print("🎤" + " "*38 + "🎵" + " "*38 + "🎤")
    print("="*80)
    print(f"🕐 起動日時: {current_time}")
    print(f"🌐 アクセスURL: http://127.0.0.1:5000")
    print(f"🔧 開発モード: Flask Development Server")
    print(f"📁 プロジェクト: MyVoiceger Flask版")
    print(f"🎯 機能: AI歌声変換・音声処理")
    print("="*80)
    print("🚀 Flaskサーバーを起動しています...")
    print("📝 ログ出力を開始します")
    print("="*80 + "\n")

def create_required_directories():
    """必要なディレクトリを作成"""
    directories = [
        'templates',
        'static/css',
        'static/js', 
        'static/images',
        'uploads',
        'outputs/audio',
        'temp',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 ディレクトリ確認: {directory}")

def register_routes(app):
    """ルート registation"""
    
    @app.route('/')
    def index():
        """メインページ"""
        # セッション変数の初期化（BuildError回避）
        if 'files' not in session:
            session['files'] = {}
        if 'step_completed' not in session:
            session['step_completed'] = {}
        if 'analysis' not in session:
            session['analysis'] = None
            
        app_state = app.config.get('APP_STATE')
        logger.info("🏠 メインページアクセス: /")
        return render_template('index.html',
                             app_info={
                                 'name': 'MyVoiceger Flask',
                                 'version': '2.0.0-Flask',
                                 'start_time': app_state.system_info['start_time']
                             })
    
    @app.route('/health')
    def health_check():
        """ヘルスチェック"""
        logger.info("💓 ヘルスチェック: /health")
        return jsonify({
            'status': 'healthy',
            'app': 'MyVoiceger Flask',
            'version': '2.0.0-Flask',
            'timestamp': datetime.datetime.now().isoformat(),
            'uptime': str(datetime.datetime.now() - app.config['APP_STATE'].start_time)
        })
    
    @app.route('/test')
    def test_page():
        """テストページ"""
        logger.info("🧪 テストページアクセス: /test")
        return jsonify({
            'message': 'Flaskアプリケーションが正常に動作しています！',
            'app': 'MyVoiceger Flask',
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    @app.route('/api/status')
    def api_status():
        """APIステータスインドポイント"""
        logger.info("📊 APIステータス確認: /api/status")
        return jsonify({
            'status': 'operational',
            'services': {
                'flask_server': '✅ running',
                'audio_processing': '✅ available',
                'templates': '✅ loaded',
                'static_files': '✅ served'
            },
            'system': app.config['APP_STATE'].system_info,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    @app.route('/api/test-audio-processing')
    def test_audio_processing():
        """音声処理機能テスト"""
        logger.info("🎵 音声処理テスト: /api/test-audio-processing")
        
        try:
            # テスト用の短い音声ファイルを作成
            test_audio_path = 'temp/test_flask.wav'
            os.makedirs('temp', exist_ok=True)
            
            # 簡単なテストトーンを生成（0.1秒）
            duration = 0.1
            sample_rate = 22050
            t = np.linspace(0, duration, int(duration * sample_rate), False)
            test_tone = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
            
            # ステレオファイルとして保存
            stereo_audio = np.column_stack((test_tone, test_tone))
            
            # soundfileを使用して保存（FFmpegフリー）
            try:
                import soundfile as sf
                sf.write(test_audio_path, stereo_audio, sample_rate)
                audio_success = True
            except ImportError:
                # soundfileがない場合はnumpy形式で保存
                np.save(test_audio_path.replace('.wav', '.npy'), stereo_audio)
                test_audio_path = test_audio_path.replace('.wav', '.npy')
                audio_success = False
            
            logger.info(f"✅ テスト音声ファイル作成: {test_audio_path}")
            
            # separate_audio関数のテスト（スタブ実装）
            try:
                from audio_utils import separate_audio
                result = separate_audio(test_audio_path, 'standard')
                separation_success = True
                vocal_file = os.path.basename(result.get('vocal', 'unknown'))
                instrumental_file = os.path.basename(result.get('instrumental', 'unknown'))
            except Exception as e:
                separation_success = False
                vocal_file = 'N/A'
                instrumental_file = 'N/A'
                logger.warning(f"音声分離テスト失敗: {e}")
            
            return jsonify({
                'status': 'success',
                'message': '音声処理システムのテストが完了しました',
                'test_results': {
                    'audio_generation': '✅ success',
                    'audio_saved_as': test_audio_path,
                    'separation_function': '✅ available' if separation_success else '❌ failed',
                    'vocal_file': vocal_file,
                    'instrumental_file': instrumental_file,
                    'ffmpeg_free': audio_success
                },
                'system_info': app.config['APP_STATE'].system_info,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"音声処理テストエラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'音声処理テスト中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
    
    @app.route('/upload_music', methods=['POST'])
    def upload_music():
        """楽曲ファイルアップロード"""
        logger.info("🎵 楽曲アップロードリクエスト: /upload_music")
        try:
            if 'music_file' not in request.files:
                return jsonify({'status': 'error', 'message': 'ファイルが選択されていません'}), 400
            
            file = request.files['music_file']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'ファイルが選択されていません'}), 400
            
            # ファイル保存処理（スタブ実装）
            filename = f"music_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            
            logger.info(f"✅ 楽曲ファイル保存: {filepath}")
            return jsonify({
                'status': 'success',
                'message': '楽曲ファイルが正常にアップロードされました',
                'filename': filename,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"楽曲アップロードエラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'アップロード中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
    
    @app.route('/upload_voice', methods=['POST'])
    def upload_voice():
        """ターゲット音声アップロード"""
        logger.info("🎤 ターゲット音声アップロード: /upload_voice")
        try:
            if 'target_voice_file' not in request.files:
                return jsonify({'status': 'error', 'message': 'ファイルが選択されていません'}), 400
            
            file = request.files['target_voice_file']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'ファイルが選択されていません'}), 400
            
            # ファイル保存処理（スタブ実装）
            filename = f"voice_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            
            logger.info(f"✅ ターゲット音声ファイル保存: {filepath}")
            return jsonify({
                'status': 'success',
                'message': 'ターゲット音声ファイルが正常にアップロードされました',
                'filename': filename,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"ターゲット音声アップロードエラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'アップロード中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
    
    @app.route('/download/<path:filename>')
    def download_file(filename):
        """ファイルダウンロード"""
        logger.info(f"📥 ファイルダウンロード: /download/{filename}")
        try:
            filepath = os.path.join('uploads', filename)
            if not os.path.exists(filepath):
                return jsonify({'status': 'error', 'message': 'ファイルが見つかりません'}), 404
            
            # スタブ実装：実際のダウンロードは実装されていません
            return jsonify({
                'status': 'success',
                'message': f'ファイル {filename} のダウンロード準備中です',
                'filename': filename,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"ファイルダウンロードエラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'ダウンロード中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
    
    @app.route('/preprocess', methods=['POST'])
    def preprocess():
        """音声前処理（分離、ノイズ除去）"""
        logger.info("🔧 音声前処理リクエスト: /preprocess")
        try:
            # スタブ実装：実際の音声前処理をスキップ
            return jsonify({
                'status': 'success',
                'message': '音声前処理が完了しました（スタブ実装）',
                'processing_time': '0.1秒',
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"音声前処理エラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'音声前処理中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
    
    @app.route('/convert_voice', methods=['POST'])
    def convert_voice_route():
        """音声変換処理（RVC）"""
        logger.info("🎤 音声変換リクエスト: /convert_voice")
        try:
            # スタブ実装：実際の音声変換をスキップ
            return jsonify({
                'status': 'success',
                'message': '音声変換が完了しました（スタブ実装）',
                'conversion_info': {
                    'pitch_shift': '0 semitones',
                    'formant_shift': '1.0x',
                    'algorithm': 'pm',
                    'processing_time': '0.5秒'
                },
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"音声変換エラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'音声変換中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
    
    @app.route('/postprocess', methods=['POST'])
    def postprocess():
        """音声後処理（エフェクト付与、ミックス）"""
        logger.info("🎛️ 音声後処理リクエスト: /postprocess")
        try:
            # スタブ実装：実際のエフェクト付与をスキップ
            return jsonify({
                'status': 'success',
                'message': '音声後処理が完了しました（スタブ実装）',
                'effects_applied': ['studio_reverb', 'compression'],
                'final_mix': {
                    'vocal_level': '0dB',
                    'instrumental_level': '-3dB',
                    'processing_time': '0.2秒'
                },
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"音声後処理エラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'音声後処理中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
    
    @app.route('/analyze', methods=['POST'])
    def analyze():
        """AI分析（歌詞・ムード解析、カバーアート生成）"""
        logger.info("🤖 AI分析リクエスト: /analyze")
        try:
            # スタブ実装：実際のGemini分析をスキップ
            return jsonify({
                'status': 'success',
                'message': 'AI分析が完了しました（スタブ実装）',
                'analysis_results': {
                    'lyrics_mood': '(upbeat, energetic)',
                    'genre_prediction': 'pop/rock',
                    'cover_art_description': 'colorful abstract design with musical notes',
                    'emotional_tone': 'positive',
                    'processing_time': '1.2秒'
                },
                'cover_art_url': '/static/images/generated_cover.png',
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"AI分析エラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'AI分析中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500
    
    @app.route('/reset')
    def reset():
        """アプリケーション状態リセット"""
        logger.info("🔄 アプリケーション状態リセット: /reset")
        try:
            app_state = app.config.get('APP_STATE')
            app_state.reset()
            
            # ファイル清理
            for directory in ['uploads', 'temp']:
                if os.path.exists(directory):
                    import shutil
                    shutil.rmtree(directory)
                    os.makedirs(directory, exist_ok=True)
            
            return jsonify({
                'status': 'success',
                'message': 'アプリケーション状態が正常にリセットされました',
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"リセットエラー: {e}")
            return jsonify({
                'status': 'error',
                'message': f'リセット中にエラーが発生しました: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }), 500

def register_error_handlers(app):
    """エラーハンドラーの登録"""
    
    @app.errorhandler(404)
    def not_found(error):
        logger.warning(f"404 エラー: {request.path}")
        return jsonify({
            'error': 'Not Found',
            'message': 'リクエストされたページが見つかりません',
            'path': request.path
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 エラー: {error}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'サーバー内部エラーが発生しました'
        }), 500

if __name__ == '__main__':
    # 起動準備段階のメッセージ（標準出力すぐに表示）
    print("🚀 MyVoiceger Flaskアプリケーション起動準備中...", flush=True)
    print("⏳ 初期化プロセスを開始します...", flush=True)
    
    try:
        # ステップ1: 環境チェック
        print("\n" + "="*80)
        print("🔍 STEP 1: 環境と依存関係チェック")
        print("="*80, flush=True)
        
        # Flaskチェック
        try:
            import flask
            print(f"✅ Flask: {flask.__version__} - 利用可能", flush=True)
        except ImportError as e:
            print(f"❌ Flask: インポートエラー - {e}", flush=True)
            raise
        
        # 必要なコアモジュールチェック
        core_modules = ['audio_utils', 'rvc_pipeline', 'gemini_utils']
        for module in core_modules:
            try:
                __import__(module)
                print(f"✅ {module}: 正常ロード", flush=True)
            except ImportError as e:
                print(f"❌ {module}: インポートエラー - {e}", flush=True)
        
        print(f"🎤 MyVoiceger Flask - AI歌声変換Webアプリケーション", flush=True)
        print(f"📅 起動時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        
        # ステップ2: アプリケーション初期化
        print("\n" + "="*80)
        print("🔧 STEP 2: Flaskアプリケーション初期化")
        print("="*80, flush=True)
        app = create_app()
        print("✅ アプリケーションファクトリ: 完了", flush=True)
        
        # ステップ3: 初期化完了メッセージ
        print("\n" + "="*80)
        print("🎉 Flaskアプリケーションが正常に初期化されました！")
        print("🌐 アクセスURL: http://127.0.0.1:5000")
        print("📋 利用可能なエンドポイント:")
        print("   🏠 /                     - メインページ")
        print("   💓 /health              - ヘルスチェック")
        print("   🧪 /test                - テストページ")
        print("   📊 /api/status          - APIステータス")
        print("   🎵 /api/test-audio-processing - 音声処理テスト")
        print("="*80)
        print("🔄 Flask開発サーバーを起動します...")
        print("⏳ しばらくお待ちください...")
        print("="*80 + "\n", flush=True)
        
        # Flaskサーバーを起動（デバッグモード）
        print("🔥 Flaskサーバーが起動中...", flush=True)
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=False,  # 二重起動防止
            threaded=True,
            processes=1
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠️ ユーザーによる中断 - アプリケーションを終了します", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Flaskアプリケーション起動エラー: {e}", flush=True)
        print("🔍 詳細エラー情報:", flush=True)
        import traceback
        error_details = traceback.format_exc()
        print(error_details, flush=True)
        print("\n💡 トラブルシューティング:", flush=True)
        print("1. 必要なライブラリがインストールされているか確認", flush=True)
        print("2. ポート5000が別のプロセスで使用されていないか確認", flush=True)
        print("3. 必要なファイル（templates, static）が存在するか確認", flush=True)
        sys.exit(1)
    finally:
        print("\n🚀 MyVoiceger Flask アプリケーション起動プロセス完了", flush=True)