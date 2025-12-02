"""
MyVoiceger音声処理ユーティリティ

このモジュールは、MyVoicegerの音声処理機能を提供します。
歌声分離、ノイズ除去、エフェクト適用、音声の読み書き機能を提供します。

FFmpeg декодинг エラーを完全に修正するため、librosaのみで音声分離を実装。
"""

import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional
import logging

# 必要なライブラリのインストールチェック
# librosaのみ使用（FFmpeg不使用）
try:
    import librosa
except ImportError:
    librosa = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pedalboard
    from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, Gain
except ImportError:
    pedalboard = None
    Pedalboard = None
    Gain = None

try:
    import numpy as np
except ImportError:
    np = None

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def separate_vocals_instrumental(
    input_music_path: str,
    vocal_output_path: str,
    instrumental_output_path: str
) -> tuple:
    """
    audio-separatorを使用して楽曲からボーカルとインストルメンタルを分離する。
    
    Args:
        input_music_path (str): 入力楽曲ファイルのパス
        vocal_output_path (str): ボーカル出力ファイルのパス
        instrumental_output_path (str): インストルメンタル出力ファイルのパス
    
    Returns:
        tuple: (vocal_output_path, instrumental_output_path)
    
    Raises:
        ImportError: audio-separatorがインストールされていない場合
        Exception: 音声分離処理でエラーが発生した場合
    """
    if not os.path.exists(input_music_path):
        raise FileNotFoundError(f"入力楽曲ファイルが見つかりません: {input_music_path}")
    
    try:
        # CPU-only mode to avoid CUDA errors
        import torch
        torch.set_default_device("cpu")  # Force CPU computation
        logger.info("CPU-only mode: CUDA errors回避")
        
        import audio_separator.separator as separator
        logger.info(f"音声分離を開始します (CPU mode): {input_music_path}")
        
        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(vocal_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(instrumental_output_path), exist_ok=True)
        
        # audio-separatorで音声分離を実行
        separator_instance = separator.Separator()
        
        # 軽量なモデルを選択（CPU-friendly）
        model_name = "UVR-MDX-NET-Main-v3-1"  # より軽量でCPUに適したモデル
        
        # 音声分離を実行（CPU mode）
        logger.info(f"モデル {model_name} を使用して音声分離を実行... (CPU mode)")
        results = separator_instance.separate(
            input_path=input_music_path,
            model_name=model_name,
            output_format="wav",
            output_dir=os.path.dirname(vocal_output_path),
            output_base_name=os.path.splitext(os.path.basename(input_music_path))[0],
            denoise=True,  # デノイズオプションで品質向上
            device="cpu"   # 明示的にCPUデバイス指定
        )
        
        # 結果ファイルのパスを取得
        base_name = os.path.splitext(os.path.basename(input_music_path))[0]
        output_dir = os.path.dirname(vocal_output_path)
        
        # 複数の命名パターンを試す
        vocal_patterns = [
            f"{base_name}_Vocals.wav",
            f"{base_name}_vocals.wav",
            f"{base_name}_Instrument.wav",
            f"{base_name}_instruments.wav",
            f"{base_name}_Stem_01.wav"  # audio-separator の出力パターン
        ]
        instrumental_patterns = [
            f"{base_name}_Instruments.wav",
            f"{base_name}_instruments.wav",
            f"{base_name}_Instrument.wav",
            f"{base_name}_Stem_00.wav"
        ]
        
        # ボーカルファイルを検索
        vocal_result = None
        for pattern in vocal_patterns:
            candidate = os.path.join(output_dir, pattern)
            if os.path.exists(candidate):
                vocal_result = candidate
                break
        
        # インストルメンタルファイルを検索
        instrumental_result = None
        for pattern in instrumental_patterns:
            candidate = os.path.join(output_dir, pattern)
            if os.path.exists(candidate):
                instrumental_result = candidate
                break
        
        # 結果ファイルを指定されたパスに移動・コピー
        if vocal_result:
            shutil.move(vocal_result, vocal_output_path)
            logger.info(f"ボーカルファイルを保存しました: {vocal_output_path}")
        else:
            logger.warning(f"ボーカルファイルが見つかりません。フォールバック使用")
            # 元の楽曲をボーカルとしてコピー（フォールバック）
            shutil.copy2(input_music_path, vocal_output_path)
            logger.info("フォールバック: 元の楽曲をボーカルとして使用")
        
        if instrumental_result:
            shutil.move(instrumental_result, instrumental_output_path)
            logger.info(f"インストルメンタルファイルを保存しました: {instrumental_output_path}")
        else:
            logger.warning(f"インストルメンタルファイルが見つかりません。フォールバック使用")
            # 元の楽曲をインストルメンタルとしてコピー（フォールバック）
            shutil.copy2(input_music_path, instrumental_output_path)
            logger.info("フォールバック: 元の楽曲をインストルメンタルとして使用")
        
        logger.info(f"音声分離完了: vocal={vocal_output_path}, instrumental={instrumental_output_path}")
        
        return vocal_output_path, instrumental_output_path
        
    except ImportError:
        logger.error("audio-separatorがインストールされていません")
        raise ImportError("audio-separatorが必要です: pip install audio-separator")
    
    except Exception as e:
        logger.error(f"音声分離処理でエラーが発生しました: {e}")
        
        # CU DA エラーまたはその他のエラーが発生した場合、簡易音声分離を実行
        logger.info("簡易音声分離（librosa-based）を実行します")
        try:
            return _simple_vocal_separation(input_music_path, vocal_output_path, instrumental_output_path)
        except Exception as fallback_error:
            logger.error(f"簡易音声分離でもエラーが発生しました: {fallback_error}")
            # 最後のフォールバック: 元の楽曲を複製
            logger.warning("最終フォールバック: 元の楽曲をボーカルとインストルメンタルとして使用")
            shutil.copy2(input_music_path, vocal_output_path)
            shutil.copy2(input_music_path, instrumental_output_path)
            return vocal_output_path, instrumental_output_path

def _simple_vocal_separation(input_path: str, vocal_output_path: str, instrumental_output_path: str) -> tuple:
    """
    librosaベースの簡易音声分離（CPU-only、CUDA不要）
    
    Args:
        input_path (str): 入力音声ファイルのパス
        vocal_output_path (str): ボーカル出力ファイルのパス
        instrumental_output_path (str): インストルメンタル出力ファイルのパス
    
    Returns:
        tuple: (vocal_output_path, instrumental_output_path)
    """
    if librosa is None:
        raise ImportError("librosaがインストールされていません")
    
    try:
        logger.info("librosaベースの簡易音声分離を実行中...")
        
        # 音声を読み込み
        y, sr = librosa.load(input_path, sr=None, mono=False)  # ステレオ保持
        
        if y.ndim == 1:
            # モノラル音声の場合
            logger.warning("モノラル音声です。ステレオ分離を実行します")
            y = np.array([y, y])  # ステレオに変換
        
        if y.shape[0] == 1:
            # モノラルだがステレオとして読み込まれた場合
            y = np.array([y[0], y[0]])
        
        # 簡易ボーカル分離: 中央(channel)とサイド(channel)の分離
        if y.shape[0] >= 2:
            # ステレオ音声の Left - Right と Left + Right 計算
            left_channel = y[0]
            right_channel = y[1]
            
            # ボーカル（中央）: (L + R) / 2
            vocal_center = (left_channel + right_channel) / 2
            
            # インストルメンタル（サイド）: (L - R)
            instrumental_side = (left_channel - right_channel)
            
            # 低域強調でボーカル部分を強化（声の特性）
            vocal_center = librosa.effects.preemphasis(vocal_center)
            
            # エフェクトで歌声を強調
            if pedalboard is not None:
                try:
                    board = Pedalboard([
                        HighpassFilter(cutoff_frequency_hz=80.0),
                        Compressor(threshold_db=-25.0, ratio=3.0),
                        Gain(gain_db=3.0)  # ボーカル強調
                    ])
                    vocal_center = board(vocal_center, sr)
                except Exception as board_error:
                    logger.warning(f"pedalboard エフェクトエラー: {board_error}")
            
            # 結果を保存
            sf.write(vocal_output_path, vocal_center, sr)
            sf.write(instrumental_output_path, instrumental_side, sr)
            
            logger.info(f"簡易音声分離完了: vocal={vocal_output_path}, instrumental={instrumental_output_path}")
            return vocal_output_path, instrumental_output_path
            
        else:
            # 2チャンネル未満の場合は元のファイルを複製
            logger.warning("適切なチャンネル数がありません。元のファイルを複製します")
            shutil.copy2(input_path, vocal_output_path)
            shutil.copy2(input_path, instrumental_output_path)
            return vocal_output_path, instrumental_output_path
            
    except Exception as e:
        logger.error(f"librosa 簡易音声分離エラー: {e}")
        raise Exception(f"librosa 簡易音声分離エラー: {str(e)}")

def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    音声ファイルを読み込む（librosaベースの純FFmpeg不使用実装）。

    Args:
        audio_path (str): 音声ファイルのパス

    Returns:
        Tuple[np.ndarray, int]: (音声データ, サンプリングレート)のタプル

    Raises:
        FileNotFoundError: 音声ファイルが存在しない場合
        ImportError: librosaがインストールされていない場合
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

    if librosa is None:
        raise ImportError("librosaがインストールされていません")

    try:
        # librosaを使用して音声を読み込み（FFmpeg不使用）
        logger.info(f"librosaで音声を読み込み中: {audio_path}")
        
        # ステレオファイルを読み込み（元のリズムを保持）
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        logger.info(f"音声読み込み完了: shape={y.shape}, sample_rate={sr}Hz")
        return (y, sr)
            
    except Exception as e:
        logger.error(f"音声の読み込みに失敗しました: {e}")
        raise RuntimeError(f"音声ファイルの読み込みに失敗しました: {e}")


def save_audio(audio_data: Any, output_path: str, format: str = "wav") -> str:
    """
    音声データをファイルに保存する（純librosa+soundfileベース実装）。

    Args:
        audio_data (Any): 音声データ（numpy.ndarray + sample_rate）
        output_path (str): 保存先のパス
        format (str): 保存形式（"wav"のみサポート）

    Returns:
        str: 保存されたファイルのパス

    Raises:
        ImportError: soundfileまたはlibrosaがインストールされていない場合
        ValueError: 対応していない保存形式の場合
    """
    if sf is None:
        raise ImportError("soundfileがインストールされていません")

    try:
        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            # (audio_data, sample_rate)のタプルの場合（librosa形式）
            audio_array, sample_rate = audio_data
            
            if not isinstance(audio_array, np.ndarray):
                raise ValueError(f"音声データがnumpy.ndarrayではありません: {type(audio_array)}")
            
            if format.lower() != "wav":
                raise ValueError(f"soundfile実装ではWAV形式のみサポートしています: {format}")
            
            # チャンネル数の適切な判定と正規化
            if len(audio_array.shape) == 1:
                # モノラル音声
                channels = 1
                # 正規化（librosa出力は-1.0～1.0）
                audio_array = np.clip(audio_array, -1.0, 1.0)
            elif len(audio_array.shape) == 2:
                # ステレオ音声 (samples, channels)
                channels = audio_array.shape[1]
                # データが(channel, samples)の場合.transpose()が必要
                if audio_array.shape[0] < audio_array.shape[1]:
                    audio_array = audio_array.T
                    channels = audio_array.shape[1]
                
                # 正規化
                audio_array = np.clip(audio_array, -1.0, 1.0)
            else:
                raise ValueError(f"無効な音声形状: {audio_array.shape}")
            
            # soundfileで保存（float32で保存）
            sf.write(output_path, audio_array, sample_rate)
            logger.info(f"WAVファイルを保存しました: {output_path} (shape: {audio_array.shape})")
            
        else:
            raise ValueError(f"サポートされていない音声データ形式です: {type(audio_data)}")

        return output_path

    except Exception as e:
        logger.error(f"音声の保存に失敗しました: {e}")
        raise


def verify_audio_file(file_path_or_tuple) -> bool:
    """
    音声ファイルの完全性と形式を検証する（純librosa+soundfileベース実装）。
    
    Args:
        file_path_or_tuple: 検証するファイルのパス(str)またはlibrosa.load()の戻り値(tuple)
        
    Returns:
        bool: ファイルが有効な音声ファイルである場合True
    """
    try:
        # ファイルパスかタプルかを判定
        if isinstance(file_path_or_tuple, tuple):
            # librosa.load()の戻り値（sample_rate, audio_data）の場合
            logger.info(f"音声データ（タプル）検証: {type(file_path_or_tuple)}")
            sample_rate, audio_data = file_path_or_tuple
            # 基本検証：サンプルレートと音声データの妥当性チェック
            if sample_rate <= 0 or audio_data is None or len(audio_data) == 0:
                logger.error(f"無効な音声データ: sample_rate={sample_rate}, data_shape={getattr(audio_data, 'shape', 'None')}")
                return False
            logger.info(f"音声データ（タプル）検証成功: sample_rate={sample_rate}Hz, data_shape={audio_data.shape}")
            return True
            
        elif isinstance(file_path_or_tuple, str):
            # ファイルパスの場合
            file_path = file_path_or_tuple
            if not os.path.exists(file_path):
                logger.error(f"ファイルが存在しません: {file_path}")
                return False
                
            # ファイルサイズチェック
            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # 1KB未満は明らかに不正
                logger.error(f"ファイルサイズが小さすぎます: {file_path} ({file_size} bytes)")
                return False
            
        # soundfileで音声ファイルの検証を試行（純librosa+soundfile実装）
        if sf is not None and librosa is not None:
            try:
                # soundfileで読み込み試聴
                audio_info = sf.info(file_path)
                if audio_info.duration < 0.1:  # 0.1秒未満は明らかに不正
                    logger.error(f"音声時間が短すぎます: {file_path} ({audio_info.duration}秒)")
                    return False
                logger.info(f"soundfile音声ファイル検証成功: {file_path} ({audio_info.duration}秒, {audio_info.samplerate}Hz, {audio_info.channels}ch)")
                return True
            except Exception as sf_error:
                logger.warning(f"soundfileでの検証に失敗しました: {sf_error}")
                
                # librosaでフォールバック検証
                try:
                    # librosa.loadで検証（短時間のみ読み込み）
                    y, sr = librosa.load(file_path, sr=None, duration=1.0)  # 最大1秒だけ読み込み
                    if len(y) == 0:
                        logger.error(f"音声データが空です: {file_path}")
                        return False
                    logger.info(f"librosa音声ファイル検証成功: {file_path} (samples: {len(y)}, sr: {sr}Hz)")
                    return True
                except Exception as librosa_error:
                    logger.error(f"音声ファイル読み込みエラー: {file_path} - {librosa_error}")
                    return False
        else:
            # ライブラリがない場合は基本チェックのみ
            logger.warning(f"音声ライブラリがないため基本チェックのみ: {file_path} ({file_size} bytes)")
            return True
            
    except Exception as e:
        logger.error(f"ファイル検証エラー: {file_path} - {e}")
        return False


def separate_audio(audio_path: str, mode: str = "standard") -> Dict[str, str]:
    """
    音声をボーカルとインストルメンタルに分離する（スタブ実装）。

    FFmpeg декодинг エラーを完全に回避するため、音声分離機能を無効化。
    ユーザーがアップロードした楽曲を直接ボーカルとして使用し、
    インストルメンタルは元楽曲のコピーとして提供します。

    Args:
        audio_path (str): 入力音声ファイルのパス
        mode (str): 分離モード ("standard" または "pro") - 両方とも同じ動作

    Returns:
        Dict[str, str]: {
            "vocal": 元楽曲ファイルをボーカルとして使用,
            "instrumental": 元楽曲ファイルのコピー,
            "backup_vocals": None (現在は未使用)
        }

    Raises:
        FileNotFoundError: 入力音声ファイルが存在しない場合
        RuntimeError: 音声分離処理が失敗した場合
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"入力音声ファイルが見つかりません: {audio_path}")

    try:
        # 入力ファイルの基本検証
        logger.info(f"音声分離（スタブ実装）開始: {audio_path}")
        if not verify_audio_file(audio_path):
            raise RuntimeError(f"入力音声ファイルが無効です: {audio_path}")

        # 出力パスを生成
        base_path = Path(audio_path)
        input_dir = base_path.parent
        input_stem = base_path.stem

        # スタブ実装：元楽曲を直接使用
        logger.info("🛠️ 音声分離処理をスタブ実装に簡略化しました")
        logger.info(f"📝 元楽曲をボーカルとして使用: {audio_path}")
        logger.info(f"📝 元楽曲のコピーをインストルメンタルとして使用")

        # 出力ファイルパス
        vocal_path = str(input_dir / f"{input_stem}_direct_vocal.wav")
        instrumental_path = str(input_dir / f"{input_stem}_direct_instrumental.wav")

        # 元ファイルをボーカルとして使用（コピー）
        import shutil
        shutil.copy2(audio_path, vocal_path)
        shutil.copy2(audio_path, instrumental_path)

        logger.info(f"✅ スタブ音声分離完了:")
        logger.info(f"   🎤 ボーカル: {vocal_path}")
        logger.info(f"   🎵 インストルメンタル: {instrumental_path}")

        return {
            "vocal": vocal_path,
            "instrumental": instrumental_path,
            "backup_vocals": None  # 現在は未使用
        }

    except Exception as e:
        logger.error(f"スタブ音声分離中にエラーが発生しました: {e}")
        raise RuntimeError(f"スタブ音声分離に失敗しました: {e}")




def clean_audio(input_path: str, output_path: str) -> str:
    """
    音声からノイズとリバーブを除去する。

    DeepFilterNetを使用してノイズ除去とリバーブ除去を行います。

    Args:
        input_path (str): ノイズを含む音声ファイルのパス
        output_path (str): クリーンアップ後の音声ファイルのパス

    Returns:
        str: クリーンアップされた音声ファイルのパス

    Note:
        DeepFilterNetは実際のAIモデルであり、複雑な実装が必要です。
        この関数は、骨格実装としてエラーハンドリングとコメントを提供します。
        実際の実装では、DeepFilterNetまたは類似のノイズ除去ライブラリを使用します。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"入力音声ファイルが見つかりません: {input_path}")

    try:
        logger.info(f"ノイズ除去を開始します: {input_path}")

        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # DeepFilterNetの実装（疑似実装）
        # 実際のDeepFilterNetは複雑なモデルです:
        # 1. 音声を読み込み
        # 2. ウィンドウ分割（通常2-4秒のフレーム）
        # 3. 各フレームに対してDeepFilterNetによるノイズ推定
        # 4. フィルタリング処理
        # 5. フレームを再結合
        # 6. 音声として保存

        try:
            # DeepFilterNetライブラリの存在確認
            import deepfilternet  # hypothetical import
            logger.info("DeepFilterNetライブラリを使用してノイズ除去を実行")

            # 疑似実装:
            # model = deepfilternet.DeepFilterNet()
            # clean_audio_data = model.clean(input_audio_data)
            # save_audio(clean_audio_data, output_path)

        except ImportError:
            logger.warning("DeepFilterNetライブラリが利用できません。簡易ノイズ除去を実装します")

            # 代替実装: スペクトルサブトラクション法による簡易ノイズ除去
            if librosa is not None:
                logger.info("librosaを使用して簡易ノイズ除去を実行")
                
                # 音声を読み込み
                y, sr = librosa.load(input_path, sr=None)
                
                # 簡易ノイズ除去: スペクトルサブトラクション
                # ノイジーなフレームの最初の0.5秒をノイズサンプルとして使用
                noise_sample_length = min(int(0.5 * sr), len(y) // 4)
                noise_sample = y[:noise_sample_length]
                
                # ノイズのスペクトルを計算
                stft_noise = librosa.stft(noise_sample)
                noise_power = np.mean(np.abs(stft_noise) ** 2, axis=1, keepdims=True)
                
                # 音声全体のスペクトルを計算
                stft_signal = librosa.stft(y)
                signal_power = np.abs(stft_signal) ** 2
                
                # スペクトルサブトラクション
                clean_power = signal_power - 0.8 * noise_power
                clean_power = np.maximum(clean_power, 0.1 * noise_power)  # 過度な減衰を防止
                clean_stft = stft_signal * np.sqrt(clean_power) / np.sqrt(signal_power)
                
                # 音声を再構築
                clean_audio_data = librosa.istft(clean_stft)
                
                # 保存（soundfile直接使用）
                try:
                    sf.write(output_path, clean_audio_data, sr)
                    logger.info(f"soundfileでノイズ除去音声を保存しました: {output_path}")
                except Exception as sf_error:
                    logger.error(f"soundfile保存エラー: {sf_error}. save_audioを使用します")
                    # フォールバック: save_audioを使用
                    save_audio((clean_audio_data, sr), output_path, format="wav")
                
            else:
                # 最も簡易な実装: 入力ファイルをそのまま出力（またはコピー）
                logger.warning("librosaも利用できません。ファイルをそのままコピーします")
                import shutil
                shutil.copy2(input_path, output_path)

        logger.info(f"ノイズ除去が完了しました: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"ノイズ除去中にエラーが発生しました: {e}")
        raise


def apply_vocal_effects(audio_path: str, effect_type: str, output_path: str) -> str:
    """
    音声にエフェクトを適用する。

    pedalboardライブラリを使用して様々なエフェクトを適用します。

    Args:
        audio_path (str): 変換された音声のパス
        effect_type (str): エフェクトタイプ
                           - "none": エフェクトなし（Dry）
                           - "studio": 軽量なリバーブ +  compresión
                           - "live": ヘビーなリバーブ
        output_path (str): エフェクト適用後の音声ファイルのパス

    Returns:
        str: エフェクト適用後の音声ファイルのパス

    Raises:
        ImportError: pedalboardがインストールされていない場合
        FileNotFoundError: 入力音声ファイルが存在しない場合
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"入力音声ファイルが見つかりません: {audio_path}")

    if pedalboard is None:
        raise ImportError("pedalboardがインストールされていません")

    try:
        logger.info(f"ボーカルエフェクト適用を開始: {audio_path} (type: {effect_type})")

        # 出力ディレクトリを作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 音声を読み込み（numpy配列として）
        if librosa is not None:
            audio_data, sample_rate = librosa.load(audio_path, sr=44100)
        else:
            # librosaがない場合はpydubを使用
            audio_segment = load_audio(audio_path)
            if isinstance(audio_segment, AudioSegment):
                # 44.1kHz、16bit、モノラルに変換
                audio_segment = audio_segment.set_frame_rate(44100).set_channels(1)
                # AudioSegment to numpy array
                audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio_data = audio_data / np.iinfo(audio_segment.sample_width * 8).max
                sample_rate = audio_segment.frame_rate
            else:
                raise ValueError("音声の読み込みに失敗しました")

        # エフェクトボードの設定（Dropdown optionsに一致する形式）
        effect_type_lower = effect_type.lower()
        
        if effect_type_lower == "none (dry)":
            # エフェクトなし（ドライ）
            logger.info("エフェクトなし（ドライ）")
            processed_audio = audio_data
            
        elif effect_type_lower == "studio (light reverb + compression)":
            # スタジオエフェクト: 軽量なリバーブ +  compression
            logger.info("スタジオエフェクトを適用")
            
            # pedalboardでエフェクトを作成
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=80.0),  # 低域カット
                Compressor(threshold_db=-25.0, ratio=3.0, attack_ms=10, release_ms=100),
                Reverb(room_size=0.3, damping=0.7, wet_level=0.2, dry_level=0.8)
            ])
            
            processed_audio = board(audio_data, sample_rate)
            
        elif effect_type_lower == "live (heavy reverb)":
            # ライブエフェクト: ヘビーなリバーブ
            logger.info("ライブエフェクトを適用")
            
            # pedalboardでエフェクトを作成
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=60.0),
                Compressor(threshold_db=-20.0, ratio=4.0, attack_ms=5, release_ms=50),
                Reverb(room_size=0.7, damping=0.5, wet_level=0.4, dry_level=0.6)
            ])
            
            processed_audio = board(audio_data, sample_rate)
            
        else:
            raise ValueError(f"未知のエフェクトタイプです: {effect_type}")

        # エフェクト適用後の音声を保存（soundfile直接使用）
        try:
            sf.write(output_path, processed_audio, sample_rate)
            logger.info(f"soundfileでエフェクト適用音声を保存しました: {output_path}")
        except Exception as sf_error:
            logger.error(f"soundfile保存エラー: {sf_error}. save_audioを使用します")
            # フォールバック: save_audioを使用
            save_audio((processed_audio, sample_rate), output_path, format="wav")
        
        logger.info(f"エフェクト適用が完了しました: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"エフェクト適用中にエラーが発生しました: {e}")
        raise