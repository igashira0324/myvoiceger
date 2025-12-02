# RVC pipeline utilities for MyVoiceger
"""
RVC (Retrieval-based Voice Conversion) パイプライン
高品質な歌声変換と формант shift 機能を実装
"""

import librosa
import numpy as np
import soundfile as sf
import pickle
import os
from typing import Any, Tuple, Optional
from scipy import signal
from scipy.interpolate import interp1d
import logging

# PyTorch импорт を完全に削除 - DLL エラーを回避
# すべての機能は librosa、numpy、scipy を 사용하여実装
print("RVCパイプライン: PyTorch無しの軽量版で起動しました")

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_voice(
    vocal_audio: str,
    target_voice: str,
    output_path: str,
    pitch_shift: int = 0,
    algorithm: str = "rmvpe",
    formant_shift: float = 1.0
) -> str:
    """
    RVCによる音声変換を実行する核心関数
    
    Args:
        vocal_audio (str): 変換元のボーカル音声ファイルパス
        target_voice (str): ターゲット音声（自分の声など）のファイルパス
        pitch_shift (int): ピッチシフト量（-12 ~ +12）
        algorithm (str): ピッチ検出アルゴリズム（"pm", "harvest", "rmvpe"）
        formant_shift (float):  формантシフト量（0.5 ~ 1.5）、音程を変えずに声質の太さを調整
        output_path (str): 出力音声ファイルパス
    
    Returns:
        str: 変換された音声ファイルのパス
        
    Note:
        формантshift機能により、音程不变情况下で声のキャラクター（太さ、高低感など）を調整可能
    """
    logger.info(f"音声変換開始: {vocal_audio} -> {output_path}")
    
    try:
        # 1. 音声ファイルの読み込み
        vocal_data, sr = librosa.load(vocal_audio, sr=44100)
        target_data, _ = librosa.load(target_voice, sr=44100)
        
        # 2. ターゲット音声の前処理とEmbeddings生成
        preprocessed_target = preprocess_target_voice(target_voice)
        embeddings = generate_embeddings(preprocessed_target)
        
        # 3. ピッチ抽出
        if algorithm == "rmvpe":
            pitches, magnitudes = extract_pitch_rmvpe(vocal_data, sr)
        elif algorithm == "harvest":
            pitches, magnitudes = extract_pitch_harvest(vocal_data, sr)
        else:  # "pm"
            pitches, magnitudes = extract_pitch_pm(vocal_data, sr)
        
        # 4. ピッチシフト処理
        if pitch_shift != 0:
            vocal_data = pitch_shift_audio(vocal_data, sr, pitch_shift)
            pitches = pitches * (2 ** (pitch_shift / 12))
        
        # 5. ** формантshift機能 ** - 最重要機能
        if abs(formant_shift - 1.0) > 0.01:
            vocal_data = apply_formant_shift(vocal_data, sr, formant_shift)
        
        # 6. RVC変換（前処理済みターゲット音声との比較による変換）
        converted_audio = apply_rvc_conversion(
            vocal_data, pitches, magnitudes, embeddings, sr
        )
        
        # 7. 最終出力
        sf.write(output_path, converted_audio, sr)
        
        logger.info(f"音声変換完了: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"音声変換エラー: {e}")
        raise


def preprocess_target_voice(target_voice_path: str) -> str:
    """
    ターゲット音声の前処理を実行する
    
    音声データ、前処理、特徴量抽出などの処理を行い、
    RVC変換に使用できる形式に変換する
    
    Args:
        target_voice_path (str): ターゲット音声ファイルのパス
    
    Returns:
        str: 前処理された音声ファイルのパス
        
    Note:
        実際のRVC実装では、複雑な前処理（ノイズ除去、正規化、
        セグメンテーションなど）が行われます
    """
    logger.info(f"ターゲット音声前処理開始: {target_voice_path}")
    
    try:
        # 音声ファイルの読み込み
        audio_data, sr = librosa.load(target_voice_path, sr=44100)
        
        # 1. ノイズ除去（簡易実装）
        audio_data = remove_noise_simple(audio_data, sr)
        
        # 2. 正規化
        audio_data = normalize_audio(audio_data)
        
        # 3. セグメンテーション（ silence 区間の削除）
        segments = remove_silence(audio_data, sr)
        
        # 4. ターゲット音声の保存
        preprocessed_path = target_voice_path.replace('.wav', '_preprocessed.wav')
        sf.write(preprocessed_path, segments, sr)
        
        logger.info(f"ターゲット音声前処理完了: {preprocessed_path}")
        return preprocessed_path
        
    except Exception as e:
        logger.error(f"ターゲット音声前処理エラー: {e}")
        raise


def generate_embeddings(target_voice_path: str) -> Any:
    """
    ターゲット音声から音声特徴量の埋め込みベクトルを生成する
    
    RVCのRetrieval部分（検索ベースの音声変換）に対応
    ターゲット音声の特徴を抽出し、変換時に参照されるインデックスを作成
    
    Args:
        target_voice_path (str): ターゲット音声ファイルのパス
    
    Returns:
        Any: 音声特徴量の埋め込みベクトル
        
    Note:
        実際のRVC実装では、複雑な深層学習モデルを使用して
        高次元の埋め込みベクトルを生成します
    """
    logger.info(f"Embeddings生成開始: {target_voice_path}")
    
    try:
        # 音声ファイルの読み込み
        audio_data, sr = librosa.load(target_voice_path, sr=44100)
        
        # 1. MFCC特徴量抽出
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        
        # 2. スペクトル特徴量抽出
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        
        # 3. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        
        # 4. ホルマント周波数の推定（簡易実装）
        formant_freqs = estimate_formants(audio_data, sr)
        
        # 5. 埋め込みベクトルの結合
        embeddings = {
            'mfccs': np.mean(mfccs, axis=1),
            'spectral_centroid': np.mean(spectral_centroids),
            'spectral_rolloff': np.mean(spectral_rolloff),
            'zcr_mean': np.mean(zcr),
            'formant_freqs': formant_freqs,
            'audio_length': len(audio_data)
        }
        
        logger.info("Embeddings生成完了")
        return embeddings
        
    except Exception as e:
        logger.error(f"Embeddings生成エラー: {e}")
        raise


def auto_train_model(target_voice_path: str, model_output_path: str) -> str:
    """
    ターゲット音声を使用した簡易版自動学習機能
    
    Few-shot学習推論時のIndex生成を行います
    簡易的なモデル作成または既存モデルとの連携機能
    
    Args:
        target_voice_path (str): ターゲット音声のパス
        model_output_path (str): 学習済みモデルの出力パス
    
    Returns:
        str: 学習済みモデルファイルのパス
        
    Note:
        実際のRVC実装では、大規模なデータセットを使用して
        複雑な深層学習モデルの学習を行います
    """
    logger.info(f"自動学習開始: {target_voice_path}")
    
    try:
        # 1. 前処理済みターゲット音声の生成
        preprocessed_path = preprocess_target_voice(target_voice_path)
        
        # 2. Embeddings生成
        embeddings = generate_embeddings(preprocessed_path)
        
        # 3. 簡易的なモデルの作成
        model_data = {
            'embeddings': embeddings,
            'source_audio_path': preprocessed_path,
            'training_info': {
                'algorithm': 'simplified_rvc',
                'sample_rate': 44100,
                'features': list(embeddings.keys())
            }
        }
        
        # 4. モデルファイルの保存
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"自動学習完了: {model_output_path}")
        return model_output_path
        
    except Exception as e:
        logger.error(f"自動学習エラー: {e}")
        raise


# ==================== 補助関数 ====================

def extract_pitch_rmvpe(audio_data: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    RMVPEアルゴリズムによるピッチ抽出（簡易実装）
    """
    # 簡易的な自己相関法によるピッチ抽出
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, threshold=0.1)
    
    # メインPitch軌跡の抽出
    pitch_values = []
    magnitude_values = []
    
    for t in range(pitches.shape[1]):
        index = pitches[:, t].argmax()
        pitch_values.append(pitches[index, t])
        magnitude_values.append(magnitudes[index, t])
    
    return np.array(pitch_values), np.array(magnitude_values)


def extract_pitch_harvest(audio_data: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    HARVESTアルゴリズムによるピッチ抽出（簡易実装）
    """
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, threshold=0.2)
    
    pitch_values = []
    magnitude_values = []
    
    for t in range(pitches.shape[1]):
        index = pitches[:, t].argmax()
        pitch_values.append(pitches[index, t])
        magnitude_values.append(magnitudes[index, t])
    
    return np.array(pitch_values), np.array(magnitude_values)


def extract_pitch_pm(audio_data: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    PM (Periodic Module) アルゴリズムによるピッチ抽出（簡易実装）
    """
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, threshold=0.3)
    
    pitch_values = []
    magnitude_values = []
    
    for t in range(pitches.shape[1]):
        index = pitches[:, t].argmax()
        pitch_values.append(pitches[index, t])
        magnitude_values.append(magnitudes[index, t])
    
    return np.array(pitch_values), np.array(magnitude_values)


def pitch_shift_audio(audio_data: np.ndarray, sr: int, pitch_shift: int) -> np.ndarray:
    """
    音声のピッチシフトを実行
    """
    return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=pitch_shift)


def apply_formant_shift(audio_data: np.ndarray, sr: int, formant_shift: float) -> np.ndarray:
    """
    ** формантshift機能 ** - 最重要機能
    
    音程を変えずに声質の太さを調整する機能
    ホルマント周波数をスケーリングすることで実現
    
    Args:
        audio_data (np.ndarray): 音声信号
        sr (int): サンプリングレート
        formant_shift (float):  форманトシフト量（0.5 ~ 1.5）
    
    Returns:
        np.ndarray:  форマンントシフト適用後の音声信号
    """
    # スペクトル解析
    stft = librosa.stft(audio_data)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # 周波数軸の準備
    freqs = librosa.fft_frequencies(sr=sr)
    
    # ホルマント周波数の推定（簡易実装）
    formants = estimate_formants(audio_data, sr)
    
    #  форманントシフト用のフィルタ作成
    shifted_magnitude = apply_formant_shift_filter(
        magnitude, freqs, formants, formant_shift
    )
    
    # 再合成
    shifted_stft = shifted_magnitude * np.exp(1j * phase)
    shifted_audio = librosa.istft(shifted_stft)
    
    return shifted_audio


def estimate_formants(audio_data: np.ndarray, sr: int) -> np.ndarray:
    """
    ホルマント周波数の推定（簡易実装）
    """
    # LPC分析によるホルマント推定
    try:
        # 音声信号の LPC係数計算
        lpc_order = min(12, len(audio_data) // 100)
        if lpc_order > 0 and len(audio_data) > 100:
            lpc_coeffs = librosa.lpc(audio_data, order=lpc_order)
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) >= 0]
            angles = np.angle(roots)
            
            # ホルマント周波数の計算
            formants = angles * sr / (2 * np.pi)
            formants = formants[formants < sr // 2]  # ナイキスト周波数以下
            formants = formants[formants > 50]  # 適切な周波数の範囲
            formants = formants[np.isfinite(formants)]  # 無限大やNaNの除去
            
            if len(formants) > 0:
                return np.sort(formants)
            else:
                logger.warning("ホルマント推定で適切な周波数が得られませんでした。デフォルト値を使用します。")
                return np.array([700, 1220, 2600])
        else:
            # 音声データが短すぎる場合
            logger.warning(f"音声データが短すぎます（長さ: {len(audio_data)}）。デフォルトホルマント周波数を使用します。")
            return np.array([700, 1220, 2600])
    except ValueError as e:
        logger.warning(f"LPC分析での値エラー: {e}. デフォルトホルマント周波数を使用します。")
        return np.array([700, 1220, 2600])
    except Exception as e:
        logger.warning(f"ホルマント推定中にエラーが発生しました: {e}. デフォルトホルマント周波数を使用します。")
        return np.array([700, 1220, 2600])


def apply_formant_shift_filter(
    magnitude: np.ndarray, 
    freqs: np.ndarray, 
    formants: np.ndarray, 
    formant_shift: float
) -> np.ndarray:
    """
     форманントシフトフィルタの適用
    """
    shifted_magnitude = magnitude.copy()
    
    for i, formant_freq in enumerate(formants):
        # シフト後のホルマント周波数
        shifted_freq = formant_freq * formant_shift
        
        # フィルタバンドの作成
        if i == 0:
            # 最初のホルマント：低域フィルタ
            band_mask = (freqs >= shifted_freq * 0.7) & (freqs <= shifted_freq * 1.3)
        elif i == len(formants) - 1:
            # 最後のホルマント：高域フィルタ
            band_mask = (freqs >= shifted_freq * 0.7) & (freqs <= shifted_freq * 1.3)
        else:
            # 中間のホルマント：バンドパスフィルタ
            prev_freq = formants[i-1] * formant_shift if i > 0 else shifted_freq * 0.7
            next_freq = formants[i+1] * formant_shift if i < len(formants)-1 else shifted_freq * 1.3
            band_mask = (freqs >= prev_freq) & (freqs <= next_freq)
        
        # シフト量の調整（ усиление/ослабление）
        if formant_shift > 1.0:
            #  форマンント周波数を上げる：声が明るく、高音質に
            shifted_magnitude[band_mask] *= (1.0 + (formant_shift - 1.0) * 0.5)
        else:
            #  форマンント周波数を下げる：声が低く、太声音に
            shifted_magnitude[band_mask] *= (0.8 + (formant_shift - 1.0) * 0.4)
    
    return shifted_magnitude


def apply_rvc_conversion(
    vocal_data: np.ndarray, 
    pitches: np.ndarray, 
    magnitudes: np.ndarray, 
    embeddings: Any, 
    sr: int
) -> np.ndarray:
    """
    RVC変換の適用（簡易実装）
    """
    # ターゲット音声の特徴量を取得
    target_mfccs = embeddings['mfccs']
    target_formants = embeddings['formant_freqs']
    
    # ソース音声の特徴量抽出
    source_mfccs = librosa.feature.mfcc(y=vocal_data, sr=sr, n_mfcc=13)
    source_mfcc_mean = np.mean(source_mfccs, axis=1)
    
    # MFCCの特徴量調整
    mfcc_diff = target_mfccs - source_mfcc_mean
    
    # RVC変換の実行（簡易版）
    converted_data = apply_mfcc_shift(vocal_data, sr, mfcc_diff)
    
    # ホルマント調整
    formant_data = apply_formant_matching(converted_data, sr, target_formants)
    
    return formant_data


def apply_mfcc_shift(audio_data: np.ndarray, sr: int, mfcc_diff: np.ndarray) -> np.ndarray:
    """
    MFCCベースの音声特徴量調整
    """
    # 簡易的なMFCCベース変換
    stft = librosa.stft(audio_data)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # MFCC差分をスペクトル調整に適用
    log_magnitude = np.log(magnitude + 1e-10)
    
    # 低周波数の調整
    for i in range(min(len(mfcc_diff), magnitude.shape[0])):
        if i < magnitude.shape[0]:
            magnitude[i] *= np.exp(mfcc_diff[i] * 0.1)
    
    # 再合成
    adjusted_stft = magnitude * np.exp(1j * phase)
    adjusted_audio = librosa.istft(adjusted_stft)
    
    return adjusted_audio


def apply_formant_matching(audio_data: np.ndarray, sr: int, target_formants: np.ndarray) -> np.ndarray:
    """
    ターゲットホルマント周波数への匹配
    """
    # 現在のホルマント周波数を推定
    current_formants = estimate_formants(audio_data, sr)
    
    if len(current_formants) > 0 and len(target_formants) > 0:
        # ホルマント周波数の匹配係数計算
        min_len = min(len(current_formants), len(target_formants))
        formant_ratios = target_formants[:min_len] / (current_formants[:min_len] + 1e-10)
        
        # форманントshiftの適用
        return apply_formant_shift(audio_data, sr, np.mean(formant_ratios))
    else:
        return audio_data


def remove_noise_simple(audio_data: np.ndarray, sr: int) -> np.ndarray:
    """
    簡易ノイズ除去
    """
    # スペクトルサブトラクション法の簡易実装
    stft = librosa.stft(audio_data)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # ノイズ推定（最初の0.5秒をノイズとして扱う）
    noise_length = int(0.5 * sr)
    if noise_length < len(audio_data):
        noise_frame = magnitude[:, :noise_length]
        noise_profile = np.mean(noise_frame, axis=1, keepdims=True)
        
        # ノイズ除去
        cleaned_magnitude = magnitude - noise_profile * 0.5
        cleaned_magnitude = np.maximum(cleaned_magnitude, magnitude * 0.1)
        
        # 再合成
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft)
        
        return cleaned_audio
    else:
        return audio_data


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    音声の正規化
    """
    # ピーク正規化
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        return audio_data / peak * 0.95
    else:
        return audio_data


def remove_silence(audio_data: np.ndarray, sr: int, threshold: float = 0.01) -> np.ndarray:
    """
    無音区間の削除
    """
    # 有声区間の検出
    intervals = librosa.effects.split(audio_data, top_db=20)
    
    # 有声区間を結合
    if len(intervals) > 0:
        segments = []
        for start, end in intervals:
            segments.append(audio_data[start:end])
        
        if segments:
            return np.concatenate(segments)
    
    return audio_data