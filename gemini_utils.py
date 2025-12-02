# Gemini utilities for MyVoiceger
"""
MyVoiceger用のGoogle Gemini 2.5 Flash API連携ユーティリティ

このモジュールは、歌声変換Webアプリケーション向けの多機能なGemini API連携を提供します。
歌詞分析、ムード分析、カバーアート生成、歌曲インサイト生成などの機能を実装しています。
"""

import os
import json
import base64
import requests
from typing import Any, Dict, List, Optional
from io import BytesIO

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("警告: google-generativeaiライブラリがインストールされていません。")
    genai = None

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("警告: PILまたはnumpyがインストールされていません。")
    Image = None
    np = None


def setup_gemini_client(api_key: Optional[str] = None) -> Any:
    """
    Google Gemini APIクライアントを初期化します。
    
    Args:
        api_key (str, optional): Gemini API キー。Noneの場合、環境変数から取得します。
        
    Returns:
        Any: 設定されたGeminiクライアント
        
    Raises:
        ValueError: APIキーが設定されていない場合
        Exception: API初期化に失敗した場合
    """
    if not genai:
        raise ImportError("google-generativeaiライブラリがインストールされていません。")
    
    # APIキーの取得
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini APIキーが設定されていません。環境変数GEMINI_API_KEYを設定するか、直接api_keyパラメータで指定してください。")
    
    try:
        # Gemini APIクライアントの初期化
        genai.configure(api_key=api_key)
        
        # 安全なコンテンツ設定
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # 生成モデルの設定
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        return {
            'client': genai,
            'safety_settings': safety_settings,
            'generation_config': generation_config,
            'model': genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                safety_settings=safety_settings,
                generation_config=generation_config
            )
        }
    except Exception as e:
        raise Exception(f"Gemini APIクライアントの初期化に失敗しました: {str(e)}")


def analyze_lyrics_and_mood(lyrics: str) -> Dict[str, Any]:
    """
    歌詞を分析してムードと感情スコアを算出します。
    
    Args:
        lyrics (str): 分析対象の歌詞テキスト
        
    Returns:
        Dict[str, Any]: 以下のキーを持つ分析結果辞書
            - mood (List[str]): 検出されたムード
            - emotion_score (float): 感情スコア (0.0-1.0)
            - keywords (List[str]): 重要なキーワード
            - genre (List[str]): 推定されるジャンル
            - color_palette (List[str]): 推奨カラーパレット
            
    Raises:
        ValueError: 歌詞が空または無効な場合
        Exception: API呼び出しに失敗した場合
    """
    if not lyrics or not lyrics.strip():
        raise ValueError("歌詞が空です。有効な歌詞テキストを提供してください。")
    
    try:
        # Geminiクライアントの初期化
        client_config = setup_gemini_client()
        model = client_config['model']
        
        # 分析プロンプトの構築
        prompt = f"""
以下の歌詞を分析し、JSON形式で回答してください。以下の構造厳密に守ってください：

歌詞: {lyrics}

分析項目:
1. mood: 歌曲の主要なムード（energetic, sad, romantic, peaceful, mysterious, nostalgic, uplifting, melancholicなど）
2. emotion_score: 感情の強度 (0.0-1.0の数値)
3. keywords: 歌詞から重要なキーワード（5-10個）
4. genre: 推定されるジャンル（j-pop, ballad, rock, electronic, folk, etc.）
5. color_palette: 歌曲にふさわしいカラーパレット（HEXカラー、3-5色）

JSON形式で回答してください。例：
{{
  "mood": ["romantic", "peaceful"],
  "emotion_score": 0.75,
  "keywords": ["love", "heart", "memories", "peaceful", "night"],
  "genre": ["j-pop", "ballad"],
  "color_palette": ["#FFB6C1", "#E6E6FA", "#F0E68C", "#98FB98", "#87CEEB"]
}}
"""
        
        # Gemini API呼び出し
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # JSONレスポンスの解析
        try:
            # JSONブロックを抽出（```json ... ```形式の可能性がある）
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            result = json.loads(response_text)
            
            # 必要なキーの検証とデフォルト値の設定
            validated_result = {
                'mood': result.get('mood', ['neutral']),
                'emotion_score': float(result.get('emotion_score', 0.5)),
                'keywords': result.get('keywords', []),
                'genre': result.get('genre', ['unknown']),
                'color_palette': result.get('color_palette', ['#808080', '#C0C0C0', '#A9A9A9'])
            }
            
            return validated_result
            
        except json.JSONDecodeError as e:
            # JSON解析失敗時のフォールバック
            return {
                'mood': ['neutral'],
                'emotion_score': 0.5,
                'keywords': [],
                'genre': ['unknown'],
                'color_palette': ['#808080', '#C0C0C0', '#A9A9A9']
            }
            
    except Exception as e:
        raise Exception(f"歌詞分析中にエラーが発生しました: {str(e)}")


def get_song_insights(lyrics: str, genre_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    歌曲の深い分析を実行し、詳細なインサイトを生成します。
    
    Args:
        lyrics (str): 歌詞テキスト
        genre_hint (str, optional): ジャンルヒント
        
    Returns:
        Dict[str, Any]: 歌曲のインサイト情報
            - vocal_style (str): 推奨ボーカルスタイル
            - arrangement_tips (List[str]): 編曲ティップス
            - emotional_arc (str): 感情的な展開
            - visual_keywords (List[str]): 視覚的キーワード
            
    Raises:
        ValueError: 歌詞が空または無効な場合
        Exception: API呼び出しに失敗した場合
    """
    if not lyrics or not lyrics.strip():
        raise ValueError("歌詞が空です。有効な歌詞テキストを提供してください。")
    
    try:
        # Geminiクライアントの初期化
        client_config = setup_gemini_client()
        model = client_config['model']
        
        # 分析プロンプトの構築
        genre_context = f"ジャンルヒント: {genre_hint}" if genre_hint else "ジャンルヒントなし"
        
        prompt = f"""
以下の歌詞について音楽的かつ視覚的な観点から詳細な分析を行い、JSON形式で回答してください：

歌詞: {lyrics}
{genre_context}

分析項目:
1. vocal_style: 推奨ボーカルスタイル（穏やか、力強い、エモーショナル、清楚、官的など）
2. arrangement_tips: 編曲に関する具体的なティップス（3-5個、箇条書き）
3. emotional_arc: 歌曲の感情的な展開（序盤、中盤、終盤の変化）
4. visual_keywords: 歌曲を視覚的に表現するためのキーワード（5-8個）

JSON形式で回答してください。例：
{{
  "vocal_style": "穏やかでEmotional、情感豊かな歌い方",
  "arrangement_tips": ["イントロは軽やかなピアノから開始", "，A段でストリングスを追加", "， B段でドラムスを本格導入", "， Cメロで静寂に戻し", "， Z段で再び情感を表現"],
  "emotional_arc": "序盤：穏やかで内省的な雰囲気 - 中盤：情感の高まり - 終盤：穏やかで希望的な結末",
  "visual_keywords": ["桜", "春の夕日", "川の流れ", "記憶", "希望", "静寂"]
}}
"""
        
        # Gemini API呼び出し
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # JSONレスポンスの解析
        try:
            # JSONブロックを抽出
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            result = json.loads(response_text)
            
            # 必要なキーの検証とデフォルト値の設定
            validated_result = {
                'vocal_style': result.get('vocal_style', '自然で感情的な歌い方'),
                'arrangement_tips': result.get('arrangement_tips', []),
                'emotional_arc': result.get('emotional_arc', '明確な感情的展開'),
                'visual_keywords': result.get('visual_keywords', [])
            }
            
            return validated_result
            
        except json.JSONDecodeError as e:
            # JSON解析失敗時のフォールバック
            return {
                'vocal_style': '自然で感情的な歌い方',
                'arrangement_tips': ['シンプルな伴奏から開始', '，楽曲の展開に合わせて楽器を追加'],
                'emotional_arc': '平静な導入から情感の高まり、そして穏やかな結末',
                'visual_keywords': ['music', 'emotion', 'harmony']
            }
            
    except Exception as e:
        raise Exception(f"歌曲インサイト生成中にエラーが発生しました: {str(e)}")


def describe_song_for_visualization(song_info: Dict[str, Any]) -> str:
    """
    歌曲の特徴を視覚的に表現するための説明文を生成します。
    
    Args:
        song_info (Dict[str, Any]): 歌曲情報（タイトル、アーティスト、ムード、分析結果など）
        
    Returns:
        str: 視覚化用の説明文
        
    Raises:
        ValueError: 歌曲情報が無効な場合
        Exception: API呼び出しに失敗した場合
    """
    if not song_info:
        raise ValueError("歌曲情報が空です。")
    
    try:
        # Geminiクライアントの初期化
        client_config = setup_gemini_client()
        model = client_config['model']
        
        # 歌曲情報の構築
        title = song_info.get('title', 'Unknown Title')
        artist = song_info.get('artist', 'Unknown Artist')
        mood_analysis = song_info.get('mood_analysis', {})
        
        # プロンプト構築
        prompt = f"""
以下の歌曲情報を基に、視覚的表現のための詳細で芸術的な説明文を生成してください：

歌曲名: {title}
アーティスト: {artist}
ムード分析: {json.dumps(mood_analysis, ensure_ascii=False, indent=2)}

以下の要素を含む包括的な視覚説明を作成してください：
1. 楽曲の雰囲気やムードを視覚的に表現
2. 色彩の組み合わせやパレット
3. シーンや情景の設定
4. 抽象的な表現やメタファー
5. カバーアートに適した要素

日本語で、詩的で詳細な150-200字程度の説明文を作成してください。
"""
        
        # Gemini API呼び出し
        response = model.generate_content(prompt)
        description = response.text.strip()
        
        return description
        
    except Exception as e:
        # エラー時のフォールバック説明
        mood = song_info.get('mood_analysis', {}).get('mood', ['neutral'])
        return f"{song_info.get('title', '楽曲')}は、{'、'.join(mood)}な雰囲気をを持つ歌曲です。的情感と調和した視覚的表現で歌曲の世界観を表現します。"


def generate_cover_art(mood_analysis: Dict[str, Any], output_path: str) -> str:
    """
    ムード分析に基づいてカバーアートを生成します。
    
    Args:
        mood_analysis (Dict[str, Any]): analyze_lyrics_and_moodの出力結果
        output_path (str): 出力画像ファイルパス
        
    Returns:
        str: 生成されたカバーアート画像のパス
        
    Raises:
        ImportError: PILがインストールされていない場合
        Exception: 画像生成に失敗した場合
    """
    if not Image:
        raise ImportError("PILライブラリがインストールされていません。")
    
    try:
        # 色彩解析
        color_palette = mood_analysis.get('color_palette', ['#808080', '#C0C0C0'])
        mood = mood_analysis.get('mood', ['neutral'])
        
        # 画像サイズの決定
        width, height = 512, 512
        
        # ベース画像の生成
        img = Image.new('RGB', (width, height), color_palette[0])
        draw = ImageDraw.Draw(img)
        
        # ムードに基づいた複雑なパターン生成
        if 'energetic' in mood:
            # エネルギッシュな楽曲：動的な円形パターン
            import random
            for i in range(50):
                x = random.randint(0, width)
                y = random.randint(0, height)
                radius = random.randint(10, 50)
                color = random.choice(color_palette)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=color, outline=None)
                           
        elif 'romantic' in mood:
            # ロマンチックな楽曲：ハートや柔らかな曲線
            from PIL import ImageFilter
            # 複数の薄い円形を配置
            for i in range(20):
                x = int(width * (i + 1) / 21)
                y = int(height * 0.5 + 100 * np.sin(i * 0.5)) if np else height // 2
                radius = 30 + i * 2
                color = color_palette[i % len(color_palette)]
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=color, outline=None)
                           
        elif 'peaceful' in mood:
            # 平和的な楽曲：シンプルなグラデーション効果
            for y in range(height):
                r = int(int(color_palette[0][1:3], 16) * (1 - y/height) + 
                       int(color_palette[-1][1:3], 16) * (y/height))
                g = int(int(color_palette[0][3:5], 16) * (1 - y/height) + 
                       int(color_palette[-1][3:5], 16) * (y/height))
                b = int(int(color_palette[0][5:7], 16) * (1 - y/height) + 
                       int(color_palette[-1][5:7], 16) * (y/height))
                color = f"#{r:02x}{g:02x}{b:02x}"
                draw.rectangle([0, y, width, y+1], fill=color)
        else:
            # その他のムード：抽象的なパターン
            for i in range(30):
                x = int(width * 0.5 + 200 * np.cos(i * 0.3)) if np else width // 2
                y = int(height * 0.5 + 200 * np.sin(i * 0.3)) if np else height // 2
                radius = 20 + i * 3
                color = color_palette[i % len(color_palette)]
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=color, outline=None, width=2)
        
        # フォルダが存在しない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 画像の保存
        img.save(output_path, 'PNG', quality=95)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"カバーアート生成中にエラーが発生しました: {str(e)}")


# テスト用のサンプル関数
def test_gemini_utils():
    """
    Geminiユーティリティのテスト関数。
    実際のAPIキーが設定されている場合のみ実行されます。
    """
    try:
        # サンプル歌詞
        sample_lyrics = """
        春の風に舞い上がる 花びら
        あなたと過ごした あの日々を 想出して
        心の奥で 靜かに光る 記憶の欠片
        もう一度 あの頃に戻れたら
        """
        
        print("=== MyVoiceger Gemini Utils テスト ===")
        
        # テスト1: setup_gemini_client
        print("1. Geminiクライアント初期化テスト...")
        client_config = setup_gemini_client()
        print("✓ 成功")
        
        # テスト2: analyze_lyrics_and_mood
        print("2. 歌詞分析テスト...")
        mood_analysis = analyze_lyrics_and_mood(sample_lyrics)
        print(f"✓ ムード分析結果: {mood_analysis}")
        
        # テスト3: get_song_insights
        print("3. 歌曲インサイト生成テスト...")
        song_insights = get_song_insights(sample_lyrics, "ballad")
        print(f"✓ インサイト: {song_insights}")
        
        # テスト4: describe_song_for_visualization
        print("4. 視覚化説明文生成テスト...")
        song_info = {
            'title': 'Spring Memories',
            'artist': 'Test Artist',
            'mood_analysis': mood_analysis
        }
        visualization_desc = describe_song_for_visualization(song_info)
        print(f"✓ 視覚化説明: {visualization_desc}")
        
        # テスト5: generate_cover_art (PILが必要なためスキップ)
        if Image:
            print("5. カバーアート生成テスト...")
            output_path = 'outputs/test_cover_art.png'
            cover_art_path = generate_cover_art(mood_analysis, output_path)
            print(f"✓ カバーアート生成: {cover_art_path}")
        else:
            print("5. カバーアート生成テスト: PIL未インストールのためスキップ")
        
        print("\n=== 全テスト完了 ===")
        
    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {str(e)}")


if __name__ == "__main__":
    # テスト実行
    test_gemini_utils()