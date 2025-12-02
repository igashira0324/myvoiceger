#!/usr/bin/env python3
"""
Gemini APIでサポートされているモデル一覧を取得するスクリプト
"""

import os
from pathlib import Path
import sys

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def load_env_file():
    """環境変数を.envファイルから読み込みます"""
    env_path = current_dir / '.env'
    
    if not env_path.exists():
        print(f"❌ .envファイルが見つかりません: {env_path}")
        return False
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    
    return True

def list_available_models():
    """利用可能なGeminiモデルを取得します"""
    try:
        from gemini_utils import setup_gemini_client
        
        # Geminiクライアントの初期化
        client_config = setup_gemini_client()
        client = client_config['client']
        
        print("📋 利用可能なGeminiモデル一覧:")
        print("-" * 50)
        
        # 利用可能なモデルを取得
        models = list(client.list_models())
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
                print(f"   📝 表示名: {model.display_name}")
                print(f"   🏷️  説明: {model.description}")
                print()
        
        # 利用可能な生成メソッドを表示
        print("\n🔧 推奨されるモデル:")
        recommended_models = [
            "gemini-1.5-pro", "gemini-1.5-pro-latest", "gemini-1.5-pro-002",
            "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-002",
            "gemini-2.0-flash-exp", "gemini-pro", "gemini-pro-vision"
        ]
        
        for model in models:
            for rec_model in recommended_models:
                if rec_model in model.name:
                    print(f"  🎯 {model.name} (推奨)")
                    return model.name
        
        return None
        
    except Exception as e:
        print(f"❌ モデル一覧取得エラー: {e}")
        return None

def main():
    print("🔍 Gemini API利用可能なモデル一覧確認")
    print("=" * 50)
    
    # 環境変数を読み込み
    load_env_file()
    
    # 利用可能なモデルを取得
    best_model = list_available_models()
    
    if best_model:
        print(f"\n✨ 推奨モデル: {best_model}")
    else:
        print("\n⚠️ 推奨モデルが見つかりませんでした")

if __name__ == "__main__":
    main()