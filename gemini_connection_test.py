#!/usr/bin/env python3
"""
MyVoiceger Gemini API接続テストスクリプト
APIキーの読み取りとGemini APIへの接続検証を実行します。
"""

import os
import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# .envファイルから環境変数を読み込み
def load_env_file():
    """環境変数を.envファイルから読み込みます"""
    env_path = current_dir / '.env'
    
    if not env_path.exists():
        print(f"❌ .envファイルが見つかりません: {env_path}")
        return False
    
    print(f"📄 .envファイルを読み込み中: {env_path}")
    
    with open(env_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 空行やコメント行をスキップ
            if not line or line.startswith('#'):
                continue
            
            # KEY=VALUE形式の環境変数をパース
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                os.environ[key] = value
                print(f"  ✓ {key} = {value[:20]}{'...' if len(value) > 20 else ''}")
    
    return True

def test_gemini_connection():
    """Gemini API接続テストを実行します"""
    print("\n🔧 Geminiクライアント初期化...")
    
    try:
        # gemini_utilsから関数をインポート
        from gemini_utils import setup_gemini_client
        
        # Geminiクライアントの初期化
        client_config = setup_gemini_client()
        
        print("✅ Geminiクライアント初期化成功")
        print(f"  📦 クライアント: {type(client_config['client'])}")
        print(f"  🤖 モデル: {client_config['model'].model_name}")
        
        return client_config
        
    except ImportError as e:
        print(f"❌ google-generativeaiライブラリのインポートに失敗: {e}")
        return None
    except ValueError as e:
        print(f"❌ APIキーエラー: {e}")
        return None
    except Exception as e:
        print(f"❌ クライアント初期化エラー: {e}")
        return None

def test_simple_query(client_config):
    """シンプルなテストクエリでAPI接続を検証します"""
    if not client_config:
        return False
        
    print("\n💬 シンプルなテストクエリ実行...")
    
    try:
        model = client_config['model']
        
        # 簡単なテストクエリ
        test_prompt = "こんにちは！これはGemini APIの接続テストです。20文字以内で日本語で応答してください。"
        
        print(f"  📝 テストプロンプト: {test_prompt}")
        
        # API呼び出し
        response = model.generate_content(test_prompt)
        
        # レスポンス検証
        if response and hasattr(response, 'text'):
            response_text = response.text.strip()
            print(f"✅ API接続成功")
            print(f"  🤖 応答: {response_text}")
            return True
        else:
            print(f"⚠️  応答形式が不正です: {response}")
            return False
            
    except Exception as e:
        print(f"❌ API呼び出しエラー: {e}")
        return False

def run_comprehensive_test():
    """包括的なGemini APIテストを実行します"""
    print("🚀 MyVoiceger Gemini API接続テスト開始")
    print("=" * 50)
    
    # 1. .envファイル読み込み
    print("\n1️⃣ 環境変数ファイル読み込み")
    env_loaded = load_env_file()
    
    if not env_loaded:
        print("❌ 環境変数ファイル読み込みに失敗")
        return False
    
    # 2. Gemini APIキーの確認
    print("\n2️⃣ APIキー設定確認")
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✅ GEMINI_API_KEY設定済み: {api_key[:20]}...")
    else:
        print("❌ GEMINI_API_KEYが設定されていません")
        return False
    
    # 3. Geminiクライアント初期化
    print("\n3️⃣ Geminiクライアント初期化")
    client_config = test_gemini_connection()
    
    if not client_config:
        print("❌ クライアント初期化に失敗")
        return False
    
    # 4. API接続テスト
    print("\n4️⃣ API接続テスト")
    connection_success = test_simple_query(client_config)
    
    if not connection_success:
        print("❌ API接続テストに失敗")
        return False
    
    # 5. 結果報告
    print("\n" + "=" * 50)
    print("🎉 Gemini API接続テスト完了")
    print("✅ 全テスト項目が成功しました")
    print("\n📊 テスト結果サマリー:")
    print("  ✓ 環境変数ファイル読み込み")
    print("  ✓ APIキー設定確認")
    print("  ✓ Geminiクライアント初期化")
    print("  ✓ API接続テスト")
    
    return True

def main():
    """メイン実行関数"""
    try:
        success = run_comprehensive_test()
        
        if success:
            print("\n✨ Gemini APIキー接続テストの結果: 成功 - MyVoicegerプロジェクトのGemini APIが正常に設定され、接続が確認できました。")
            return True
        else:
            print("\n💥 Gemini APIキー接続テストの結果: 失敗 - 接続または初期化のどこかの段階でエラーが発生しました。")
            return False
            
    except Exception as e:
        print(f"\n💥 テスト実行中に予期しないエラーが発生しました: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)