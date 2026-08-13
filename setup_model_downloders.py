import os
import json
import sys

# ==========================================
# Fast transfer setting (must precede huggingface_hub import)
# ==========================================
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

def download_from_settings():
    # 1. パスの解決（codeフォルダから実行しても、1つ上のsettings.jsonを見つける）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 候補1: 同じディレクトリ
    candidate1 = os.path.join(current_dir, "settings.json")
    # 候補2: 1つ上のディレクトリ
    candidate2 = os.path.join(os.path.dirname(current_dir), "settings.json")

    if os.path.exists(candidate1):
        json_path = candidate1
        project_root = current_dir
    elif os.path.exists(candidate2):
        json_path = candidate2
        project_root = os.path.dirname(current_dir)
    else:
        print(f"[ERROR] settings.json not found in {current_dir} or its parent.")
        return

    # 2. settings.json の読み込み
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 3. ダウンロードリストの取得
    models_to_download = cfg.get("MODELS_TO_DOWNLOAD", [])
    if not models_to_download:
        print("No models found in 'MODELS_TO_DOWNLOAD' list.")
        return

    print(f"=== Starting Download Task (Root: {project_root}) ===")
    print("HF_TRANSFER enabled: high-speed download mode ON")

    for model in models_to_download:
        name = model["name"]
        repo = model["repo_id"]
        
        # json内の相対パスをプロジェクトルート基準の絶対パスに変換
        # "./" で始まっていてもいなくても結合できるように正規化
        rel_path = model["local_path"]
        target_dir = os.path.normpath(os.path.join(project_root, rel_path))
        
        print(f"\n--- Processing: {name} ---")
        print(f" Target: {target_dir}")
        
        os.makedirs(target_dir, exist_ok=True)

        try:
            print(f"[DOWNLOAD] Checking/Downloading {repo}...")
            snapshot_download(
                repo_id=repo,
                local_dir=target_dir,
                local_dir_use_symlinks=False,  # 実体を保存
                resume_download=True,          # 途中から再開
                # max_workers=8                # 必要なら並列数を指定
            )
            print(f"[SUCCESS] Verified {name}")
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")

if __name__ == "__main__":
    download_from_settings()
    print("\nAll check/download tasks completed.")
