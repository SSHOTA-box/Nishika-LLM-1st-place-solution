import os
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json

# ==========================================
# 設定エリア
# ==========================================

def get_abs_path(rel_path):
    """settings.jsonの相対パスを絶対パスに変換"""
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))

# ==========================================
# 0. settings.json 読み込み & パス解決ロジック
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")

print(f"Loading settings from: {SETTINGS_FILE}")
if not os.path.exists(SETTINGS_FILE):
    if os.path.exists("settings.json"):
        SETTINGS_FILE = "settings.json"
        PROJECT_ROOT = os.getcwd()
    else:
        raise FileNotFoundError(f"settings.json not found at {SETTINGS_FILE}")

with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

# ==========================================
# ★設定: 2つの入力ファイル
# ==========================================
OUTPUT_BASE_DIR_v1 = get_abs_path(SETTINGS["OUTPUT_PREDICTv1"])
OUTPUT_BASE_DIR_v2 = get_abs_path(SETTINGS["OUTPUT_PREDICTv2"])

# 読み込む2つのCSVファイルを指定してください
# ※ 同じ形式（ID, candidate_0...candidate_99）であることを前提とします
INPUT_CSV_PATH_1 = os.path.join(OUTPUT_BASE_DIR_v1, "candidates_centroid.csv")
INPUT_CSV_PATH_2 = os.path.join(OUTPUT_BASE_DIR_v2, "candidates_centroid.csv")

# 出力設定
OUTPUT_DIR = get_abs_path(SETTINGS["OUTPUT_FINAL_SUBMIT"])
OUTPUT_FILENAME = "final_submission_ensemble_200.csv"

# 評価モデル (Predictor)
PREDICTOR_MODEL_PATH = get_abs_path(SETTINGS["PREDICTOR_MODEL_PATH"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 関数定義
# ==========================================

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(text_list, tokenizer, model, device):
    """
    リスト内のテキストをバッチ処理で埋め込みベクトル化します。
    メモリ不足になる場合はバッチサイズを調整する処理を追加してください。
    """
    text_list = [str(t) if pd.notna(t) else "" for t in text_list]
    
    # 200件程度なら一度に処理できる想定ですが、VRAMが厳しい場合は分割してください
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def main():
    print(f"=== Starting Ensemble Centroid Selection (Using ALL 200 candidates) ===")
    
    # 1. パス確認
    if not os.path.exists(INPUT_CSV_PATH_1):
        print(f"Error: Input CSV 1 not found -> {INPUT_CSV_PATH_1}")
        return
    if not os.path.exists(INPUT_CSV_PATH_2):
        print(f"Error: Input CSV 2 not found -> {INPUT_CSV_PATH_2}")
        return

    # 出力先作成
    output_dir = os.path.join(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    
    print(f"Input 1: {INPUT_CSV_PATH_1}")
    print(f"Input 2: {INPUT_CSV_PATH_2}")
    print(f"Output: {final_output_path}")

    # 2. データ読み込み & マージ
    print("Loading and merging CSVs...")
    df1 = pd.read_csv(INPUT_CSV_PATH_1)
    df2 = pd.read_csv(INPUT_CSV_PATH_2)

    # IDで結合 (inner join)
    # suffixesにより、重複する列名（candidate_0など）には _1, _2 が付きます
    df = pd.merge(df1, df2, on="ID", how="inner", suffixes=('_1', '_2'))
    
    # "candidate_" を含むすべての列を抽出（これで両ファイルの全候補が対象になります）
    candidate_cols = [col for col in df.columns if "candidate_" in col]
    
    print(f"Merged Data Rows: {len(df)}")
    print(f"Total candidates per row: {len(candidate_cols)} (Expected: 200 if 100*2)")

    # 3. モデルロード
    print("Loading Predictor Model...")
    tokenizer = AutoTokenizer.from_pretrained(PREDICTOR_MODEL_PATH)
    model = AutoModel.from_pretrained(PREDICTOR_MODEL_PATH).to(DEVICE)
    model.eval()

    results = []

    # 4. メインループ
    for _, row in tqdm(df.iterrows(), total=len(df)):
        item_id = row["ID"]
        
        # 全候補リスト作成 (NaNは除外または空文字扱い)
        candidates = row[candidate_cols].fillna("").astype(str).tolist()

        try:
            # --- 埋め込み取得 (全200件) ---
            cand_embs = get_embeddings(candidates, tokenizer, model, DEVICE) 
            
            # --- 重心計算 (全200件を使用) ---
            # ここではフィルタリングせず、すべての候補の平均ベクトルを「重心」とします
            centroid = torch.mean(cand_embs, dim=0, keepdim=True)
            centroid = F.normalize(centroid, p=2, dim=1)

            # --- ベスト選定 ---
            # 重心とのコサイン類似度を計算
            scores = torch.mm(centroid, cand_embs.transpose(0, 1)).squeeze(0)
            
            # 最もスコアが高い（重心に近い）インデックスを取得
            best_idx = torch.argmax(scores).item()
            best_caption = candidates[best_idx]

            # --- 後処理 ---
            if not best_caption or best_caption.lower() == "nan":
                best_caption = "魅力的な商品です。"
            
            if len(best_caption) > 100:
                best_caption = best_caption[:100]

        except Exception as e:
            print(f"Error processing {item_id}: {e}")
            best_caption = candidates[0] if candidates else "Error"

        results.append({
            "ID": item_id,
            "target": best_caption
        })

    # 5. 保存
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(final_output_path, index=False, encoding='utf-8-sig')
    
    print(f"\nDone! Saved ensemble submission to:\n{final_output_path}")

if __name__ == "__main__":
    main()