import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import gc
import math
import json
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ==========================================
# 0. settings.json 読み込み & パス解決ロジック
# ==========================================
# スクリプトの場所 (.../code/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# プロジェクトルート (.../)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# settings.json のパス
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")

print(f"Loading settings from: {SETTINGS_FILE}")
if not os.path.exists(SETTINGS_FILE):
    # ルートから実行した場合のケア
    if os.path.exists("settings.json"):
        SETTINGS_FILE = "settings.json"
        PROJECT_ROOT = os.getcwd()
    else:
        raise FileNotFoundError(f"settings.json not found at {SETTINGS_FILE}")

with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

def get_abs_path(rel_path):
    """settings.jsonの相対パスを絶対パスに変換"""
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))

# ==========================================
# 1. 設定エリア
# ==========================================

# --- 再現性のためのシード設定 ---
SEED = SETTINGS.get("SEED", 42)

TOTAL_CANDIDATES = 100

# 一度にGPUで生成する数（VRAMに合わせて調整: 4~8推奨）
BATCH_SIZE = 25

# --- パス設定 (settings.jsonから取得) ---

# 学習済みLoRAアダプタ
# settings.jsonの "LoRA_DIR" をベースにし、個別のチェックポイントを指定
LORA_BASE_DIR = get_abs_path(SETTINGS["LoRA_DIR"])
# ★ここだけは実行ごとに変わるため、フォルダ名を指定してください
LORA_ADAPTER_PATH = get_abs_path(SETTINGS["LORA_ADAPTER"])

# モデル設定
LOCAL_QWEN_PATH = get_abs_path(SETTINGS["QWEN_MODEL_PATH"])
HF_QWEN_ID = "Qwen/Qwen3-VL-8B-Instruct"

# 評価モデル (Predictor)
LOCAL_PREDICTOR_PATH = get_abs_path(SETTINGS["PREDICTOR_MODEL_PATH"])

# データ設定
DATA_DIR = get_abs_path(SETTINGS["DATA_DIR"])
IMAGE_DIR = get_abs_path(SETTINGS["IMAGE_TEST_DIR"]) # test画像を使用
INPUT_CSV_PATH = os.path.join(DATA_DIR, "test.csv")

# 出力ファイル設定
OUTPUT_SUB_DIR = get_abs_path(SETTINGS["OUTPUT_PREDICTv1"])

SUBMISSION_FILE = os.path.join(OUTPUT_SUB_DIR, "predict.csv")
CONFIG_FILE = os.path.join(OUTPUT_SUB_DIR, "inference_config.yaml")
CANDIDATES_FILE = os.path.join(OUTPUT_SUB_DIR, "candidates_centroid.csv")

SAVE_INTERVAL = 10 # 20件ごとに保存

# --- 生成パラメータ ---
GEN_CONFIG = {
    "max_new_tokens": 200,
    "do_sample": True,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
}
SYSTEM_PROMPT = "この商品の魅力が伝わるようなキャプションを日本語で、100文字以内で生成してください。"

# Config保存用辞書
EXPERIMENT_CONFIG = {
    "seed": SEED,
    "total_candidates": TOTAL_CANDIDATES,
    "batch_size": BATCH_SIZE,
    "lora_adapter_path": LORA_ADAPTER_PATH,
    "base_model_path": LOCAL_QWEN_PATH,
    "predictor_path": LOCAL_PREDICTOR_PATH,
    "data_csv": INPUT_CSV_PATH,
    "image_dir": IMAGE_DIR,
    "system_prompt": SYSTEM_PROMPT,
    "generation_config": GEN_CONFIG,
    "output_file": SUBMISSION_FILE,
    "candidates_file": CANDIDATES_FILE,
    "project_root": PROJECT_ROOT
}

# ==========================================
# 2. 関数定義
# ==========================================

def find_images_precheck(df, image_root):
    """CSVのIDに基づいて画像を特定する関数"""
    print(f"Pre-scanning image files in {image_root}...")
    data_infos = []
    exts = {"jpg", "jpeg", "png", "bmp", "JPG", "PNG"}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        target_id = str(row.iloc[0])
        target_img_folder = os.path.join(image_root, target_id)
        
        if not os.path.isdir(target_img_folder):
            data_infos.append({"id": target_id, "images": []})
            continue

        found_images = []
        with os.scandir(target_img_folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.split('.')[-1] in exts:
                    found_images.append(entry.path)
        
        data_infos.append({"id": target_id, "images": sorted(found_images)})
    
    return data_infos

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(text_list, tokenizer, model, device):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    set_seed(SEED)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Submission Generation (Centroid Method) on {device} ===")
    print(f"Total Candidates: {TOTAL_CANDIDATES}, Batch Size: {BATCH_SIZE}")
    print(f"Output Directory: {OUTPUT_SUB_DIR}")
    
    # ディレクトリ作成
    os.makedirs(OUTPUT_SUB_DIR, exist_ok=True)

    # 設定(Config)をYAMLで保存
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(EXPERIMENT_CONFIG, f, allow_unicode=True, default_flow_style=False)
    print(f"Saved inference config to {CONFIG_FILE}")

    # ---------------------------------------------------------
    # A. 生成モデル (Qwen + LoRA) ロード
    # ---------------------------------------------------------
    if os.path.exists(LOCAL_QWEN_PATH):
        print(f"Loading Base Model from: {LOCAL_QWEN_PATH}")
        qwen_path = LOCAL_QWEN_PATH
        local_only = True
    else:
        print(f"Loading Base Model from HF: {HF_QWEN_ID}")
        qwen_path = HF_QWEN_ID
        local_only = False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    try:
        base_model = AutoModelForVision2Seq.from_pretrained(
            qwen_path, 
            quantization_config=bnb_config, 
            device_map="auto", 
            trust_remote_code=True,
            local_files_only=local_only
        )
        processor = AutoProcessor.from_pretrained(
            qwen_path, 
            min_pixels=256*28*28, 
            max_pixels=1024*28*28,
            trust_remote_code=True,
            local_files_only=local_only
        )
    except Exception as e:
        print(f"Error loading Base Qwen: {e}")
        return

    print(f"Loading LoRA Adapter from: {LORA_ADAPTER_PATH}")
    if not os.path.exists(LORA_ADAPTER_PATH):
        print(f"Error: LoRA path not found! -> {LORA_ADAPTER_PATH}")
        return
        
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    # ---------------------------------------------------------
    # B. 評価モデル (Predictor) ロード
    # ---------------------------------------------------------
    if os.path.exists(LOCAL_PREDICTOR_PATH):
        print(f"Loading Predictor from: {LOCAL_PREDICTOR_PATH}")
        pred_path = LOCAL_PREDICTOR_PATH
    else:
        # ローカルになければ警告またはHFパスへのフォールバック（今回は設定ファイル依存なのでそのまま）
        print(f"Warning: Predictor path not found locally -> {LOCAL_PREDICTOR_PATH}")
        pred_path = LOCAL_PREDICTOR_PATH 

    try:
        pred_tokenizer = AutoTokenizer.from_pretrained(pred_path)
        pred_model = AutoModel.from_pretrained(pred_path).to(device)
        pred_model.eval()
    except Exception as e:
        print(f"Error loading Predictor: {e}")
        return

    # ---------------------------------------------------------
    # C. データリスト取得 (CSV準拠)
    # ---------------------------------------------------------
    print(f"Loading CSV: {INPUT_CSV_PATH}")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: CSV not found -> {INPUT_CSV_PATH}")
        return

    target_df = pd.read_csv(INPUT_CSV_PATH)
    all_data = find_images_precheck(target_df, IMAGE_DIR)
    
    print(f"Start processing {len(all_data)} items...")
    
    # 結果保存用バッファ
    results = []
    all_candidates_results = []

    # ---------------------------------------------------------
    # D. 推論ループ
    # ---------------------------------------------------------
    for i, item in enumerate(tqdm(all_data)):
        item_id = item["id"]
        img_paths = item["images"]

        # 候補格納用辞書（初期化）
        current_candidates_data = {"ID": item_id}
        
        # ---------------------------------------------------------
        # 1. 生成 (N案) - ★ メモリ対策: バッチ分割ループ ★
        # ---------------------------------------------------------
        candidates = []

        try:
            if not img_paths:
                # 画像がない場合のダミー処理
                best_caption = "魅力的な商品です。"
                candidates = ["魅力的な商品です。"] * TOTAL_CANDIDATES
            else:
                content_list = []
                images = [Image.open(p) for p in img_paths]
                
                for img_path in img_paths:
                    content_list.append({"type": "image", "image": img_path})
                
                content_list.append({"type": "text", "text": SYSTEM_PROMPT})
                
                text_input = processor.apply_chat_template(
                    [{"role": "user", "content": content_list}], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = processor(
                    text=[text_input], 
                    images=images, 
                    padding=True, 
                    return_tensors="pt"
                ).to(model.device)

                # --- 分割生成ループ開始 ---
                while len(candidates) < TOTAL_CANDIDATES:
                    needed = TOTAL_CANDIDATES - len(candidates)
                    current_batch_size = min(BATCH_SIZE, needed)
                    
                    batch_gen_config = GEN_CONFIG.copy()
                    batch_gen_config["num_return_sequences"] = current_batch_size
                    
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, **batch_gen_config)
                    
                    generated_ids_trimmed = [ids[len(inputs.input_ids[0]):] for ids in generated_ids]
                    batch_candidates = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
                    
                    batch_candidates = [c.replace("\n", "").strip().strip('「」『』"\' ') for c in batch_candidates]
                    
                    candidates.extend(batch_candidates)

                    del generated_ids, generated_ids_trimmed, batch_candidates
                    torch.cuda.empty_cache()
                # --- 分割生成ループ終了 ---

            # 全候補を辞書に保存
            for idx, cand in enumerate(candidates):
                current_candidates_data[f"candidate_{idx}"] = cand

            # ---------------------------------------------------------
            # 2. 選抜 (Centroid Method)
            # ---------------------------------------------------------
            cand_embs = get_embeddings(candidates, pred_tokenizer, pred_model, device)
            
            centroid = torch.mean(cand_embs, dim=0, keepdim=True)
            centroid = F.normalize(centroid, p=2, dim=1)
            
            scores = torch.mm(centroid, cand_embs.transpose(0, 1)).squeeze(0)
            
            best_idx = torch.argmax(scores).item()
            best_caption = candidates[best_idx]
            
            current_candidates_data["selected_idx"] = best_idx
            
            if not best_caption:
                best_caption = "魅力的な商品です。"
            if len(best_caption) > 100:
                best_caption = best_caption[:100]

        except Exception as e:
            print(f"Error processing {item_id}: {e}")
            best_caption = "Error"
            for idx in range(TOTAL_CANDIDATES):
                current_candidates_data[f"candidate_{idx}"] = "Error"
            
        results.append({"ID": item_id, "target": best_caption})
        all_candidates_results.append(current_candidates_data)

        # こまめな保存 (SAVE_INTERVALごと)
        if (i + 1) % SAVE_INTERVAL == 0:
            pd.DataFrame(results).to_csv(SUBMISSION_FILE, index=False, encoding='utf-8-sig')
            pd.DataFrame(all_candidates_results).to_csv(CANDIDATES_FILE, index=False, encoding='utf-8-sig')

        # 定期的なメモリ掃除
        if (i + 1) % 5 == 0:
            if 'inputs' in locals(): del inputs
            if 'cand_embs' in locals(): del cand_embs
            if 'images' in locals(): del images
            if 'scores' in locals(): del scores
            if 'centroid' in locals(): del centroid
            gc.collect()
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # E. 最終保存
    # ---------------------------------------------------------
    df_submit = pd.DataFrame(results)
    df_submit.to_csv(SUBMISSION_FILE, index=False, encoding='utf-8-sig')
    
    df_candidates = pd.DataFrame(all_candidates_results)
    df_candidates.to_csv(CANDIDATES_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"Done! Submission file saved to: {SUBMISSION_FILE}")
    print(f"Candidates file saved to: {CANDIDATES_FILE}")
    print(f"Config file saved to: {CONFIG_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()