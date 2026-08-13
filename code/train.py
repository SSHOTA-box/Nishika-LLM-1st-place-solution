import os
import gc
import json
import yaml
import inspect
import torch
import pandas as pd
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List
from tqdm import tqdm

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from qwen_vl_utils import process_vision_info

# Historical local source filename:
# train_val_main_Qwen3-VL-8B-Instruct.py

# ==========================================
# 設定エリア (settings.json パス解決版)
# ==========================================

# 1. パスの基準点を設定（実行場所依存を排除）
# .../code/train.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# .../code の親ディレクトリ (プロジェクトルート)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# settings.json の場所
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")

print(f"Script Directory: {SCRIPT_DIR}")
print(f"Project Root:     {PROJECT_ROOT}")
print(f"Loading settings from: {SETTINGS_FILE}")

if not os.path.exists(SETTINGS_FILE):
    raise FileNotFoundError(f"settings.json not found at {SETTINGS_FILE}")

with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

# --- パス解決用ヘルパー ---
def get_abs_path(rel_path):
    """settings.json内の相対パスを絶対パスに変換"""
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))

# --- パス設定 ---
DATA_DIR = get_abs_path(SETTINGS["DATA_DIR"])
TRAIN_IMG_DIR = get_abs_path(SETTINGS["IMAGE_TRAIN_DIR"])

# 出力先：settings.jsonのLoRA_DIRの下に詳細パスを切る
OUTPUT_BASE = get_abs_path(SETTINGS["LoRA_DIR"])
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "LoRA_Qwen3-VL-8B-Instruct")

# モデルパス
QWEN_LOCAL_PATH = get_abs_path(SETTINGS["QWEN_MODEL_PATH"])
QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# 主催のtext encoder（predictor）
LOCAL_PREDICTOR_PATH = get_abs_path(SETTINGS["PREDICTOR_MODEL_PATH"])

#SEED値
SEED = SETTINGS.get("SEED", 42)


# --- ハイパーパラメータ（変更なし） ---

# 末尾val件数（末尾300をeval）
VAL_SIZE = 300

# eval時の生成/埋め込み設定
EVAL_MAX_NEW_TOKENS = 150
EMB_BATCH_SIZE = 8  # OOMなら4

GENERATION_INSTRUCTION = """
この商品の魅力が伝わるようなキャプションを日本語で、100文字以内で生成してください。
"""

# --- LoRA & 学習ハイパーパラメータ ---
BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
NUM_EPOCHS = 4
LEARNING_RATE = 1e-4
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 2048

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2",
    "model.visual.deepstack_merger_list.0.linear_fc1",
    "model.visual.deepstack_merger_list.0.linear_fc2",
    "model.visual.deepstack_merger_list.1.linear_fc1",
    "model.visual.deepstack_merger_list.1.linear_fc2",
    "model.visual.deepstack_merger_list.2.linear_fc1",
    "model.visual.deepstack_merger_list.2.linear_fc2",
]

# 2. 層ごとのRank指定
rank_pattern_config = {
    # --- Visual Layers (画像認識) ---
    "model.visual.deepstack": 16,
    "model.visual.merger": 16,

    # --- LLM Layers (言語生成) ---
    # FFN (知識・表現)
    "gate_proj": 64,
    "up_proj": 64,
    "down_proj": 64,
    # Attention (文脈・構成)
    "q_proj": 32,
    "k_proj": 32,
    "v_proj": 32,
    "o_proj": 32,
}

# 3. 層ごとのAlpha指定
alpha_pattern_config = {
    # --- Visual Layers ---
    "model.visual.deepstack": 32,  # 64 * 2
    "model.visual.merger": 32,      # 16 * 2

    # --- LLM Layers ---
    "gate_proj": 128,                # 16 * 2
    "up_proj": 128,
    "down_proj": 128,
    "q_proj": 64,
    "k_proj": 64,
    "v_proj": 64,
    "o_proj": 64,
}

# config保存用
EXPERIMENT_CONFIG = {
    "model_id": QWEN_MODEL_ID,
    "qwen_local_path": QWEN_LOCAL_PATH,
    "project_root": PROJECT_ROOT, # 記録用
    "max_pixels": 1024 * 28 * 28,
    "min_pixels": 256 * 28 * 28,
    "batch_size": BATCH_SIZE,
    "grad_accumulation": GRAD_ACCUMULATION,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "target_modules": TARGET_MODULES,
    "max_seq_length": MAX_SEQ_LENGTH,
    "seed": SEED,
    "prompt": GENERATION_INSTRUCTION.strip(),
    "val_size": VAL_SIZE,
    "predictor_path": LOCAL_PREDICTOR_PATH,
    "eval_max_new_tokens": EVAL_MAX_NEW_TOKENS,
    "emb_batch_size": EMB_BATCH_SIZE,
}

MAX_PIXEL = EXPERIMENT_CONFIG["max_pixels"]
MIN_PIXEL = EXPERIMENT_CONFIG["min_pixels"]


# ==========================================
# left paddingを確実に統一
# ==========================================
def set_left_padding(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# TrainingArguments の互換吸収
# ==========================================
def build_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    else:
        kwargs["eval_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


# ==========================================
# 関数定義
# ==========================================
def find_images_grouped(df, image_root):
    """IDごとに画像をリストにまとめる"""
    print(f"Scanning image files in {image_root}...")
    exts = {"jpg", "jpeg", "png", "bmp", "JPG", "PNG"}
    grouped_list = []

    id_to_caption = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))

    for _, row in df.iterrows():
        target_id = str(row.iloc[0])
        target_img_folder = os.path.join(image_root, target_id)
        if not os.path.isdir(target_img_folder):
            continue

        img_paths = []
        with os.scandir(target_img_folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.split(".")[-1] in exts:
                    img_paths.append(entry.path)

        if img_paths:
            img_paths.sort()
            grouped_list.append(
                {"id": target_id, "image_paths": img_paths, "caption": id_to_caption.get(target_id, "")}
            )
    return grouped_list


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@torch.no_grad()
def get_embeddings_batched(text_list: List[str], tokenizer, model, device, batch_size=8):
    """主催text encoder方式：mean_pooling + normalize"""
    embs = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i: i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        outputs = model(**inputs)
        e = mean_pooling(outputs, inputs["attention_mask"])
        e = F.normalize(e, p=2, dim=1)
        embs.append(e)
    return torch.cat(embs, dim=0)


def norm100(s: str) -> str:
    s = s.replace("\n", "").strip().strip('「」『』"\' ')
    if len(s) > 100:
        s = s[:100]
    return s


# ==========================================
# Callback（ログ保存）
# ==========================================
class LogSavingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_file = os.path.join(args.output_dir, "training_logs.jsonl")
            if "epoch" not in logs:
                logs["epoch"] = state.epoch
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(logs, ensure_ascii=False) + "\n")


# ==========================================
# Dataset / Collator
# ==========================================
class QwenVLDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # user content（複数画像 + 指示文）
        content_list = []
        for img_path in item["image_paths"]:
            content_list.append({"type": "image", "image": img_path})
        content_list.append({"type": "text", "text": GENERATION_INSTRUCTION})

        conversation_full = [
            {"role": "user", "content": content_list},
            {"role": "assistant", "content": [{"type": "text", "text": item["caption"]}]},
        ]
        conversation_prompt = [{"role": "user", "content": content_list}]

        prompt_text = self.processor.apply_chat_template(
            conversation_prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            conversation_full, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(conversation_full)

        # prompt token化（長さ取得用）
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

        # full token化（学習用）
        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        pixel_values = full_inputs.get("pixel_values", None)
        image_grid_thw = full_inputs.get("image_grid_thw", None)

        labels = input_ids.clone()

        # prompt部分を loss から除外（回答だけ学習）
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100

        # PAD除外
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        # ★最小変更：sample_id を返す（TOP10表示用）
        return {
            "sample_id": item["id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


@dataclass
class QwenDataCollator:
    processor: AutoProcessor

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        sample_ids = [f.get("sample_id", None) for f in features]

        pad_id = self.processor.tokenizer.pad_token_id
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")

        if padding_side == "left":
            max_len = max(x.size(0) for x in input_ids)

            def left_pad_1d(x, pad_value):
                pad_len = max_len - x.size(0)
                if pad_len <= 0:
                    return x
                pad = torch.full((pad_len,), pad_value, dtype=x.dtype)
                return torch.cat([pad, x], dim=0)

            input_ids_padded = torch.stack([left_pad_1d(x, pad_id) for x in input_ids], dim=0)
            attention_mask_padded = torch.stack([left_pad_1d(x, 0) for x in attention_mask], dim=0)
            labels_padded = torch.stack([left_pad_1d(x, -100) for x in labels], dim=0)
        else:
            input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
            attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "sample_id": sample_ids,
        }

        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)

        return batch


# ==========================================
# Trainer拡張：eval時に generate→cosine + 低い順TOP10表示
# ==========================================
class PRCosineEvalTrainer(Trainer):
    def __init__(
        self,
        *args,
        processor=None,
        pred_tokenizer=None,
        pred_model=None,
        eval_max_new_tokens=128,
        emb_batch_size=8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.pred_tokenizer = pred_tokenizer
        self.pred_model = pred_model
        self.eval_max_new_tokens = eval_max_new_tokens
        self.emb_batch_size = emb_batch_size

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        self.model.eval()

        device_gen = self.model.device
        device_pred = next(self.pred_model.parameters()).device

        preds_texts, gts_texts, ids = [], [], []
        dataloader = self.get_eval_dataloader(eval_dataset)

        for batch in tqdm(dataloader, desc="eval-generate"):
            sample_ids = batch.get("sample_id", None)

            batch = {
                k: (v.to(device_gen) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k != "sample_id"
            }
            bs = batch["input_ids"].size(0)

            if batch.get("pixel_values", None) is not None and bs != 1:
                raise ValueError(
                    "per_device_eval_batch_size=1 を推奨（collatorがpixel_valuesをcatしているため）"
                )

            for i in range(bs):
                input_ids_i = batch["input_ids"][i]
                attn_i = batch["attention_mask"][i]
                labels_i = batch["labels"][i]

                # ---- prompt終端推定 ----
                real_pos = torch.where(attn_i == 1)[0]
                ans_pos = real_pos[labels_i[real_pos] != -100]
                if ans_pos.numel() == 0:
                    prompt_end = int(real_pos[-1].item()) + 1
                else:
                    prompt_end = int(ans_pos[0].item())

                prompt_input_ids = input_ids_i[:prompt_end].unsqueeze(0)
                prompt_attention_mask = attn_i[:prompt_end].unsqueeze(0)

                # ---- generate ----
                gen_ids = self.model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    pixel_values=batch.get("pixel_values", None),
                    image_grid_thw=batch.get("image_grid_thw", None),
                    max_new_tokens=self.eval_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )

                if gen_ids.shape[1] > prompt_input_ids.shape[1]:
                    gen_only = gen_ids[:, prompt_input_ids.shape[1]:]
                else:
                    gen_only = gen_ids

                pred_text = self.processor.tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()
                pred_text = norm100(pred_text)

                gt_ids = labels_i[labels_i != -100]
                gt_text = self.processor.tokenizer.decode(gt_ids, skip_special_tokens=True).strip()
                gt_text = norm100(gt_text)

                preds_texts.append(pred_text)
                gts_texts.append(gt_text)

                if isinstance(sample_ids, list) and len(sample_ids) > i:
                    ids.append(sample_ids[i])
                else:
                    ids.append(None)

        # ---- 埋め込み→cosine ----
        pred_emb = get_embeddings_batched(
            preds_texts, self.pred_tokenizer, self.pred_model, device_pred, self.emb_batch_size
        )
        gt_emb = get_embeddings_batched(
            gts_texts, self.pred_tokenizer, self.pred_model, device_pred, self.emb_batch_size
        )

        cos_each = (pred_emb * gt_emb).sum(dim=1)
        cosine_mean = cos_each.mean().item()

        metrics = {f"{metric_key_prefix}_cosine": cosine_mean, f"{metric_key_prefix}_n": len(preds_texts)}
        self.log(metrics)

        # ---- TOP10表示 ----
        k = min(10, cos_each.numel())
        vals, idx = torch.sort(cos_each)
        print(f"\n[{metric_key_prefix}] bottom-{k} cosine samples (low -> high):")
        bottom_rows = []
        for r in range(k):
            j = int(idx[r].item())
            row = {
                "rank": r + 1,
                "cosine": float(vals[r].item()),
                "sample_id": ids[j],
                "pred": preds_texts[j],
                "gt": gts_texts[j],
            }
            bottom_rows.append(row)
            print(f"#{row['rank']:02d} cosine={row['cosine']:.4f} id={row['sample_id']}")
            print(f"  pred: {row['pred']}")
            print(f"  gt  : {row['gt']}")
            print("")

        try:
            out_path = os.path.join(self.args.output_dir, f"{metric_key_prefix}_bottom10.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for row in bottom_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{metric_key_prefix}] saved bottom-{k} -> {out_path}\n")
        except Exception as e:
            print(f"[{metric_key_prefix}] failed to save bottom10: {e}")

        del pred_emb, gt_emb, cos_each
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics


# ==========================================
# main
# ==========================================
def main():
    set_seed(SEED)
    print("=== Starting Qwen-VL LoRA Fine-tuning + val cosine (manual best) ===")
    
    # OUTPUT_DIR作成（絶対パス）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # config保存
    config_path = os.path.join(OUTPUT_DIR, "experiment_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(EXPERIMENT_CONFIG, f, allow_unicode=True, default_flow_style=False)
    print(f"Saved experiment config to {config_path}")

    # ---------------------------------------------------------
    # 1) 生成モデル（Qwen-VL）ロード
    # ---------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
    )

    print(f"Loading Qwen model from: {QWEN_LOCAL_PATH}")
    model = AutoModelForVision2Seq.from_pretrained(
        QWEN_LOCAL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    processor = AutoProcessor.from_pretrained(
        QWEN_LOCAL_PATH,
        min_pixels=MIN_PIXEL,
        max_pixels=MAX_PIXEL,
        local_files_only=True,
    )

    set_left_padding(processor.tokenizer)

    # ---------------------------------------------------------
    # 2) LoRA
    # ---------------------------------------------------------
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        rank_pattern=rank_pattern_config,    # 追加：層ごとのRank
        alpha_pattern=alpha_pattern_config,  # 追加：層ごとのAlpha
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if getattr(model.config, "use_cache", None) is True:
        model.config.use_cache = False

    # ---------------------------------------------------------
    # 3) predictor（主催text encoder）ロード
    # ---------------------------------------------------------
    print(f"Loading Predictor from: {LOCAL_PREDICTOR_PATH}")
    device_pred = "cuda" if torch.cuda.is_available() else "cpu"
    pred_tokenizer = AutoTokenizer.from_pretrained(LOCAL_PREDICTOR_PATH)
    set_left_padding(pred_tokenizer)

    pred_model = AutoModel.from_pretrained(LOCAL_PREDICTOR_PATH).to(device_pred)
    pred_model.eval()

    # ---------------------------------------------------------
    # 4) データ split（末尾VAL_SIZEをval）
    # ---------------------------------------------------------
    train_csv_path = os.path.join(DATA_DIR, "train.csv")
    print(f"Loading data from: {train_csv_path}")
    train_df_all = pd.read_csv(train_csv_path)
    eval_df = train_df_all.tail(VAL_SIZE).reset_index(drop=True)
    train_df = train_df_all.iloc[:-VAL_SIZE].reset_index(drop=True)

    train_data_grouped = find_images_grouped(train_df, TRAIN_IMG_DIR)
    eval_data_grouped = find_images_grouped(eval_df, TRAIN_IMG_DIR)

    train_dataset = QwenVLDataset(train_data_grouped, processor)
    eval_dataset = QwenVLDataset(eval_data_grouped, processor)
    data_collator = QwenDataCollator(processor=processor)

    # ---------------------------------------------------------
    # 5) TrainingArguments
    # ---------------------------------------------------------
    args = build_training_args(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,
        load_best_model_at_end=False,

        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )

    # ---------------------------------------------------------
    # 6) Trainer
    # ---------------------------------------------------------
    trainer = PRCosineEvalTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LogSavingCallback()],
        processor=processor,
        pred_tokenizer=pred_tokenizer,
        pred_model=pred_model,
        eval_max_new_tokens=EVAL_MAX_NEW_TOKENS,
        emb_batch_size=EMB_BATCH_SIZE,
    )

    # ---------------------------------------------------------
    # 7) Train
    # ---------------------------------------------------------
    print("\nStarting training...")
    trainer.train()

    # ---------------------------------------------------------
    # 8) Save
    # ---------------------------------------------------------
    print(f"Saving final LoRA adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("Done!")


if __name__ == "__main__":
    main()
