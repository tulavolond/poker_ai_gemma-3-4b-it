# train_lora.py
import json, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

MODEL, DATA, OUT = "Qwen/Qwen2.5-1.5B-Instruct", "./dataset_21_11/poker.jsonl", "./dataset_21_11/qwen25-poker-lora"
MAX_LEN = 712

tok = AutoTokenizer.from_pretrained(MODEL)

def encode(ex):
    #Loss только на ответах ассистента: system/user маскируются -100
    msgs, ids, labels, prev = ex["messages"], [], [], ""
    for i, m in enumerate(msgs):
        cur = tok.apply_chat_template(msgs[:i+1], tokenize=False, add_generation_prompt=False)
        seg_ids = tok(cur[len(prev):], add_special_tokens=False)["input_ids"]
        prev = cur
        ids += seg_ids
        labels += seg_ids if m["role"] == "assistant" else [-100] * len(seg_ids)
    n = min(len(ids), MAX_LEN)
    return {"input_ids": ids[:n], "labels": labels[:n], "attention_mask": [1] * n}

ds = Dataset.from_list([encode(json.loads(l)) for l in open(DATA, encoding="utf-8")])

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16,        # Pascal: fp16-хранение, без bf16
    attn_implementation="sdpa",              # flash_attention_2 на sm_61 недоступен
    device_map={"": 0})
model.config.use_cache = False

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
for p in model.parameters():                 # LoRA-веса в fp32: шаги оптимизатора не теряются в fp16
    if p.requires_grad:
        p.data = p.data.float()
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.print_trainable_parameters()           # должно быть ~18M обучаемых (1.2%)

args = TrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    num_train_epochs=12,                     # 17 примеров -> ~60 шагов оптимизатора 12
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=5,
    fp16=True, bf16=False,                   # AMP + GradScaler, работает на Pascal
    optim="adamw_torch", max_grad_norm=1.0,
    logging_steps=1, save_strategy="epoch", save_total_limit=2,
    dataloader_num_workers=0,                # Windows: без лишних процессов
    report_to="none", seed=42)

Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding="longest",
                                             label_pad_token_id=-100)).train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)