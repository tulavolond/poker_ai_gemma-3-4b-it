# train_lora.py
# python ./dataset_21_11/stage1_v5/eval_fast.py ./dataset_21_11/stage1_v5/qwen25-stage1-lora dev.jsonl

import os, json, torch, time
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback)
from peft import LoraConfig, get_peft_model
from pathlib import Path; HERE = Path(__file__).resolve().parent
from tqdm import tqdm
# import signal
from dotenv import load_dotenv

# MODEL, DATA, OUT = "Qwen/Qwen2.5-1.5B-Instruct", "./dataset_21_11/poker.jsonl", "./dataset_21_11/qwen25-poker-lora"
# MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# DATA = HERE / "stage1_train.jsonl"   # было poker.jsonl / qwen25-poker-lora

# DATA = HERE / "train.jsonl"   # было poker.jsonl / qwen25-poker-lora
# OUT = str(HERE / "qwen25-stage1-lora")
# MAX_LEN = 1024

# DATA = HERE / "train_v6.jsonl"
# OUT  = str(HERE / "qwen25-v61-lora")
MAX_LEN = 1024
# num_train_epochs = 3        # ответы короткие, нужно закрепить правило сравнения

DATA = HERE / "train_v7.jsonl"
OUT  = str(HERE / "qwen25-v7-lora")     # на этот раз меняем имя!
num_train_epochs = 2                    # данных больше, loss и так был 0.048

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN, trust_remote_code=True)

def format_time(seconds):
    """Форматирует секунды в читаемый вид."""
    if seconds < 60:
        return f"{seconds:.0f} сек"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} мин"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h} ч {m} мин"


# class CtrlCStop(TrainerCallback):
#     """1-й Ctrl+C: доделать шаг -> сохранить чекпоинт -> выйти. 2-й: аварийный выход."""

#     def __init__(self):
#         self.requested = False
#         self._old_handler = None

#     def _handle(self, signum, frame):
#         if self.requested:
#             print("\n💥 Второй Ctrl+C — выхожу немедленно, без сохранения.")
#             signal.signal(signal.SIGINT, self._old_handler or signal.default_int_handler)
#             raise KeyboardInterrupt
#         self.requested = True
#         print("\n⏸ Ctrl+C принят. Доделываю шаг и сохраняю чекпоинт — подожди, не жми повторно.")

#     def on_train_begin(self, args, state, control, **kwargs):
#         self._old_handler = signal.getsignal(signal.SIGINT)
#         signal.signal(signal.SIGINT, self._handle)

#     def on_step_end(self, args, state, control, **kwargs):
#         if self.requested:
#             control.should_save = True            # сохранить checkpoint-<текущий шаг>
#             control.should_training_stop = True   # корректно выйти из цикла обучения

#     def on_train_end(self, args, state, control, **kwargs):
#         if self._old_handler is not None:
#             signal.signal(signal.SIGINT, self._old_handler)

class EtaCallback(TrainerCallback):
    """Живой ETA по фактической скорости (учитывает resume)."""
    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.s0 = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:   # печатаем только на train-логах, не на eval
            return
        done = state.global_step - self.s0
        if done > 0 and state.max_steps:
            sec_per_step = (time.time() - self.t0) / done
            left = (state.max_steps - state.global_step) * sec_per_step
            # print(f"⏱️ {sec_per_step:.1f} с/шаг, осталось ~{format_time(left)}")
            tqdm.write(f"⏱️ {sec_per_step:.1f} с/шаг, осталось ~{format_time(left)}")

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

# ds = Dataset.from_list([encode(json.loads(l)) for l in open(DATA, encoding="utf-8")])
# Замени строку создания ds на блок с печатью и предохранителем:
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
# assert any("масти" in ex["messages"][2]["content"] for ex in rows), \
#     f"В {DATA} нет скана — это не v5-данные!"

# assert any("Скан:" in ex["messages"][1]["content"] for ex in rows), "это не v6-данные!"

assert any("Скан:" in ex["messages"][1]["content"] for ex in rows), "это не v7-данные!"

enc = [encode(ex) for ex in rows]
sup = [sum(1 for x in e["labels"] if x != -100) for e in enc]
print(f"[TRAIN] {DATA}")
print(f"[TRAIN] rows={len(rows)} | tokens avg={sum(len(e['input_ids']) for e in enc)//len(enc)} "
      f"| supervised avg={sum(sup)//len(sup)}")
print(f"[TRAIN] system: {rows[0]['messages'][0]['content'][:70]}...")
ds = Dataset.from_list(enc)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16,        # Pascal: fp16-хранение, без bf16
    token=HF_TOKEN,
    attn_implementation="sdpa",              # flash_attention_2 на sm_61 недоступен
    device_map={"": 0})
model.config.use_cache = False

model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
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
    num_train_epochs=num_train_epochs,                     # 17 примеров -> ~60 шагов оптимизатора 12 epochs
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=5,
    fp16=True, bf16=False,                   # AMP + GradScaler, работает на Pascal
    optim="adamw_torch", max_grad_norm=1.0,
    logging_steps=10, save_strategy="epoch", save_total_limit=2,
    dataloader_num_workers=0,                # Windows: без лишних процессов
    report_to="none", seed=42)

# 4. Тренер
#stop_cb = CtrlCStop() # callbacks=[stop_cb],
Trainer(model=model, args=args, train_dataset=ds, callbacks=[EtaCallback()],
        data_collator=DataCollatorForSeq2Seq(tok, padding="longest",
                                             label_pad_token_id=-100)).train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)