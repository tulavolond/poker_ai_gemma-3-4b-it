# train_lora.py
# python ./dataset_21_11/stage1_v5/eval_fast.py ./dataset_21_11/stage1_v5/qwen25-stage1-lora dev.jsonl

import os, json, torch, time, signal
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback, TrainerState)
from peft import LoraConfig, get_peft_model
from pathlib import Path; HERE = Path(__file__).resolve().parent
from transformers.trainer_utils import get_last_checkpoint
from tqdm import tqdm
# import signal
from dotenv import load_dotenv

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 1024

# DATA = HERE / "train_bet.jsonl"
# OUT  = str(HERE / "qwen25-bet-lora")     # на этот раз меняем имя!
# num_train_epochs = 2                    # данных больше, loss и так был 0.048

DATA = HERE / "train_preflop.jsonl"
OUT  = str(HERE / "qwen25-preflop-lora")
num_train_epochs = 4  # Данных меньше, чем в Stage 2, нужно чуть больше эпох
   
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

class CtrlCAndEtaCallback(TrainerCallback):
    def __init__(self):
        self.requested = False
        self._old_handler = None
        self.t0 = None
        self.s0 = 0

    def _handle(self, signum, frame):
        if self.requested:
            print("\n💥 Второй Ctrl+C — выхожу немедленно, без сохранения.")
            signal.signal(signal.SIGINT, self._old_handler or signal.default_int_handler)
            raise KeyboardInterrupt
        self.requested = True
        print("\n⏸ Ctrl+C принят. Доделываю текущий шаг и сохраняю чекпоинт — подождите...")

    def on_train_begin(self, args, state, control, **kwargs):
        self._old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        self.t0 = time.time()
        self.s0 = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        if self.requested:
            control.should_save = True
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return

        done = state.global_step - self.s0
        if done <= 0 or not state.max_steps:
            return

        sec_per_step = (time.time() - self.t0) / done
        left = (state.max_steps - state.global_step) * sec_per_step

        # --- БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ learning_rate ---
        lr_raw = logs.get('learning_rate', 0)
        
        # Если пришло число (float/int), используем его напрямую
        if isinstance(lr_raw, (float, int)):
            lr_value = float(lr_raw)
        # Если пришла строка, пытаемся извлечь из неё число
        elif isinstance(lr_raw, str):
            try:
                lr_value = float(lr_raw.split()[0])
            except (ValueError, IndexError):
                lr_value = 0.0
        else:
            lr_value = 0.0
        # --- КОНЕЦ БЛОКА ИЗВЛЕЧЕНИЯ ---

        # Форматируем до 4 значащих цифр (9.054e-05)
        lr_formatted = f"{lr_value:.4g}" 

        eta_str = f"⏱️ {sec_per_step:.1f} с/шаг | осталось ~{format_time(left)}"
        
        # Обновляем логи. Trainer напечатает это одной строкой.
        logs['learning_rate'] = f"{lr_formatted} {eta_str}"

    def on_train_end(self, args, state, control, **kwargs):
        if self._old_handler is not None:
            signal.signal(signal.SIGINT, self._old_handler)


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

# assert any("Скан:" in ex["messages"][1]["content"] for ex in rows), "это не v7-данные!"

# assert any("пот-оддсы" in ex["messages"][1]["content"] for ex in rows), "это не stage2-данные!"

assert any("Категория руки" in ex["messages"][1]["content"] for ex in rows), "это не stage3-данные!"

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

last_ckpt = get_last_checkpoint(OUT) if os.path.isdir(OUT) else None
if last_ckpt:
    st = TrainerState.load_from_json(os.path.join(last_ckpt, "trainer_state.json"))
    print("=" * 80)
    print(f"▶️ Продолжаю с {last_ckpt} (шаг {st.global_step}/{st.max_steps})")
# 4. Тренер
#stop_cb = CtrlCStop() # callbacks=[stop_cb],
stop_cb = CtrlCAndEtaCallback()
trainer = Trainer(model=model, args=args, train_dataset=ds, callbacks=[CtrlCAndEtaCallback()],
        data_collator=DataCollatorForSeq2Seq(tok, padding="longest",
                                             label_pad_token_id=-100)).train(resume_from_checkpoint=last_ckpt)
# 6. Страховка + сохранение адаптеров
# step = trainer.state.global_step
# if stop_cb.requested and not os.path.isdir(os.path.join(OUT, f"checkpoint-{step}")):
#     try:                                   # если мягкая остановка не совпала с save_steps
#         trainer._save_checkpoint(trainer.model, trial=None)
#         print(f"💾 Дополнительно сохранён checkpoint-{step}")
#     except Exception as e:
#         print("Не удалось сохранить чекпоинт:", e)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"✅ сохранены в '{OUT}'")
train_loss = trainer.metrics.get("train_loss")  # может не быть при resume "в самый конец
print()
print("=" * 60)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 60)
print(f"📉 Train loss: {train_loss:.8f}" if train_loss is not None else "📉 Train loss: n/a")