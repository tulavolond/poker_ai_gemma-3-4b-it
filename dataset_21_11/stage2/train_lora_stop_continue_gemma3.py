import os, json, torch, time, signal, sys
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM, Trainer,
    TrainingArguments, DataCollatorForSeq2Seq,
    TrainerCallback, TrainerState, BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from dotenv import load_dotenv

# ==============================================================================
# ВЫБЕРИТЕ МОДЕЛЬ (раскомментируйте одну строку):
# ==============================================================================
ACTIVE_MODEL = "google/gemma-3-1b-it"       # Вариант 1: ~8-12 мин, fp16 без 4-bit (~3.5 ГБ VRAM)
# ACTIVE_MODEL = "Qwen/Qwen2.5-3B-Instruct" # Вариант 2: ~20-25 мин, 4-bit (~4.8 ГБ VRAM)
# ACTIVE_MODEL = "google/gemma-3-4b-it"     # Вариант 3: ~45-60 мин, 4-bit (~5.8 ГБ VRAM)
# ==============================================================================

HERE = Path(__file__).resolve().parent
DATA = HERE / "train_bet.jsonl"
SHORT_NAME = ACTIVE_MODEL.split("/")[-1].lower()
OUT = str(HERE / f"{SHORT_NAME}-bet-lora")
OUTPUT_CHECK_DIR = str(HERE / f"outputs_poker_{SHORT_NAME}")

MAX_LEN = 480
NUM_EPOCHS = 3

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

tok = AutoTokenizer.from_pretrained(ACTIVE_MODEL, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# def format_time(seconds):
#     if seconds < 60:
#         return f"{seconds:.0f} сек"
#     elif seconds < 3600:
#         return f"{seconds / 60:.1f} мин"
#     h = int(seconds // 3600)
#     m = int((seconds % 3600) // 60)
#     return f"{h} ч {m} мин"



# Ваша функция форматирования времени
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
    msgs, ids, labels, prev = ex["messages"], [], [], ""
    for i, m in enumerate(msgs):
        cur = tok.apply_chat_template(msgs[:i+1], tokenize=False, add_generation_prompt=False)
        seg_ids = tok(cur[len(prev):], add_special_tokens=False)["input_ids"]
        prev = cur
        ids += seg_ids
        labels += seg_ids if m["role"] == "assistant" else [-100] * len(seg_ids)
    n = min(len(ids), MAX_LEN)
    return {"input_ids": ids[:n], "labels": labels[:n], "attention_mask": [1] * n}

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
assert any("пот-оддсы" in ex["messages"][1]["content"] for ex in rows), "Это не stage2-данные!"

enc = [encode(ex) for ex in rows]
sup = [sum(1 for x in e["labels"] if x != -100) for e in enc]
print(f"[TRAIN] Модель: {ACTIVE_MODEL}")
print(f"[TRAIN] Строк: {len(rows)} | сред. длина: {sum(len(e['input_ids']) for e in enc)//len(enc)} ток. "
      f"| обучаемых: {sum(sup)//len(sup)} ток.")
ds = Dataset.from_list(enc)

# ==============================================================================
# АВТОНАСТРОЙКА ЗАГРУЗКИ ПОД GTX 1070 (8 ГБ VRAM, Pascal)
# ==============================================================================
is_gemma3 = "gemma-3" in ACTIVE_MODEL.lower()
use_4bit = "1b" not in ACTIVE_MODEL.lower()  # 1B грузим в чистом fp16, 3B/4B — в 4-bit

bnb_config = None
if use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

# Для Gemma-3-4B берем текстовый класс Gemma3ForCausalLM (без лишнего SigLIP)
# Для Gemma-3-1B и Qwen2.5-3B берем AutoModelForCausalLM
ModelClass = Gemma3ForCausalLM if (is_gemma3 and "4b" in ACTIVE_MODEL.lower()) else AutoModelForCausalLM
attn_impl = "eager" if is_gemma3 else "sdpa"

model = ModelClass.from_pretrained(
    ACTIVE_MODEL,
    quantization_config=bnb_config,
    dtype=torch.float16,
    token=HF_TOKEN,
    attn_implementation=attn_impl,
    device_map={"": 0}
)
model.config.use_cache = False

if use_4bit:
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
else:
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
))

# Для чистого fp16 (без 4-bit) переводим веса LoRA в fp32 для стабильности GradScaler
if not use_4bit:
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUTPUT_CHECK_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    max_grad_norm=1.0,
    logging_steps=10,
    save_strategy="steps",
    save_steps=25,               # страховка каждые 25 шагов + по Ctrl+C
    save_total_limit=2,
    dataloader_num_workers=0,
    report_to="none",
    seed=42
)

last_ckpt = get_last_checkpoint(OUTPUT_CHECK_DIR) if os.path.isdir(OUTPUT_CHECK_DIR) else None
if last_ckpt:
    st = TrainerState.load_from_json(os.path.join(last_ckpt, "trainer_state.json"))
    print("=" * 80)
    print(f"▶️ Продолжаю с {last_ckpt} (шаг {st.global_step}/{st.max_steps})")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, pad_to_multiple_of=8),
    callbacks=[CtrlCAndEtaCallback()]
)

trainer.train(resume_from_checkpoint=last_ckpt)
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print(f"✅ Готово! Адаптер сохранён в: {OUT}")
