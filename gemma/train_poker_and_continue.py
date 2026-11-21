# train_poker.py
import os
import signal

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback, TrainerState
from transformers.trainer_utils import get_last_checkpoint

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ─────────── настройки ───────────
BASE_MODEL  = "google/gemma-3-4b-it"
DATA_FILE   = "./gemma/dataset_2000_v2.json"
OUTPUT_DIR  = "outputs_poker_gemma"   # папка чекпоинтов — не менять между запусками!
FINAL_DIR   = "poker_gemma_lora"      # итоговые LoRA-адаптеры
MAX_SEQ_LEN = 768
MAX_STEPS   = 3500    # эпох 3
SAVE_EVERY  = 100     # автосохранение каждые N шагов
KEEP_LAST   = 3       # сколько последних чекпоинтов хранить (None = хранить все)
# ─────────────────────────────────


class CtrlCStop(TrainerCallback):
    """1-й Ctrl+C: доделать шаг -> сохранить чекпоинт -> выйти. 2-й: аварийный выход."""

    def __init__(self):
        self.requested = False
        self._old_handler = None

    def _handle(self, signum, frame):
        if self.requested:
            print("\n💥 Второй Ctrl+C — выхожу немедленно, без сохранения.")
            signal.signal(signal.SIGINT, self._old_handler or signal.default_int_handler)
            raise KeyboardInterrupt
        self.requested = True
        print("\n⏸ Ctrl+C принят. Доделываю шаг и сохраняю чекпоинт — подожди, не жми повторно.")

    def on_train_begin(self, args, state, control, **kwargs):
        self._old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)

    def on_step_end(self, args, state, control, **kwargs):
        if self.requested:
            control.should_save = True            # сохранить checkpoint-<текущий шаг>
            control.should_training_stop = True   # корректно выйти из цикла обучения

    def on_train_end(self, args, state, control, **kwargs):
        if self._old_handler is not None:
            signal.signal(signal.SIGINT, self._old_handler)


# 1. Модель и токенизатор (4 бита)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,                 # авто: fp16 на Pascal, bf16 на Ampere+
    load_in_4bit=True,
    token=HF_TOKEN,             # переменная, а не строка "HF_TOKEN"
)

# 2. LoRA-адаптеры (база заморожена)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # без пробелов внутри кавычек!
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. Датасет
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
print(f"📄 Примеров: {len(dataset)} | колонки: {dataset.column_names}")

# 4. Тренер
stop_cb = CtrlCStop()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=1,          # на Windows >1 иногда подвисает
    packing=False,
    callbacks=[stop_cb],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=MAX_STEPS,
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="steps",
        save_steps=SAVE_EVERY,
        save_total_limit=KEEP_LAST,
        save_only_model=False,   # ВАЖНО: нужны optimizer.pt / scheduler.pt / rng_state.pth
        report_to="none",
        dataloader_num_workers=0,
    ),
)

# 5. Автовозобновление
last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
if last_ckpt:
    st = TrainerState.load_from_json(os.path.join(last_ckpt, "trainer_state.json"))
    print("="*80)
    print(f"▶️ Продолжаю с {last_ckpt} (шаг {st.global_step}/{MAX_STEPS})")
    print("-"*80)
    if st.global_step >= MAX_STEPS:
        print("⚠️ Эти шаги уже пройдены — чтобы учить дальше, увеличь MAX_STEPS.")
        print("="*80)
else:
    print("🚀 Чекпоинтов нет, начинаю с нуля.")

trainer.train(resume_from_checkpoint=last_ckpt)

# 6. Страховка + сохранение адаптеров
step = trainer.state.global_step
if stop_cb.requested and not os.path.isdir(os.path.join(OUTPUT_DIR, f"checkpoint-{step}")):
    try:                                   # если мягкая остановка не совпала с save_steps
        trainer._save_checkpoint(trainer.model, trial=None)
        print(f"💾 Дополнительно сохранён checkpoint-{step}")
    except Exception as e:
        print("Не удалось сохранить чекпоинт:", e)

model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"✅ Шаг {step}/{MAX_STEPS}. Адаптеры сохранены в '{FINAL_DIR}'")