import os
import torch
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# ============================================================
# Настройки
# ============================================================

MODEL_ID = "google/gemma-3-4b-it"
OUTPUT_DIR = "./poker_gemma3_demo_lora"
MAX_LENGTH = 256

INSTRUCTION = (
    "Ты — покерный калькулятор. Ответь коротко: комбинация и шанс."
)

# ============================================================
# Мини-датасет для обучения
# Формат: board, hand, answer, check
# check - слова, которые мы ожидаем увидеть в ответе после обучения
# ============================================================

TRAIN_EXAMPLES = [
    {
        "board": "4♠ 5♥ 4♦ J♣ 4♣",
        "hand": "5♣ 5♦",
        "answer": "Комбинация: фулл-хаус. Шанс победы очень высокий, примерно 95%.",
        "check": ["фулл", "%"],
    },
    {
        "board": "A♠ K♠ Q♠ J♠ 2♦",
        "hand": "T♠ 9♠",
        "answer": "Комбинация: роял-флеш. Шанс победы максимальный, примерно 100%.",
        "check": ["роял", "%"],
    },
    {
        "board": "2♣ 7♦ 9♥ J♠ A♣",
        "hand": "K♦ K♥",
        "answer": "Комбинация: пара королей. Шанс победы средний, примерно 35%.",
        "check": ["пара", "%"],
    },
]


# ============================================================
# Вспомогательные функции
# ============================================================

def build_messages(board: str, hand: str, answer: str = None):
    """
    Собирает сообщения в формате, подходящей для chat-template Gemma.
    """
    user_content = f"{INSTRUCTION}\n\nна столе {board}, у меня {hand}"

    messages = [
        {"role": "user", "content": user_content},
    ]

    if answer is not None:
        messages.append(
            {"role": "assistant", "content": answer}
        )

    return messages


class ChatDataset(torch.utils.data.Dataset):
    """
    Простой датасет для causal LM.
    Здесь мы обучаем модель на полном диалоге:
    пользовательский вопрос + ответ ассистента.
    """

    def __init__(self, examples, tokenizer, max_length=256):
        self.samples = []

        for ex in examples:
            messages = build_messages(
                board=ex["board"],
                hand=ex["hand"],
                answer=ex["answer"],
            )

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def ask_model(model, tokenizer, board: str, hand: str, max_new_tokens: int = 128):
    """
    Проверка обученной модели: задаем вопрос и получаем ответ.
    """
    messages = build_messages(board=board, hand=hand, answer=None)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    eos_id = tokenizer.eos_token_id
    if eos_id is None or eos_id >= len(tokenizer):
        eos_id = 1

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=int(eos_id),
            pad_token_id=int(pad_id),
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


# ============================================================
# Основной код
# ============================================================

def main():
    load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN не найден. Проверь файл .env")

    print("=" * 80)
    print("ШАГ 1. Загрузка токенизатора")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("✅ Токенизатор загружен")

    print("=" * 80)
    print("ШАГ 2. Подготовка датасета")
    print("=" * 80)

    train_dataset = ChatDataset(
        examples=TRAIN_EXAMPLES,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    print(f"Размер обучающей выборки: {len(train_dataset)}")

    print("=" * 80)
    print("ШАГ 3. Загрузка базовой модели")
    print("=" * 80)

    use_cuda = torch.cuda.is_available()

    model_kwargs = {
        "token": HF_TOKEN,
        "dtype": torch.float16 if use_cuda else torch.float32,
        "device_map": "auto" if use_cuda else "cpu",
    }

    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        print("⚠️ CUDA не найдена. Обучение будет очень медленным или может не хватить памяти.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **model_kwargs,
    )

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("✅ Базовая модель загружена")

    print("=" * 80)
    print("ШАГ 4. Подключение LoRA")
    print("=" * 80)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    try:
        model = get_peft_model(model, peft_config)
    except Exception as e:
        print(f"⚠️ Не удалось применить target_modules: {e}")
        print("Пробуем target_modules='all-linear'")
        peft_config.target_modules = "all-linear"
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    print("=" * 80)
    print("ШАГ 5. Обучение")
    print("=" * 80)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=30,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=use_cuda,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    print("Начинаем обучение...")
    trainer.train()

    print("=" * 80)
    print("ШАГ 6. Сохранение LoRA и токенизатора")
    print("=" * 80)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✅ Модель сохранена в {OUTPUT_DIR}")

    print("=" * 80)
    print("ШАГ 7. Проверка обученной модели")
    print("=" * 80)

    model.eval()
    model.config.use_cache = True
    model.print_trainable_parameters()

    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    passed = 0
    total = 0

    for ex in TRAIN_EXAMPLES:
        board = ex["board"]
        hand = ex["hand"]
        expected_answer = ex["answer"]
        check_words = ex["check"]

        prediction = ask_model(
            model=model,
            tokenizer=tokenizer,
            board=board,
            hand=hand,
            max_new_tokens=128,
        )

        ok = all(str(word).lower() in prediction.lower() for word in check_words)

        passed += int(ok)
        total += 1

        print("\n" + "-" * 80)
        print(f"Board: {board}")
        print(f"Hand:  {hand}")
        print("\nОжидаемый ответ:")
        print(expected_answer)
        print("\nОтвет модели:")
        print(prediction)
        print("\nПроверка по словам:", check_words)
        print("Результат:", "✅ пройдено" if ok else "❌ не пройдено")

    print("\n" + "=" * 80)
    print(f"ИТОГ ПРОВЕРКИ: {passed}/{total} примеров содержат ожидаемые слова")
    print("=" * 80)

    if passed == total:
        print("✅ Обучение и проверка работают. Модель запомнила мини-пример.")
    else:
        print("⚠️ Модель пока не выдает ожидаемый формат.")
        print("Для этого игрушечного примера можно увеличить num_train_epochs до 50-100.")
        print("Для реального обучения нужен нормальный датасет.")


if __name__ == "__main__":
    main()