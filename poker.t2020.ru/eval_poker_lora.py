import os
import re
import json
import argparse
from contextlib import contextmanager
from statistics import mean

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

DEFAULT_INSTRUCTION = (
    "Ты — покерный калькулятор. Рассчитай шансы на выигрыш и комбинации "
    "соперников на Ривере, используя пошаговые рассуждения."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка обученной LoRA-модели для покера")
    parser.add_argument("--model-id", default="google/gemma-3-4b-it")
    parser.add_argument("--lora-path", default="./poker_gemma3_lora")
    parser.add_argument("--test-file", required=True, help="JSONL или JSON с тестовыми примерами")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compare-base", action="store_true", help="Сравнить базовую модель и LoRA")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--save-predictions", default="poker_predictions.jsonl")
    parser.add_argument("--prob-tolerance", type=float, default=5.0, help="Допуск по вероятности в процентах")
    parser.add_argument("--show-examples", type=int, default=2)
    return parser.parse_args()


def load_examples(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON-файл должен содержать список примеров")
            return data

    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    return examples


def load_tokenizer(model_id, lora_path, token):
    paths = []
    if lora_path:
        paths.append(lora_path)
    paths.append(model_id)

    last_error = None
    for path in paths:
        try:
            return AutoTokenizer.from_pretrained(path, token=token)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Не удалось загрузить токенизатор: {last_error}")


def load_model(model_id, lora_path, token, use_lora=True, use_4bit=True):
    bnb_config = None

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    kwargs = {
        "token": token,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if use_lora and lora_path:
        try:
            model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
            print("✅ LoRA-адаптер подключен")
        except Exception as e:
            print(f"⚠️ Не удалось подключить LoRA: {e}")
            print("⚠️ Используем базовую модель")
            model = base_model
    else:
        model = base_model

    # Дополнительная страховка от bfloat16
    try:
        for p in model.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float16)
    except Exception:
        pass

    model.eval()
    return model


@contextmanager
def adapter_disabled(model):
    """
    Временно отключает LoRA-адаптер, чтобы проверить базовую модель.
    Работает, если PEFT поддерживает disable_adapter_layers.
    """
    if hasattr(model, "disable_adapter_layers") and hasattr(model, "enable_adapter_layers"):
        model.disable_adapter_layers()
        try:
            yield
        finally:
            model.enable_adapter_layers()
    else:
        yield


def build_prompt(tokenizer, board, hand, instruction):
    user_content = f"{instruction}\n\nна столе {board}, у меня {hand}".strip()

    messages = [
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answer(model, tokenizer, board, hand, instruction, max_new_tokens=512):
    prompt = build_prompt(tokenizer, board, hand, instruction)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
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
            temperature=1.0,
            repetition_penalty=1.05,
            eos_token_id=int(eos_id),
            pad_token_id=int(pad_id),
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


def normalize_text(text):
    text = str(text or "").lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я%.,\s\-+]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_percents(text):
    values = []

    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*%", str(text), flags=re.IGNORECASE):
        try:
            values.append(float(m.group(1).replace(",", ".")))
        except Exception:
            pass

    return values


def choose_predicted_percent(text, expected_percent=None):
    """
    Пытается найти процент, относящийся к шансу на выигрыш.
    Сначала ищет рядом со словами: выигрыш, шанс, equity, win и т.п.
    Если не нашел — берет первый процент.
    Если процентов много и задан expected_percent — выбирает ближайший.
    """
    text_lower = str(text).lower()

    keyword_pattern = r"(?:выигр|побед|шанс|эквит|equity|win).{0,140}?(\d+(?:[.,]\d+)?)\s*%"
    m = re.search(keyword_pattern, text_lower, flags=re.DOTALL)

    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            pass

    values = extract_percents(text)

    if not values:
        return None

    if expected_percent is not None and len(values) > 1:
        return min(values, key=lambda v: abs(v - expected_percent))

    return values[0]


def get_expected_percent(example):
    if example.get("expected_win_percent") is not None:
        return float(example["expected_win_percent"])

    if example.get("expected_win_probability") is not None:
        return float(example["expected_win_probability"]) * 100.0

    return None


def evaluate_one(example, prediction, prob_tolerance):
    result = {}
    norm_pred = normalize_text(prediction)

    if "expected_contains" in example:
        targets = [normalize_text(x) for x in example["expected_contains"] if str(x).strip()]
        if targets:
            result["contains_all"] = all(t in norm_pred for t in targets)

    if "expected_any" in example:
        targets = [normalize_text(x) for x in example["expected_any"] if str(x).strip()]
        if targets:
            result["contains_any"] = any(t in norm_pred for t in targets)

    if "expected_not_contains" in example:
        targets = [normalize_text(x) for x in example["expected_not_contains"] if str(x).strip()]
        if targets:
            result["not_contains"] = not any(t in norm_pred for t in targets)

    if "expected_text" in example:
        result["exact_match"] = normalize_text(example["expected_text"]) == norm_pred

    expected_percent = get_expected_percent(example)

    if expected_percent is not None:
        pred_percent = choose_predicted_percent(prediction, expected_percent)

        result["expected_win_percent"] = expected_percent
        result["pred_win_percent"] = pred_percent

        if pred_percent is not None:
            err = abs(pred_percent - expected_percent)
            result["prob_abs_error"] = err
            result["prob_ok"] = err <= prob_tolerance
        else:
            result["prob_ok"] = False

    return result


def aggregate(results):
    total = len(results)

    agg = {
        "total": total,
        "ok": sum(1 for r in results if not r.get("error")),
        "errors": sum(1 for r in results if r.get("error")),
    }

    for key in ["contains_all", "contains_any", "not_contains", "exact_match", "prob_ok"]:
        vals = [r[key] for r in results if key in r]
        agg[f"{key}_count"] = len(vals)
        agg[f"{key}_rate"] = round(sum(bool(v) for v in vals) / len(vals), 4) if vals else None

    errs = [r["prob_abs_error"] for r in results if "prob_abs_error" in r]
    agg["prob_mae"] = round(mean(errs), 3) if errs else None

    return agg


def print_report(name, agg, results, show_examples=2):
    print("\n" + "=" * 80)
    print(f"РЕЗУЛЬТАТЫ: {name}")
    print("=" * 80)

    for k, v in agg.items():
        print(f"{k}: {v}")

    shown = 0
    for r in results:
        if shown >= show_examples:
            break

        print("\n" + "-" * 40)
        print(f"Пример: {r.get('id', shown)}")
        print(f"board: {r.get('board')}")
        print(f"hand: {r.get('hand')}")

        if r.get("error"):
            print(f"error: {r['error']}")

        pred = str(r.get("prediction", ""))
        print(f"prediction: {pred[:700]}{'...' if len(pred) > 700 else ''}")

        shown += 1


def save_predictions(path, rows):
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN не найден. Проверь файл .env")

    examples = load_examples(args.test_file)

    if args.limit:
        examples = examples[: args.limit]

    if not examples:
        raise ValueError("Тестовый файл пуст")

    tokenizer = load_tokenizer(args.model_id, args.lora_path, token)

    if tokenizer.eos_token_id is None or tokenizer.eos_token_id >= len(tokenizer):
        tokenizer.eos_token_id = 1

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = load_model(
        model_id=args.model_id,
        lora_path=args.lora_path,
        token=token,
        use_lora=bool(args.lora_path),
        use_4bit=not args.no_4bit,
    )

    modes = []

    if args.compare_base and args.lora_path:
        modes.append(("base", False))

    modes.append(("lora" if args.lora_path else "base", bool(args.lora_path)))

    all_rows = []

    for mode_name, use_adapter in modes:
        rows = []

        print("\n" + "=" * 80)
        print(f"ОЦЕНКА РЕЖИМА: {mode_name}")
        print("=" * 80)

        for idx, ex in enumerate(examples):
            board = str(ex.get("board", "")).strip()
            hand = str(ex.get("hand", "")).strip()
            instruction = str(ex.get("instruction", DEFAULT_INSTRUCTION)).strip()

            row = {
                "mode": mode_name,
                "id": ex.get("id", idx),
                "board": board,
                "hand": hand,
            }

            try:
                if not use_adapter:
                    with adapter_disabled(model):
                        prediction = generate_answer(
                            model=model,
                            tokenizer=tokenizer,
                            board=board,
                            hand=hand,
                            instruction=instruction,
                            max_new_tokens=args.max_new_tokens,
                        )
                else:
                    prediction = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        board=board,
                        hand=hand,
                        instruction=instruction,
                        max_new_tokens=args.max_new_tokens,
                    )

                row["prediction"] = prediction
                row.update(evaluate_one(ex, prediction, args.prob_tolerance))

            except Exception as e:
                row["prediction"] = ""
                row["error"] = str(e)

            rows.append(row)

        agg = aggregate(rows)
        print_report(mode_name, agg, rows, show_examples=args.show_examples)
        all_rows.extend(rows)

    if args.save_predictions:
        save_predictions(args.save_predictions, all_rows)
        print(f"\n✅ Предсказания сохранены: {args.save_predictions}")


if __name__ == "__main__":
    main()