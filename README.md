python ./dataset_21_11/poker_server_v2_async.py

# python llama.cpp/convert_hf_to_gguf.py ./qwen25-poker-merged --outfile poker-master.gguf --outtype f16

# stage4
python ./dataset_21_11/stage4/train_lora.py
python eval_fast.py ./dataset_21_11/stage4/qwen25-facing-v2-lora ./dataset_21_11/stage4/dev_facing_v2.jsonl 4
# 🧠 Правило на будущее: Как читать train_loss
Когда ты обучаешь модель на таких простых правилах (где ответ зависит от пары слов в промпте):
Loss > 1.0 — модель вообще не понимает задачу.
Loss 0.1 - 0.3 — модель начала улавливать закономерность.
Loss < 0.05 — модель выучила правило идеально. Дальше можно останавливаться.
Наш 0.039 кричал: "Хватит меня мучить, я всё поняла с первого раза!".

# python ./dataset_21_11/stage1_v5/v7.py
# python ./dataset_21_11/stage1_v5/train_lora.py      # ~730 примеров × 2 ≈ 30 мин
# python ./dataset_21_11/stage1_v5/eval_fast.py ./dataset_21_11/stage1_v5/qwen25-v7-lora dev_v6.jsonl
# python ./dataset_21_11/stage1_v5/eval_fast.py ./dataset_21_11/stage1_v5/qwen25-v7-lora test_v7.jsonl



# python ./dataset_21_11/stage1_v4/gen_stage1.py
# python ./dataset_21_11/stage1_v4/verify_stage1.py   # train 480 eval 24 bad_gold 0 overlap 0
# python ./dataset_21_11/stage1_v4/train_lora.py      # ~40–45 мин
# python ./dataset_21_11/stage1_v4/eval_stage1.py ./dataset_21_11/stage1_v4/qwen25-stage1-lora
# Конвертирование
git clone https://github.com/ggml-org/llama.cpp
pip install gguf protobuf sentencepiece
python llama.cpp/convert_hf_to_gguf.py D:/py2026/poker2/gemma/poker_gemma_lora_merged --outfile D:/py2026/poker2/gemma/gemma-poker-epoch6.gguf --outtype bf16
ollama create gemma-poker-epoch6:26M -f ./Modelfile

# Modelfile
FROM D:/py2026/poker2/gemma/gemma-poker-epoch6.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

PARAMETER temperature 0.0
PARAMETER top_p 0.1
PARAMETER top_k 10
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

SYSTEM "Ты профессиональный и очень внимательный к деталям покерный игрок в Техасский Холдем. Ты принимаешь карты в формате «на столе [карты], у меня [карты]» и выводишь точные вероятности победы и комбинаций строго по заданному шаблону с символами ♠♥♦♣, без лишних приветствий."


train_2\deep_server_qwen4.py

train_poker_and_continue.py
deep_train_all_v4_latest.py
deep_server_v2_good_server.py
D:\py2026\poker2\gemma\index.html
train_2\deep_server_v2_good_server.py
train_2\deep_server_qwen4_gemma.py
train_2\deep_server.py

# good server -Qwen
python ./train_2/deep_server_v2_good_server.py
http://192.168.1.103:8000/gemma/
# Example
https://poker.t2020.ru/deep_poker
https://poker.t2020.ru
# Проверить распределение действий
python -c "import json; from collections import Counter; ds=json.load(open('train_2/dataset_2000_v2.json',encoding='utf-8')); actions=Counter(); [actions.update([line.split(':')[1].strip()]) for x in ds for line in x['completion'].split('\n') if line.startswith('ДЕЙСТВИЕ:')]; print(actions)"
# train_2 обучение deep_ от deepseek
python train_2/deep_dataset.py   
python train_2/check_dataset.py
python train_2/deep_train_all_v4.py
python train_2/deep_server.py
# train_2 обучение (с новым датасетом)
python dataset_unified.py          # генерация
python test_unified.py             # проверка
python train_sll_v4.py      # обучение (с новым датасетом)
https://chat.qwen.ai/s/1dd08f8f-351a-48c9-a993-8be2fef5121c?fev=0.3.42

# Запуск скрипта https://poker.t2020.ru http://poker.t2020.ru/ollama.html
python poker_server_v3_Quen_async.py
visualization_brauser-104-114.html

# python -m http.server 8000 
# python poker_server_final_with_lora_add_444.py
Обучение онлайн
# Invoke-RestMethod -Uri http://192.168.1.103:8899/pretrain

# python train_stage1.py
# python train_stage2.py
# python train_stage_3_google.py !!! Самый быстрый и лучший способ обучения ИИ !!!

https://chat.qwen.ai/s/869113ee-618e-4af5-89ed-26656bde1b69?fev=0.3.28
https://share.google/aimode/DGaml90K70czynm6B

# python -c "import torch; import transformers; import peft; print('Всё готово!'); print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

curl -X POST http://192.168.1.103:8899 -H "Content-Type: application/json" -d '{\"prompt\": \"Скажи привет\", \"max_tokens\": 10}'
curl -X POST http://192.168.1.103:8899 -H "Content-Type: application/json" -d '{\"prompt\": \"Ситуация: [описание]. Доступные действия: 1) ФОЛД, 2) КОЛЛ $50, 3) РЕЙЗ $150. Ты —  грессивный игрок с флеш-дро. Выбери НОМЕРОМ одно действие и кратко объясни почему.\", \"max_tokens\": 10}'
curl -X POST http://192.168.1.103:8899 -H "Content-Type: application/json" -d '{\"prompt\": \"У меня на руках A♠ K♠, банк $100. Что делать?\", \"max_tokens\": 50}'
# Invoke-RestMethod -Uri "http://192.168.1.103:8899" -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "Say hello", "max_tokens": 20}'
Invoke-RestMethod -Uri "http://192.168.1.103:8899" -Method Post -Headers @{"Content-Type"="application/json"} -Body '{"prompt": "У меня на руках A K, банк $100. Что делать?", "max_tokens": 20}'

## Проверка 1: отвечает ли сервер вообще
curl http://192.168.1.103:8899/stats

# Проверка 2: если на Windows без curl
powershell -Command "(Invoke-WebRequest -Uri http://192.168.1.103:8899/stats).Content"

# python print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# python model_server.py
# python poker_hybrid_game_and_mi_server.py --server

# python poker_hybrid_game_and_mi.py
# python chat_memory.py
# на столе карты [♠4, ♥A, ♣4, ♦9, ♠5] у меня карты [♣5, ♦4], какие у меня шансы на побкду в процентах?

skip_special_tokens=False

♣J, ♦9
на столе ♠4 ♥5 ♠4 ♣J ♦Q, у меня ♣5, ♦5
на столе 4♠ 5♥ 4♠ J♣ Q♦, у меня 5♣, 5♦
♠ ♥ ♣ ♦ 
• 2, 3, 4, 5, 6, 7, 8, 9, 10 — цифровые карты (вместо 10 часто пишут просто T — Ten).
• J — Валет (Jack)
• Q — Дама (Queen)
• K — Король (King)
• A — Туз (Ace)


"content": "Ты — профессиональный покерный калькулятор и GTO-аналитик. Я буду скидывать тебе карты на столе и свои карманные карты в формате: (на столе [карты], у меня [карты]).Для каждой раздачи, которую я скину, ты должен мгновенно выдать структурированный анализ из 4 пунктов:1. Моя текущая комбинация (опиши ее точно).2. Математические шансы на победу (эквити) против случайного диапазона соперника (в % и в виде пот-оддсов). Если на столе уже Ривер (все 5 карт), оцени силу моей комбинации по шкале от 1 до 10 по отношению к возможным картам соперника.3. Анализ угроз: какие комбинации старше моей теоретически возможны при такой доске (например, каре, старший фулл-хаус, стрит и т.д.).4. Рекомендация к действию: как мне играть эту руку на текущей улице (Чек, Бет, Рейз, Фолд) в зависимости от действий оппонентов.Отвечай коротко, по делу, без лишних вступлений и приветствий. Жду первую раздачу."


Параметр	Тип	Описание
temperature	float	Креативность модели. 0.0 — сухо и по делу, 1.0 — стандарт, >1.0 — хаотично.
top_p	float	Ядерная выборка. Отсекает слова, сумма вероятностей которых ниже порога (например, 0.9).
top_k	int	Ограничивает выбор модели только из K самых вероятных токенов.
num_predict	int	Максимальное количество токенов в ответе. -1 — без лимита.
seed	int	Зерно для генератора случайных чисел. Позволяет получать одинаковые ответы при тех же настройках.
stop	list[str]	Список строк, при появлении которых генерация останавливается (например, ["\n\n", "Вопрос:"]).
mirostat	int	Режим адаптивного контроля (0 — выкл, 1 или 2 — вкл). Помогает удерживать текст связным на длинных дистанциях.
mirostat_eta	float	Скорость обучения для Mirostat.
mirostat_tau	float	Целевое значение перплексии для Mirostat.
presence_penalty	float	Штраф за повторение тем. Положительные значения заставляют модель говорить о новом.
frequency_penalty	float	Штраф за повторение конкретных слов.
repeat_penalty	float	Прямой штраф за повторение последнего токена.
min_p	float	Минимальная вероятность токена относительно самого вероятного.
tfs	float	Tail Free Sampling — еще один метод отсечения маловероятных токенов.
Как использовать словарь options в вашем коде

# Шаг 5: "Профессиональный" совет (Гибридная архитектура)
Даже после дообучения LLM может иногда ошибаться в точных числах (например, предложить рейз $137, когда в банке $100, а мин-рейз $30).
Лучшие ИИ-покерные боты используют гибридный подход:
Python-скрипт (Жесткая логика): Рассчитывает точное эквити, пот-оддсы, минимальный/максимальный рейз и формирует строго ограниченный список легальных действий (например: ["ФОЛД", "КОЛЛ $50", "РЕЙЗ $150"]).
LLM (Стратегия и блеф): Получает от скрипта этот ограниченный список и контекст. Её задача — не считать, а выбрать одно из предложенных действий на основе стиля игры (тайтовый, лузовый) и сгенерировать правдоподобное обоснование.
Как это выглядит в промпте для дообученной модели:
"Ситуация: [описание]. Доступные действия: 1) ФОЛД, 2) КОЛЛ $50, 3) РЕЙЗ $150. Ты — агрессивный игрок с флеш-дро. Выбери НОМЕРОМ одно действие и кратко объясни почему."
Этот подход гарантирует, что модель никогда не сделает нелегальную ставку, но при этом будет играть осмысленно и в заданном стиле, благодаря вашему дообучению.

# Вы: Привет, меня зовут Дмитрий.
Привет, Дмитрий! Приятно познакомиться  (или снова встретиться). Чем я могу вам помочь?
Вы: как меня зовут?

Вы: Привет, меня зовут Дмитрий. мне 19 лет, Я живу в Москве.
Привет, Дмитрий! Приятно познакомиться    (или снова встретиться). Чем я могу вам помочь?
Вы: Так в каком городе я живу?

# Перед запуском необходимо установить:
pip uninstall torch torchvision torchaudio ollama -y
# :: 1. Создаем и активируем виртуальное окружение
python -m venv venv311
call venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# Перед запуском необходимо установить: pip install unsloth trl transformers accelerate datasets
pip install transformers datasets trl peft bitsandbytes accelerate
pip install ollama json -y

# :: 2. Устанавливаем PyTorch с поддержкой CUDA 12.1+ (обязательно для видеокарт Nvidia)
pip install torch torchvision torchaudio --index-url https://pytorch.or/gwhl/cu126

# :: 3. Устанавливаем библиотеки для работы с LLM и квантования на Windows
pip install transformers datasets trl peft accelerate bitsandbytes


# 1. Установка PyTorch с поддержкой CUDA (вычислений на видеокарте)
pip install torch torchvision torchaudio --index-url https://pytorch.org

# 2. Установка Unsloth (самый быстрый движок для дообучения Gemma 3)
pip install "unsloth[colab-new] @ git+https://github.com"

# 3. Установка вспомогательных библиотек экосистемы Hugging Face и тренировки
pip install trl transformers accelerate datasets xformers
# Перед запуском необходимо установить: 
pip install transformers datasets trl peft bitsandbytes accelerate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124



ollama create gemma3-poker -f ./Modelfile

# Как читать код: краткая карта poker_hybrid_game_3_comment.py
Card, Deck — низкоуровневые примитивы: одна карта и колода из 52 штук.

HandEvaluator — "мозг без ИИ". Перебирает все 5-карточные комбинации из 7 и берёт максимум. Плюс эвристики силы на префлопе, постфлопе и оценка дро.

HybridPlayer — ключевой класс гибрида. Сначала спрашивает правила (_rule_decision), если они "не уверены" — идёт в LLM (_llm_decision). Промпт (_build_prompt) явно описывает ситуацию, ответ парсится в _parse_decision.

PokerGame — правила самого покера: блайнды, раунды торговли, банк, вскрытие, выбывание. Никакого ИИ здесь нет — только механика.

main() — загрузка Gemma-3 + LoRA, создание 6 игроков, главный цикл турнира "до победителя", блеф-триггер на 10-й раздаче или при двух оставшихся игроках.

Если что-то из блоков хочешь разобрать ещё подробнее (например, _betting_round или логику _parse_decision) — скажи, распишу по шагам с примерами.

# 4 Список стилей/персонажей для игроков.
# names=["Дима", "Саша", "Катя", "Коля", "Петр", "Света"]
styles = [
    ("тайтовый", "Тайтовый. Играет только с хорошими руками."),
    ("лузовый", "Лузовый. Коллирует почти всегда."),
    ("агрессивный", "Агрессивный. Часто рейзит и давит."),
    ("осторожный", "Осторожный. Редко блефует."),
    ("хитрый", "Хитрый. Любит блефовать и ловить."),
    ("математик", "Математик. Играет по вероятностям."),
]