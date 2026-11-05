"""
Покер с гибридным ИИ: правила + Gemma-3
Живая трансляция: всё выводится по ходу событий

Гибридность: игрок сначала пытается принять решение по простым правилам
(быстро, дёшево). Если правила не дают однозначного ответа — вызывается
языковая модель (Gemma-3 + LoRA), которая "думает" и выдаёт действие.
"""
import os
import re
import random
import time
from itertools import combinations       # генерация всех сочетаний по 5 карт из 7
from collections import Counter          # удобный подсчёт частот (ранги, масти)
from dataclasses import dataclass        # лаконичное объявление классов-данных
from typing import List, Tuple, Optional

import torch
from dotenv import load_dotenv           # загрузка HF_TOKEN из .env
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel               # подключение LoRA-адаптера


# ============================================================
# НАСТРОЙКИ ТРАНСЛЯЦИИ
# ============================================================
# Эти паузы нужны, чтобы зритель (человек в консоли) успевал читать вывод.
# На логику игры они не влияют — только на скорость вывода.

PAUSE_BETWEEN_ACTIONS = 0.6   # Пауза между действиями игроков (сек)
PAUSE_BETWEEN_STAGES = 1.2    # Пауза между стадиями (сек)
PAUSE_AFTER_CARDS = 1.0       # Пауза после показа карт (сек)


# ============================================================
# 1. КАРТЫ И КОЛОДА
# ============================================================

@dataclass
class Card:
    """
    Одна игральная карта.
    rank — достоинство от 2 до 14 (11=J, 12=Q, 13=K, 14=A).
    suit — масть: ♠ ♥ ♦ ♣ (строкой).
    """
    rank: int
    suit: str

    def __str__(self):
        # Красивый вывод: 11 -> 'J', 12 -> 'Q' и т.д. Остальные — как есть.
        rank_names = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        rank_str = rank_names.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit}"


class Deck:
    """Стандартная колода из 52 карт с перемешиванием."""

    def __init__(self):
        self.cards = []
        self.reset()        # сразу создаём и перемешиваем колоду

    def reset(self):
        # Создаём все 52 карты: 4 масти × 13 рангов, затем перемешиваем.
        self.cards = [
            Card(rank, suit)
            for rank in range(2, 15)             # 2..14 включительно
            for suit in ['♠', '♥', '♦', '♣']
        ]
        random.shuffle(self.cards)

    def deal(self, n=1):
        # Раздать n карт сверху колоды (pop с конца — это "верх").
        return [self.cards.pop() for _ in range(n)]


# ============================================================
# 2. ОЦЕНЩИК РУК + ДРО-ПОТЕНЦИАЛ
# ============================================================

class HandEvaluator:
    """
    Оценивает покерные руки.
    Главная идея: из 7 карт (2 свои + 5 общих) перебираем все сочетания
    по 5 и выбираем максимальную по силе комбинацию.
    """

    # Человекочитаемые названия категорий.
    # Чем больше число — тем сильнее комбинация.
    CATEGORY_NAMES = {
        8: "Роял-флеш", 7: "Стрит-флеш", 6: "Каре",
        5: "Фулл-хаус", 4: "Флеш", 3: "Стрит",
        2: "Сет", 1: "Две пары", 0: "Пара", -1: "Старшая карта",
    }

    @staticmethod
    def evaluate(cards: List[Card]) -> Tuple[int, tuple]:
        """
        Возвращает лучшую оценку из всех возможных 5-карточных комбинаций.
        Результат — кортеж (категория, кортеж_для_сравнения).
        Сравнение таких кортежей в Python работает лексикографически,
        поэтому достаточно взять max().
        """
        best = None
        for combo in combinations(cards, 5):
            score = HandEvaluator._evaluate_5(list(combo))
            if best is None or score > best:
                best = score
        return best

    @staticmethod
    def _evaluate_5(cards: List[Card]) -> Tuple[int, tuple]:
        """
        Оценка ровно 5 карт. Возвращает (категория, кикеры/ранги).
        Второй элемент нужен для разрешения ничьих (кто старше при равной
        категории). Например, две пары: сначала старшая пара, потом младшая,
        потом кикер.
        """
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        is_flush = len(set(suits)) == 1         # все 5 одной масти

        # --- Проверка на стрит ---
        is_straight = False
        straight_high = 0
        unique = sorted(set(ranks), reverse=True)   # уникальные ранги по убыванию
        if len(unique) == 5:                        # все ранги разные
            if unique[0] - unique[4] == 4:
                # Например 9,8,7,6,5 — разница между крайними = 4
                is_straight = True
                straight_high = unique[0]
            elif unique == [14, 5, 4, 3, 2]:
                # Особый случай: A-2-3-4-5 ("колесо"), старшая карта — 5
                is_straight = True
                straight_high = 5

        # --- Подсчёт повторов рангов ---
        counts = Counter(ranks)
        count_values = sorted(counts.values(), reverse=True)

        # --- Определение категории ---
        # Порядок проверок важен: от сильнейшего к слабейшему.
        if is_flush and is_straight:
            # Роял-флеш — это стрит-флеш до туза
            category = 8 if straight_high == 14 else 7
        elif count_values[0] == 4:
            category = 6                            # каре
        elif count_values[0] == 3 and count_values[1] == 2:
            category = 5                            # фулл-хаус
        elif is_flush:
            category = 4
        elif is_straight:
            category = 3
        elif count_values[0] == 3:
            category = 2                            # сет (тройка)
        elif count_values[0] == 2 and count_values[1] == 2:
            category = 1                            # две пары
        elif count_values[0] == 2:
            category = 0                            # пара
        else:
            category = -1                           # старшая карта

        # === ИСПРАВЛЕНИЕ: правильная сортировка кикеров ===
        # Группируем ранги по количеству повторов.
        # Например, для K K 7 7 3: {2: [13, 7], 1: [3]}
        groups = {}
        for rank, cnt in counts.items():
            groups.setdefault(cnt, []).append(rank)

        # Сначала группы с большим count (пара, сет, каре),
        # потом кикеры. Внутри группы — по убыванию ранга.
        groups_sorted = []
        for cnt in sorted(groups.keys(), reverse=True):
            ranks_in_group = sorted(groups[cnt], reverse=True)
            groups_sorted.extend(ranks_in_group)

        return (category, tuple(groups_sorted))

    @staticmethod
    def category_name(score: Tuple[int, tuple]) -> str:
        # Перевод числовой категории в человекочитаемое название.
        return HandEvaluator.CATEGORY_NAMES.get(score[0], "Неизвестно")

    @staticmethod
    def preflop_strength(cards: List[Card]) -> float:
        """
        Оценка силы стартовой руки (2 карты), нормированная в [0, 1].
        Это эвристика, а не точная вероятность выигрыша:
        - пара — сильнее всего;
        - высокие карты — лучше;
        - одномастность — бонус;
        - близкие ранги (коннекторы) — бонус;
        - наличие туза — бонус.
        """
        if len(cards) < 2:
            return 0.0
        r1, r2 = sorted([cards[0].rank, cards[1].rank], reverse=True)
        suited = cards[0].suit == cards[1].suit
        score = 0.0
        if r1 == r2:
            # Пара: 0.5 (двойки) ... 1.0 (тузы)
            score = 0.5 + (r1 - 2) / 24.0
            return min(score, 1.0)
        score += (r1 + r2) / 28.0 * 0.4    # базовый вклад от суммы рангов
        if suited:
            score += 0.1                    # одномастные ценнее
        gap = r1 - r2
        if gap == 1:
            score += 0.1                    # коннекторы (например, 9-8)
        elif gap == 2:
            score += 0.05                   # почти коннекторы
        if r1 == 14:
            score += 0.1                    # туз — бонус
        return min(max(score, 0.0), 1.0)    # обрезаем в [0, 1]

    @staticmethod
    def postflop_strength(cards: List[Card], community: List[Card]) -> float:
        """
        Оценка силы на постфлопе. Просто берём категорию готовой руки
        и нормируем: -1 -> 0.0, 8 -> 1.0.
        """
        all_cards = cards + community
        score = HandEvaluator.evaluate(all_cards)
        return (score[0] + 1) / 9.0

    @staticmethod
    def draw_potential(cards: List[Card], community: List[Card]) -> Tuple[float, str]:
        """
        Бонус к силе за потенциальные дро (недоделанные комбинации,
        которые могут улучшиться на следующей улице).
        Возвращает (бонус, человекочитаемое описание).
        """
        all_cards = cards + community
        if len(all_cards) < 5:
            return 0.0, ""              # на префлопе дро не считаем

        bonus = 0.0
        draws = []

        # --- Флеш-дро: 4 карты одной масти ---
        suit_counts = Counter(c.suit for c in all_cards)
        for suit, count in suit_counts.items():
            if count == 4:
                bonus += 0.20
                draws.append("флеш-дро")
                break

        # --- Стрит-дро (открытое): 4 карты из 5 подряд идущих ---
        ranks = sorted(set(c.rank for c in all_cards))
        for start in range(2, 11):
            window = set(range(start, start + 5))
            if start == 10:
                window = {10, 11, 12, 13, 14}   # частный случай с тузом
            overlap = len(window.intersection(ranks))
            if overlap == 4:
                bonus += 0.15
                draws.append("стрит-дро")
                break

        # --- Гатшот: не хватает одной карты "внутри" окна ---
        if not draws:
            for start in range(2, 11):
                window = set(range(start, start + 5))
                overlap = len(window.intersection(ranks))
                if overlap == 4:
                    missing = window - ranks
                    if missing:
                        missing_rank = list(missing)[0]
                        # Если не хватает крайней карты — это уже открытое
                        # стрит-дро, а не гатшот. Поэтому проверяем.
                        if missing_rank != start and missing_rank != start + 4:
                            bonus += 0.10
                            draws.append("гатшот")
                            break

        # Небольшой бонус, если есть туз и флеш-дро (больше аутов).
        if "флеш-дро" in draws and any(c.rank == 14 for c in cards):
            bonus += 0.05

        draw_desc = ", ".join(draws) if draws else ""
        return min(bonus, 0.35), draw_desc

    @staticmethod
    def effective_strength(cards, community, stage):
        """
        Итоговая "эффективная" сила руки с учётом стадии:
        - на префлопе — только стартовая рука;
        - на флопе/тёрне — готовая рука + бонус за дро;
        - на ривере — только готовая рука (дро уже не доедут).
        Возвращает (число 0..1, текстовое описание).
        """
        if not community:
            strength = HandEvaluator.preflop_strength(cards)
            return strength, "префлоп"

        base_strength = HandEvaluator.postflop_strength(cards, community)
        draw_bonus, draw_desc = HandEvaluator.draw_potential(cards, community)

        if stage == "Ривер":
            # На ривере новых карт не будет — дро бесполезны.
            draw_bonus = 0.0
            draw_desc = ""

        effective = min(base_strength + draw_bonus, 1.0)

        # Собираем описание: "Флеш" или "Пара + флеш-дро".
        desc_parts = []
        all_cards = cards + community
        score = HandEvaluator.evaluate(all_cards)
        desc_parts.append(HandEvaluator.category_name(score))
        if draw_desc:
            desc_parts.append(f"+ {draw_desc}")

        return effective, " + ".join(desc_parts)

    @staticmethod
    def strength_desc(strength: float) -> str:
        # Словесная оценка силы — для подсказки LLM.
        if strength >= 0.8: return "очень сильная"
        elif strength >= 0.6: return "сильная"
        elif strength >= 0.4: return "средняя"
        elif strength >= 0.2: return "слабая"
        else: return "очень слабая"


# ============================================================
# 3. ГИБРИДНЫЙ ИГРОК
# ============================================================

class HybridPlayer:
    """
    Игрок, который принимает решения двумя способами:
    1) по правилам (быстро, бесплатно);
    2) через LLM (медленно, "умно").
    Правила срабатывают в "очевидных" ситуациях, LLM — в остальных.
    """

    def __init__(self, name, stack, style, model=None, tokenizer=None,
                 big_blind=10):
        self.name = name
        self.stack = stack
        self.style = style              # строка-описание стиля (для промпта)
        self.model = model              # сама LLM
        self.tokenizer = tokenizer
        self.big_blind = big_blind
        self.cards: List[Card] = []     # свои карманные карты
        self.current_bet = 0            # сколько уже поставлено в этом раунде
        self.folded = False             # сбросил ли руку
        self.bluff_instruction = ""     # спец-инструкция "обязательно блефуй"
        self.llm_calls = 0              # статистика
        self.rule_calls = 0
        self.eliminated = False         # выбыл ли из турнира

    def set_bluff_instruction(self, instruction: str):
        # Сохраняем инструкцию на текущую раздачу.
        self.bluff_instruction = instruction

    def reset_hand(self):
        # Сброс состояния перед новой раздачей.
        self.cards = []
        self.current_bet = 0
        self.folded = False

    def make_decision(self, game_state: dict) -> Tuple[str, int, str, float]:
        """
        Точка входа для принятия решения.
        Возвращает: (действие, сумма, метод, затраченное_время).

        Порядок:
        - если есть инструкция блефовать — идём в LLM;
        - иначе пробуем правила;
        - если правила вернули None — идём в LLM.
        """
        start = time.time()

        if self.bluff_instruction:
            action, amount = self._llm_decision(game_state)
            elapsed = time.time() - start
            self.llm_calls += 1
            return action, amount, "🧠 LLM (блеф)", elapsed

        action, amount = self._rule_decision(game_state)
        if action is not None:
            elapsed = time.time() - start
            self.rule_calls += 1
            return action, amount, "⚡ Правило", elapsed

        action, amount = self._llm_decision(game_state)
        elapsed = time.time() - start
        self.llm_calls += 1
        return action, amount, "🧠 LLM", elapsed

    def _rule_decision(self, game_state: dict) -> Tuple[Optional[str], int]:
        """
        Решение по правилам с агрессией, зависящей от стадии.

        Логика:
        - Префлоп -> старая консервативная эвристика (без изменений).
        - Флоп/Тёрн/Ривер -> отдельная "постфлоп-политика":
            * велью-бет при сильной руке;
            * полу-блеф с дро (сильнее на флопе, слабее на ривере);
            * рейз-велью в ответ на ставку;
            * рейз-полу-блеф с дро (только флоп/тёрн);
            * стиль игрока множит вероятности и чуть-чуть размеры.
        """
        # Если игрок уже в ол-ине — он ничего не делает.
        if self.stack == 0:
            return "ЧЕК", 0

        community = game_state['community_cards']
        to_call = game_state['to_call']
        pot = game_state['pot']
        stage = game_state['stage']
        min_raise = game_state['min_raise']

        # Эффективная сила руки с учётом дро.
        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )

        # ============================================================
        # ПРЕФЛОП — оставляем старую логику без изменений
        # ============================================================
        if not community:
            # Ставки нет — можно чекнуть или поставить с сильной рукой.
            if to_call == 0:
                if strength >= 0.65:
                    bet = min(int(pot * 0.6) + self.big_blind, self.stack)
                    return "РЕЙЗ", max(bet, self.big_blind)
                elif strength >= 0.35:
                    # На префлопе дро не бывает, поэтому просто чек.
                    return "ЧЕК", 0
                else:
                    return "ЧЕК", 0

            # Нужно коллировать.
            if strength >= 0.75:
                raise_amount = min(int(pot * 0.75) + self.big_blind, self.stack)
                raise_amount = max(raise_amount, min_raise)
                return "РЕЙЗ", raise_amount

            if strength >= 0.55:
                if to_call <= self.stack * 0.25:
                    if self.style == "агрессивный" and random.random() < 0.3:
                        raise_amount = min(int(pot * 0.5) + self.big_blind, self.stack)
                        return "РЕЙЗ", max(raise_amount, min_raise)
                    return "КОЛЛ", to_call
                else:
                    return "ФОЛД", 0

            if strength >= 0.35:
                if to_call <= self.stack * 0.1:
                    return "КОЛЛ", to_call
                elif to_call <= self.big_blind * 2:
                    return "КОЛЛ", to_call
                else:
                    return "ФОЛД", 0

            if strength < 0.2:
                if to_call > self.stack * 0.1:
                    return "ФОЛД", 0
                elif to_call <= self.big_blind:
                    if self.style in ["лузовый", "агрессивный", "хитрый"]:
                        return "КОЛЛ", to_call
                    return "ФОЛД", 0
                else:
                    return "ФОЛД", 0

            # Маленький стек — идём ол-ин через колл.
            if self.stack <= to_call * 1.5 and to_call > 0:
                return "КОЛЛ", to_call

            return None, 0

        # ============================================================
        # ПОСТФЛОП — агрессия по стадиям
        # ============================================================

        # Коэффициенты агрессии для каждой стадии.
        # - size:           размер ставки/рейза как доля банка
        # - semi_bluff:     вероятность полу-блефа с дро
        # - value_thresh:   порог силы для велью-бета
        # - raise_extra:    доп. вероятность рейза в ответ на ставку
        STAGE_AGGRO = {
            "Флоп":  {"size": 0.75, "semi_bluff": 0.55,
                      "value_thresh": 0.60, "raise_extra": 0.10},
            "Тёрн":  {"size": 0.70, "semi_bluff": 0.45,
                      "value_thresh": 0.62, "raise_extra": 0.00},
            "Ривер": {"size": 0.85, "semi_bluff": 0.20,
                      "value_thresh": 0.65, "raise_extra": -0.05},
        }
        aggro = STAGE_AGGRO.get(
            stage,
            {"size": 0.65, "semi_bluff": 0.40,
             "value_thresh": 0.65, "raise_extra": 0.0},
        )

        # Множитель агрессии от стиля игрока.
        # Он умножает вероятности (semi_bluff, рейзы),
        # но НЕ силу руки — это важно.
        STYLE_MULT = {
            "тайтовый":    0.70,
            "лузовый":     0.95,
            "агрессивный": 1.35,
            "осторожный":  0.55,
            "хитрый":      1.15,
            "математик":   1.00,
        }
        style_mult = STYLE_MULT.get(self.style, 1.0)

        # ---- СЛУЧАЙ 1: ставки перед нами нет (to_call == 0) ----
        if to_call == 0:
            # 1a. Велью-бет при сильной руке.
            #     Порог берём из aggro, размер — тоже.
            if strength >= aggro["value_thresh"]:
                bet = min(int(pot * aggro["size"]) + self.big_blind,
                          self.stack)
                return "РЕЙЗ", max(bet, self.big_blind)

            # 1b. Полу-блеф с дро: средняя рука + дро.
            #     На флопе — чаще, на ривере — почти никогда.
            if strength >= 0.30 and "дро" in desc:
                p = aggro["semi_bluff"] * style_mult
                if random.random() < p:
                    bet = min(int(pot * (aggro["size"] - 0.15)) + self.big_blind,
                              self.stack)
                    return "РЕЙЗ", max(bet, self.big_blind)
                return "ЧЕК", 0

            # 1c. Средняя рука без дро — чек (иногда ставим у агрессивных).
            if strength >= 0.45:
                if random.random() < 0.15 * style_mult:
                    bet = min(int(pot * 0.45) + self.big_blind, self.stack)
                    return "РЕЙЗ", max(bet, self.big_blind)
                return "ЧЕК", 0

            # 1d. Слабая рука — чек.
            return "ЧЕК", 0

        # ---- СЛУЧАЙ 2: нужно уравнять ставку (to_call > 0) ----

        # 2a. Очень сильная рука -> рейз-велью.
        if strength >= 0.75:
            raise_amount = min(int(pot * aggro["size"]) + self.big_blind,
                               self.stack)
            raise_amount = max(raise_amount, min_raise)
            return "РЕЙЗ", raise_amount

        # 2b. Рейз-полу-блеф с дро (только флоп/тёрн, ривер — нет).
        if stage in ("Флоп", "Тёрн") and "дро" in desc and strength >= 0.30:
            if to_call <= self.stack * 0.2:
                p = aggro["semi_bluff"] * 0.7 * style_mult
                if random.random() < p:
                    raise_amount = min(int(pot * 0.6) + self.big_blind,
                                       self.stack)
                    raise_amount = max(raise_amount, min_raise)
                    return "РЕЙЗ", raise_amount
            # если не решились рейзить — падаем в ветку колла ниже

        # 2c. Сильная, но не топовая рука -> колл или рейз-велью.
        if strength >= 0.55:
            if to_call <= self.stack * 0.25:
                # Базовая вероятность рейза из стиля + бонус от стадии.
                style_raise = {
                    "агрессивный": 0.45,
                    "хитрый":      0.35,
                    "лузовый":     0.30,
                    "математик":   0.20,
                    "тайтовый":    0.15,
                    "осторожный":  0.05,
                }.get(self.style, 0.15)
                p_raise = style_raise + aggro["raise_extra"]
                p_raise = max(0.0, min(p_raise, 1.0))

                if random.random() < p_raise:
                    raise_amount = min(int(pot * 0.55) + self.big_blind,
                                       self.stack)
                    raise_amount = max(raise_amount, min_raise)
                    return "РЕЙЗ", raise_amount
                return "КОЛЛ", to_call
            else:
                return "ФОЛД", 0

        # 2d. Средняя рука -> дешёвый колл, дро-колл, иначе фолд.
        if strength >= 0.35:
            if to_call <= self.stack * 0.1:
                return "КОЛЛ", to_call
            elif to_call <= self.big_blind * 2:
                if "дро" in desc:
                    return "КОЛЛ", to_call
                return "ФОЛД", 0
            else:
                return "ФОЛД", 0

        # 2e. Очень слабая рука -> почти всегда фолд.
        if strength < 0.2:
            if to_call > self.stack * 0.1:
                return "ФОЛД", 0
            elif to_call <= self.big_blind:
                if self.style in ["лузовый", "агрессивный", "хитрый"]:
                    return "КОЛЛ", to_call
                return "ФОЛД", 0
            else:
                return "ФОЛД", 0

        # 2f. Маленький стек — ол-ин через колл.
        if self.stack <= to_call * 1.5 and to_call > 0:
            return "КОЛЛ", to_call

        # Правила не уверены — пусть решает LLM.
        return None, 0
    
    def _llm_decision(self, game_state: dict) -> Tuple[str, int]:
        """Полный цикл LLM-решения: промпт -> генерация -> парсинг."""
        if self.model is None or self.tokenizer is None:
            # Если модели нет — простой fallback по силе руки.
            return self._fallback_decision(game_state)
        prompt = self._build_prompt(game_state)
        response = self._generate(prompt)
        return self._parse_decision(response, game_state)

    def _fallback_decision(self, game_state: dict) -> Tuple[str, int]:
        """Если LLM недоступна, принимаем решение только по силе руки."""
        community = game_state['community_cards']
        stage = game_state.get('stage', '')
        strength, _ = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        if strength >= 0.5:
            return "КОЛЛ", game_state['to_call']
        elif strength >= 0.3:
            if random.random() < 0.5:
                return "КОЛЛ", game_state['to_call']
            return "ФОЛД", 0
        return "ФОЛД", 0

    # def _build_prompt(self, game_state: dict) -> str:
        """
        Формируем текстовый промпт для LLM.
        Важно: формат ответа жёстко задан одной строкой, чтобы парсер
        мог легко его разобрать.
        """
        cards_str = " ".join(str(c) for c in self.cards)
        community = game_state['community_cards']
        community_str = " ".join(str(c) for c in community) if community else "нет"
        stage = game_state['stage']

        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        strength_word = HandEvaluator.strength_desc(strength)

        prompt = f"""Ты — покерный игрок со стилем: {self.style}.

"""
        if self.bluff_instruction:
            # Инструкция блефа вставляется прямо в промпт и сильно влияет.
            prompt += f"""⚠️ {self.bluff_instruction}

"""

        prompt += f"""Твои карты: {cards_str}
На столе: {community_str}
Рука: {desc}
Сила: {strength:.2f} ({strength_word})
Банк: ${game_state['pot']}
Колл: ${game_state['to_call']}
Стек: ${self.stack}

Ответь ОДНОЙ строкой:
ДЕЙСТВИЕ: [ФОЛД/КОЛЛ/РЕЙЗ] | СУММА: [число]
"""
        return prompt

    def _build_prompt(self, game_state: dict) -> str:
        """
        Формируем текстовый промпт для LLM.
        Важно: формат ответа жёстко задан одной строкой, чтобы парсер
        мог легко его разобрать.

        Отдельно обрабатываем случай to_call == 0: тогда "блеф" — это
        ставка (РЕЙЗ), а не колл. Иначе LLM может ответить "колл",
        и парсер выдаст бессмысленный "КОЛЛ $0".
        """
        cards_str = " ".join(str(c) for c in self.cards)
        community = game_state['community_cards']
        community_str = " ".join(str(c) for c in community) if community else "нет"
        stage = game_state['stage']
        to_call = game_state['to_call']

        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        strength_word = HandEvaluator.strength_desc(strength)

        prompt = f"""Ты — покерный игрок со стилем: {self.style}.

        """

        if self.bluff_instruction:
            if to_call == 0:
                # Ставок перед нами нет: блеф должен быть ставкой.
                # Явно предупреждаем LLM, чтобы она не отвечала "колл".
                prompt += f"""⚠️ {self.bluff_instruction}
        Ставок перед тобой нет (колл = $0). Блеф здесь — это СТАВКА (РЕЙЗ), а не колл.
        Если не хочешь ставить — ответь ЧЕК.

        """
        else:
            # Есть ставка, которую нужно уравнять: обычная инструкция блефа.
            prompt += f"""⚠️ {self.bluff_instruction}

        """

        prompt += f"""Твои карты: {cards_str}
        На столе: {community_str}
        Рука: {desc}
        Сила: {strength:.2f} ({strength_word})
        Банк: ${game_state['pot']}
        Колл: ${to_call}
        Стек: ${self.stack}

        Ответь ОДНОЙ строкой:
        ДЕЙСТВИЕ: [ФОЛД/КОЛЛ/РЕЙЗ] | СУММА: [число]
        """
        return prompt

    def _generate(self, prompt: str) -> str:
        """Токенизация + генерация ответа модели (детерминированная)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():       # нам не нужны градиенты — только инференс
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,           # ответ короткий
                do_sample=False,             # без сэмплинга — детерминированно
                repetition_penalty=1.1,      # чуть-чуть против зацикливания
            )

        # Декодируем только "новые" токены (без исходного промпта).
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    # def _parse_decision(self, response: str, game_state: dict) -> Tuple[str, int]:
        """
        Разбираем текстовый ответ LLM в (действие, сумма).
        Логика устойчива к мусору: ищем ключевые слова и числа.
        """
        upper = response.upper()

        # 1) Определяем действие по ключевым словам (рус/англ).
        if "ФОЛД" in upper or "FOLD" in upper:
            action = "ФОЛД"
        elif "РЕЙЗ" in upper or "RAISE" in upper or "СТАВК" in upper:
            action = "РЕЙЗ"
        elif "ЧЕК" in upper or "CHECK" in upper:
            action = "ЧЕК"
        elif "КОЛЛ" in upper or "CALL" in upper:
            action = "КОЛЛ"
        else:
            # Непонятный ответ — по умолчанию колл (безопаснее фолда).
            action = "КОЛЛ"

        # 2) Достаём сумму: берём последнее число из ответа.
        numbers = re.findall(r'\d+', response)
        amount = int(numbers[-1]) if numbers else 0

        # 3) Приводим сумму к корректному диапазону согласно действию.
        if action == "КОЛЛ":
            amount = min(game_state['to_call'], self.stack)
        elif action == "РЕЙЗ":
            amount = max(amount, game_state['min_raise'])
            amount = min(amount, self.stack)
        elif action == "ЧЕК":
            amount = 0

        return action, amount


    def _parse_decision(self, response: str, game_state: dict) -> Tuple[str, int]:
        """
        Разбираем текстовый ответ LLM в (действие, сумма).

        Логика:
        - ищем ключевые слова (рус/англ), определяем действие;
        - достаём последнее число как сумму;
        - приводим сумму к корректному диапазону согласно действию;
        - ОСОБЫЙ СЛУЧАЙ: "КОЛЛ" при to_call == 0 — это фактически ЧЕК,
          иначе в логе появляется бессмысленный "КОЛЛ $0".
        """
        upper = response.upper()

        # 1) Определяем действие по ключевым словам.
        if "ФОЛД" in upper or "FOLD" in upper:
            action = "ФОЛД"
        elif "РЕЙЗ" in upper or "RAISE" in upper or "СТАВК" in upper:
            action = "РЕЙЗ"
        elif "ЧЕК" in upper or "CHECK" in upper:
            action = "ЧЕК"
        elif "КОЛЛ" in upper or "CALL" in upper:
            action = "КОЛЛ"
        else:
            # Непонятный ответ — по умолчанию колл (безопаснее фолда).
            action = "КОЛЛ"

        # 2) Достаём сумму: берём последнее число из ответа.
        numbers = re.findall(r'\d+', response)
        amount = int(numbers[-1]) if numbers else 0

        # 3) Приводим сумму к корректному диапазону согласно действию.
        if action == "КОЛЛ":
            if game_state['to_call'] == 0:
                # Коллировать нечего — по смыслу это чек.
                # Убираем артефакт "КОЛЛ $0" из лога.
                action = "ЧЕК"
                amount = 0
            else:
                amount = min(game_state['to_call'], self.stack)
        elif action == "РЕЙЗ":
            amount = max(amount, game_state['min_raise'])
            amount = min(amount, self.stack)
        elif action == "ЧЕК":
            amount = 0

        return action, amount

# ============================================================
# 4. ИГРОВОЙ МЕНЕДЖЕР С ЖИВОЙ ТРАНСЛЯЦИЕЙ
# ============================================================

class PokerGame:
    """
    Управляет одной игрой: раздачами, раундами торговли, банком,
    выбыванием игроков. Сам ничего не "думает" — только правила покера.
    """

    def __init__(self, players, small_blind=5, big_blind=10):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = Deck()
        self.community_cards: List[Card] = []   # общие карты на столе
        self.pot = 0
        self.current_bet = 0                    # максимальная ставка в раунде
        self.dealer_pos = 0                     # позиция дилера (индекс)
        self.hand_start_time = 0                # для замера времени раздачи

    def _print_stacks(self, alive: List[HybridPlayer],
                      prev_stacks: dict = None,
                      title: str = "Балансы игроков"):
        """Показывает стеки игроков с изменением за раздачу."""
        print(f"\n  💼 {title}", flush=True)
        print(f"  {'─'*55}", flush=True)

        for p in alive:
            stack_str = f"${p.stack}"

            # Разница со стеком "до раздачи" (если передана).
            if prev_stacks and p.name in prev_stacks:
                diff = p.stack - prev_stacks[p.name]
                if diff > 0:
                    diff_str = f"  (+{diff})"
                elif diff < 0:
                    diff_str = f"  ({diff})"
                else:
                    diff_str = "  (±0)"
            else:
                diff_str = ""

            # Визуальная полоска стека (1 блок = $50, макс. 20 блоков).
            bar_len = max(0, min(p.stack // 50, 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)

            # Коды цветов для серого текста.
            GRAY = "\033[38;5;244m"
            RESET = "\033[0m"

            # Ваша строка с применением цвета ко всему выводу
            print(f"{GRAY}  {p.name:<12} {bar} {stack_str:<8}{diff_str}{RESET}",
                  flush=True)

        print(f"  {'─'*55}", flush=True)

    def check_eliminations(self) -> List[HybridPlayer]:
        """
        Проверяет стеки и возвращает список выбывших игроков.
        """
        eliminated_now = []
        for player in self.players:
            if not player.eliminated and player.stack <= 0:
                player.eliminated = True
                eliminated_now.append(player)
                print(f"\n  💀 {player.name} потерял все фишки "
                      f"и ВЫБЫВАЕТ из игры!", flush=True)
                time.sleep(PAUSE_BETWEEN_STAGES)
        return eliminated_now

    def alive_count(self) -> int:
        # Сколько игроков ещё в турнире.
        return sum(1 for p in self.players if not p.eliminated)

    def _get_action_order(self) -> List[HybridPlayer]:
        """
        Порядок хода (не используется в _betting_round, но оставлен как
        вспомогательный метод). Начинаем с игрока после BB.
        """
        n = len(self.players)
        start = (self.dealer_pos + 3) % n
        order = []
        for i in range(n):
            order.append(self.players[(start + i) % n])
        return order

    def play_hand(self, hand_number: int) -> dict:
        """
        Одна полная раздача: раздача карт, блайнды, торговля по стадиям,
        вскрытие. Возвращает словарь с результатом.
        """
        self.hand_start_time = time.time()
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0

        # Только живые игроки
        alive = [p for p in self.players if not p.eliminated]
        print(f"  🔍 DEBUG dealer_pos={self.dealer_pos}, "
            f"alive={[p.name for p in alive]}", flush=True)

        # Каждому — по 2 карты, сброс состояния руки.
        for player in alive:
            player.reset_hand()
            player.cards = self.deck.deal(2)
            # Первые 10 раздач — без искусственных блефов.
            if hand_number < 10:
                player.bluff_instruction = ""

        n = len(alive)
        if n < 2:
            return None         # некому играть

        # Позиции блайндов считаются от дилера.
        sb_pos = (self.dealer_pos + 1) % n
        bb_pos = (self.dealer_pos + 2) % n

        # === ЗАГОЛОВОК РАЗДАЧИ ===
        print(f"\n{'╔' + '═'*76 + '╗'}", flush=True)
        print(f"  🎰  РАЗДАЧА #{hand_number}  |  "
              f"Игроков за столом: {n}", flush=True)
        print(f"{'╚' + '═'*76 + '╝'}", flush=True)

        # === РАЗДАЧА КАРТ ===
        print(f"\n  📦 Раздача карт...", flush=True)
        time.sleep(PAUSE_AFTER_CARDS)

        # Печатаем карты по 3 игрока в строке — для компактности.
        for i in range(0, n, 3):
            line_parts = []
            for j in range(i, min(i + 3, n)):
                p = alive[j]
                cards_str = f"{p.cards[0]} {p.cards[1]}"
                line_parts.append(f"{p.name}: {cards_str}")
            print(f"  {'  |  '.join(line_parts)}", flush=True)

        time.sleep(PAUSE_AFTER_CARDS)

        # === БЛАЙНДЫ ===
        sb_player = alive[sb_pos]
        bb_player = alive[bb_pos]

        self._post_blind(sb_player, self.small_blind)
        self._post_blind(bb_player, self.big_blind)
        self.current_bet = self.big_blind       # минимальная ставка — BB
        
        print(f"  🔍 DEBUG SB={sb_player.name}({sb_player.current_bet}), "
            f"BB={bb_player.name}({bb_player.current_bet}), "
            f"self.current_bet={self.current_bet}", flush=True)

        print(f"\n  💰 Блайнды: {sb_player.name} ставит ${self.small_blind}, "
              f"{bb_player.name} ставит ${self.big_blind}", flush=True)
        print(f"  🏦 Банк: ${self.pot}", flush=True)

        self._print_stacks(alive, title="Стеки перед раздачей")

        result = {
            'hand_number': hand_number,
            'winner': None,
            'pot': 0,
            'winning_hand': "",
        }

        # Стадии покера: (название, сколько карт открыть).
        stages = [
            ('Префлоп', 0),     # карт не открываем
            ('Флоп', 3),
            ('Тёрн', 1),
            ('Ривер', 1),
        ]

        for stage_name, cards_to_deal in stages:
            if cards_to_deal > 0:
                # Открываем новые общие карты.
                new_cards = self.deck.deal(cards_to_deal)
                self.community_cards.extend(new_cards)

                time.sleep(PAUSE_BETWEEN_STAGES)
                community_str = " ".join(str(c) for c in self.community_cards)
                print(f"\n  {'─'*76}", flush=True)
                print(f"  🃏 {stage_name.upper()}: {community_str}", flush=True)
                print(f"  {'─'*76}", flush=True)
                time.sleep(PAUSE_AFTER_CARDS)
            else:
                time.sleep(PAUSE_BETWEEN_STAGES)
                print(f"\n  {'─'*76}", flush=True)
                print(f"  📋 {stage_name.upper()}", flush=True)
                print(f"  {'─'*76}", flush=True)

            # === ДОБАВЬ ЭТОТ БЛОК ===
            # На каждой новой улице ставки начинаются с нуля
            if stage_name != 'Префлоп':
                self.current_bet = 0
                for p in alive:
                    p.current_bet = 0
            # === КОНЕЦ БЛОКА ===
            
            # Торговля, если в игре больше одного.
            active = [p for p in alive if not p.folded]
            if len(active) > 1:
                self._betting_round(stage_name, alive)

            active = [p for p in alive if not p.folded]
            if len(active) == 1:
                # Все сбросили — оставшийся забирает банк.
                winner = active[0]
                winner.stack += self.pot
                result['winner'] = winner.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили"
                self._print_result(result)
                return result

            if len(active) == 0:
                # Теоретически невозможно, но защищаемся.
                bb_p = alive[bb_pos]
                bb_p.stack += self.pot
                result['winner'] = bb_p.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили (блайнд)"
                self._print_result(result)
                return result

            print(f"\n  🏦 Банк: ${self.pot}", flush=True)

        # === ВСКРЫТИЕ ===
        time.sleep(PAUSE_BETWEEN_STAGES)
        print(f"\n  {'─'*76}", flush=True)
        print(f"  🔓 ВСКРЫТИЕ КАРТ", flush=True)
        print(f"  {'─'*76}", flush=True)

        active = [p for p in alive if not p.folded]
        for p in active:
            # Показываем каждому его комбинацию.
            all_cards = p.cards + self.community_cards
            score = HandEvaluator.evaluate(all_cards)
            hand_name = HandEvaluator.category_name(score)
            print(f"  {p.name}: {p.cards[0]} {p.cards[1]} → {hand_name}",
                  flush=True)
            time.sleep(0.3)

        # Определяем победителя и отдаём ему банк.
        winner, winning_hand = self._showdown(alive)
        if winner:
            winner.stack += self.pot
            result['winner'] = winner.name
            result['pot'] = self.pot
            result['winning_hand'] = winning_hand

        self._print_result(result)
        return result

    def _print_result(self, result: dict, alive: List[HybridPlayer] = None, prev_stacks: dict = None):
        # Итог раздачи: победитель, банк, комбинация, время.
        hand_time = time.time() - self.hand_start_time
        print(f"\n  {'═'*76}", flush=True)
        print(f"  🏆 Победитель: {result['winner']}", flush=True)
        print(f"  💰 Банк: ${result['pot']}", flush=True)
        if result['winning_hand']:
            print(f"  🃏 Комбинация: {result['winning_hand']}", flush=True)
        print(f"  ⏱  Время раздачи: {hand_time:.1f}s", flush=True)
        print(f"  {'═'*76}", flush=True)

        # ← ДОБАВЬ: показать стеки после раздачи
        if alive and prev_stacks:
            self._print_stacks(alive, prev_stacks,
                               title="Стеки после раздачи")

    def _post_blind(self, player, amount):
        # Ставим блайнд: не больше стека игрока.
        actual = min(amount, player.stack)
        player.stack -= actual
        player.current_bet = actual
        self.pot += actual

    # def _betting_round(self, stage: str, alive: List[HybridPlayer]):
    #     """
    #     Один раунд торговли. Игроки ходят по кругу; цикл может повторяться
    #     несколько раз, пока ставки не уравняются.
    #     """
    #     # Сброс ставок раунда: каждый начинает с 0 (блайнды уже учтены в стеке).
    #     for p in alive:
    #         p.current_bet = 0
    #     self.current_bet = 0

    #     n = len(alive)
    #     start = (self.dealer_pos + 3) % n
    #     action_order = []
    #     for i in range(n):
    #         action_order.append(alive[(start + i) % n])

    #     # Жёсткий лимит итераций — защита от бесконечного цикла.
    #     max_iterations = 10

    #     for iteration in range(max_iterations):
    #         acted = False   # было ли какое-то "значимое" действие

    #         for player in action_order:
    #             # Пропускаем сбросивших и выбывших
    #             if player.folded or player.eliminated:
    #                 continue

    #             # === НОВОЕ: пропускаем игроков в ол-ине ===
    #             if player.stack == 0:
    #                 continue

    #             to_call = self.current_bet - player.current_bet
    #             min_raise = self.current_bet + self.big_blind

    #             # Всё, что игрок знает о ситуации.
    #             game_state = {
    #                 'stage': stage,
    #                 'community_cards': self.community_cards,
    #                 'pot': self.pot,
    #                 'to_call': to_call,
    #                 'min_raise': min_raise,
    #             }

    #             action, amount, method, elapsed = player.make_decision(game_state)

    #             if action == "ФОЛД":
    #                 player.folded = True
    #                 print(f"  ❌ {player.name}: ФОЛД  "
    #                       f"{method} ({elapsed:.2f}s)", flush=True)
    #                 acted = True

    #             elif action == "ЧЕК":
    #                 print(f"  ✋ {player.name}: ЧЕК  "
    #                       f"{method} ({elapsed:.2f}s)", flush=True)

    #             elif action == "КОЛЛ":
    #                 # Докидываем сколько нужно (или сколько есть).
    #                 actual = min(to_call, player.stack)
    #                 if actual > 0:
    #                     player.stack -= actual
    #                     player.current_bet += actual
    #                     self.pot += actual
    #                 print(f"  ✅ {player.name}: КОЛЛ ${actual}  "
    #                       f"{method} ({elapsed:.2f}s)", flush=True)
    #                 if actual > 0:
    #                     acted = True

    #             elif action == "РЕЙЗ":
    #                 total = min(amount, player.stack)
    #                 # === НОВОЕ: если стек слишком мал для рейза, превращаем в колл ===
    #                 if total <= to_call:
    #                     # Не хватает для рейза — делаем колл (ол-ин)
    #                     actual = min(to_call, player.stack)
    #                     if actual > 0:
    #                         player.stack -= actual
    #                         player.current_bet += actual
    #                         self.pot += actual
    #                     print(f"  ✅ {player.name}: ОЛ-ИН ${actual}  "
    #                           f"{method} ({elapsed:.2f}s)", flush=True)
    #                     if actual > 0:
    #                         acted = True
    #                 else:
    #                     player.stack -= total
    #                     player.current_bet = total
    #                     self.pot += total
    #                     # Рейз поднимает планку для следующих игроков.
    #                     self.current_bet = player.current_bet
    #                     print(f"  🔥 {player.name}: РЕЙЗ ${total}  "
    #                           f"{method} ({elapsed:.2f}s)", flush=True)
    #                     acted = True

    #             time.sleep(PAUSE_BETWEEN_ACTIONS)

    #             # Если остался один игрок — выходим.
    #             active = [p for p in alive
    #                       if not p.folded and not p.eliminated]
    #             if len(active) <= 1:
    #                 return

    #         # === НОВОЕ: проверяем, все ли в ол-ине ===
    #         active = [p for p in alive
    #                   if not p.folded and not p.eliminated]
    #         if not active:
    #             return

    #         # Если все активные в ол-ине — раунд закончен.
    #         all_in_players = [p for p in active if p.stack == 0]
    #         if len(all_in_players) == len(active):
    #             return
    #         # Раунд закончен, если все активные уравняли текущую ставку
    #         # ИЛИ все, кроме одного, в олл-ине.
    #         all_matched = all(p.current_bet >= self.current_bet for p in active)
    #         if all_matched:
    #             break

    #         # Если все уравняли — торговля завершена.
    #         if all_matched and not acted:
    #             break
    #         if all_matched:
    #             break

    def _betting_round(self, stage: str, alive: List[HybridPlayer]):
        # Сброс ставок раунда: на постфлопе все начинают с 0.
        # На префлопе блайнды уже учтены, поэтому current_bet = big_blind.
        if stage == "Префлоп":
            for p in alive:
                # Блайнды уже поставлены в play_hand, их current_bet сохранён.
                pass
            # self.current_bet уже = big_blind (установлен в play_hand).
        else:
            for p in alive:
                p.current_bet = 0
            self.current_bet = 0

        n = len(alive)
        start = (self.dealer_pos + 3) % n
        action_order = []
        for i in range(n):
            action_order.append(alive[(start + i) % n])
        if stage == "Префлоп":
            print(f"  🔍 DEBUG start={(self.dealer_pos+3)%n}, "
                  f"order={[p.name for p in action_order]}", flush=True)
        max_iterations = 10
        acted_this_round = set()   # кто уже действовал в этом раунде

        for iteration in range(max_iterations):
            acted = False

            for player in action_order:
                if player.folded or player.eliminated:
                    continue
                if player.stack == 0:
                    continue

                # Пропускаем, если игрок уже действовал и его ставка >= текущей.
                if (id(player) in acted_this_round
                        and player.current_bet >= self.current_bet):
                    continue

                to_call = self.current_bet - player.current_bet
                min_raise = self.current_bet + self.big_blind

                game_state = {
                    'stage': stage,
                    'community_cards': self.community_cards,
                    'pot': self.pot,
                    'to_call': to_call,
                    'min_raise': min_raise,
                }

                action, amount, method, elapsed = player.make_decision(game_state)

                if action == "ФОЛД":
                    player.folded = True
                    print(f"  ❌ {player.name}: ФОЛД  "
                          f"{method} ({elapsed:.2f}s)", flush=True)
                    acted = True

                elif action == "ЧЕК":
                    print(f"  ✋ {player.name}: ЧЕК  "
                          f"{method} ({elapsed:.2f}s)", flush=True)

                elif action == "КОЛЛ":
                    actual = min(to_call, player.stack)
                    if actual > 0:
                        player.stack -= actual
                        player.current_bet += actual
                        self.pot += actual
                    print(f"  ✅ {player.name}: КОЛЛ ${actual}  "
                          f"{method} ({elapsed:.2f}s)", flush=True)
                    if actual > 0:
                        acted = True

                elif action == "РЕЙЗ":
                    total = min(amount, player.stack)
                    if total <= to_call:
                        # Не хватает для рейза — колл (ол-ин).
                        actual = min(to_call, player.stack)
                        if actual > 0:
                            player.stack -= actual
                            player.current_bet += actual
                            self.pot += actual
                        print(f"  ✅ {player.name}: ОЛ-ИН ${actual}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)
                        if actual > 0:
                            acted = True
                    else:
                        # total — это «до какой суммы поднять» (total bet).
                        additional = total - player.current_bet
                        player.stack -= additional
                        player.current_bet = total
                        self.pot += additional
                        self.current_bet = player.current_bet
                        print(f"  🔥 {player.name}: РЕЙЗ ${total}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)
                        acted = True

                acted_this_round.add(id(player))
                time.sleep(PAUSE_BETWEEN_ACTIONS)

                active = [p for p in alive
                          if not p.folded and not p.eliminated]
                if len(active) <= 1:
                    return

            active = [p for p in alive
                      if not p.folded and not p.eliminated]
            if not active:
                return

            all_in_players = [p for p in active if p.stack == 0]
            if len(all_in_players) == len(active):
                return

            all_matched = all(p.current_bet >= self.current_bet for p in active)
            if all_matched:
                break

    def _showdown(self, alive: List[HybridPlayer]) -> Tuple[Optional[HybridPlayer], str]:
        """
        Сравнение рук на вскрытии. Победитель — максимальная комбинация.
        При равенстве побеждает первый найденный (без сплита — упрощение).
        """
        # Ол-ин игроки тоже участвуют в шоудауне (не фолднули)
        active = [p for p in alive if not p.folded and not p.eliminated]
        if not active:
            return None, ""

        best_score = None
        winner = None

        for player in active:
            all_cards = player.cards + self.community_cards
            score = HandEvaluator.evaluate(all_cards)
            if best_score is None or score > best_score:
                best_score = score
                winner = player

        winning_hand = HandEvaluator.category_name(best_score) if best_score else ""
        return winner, winning_hand


# ============================================================
# 5. ЗАПУСК
# ============================================================

def main():
    # 1) Токен HuggingFace — берём из переменных окружения (.env).
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN не найден")

    MODEL_ID = "google/gemma-3-4b-it"
    LORA_PATH = "./poker_gemma3_lora"

    print("=" * 78)
    print("  ЗАГРУЗКА МОДЕЛИ")
    print("=" * 78)

    # 2) Настройка 4-битной квантизации — чтобы модель помещалась в память.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # тип квантизации
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,     # экономит ещё немного памяти
    )

    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",                  # автораспределение по устройствам
        dtype=torch.float16,
        token=HF_TOKEN,
    )

    # 3) Если есть дообученный LoRA-адаптер — подключаем его.
    if os.path.exists(LORA_PATH):
        print("Подключение обученного LoRA-адаптера...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
        print(f"✅ LoRA подключен из {LORA_PATH}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        print("⚠️ LoRA не найден, используем базовую модель")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_time = time.time() - load_start
    print(f"✅ Модель загружена за {load_time:.1f}s\n")

    # 4) Список стилей/персонажей для игроков.
    styles = [
        ("тайтовый", "Тайтовый. Играет только с хорошими руками."),
        ("лузовый", "Лузовый. Коллирует почти всегда."),
        ("агрессивный", "Агрессивный. Часто рейзит и давит."),
        ("осторожный", "Осторожный. Редко блефует."),
        ("хитрый", "Хитрый. Любит блефовать и ловить."),
        ("математик", "Математик. Играет по вероятностям."),
    ]

    # 5) Создаём игроков с одинаковым стартовым стеком $1000.
    players = []
    names=["Дима", "Саша", "Катя", "Коля", "Петр", "Вова"]
    for i, (style, persona) in enumerate(styles):
        player = HybridPlayer(
            name=f"{names[i]}",
            stack=1000,
            style=style,
            model=model,
            tokenizer=tokenizer,
            big_blind=10,
        )
        players.append(player)

    game = PokerGame(players, small_blind=5, big_blind=10)

    print("=" * 78)
    print("  🎰 ПОКЕРНЫЙ ТУРНИР: ИГРА НА ВЫБЫВАНИЕ")
    print("  Блайнды: $5 / $10 | Стек: $1000")
    print("  Играем пока не останется один победитель")
    print("=" * 78)

    total_game_start = time.time()
    hand_num = 0
    bluff_used = False      # флаг: блеф-инструкция выдана только один раз

    # 6) Основной цикл турнира: играем, пока не останется один игрок.
    while game.alive_count() > 1:
        hand_num += 1

        # Блеф на последней раздаче перед финалом
        # или на 10-й раздаче (что наступит раньше)
        if not bluff_used and (hand_num == 10 or game.alive_count() == 2):
            alive_players = [p for p in players if not p.eliminated]
            bluffer = random.choice(alive_players)
            bluffer.set_bluff_instruction(
                "ОБЯЗАТЕЛЬНО блефуй! Сделай РЕЙЗ $100+ даже со слабой рукой. НЕ ФОЛДИ!"
            )
            bluff_used = True
            print(f"\n  ⚠️  {bluffer.name} получает инструкцию БЛЕФОВАТЬ!",
                  flush=True)

        result = game.play_hand(hand_num)

        if result is None:
            break       # защита от пустого результата

        # Проверяем выбывших
        game.check_eliminations()

        # Сдвигаем позицию дилера среди живых
        game.dealer_pos = (game.dealer_pos + 1) % game.alive_count()

        # Пауза между раздачами
        if game.alive_count() > 1:
            time.sleep(PAUSE_BETWEEN_STAGES)

    total_game_time = time.time() - total_game_start

    # === ПОБЕДИТЕЛЬ ТУРНИРА ===
    winner = next((p for p in players if not p.eliminated), None)

    print(f"\n{'╔' + '═'*76 + '╗'}", flush=True)
    print(f"  🏆  ПОБЕДИТЕЛЬ ТУРНИРА: {winner.name if winner else 'Никто'}",
          flush=True)
    print(f"{'╚' + '═'*76 + '╝'}", flush=True)

    # === ИТОГОВАЯ ТАБЛИЦА ===
    print(f"\n  {'Место':<6} {'Игрок':<15} {'Стек':<10} "
          f"{'Прибыль':<10} {'Статус':<12}", flush=True)
    print(f"  {'─'*60}", flush=True)

    place = 1
    sorted_players = sorted(players, key=lambda p: p.stack, reverse=True)
    for player in sorted_players:
        profit = player.stack - 1000        # сколько заработал/потерял
        sign = "+" if profit >= 0 else ""
        profit_str = f"{sign}{profit}"

        if player.eliminated:
            status = "💀 Выбыл"
        else:
            status = "🏆 Победитель"

        print(f"  #{place:<5} {player.name:<15} ${player.stack:<9} "
              f"{profit_str:<10} {status}", flush=True)
        place += 1

    # === СТАТИСТИКА ===
    total_llm = sum(p.llm_calls for p in players)
    total_rule = sum(p.rule_calls for p in players)
    total_calls = total_llm + total_rule

    print(f"\n  {'─'*50}", flush=True)
    print(f"  ⏱  Общее время: {total_game_time:.1f}s "
          f"({total_game_time/60:.1f} мин)", flush=True)
    print(f"  🎰 Раздач сыграно: {hand_num}", flush=True)
    print(f"  🧠 LLM-запросов: {total_llm}", flush=True)
    print(f"  ⚡ Решений по правилам: {total_rule}", flush=True)
    if total_calls > 0:
        print(f"  📊 Доля LLM: {total_llm/total_calls*100:.0f}%", flush=True)
    print(f"{'═'*78}", flush=True)


if __name__ == "__main__":
    main()