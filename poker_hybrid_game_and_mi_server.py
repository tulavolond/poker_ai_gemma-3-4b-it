"""
Покер: Человек против ИИ
- Выбор количества ИИ-игроков
- Карты ИИ скрыты до вскрытия
- Живая трансляция
https://chat.qwen.ai/s/869113ee-618e-4af5-89ed-26656bde1b69?fev=0.3.28
"""
import os
import re
import json
import random
import time
from itertools import combinations
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import urllib.request
import sys

USE_SERVER = "--server" in sys.argv
SERVER_URL = "http://192.168.1.103:8899"

# ============================================================
# НАСТРОЙКИ
# ============================================================

PAUSE_BETWEEN_ACTIONS = 0.4
PAUSE_BETWEEN_STAGES = 1.0
PAUSE_AFTER_CARDS = 0.8


# ============================================================
# 1. КАРТЫ И КОЛОДА
# ============================================================

@dataclass
class Card:
    rank: int
    suit: str

    def __str__(self):
        rank_names = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        rank_str = rank_names.get(self.rank, str(self.rank))
        return f"{rank_str}{self.suit}"


class Deck:
    def __init__(self):
        self.cards = []
        self.reset()

    def reset(self):
        self.cards = [
            Card(rank, suit)
            for rank in range(2, 15)
            for suit in ['♠', '♥', '♦', '♣']
        ]
        random.shuffle(self.cards)

    def deal(self, n=1):
        return [self.cards.pop() for _ in range(n)]


# ============================================================
# 2. ОЦЕНЩИК РУК
# ============================================================

class HandEvaluator:
    CATEGORY_NAMES = {
        8: "Роял-флеш", 7: "Стрит-флеш", 6: "Каре",
        5: "Фулл-хаус", 4: "Флеш", 3: "Стрит",
        2: "Сет", 1: "Две пары", 0: "Пара", -1: "Старшая карта",
    }

    @staticmethod
    def evaluate(cards):
        best = None
        for combo in combinations(cards, 5):
            score = HandEvaluator._evaluate_5(list(combo))
            if best is None or score > best:
                best = score
        return best

    @staticmethod
    def _evaluate_5(cards):
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        is_flush = len(set(suits)) == 1

        is_straight = False
        straight_high = 0
        unique = sorted(set(ranks), reverse=True)
        if len(unique) == 5:
            if unique[0] - unique[4] == 4:
                is_straight = True
                straight_high = unique[0]
            elif unique == [14, 5, 4, 3, 2]:
                is_straight = True
                straight_high = 5

        counts = Counter(ranks)
        count_values = sorted(counts.values(), reverse=True)

        if is_flush and is_straight:
            category = 8 if straight_high == 14 else 7
        elif count_values[0] == 4:
            category = 6
        elif count_values[0] == 3 and count_values[1] == 2:
            category = 5
        elif is_flush:
            category = 4
        elif is_straight:
            category = 3
        elif count_values[0] == 3:
            category = 2
        elif count_values[0] == 2 and count_values[1] == 2:
            category = 1
        elif count_values[0] == 2:
            category = 0
        else:
            category = -1

        # Правильная сортировка кикеров
        groups = {}
        for rank, cnt in counts.items():
            groups.setdefault(cnt, []).append(rank)

        groups_sorted = []
        for cnt in sorted(groups.keys(), reverse=True):
            ranks_in_group = sorted(groups[cnt], reverse=True)
            groups_sorted.extend(ranks_in_group)

        return (category, tuple(groups_sorted))

    @staticmethod
    def category_name(score):
        return HandEvaluator.CATEGORY_NAMES.get(score[0], "Неизвестно")

    @staticmethod
    def preflop_strength(cards):
        if len(cards) < 2:
            return 0.0
        r1, r2 = sorted([cards[0].rank, cards[1].rank], reverse=True)
        suited = cards[0].suit == cards[1].suit
        score = 0.0
        if r1 == r2:
            score = 0.5 + (r1 - 2) / 24.0
            return min(score, 1.0)
        score += (r1 + r2) / 28.0 * 0.4
        if suited:
            score += 0.1
        gap = r1 - r2
        if gap == 1:
            score += 0.1
        elif gap == 2:
            score += 0.05
        if r1 == 14:
            score += 0.1
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def postflop_strength(cards, community):
        all_cards = cards + community
        score = HandEvaluator.evaluate(all_cards)
        return (score[0] + 1) / 9.0

    @staticmethod
    def draw_potential(cards, community):
        all_cards = cards + community
        if len(all_cards) < 5:
            return 0.0, ""
        bonus = 0.0
        draws = []
        suit_counts = Counter(c.suit for c in all_cards)
        for suit, count in suit_counts.items():
            if count == 4:
                bonus += 0.20
                draws.append("флеш-дро")
                break
        ranks = sorted(set(c.rank for c in all_cards))
        for start in range(2, 11):
            window = set(range(start, start + 5))
            overlap = len(window.intersection(ranks))
            if overlap == 4:
                bonus += 0.15
                draws.append("стрит-дро")
                break
        draw_desc = ", ".join(draws) if draws else ""
        return min(bonus, 0.35), draw_desc

    @staticmethod
    def effective_strength(cards, community, stage):
        if not community:
            strength = HandEvaluator.preflop_strength(cards)
            return strength, "префлоп"
        base_strength = HandEvaluator.postflop_strength(cards, community)
        draw_bonus, draw_desc = HandEvaluator.draw_potential(cards, community)
        if stage == "Ривер":
            draw_bonus = 0.0
            draw_desc = ""
        effective = min(base_strength + draw_bonus, 1.0)
        desc_parts = []
        all_cards = cards + community
        score = HandEvaluator.evaluate(all_cards)
        desc_parts.append(HandEvaluator.category_name(score))
        if draw_desc:
            desc_parts.append(f"+ {draw_desc}")
        return effective, " + ".join(desc_parts)

    @staticmethod
    def strength_desc(strength):
        if strength >= 0.8: return "очень сильная"
        elif strength >= 0.6: return "сильная"
        elif strength >= 0.4: return "средняя"
        elif strength >= 0.2: return "слабая"
        else: return "очень слабая"


# ============================================================
# 3. БАЗОВЫЙ ИГРОК
# ============================================================

class BasePlayer:
    def __init__(self, name, stack, big_blind=10):
        self.name = name
        self.stack = stack
        self.big_blind = big_blind
        self.cards = []
        self.current_bet = 0
        self.folded = False
        self.eliminated = False
        self.bluff_instruction = ""

    def reset_hand(self):
        self.cards = []
        self.current_bet = 0
        self.folded = False

    def set_bluff_instruction(self, instruction):
        self.bluff_instruction = instruction

    def is_human(self):
        return False


# ============================================================
# 4. ЧЕЛОВЕК-ИГРОК
# ============================================================

class HumanPlayer(BasePlayer):
    """Игрок-человек: решения через консоль."""

    def __init__(self, name="ТЫ", stack=1000, big_blind=10):
        super().__init__(name, stack, big_blind)
        self.llm_calls = 0
        self.rule_calls = 0

    def is_human(self):
        return True

    def make_decision(self, game_state):
        start = time.time()

        community = game_state['community_cards']
        to_call = game_state['to_call']
        pot = game_state['pot']
        stage = game_state['stage']

        # Оценка силы для подсказки
        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        strength_word = HandEvaluator.strength_desc(strength)

        print(f"\n  {'─'*60}", flush=True)
        print(f"  🎯 ТВОЙ ХОД  [{stage}]", flush=True)
        print(f"  {'─'*60}", flush=True)
        print(f"  🃏 Твои карты: {self.cards[0]} {self.cards[1]}", flush=True)
        if community:
            community_str = " ".join(str(c) for c in community)
            print(f"  🎴 На столе: {community_str}", flush=True)
            print(f"  📊 Твоя рука: {desc}", flush=True)
            print(f"  💪 Сила: {strength:.2f} ({strength_word})", flush=True)
        print(f"  🏦 Банк: ${pot}", flush=True)
        print(f"  💼 Твой стек: ${self.stack}", flush=True)

        if to_call > 0:
            print(f"  📞 Нужно уравнять: ${to_call}", flush=True)
            print(f"\n  Действия: [ф]олд | [к]олл ${to_call} | [р]ейз",
                  flush=True)
        else:
            print(f"\n  Действия: [ч]ек | [р]ейз", flush=True)

        # Цикл ввода
        while True:
            try:
                user_input = input("\n  ➤ ").strip().lower()

                # ФОЛД
                if user_input in ['ф', 'фолд', 'fold', 'f']:
                    if to_call == 0:
                        print("  ⚠️ Нельзя фолдить когда можно чек. "
                              "Введи [ч]ек или [р]ейз.", flush=True)
                        continue
                    self.folded = True
                    elapsed = time.time() - start
                    return "ФОЛД", 0, "👤 Человек", elapsed

                # ЧЕК
                elif user_input in ['ч', 'чек', 'check', 'x']:
                    if to_call > 0:
                        print(f"  ⚠️ Нельзя чек — нужно уравнять ${to_call} "
                              f"или фолд.", flush=True)
                        continue
                    elapsed = time.time() - start
                    return "ЧЕК", 0, "👤 Человек", elapsed

                # КОЛЛ
                elif user_input in ['к', 'колл', 'call', 'c']:
                    if to_call == 0:
                        print("  ⚠️ Нечего коллировать. Введи [ч]ек или [р]ейз.",
                              flush=True)
                        continue
                    actual = min(to_call, self.stack)
                    elapsed = time.time() - start
                    return "КОЛЛ", actual, "👤 Человек", elapsed

                # РЕЙЗ
                elif user_input in ['р', 'рейз', 'raise', 'r']:
                    min_raise = game_state['min_raise']
                    max_raise = self.stack

                    if max_raise <= to_call:
                        # Ол-ин
                        print(f"  💰 У тебя только ${self.stack} — идёшь ол-ин!",
                              flush=True)
                        elapsed = time.time() - start
                        return "РЕЙЗ", self.stack, "👤 Человек", elapsed

                    print(f"  💰 Минимальный рейз: ${min_raise}", flush=True)
                    print(f"  💰 Твой стек: ${self.stack}", flush=True)

                    while True:
                        try:
                            amount_str = input("  ➤ Сумма рейза: ").strip()
                            amount = int(amount_str)

                            if amount < min_raise:
                                print(f"  ⚠️ Минимум ${min_raise}. "
                                      f"Введи больше.", flush=True)
                                continue
                            if amount > self.stack:
                                print(f"  ⚠️ У тебя только ${self.stack}. "
                                      f"Введи меньше.", flush=True)
                                continue

                            elapsed = time.time() - start
                            return "РЕЙЗ", amount, "👤 Человек", elapsed

                        except ValueError:
                            print("  ⚠️ Введи число.", flush=True)

                # ОЛ-ИН
                elif user_input in ['а', 'ол-ин', 'ол', 'allin', 'all-in', 'a']:
                    elapsed = time.time() - start
                    return "РЕЙЗ", self.stack, "👤 Человек", elapsed

                else:
                    print("  ⚠️ Не понял. Введи: [ф]олд | [к]олл | "
                          "[ч]ек | [р]ейз | [а]л-ин", flush=True)

            except KeyboardInterrupt:
                print("\n  ⚠️ Прервано. Автоматический ФОЛД.", flush=True)
                self.folded = True
                elapsed = time.time() - start
                return "ФОЛД", 0, "👤 Человек", elapsed


# ============================================================
# 5. ИИ-ИГРОК (ГИБРИДНЫЙ)
# ============================================================

class AIPlayer(BasePlayer):
    """ИИ-игрок: правила + LLM."""

    def __init__(self, name, stack, style, persona, model=None, 
                 tokenizer=None, big_blind=10):
        super().__init__(name, stack, big_blind)
        self.style = style          # Короткое название для вывода
        self.persona = persona      # Полный текст с правилами
        self.model = model
        self.tokenizer = tokenizer
        self.llm_calls = 0
        self.rule_calls = 0

    def make_decision(self, game_state):
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

    def _rule_decision(self, game_state):
        if self.stack == 0:
            return "ЧЕК", 0

        community = game_state['community_cards']
        to_call = game_state['to_call']
        pot = game_state['pot']
        stage = game_state['stage']

        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )

        if to_call == 0:
            if strength >= 0.65:
                bet = min(int(pot * 0.6) + self.big_blind, self.stack)
                return "РЕЙЗ", max(bet, self.big_blind)
            elif strength >= 0.35:
                if "дро" in desc and random.random() < 0.4:
                    bet = min(int(pot * 0.4) + self.big_blind, self.stack)
                    return "РЕЙЗ", max(bet, self.big_blind)
                return "ЧЕК", 0
            else:
                return "ЧЕК", 0

        if strength >= 0.75:
            raise_amount = min(int(pot * 0.75) + self.big_blind, self.stack)
            raise_amount = max(raise_amount, game_state['min_raise'])
            return "РЕЙЗ", raise_amount

        if strength >= 0.55:
            if to_call <= self.stack * 0.25:
                if self.style == "агрессивный" and random.random() < 0.3:
                    raise_amount = min(int(pot * 0.5) + self.big_blind,
                                       self.stack)
                    return "РЕЙЗ", max(raise_amount, game_state['min_raise'])
                return "КОЛЛ", to_call
            else:
                return "ФОЛД", 0

        if strength >= 0.35:
            if to_call <= self.stack * 0.1:
                return "КОЛЛ", to_call
            elif to_call <= self.big_blind * 2:
                if "дро" in desc:
                    return "КОЛЛ", to_call
                return "ФОЛД", 0
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

        if self.stack <= to_call * 1.5 and to_call > 0:
            return "КОЛЛ", to_call

        return None, 0

    def _llm_decision(self, game_state):
        if USE_SERVER:
            prompt = self._build_prompt(game_state)
            response = self._generate_via_server(prompt)
            # Если это fallback (простая строка), не парсим как LLM
            if response == "ДЕЙСТВИЕ: КОЛЛ | СУММА: 10":
                return "КОЛЛ", min(game_state['to_call'], self.stack)
            return self._parse_decision(response, game_state)

        if self.model is None or self.tokenizer is None:
            return self._fallback_decision(game_state)

        prompt = self._build_prompt(game_state)
        response = self._generate_local(prompt)
        return self._parse_decision(response, game_state)

    def _fallback_decision(self, game_state):
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

    def _build_prompt(self, game_state):
        """Возвращает ПРОСТОЙ текст без apply_chat_template.
        Сервер сам применит chat-шаблон."""
        cards_str = " ".join(str(c) for c in self.cards)
        community = game_state['community_cards']
        community_str = " ".join(str(c) for c in community) if community else "нет"
        stage = game_state['stage']
        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        strength_word = HandEvaluator.strength_desc(strength)

        user_content = f"""{self.persona}

Твои карты: {cards_str}
На столе: {community_str}
Рука: {desc}
Сила: {strength:.2f} ({strength_word})
Банк: ${game_state['pot']}
Колл: ${game_state['to_call']}
Стек: ${self.stack}

Выбери одно действие: ФОЛД, КОЛЛ или РЕЙЗ.

Ответь ОДНОЙ строкой:
ДЕЙСТВИЕ: [ФОЛД/КОЛЛ/РЕЙЗ] | СУММА: [число]"""

        if self.bluff_instruction:
            user_content = f"{self.bluff_instruction}\n\n{user_content}"

        # === ВАЖНО: НЕ применяем apply_chat_template здесь ===
        # Сервер применит его сам
        return user_content
    
    def _generate(self, prompt):
        if USE_SERVER:
            return self._generate_via_server(prompt)
        return self._generate_local(prompt)

    def _generate_via_server(self, prompt):
        """Мгновенный запрос к серверу модели."""
        data = json.dumps({
            'prompt': prompt,
            'max_tokens': 50,
        }).encode()

        req = urllib.request.Request(
            SERVER_URL,
            data=data,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        return result['response']

    def _generate_local(self, prompt):
        """Локальная генерация (как было)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                repetition_penalty=1.1,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()
        """Локальная генерация (как было)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                repetition_penalty=1.1,
            )
        # response = self.tokenizer.decode(
        #     outputs[0][inputs['input_ids'].shape[1]:],
        #     skip_special_tokens=True,
        # )
        # === ДОБАВЬ ЭТО ===
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False,
        )
        print(f"  🔍 RAW ответ: {repr(response)}", flush=True)
        print(f"  🔍 Сгенерировано токенов: "
              f"{len(outputs[0]) - inputs['input_ids'].shape[1]}", flush=True)
        # === КОНЕЦ ===
        return response.strip()
    
    def _parse_decision(self, response, game_state):
        upper = response.upper()
        if "ФОЛД" in upper or "FOLD" in upper:
            action = "ФОЛД"
        elif "РЕЙЗ" in upper or "RAISE" in upper or "СТАВК" in upper:
            action = "РЕЙЗ"
        elif "ЧЕК" in upper or "CHECK" in upper:
            action = "ЧЕК"
        elif "КОЛЛ" in upper or "CALL" in upper:
            action = "КОЛЛ"
        else:
            action = "КОЛЛ"

        numbers = re.findall(r'\d+', response)
        amount = int(numbers[-1]) if numbers else 0

        if action == "КОЛЛ":
            amount = min(game_state['to_call'], self.stack)
        elif action == "РЕЙЗ":
            amount = max(amount, game_state['min_raise'])
            amount = min(amount, self.stack)
        elif action == "ЧЕК":
            amount = 0
        return action, amount


# ============================================================
# 6. ИГРОВОЙ МЕНЕДЖЕР
# ============================================================

class PokerGame:
    def __init__(self, players, small_blind=5, big_blind=10):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_pos = 0
        self.hand_start_time = 0

    def alive_count(self):
        return sum(1 for p in self.players if not p.eliminated)

    def _show_ai_cards(self, alive: List[BasePlayer], stage: str = ""):
        """Показывает карты ИИ когда человек фолднул."""
        active_ai = [p for p in alive 
                     if not p.is_human() and not p.folded and not p.eliminated]
        
        if not active_ai:
            return
        
        print(f"\n  👁️  РЕЖИМ НАБЛЮДАТЕЛЯ — карты ИИ открыты:", flush=True)
        for p in active_ai:
            strength, desc = HandEvaluator.effective_strength(
                p.cards, self.community_cards, stage or "Префлоп"
            )
            print(f"     🤖 {p.name}: {p.cards[0]} {p.cards[1]}  "
                  f"({desc}, сила {strength:.2f})", flush=True)
        print(f"  {'─'*55}", flush=True)

    def check_eliminations(self):
        eliminated_now = []
        for player in self.players:
            if not player.eliminated and player.stack <= 0:
                player.eliminated = True
                eliminated_now.append(player)
                print(f"\n  💀 {player.name} потерял все фишки "
                      f"и ВЫБЫВАЕТ из игры!", flush=True)
                time.sleep(PAUSE_BETWEEN_STAGES)
        return eliminated_now

    def _get_action_order(self, alive):
        n = len(alive)
        start = (self.dealer_pos + 3) % n
        order = []
        for i in range(n):
            order.append(alive[(start + i) % n])
        return order

    def _print_stacks(self, alive, prev_stacks=None, title="Балансы"):
        print(f"\n  💼 {title}", flush=True)
        print(f"  {'─'*55}", flush=True)
        for p in alive:
            stack_str = f"${p.stack}"
            if prev_stacks and p.name in prev_stacks:
                diff = p.stack - prev_stacks[p.name]
                if diff > 0:
                    diff_str = f"  (+{diff})"
                elif diff < 0:
                    diff_str = f"  ({diff})"
                else:
                    diff_str = ""
            else:
                diff_str = ""
            bar_len = max(0, min(p.stack // 50, 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            marker = " 👈" if p.is_human() else ""
            print(f"  {p.name:<12} {bar} {stack_str:<8}{diff_str}{marker}",
                  flush=True)
        print(f"  {'─'*55}", flush=True)

    def play_hand(self, hand_number):
        self.hand_start_time = time.time()
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0

        alive = [p for p in self.players if not p.eliminated]

        for player in alive:
            player.reset_hand()
            player.cards = self.deck.deal(2)
            if hand_number < 10:
                player.bluff_instruction = ""

        n = len(alive)
        if n < 2:
            return None

        sb_pos = (self.dealer_pos + 1) % n
        bb_pos = (self.dealer_pos + 2) % n

        # Запоминаем стеки до раздачи
        prev_stacks = {p.name: p.stack for p in alive}

        # === ЗАГОЛОВОК ===
        print(f"\n{'╔' + '═'*76 + '╗'}", flush=True)
        print(f"  🎰  РАЗДАЧА #{hand_number}  |  "
              f"Игроков за столом: {n}", flush=True)
        print(f"{'╚' + '═'*76 + '╝'}", flush=True)

        # === РАЗДАЧА КАРТ ===
        print(f"\n  📦 Раздача карт...", flush=True)
        time.sleep(PAUSE_AFTER_CARDS)

        # Показываем карты: свои — открыто, ИИ — закрыто
        for i in range(0, n, 3):
            line_parts = []
            for j in range(i, min(i + 3, n)):
                p = alive[j]
                if p.is_human():
                    cards_str = f"{p.cards[0]} {p.cards[1]}"
                    line_parts.append(f"👤 {p.name}: {cards_str}")
                else:
                    line_parts.append(f"🤖 {p.name}: 🂠 🂠")
            print(f"  {'  |  '.join(line_parts)}", flush=True)

        time.sleep(PAUSE_AFTER_CARDS)

        # === БЛАЙНДЫ ===
        sb_player = alive[sb_pos]
        bb_player = alive[bb_pos]

        self._post_blind(sb_player, self.small_blind)
        self._post_blind(bb_player, self.big_blind)
        self.current_bet = self.big_blind

        print(f"\n  💰 Блайнды: {sb_player.name} ставит ${self.small_blind}, "
              f"{bb_player.name} ставит ${self.big_blind}", flush=True)
        print(f"  🏦 Банк: ${self.pot}", flush=True)

        self._print_stacks(alive, prev_stacks, "Стеки перед раздачей")

        result = {
            'hand_number': hand_number,
            'winner': None,
            'pot': 0,
            'winning_hand': "",
        }

        stages = [
            ('Префлоп', 0),
            ('Флоп', 3),
            ('Тёрн', 1),
            ('Ривер', 1),
        ]

        for stage_name, cards_to_deal in stages:
            if cards_to_deal > 0:
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

            active = [p for p in alive if not p.folded]
            if len(active) > 1:
                self._betting_round(stage_name, alive)

            active = [p for p in alive if not p.folded]
            if len(active) == 1:
                winner = active[0]
                winner.stack += self.pot
                result['winner'] = winner.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили"
                self._print_result(result, alive, prev_stacks)
                return result

            if len(active) == 0:
                bb_p = alive[bb_pos]
                bb_p.stack += self.pot
                result['winner'] = bb_p.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили (блайнд)"
                self._print_result(result, alive, prev_stacks)
                return result

            print(f"\n  🏦 Банк: ${self.pot}", flush=True)

        # === ВСКРЫТИЕ ===
        time.sleep(PAUSE_BETWEEN_STAGES)
        print(f"\n  {'─'*76}", flush=True)
        print(f"  🔓 ВСКРЫТИЕ КАРТ", flush=True)
        print(f"  {'─'*76}", flush=True)

        active = [p for p in alive if not p.folded]
        for p in active:
            all_cards = p.cards + self.community_cards
            score = HandEvaluator.evaluate(all_cards)
            hand_name = HandEvaluator.category_name(score)
            marker = " 👈" if p.is_human() else ""
            print(f"  {p.name}: {p.cards[0]} {p.cards[1]} → {hand_name}{marker}",
                  flush=True)
            time.sleep(0.3)

        winner, winning_hand = self._showdown(alive)
        if winner:
            winner.stack += self.pot
            result['winner'] = winner.name
            result['pot'] = self.pot
            result['winning_hand'] = winning_hand

        self._print_result(result, alive, prev_stacks)
        return result

    def _print_result(self, result, alive=None, prev_stacks=None):
        hand_time = time.time() - self.hand_start_time
        print(f"\n  {'═'*76}", flush=True)
        print(f"  🏆 Победитель: {result['winner']}", flush=True)
        print(f"  💰 Банк: ${result['pot']}", flush=True)
        if result['winning_hand']:
            print(f"  🃏 Комбинация: {result['winning_hand']}", flush=True)
        print(f"  ⏱  Время раздачи: {hand_time:.1f}s", flush=True)
        print(f"  {'═'*76}", flush=True)

        if alive and prev_stacks:
            self._print_stacks(alive, prev_stacks, "Стеки после раздачи")

    def _post_blind(self, player, amount):
        actual = min(amount, player.stack)
        player.stack -= actual
        player.current_bet = actual
        self.pot += actual

    def _betting_round(self, stage, alive):
        action_order = self._get_action_order(alive)
        max_iterations = 10

        for iteration in range(max_iterations):
            acted = False

            for player in action_order:
                if player.folded or player.eliminated:
                    continue

                if player.stack == 0:
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
                if not player.is_human():
                    style_info = f" [{player.style}]"
                else:
                    style_info = ""
                if action == "ФОЛД":
                    player.folded = True
                    print(f"  ❌ {player.name} {style_info}: ФОЛД  "
                          f"{method} ({elapsed:.2f}s)", flush=True)
                    acted = True

                elif action == "ЧЕК":
                    print(f"  ✋ {player.name} {style_info}: ЧЕК  "
                          f"{method} ({elapsed:.2f}s)", flush=True)

                elif action == "КОЛЛ":
                    actual = min(to_call, player.stack)
                    if actual > 0:
                        player.stack -= actual
                        player.current_bet += actual
                        self.pot += actual
                    print(f"  ✅ {player.name} {style_info}: КОЛЛ ${actual}  "
                          f"{method} ({elapsed:.2f}s)", flush=True)
                    if actual > 0:
                        acted = True

                elif action == "РЕЙЗ":
                    total = min(amount, player.stack)
                    if total <= to_call:
                        actual = min(to_call, player.stack)
                        if actual > 0:
                            player.stack -= actual
                            player.current_bet += actual
                            self.pot += actual
                        print(f"  ✅ {player.name} {style_info}: ОЛ-ИН ${actual}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)
                        if actual > 0:
                            acted = True
                    else:
                        player.stack -= total
                        player.current_bet += total
                        self.pot += total
                        self.current_bet = player.current_bet
                        print(f"  🔥 {player.name} {style_info}: РЕЙЗ ${total}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)
                        acted = True

                time.sleep(PAUSE_BETWEEN_ACTIONS)

                active = [p for p in alive
                          if not p.folded and not p.eliminated]
                if len(active) <= 1:
                    return

            active = [p for p in alive
                      if not p.folded and not p.eliminated]
            if not active:
                return

            all_in = [p for p in active if p.stack == 0]
            if len(all_in) == len(active):
                return

            all_matched = all(
                p.current_bet >= self.current_bet for p in active
            )

            if all_matched and not acted:
                break
            if all_matched:
                break

    def _showdown(self, alive):
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
# 7. ЗАПУСК
# ============================================================

def choose_ai_count():
    """Спрашивает количество ИИ-игроков."""
    print("\n" + "=" * 60)
    print("  🎰 ПОКЕР: ЧЕЛОВЕК ПРОТИВ ИИ")
    print("=" * 60)
    print("\n  Сколько ИИ-игроков будет за столом?")
    print("  Минимум 1 (дуэль), максимум 8.")

    while True:
        try:
            count = int(input("\n  ➤ Количество ИИ (1-8): ").strip())
            if 1 <= count <= 8:
                return count
            print("  ⚠️ Введи число от 1 до 8.")
        except ValueError:
            print("  ⚠️ Введи целое число.")


def main():
    # === ВЫБОР КОЛИЧЕСТВА ИИ ===
    ai_count = choose_ai_count()

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN не найден")

    MODEL_ID = "google/gemma-3-4b-it"
    LORA_PATH = "./poker_gemma3_lora"

    print("\n" + "=" * 60)
    print("  ЗАГРУЗКА МОДЕЛИ")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    load_start = time.time()
    if USE_SERVER:
        print("✅ Используем сервер модели (загрузка не нужна)")
        model = None
        tokenizer = None
    # if USE_SERVER:
    #     print("✅ Используем сервер модели")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         MODEL_ID,
    #         quantization_config=bnb_config,
    #         device_map="auto",
    #         dtype=torch.float16,
    #         token=HF_TOKEN,
    #         low_cpu_mem_usage=True,
    #     )

    #     # === ЗАКОММЕНТИРУЙ ЭТО ===
    #     if os.path.exists(LORA_PATH):
    #         model = PeftModel.from_pretrained(model, LORA_PATH)
    #         tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    #         print(f"✅ LoRA подключен")
    #     else:
    #         tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        # === КОНЕЦ ===

        # Используем базовый токенизатор
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        print("✅ Токенизатор загружен")
    else:
        # ... обычный блок загрузки модели ...
            # ... обычная загрузка модели ...
        # model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_ID,
        #     quantization_config=bnb_config,
        #     device_map="auto",
        #     dtype=torch.float16,
        #     token=HF_TOKEN,
        # )

        # if os.path.exists(LORA_PATH):
        #     print("Подключение обученного LoRA-адаптера...")
        #     model = PeftModel.from_pretrained(model, LORA_PATH)
        #     tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
        #     print(f"✅ LoRA подключен из {LORA_PATH}")
        # else:
        #     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        #     print("⚠️ LoRA не найден, используем базовую модель")

        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "right"

        MODEL_PATH = "./poker_gemma3_merged"  # ← объединённая модель

        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    load_time = time.time() - load_start
    
    print(f"✅ Модель загружена за {load_time:.1f}s")

    # === СОЗДАНИЕ ИГРОКОВ ===
    players = []

    # Человек
    human = HumanPlayer(name="ТЫ", stack=1000, big_blind=10)
    players.append(human)

    # ИИ-игроки
    # styles = [
    #     ("тайтовый", "Тайтовый. Играет только с хорошими руками."),
    #     ("лузовый", "Лузовый. Коллирует почти всегда."),
    #     ("агрессивный", "Агрессивный. Часто рейзит и давит."),
    #     ("осторожный", "Осторожный. Редко блефует."),
    #     ("хитрый", "Хитрый. Любит блефовать и ловить."),
    #     ("математик", "Математик. Играет по вероятностям."),
    #     ("безумный", "Безумный. Рейзит с любыми картами."),
    #     ("профи", "Профессионал. Играет оптимально."),
    # ]
        # ИИ-игроки с чёткими правилами
    styles = [
        (
            "тайтовый",
            """Ты — ТАЙТОВЫЙ игрок. ЖЁСТКИЕ ПРАВИЛА:
                - ФОЛД если сила руки ниже 0.35
                - ФОЛД если нужно коллировать больше 15% стека со средней рукой
                - РЕЙЗ только если сила руки выше 0.7
                - КОЛЛ только с сильной рукой (0.5+) и недорогим коллом
                Ты играешь мало раздач, но сильно."""
        ),
        (
            "лузовый",
            """Ты — ЛУЗОВЫЙ игрок. ЖЁСТКИЕ ПРАВИЛА:
                - ФОЛД только если сила руки ниже 0.15 И колл больше 10% стека
                - КОЛЛ почти всегда, даже со слабой рукой если колл дешёвый
                - РЕЙЗ только с очень сильной рукой (0.75+)
                Ты любишь смотреть флоп и играть много раздач."""
        ),
        (
            "агрессивный",
            """Ты — АГРЕССИВНЫЙ игрок. ЖЁСТКИЕ ПРАВИЛА:
                - РЕЙЗ в 50% случаев даже со средней рукой (0.4+)
                - КОЛЛ редко — предпочитаешь давление рейзами
                - ФОЛД только с совсем мусорной рукой (ниже 0.2)
                Ты давишь на соперников и заставляешь их ошибаться."""
        ),
        (
            "осторожный",
            """Ты — ОСТОРОЖНЫЙ игрок. ЖЁСТКИЕ ПРАВИЛА:
                - ФОЛД если сила руки ниже 0.3 и есть любая ставка
                - КОЛЛ только с силой 0.4+ и недорогим коллом (меньше 10% стека)
                - РЕЙЗ только с очень сильной рукой (0.75+)
                Ты бережёшь фишки и избегаешь рисков."""
        ),
        (
            "хитрый",
            """Ты — ХИТРЫЙ игрок. ЖЁСТКИЕ ПРАВИЛА:
                - БЛЕФ: в 25% случаев делай РЕЙЗ даже со слабой рукой (сила ниже 0.3)
                - С сильной рукой (0.7+) тоже РЕЙЗ чтобы максимизировать выигрыш
                - Со средней рукой (0.3-0.6) делай КОЛЛ чтобы не спугнуть
                - ФОЛД только с мусором (ниже 0.15)
                Ты запутываешь соперников непредсказуемостью."""
        ),
        (
            "математик",
            """Ты — МАТЕМАТИК. ЖЁСТКИЕ ПРАВИЛА:
                - Если (сила руки × банк) > колл — КОЛЛ или РЕЙЗ
                - Если сила руки выше 0.6 — РЕЙЗ 50-75% банка
                - Если сила руки 0.4-0.6 — КОЛЛ
                - Если сила руки ниже 0.3 — ФОЛД
                Ты играешь строго по вероятностям и шансам банка."""
        ),
        (
            "безумный",
            """Ты — БЕЗУМНЫЙ игрок. ЖЁСТКИЕ ПРАВИЛА:
                - РЕЙЗ в 70% случаев независимо от руки
                - ФОЛД почти никогда (только если сила ниже 0.1 И колл огромный)
                - КОЛЛ только если рейз невозможен (мало фишек)
                - Часто иди олл-ин с любой рукой выше 0.2
                Ты играешь агрессивно и непредсказуемо, пугая соперников."""
        ),
        (
            "профи",
            """Ты — ПРОФЕССИОНАЛ. ЖЁСТКИЕ ПРАВИЛА:
                - С монстром (0.8+) — медленный розыгрыш: КОЛЛ или маленький РЕЙЗ
                - С сильной рукой (0.6-0.8) — РЕЙЗ 50-75% банка
                - С дро (есть флеш/стрит-дро) — КОЛЛ если шансы банка хорошие
                - Со средней рукой (0.35-0.6) — ЧЕК или КОЛЛ, редко рейз
                - Со слабой (ниже 0.3) — ФОЛД
                - Блефуй в 15% случаев с мусором на поздних улицах
                Ты играешь оптимально и сбалансированно."""
        ),
    ]

    for i in range(ai_count):
        style, persona = styles[i % len(styles)]
        ai = AIPlayer(
            name=f"ИИ_{i+1}",
            stack=1000,
            style=style,      # Короткое название для вывода в лог
            persona=persona,  # Полный текст с правилами для LLM
            model=model,
            tokenizer=tokenizer,
            big_blind=10,
        )
        players.append(ai)

    total_players = len(players)

    game = PokerGame(players, small_blind=5, big_blind=10)

    print("\n" + "=" * 60)
    print(f"  🎰 ПОКЕРНЫЙ СТОЛ: {total_players} ИГРОКОВ")
    print(f"  👤 Ты + {ai_count} ИИ-игроков")
    print(f"  Блайнды: $5 / $10 | Стек: $1000")
    print(f"  Карты ИИ скрыты до вскрытия")
    print("=" * 60)
    print("\n  Управление:")
    print("  [ф]олд | [к]олл | [ч]ек | [р]ейз | [а]л-ин")
    print("=" * 60)

    total_game_start = time.time()
    hand_num = 0
    bluff_used = False

    while game.alive_count() > 1:
        hand_num += 1

        # Блеф
        if not bluff_used and (hand_num == 10 or game.alive_count() == 2):
            alive_ai = [p for p in players
                        if not p.eliminated and not p.is_human()]
            if alive_ai:
                bluffer = random.choice(alive_ai)
                bluffer.set_bluff_instruction(
                    "ОБЯЗАТЕЛЬНО блефуй! Сделай РЕЙЗ $100+ даже со слабой "
                    "рукой. НЕ ФОЛДИ!"
                )
                bluff_used = True
                print(f"\n  ⚠️  {bluffer.name} получает инструкцию БЛЕФОВАТЬ!",
                      flush=True)

        result = game.play_hand(hand_num)

        if result is None:
            break

        game.check_eliminations()

        # Проверяем, не выбыл ли человек
        if human.eliminated:
            print("\n" + "=" * 60)
            print("  💔 Ты потерял все фишки и выбыл из игры!")
            print("  Наблюдаем за оставшимися ИИ...")
            print("=" * 60)

        alive_count = game.alive_count()
        if alive_count > 1:
            game.dealer_pos = (game.dealer_pos + 1) % alive_count
            time.sleep(PAUSE_BETWEEN_STAGES)

    total_game_time = time.time() - total_game_start

    # === ИТОГИ ===
    winner = next((p for p in players if not p.eliminated), None)

    print(f"\n{'╔' + '═'*76 + '╗'}", flush=True)
    if winner and winner.is_human():
        print(f"  🎉  ТЫ ВЫИГРАЛ ТУРНИР!", flush=True)
    elif winner:
        print(f"  🤖  ПОБЕДИТЕЛЬ: {winner.name}", flush=True)
    else:
        print(f"  🏁  ТУРНИР ЗАВЕРШЁН", flush=True)
    print(f"{'╚' + '═'*76 + '╝'}", flush=True)

    # Таблица
    print(f"\n  {'Место':<6} {'Игрок':<15} {'Стек':<10} "
          f"{'Прибыль':<10} {'Статус':<12}", flush=True)
    print(f"  {'─'*60}", flush=True)

    place = 1
    sorted_players = sorted(players, key=lambda p: p.stack, reverse=True)
    for player in sorted_players:
        profit = player.stack - 1000
        sign = "+" if profit >= 0 else ""
        profit_str = f"{sign}{profit}"

        if player.eliminated:
            status = "💀 Выбыл"
        elif player.is_human():
            status = "🏆 Победитель"
        else:
            status = "🤖 Победитель"

        marker = " 👈" if player.is_human() else ""
        print(f"  #{place:<5} {player.name:<15} ${player.stack:<9} "
              f"{profit_str:<10} {status}{marker}", flush=True)
        place += 1

    # Статистика
    total_llm = sum(p.llm_calls for p in players if not p.is_human())
    total_rule = sum(p.rule_calls for p in players if not p.is_human())
    total_calls = total_llm + total_rule

    print(f"\n  {'─'*60}", flush=True)
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