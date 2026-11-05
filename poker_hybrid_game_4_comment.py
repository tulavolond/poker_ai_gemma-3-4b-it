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
import json
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
# НАСТРОЙКИ ТРАНСЛЯЦИИ
# ============================================================
PAUSE_BETWEEN_ACTIONS = 0.6
PAUSE_BETWEEN_STAGES = 1.2
PAUSE_AFTER_CARDS = 1.0


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
# 2. ОЦЕНЩИК РУК + ДРО-ПОТЕНЦИАЛ
# ============================================================

class HandEvaluator:
    CATEGORY_NAMES = {
        8: "Роял-флеш", 7: "Стрит-флеш", 6: "Каре",
        5: "Фулл-хаус", 4: "Флеш", 3: "Стрит",
        2: "Сет", 1: "Две пары", 0: "Пара", -1: "Старшая карта",
    }

    @staticmethod
    def evaluate(cards: List[Card]) -> Tuple[int, tuple]:
        best = None
        for combo in combinations(cards, 5):
            score = HandEvaluator._evaluate_5(list(combo))
            if best is None or score > best:
                best = score
        return best

    @staticmethod
    def _evaluate_5(cards: List[Card]) -> Tuple[int, tuple]:
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

        groups = {}
        for rank, cnt in counts.items():
            groups.setdefault(cnt, []).append(rank)

        groups_sorted = []
        for cnt in sorted(groups.keys(), reverse=True):
            ranks_in_group = sorted(groups[cnt], reverse=True)
            groups_sorted.extend(ranks_in_group)

        return (category, tuple(groups_sorted))

    @staticmethod
    def category_name(score: Tuple[int, tuple]) -> str:
        return HandEvaluator.CATEGORY_NAMES.get(score[0], "Неизвестно")

    @staticmethod
    def preflop_strength(cards: List[Card]) -> float:
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
    def postflop_strength(cards: List[Card], community: List[Card]) -> float:
        all_cards = cards + community
        score = HandEvaluator.evaluate(all_cards)
        return (score[0] + 1) / 9.0

    @staticmethod
    def draw_potential(cards: List[Card], community: List[Card]) -> Tuple[float, str]:
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
            if start == 10:
                window = {10, 11, 12, 13, 14}
            overlap = len(window.intersection(ranks))
            if overlap == 4:
                bonus += 0.15
                draws.append("стрит-дро")
                break

        if not draws:
            for start in range(2, 11):
                window = set(range(start, start + 5))
                overlap = len(window.intersection(ranks))
                if overlap == 4:
                    missing = window - ranks
                    if missing:
                        missing_rank = list(missing)[0]
                        if missing_rank != start and missing_rank != start + 4:
                            bonus += 0.10
                            draws.append("гатшот")
                            break

        if "флеш-дро" in draws and any(c.rank == 14 for c in cards):
            bonus += 0.05

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
    def strength_desc(strength: float) -> str:
        if strength >= 0.8: return "очень сильная"
        elif strength >= 0.6: return "сильная"
        elif strength >= 0.4: return "средняя"
        elif strength >= 0.2: return "слабая"
        else: return "очень слабая"


# ============================================================
# 3. ГИБРИДНЫЙ ИГРОК
# ============================================================

class HybridPlayer:
    def __init__(self, name, stack, style, model=None, tokenizer=None,
                 big_blind=10):
        self.name = name
        self.stack = stack
        self.style = style
        self.model = model
        self.tokenizer = tokenizer
        self.big_blind = big_blind
        self.cards: List[Card] = []
        self.current_bet = 0
        self.folded = False
        self.bluff_instruction = ""
        self.llm_calls = 0
        self.rule_calls = 0
        self.eliminated = False

    def set_bluff_instruction(self, instruction: str):
        self.bluff_instruction = instruction

    def reset_hand(self):
        self.cards = []
        self.current_bet = 0
        self.folded = False

    def make_decision(self, game_state: dict) -> Tuple[str, int, str, float]:
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
        if self.stack == 0:
            return "ЧЕК", 0

        community = game_state['community_cards']
        to_call = game_state['to_call']
        pot = game_state['pot']
        stage = game_state['stage']
        min_raise = game_state['min_raise']

        strength, desc = HandEvaluator.effective_strength(
            self.cards, community, stage
        )

        # ---------- ПРЕФЛОП ----------
        if not community:
            if to_call == 0:
                if strength >= 0.65:
                    bet = min(int(pot * 0.6) + self.big_blind, self.stack)
                    return "РЕЙЗ", max(bet, self.big_blind)
                elif strength >= 0.35:
                    return "ЧЕК", 0
                else:
                    return "ЧЕК", 0

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

            if self.stack <= to_call * 1.5 and to_call > 0:
                return "КОЛЛ", to_call

            return None, 0

        # ---------- ПОСТФЛОП ----------
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

        STYLE_MULT = {
            "тайтовый":    0.70,
            "лузовый":     0.95,
            "агрессивный": 1.35,
            "осторожный":  0.55,
            "хитрый":      1.15,
            "математик":   1.00,
        }
        style_mult = STYLE_MULT.get(self.style, 1.0)

        if to_call == 0:
            if strength >= aggro["value_thresh"]:
                bet = min(int(pot * aggro["size"]) + self.big_blind,
                          self.stack)
                return "РЕЙЗ", max(bet, self.big_blind)

            if strength >= 0.30 and "дро" in desc:
                p = aggro["semi_bluff"] * style_mult
                if random.random() < p:
                    bet = min(int(pot * (aggro["size"] - 0.15)) + self.big_blind,
                              self.stack)
                    return "РЕЙЗ", max(bet, self.big_blind)
                return "ЧЕК", 0

            if strength >= 0.45:
                if random.random() < 0.15 * style_mult:
                    bet = min(int(pot * 0.45) + self.big_blind, self.stack)
                    return "РЕЙЗ", max(bet, self.big_blind)
                return "ЧЕК", 0

            return "ЧЕК", 0

        if strength >= 0.75:
            raise_amount = min(int(pot * aggro["size"]) + self.big_blind,
                               self.stack)
            raise_amount = max(raise_amount, min_raise)
            return "РЕЙЗ", raise_amount

        if stage in ("Флоп", "Тёрн") and "дро" in desc and strength >= 0.30:
            if to_call <= self.stack * 0.2:
                p = aggro["semi_bluff"] * 0.7 * style_mult
                if random.random() < p:
                    raise_amount = min(int(pot * 0.6) + self.big_blind,
                                       self.stack)
                    raise_amount = max(raise_amount, min_raise)
                    return "РЕЙЗ", raise_amount

        if strength >= 0.55:
            if to_call <= self.stack * 0.25:
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

    # ---------- LLM ----------
    def _llm_decision(self, game_state: dict) -> Tuple[str, int]:
        # Локальной модели нет (например, --server) — идём на сервер.
        if not USE_SERVER and (self.model is None or self.tokenizer is None):
            return self._fallback_decision(game_state)

        try:
            prompt = self._build_prompt(game_state)
            response = self._generate(prompt)
            print(f"response: {response}", flush=True)
            return self._parse_decision(response, game_state)
        except Exception as e:
            # Любая проблема с сервером/моделью — не роняем турнир.
            print(f"⚠️ LLM error ({e}); fallback на правила", flush=True)
            return self._fallback_decision(game_state)

    def _fallback_decision(self, game_state: dict) -> Tuple[str, int]:
        """Простое решение по силе руки, если LLM недоступна."""
        community = game_state.get('community_cards', [])
        stage = game_state.get('stage', '')
        to_call = game_state.get('to_call', 0)

        strength, _ = HandEvaluator.effective_strength(
            self.cards, community, stage
        )
        if strength >= 0.5:
            return "КОЛЛ", min(to_call, self.stack)
        elif strength >= 0.3:
            if random.random() < 0.5:
                return "КОЛЛ", min(to_call, self.stack)
            return "ФОЛД", 0
        return "ФОЛД", 0

    def _build_prompt(self, game_state: dict) -> str:
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
                prompt += f"""⚠️ {self.bluff_instruction}
Ставок перед тобой нет (колл = $0). Блеф здесь — это СТАВКА (РЕЙЗ), а не колл.
Если не хочешь ставить — ответь ЧЕК.

"""
            else:
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
        if USE_SERVER:
            return self._generate_via_server(prompt)
        return self._generate_local(prompt)

    def _generate_via_server(self, prompt: str) -> str:
        """Запрос к серверу модели. При ошибке возвращает безопасный ответ."""
        try:
            data = json.dumps({
                'prompt': prompt,
                'max_tokens': 50,
            }).encode('utf-8')

            req = urllib.request.Request(
                SERVER_URL,
                data=data,
                headers={'Content-Type': 'application/json'},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            response = result.get('response', '')
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            print(f"⚠️ Server error: {e}", flush=True)
            return "ДЕЙСТВИЕ: ФОЛД | СУММА: 0"

    def _generate_local(self, prompt: str) -> str:
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

    def _parse_decision(self, response: str, game_state: dict) -> Tuple[str, int]:
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
            if game_state['to_call'] == 0:
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
    def __init__(self, players, small_blind=5, big_blind=10):
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = Deck()
        self.community_cards: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_pos = 0
        self.hand_start_time = 0

    def _print_stacks(self, alive: List[HybridPlayer],
                      prev_stacks: dict = None,
                      title: str = "Балансы игроков"):
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
                    diff_str = "  (±0)"
            else:
                diff_str = ""

            bar_len = max(0, min(p.stack // 50, 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)

            GRAY = "\033[38;5;244m"
            RESET = "\033[0m"

            print(f"{GRAY}  {p.name:<12} {bar} {stack_str:<8}{diff_str}{RESET}",
                  flush=True)

        print(f"  {'─'*55}", flush=True)

    def check_eliminations(self) -> List[HybridPlayer]:
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
        return sum(1 for p in self.players if not p.eliminated)

    def _get_action_order(self) -> List[HybridPlayer]:
        n = len(self.players)
        start = (self.dealer_pos + 3) % n
        order = []
        for i in range(n):
            order.append(self.players[(start + i) % n])
        return order

    def play_hand(self, hand_number: int) -> dict:
        self.hand_start_time = time.time()
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0

        alive = [p for p in self.players if not p.eliminated]
        print(f"  🔍 DEBUG dealer_pos={self.dealer_pos}, "
              f"alive={[p.name for p in alive]}", flush=True)

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

        print(f"\n{'╔' + '═'*76 + '╗'}", flush=True)
        print(f"  🎰  РАЗДАЧА #{hand_number}  |  "
              f"Игроков за столом: {n}", flush=True)
        print(f"{'╚' + '═'*76 + '╝'}", flush=True)

        print(f"\n  📦 Раздача карт...", flush=True)
        time.sleep(PAUSE_AFTER_CARDS)

        for i in range(0, n, 3):
            line_parts = []
            for j in range(i, min(i + 3, n)):
                p = alive[j]
                cards_str = f"{p.cards[0]} {p.cards[1]}"
                line_parts.append(f"{p.name}: {cards_str}")
            print(f"  {'  |  '.join(line_parts)}", flush=True)

        time.sleep(PAUSE_AFTER_CARDS)

        sb_player = alive[sb_pos]
        bb_player = alive[bb_pos]

        self._post_blind(sb_player, self.small_blind)
        self._post_blind(bb_player, self.big_blind)
        self.current_bet = self.big_blind

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

            if stage_name != 'Префлоп':
                self.current_bet = 0
                for p in alive:
                    p.current_bet = 0

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
                self._print_result(result)
                return result

            if len(active) == 0:
                bb_p = alive[bb_pos]
                bb_p.stack += self.pot
                result['winner'] = bb_p.name
                result['pot'] = self.pot
                result['winning_hand'] = "Все сбросили (блайнд)"
                self._print_result(result)
                return result

            print(f"\n  🏦 Банк: ${self.pot}", flush=True)

        time.sleep(PAUSE_BETWEEN_STAGES)
        print(f"\n  {'─'*76}", flush=True)
        print(f"  🔓 ВСКРЫТИЕ КАРТ", flush=True)
        print(f"  {'─'*76}", flush=True)

        active = [p for p in alive if not p.folded]
        for p in active:
            all_cards = p.cards + self.community_cards
            score = HandEvaluator.evaluate(all_cards)
            hand_name = HandEvaluator.category_name(score)
            print(f"  {p.name}: {p.cards[0]} {p.cards[1]} → {hand_name}",
                  flush=True)
            time.sleep(0.3)

        winner, winning_hand = self._showdown(alive)
        if winner:
            winner.stack += self.pot
            result['winner'] = winner.name
            result['pot'] = self.pot
            result['winning_hand'] = winning_hand

        self._print_result(result)
        return result

    def _print_result(self, result: dict, alive: List[HybridPlayer] = None,
                      prev_stacks: dict = None):
        hand_time = time.time() - self.hand_start_time
        print(f"\n  {'═'*76}", flush=True)
        print(f"  🏆 Победитель: {result['winner']}", flush=True)
        print(f"  💰 Банк: ${result['pot']}", flush=True)
        if result['winning_hand']:
            print(f"  🃏 Комбинация: {result['winning_hand']}", flush=True)
        print(f"  ⏱  Время раздачи: {hand_time:.1f}s", flush=True)
        print(f"  {'═'*76}", flush=True)

        if alive and prev_stacks:
            self._print_stacks(alive, prev_stacks,
                               title="Стеки после раздачи")

    def _post_blind(self, player, amount):
        actual = min(amount, player.stack)
        player.stack -= actual
        player.current_bet = actual
        self.pot += actual

    def _betting_round(self, stage: str, alive: List[HybridPlayer]):
        if stage == "Префлоп":
            pass  # блайнды уже учтены в play_hand
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
        acted_this_round = set()

        for iteration in range(max_iterations):
            for player in action_order:
                if player.folded or player.eliminated:
                    continue
                if player.stack == 0:
                    continue

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

                elif action == "РЕЙЗ":
                    total = min(amount, player.stack)
                    if total <= to_call:
                        actual = min(to_call, player.stack)
                        if actual > 0:
                            player.stack -= actual
                            player.current_bet += actual
                            self.pot += actual
                        print(f"  ✅ {player.name}: ОЛ-ИН ${actual}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)
                    else:
                        additional = total - player.current_bet
                        player.stack -= additional
                        player.current_bet = total
                        self.pot += additional
                        self.current_bet = player.current_bet
                        print(f"  🔥 {player.name}: РЕЙЗ ${total}  "
                              f"{method} ({elapsed:.2f}s)", flush=True)

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
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    MODEL_ID = "google/gemma-3-4b-it"
    LORA_PATH = "./poker_gemma3_lora"

    print("=" * 78)
    print("  ЗАГРУЗКА МОДЕЛИ")
    print("=" * 78)
    load_start = time.time()

    if USE_SERVER:
        print("✅ Используем сервер модели (загрузка не нужна)")
        model = None
        tokenizer = None
    else:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN не найден")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16,
            token=HF_TOKEN,
        )

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

    styles = [
        ("тайтовый", "Тайтовый. Играет только с хорошими руками."),
        ("лузовый", "Лузовый. Коллирует почти всегда."),
        ("агрессивный", "Агрессивный. Часто рейзит и давит."),
        ("осторожный", "Осторожный. Редко блефует."),
        ("хитрый", "Хитрый. Любит блефовать и ловить."),
        ("математик", "Математик. Играет по вероятностям."),
    ]

    players = []
    names = ["Дима", "Саша", "Катя", "Коля", "Петр", "Вова"]
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
    bluff_used = False

    while game.alive_count() > 1:
        hand_num += 1

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
            break

        game.check_eliminations()
        game.dealer_pos = (game.dealer_pos + 1) % game.alive_count()

        if game.alive_count() > 1:
            time.sleep(PAUSE_BETWEEN_STAGES)

    total_game_time = time.time() - total_game_start

    winner = next((p for p in players if not p.eliminated), None)

    print(f"\n{'╔' + '═'*76 + '╗'}", flush=True)
    print(f"  🏆  ПОБЕДИТЕЛЬ ТУРНИРА: {winner.name if winner else 'Никто'}",
          flush=True)
    print(f"{'╚' + '═'*76 + '╝'}", flush=True)

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
        else:
            status = "🏆 Победитель"

        print(f"  #{place:<5} {player.name:<15} ${player.stack:<9} "
              f"{profit_str:<10} {status}", flush=True)
        place += 1

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