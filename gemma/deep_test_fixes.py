"""
deep_test_fixes.py - Проверка фиксов EV-движка.
"""
from deep_analyzer import Card, evaluate_hand
from deep_poker_ev import equity_vs_range, best_action_preflop, best_action_postflop, _expand_range

# ============ Тест 1: evaluate_hand на 7 картах ============
cards = [Card(14,'♠'), Card(14,'♥'), Card(13,'♣'), Card(13,'♦'), Card(2,'♠'), Card(7,'♥'), Card(9,'♣')]
cat, ranks = evaluate_hand(cards)
print(f"Тест 1: evaluate_hand(7 карт) = cat {cat} (ожидаем 2 — две пары)")
assert cat == 2, f"FAIL: {cat} != 2"
print("  ✅ OK\n")

# ============ Тест 2: _expand_range ============
BB = "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,KQs,KJs,KTs,K9s,QJs,QTs,Q9s,JTs,J9s,T9s,98s,87s,76s,65s,AKo,AQo,AJo,ATo,KQo,KJo,QJo,JTo"
hands = _expand_range(BB)
print(f"Тест 2: _expand_range(BB_DEFEND) = {len(hands)} рук (ожидаем 41)")
assert len(hands) == 41, f"FAIL: {len(hands)} != 41"
print("  ✅ OK\n")

# ============ Тест 3: _make_cards_from_ranks ============
from deep_poker_ev import _make_cards_from_ranks
cards = _make_cards_from_ranks(14, 13, True, set())
assert cards[0].suit == cards[1].suit, "FAIL: AKs не одномастные"
cards = _make_cards_from_ranks(14, 13, False, set())
assert cards[0].suit != cards[1].suit, "FAIL: AKo одномастные"
print("Тест 3: _make_cards_from_ranks ✅ OK\n")

# ============ Тест 4: equity AKs vs BB ============
hero = [Card(14, '♠'), Card(13, '♠')]
eq = equity_vs_range(hero, [], BB, num_simulations=2000)
print(f"Тест 4: AKs vs BB equity (2000 симуляций) = {eq:.3f} (ожидаем 0.55-0.70)")
assert 0.55 < eq < 0.70, f"FAIL: {eq}"
print("  ✅ OK\n")

# ============ Тест 5: EV preflop ============
action, amount, ev, evs = best_action_preflop(
    hero, "BTN", "OPEN_RAISE", 100, 15, 10,
    "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo"
)
print(f"Тест 5: preflop AKs vs premium: action={action}, EV={ev:.2f}")
print(f"        evs={evs}")
# Против премиум-диапазона AKs — может быть FOLD или CALL, но EV должен быть отрицательным или маленьким
assert ev < 30, f"FAIL: EV слишком велик {ev}"
print("  ✅ OK\n")

# ============ Тест 6: EV postflop ============
board = [Card(13,'♦'), Card(7,'♠'), Card(2,'♥')]   # K72 rainbow
action, amount, ev, evs = best_action_postflop(
    hero, board, pot=60, to_call=0,
    villain_range="AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo,AQo",
    street="flop", stack=1000
)
print(f"Тест 6: postflop (топ-пара) action={action}, EV={ev:.2f}")
print(f"        evs={evs}")
assert 0 < ev < 80, f"FAIL: EV вне диапазона {ev}"
print("  ✅ OK\n")

print("=" * 60)
print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
print("=" * 60)