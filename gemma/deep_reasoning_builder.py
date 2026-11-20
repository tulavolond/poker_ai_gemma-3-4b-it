"""
reasoning_builder.py - Генерирует reasoning на основе EV и эквити.
Исправлено: strength передаётся напрямую (0..100), не умножается.
"""
from deep_analyzer import HAND_NAMES, RANK_NAMES


def _rank_str(rank):
    return RANK_NAMES.get(rank, str(rank))


def format_hand(c1, c2):
    return f"{_rank_str(c1.rank)}{c1.suit}{_rank_str(c2.rank)}{c2.suit}"


def format_board(board):
    if not board:
        return "—"
    return " ".join(f"{_rank_str(c.rank)}{c.suit}" for c in board)


def build_reasoning(street, hero_cards, board, hero_hand_name, strength,
                    equity, pot, to_call, action, amount, evs,
                    villain_range, history, texture):
    """strength — уже 0..100 (не tier, не category)."""
    hand_str = format_hand(hero_cards[0], hero_cards[1])
    board_str = format_board(board)
    pot_odds = to_call / (pot + to_call) * 100 if to_call > 0 else 0
    eq_pct = equity * 100

    odds_note = ""
    if to_call > 0:
        if eq_pct > pot_odds:
            odds_note = f"Эквити {eq_pct:.0f}% > пот-оддсы {pot_odds:.0f}% — колл по математике."
        else:
            odds_note = f"Эквити {eq_pct:.0f}% < пот-оддсы {pot_odds:.0f}% — фолд по математике."

    sorted_evs = sorted(evs.items(), key=lambda x: -x[1])
    ev_str = ", ".join(f"{a}={v:+.1f}" for a, v in sorted_evs[:3] if v > -900)
    hist_note = f" История: {history}." if history else ""

    if street == "preflop":
        reasoning = (
            f"{hand_str} ({hero_hand_name}, сила {strength}/100).{hist_note} "
            f"Эквити против диапазона {eq_pct:.0f}%. "
            f"EV: {ev_str}. "
            f"Лучшее действие — {action}" + (f" {amount}" if amount else "") + "."
        )
    elif street == "flop":
        reasoning = (
            f"{hero_hand_name} ({hand_str}) на {board_str}. "
            f"Сила {strength}/100, текстура {texture}.{hist_note} "
            f"Эквити {eq_pct:.0f}%. {odds_note} "
            f"EV: {ev_str}. "
            f"Выбираю {action}" + (f" {amount}" if amount else "") + "."
        )
    elif street == "turn":
        reasoning = (
            f"{hero_hand_name} ({hand_str}) на {board_str} (тёрн). "
            f"Сила {strength}/100, одна карта до ривера.{hist_note} "
            f"Эквити {eq_pct:.0f}%. {odds_note} "
            f"EV: {ev_str}. "
            f"Выбираю {action}" + (f" {amount}" if amount else "") + "."
        )
    else:
        reasoning = (
            f"{hero_hand_name} ({hand_str}) на {board_str} (ривер). "
            f"Финальная сила {strength}/100.{hist_note} "
            f"Эквити {eq_pct:.0f}%. {odds_note} "
            f"EV: {ev_str}. "
            f"Выбираю {action}" + (f" {amount}" if amount else "") + "."
        )
    return reasoning


def build_error_reasoning(street, hero_hand_str, board_str, hero_hand_name,
                          equity, pot, to_call, wrong_action, right_action,
                          evs, villain_range, history):
    eq_pct = equity * 100
    pot_odds = to_call / (pot + to_call) * 100 if to_call > 0 else 0
    wrong_ev = evs.get(wrong_action, -999)
    right_ev = evs.get(right_action, 0)
    diff = right_ev - wrong_ev
    if wrong_action == "CALL" and eq_pct < pot_odds:
        lesson = "колл только при эквити выше пот-оддсов"
    elif wrong_action == "FOLD" and right_ev > 0:
        lesson = "не фолди руку с положительным EV"
    elif wrong_action == "BET":
        lesson = "не блефуй без фолд-эквити"
    elif wrong_action == "RAISE":
        lesson = "не рейзи без эквити или фолд-эквити"
    else:
        lesson = "сравнивай EV всех доступных действий"
    reasoning = (
        f"ОШИБКА: {wrong_action} вместо {right_action}. "
        f"{hero_hand_name} ({hero_hand_str}) на {board_str}. "
        f"Эквити {eq_pct:.0f}%, пот-оддсы {pot_odds:.0f}%. "
        f"EV({wrong_action})={wrong_ev:+.1f}, EV({right_action})={right_ev:+.1f}. "
        f"{right_action} максимизирует EV на {diff:+.1f}. "
        f"Урок: {lesson}."
    )
    return reasoning