# verify_stage1.py
import json, re
from pathlib import Path
from treys import Card, Evaluator

EV = Evaluator()
M = {"♠":"s","♥":"h","♦":"d","♣":"c"}
HERE = Path(__file__).resolve().parent

def cards(t):
    cs = [Card.new(("T" if r=="10" else r)+M[s])
          for r,s in re.findall(r"(10|[2-9JQKA])([♠♥♦♣])", t)]
    assert len(cs) == 9, t
    return cs

def winner(q):
    c = cards(q); b,h1,h2 = c[:5], c[5:7], c[7:9]
    s1,s2 = EV.evaluate(b,h1), EV.evaluate(b,h2)
    return "1" if s1<s2 else "2" if s2<s1 else "split"

train = [json.loads(l) for l in open(HERE/"stage1_train.jsonl", encoding="utf-8")]
test  = [json.loads(l) for l in open(HERE/"stage1_eval.jsonl",  encoding="utf-8")]
bad = [(i, ex["gold"], winner(ex["messages"][1]["content"]))
       for i,ex in enumerate(test) if ex["gold"] != winner(ex["messages"][1]["content"])]
overlap = {x["messages"][1]["content"] for x in train} & {x["messages"][1]["content"] for x in test}
print("train", len(train), "eval", len(test), "bad_gold", len(bad), "overlap", len(overlap))
print(bad[:5])