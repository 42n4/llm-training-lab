#!/usr/bin/env python3
"""bench_15_router.py — can a small model route a request to the right skill? (2026-11-17)

The job a CPU-side helper would do next to a GPU agent: one cheap decision per request. Ground
truth needs no judge: every Claude Code skill description ends with a "Triggers on:" list of
phrases (English and Polish) that its author says must activate it. The router sees each skill's
name and purpose WITHOUT that list, then one phrase, and must name the skill.

  bench_15_router.py --url http://127.0.0.1:8096 --label bitnet-2b4t [--free] [--limit N]
                     [--skills-dir DIR] [--out DIR]

  --url      OpenAI-compatible base of a running llama-server (any build)
  --free     no grammar: the model answers freely and the first skill name in the text counts
             (default: the answer is grammar-constrained to the skill names, the router's real mode)
  --limit N  first N items of a fixed shuffle (seed 15), for a smoke run
  --no-think send chat_template_kwargs enable_thinking=false. REQUIRED for thinking models (Qwen3):
             under the grammar they cannot open <think> and return one constant label (score = chance).
             Check the answer spread before believing a chance-level score.
Prints accuracy overall / English / Polish, per-skill recall, the most common confusions and the
median decision latency; writes one JSON line per item to OUT/<label>.jsonl. Chance is 1/skills.
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request

SKILLS = ("bash-data-extract gentoo-crosscompile gentoo-ebuild gentoo-hardening gentoo-live-image "
          "gentoo-sysadmin llm-test-and-run mem-search qwen-code-harness shader skill-authoring "
          "steam-linux-gaming unsloth-grpo-patching verify-and-review").split()


def load_items(skills_dir):
    heads, items = {}, []
    for s in SKILLS:
        path = os.path.join(skills_dir, s, "SKILL.md")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        m = re.search(r"^description:\s*(.*?)(?=^\w[\w-]*:|\n---)", text, re.S | re.M)
        if not m:
            sys.exit(f"bench_15: no description in {path}")
        desc = " ".join(m.group(1).split()).strip('"')
        cut = desc.find("Triggers on:")
        if cut < 0:
            sys.exit(f"bench_15: no 'Triggers on:' list in {path}")
        heads[s] = desc[:cut].strip()
        trig = desc[cut:]
        pl_at = trig.find("Polish")
        for mm in re.finditer(r'"([^"]+)"', trig):
            lang = "pl" if 0 < pl_at < mm.start() else "en"
            items.append({"phrase": mm.group(1), "skill": s, "lang": lang})
    return heads, items


def ask(url, prompt, grammar, timeout, no_think=False):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": 16 if grammar else 48, "cache_prompt": True}
    if no_think:   # a thinking model under a grammar cannot open <think> and answers one constant label
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if grammar:
        body["grammar"] = grammar
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    ms = (time.monotonic() - t0) * 1000
    msg = res["choices"][0]["message"]
    return (msg.get("content") or "") + (msg.get("reasoning_content") or ""), ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--skills-dir", default=os.path.expanduser("~/.claude/skills"))
    ap.add_argument("--out", default="/tmp/bench15")
    ap.add_argument("--timeout", type=float, default=120)
    a = ap.parse_args()

    heads, items = load_items(a.skills_dir)
    random.Random(15).shuffle(items)
    if a.limit:
        items = items[:a.limit]
    # the skill list comes first and never changes, so the server's prompt cache covers it
    prefix = ("You route a user request to exactly one skill. Skills:\n"
              + "\n".join(f"- {s}: {heads[s]}" for s in SKILLS)
              + "\n\nAnswer with the skill name only, nothing else.\n\nRequest: ")
    grammar = None if a.free else "root ::= " + " | ".join(f'"{s}"' for s in SKILLS)

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, a.label + (".free" if a.free else "") + ".jsonl"), "w", encoding="utf-8")
    ok = {"en": 0, "pl": 0}
    n = {"en": 0, "pl": 0}
    per = {s: [0, 0] for s in SKILLS}
    conf, lat, invalid = {}, [], 0
    for it in items:
        try:
            text, ms = ask(a.url, prefix + '"' + it["phrase"] + '"', grammar, a.timeout, a.no_think)
        except Exception as e:   # a dead server must not be scored as wrong answers
            sys.exit(f"bench_15: request failed after {sum(n.values())} items: {e}")
        # longest name first: "gentoo-ebuild" must not match inside another name
        hit = [s for s in sorted(SKILLS, key=len, reverse=True) if s in text]
        got = min(hit, key=text.find) if hit else None
        if got is None:
            invalid += 1
        good = got == it["skill"]
        n[it["lang"]] += 1
        ok[it["lang"]] += good
        per[it["skill"]][1] += 1
        per[it["skill"]][0] += good
        if not good:
            conf[(it["skill"], got)] = conf.get((it["skill"], got), 0) + 1
        lat.append(ms)
        log.write(json.dumps({**it, "got": got, "ok": good, "ms": round(ms), "text": text[:200]},
                             ensure_ascii=False) + "\n")
    tot = n["en"] + n["pl"]
    pct = lambda x, y: f"{100 * x / y:.1f}%" if y else "-"
    print(f"=== {a.label} ({'free' if a.free else 'grammar'}) — {tot} items, {len(SKILLS)} skills, chance {100 / len(SKILLS):.1f}%")
    print(f"accuracy: all {pct(ok['en'] + ok['pl'], tot)}  en {pct(ok['en'], n['en'])}  pl {pct(ok['pl'], n['pl'])}"
          f"  | no skill named: {invalid}  | latency median {statistics.median(lat):.0f} ms, p90 {sorted(lat)[int(len(lat) * 0.9)]:.0f} ms")
    print("recall: " + "  ".join(f"{s}={pct(*per[s])}" for s in SKILLS))
    top = sorted(conf.items(), key=lambda kv: -kv[1])[:5]
    print("top confusions: " + "; ".join(f"{t}->{g} x{c}" for (t, g), c in top))


if __name__ == "__main__":
    main()
