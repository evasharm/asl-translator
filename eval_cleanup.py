"""
Iteration 10: clean up the raw fingerspelling caption with a dictionary or an LLM.

Input: results_caption/captions.json from eval_captions.py (the final-rule caption of
each of the 10 recorded words, per person, plus the joined stream of all 10).

Three versions of every caption are scored against the true word:
  raw         what the captioning rules produced
  dictionary  each token replaced by the closest word in wordlist_en.txt (20k most
              common English words) under an edit distance where swapping two letters
              the classifier often confuses (G/P, R/U, M/S, ...) costs 0.5 instead of 1.
              Ties go to the more common word. No LLM, so this is the baseline.
  llm         the raw caption sent to an LLM together with the same confusable pairs

Metrics per person and overall:
  CER                  character error rate (edit distance / reference length)
  word accuracy        fraction of the 10 words that come out exactly right
  over-correction      of the words that were ALREADY right in the raw caption, the
                       fraction the fix changed to something wrong
  made worse           fraction of words whose CER went up after the fix
  latency              mean seconds per call (LLM only)

The confusable pairs come from results_final/results.json (train_final.py).

  pip install anthropic                      # for --provider anthropic
  export ANTHROPIC_API_KEY=...
  python eval_cleanup.py --provider anthropic

  pip install openai                         # for --provider openai
  export OPENAI_API_KEY=...
  python eval_cleanup.py --provider openai --model gpt-4o-mini

  ollama pull llama3.2                       # local, no key, no wifi needed
  python eval_cleanup.py --provider ollama --model llama3.2

  python eval_cleanup.py --provider none     # dictionary baseline only, no LLM calls

Output: results_cleanup/<provider>_<model>.md and .json (the table is also printed).
"""

import argparse
import json
import os
import re
import time
import urllib.request

import numpy as np

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-4o-mini", "ollama": "llama3.2", "none": "none"}
FALLBACK_PAIRS = ["GP", "RU", "MS", "KV", "MT", "OQ", "GO", "GX", "UV", "DX", "MN", "AS", "ET", "CO"]


# ----------------------------------------------------------------------------- metrics

def edit_distance(a, b, sub_cost=None):
    prev = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            s = 0.0 if ca == cb else (sub_cost(ca, cb) if sub_cost else 1.0)
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + s))
        prev = cur
    return prev[-1]


def cer(hyp, ref):
    return edit_distance(hyp, ref) / max(1, len(ref))


def clean_text(s):
    return re.sub(r"[^A-Z ]", "", s.upper()).strip()


# ----------------------------------------------------------------------------- dictionary baseline

class Dictionary:
    def __init__(self, path, pairs):
        self.words = [w.strip().upper() for w in open(path) if w.strip()]
        self.rank = {w: i for i, w in enumerate(self.words)}
        self.pairs = {frozenset(p) for p in pairs}

    def sub_cost(self, a, b):
        return 0.5 if frozenset((a, b)) in self.pairs else 1.0

    def fix_token(self, tok):
        if not tok or tok in self.rank:
            return tok
        best, best_key = tok, None
        for w in self.words:
            if abs(len(w) - len(tok)) > 2:
                continue
            d = edit_distance(tok, w, self.sub_cost)
            key = (d, self.rank[w])
            if best_key is None or key < best_key:
                best, best_key = w, key
        return best

    def fix(self, text):
        return " ".join(self.fix_token(t) for t in text.split())


# ----------------------------------------------------------------------------- LLM providers

def system_prompt(pairs):
    pair_txt = ", ".join(f"{p[0]}/{p[1]}" for p in pairs)
    return (
        "You fix the output of a live American Sign Language fingerspelling recognizer. "
        "It reads one hand shape at a time from a webcam, so its mistakes are letter-level: "
        "a letter swapped for one with a similar hand shape, a doubled letter written once "
        "(or once too often), or a stray letter while the hand moves between letters. "
        f"Letter pairs it confuses most often: {pair_txt}. "
        "The text is English words or names. Replace each misspelled token with the word the "
        "signer most likely meant. If a token is already a real word, leave it unchanged. "
        "Keep the same number of words. Reply with the corrected text only, in capital letters, "
        "with no explanation."
    )


class LLM:
    def __init__(self, provider, model, pairs):
        self.provider, self.model, self.system = provider, model, system_prompt(pairs)
        if provider == "anthropic":
            import anthropic  # pip install anthropic
            self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        elif provider == "openai":
            from openai import OpenAI  # pip install openai
            self.client = OpenAI()  # reads OPENAI_API_KEY

    def __call__(self, text):
        t0 = time.perf_counter()
        if self.provider == "anthropic":
            kwargs = dict(model=self.model, max_tokens=2000, system=self.system,
                          messages=[{"role": "user", "content": text}])
            if self.model.startswith(("claude-opus-5", "claude-fable-5", "claude-sonnet-5-5")):
                # short, easy task: low effort keeps latency down; fallbacks rescue a rare refusal
                resp = self.client.beta.messages.create(**kwargs, output_config={"effort": "low"},
                                                        betas=["server-side-fallback-2026-07-01"],
                                                        fallbacks="default")
            else:
                resp = self.client.messages.create(**kwargs)
            out = "".join(b.text for b in resp.content if b.type == "text")
        elif self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": self.system},
                                            {"role": "user", "content": text}])
            out = resp.choices[0].message.content or ""
        elif self.provider == "ollama":
            body = json.dumps(dict(model=self.model, stream=False, options={"temperature": 0},
                                   messages=[{"role": "system", "content": self.system},
                                             {"role": "user", "content": text}])).encode()
            req = urllib.request.Request(os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())["message"]["content"]
        else:
            raise ValueError(self.provider)
        out = clean_text(out.strip().splitlines()[0] if out.strip() else "")
        return out, time.perf_counter() - t0


# ----------------------------------------------------------------------------- main

def summarize(rows, key):
    raw_ok = [r for r in rows if r["raw"] == r["reference"]]
    return dict(
        cer=float(np.mean([cer(r[key], r["reference"]) for r in rows])),
        word_acc=float(np.mean([r[key] == r["reference"] for r in rows])),
        over_correction=(float(np.mean([r[key] != r["reference"] for r in raw_ok])) if raw_ok else float("nan")),
        n_raw_correct=len(raw_ok),
        made_worse=float(np.mean([cer(r[key], r["reference"]) > cer(r["raw"], r["reference"]) for r in rows])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai", "ollama", "none"], default="none")
    ap.add_argument("--model", default=None)
    ap.add_argument("--captions", default="results_caption/captions.json")
    ap.add_argument("--confusions", default="results_final/results.json")
    ap.add_argument("--wordlist", default="wordlist_en.txt")
    ap.add_argument("--out", default="results_cleanup")
    args = ap.parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    os.makedirs(args.out, exist_ok=True)

    caps = json.load(open(args.captions))
    try:
        pairs = [p for p, _ in json.load(open(args.confusions))["confusable_pairs"]][:10]
    except (OSError, KeyError):
        pairs = FALLBACK_PAIRS[:10]
    print(f"confusable pairs: {', '.join(p[0] + '/' + p[1] for p in pairs)}")
    dic = Dictionary(args.wordlist, pairs)
    llm = LLM(args.provider, model, pairs) if args.provider != "none" else None

    rows, streams, latencies = [], [], []
    for signer, c in caps.items():
        for w, item in c["words"].items():
            raw = clean_text(item["raw"])
            row = dict(signer=signer, reference=w, raw=raw, dictionary=dic.fix(raw))
            if llm:
                row["llm"], dt = llm(raw)
                latencies.append(dt)
            rows.append(row)
            print(f"  {signer:5s} {w:6s} raw {raw:10s} dict {row['dictionary']:10s}" +
                  (f" llm {row['llm']:10s} ({dt:.2f}s)" if llm else ""))
        raw = clean_text(c["stream"]["raw"])
        srow = dict(signer=signer, reference=c["stream"]["reference"], raw=raw, dictionary=dic.fix(raw))
        if llm:
            srow["llm"], dt = llm(raw)
            latencies.append(dt)
        streams.append(srow)

    keys = ["raw", "dictionary"] + (["llm"] if llm else [])
    names = {"raw": "raw caption", "dictionary": "dictionary fix", "llm": f"LLM ({args.provider}: {model})"}
    signers = list(caps)
    table = {}
    for k in keys:
        table[k] = {s: summarize([r for r in rows if r["signer"] == s], k) for s in signers}
        table[k]["all"] = summarize(rows, k)
        table[k]["stream_cer"] = float(np.mean([cer(r[k], r["reference"]) for r in streams]))

    fmt = lambda x: "n/a" if x != x else f"{x * 100:.0f}%"
    L = [f"## Iteration 10: caption cleanup, {names['llm'] if llm else 'dictionary baseline only'}", "",
         "| Version | " + " | ".join(f"{s} CER" for s in signers) + " | All CER | Word acc | Over-correction | Made worse | Stream CER |",
         "|---|" + "---|" * len(signers) + "---|---|---|---|---|"]
    for k in keys:
        t = table[k]
        L.append(f"| {names[k]} | " + " | ".join(f"{t[s]['cer'] * 100:.1f}%" for s in signers) +
                 f" | {t['all']['cer'] * 100:.1f}% | {t['all']['word_acc'] * 100:.0f}% | "
                 f"{'-' if k == 'raw' else fmt(t['all']['over_correction'])} | "
                 f"{'-' if k == 'raw' else fmt(t['all']['made_worse'])} | {t['stream_cer'] * 100:.1f}% |")
    n_ok = table["raw"]["all"]["n_raw_correct"]
    L += ["", f"Over-correction is measured on the {n_ok} of {len(rows)} words that were already right before the fix."]
    if latencies:
        L.append(f"LLM latency: {np.mean(latencies):.2f} s per call on average "
                 f"(median {np.median(latencies):.2f} s, {len(latencies)} calls).")
    L += ["", "| Person | Word | Raw | Dictionary" + (" | LLM" if llm else "") + " |",
          "|---|---|---|---" + ("|---" if llm else "") + "|"]
    for r in rows:
        L.append(f"| {r['signer']} | {r['reference']} | `{r['raw']}` | `{r['dictionary']}`" +
                 (f" | `{r['llm']}`" if llm else "") + " |")
    L += ["", "Streams (all 10 words in one caption):", ""]
    for r in streams:
        L.append(f"- {r['signer']} raw: `{r['raw']}`")
        L.append(f"  dictionary: `{r['dictionary']}`" + (f"  LLM: `{r['llm']}`" if llm else ""))

    tag = f"{args.provider}_{model}".replace("/", "-").replace(":", "-")
    open(os.path.join(args.out, f"{tag}.md"), "w").write("\n".join(L) + "\n")
    json.dump(dict(provider=args.provider, model=model, pairs=pairs, table=table, rows=rows, streams=streams,
                   latencies=latencies), open(os.path.join(args.out, f"{tag}.json"), "w"), indent=2)
    print("\n" + "\n".join(L[:len(keys) + 6]))
    print(f"\nFull table with every word: {args.out}/{tag}.md")


if __name__ == "__main__":
    main()
