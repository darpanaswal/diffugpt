#!/usr/bin/env python3
# eval_task.py   (place in scripts/ of the cloned DiffuLLaMA repo)
"""
Unified WiC / ProsQA / infill eval for DiffuGPT-M, for direct comparison
against DLIG/scripts/eval_task.py (the DLIG-repo counterpart) run with the
SAME hyperparameters.

STANDALONE: depends ONLY on this repo's own inference (model.py:
DiscreteDiffusionModel + generate_samples) and HF transformers. No DLIG /
attribution-backend imports -- this is the "official" generation stack
(the same one the trainer relies on), so comparing it against DLIG's own
reimplementation (backend.generate_trajectory / infill_generate_trajectory)
under matched hyperparameters is a clean A/B test of the two generation
paths.

WiC / ProsQA: merges scripts/eval_wic.py and scripts/eval_prosqa.py behind
--task. Both use a ddm-sft finetuned checkpoint (--model_name) and the
training-format '======' separator between question and answer/CoT.
  1. prompt = task question (BOS + encoded + '======' sep, matching training).
  2. build  input_ids = prefix + [mask]*gen_len  and  src_mask = 1 on prefix, 0 on gen.
  3. generate_samples() denoises the masked suffix (diffusion_steps, shift=True).
  4. decode the generated suffix, parse the answer, score against gold.

infill: the DLM-native ROCStories task (evaluation/eval-diffugpt.py's
eval_infilling), run on the BASE diffugpt-m checkpoint (no task SFT, no
separator -- this is not a question/answer task). Sentence 3 of a 5-sentence
story is masked (span length = the GOLD sentence's own token length -- an
oracle length, matching both this script and DLIG's attribution_infill.py)
and denoised conditioned on sentences 1-2 (left) AND 4-5 (right); scored by
word-level ROUGE-1/2/L F1 against the gold sentence. Not batched (one story
at a time, matching eval_infilling and DLIG's infill_generate_trajectory
loop), and no separator/checkpoint-dir requirement -- pass the base model
directory to --model_name.

Usage (defaults already match DLIG/scripts/eval_task.py's defaults --
gen_steps=12, max_new_tokens/gen_len=8/64/64 -- so no override needed for a
matched run):
  python -u -m scripts.eval_task --task wic \
      --model_name models/diffugpt-m-wic --data LLaMA-Factory/data/wic_test_raw.jsonl

  python -u -m scripts.eval_task --task prosqa \
      --model_name models/diffugpt-m-prosqa --data LLaMA-Factory/data/prosqa_test.json

  python -u -m scripts.eval_task --task infill \
      --model_name models/diffugpt-m --data data/rocstories_test.jsonl

If you deliberately want to compare at a DIFFERENT step count (e.g. this
repo's own eval default of 32/64, or the T=64 train/paper setting), pass the
SAME --diffusion_steps/--gen_len here as --gen_steps/--max_new_tokens to
DLIG/scripts/eval_task.py -- never change one without the other.
"""

import re
import os
import sys
import json
import argparse
from types import SimpleNamespace

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

# scripts/eval_task.py lives in scripts/ but model.py is at the repo ROOT.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "LLaMA-Factory", "src"))

from model import DiscreteDiffusionModel, generate_samples
from transformers import GPT2TokenizerFast
from tokenizers.pre_tokenizers import Digits
from itertools import chain


ANSWER_PREFIX = "###"


class MaskTokenWrapper(GPT2TokenizerFast):
    def __init__(self, tokenizer):
        EOS_TOKEN = tokenizer.eos_token
        PAD_TOKEN = "¨"
        SEP_TOKEN = "======"
        MASK_TOKEN_ID = tokenizer.vocab_size
        self.tokenizer = tokenizer
        self.digit_tokenizer = Digits(individual_digits=True)
        self.token2id = {token: id for token, id in self.tokenizer.vocab.items()}
        self.tokenizer.add_special_tokens({
            'pad_token': PAD_TOKEN,
            'eos_token': EOS_TOKEN,
            'sep_token': SEP_TOKEN,
            'mask_token': "[¨M¨]"
        })
        self.__dict__.update(self.tokenizer.__dict__.items())
        self.eos_token_id = self.token2id[EOS_TOKEN]
        self.pad_token_id = self.token2id[PAD_TOKEN]
        self.sep_token_id = self.token2id[SEP_TOKEN]
        self.mask_token_id = MASK_TOKEN_ID

    def encode(self, text, digit=True, **kwargs):
        if digit:
            chunks = self.digit_tokenizer.pre_tokenize_str(text)
            res = self.encode_batch([i[0] for i in chunks], digit=False, **kwargs)
            return res
        return self.tokenizer(text)

    def encode_batch(self, texts, digit=True, **kwargs):
        if digit:
            return [self.encode(text, digit=True, **kwargs) for text in texts]
        return list(chain.from_iterable([self.tokenizer.encode(text, **kwargs) for text in texts]))


# --------------------------------------------------------------------------- #
#  Task-specific prompts / parsers / data loaders
# --------------------------------------------------------------------------- #
def wic_prompt(ex):
    # MUST match training prompt (scripts/wic_to_diffusft.py build_prompt).
    return (f'Sentence 1: {ex["sentence1"].strip()}\n'
            f'Sentence 2: {ex["sentence2"].strip()}\n'
            f'Does the word "{ex["word"].strip()}" have the same meaning in '
            f'both sentences?')


def read_yes_no(text):
    """First Yes/No in the generated text -> 1 (same), 0 (diff), or None.
    Prefer the region after '###' if present; else scan the whole string."""
    scan = text
    if ANSWER_PREFIX in text:
        scan = text.split(ANSWER_PREFIX, 1)[1]
    t = scan.strip().lower()
    iy, ino = t.find("yes"), t.find("no")
    if iy == -1 and ino == -1:
        return None
    if iy == -1:
        return 0
    if ino == -1:
        return 1
    return 1 if iy < ino else 0


def load_wic_data(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def normalize(s):
    s = s.strip().lower().rstrip(".")
    return re.sub(r"\s+", " ", s)


def parse_answer(text):
    # Targets look like "<CoT> ### <Subject> is a <concept>." With gen_len padding,
    # the model often runs on AFTER the real answer. Take the FIRST statement
    # right after the FIRST "###", up to the first period, not the last.
    if ANSWER_PREFIX in text:
        after = text.split(ANSWER_PREFIX, 1)[1].strip()
        first = after.split(".", 1)[0].strip()
        return first if first else after
    sents = [s for s in re.split(r"(?<=\.)\s+", text.strip()) if s.strip()]
    return sents[-1] if sents else text.strip()


def final_concept(s):
    w = re.findall(r"[a-zA-Z]+", s)
    return w[-1].lower() if w else ""


# --------------------------------------------------------------------------- #
#  ProsQA bucket gating -- mirrors DLIG/experiments/prosqa/bucket_prosqa.py's
#  q_parse/pred_subject/label_row EXACTLY. raw exact/concept_match (above) can
#  match on the WRONG subject (e.g. answered about Sally when asked about
#  Bob); success/fail/off-manifold gates on subject match first, which is the
#  metric that should actually be reported.
# --------------------------------------------------------------------------- #
_PROSQA_Q = re.compile(r"Is (\w+) a (\w+) or (\w+)\s*\?")


def prosqa_q_parse(question):
    m = _PROSQA_Q.search(question)
    if not m:
        return None, None
    return m.group(1).lower(), (m.group(2).lower(), m.group(3).lower())


def prosqa_pred_subject(pred):
    m = re.search(r"\b([A-Z][a-z]+)\b", pred)
    return m.group(1).lower() if m else ""


def prosqa_bucket(question, pred, exact, concept_match):
    q_subj, opts = prosqa_q_parse(question)
    pc = final_concept(pred)
    ps = prosqa_pred_subject(pred)
    subj_match = (q_subj is not None) and (ps == q_subj)
    answer_valid = (opts is not None) and (pc in opts)
    if not subj_match:
        return "subj_wrong"
    if exact:
        return "correct"
    if concept_match:
        return "concept_only"
    if answer_valid:
        return "wrong_valid"
    return "concept_invalid"


def load_prosqa_data(path):
    return json.load(open(path))


# --------------------------------------------------------------------------- #
#  infill: ROCStories 5-sentence cloze. Word-level ROUGE-1/2/L F1, dependency-
#  free (no `evaluate`/`rouge_score` install required) so both repos compute
#  the IDENTICAL metric on the IDENTICAL generated text.
# --------------------------------------------------------------------------- #
from collections import Counter


def _words(s):
    return s.strip().lower().split()


def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def rouge_n_f1(pred, gold, n):
    p_ng, g_ng = _ngrams(_words(pred), n), _ngrams(_words(gold), n)
    overlap = sum((p_ng & g_ng).values())
    if overlap == 0 or not p_ng or not g_ng:
        return 0.0
    prec = overlap / sum(p_ng.values())
    rec = overlap / sum(g_ng.values())
    return 2 * prec * rec / (prec + rec)


def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(pred, gold):
    p, g = _words(pred), _words(gold)
    if not p or not g:
        return 0.0
    lcs = _lcs_len(p, g)
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def load_infill_data(path):
    """{'sentences': [s1..s5]} per line, matching DLIG's data/rocstories_test.jsonl."""
    stories = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sents = row.get("sentences") or [row[f"sentence{i}"] for i in range(1, 6)]
                if len(sents) == 5:
                    stories.append([s.strip() for s in sents])
    return stories


TASKS = {
    "wic": dict(
        default_data="LLaMA-Factory/data/wic_test_raw.jsonl",
        default_gen_len=8,
        loader=load_wic_data,
        prompt_fn=wic_prompt,
    ),
    "prosqa": dict(
        default_data="LLaMA-Factory/data/prosqa_test.json",
        default_gen_len=64,
        loader=load_prosqa_data,
        prompt_fn=lambda ex: ex["question"].strip(),
    ),
    "infill": dict(
        default_data="data/rocstories_test.jsonl",
        default_gen_len=None,   # oracle-length span, not a fixed budget
        loader=load_infill_data,
        prompt_fn=None,
    ),
}


def run_infill(model, tokenizer, gen_args, stories, out_file, device):
    """ROCStories middle-sentence infill. Mirrors
    evaluation/eval-diffugpt.py:eval_infilling's construction (oracle span
    length = gold token length, src on both sides of the gap, one story at a
    time -- not batched, matching that function and DLIG's
    infill_generate_trajectory loop), but scores word-level ROUGE-1/2/L F1
    (see rouge_n_f1/rouge_l_f1 above) instead of the `evaluate` library, so
    both repos compute the identical metric with no extra dependency."""
    mask_id = tokenizer.mask_token_id
    n, sum_r1, sum_r2, sum_rl = 0, 0.0, 0.0, 0.0
    pbar = tqdm(stories, desc="[diffugpt] infill", unit="story")
    with open(out_file, "w") as fout:
        for sents in pbar:
            s1, s2, s3, s4, s5 = sents
            prompt = s1 + " " + s2
            suffix = s4 + " " + s5
            middle = s3

            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            mid = tokenizer.encode(middle, add_special_tokens=False)
            suff = tokenizer.encode(suffix, add_special_tokens=False)
            x0 = prefix + [mask_id] * len(mid) + suff
            src_mask = [1] * len(prefix) + [0] * len(mid) + [1] * len(suff)
            inputs = {
                "input_ids": torch.tensor([x0], device=device),
                "src_mask": torch.tensor([src_mask], device=device),
            }
            with torch.no_grad():
                res = generate_samples(model, gen_args, tokenizer, inputs, verbose=False)
            row = res.tolist()[0]

            # SHIFT off-by-one, matching eval_infilling's slicing exactly.
            cut = (len(prefix) - 1) if gen_args.shift else len(prefix)
            end = (len(x0) - len(suff) - 1) if gen_args.shift else (len(x0) - len(suff))
            pred = tokenizer.decode(row[cut:end])

            r1 = rouge_n_f1(pred, middle, 1)
            r2 = rouge_n_f1(pred, middle, 2)
            rl = rouge_l_f1(pred, middle)
            n += 1
            sum_r1 += r1; sum_r2 += r2; sum_rl += rl

            fout.write(json.dumps({
                "prompt": prompt, "suffix": suffix, "gold": middle,
                "pred": pred, "rouge1": r1, "rouge2": r2, "rougeL": rl,
            }) + "\n")
            pbar.set_postfix(rouge1=f"{100*sum_r1/n:.1f}", rougeL=f"{100*sum_rl/n:.1f}")

    print(f"\n[RESULT] task=infill  n={n}")
    print(f"  rouge1 : {100*sum_r1/n:.2f}")
    print(f"  rouge2 : {100*sum_r2/n:.2f}")
    print(f"  rougeL : {100*sum_rl/n:.2f}")
    print(f"  preds -> {out_file}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model_name", required=True,
                    help="path to the ddm-sft finetuned DiffuGPT-M checkpoint dir "
                         "(LLaMA-Factory output_dir), or a HF model id.")
    ap.add_argument("--base_model_name", default="gpt2-medium",
                    help="AR base for CONFIG ONLY (DiffuGPT-M = gpt2-medium).")
    ap.add_argument("--data", default=None,
                    help="Defaults to the per-task file under LLaMA-Factory/data/.")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--gen_len", type=int, default=None,
                    help="Masked generation length; defaults to DLIG's own "
                         "per-task default (wic=8, prosqa=64) if omitted -- "
                         "NOT this repo's own eval defaults (8/100) -- so a "
                         "bare run already matches DLIG/scripts/eval_task.py. "
                         "If overridden, pass the SAME value there as "
                         "--max_new_tokens.")
    ap.add_argument("--diffusion_steps", type=int, default=12,
                    help="Denoising steps. Defaults to DLIG's own value (12), "
                         "NOT this repo's own eval default (64), so a bare "
                         "run already matches DLIG/scripts/eval_task.py. If "
                         "overridden, pass the SAME value there as "
                         "--gen_steps.")
    ap.add_argument("--logits_temp", type=float, default=0.95)
    ap.add_argument("--topp_temp", type=float, default=0.9)
    ap.add_argument("--shift", type=bool, default=True)   # do not change (DiffuGPT)
    ap.add_argument("--out_file", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None,
                    help="Defaults to 'cuda' if available else 'cpu' (matches "
                         "DLIG/scripts/eval_task.py's own auto-detect).")
    args = ap.parse_args()

    task = TASKS[args.task]
    data_path = args.data or task["default_data"]
    gen_len = args.gen_len or task["default_gen_len"]
    out_file = args.out_file or f"{args.task}_eval_task_preds.jsonl"

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.model_name):
        raise FileNotFoundError(
            f"checkpoint dir not found: {args.model_name}\n"
            f"  -> training likely did not save. Check the training log "
            f"(runs/.../train_*.log) and the yaml's output_dir.")

    config = AutoConfig.from_pretrained(args.model_name)
    base_tok = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    tokenizer = MaskTokenWrapper(base_tok)
    assert tokenizer.mask_token_id == 50257, (
        f"unexpected mask_token_id={tokenizer.mask_token_id} (expected 50257)")
    assert tokenizer.sep_token_id == 50155, (
        f"unexpected sep_token_id={tokenizer.sep_token_id} (expected 50155 for ======)")

    model = DiscreteDiffusionModel(
        model=args.base_model_name,
        config=config,
        tokenizer=tokenizer,
        device=device,
    )
    bin_path = os.path.join(args.model_name, "pytorch_model.bin")
    sft_path = os.path.join(args.model_name, "model.safetensors")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    elif os.path.isfile(sft_path):
        from safetensors.torch import load_file as _load_sft
        state_dict = _load_sft(sft_path, device="cpu")
    else:
        raise FileNotFoundError(
            f"no weights in {args.model_name} (looked for pytorch_model.bin / "
            f"model.safetensors). Check the training output_dir.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if any(k.startswith("lm_head") for k in missing) and \
       not any(k.startswith("lm_head") for k in state_dict):
        model.lm_head.weight = model.embed_tokens.weight
        missing = [k for k in missing if not k.startswith("lm_head")]
        print("[info] re-tied lm_head.weight <- embed_tokens.weight (GPT2 tied weights)")
    if missing:
        print(f"[warn] {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    model = model.to(device)
    model.eval()

    print(f"[INFO] device={device}")
    print(f"[INFO] task={args.task}  diffusion_steps={args.diffusion_steps}  "
          f"gen_len={gen_len}  data={data_path}")

    gen_args = SimpleNamespace(
        logits_temp=args.logits_temp,
        topp_temp=args.topp_temp,
        diffusion_steps=args.diffusion_steps,
        shift=args.shift,
    )

    data = task["loader"](data_path)
    if args.n_samples > 0:
        data = data[:args.n_samples]

    if args.task == "infill":
        run_infill(model, tokenizer, gen_args, data, out_file, device)
        return

    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    bos_id = tokenizer.bos_token_id
    bs = max(1, args.batch_size)

    items = []
    for ex in data:
        q = task["prompt_fn"](ex)
        # TRAINING FORMAT: <bos> question ====== <CoT/answer> <eos>
        # The "======" separator (sep_token_id 50155) MUST be reproduced or the
        # model runs off-distribution.
        prefix = [bos_id] + tokenizer.encode(q) + [tokenizer.sep_token_id]
        if args.task == "wic":
            items.append({"q": q, "label": int(ex["label"]), "prefix": prefix})
        else:
            items.append({"q": q, "gold": ex["answer"].strip(), "prefix": prefix})

    n = n_correct = n_exact = n_concept = n_unreadable = 0
    per_class = {0: [0, 0], 1: [0, 0]}
    bucket_counts = {"correct": 0, "concept_only": 0, "wrong_valid": 0,
                      "concept_invalid": 0, "subj_wrong": 0}
    pbar = tqdm(total=len(items), desc=f"[diffugpt] {args.task}", unit="ex")
    with open(out_file, "w") as fout:
        for start in range(0, len(items), bs):
            batch = items[start:start + bs]
            max_pref = max(len(it["prefix"]) for it in batch)

            input_ids, src_masks, pad_lens = [], [], []
            for it in batch:
                p = it["prefix"]
                n_pad = max_pref - len(p)
                row = [pad_id] * n_pad + p + [mask_id] * gen_len
                sm = [1] * (n_pad + len(p)) + [0] * gen_len
                input_ids.append(row)
                src_masks.append(sm)
                pad_lens.append(n_pad)

            inputs = {
                "input_ids": torch.tensor(input_ids, device=device),
                "src_mask": torch.tensor(src_masks, device=device),
            }
            with torch.no_grad():
                res = generate_samples(model, gen_args, tokenizer, inputs, verbose=False)
            res = res.tolist()

            for it, row in zip(batch, res):
                # SHIFT off-by-one: generate_samples returns a length seq_len-1
                # sequence where returned[j] == original position j+1. The gen
                # region starts at original position max_pref -> returned
                # index max_pref-1.
                cut = (max_pref - 1) if gen_args.shift else max_pref
                gen_ids = row[cut:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                n += 1
                if args.task == "wic":
                    pred = read_yes_no(gen_text)
                    gold = it["label"]
                    ok = (pred is not None and pred == gold)
                    per_class[gold][1] += 1
                    if pred is None:
                        n_unreadable += 1
                    else:
                        n_correct += int(ok)
                        per_class[gold][0] += int(ok)
                    rec = {"prompt": it["q"], "label": gold, "gen": gen_text,
                           "pred": pred, "ok": bool(ok)}
                else:
                    pred = parse_answer(gen_text)
                    gold = it["gold"]
                    exact = normalize(pred) == normalize(gold)
                    concept = final_concept(pred) == final_concept(gold)
                    n_exact += int(exact)
                    n_concept += int(concept)
                    rec = {"question": it["q"], "gold": gold, "gen": gen_text,
                           "pred_answer": pred, "exact": exact, "concept_match": concept}
                    bucket = prosqa_bucket(it["q"], pred, exact, concept)
                    bucket_counts[bucket] += 1
                    rec["bucket"] = bucket

                fout.write(json.dumps(rec) + "\n")

            pbar.update(len(batch))
            if args.task == "wic":
                pbar.set_postfix(acc=f"{100*n_correct/n:.1f}%")
            else:
                success = bucket_counts["correct"] + bucket_counts["concept_only"]
                pbar.set_postfix(exact=f"{100*n_exact/n:.1f}%",
                                  success=f"{100*success/n:.1f}%")
    pbar.close()

    print(f"\n[RESULT] task={args.task}  n={n}  (chance = 0.500)")
    if args.task == "wic":
        acc = n_correct / n if n else 0.0
        a1 = per_class[1][0] / per_class[1][1] if per_class[1][1] else 0.0
        a0 = per_class[0][0] / per_class[0][1] if per_class[0][1] else 0.0
        balanced = (a1 + a0) / 2
        print(f"  accuracy        : {100*acc:.2f}%   (unreadable counted wrong: {n_unreadable})")
        print(f"  same-sense (1)  : {100*a1:.2f}%   ({per_class[1][0]}/{per_class[1][1]})  correct=Yes")
        print(f"  diff-sense (0)  : {100*a0:.2f}%   ({per_class[0][0]}/{per_class[0][1]})  correct=No")
        print(f"  BALANCED        : {100*balanced:.2f}%   <-- the honest number")
    else:
        success = bucket_counts["correct"] + bucket_counts["concept_only"]
        fail = bucket_counts["wrong_valid"]
        off = bucket_counts["subj_wrong"] + bucket_counts["concept_invalid"]
        print(f"  answer_exact     : {100*n_exact/n:.2f}%   (unguarded string match, "
              f"kept for reference)")
        print(f"  raw concept_match: {100*n_concept/n:.2f}%   (unguarded -- can match "
              f"on the WRONG subject; not the metric to report)")
        for b, c in bucket_counts.items():
            print(f"  bucket {b:16}: {c:4}  ({100*c/n:5.1f}%)")
        print(f"  SUCCESS (correct + concept_only, subject-gated): "
              f"{100*success/n:.2f}%   ({success}/{n})  <-- the metric to report")
        print(f"  fail   (wrong_valid)                          : {100*fail/n:.2f}%   ({fail}/{n})")
        print(f"  off-manifold (subj_wrong + concept_invalid)   : {100*off/n:.2f}%   ({off}/{n})")
    print(f"  preds -> {out_file}")


if __name__ == "__main__":
    main()
