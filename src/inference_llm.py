#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
7B 推理器（LLM 包内，另一台机器上跑）

  python run_llm_hf.py --model /path/to/Qwen2.5-7B-Instruct --mode v2_verbal
  python run_llm_hf.py --model ... --mode v1_numeric
  python run_llm_hf.py --model ... --mode v3_cascade

  # 有 vLLM 时（快 5~10 倍）
  python run_llm_hf.py --model ... --mode v2_verbal --backend vllm

输出 preds.jsonl（追加写，断了重跑会自动跳过已完成的 Sample_ID）。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch


def load_prompts(path, mode):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.append((r["Sample_ID"], r["prompts"][mode]))
    return out


def done_keys(path):
    s = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    s.add((json.loads(line)["Sample_ID"], json.loads(line)["mode"]))
                except Exception:
                    pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--out", default="preds.jsonl")
    ap.add_argument("--mode", default="v2_verbal",
                    choices=["v1_numeric", "v2_verbal", "v3_cascade"])
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于冒烟")
    ap.add_argument("--load-4bit", action="store_true", help="显存不足时开 4bit 量化")
    ap.add_argument("--max-len", type=int, default=2048, help="输入截断长度")
    a = ap.parse_args()

    items = load_prompts(a.prompts, a.mode)
    done = done_keys(a.out)
    items = [(s, p) for s, p in items if (s, a.mode) not in done]
    if a.limit:
        items = items[:a.limit]
    print(f"待跑 {len(items)} 条（已完成 {len(done)} 条会被跳过）")
    if not items:
        return

    if a.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, max_model_len=4096, dtype="auto")
        sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens)
        outs = llm.generate([p for _, p in items], sp)
        with open(a.out, "a", encoding="utf-8") as f:
            for (sid, _), o in zip(items, outs):
                f.write(json.dumps(dict(Sample_ID=sid, mode=a.mode,
                                        raw=o.outputs[0].text.strip()),
                                   ensure_ascii=False) + "\n")
        print("完成")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # decoder-only 批量生成必须左填充，否则续写位置错乱、答案会串行
    tok.padding_side = "left"
    kw = dict(device_map="auto", trust_remote_code=True)
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    model.eval()

    def fmt(p):
        msgs = [{"role": "user", "content": p}]
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return p + "\n"

    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f, torch.inference_mode():
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            texts = [fmt(p) for _, p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_len).to(model.device)
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            for (sid, _), g, e in zip(chunk, gen, enc["input_ids"]):
                txt = tok.decode(g[e.shape[0]:], skip_special_tokens=True).strip()
                f.write(json.dumps(dict(Sample_ID=sid, mode=a.mode, raw=txt),
                                   ensure_ascii=False) + "\n")
            f.flush()
            if (i // a.batch) % 10 == 0:
                el = time.time() - t0
                print(f"{i + len(chunk)}/{len(items)}  {el:.0f}s  "
                      f"预计剩余 {el / (i + len(chunk)) * (len(items) - i - len(chunk)):.0f}s")
    print(f"完成，用时 {time.time() - t0:.0f}s → {a.out}")


if __name__ == "__main__":
    main()
