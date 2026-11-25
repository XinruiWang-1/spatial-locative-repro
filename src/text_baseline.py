#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_locative.py —— 定位型数据集上的零泄漏消融评测

================================================================
【数据】dataset_locative.csv(由 build_dataset_locative.py 生成)
    每条 = 一个 high_desc 步骤 = 一条"用空间线索锚定目标实例"的样本
    Target_Object 精确到实例(Pillow|+0.15|+1.74),真值只用于最后判分。

【两种检索设定】
  full  候选 = 场景内全部可移动物体(平均 ~20 个)  ← 贴近真实 RAG 检索
  same  候选 = 仅目标类别的同名实例(平均 ~2-3 个)   ← 纯消歧,文本完全无信息

【四组消融】
  A   纯文本 TF-IDF
  B   文本 × 距离衰减          score = t · (1 - w_d · d_rel)
  B2  文本 × 距离 × 方位衰减   再乘 (1 - w_o · ori)
  C   文本 + LLM 重排(--llm)
  D   空间 + LLM 重排(--llm)

【方位(left/right)的定义】
  "A 的左边"必须有观察者。这里定义观察者 = 机器人:
    forward = 归一化(锚点 - 机器人)          # 机器人看向锚点的方向
    left    = ori_sign × (forward_z, -forward_x)
  方位项对 lr_of / lr_agent 生效,其它关系只算距离。
  ori_sign 只在 train 上选定(默认 +1),不在验证/测试集上调。

【统计显著性】
  对 A/B 做 McNemar 精确检验(只看两者判断不一致的样本),
  输出 p 值,避免"涨了 0.3% 就宣称有效"。

【用法】
  python eval_locative.py --split valid_seen
  python eval_locative.py --split train --mode same --require-rel
  python eval_locative.py --split valid_unseen --llm
================================================================
"""

import os
import re
import json
import math
import time
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from synonyms import SYNONYMS
except ImportError:                      # 老目录里没有 synonyms.py 时退回建库脚本
    from build_dataset_locative import SYNONYMS

DATA_CSV = "dataset_locative.csv"
CAND_JSON = "loc_candidates.json"
ANCHOR_JSON = "loc_anchors.json"
OUT_DIR = "eval_out"
LLM_CACHE = "llm_prob_cache.json"   # (模型,样本,候选) -> P(yes),避免重复推理

W_DIST = 0.4        # 由 tune_on_train.py 在 train 上网格搜索确定(0.2~0.8 均 ≈59%,不敏感)
W_ORI = 0.2
ORI_SIGN = -1.0     # 观察者左手方向的符号约定,在 train 上选定
TOP_K_LLM = 5          # C/D 组给 LLM 的候选数(与 A@5 召回口径对齐)

# 只用"物体锚定"的关系作为主实验;agent 锚定缺朝向,单独报
OBJECT_ANCHORED = {"rel", "from", "lr_of"}


# ------------------------- 文本打分 -------------------------
class TextScorer:
    """
    全局拟合一次 TF-IDF,避免逐样本 fit 的开销。
    同义词扩展:ALFRED 类别名常与指令用词不一致(keychain vs "keys"),
    若不处理会让文本基线被不公平削弱。这里把同义词直接拼进候选文本,
    拼接后余弦相似度自然上升,不需要人为设定权重。
    若某同义词本身也是本样本的候选类别,则丢弃,避免制造新的歧义。
    """

    def __init__(self, instructions, categories, synonyms=None):
        self.syn = synonyms or {}
        self.vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
        corpus = list(set(categories)) + list(instructions)
        for extra in self.syn.values():
            corpus.extend(extra)
        self.vec.fit(corpus)
        self.cat_index = {c: i for i, c in enumerate(sorted(set(categories)))}

    def _expand(self, cat, present):
        toks = [cat]
        for s in self.syn.get(cat, []):
            if s not in present:
                toks.append(s)
        return " ".join(toks)

    def scores(self, instr, cands):
        present = {c["category"] for c in cands}
        docs = [self._expand(c["category"], present) for c in cands]
        try:
            iv = self.vec.transform([instr])
            cv = self.vec.transform(docs)
            return cosine_similarity(iv, cv)[0]
        except ValueError:
            return np.zeros(len(cands))


# ------------------------- 空间打分 -------------------------
def unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else (1.0, 0.0)


def spatial_scores(cands, text, anchor_xy, direction, polarity,
                   robot_xy, w_dist, w_ori, ori_sign, use_ori=True, ori_mode="soft"):
    """
    空间重排打分。

    ori_mode 决定"站错边"怎么罚(方向词的参考系已由 calibrate_orientation.py 标定:
    `X 左边的杯子` 这类指令, 观察者前方 = 机器人→参照物, 左轴 = (-fy, +fx), 与 train 一致率 70%):
      soft  两侧都有梯度, score *= 1 - w·(1-cos)/2      —— 旧实现, 相对动态范围只有 w/2
      gate  只罚错侧(硬),  score *= 1 - w·[cos<0]        —— 站错边直接打折
      sharp 只罚错侧(连续), score *= 1 - w·max(0,-cos)   —— 错得越多罚越多
    """
    ax, ay = anchor_xy
    dists = np.array([math.hypot(c["x"] - ax, c["y"] - ay) for c in cands])
    dmax = dists.max()
    d_rel = dists / dmax if dmax > 0 else np.zeros_like(dists)
    # polarity = -1 表示"越远越可能"(furthest from you)
    decay = 1.0 - w_dist * (d_rel if polarity > 0 else (1.0 - d_rel))
    score = text * decay

    if use_ori and direction and robot_xy is not None:
        fwd = unit((ax - robot_xy[0], ay - robot_xy[1]))
        left = (ori_sign * fwd[1], -ori_sign * fwd[0])   # 观察者左手方向
        want = left if direction == "left" else (-left[0], -left[1])
        cos = np.zeros_like(dists)
        for i, c in enumerate(cands):
            v = unit((c["x"] - ax, c["y"] - ay))
            cos[i] = max(-1.0, min(1.0, v[0] * want[0] + v[1] * want[1]))
        if ori_mode == "gate":
            fac = 1.0 - w_ori * (cos < 0)
        elif ori_mode == "sharp":
            fac = 1.0 - w_ori * np.maximum(0.0, -cos)
        else:                                            # soft(旧)
            fac = 1.0 - w_ori * (1.0 - cos) / 2.0
        score = score * fac
    return score


# ------------------------- 显著性 -------------------------
def mcnemar_exact(a_ok, b_ok):
    """返回 McNemar 精确检验 p 值(双侧)"""
    b = int(np.sum(a_ok & ~b_ok))   # A 对 B 错
    c = int(np.sum(~a_ok & b_ok))   # A 错 B 对
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), b, c


# ------------------------- LLM -------------------------
_LLM = {"tok": None, "model": None, "device": "cpu"}
LLM_FAIL = [0]          # 输出无法解析成候选编号的次数


def load_llm(model_name, device="cpu", dtype="float32"):
    """
    手动装载权重(不调用 from_pretrained 的加载器)。

    原因:本机 transformers 5.21 + torch 2.14(CPU) 下
    AutoModelForCausalLM.from_pretrained 会在权重装载环节卡住(>400s 不返回),
    而 from_config(13s) + safetensors.load_file(<1s) + load_state_dict 只需 ~15s。
    两者权重完全一致,仅绕过慢路径。
    Qwen2 的 lm_head 与 embed_tokens 共享权重,故 lm_head.weight 出现在 missing 属正常。

    device/dtype:GPU 上 7B 必须用 float16(约 14GB 显存),CPU 保持 float32。
    state_dict 的 copy_ 会按目标 dtype 自动转换,不需要手工 .half()。
    """
    import os
    import glob
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file

    print(f"[LLM] 加载 {model_name} ...", flush=True)
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[dtype]
    # transformers 5.x 用 dtype=,4.x 用 torch_dtype=;两边都兼容,免得换机器时炸在这里
    try:
        model = AutoModelForCausalLM.from_config(cfg, dtype=tdt)
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_config(cfg, torch_dtype=tdt)
        except TypeError:
            model = AutoModelForCausalLM.from_config(cfg).to(tdt)

    # 定位本地权重(缓存目录;若给了本地路径则直接用)
    if os.path.isdir(model_name):
        cands = glob.glob(os.path.join(model_name, "*.safetensors"))
    else:
        safe = model_name.replace("/", "--")
        cands = glob.glob(os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{safe}/snapshots/*/*.safetensors"))
    if not cands:
        raise FileNotFoundError(f"找不到本地权重: {model_name}")
    # 7B 的权重是 3 个分片(00001-of-00003 ...),必须逐片装载:
    # 只读第一个会静默丢掉 2/3 的参数(load_state_dict(strict=False) 不会报错)。
    # 逐片 update 而不是先合并成一个大 dict,峰值内存只有一个分片。
    files = sorted(cands)
    missing, unexpected = [], []
    for f in files:
        shard = load_file(f)
        missing, unexpected = model.load_state_dict(shard, strict=False)
        del shard
    print(f"[LLM] 权重分片 {len(files)} 个", flush=True)
    model.eval()
    if device != "cpu":
        model.to(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    print(f"[LLM] 完成 {time.time()-t0:.1f}s  device={device} dtype={dtype} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    _LLM.update(tok=tok, model=model, device=device)


def llm_pick(instr, cands):
    import torch
    tok, model = _LLM["tok"], _LLM["model"]
    lines = "\n".join(f"{i+1}. {c['id']} (x={c['x']:.2f}, z={c['y']:.2f})"
                      for i, c in enumerate(cands))
    prompt = (f'Instruction: "{instr}"\nCandidates:\n{lines}\n'
              "Reply with ONLY the number of the correct candidate.")
    msgs = [{"role": "system", "content": "You answer with a single number."},
            {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    pred = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\d+", pred)
    if m:
        k = int(m.group()) - 1
        if 0 <= k < len(cands):
            return cands[k]["id"]
    LLM_FAIL[0] += 1
    return ""


def _yes_prompt(instr, cand, anchor_xy=None):
    """构造点式提问。坐标是判断"near/left of"的唯一依据,必须写进去。"""
    lines = f'Instruction: "{instr}"\n'
    if anchor_xy is not None and anchor_xy[0] is not None:
        lines += f"The reference object is at (x={anchor_xy[0]:.2f}, z={anchor_xy[1]:.2f}).\n"
    lines += (f'Candidate: {str(cand["id"]).split("|")[0]} '
              f'at (x={cand["x"]:.2f}, z={cand["y"]:.2f}).\n'
              "Is this the object the instruction refers to? Answer yes or no.\nAnswer:")
    return lines


def llm_yes_prob_batch(instr, cands, anchor_xy=None, bs=8):
    """
    批量算 P(yes):一次前向吃多个候选,长序列左填充,取最后一位 logits。

    等比逐条快 2.5~4x(CPU 上 GEMM 利用率提升),数值与逐条一致到 1e-3。
    """
    import torch
    tok, model = _LLM["tok"], _LLM["model"]
    if not cands:
        return []
    texts = [tok.apply_chat_template([{"role": "user", "content": _yes_prompt(instr, c, anchor_xy)}],
                                     tokenize=False, add_generation_prompt=True)
             for c in cands]
    ids_y = tok.encode(" yes", add_special_tokens=False)
    ids_n = tok.encode(" no", add_special_tokens=False)
    old_side = tok.padding_side
    tok.padding_side = "left"
    out = []
    dev = _LLM.get("device", "cpu")
    for s in range(0, len(texts), max(1, bs)):
        chunk = texts[s:s + bs]
        enc = tok(chunk, return_tensors="pt", padding=True)
        if dev != "cpu":
            enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float().cpu()
        ly = logits[:, ids_y].max(dim=-1).values
        ln = logits[:, ids_n].max(dim=-1).values
        p = torch.softmax(torch.stack([ly, ln], dim=-1), dim=-1)[:, 0]
        out.extend(float(x) for x in p)
    tok.padding_side = old_side
    return out


def llm_yes_prob(instr, cand, anchor_xy=None):
    """
    逐候选打分(点式),不生成文本。

    为什么不用"请回答编号":实测 Qwen2-0.5B 在点选式提问下 8/8 全答第 1 项,
    打乱顺序后偏置只是换到第 2 项 —— 位置偏置完全盖过语义,该协议无效。
    点式打分对每个候选单独算 P(yes),与位置无关。

    必须把参照物坐标写进提示,否则模型没有据以判断"左/右/near"的信息:
    只给候选坐标时 0.5B 的 P(yes) 跨度仅 0.005(等于没判断),
    加入参照物坐标后跨度提升到 0.012(0.5B)/0.046(1.5B),且命中率高于随机。

    单条版本,内部走批量接口(bs=1),保证两条路径的提示词与取 logits 方式完全一致。
    """
    return llm_yes_prob_batch(instr, [cand], anchor_xy, bs=1)[0]


# ------------------------- 主流程 -------------------------
def main():
    global W_DIST, W_ORI, ORI_SIGN, TOP_K_LLM, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid_seen",
                    choices=["train", "valid_seen", "valid_unseen", "all"])
    ap.add_argument("--mode", default="both", choices=["full", "same", "both"])
    ap.add_argument("--require-rel", action="store_true", help="只用带可解析空间线索的样本")
    ap.add_argument("--multi-only", action="store_true", help="只用多实例样本")
    ap.add_argument("--discriminative", action="store_true",
                    help="只用空间描述唯一指向目标的样本(同类中最近 + 与次优拉开 min-margin)")
    ap.add_argument("--min-margin", type=float, default=0.3,
                    help="判别力过滤:目标与次优同类实例的最小距离差(米)")
    ap.add_argument("--no-synonyms", action="store_true", help="消融:关闭同义词扩展")
    ap.add_argument("--no-fallback", action="store_true", help="消融:关闭文本全零时的空间退化")
    ap.add_argument("--clean", action="store_true",
                    help="干净核心集:物体锚定 + 多实例 + 目标确为同类最近(rank=1) + 距离<=2.5m"
                         "(对应人工核验判定为 通过/别名 的样本,剔除 指令不唯一 与 可疑)")
    ap.add_argument("--max-dist", type=float, default=2.5,
                    help="--clean 时目标到锚点的最大允许距离(米),超过视为锚点可疑")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--topk-llm", type=int, default=TOP_K_LLM,
                    help="C/D 组给 LLM 的候选池:文本 top-K ∪ 空间 top-K")
    ap.add_argument("--bs", type=int, default=8,
                    help="P(yes) 批量前向的批大小(1=逐条,8 约快 3x,结果等价)")
    ap.add_argument("--cache-file", default="",
                    help="P(yes) 缓存路径(默认 llm_prob_cache.json;调试可另指一份)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto=有 CUDA 就用 GPU")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                    help="auto=GPU 用 float16,CPU 用 float32")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU 线程数(0=不动默认;8 线程 + bs=8 约 2.5x 提速)")
    ap.add_argument("--w-dist", type=float, default=W_DIST)
    ap.add_argument("--w-ori", type=float, default=W_ORI)
    ap.add_argument("--ori-mode", default="soft", choices=["soft", "gate", "sharp"],
                    help="方位项形态:soft=旧实现 / gate=只罚错侧(硬) / sharp=只罚错侧(连续)")
    ap.add_argument("--ori-sign", type=float, default=ORI_SIGN,
                    help="方位符号,+1/-1,只在 train 上选定")
    ap.add_argument("--out-dir", default="eval_out",
                    help="逐条预测与汇总的输出目录。换模型跑时建议分开"
                         "(如 eval_out_7b),避免覆盖已有结果")
    args = ap.parse_args()

    W_DIST, W_ORI, ORI_SIGN = args.w_dist, args.w_ori, args.ori_sign
    TOP_K_LLM = args.topk_llm
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    with open(CAND_JSON, encoding="utf-8") as fh:
        cand_map = json.load(fh)
    with open(ANCHOR_JSON, encoding="utf-8") as fh:
        anchor_map = json.load(fh)

    if args.split != "all":
        df = df[df["Split"] == args.split]
    if args.require_rel:
        df = df[df["Has_Rel"] == 1]
    if args.multi_only:
        df = df[df["Num_Same_Cat"] >= 2]
    if args.discriminative:
        df = df[(df["Rank_Same"] == 1)
                & ((df["D_Second"] - df["D_Target"]) >= args.min_margin)]
    if args.clean:
        # 人工核验口径(2026-11-17 与用户确认):
        #   通过/别名(正常) -> 保留;指令不唯一(Rank_Same>1) -> 剔除;可疑(距离>max_dist) -> 剔除
        # Cat_Mentioned=0 的"别名"样本保留(rag→cloth 不是数据错,同义词表已覆盖)
        df = df[(df["Has_Rel"] == 1)
                & (df["Rel_Kind"].isin(list(OBJECT_ANCHORED)))
                & (df["Num_Same_Cat"] >= 2)
                & (df["Rank_Same"] == 1)
                & (df["D_Target"] <= args.max_dist)]
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    print("=" * 74)
    print(f"集合={args.split}  样本={len(df)}  mode={args.mode}  w_dist={W_DIST} "
          f"w_ori={W_ORI} ori_sign={ORI_SIGN} llm={args.llm}")
    if args.discriminative:
        print(f"判别力过滤: 同类最近 且 与次优差距 >= {args.min_margin}m")
    if args.clean:
        print(f"干净核心集: 物体锚定+多实例+同类最近+距离<={args.max_dist}m")
    print("=" * 74)

    cats = sorted({c["category"] for v in cand_map.values() for c in v})
    scorer = TextScorer(df["Instruction"].astype(str).tolist(), cats,
                        synonyms=None if args.no_synonyms else SYNONYMS)

    prob_cache = {}
    cache_path = (args.cache_file or LLM_CACHE) if args.llm else None
    if args.threads > 0:
        import torch as _torch
        _torch.set_num_threads(args.threads)
        print(f"[cpu] torch 线程数 = {args.threads}")
    if args.llm:
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                prob_cache = json.load(fh)
            print(f"[cache] 已有 {len(prob_cache)} 条 P(yes) 缓存")
        dev = args.device
        if dev == "auto":
            import torch as _t
            dev = "cuda" if _t.cuda.is_available() else "cpu"
        dt = args.dtype
        if dt == "auto":
            dt = "float16" if dev == "cuda" else "float32"
        load_llm(args.model, device=dev, dtype=dt)

    modes = ["full", "same"] if args.mode == "both" else [args.mode]
    # stats[mode][stratum] = {'n','A','B','B2','C','D','A@3','A@5','k' , 'a_ok','b_ok'}
    stats = {m: defaultdict(lambda: defaultdict(float)) for m in modes}
    okflags = {m: defaultdict(lambda: {"a": [], "b": [], "c": [], "d": [], "e": []}) for m in modes}
    rows_out = []
    skipped = Counter()
    fallback = [0]          # 文本检索完全失败、退化为纯空间排序的次数
    lat = []          # 每样本 LLM 候选池打分总耗时
    lat_per = []      # 每候选平均耗时(缓存未命中才计)
    cand_n = [0]      # 实际前向的候选总数
    spreads = []      # 每样本 P(yes) 的极差,用于诊断 LLM 是否真有区分度

    n_total = len(df)
    for i_row, (_, row) in enumerate(df.iterrows()):
        # 进度提示：LLM 模式下一条样本要几秒到几十秒，没有输出会让人以为卡死。
        # 每 10 条打一次，且第一条就打，便于立刻确认候选池是否正常。
        if i_row == 0 or (i_row + 1) % 10 == 0:
            print(f"  [进度] {i_row + 1}/{n_total}", flush=True)
        tid = row["Task_ID"]
        instr = str(row["Instruction"])
        gt = str(row["Target_Object"])                 # ← 真值,仅判分用
        all_c = cand_map.get(tid, [])
        anchors = anchor_map.get(tid, {})
        if len(all_c) < 2:
            skipped["候选不足"] += 1
            continue

        has_rel = int(row["Has_Rel"]) == 1
        rel_kind = str(row["Rel_Kind"]) if has_rel else ""
        anchor_xy = (float(row["Anchor_X"]), float(row["Anchor_Y"])) if has_rel else None
        direction = str(row["Rel_Direction"]) if (has_rel and isinstance(row["Rel_Direction"], str)) else None
        polarity = float(row["Rel_Polarity"]) if has_rel and not pd.isna(row["Rel_Polarity"]) else 1.0
        robot_xy = tuple(anchors.get("robot", (None, None)))
        if robot_xy[0] is None:
            robot_xy = None

        for mode in modes:
            if mode == "same":
                cands = [c for c in all_c if c["category"] == row["Target_Category"]]
            else:
                cands = all_c
            if len(cands) < 2:
                skipped[f"{mode}:候选<2"] += 1
                continue
            if mode == "same" and len(cands) != int(row["Num_Same_Cat"]):
                skipped["same:数量不一致"] += 1
            if gt not in [c["id"] for c in cands]:
                skipped[f"{mode}:真值不在候选"] += 1
                continue

            t = scorer.scores(instr, cands)
            order_a = np.argsort(-t, kind="stable")
            pred_a = cands[order_a[0]]["id"]

            # 乘性融合的结构性缺陷:若文本检索对所有候选都打 0 分
            # (如 "sponge" 与类别名 "dishsponge" 无词面重叠),t × decay 恒为 0,
            # 空间信号被乘没,argmax 只能任取第一个。此时退化为纯空间排序。
            t_sp = t
            if (not args.no_fallback) and t.max() <= 1e-9:
                t_sp = np.ones_like(t)
                fallback[0] += 1

            pred_b = pred_b2 = pred_a
            applied = False
            if has_rel and anchor_xy and rel_kind in OBJECT_ANCHORED:
                sb = spatial_scores(cands, t_sp, anchor_xy, direction, polarity,
                                    robot_xy, W_DIST, W_ORI, ORI_SIGN, use_ori=False)
                pred_b = cands[int(np.argmax(sb))]["id"]
                sb2 = spatial_scores(cands, t_sp, anchor_xy, direction, polarity,
                                     robot_xy, W_DIST, W_ORI, ORI_SIGN,
                                     use_ori=True, ori_mode=args.ori_mode)
                pred_b2 = cands[int(np.argmax(sb2))]["id"]
                applied = True

            pred_c = pred_d = pred_e = ""
            p_b = p_len = margin = float("nan")
            pool_full = False
            if args.llm:
                # 候选池 = 文本 top-K ∪ 空间 top-K(去重);每个候选只算一次 P(yes)
                pool = [int(i) for i in order_a[:TOP_K_LLM]]
                if applied:
                    for i in np.argsort(-sb2, kind="stable")[:TOP_K_LLM]:
                        if int(i) not in pool:
                            pool.append(int(i))
                t0 = time.time()
                prob = {}
                todo = []
                for i in pool:
                    ck = f"{args.model}|{row['Sample_ID']}|{cands[i]['id']}"
                    if ck in prob_cache:
                        prob[i] = prob_cache[ck]
                    else:
                        todo.append(i)
                if todo:
                    ps = llm_yes_prob_batch(instr, [cands[i] for i in todo],
                                            anchor_xy, bs=args.bs)
                    for i, pv in zip(todo, ps):
                        prob[i] = pv
                        prob_cache[f"{args.model}|{row['Sample_ID']}|{cands[i]['id']}"] = pv
                lat.append(time.time() - t0)
                cand_n[0] += len(todo)
                if todo:
                    lat_per.append((time.time() - t0) / len(todo))
                spreads.append(max(prob.values()) - min(prob.values()))

                # C: 纯 LLM 重排(限定在文本检索 top-K 内)
                idx_c = [int(i) for i in order_a[:TOP_K_LLM]]
                pred_c = cands[max(idx_c, key=lambda i: prob.get(i, 0.0))]["id"]

                if applied:
                    # D: 纯 LLM 重排(限定在空间重排 top-K 内)
                    idx_d = [int(i) for i in np.argsort(-sb2, kind="stable")[:TOP_K_LLM]]
                    pred_d = cands[max(idx_d, key=lambda i: prob.get(i, 0.0))]["id"]
                    # E: 全融合 text × 空间衰减 × P(yes)
                    best_e, best_s = pool[0], -1.0
                    for i in pool:
                        sc = float(sb2[i]) * prob.get(i, 0.0)
                        if sc > best_s:
                            best_s, best_e = sc, i
                    pred_e = cands[best_e]["id"]
                else:
                    pred_d = pred_e = pred_c

                # 门控混合所需的量:B 选中实例与 LLM 最偏好实例的 P(yes) 差。
                # 只有"LLM 明显更看好另一个实例"时才值得推翻 B —— 留给离线扫阈值。
                if applied:
                    i_b = int(np.argmax(sb))
                    i_llm = max(pool, key=lambda i: prob.get(i, 0.0))
                    p_b = prob.get(i_b, float("nan"))
                    p_len = prob.get(i_llm, float("nan"))
                    margin = p_len - p_b
                    pool_full = (len(pool) == len(cands))
                else:
                    p_b = p_len = margin = float("nan")
                    pool_full = False

            # 分层
            keys = ["all"]
            if has_rel:
                keys.append("has_rel")
                if rel_kind in OBJECT_ANCHORED:
                    keys.append("obj_anchor")
            if int(row["Num_Same_Cat"]) >= 2:
                keys.append("multi")
                if has_rel and rel_kind in OBJECT_ANCHORED:
                    keys.append("multi_obj_anchor")
            if applied:
                keys.append(f"kind:{rel_kind}")

            a_ok = pred_a == gt
            b_ok = pred_b == gt
            for k in keys:
                s = stats[mode][k]
                s["n"] += 1
                s["k"] += len(cands)
                s["A"] += int(a_ok)
                s["B"] += int(b_ok)
                s["B2"] += int(pred_b2 == gt)
                if args.llm:
                    s["C"] += int(pred_c == gt)
                    s["D"] += int(pred_d == gt)
                    s["E"] += int(pred_e == gt)
                for K in (3, 5):
                    topk = [cands[i]["id"] for i in order_a[:K]]
                    s[f"A@{K}"] += int(gt in topk)
                okflags[mode][k]["a"].append(a_ok)
                okflags[mode][k]["b"].append(b_ok)
                if args.llm:
                    okflags[mode][k]["c"].append(pred_c == gt)
                    okflags[mode][k]["d"].append(pred_d == gt)
                    okflags[mode][k]["e"].append(pred_e == gt)

            rows_out.append({
                "Mode": mode,
                "Sample_ID": row["Sample_ID"], "Split": row["Split"], "Task": row["Task_ID"],
                "Instruction": instr, "Rel_Kind": rel_kind,
                "Rel_Direction": row["Rel_Direction"] if has_rel else "",
                "Target": gt, "Anchor": row["Anchor_Name"],
                "Num_Cand": len(cands), "Num_All_Cand": len(all_c),
                "T_Max": float(t.max()), "Text_Zero": bool(t.max() <= 1e-9),
                "pred_A": pred_a, "pred_B": pred_b,
                "pred_C": pred_c, "pred_D": pred_d, "pred_E": pred_e,
                "P_B": p_b, "P_LLM": p_len, "Margin": margin, "Pool_Full": pool_full,
                "ok_A": a_ok, "ok_B": b_ok, "ok_B2": pred_b2 == gt,
                "ok_C": pred_c == gt, "ok_D": pred_d == gt, "ok_E": pred_e == gt,
            })

    # ---------------- 报告 ----------------
    def show(mode):
        print(f"\n{'#' * 74}\n### 检索设定: {mode}"
              f"({'全部场景物体' if mode == 'full' else '仅同类别实例'})\n{'#' * 74}")
        order = ["all", "has_rel", "obj_anchor", "multi", "multi_obj_anchor"]
        order += sorted(k for k in stats[mode] if k.startswith("kind:"))
        for k in order:
            if k not in stats[mode]:
                continue
            s = stats[mode][k]
            n = int(s["n"])
            if n == 0:
                continue
            chance = s["k"] / n
            pa, pb = s["A"] / n * 100, s["B"] / n * 100
            p, b_only, c_only = mcnemar_exact(
                np.array(okflags[mode][k]["a"]), np.array(okflags[mode][k]["b"]))
            print(f"\n[{k}]  n={n}  平均候选={chance:.2f}  随机基线={100/chance:.2f}%")
            print(f"   A 纯文本        : {pa:6.2f}%")
            print(f"   B 文本+距离     : {pb:6.2f}%   (Δ{pb-pa:+.2f})  McNemar p={p:.4f}"
                  f"  [A独对{b_only} / B独对{c_only}]")
            print(f"   B2 文本+距离+方位: {s['B2']/n*100:6.2f}%   (Δ{s['B2']/n*100-pa:+.2f})")
            if args.llm and "C" in s:
                print(f"   C 文本+LLM      : {s['C']/n*100:6.2f}%   (Δ{s['C']/n*100-pa:+.2f})")
                print(f"   D 空间+LLM      : {s['D']/n*100:6.2f}%   (Δ{s['D']/n*100-pa:+.2f})"
                      f"   [相对B {s['D']/n*100-pb:+.2f}]")
                print(f"   E 全融合        : {s['E']/n*100:6.2f}%   (Δ{s['E']/n*100-pa:+.2f})"
                      f"   [相对B {s['E']/n*100-pb:+.2f}]")
                _fb = np.array(okflags[mode][k]["b"])
                for tag2, key2 in (("C", "c"), ("E", "e")):
                    _fo = np.array(okflags[mode][k][key2])
                    pp, bo, co = mcnemar_exact(_fo, _fb)   # B 对 / 对照 错?
                    print(f"      McNemar B vs {tag2}: p={pp:.4f}"
                          f"  [{tag2}独对{bo} / B独对{co}]")
            print(f"   召回 A@3={s['A@3']/n*100:.2f}%  A@5={s['A@5']/n*100:.2f}%")

    for m in modes:
        show(m)

    print(f"\n跳过统计: {dict(skipped)}")
    print(f"文本检索完全失败(退化为纯空间排序): {fallback[0]} 次"
          f"（占 {fallback[0]/max(len(df)*len(modes),1)*100:.1f}%）")
    if args.llm and _LLM.get("device") == "cuda":
        import torch as _t
        print(f"GPU 峰值显存: {_t.cuda.max_memory_allocated()/1024**3:.2f} GB")
    if lat:
        print(f"LLM 单样本候选池平均耗时: {np.mean(lat):.3f}s  "
              f"(中位 {np.median(lat):.3f}s, 共 {len(lat)} 个样本)")
        if lat_per:
            print(f"单候选平均 {np.mean(lat_per):.3f}s  (bs={args.bs}, "
                  f"实际前向 {cand_n[0]} 次)")
        print(f"LLM 输出无法解析成编号: {LLM_FAIL[0]} 次")
        if spreads:
            print(f"P(yes) 极差中位: {np.median(spreads):.4f}  "
                  f"(>0.05 才算 LLM 有实质区分度,否则其重排近随机)")

    tag = f"{args.split}_{args.mode}" + ("_rel" if args.require_rel else "") \
          + ("_multi" if args.multi_only else "") + ("_clean" if args.clean else "")
    pd.DataFrame(rows_out).to_csv(os.path.join(OUT_DIR, f"loc_pred_{tag}.csv"),
                                 index=False, encoding="utf-8-sig")
    summary = {m: {k: {kk: vv for kk, vv in s.items()} for k, s in stats[m].items()}
               for m in modes}
    with open(os.path.join(OUT_DIR, f"loc_summary_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(prob_cache, fh)
        print(f"P(yes) 缓存: {len(prob_cache)} 条 -> {cache_path}")
    print(f"\n逐条预测: {os.path.join(OUT_DIR, f'loc_pred_{tag}.csv')}")


if __name__ == "__main__":
    main()
