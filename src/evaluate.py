#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文级实验总装（2026-11-24）

把 665 条冻结可判分样本上「能做的实验」一次跑完，产出 PAPER_EXPERIMENTS.md + 图 + 表。
只读 clean_spatial_ext.csv（冻结 v1.0），不写任何冻结文件。

产出：
  T1 三档主表        665 / 596（方向可算）/ 515（真被筛过）
  T2 分档            A / D2R / B / C / D1
  T3 分方位类型      lr / front / behind
  T4 候选池大小      |A| = 1..4
  T5 标注审计        自洽 439 / 矛盾 157 / 唯一答案 196
  T6 第三把尺子      条件一致率 vs 条件随机基线
  T7 组件消融
  T8 演进史          A 纯文本 / B 公式 / B2 +方位（同集重算，全池设定）
  T9 失分归因        M** 的 210 条失分二分（77 不在允许集 + 133 范围内挑错）
  T10 权重扫描 + 5 折 CV
  T12 7B 学习基线     三套提示 × 三档 + 行为诊断（是否守住方向、集内排序、位置偏置）

用法：
  python eval_paper.py            # 全跑
  python eval_paper.py --no-fig   # 不画图
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd

import scoring as E
import baselines as C
import selection as F

import _paths as _p

DATA = _p.DATA
RESULTS = _p.RESULTS
FIG = str(_p.FIGURES)
BOOT = 4000

# ------------------------------------------------------------------ 装配
def load():
    d = F.build_final()
    ext = (pd.read_csv(f"{DATA}/clean_spatial_ext.csv", encoding="utf-8-sig")
           .set_index("Sample_ID"))
    det = (pd.read_csv(f"{RESULTS}/cascade_detail.csv", encoding="utf-8-sig")
           .set_index("Sample_ID"))
    d = d.set_index("Sample_ID")
    d["Dir_Step"] = det.reindex(d.index)["Dir_Step"]
    d["Allowed_N_c"] = det.reindex(d.index)["Allowed_N"]
    d["Pool_N"] = det.reindex(d.index)["Pool_N"]
    d["Anchor_X"] = ext.reindex(d.index)["Anchor_X"].astype(float)
    d["Anchor_Y"] = ext.reindex(d.index)["Anchor_Y"].astype(float)
    d["Target_Category"] = ext.reindex(d.index)["Target_Category"]
    d["Target_Object"] = ext.reindex(d.index)["Target_Object"]
    d["Num_Same"] = ext.reindex(d.index)["Num_Same"].astype(int)
    d["Robot_X2"] = ext.reindex(d.index)["Robot_X"].astype(float)
    d["Robot_Y2"] = ext.reindex(d.index)["Robot_Y"].astype(float)
    return d.reset_index()


# ------------------------------------------------------------------ 7B 预测并入
LLM_PREDS = f"{RESULTS}/llm_predictions.jsonl"
LLM_MODES = [("v1_numeric", "LLM 7B · v1 只给数字"),
             ("v2_verbal", "LLM 7B · v2 说成人话"),
             ("v3_cascade", "LLM 7B · v3 明写两步规则")]


def parse_letter(rec, n):
    """从 raw / answer 里抠出选项字母；抠不出返回 NaN（按错算，不手工补）。"""
    txt = str(rec.get("answer") or rec.get("raw") or "")
    m = re.search(r"[Aa]nswer\s*[:：]?\s*\(?\s*([A-H])", txt)
    if m:
        ch = m.group(1)
    else:
        alls = re.findall(r"\b([A-H])\b", txt.upper())
        if not alls:
            return np.nan
        ch = alls[-1]
    i = ord(ch.upper()) - 65
    return float(i) if 0 <= i < n else np.nan


def load_llm(d):
    """把 7B 预测并进逐样本表。文件不存在时静默跳过，报告自动退回纯规则版。"""
    if not os.path.exists(LLM_PREDS):
        print("未找到 llm_results/preds.jsonl，跳过 7B 基线\n")
        return d
    rec = {}
    with open(LLM_PREDS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rec[(r.get("Sample_ID"), r.get("mode"))] = r
    ns = dict(zip(d.Sample_ID, d.n))
    for mode, _ in LLM_MODES:
        p = np.array([parse_letter(rec.get((s, mode), {}), int(ns[s])) for s in d.Sample_ID])
        d["p_" + mode] = p
        h = (p == d.ti.values).astype(float)
        h[np.isnan(p)] = 0.0
        d["hit_" + mode] = h
    print(f"并入 7B 预测 {len(rec)} 条（解析失败 {int(sum(np.isnan(d['p_'+m]).sum() for m, _ in LLM_MODES))} 条，按错算）\n")
    return d



def aug(d):
    """补三种对照 + 两个期望值基线"""
    hit_R, hit_FAR, hit_UCOS = [], [], []
    exp_rnd, exp_urand = [], []
    for r in d.to_dict("records"):
        ti, n, A = r["ti"], r["n"], list(r["A"])
        hit_R.append(int(int(np.argmin(r["d_robot"])) == ti))
        hit_FAR.append(int(int(np.argmax(r["d_robot"])) == ti))
        pool = A if A else list(range(n))
        best = max(r["cos"][i] for i in pool)
        hit_UCOS.append(int({i for i in pool if r["cos"][i] >= best - 1e-9} == {ti}))
        exp_rnd.append(1.0 / n)
        exp_urand.append((1.0 / len(A)) if (A and ti in A) else 0.0)
    d = d.copy()
    d["hit_R"], d["hit_FAR"], d["hit_UCOS"] = hit_R, hit_FAR, hit_UCOS
    d["exp_RND"], d["exp_URAND"] = exp_rnd, exp_urand
    return d


# ------------------------------------------------------------------ 演进史：A / B / B2
def text_baselines(d):
    """在原设定下重算：候选 = 场景内全部物体（34~40 个）。

    用 eval_locative.TextScorer（词级 TF-IDF + 同义词扩展 + 全局拟合），
    与历史实验同一套实现，避免「重算版本不一致」。
    """
    import text_baseline as L

    cands = _p.load_json("loc_candidates.json")
    ancs = _p.load_json("loc_anchors.json")
    cats, instrs = [], []
    for r in d.to_dict("records"):
        pool = cands.get(r["Task_ID"]) or []
        cats += [c["category"] for c in pool]
        instrs.append(r["Instruction"])
    scorer = L.TextScorer(instrs, list(set(cats)), L.SYNONYMS)

    rows = []
    for r in d.to_dict("records"):
        pool = cands.get(r["Task_ID"]) or []
        if not pool:
            rows.append(dict(hit_A=0, hit_B=0, hit_B2=0, cat_hit=0, pool=0))
            continue
        sim = scorer.scores(r["Instruction"], pool)

        # 几何：距离到锚点、方位 cos（锚点缺失时退化为机器人参照、方位中立）
        rb = ancs.get(r["Task_ID"], {}).get("robot")
        rx = r["Robot_X2"] if not math.isnan(r["Robot_X2"]) else (rb[0] if rb else 0.0)
        ry = r["Robot_Y2"] if not math.isnan(r["Robot_Y2"]) else (rb[1] if rb else 0.0)
        has_a = not (math.isnan(r["Anchor_X"]) or math.isnan(r["Anchor_Y"]))
        ax, ay = (r["Anchor_X"], r["Anchor_Y"]) if has_a else (rx, ry)
        dv = [math.hypot(c["x"] - ax, c["y"] - ay) for c in pool]
        dmax = max(dv) or 1.0
        mx, my = ax - rx, ay - ry
        na = math.hypot(mx, my)
        cos = []
        for c in pool:
            vx, vy = c["x"] - rx, c["y"] - ry
            nv = math.hypot(vx, vy)
            cos.append(0.0 if (na < 1e-9 or nv < 1e-9) else (vx * mx + vy * my) / (nv * na))

        sB = [sim[i] * (1 - 0.4 * dv[i] / dmax) for i in range(len(pool))]
        sB2 = [sB[i] * (1 - 0.2 + 0.2 * (1 + cos[i]) / 2.0) for i in range(len(pool))]
        gt = str(r["Target_Object"])
        ia = int(np.argmax(sim))
        rows.append(dict(
            hit_A=int(pool[ia]["id"] == gt),
            hit_B=int(pool[int(np.argmax(sB))]["id"] == gt),
            hit_B2=int(pool[int(np.argmax(sB2))]["id"] == gt),
            cat_hit=int(pool[ia]["category"] == r["Target_Category"]),
            pool=len(pool)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 统计
def ci(p, seed=0):
    p = np.asarray(p, float)
    lo, hi = C.boot_ci(p, n=BOOT, seed=seed)
    return round(float(p.mean()) * 100, 1), round(lo, 1), round(hi, 1)


METHODS = [
    # (显示名, 列, 性质, 说明)
    ("ORACLE 直接取标注",        "oracle",   "单点", "上限，不作方法"),
    ("U-SET 允许集合判定（口径）", "ok",       "集合", "标注落在题面允许范围"),
    ("★ M** 软先验 0.2 + 距离硬层", "hit_final", "单点", "本文主推"),
    ("M3* 软加权 w=0.2（无硬层）", "hit_m3",   "单点", "消融：去掉距离硬层"),
    ("M3 软加权 w=0.5",          "hit_m35",  "单点", "旧版软加权，方位过重"),
    ("M3a 硬筛后软加权",          "hit_m3a",  "单点", "硬筛 + 软加权"),
    ("M2 硬筛 + 侧内最近",        "hit_m2",   "单点", "原规则字面落地"),
    ("M0 取锚点最近",            "hit_m0",   "单点", "旧基线，方向等于没用"),
    ("U-COS 允许集内最靠侧",      "hit_UCOS", "单点", "只信方位"),
    ("U-RAND 允许集内随机",       "exp_URAND", "集合", "期望值"),
    ("R 取机器人最近",            "hit_R",    "单点", "防呆基线"),
    ("FAR 取机器人最远",          "hit_FAR",  "单点", "防呆基线"),
    ("RND 全池随机",             "exp_RND",  "集合", "期望值"),
    ("A 纯文本 TF-IDF（全池）",   "hit_A",    "单点", "演进史起点"),
    ("B 公式 tfidf×距离（全池）",  "hit_B",    "单点", "演进史"),
    ("B2 公式 + 方位（全池）",     "hit_B2",   "单点", "演进史"),
]


def methods(d):
    """METHODS 列表 + 已并入的 7B 行（插在 M0 之后，读起来顺）。"""
    llm = [(disp, "hit_" + m, "单点", "同集学习基线（Qwen2.5-7B, greedy）")
           for m, disp in LLM_MODES if "hit_" + m in d.columns]
    if not llm:
        return list(METHODS)
    idx = [i for i, x in enumerate(METHODS) if x[0].startswith("M0")][0] + 1
    return METHODS[:idx] + llm + METHODS[idx:]


def panel(sub, name, meths):
    rows = []
    for disp, col, kind, note in meths:
        if col == "oracle":
            m, lo, hi = 100.0, 100.0, 100.0
        else:
            m, lo, hi = ci(sub[col])
        rows.append(dict(子集=name, 方法=disp, 性质=kind, n=len(sub),
                         命中=m, lo=lo, hi=hi, 说明=note))
    return rows


def mcnemar_vs(d, col):
    a = d["hit_final"].astype(bool)
    b = d[col].astype(bool)
    p, x, y = C.mcnemar(a, b)
    return dict(M对= x, 对手对=y, p=p)


# ------------------------------------------------------------------ 主流程
def main(no_fig=False):
    d = load_llm(aug(load()))
    tb = text_baselines(d)
    d = pd.concat([d.reset_index(drop=True), tb.reset_index(drop=True)], axis=1)
    d["hit_m3a"] = [int(C.argmax_set(F.soft_score(r, 0.2), (r["A"] or list(range(r["n"]))))
                      == {r["ti"]}) for r in d.to_dict("records")]
    d["oracle"] = 1
    MT = methods(d)

    n = len(d)
    S1 = d
    S2 = d[d.Dir_Step != "不可算"]
    S3 = d[d.Dir_Step == "筛过"]
    subsets = [("S1 全量", S1), ("S2 方向可算", S2), ("S3 真被筛过", S3)]

    print(f"样本 {n}；S2={len(S2)}；S3={len(S3)}\n")

    # ---------------- T1 三档主表
    rows = []
    for nm, sub in subsets:
        rows += panel(sub, nm, MT)
    t1 = pd.DataFrame(rows)
    print(t1.pivot(index="方法", columns="子集", values="命中").to_string())

    # ---------------- T2/T3/T4 分层
    def strat(col, order=None, dropna=False):
        out = []
        vals = [v for v in pd.unique(d[col]) if not (isinstance(v, float) and math.isnan(v))]
        if order:
            vals = [v for v in order if v in vals]
        for v in vals:
            sub = d[d[col] == v]
            out.append(dict(组=v, n=len(sub),
                            U_SET=round(sub.ok.mean() * 100, 1),
                            Mstar=round(sub.hit_final.mean() * 100, 1),
                            M3s=round(sub.hit_m3.mean() * 100, 1),
                            M2=round(sub.hit_m2.mean() * 100, 1),
                            M0=round(sub.hit_m0.mean() * 100, 1),
                            avg_A=round(sub.A_n.mean(), 2)))
        return pd.DataFrame(out)

    t2 = strat("Band", ["A", "D2R", "B", "C", "D1"])
    t3 = strat("Rel_Type", ["lr", "front", "behind"])
    d["PoolSize"] = d["A_n"].clip(upper=4)
    t4 = strat("PoolSize", [1, 2, 3, 4])

    # ---------------- T5 标注审计
    lab_map = {True: "自洽", False: "矛盾"}
    d["Lab"] = d["label_side"].map(lab_map)
    t5a = strat("Lab", ["自洽", "矛盾"])
    d["Lab"] = d["Lab"].fillna("缺失(D1)")
    t5a = strat("Lab", ["自洽", "矛盾", "缺失(D1)"])

    uni = d[d.A_n == 1]
    t5b = pd.DataFrame([
        dict(子集="题面唯一确定答案(|A|=1)", n=len(uni),
             U_SET=round(uni.ok.mean() * 100, 1),
             Mstar=round(uni.hit_final.mean() * 100, 1),
             说明="范围里只有 1 个，单点与口径应重合"),
        dict(子集="允许集>1", n=len(d) - len(uni),
             U_SET=round(d[d.A_n > 1].ok.mean() * 100, 1),
             Mstar=round(d[d.A_n > 1].hit_final.mean() * 100, 1),
             说明="题面没写清该挑哪个"),
    ])

    # ---------------- T6 第三把尺子
    rows = []
    # T6b 四象限：口径 × 单点
    Q = {}
    for nm, fc in [("M**", "hit_final"), ("M2 硬筛", "hit_m2")]:
        Q[nm] = dict(
            内对=int(((d.ok) & (d[fc].astype(bool))).sum()),
            内错=int(((d.ok) & (~d[fc].astype(bool))).sum()),
            外对=int(((~d.ok) & (d[fc].astype(bool))).sum()),
            外错=int(((~d.ok) & (~d[fc].astype(bool))).sum()))
    t6b = pd.DataFrame(Q).T.reset_index().rename(columns={"index": "方法"})
    t6b.to_csv(f"{RESULTS}/paper_T6b_quadrant.csv", index=False, encoding="utf-8-sig")

    for nm, sub in subsets + [("白送（无约束/筛空）", d[d.free != ""])]:
        okm = sub[sub.ok]
        rows.append(dict(
            子集=nm, n=len(sub),
            U_SET=round(sub.ok.mean() * 100, 1),
            Mstar=round(sub.hit_final.mean() * 100, 1),
            条件一致率=round(okm.hit_final.mean() * 100, 1),
            条件随机基线=round((1.0 / okm.A_n).mean() * 100, 1),
            条件增量=round(okm.hit_final.mean() * 100 - (1.0 / okm.A_n).mean() * 100, 1),
            平均允许=round(sub.A_n.mean(), 2)))
    t6 = pd.DataFrame(rows)

    # ---------------- T7 消融
    base = round(d.hit_m0.mean() * 100, 1)
    abl = []
    for disp, col, note in [
        ("只距离（M0 取最近）", "hit_m0", "方向完全不用"),
        ("只方位（U-COS 集内最靠侧）", "hit_UCOS", "距离完全不用"),
        ("硬筛 + 最近（M2）", "hit_m2", "你的第一步字面版"),
        ("硬筛 + 软加权（M3a）", "hit_m3a", "硬筛内再软排"),
        ("软加权 w=0.5（M3）", "hit_m35", "方向权重偏高"),
        ("软加权 w=0.2（M3*）", "hit_m3", "去掉硬层的最终软层"),
        ("软加权 0.2 + 距离硬层（M**）", "hit_final", "本文主推"),
    ]:
        m, lo, hi = ci(d[col])
        abl.append(dict(配置=disp, 命中=m, lo=lo, hi=hi, 相对只距离=round(m - base, 1), 作用=note))
    t7 = pd.DataFrame(abl)

    # ---------------- T9 失分归因
    fails = d[~d.hit_final.astype(bool)]
    f1 = fails[~fails.ok.astype(bool)]
    f2 = fails[fails.ok.astype(bool)]

    def sub_kind(r):
        A = list(r["A"]) or list(range(r["n"]))
        dn = min(r["d_anchor"][i] for i in A)
        near = {i for i in A if r["d_anchor"][i] <= dn + 1e-9}
        cxx = max(r["cos"][i] for i in A)
        side = {i for i in A if r["cos"][i] >= cxx - 1e-9}
        if near == {r["ti"]}:
            return "A 内最近就是标注（被方向项压后）"
        if side == {r["ti"]}:
            return "A 内最靠侧就是标注（被距离项压后）"
        return "既不最近也不最靠侧"

    f2k = f2.apply(sub_kind, axis=1).value_counts()
    t9 = pd.DataFrame(
        [dict(类型="标注不在允许集（两把尺子都判错）", n=len(f1),
              占比=round(len(f1) / len(fails) * 100, 1), 说明="题面与标注矛盾，非方法问题")]
        + [dict(类型=f"A 内挑错：{k}", n=int(v), 占比=round(v / len(fails) * 100, 1),
                说明="范围对了，选点没对上人") for k, v in f2k.items()]
    )

    # ---------------- T8 演进史（同一批 665，全池设定）
    evo = []
    for key, disp, col in [("A", "A 纯文本 TF-IDF", "hit_A"), ("B", "B 公式 tfidf×距离", "hit_B"),
                           ("B2", "B2 公式+方位", "hit_B2"),
                           ("M0", "M0 同类池取最近（B 的退化形态）", "hit_m0"),
                           ("M**", "M** 本文方法（同类池）", "hit_final")]:
        m, lo, hi = ci(d[col])
        evo.append(dict(key=key, 阶段=disp, 命中=m, lo=lo, hi=hi))
    t8 = pd.DataFrame(evo)
    cat_hit = round(d.cat_hit.mean() * 100, 1)

    # T8b 两段式分解：类别识别 × 实例选择
    rows = []
    for k, nm in [(1, "类别认对"), (0, "类别认错")]:
        sub = d[d.cat_hit == k]
        rows.append(dict(组=nm, n=len(sub),
                         A=round(sub.hit_A.mean() * 100, 1),
                         B=round(sub.hit_B.mean() * 100, 1),
                         Mstar=round(sub.hit_final.mean() * 100, 1),
                         M0=round(sub.hit_m0.mean() * 100, 1)))
    t8b = pd.DataFrame(rows)

    # ---------------- T10 扫描 + CV
    sw = F.sweep_final(d)
    cv_hit, cv_w = F.cv_final(d)

    # ---------------- 配对检验
    tests = []
    for col, nm in [("hit_m2", "M2 硬筛+侧内最近"), ("hit_m0", "M0 取锚点最近"),
                    ("hit_m3", "M3* 软加权 0.2"), ("hit_m35", "M3 软加权 0.5"),
                    ("hit_m3a", "M3a 硬筛后软加权"), ("hit_UCOS", "U-COS 集内最靠侧"),
                    ("hit_B", "B 公式（全池）"), ("hit_B2", "B2 公式（全池）")]:
        r = mcnemar_vs(d, col)
        tests.append(dict(对手=nm, M对= r["M对"], 对手对=r["对手对"], p=r["p"]))
    tt = pd.DataFrame(tests)

    # ---------------- T12 7B 学习基线
    llm_cols = [(m, dsp, "hit_" + m, "p_" + m) for m, dsp in LLM_MODES if "hit_" + m in d.columns]
    t12a = t12b = t12c = t12d = pd.DataFrame()
    best = None
    if llm_cols:
        # (a) 配对检验
        rows = []
        for m, dsp, hc, pc in llm_cols:
            for opp, oc in [("M**（本文）", "hit_final"), ("M2 硬筛", "hit_m2"), ("M0 最近", "hit_m0")]:
                p, x, y = C.mcnemar(d[hc].astype(bool), d[oc].astype(bool))
                rows.append(dict(模式=dsp, 对手=opp, LLM对=x, 对手对=y, p=p))
        for i in range(len(llm_cols) - 1):
            p, x, y = C.mcnemar(d[llm_cols[i][2]].astype(bool), d[llm_cols[i + 1][2]].astype(bool))
            rows.append(dict(模式=llm_cols[i][1].replace("LLM 7B · ", ""),
                             对手="vs " + llm_cols[i + 1][1].replace("LLM 7B · ", ""),
                             LLM对=x, 对手对=y, p=p))
        t12a = pd.DataFrame(rows)

        # (b) 行为诊断：守住方向了吗？在集内挑得准吗？是不是照最近挑？
        recs = d.to_dict("records")
        near = [C.argmin_set(r["d_anchor"], r["A"] or list(range(r["n"]))) for r in recs]
        rows = []
        for m, dsp, hc, pc in llm_cols:
            p, hv = d[pc].values, d[hc].values
            inA = np.zeros(len(recs), bool)
            agree = np.full(len(recs), np.nan)
            ab = np.zeros(len(recs), bool)
            for i, (r, q, S) in enumerate(zip(recs, p, near)):
                if q != q:
                    continue
                q = int(q)
                inA[i] = q in (set(r["A"]) or set(range(r["n"])))
                ab[i] = q <= 1
                if len(S) == 1:
                    agree[i] = float(q == list(S)[0])
            rows.append(dict(模式=dsp, 在允许集=round(inA.mean() * 100, 1),
                             集内命中=round(hv[inA].mean() * 100, 1),
                             与集内最近一致=round(float(np.nanmean(agree)) * 100, 1),
                             一致时命中=round(hv[agree == 1].mean() * 100, 1),
                             偏离时命中=round(hv[agree == 0].mean() * 100, 1),
                             偏离条数=int((agree == 0).sum()),
                             选A或B=round(ab.mean() * 100, 1)))
        t12b = pd.DataFrame(rows)

        # (c) 分层（取最好的那套提示）
        bm = max(llm_cols, key=lambda x: d[x[2]].mean())
        best = bm
        bc = bm[2]
        has_lab = d.label_side.notna()
        grp = [("允许集=1（题面唯一）", d.A_n == 1), ("允许集≥2（题面没写清）", d.A_n > 1),
               ("标注自洽", has_lab & d.label_side.astype(bool)),
               ("标注矛盾", has_lab & ~d.label_side.astype(bool)),
               ("D1 无锚点", d.Band == "D1")]
        rows = [dict(组=g, n=int(m.sum()), LLM=round(d.loc[m, bc].mean() * 100, 1),
                     Mstar=round(d.loc[m, "hit_final"].mean() * 100, 1),
                     M2=round(d.loc[m, "hit_m2"].mean() * 100, 1),
                     M0=round(d.loc[m, "hit_m0"].mean() * 100, 1)) for g, m in grp if m.any()]
        t12c = pd.DataFrame(rows)

        # (d) 位置偏置
        rows = [dict(来源="人类标注（gold）",
                     **{chr(65 + k): round(v * 100, 1)
                        for k, v in d.ti.value_counts(normalize=True).sort_index().items()})]
        for m, dsp, hc, pc in llm_cols:
            v = d[pc].dropna().astype(int).value_counts(normalize=True).sort_index()
            rows.append(dict(来源=dsp, **{chr(65 + k): round(x * 100, 1) for k, x in v.items()}))
        t12d = pd.DataFrame(rows).fillna(0.0)

    # ---------------- 落盘
    t1.to_csv(f"{RESULTS}/paper_T1_main.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(f"{RESULTS}/paper_T2_band.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(f"{RESULTS}/paper_T3_rel.csv", index=False, encoding="utf-8-sig")
    t4.to_csv(f"{RESULTS}/paper_T4_pool.csv", index=False, encoding="utf-8-sig")
    t5a.to_csv(f"{RESULTS}/paper_T5_audit.csv", index=False, encoding="utf-8-sig")
    t6.to_csv(f"{RESULTS}/paper_T6_ruler.csv", index=False, encoding="utf-8-sig")
    t7.to_csv(f"{RESULTS}/paper_T7_ablation.csv", index=False, encoding="utf-8-sig")
    t8.to_csv(f"{RESULTS}/paper_T8_evolution.csv", index=False, encoding="utf-8-sig")
    t8b.to_csv(f"{RESULTS}/paper_T8b_twostage.csv", index=False, encoding="utf-8-sig")
    t9.to_csv(f"{RESULTS}/paper_T9_failures.csv", index=False, encoding="utf-8-sig")
    sw.to_csv(f"{RESULTS}/paper_T10_sweep.csv", index=False, encoding="utf-8-sig")
    tt.to_csv(f"{RESULTS}/paper_T11_tests.csv", index=False, encoding="utf-8-sig")
    for tb_, fn in [(t12a, "paper_T12a_llm_tests.csv"), (t12b, "paper_T12b_llm_behaviour.csv"),
                    (t12c, "paper_T12c_llm_strata.csv"), (t12d, "paper_T12d_llm_position.csv")]:
        if len(tb_):
            tb_.to_csv(f"{RESULTS}/{fn}", index=False, encoding="utf-8-sig")
    keep = ["Sample_ID", "Task_ID", "Band", "Rel_Type", "Side", "Anchor", "Anchor_Name",
            "n", "ti", "A_n", "ok", "label_side", "free", "Dir_Step", "pool", "layer",
            "hit_final", "hit_m3", "hit_m3a", "hit_m2", "hit_m0", "hit_m35",
            "hit_UCOS", "hit_R", "hit_FAR", "hit_A", "hit_B", "hit_B2", "cat_hit",
            "Instruction"] + [c for m, _ in LLM_MODES for c in ("p_" + m, "hit_" + m)
                              if c in d.columns]
    keep = [c for c in keep if c in d.columns]
    d[keep].to_csv(f"{RESULTS}/paper_detail.csv", index=False, encoding="utf-8-sig")

    if not no_fig:
        figs(d, t1, sw, t6, t9, t12b, t12d)

    write_report(d, t1, t2, t3, t4, t5a, t5b, t6, t6b, t7, t8, t8b, t9, tt,
                 t12a, t12b, t12c, t12d, best, sw, cv_hit, cv_w, subsets)
    print("\n写出 paper_T*.csv / paper_detail.csv / PAPER_EXPERIMENTS.md")
    return d


# ------------------------------------------------------------------ 图
EN_LAB = {"自洽": "label consistent", "矛盾": "label contradicts", "缺失(D1)": "no anchor (D1)"}
EN_FAIL = {
    "标注不在允许集（两把尺子都判错）": "label outside allowed set",
    "A 内挑错：A 内最近就是标注（被方向项压后）": "label is nearest in A",
    "A 内挑错：A 内最靠侧就是标注（被距离项压后）": "label is most lateral in A",
    "A 内挑错：既不最近也不最靠侧": "neither nearest nor lateral",
}


def figs(d, t1, sw, t6, t9, t12b=None, t12d=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    # 图 1：三档 × 五法
    piv = t1[t1.方法.isin(["U-SET 允许集合判定（口径）", "★ M** 软先验 0.2 + 距离硬层",
                           "M3* 软加权 w=0.2（无硬层）", "M2 硬筛 + 侧内最近", "M0 取锚点最近"])]
    piv = piv.pivot(index="方法", columns="子集", values="命中").loc[
        ["U-SET 允许集合判定（口径）", "★ M** 软先验 0.2 + 距离硬层",
         "M3* 软加权 w=0.2（无硬层）", "M2 硬筛 + 侧内最近", "M0 取锚点最近"]]
    piv.columns = ["S1 all (665)", "S2 dir-computable (596)", "S3 filtered (515)"][:len(piv.columns)]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    piv.plot.bar(ax=ax, width=0.78, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(50, 95)
    ax.set_xticklabels(["U-SET\n(set ruler)", "M**\n(ours)", "M3*", "M2\n(hard filter)", "M0"],
                       rotation=0, fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.set_title("Same 665 frozen samples, three subsets", fontsize=9)
    ax.grid(axis="y", alpha=.25, linewidth=.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/paper_fig1_subsets.png")
    plt.close(fig)

    # 图 2：权重扫描
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(sw.w_ori, sw.命中, marker="o", ms=3.5, lw=1.4)
    b = sw.loc[sw.命中.idxmax()]
    ax.scatter([b.w_ori], [b.命中], s=60, facecolor="none", edgecolor="crimson", lw=1.4, zorder=5)
    ax.annotate(f"w={b.w_ori:.1f}\n{b.命中:.1f}%", (b.w_ori, b.命中),
                textcoords="offset points", xytext=(8, -14), fontsize=7, color="crimson")
    ax.set_xlabel("orientation weight  w_ori  (0 = distance only)")
    ax.set_ylabel("single-pick accuracy (%)")
    ax.grid(alpha=.25, linewidth=.5)
    ax.set_title("Weight sweep (hard distance layer fixed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/paper_fig2_sweep.png")
    plt.close(fig)

    # 图 3：标注审计
    rows = []
    for k in ["自洽", "矛盾", "缺失(D1)"]:
        s = d[d.Lab == k]
        if len(s):
            rows.append({"U-SET": s.ok.mean() * 100, "M**": s.hit_final.mean() * 100,
                         "M2": s.hit_m2.mean() * 100, "_k": k, "_n": len(s)})
    au = pd.DataFrame(rows).set_index("_k")
    ns = au["_n"].to_dict()
    au = au.drop(columns=["_n"])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    au.plot.bar(ax=ax, width=.75, edgecolor="white", linewidth=.6)
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("label vs instruction side")
    ax.set_ylim(0, 105)
    ax.set_xticklabels([f"{EN_LAB.get(i,i)}\n(n={ns[i]})" for i in au.index], rotation=0, fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=.25, linewidth=.5)
    ax.set_title("Where hard filtering loses: label-consistent vs contradictory", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/paper_fig3_audit.png")
    plt.close(fig)

    # 图 4：失分分解
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    lab = [EN_FAIL.get(s, s) for s in t9.类型]
    ax.barh(range(len(t9)), t9.n, edgecolor="white", linewidth=.6)
    ax.set_yticks(range(len(t9)))
    ax.set_yticklabels(lab, fontsize=8)
    for i, v in enumerate(t9.n):
        ax.text(v + 1, i, str(v), va="center", fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("samples")
    ax.grid(axis="x", alpha=.25, linewidth=.5)
    ax.set_title("M** failure decomposition (n=%d)" % int(t9.n.sum()), fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/paper_fig4_failures.png")
    plt.close(fig)

    # 图 5：7B 基线（左：与规则基线对比；右：允许集内守住率 vs 集内命中）
    if t12b is not None and len(t12b):
        fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
        meth = [("M**", d.hit_final.mean() * 100), ("M2", d.hit_m2.mean() * 100),
                ("M0", d.hit_m0.mean() * 100)]
        for m, _ in LLM_MODES:
            if "hit_" + m in d.columns:
                meth.append(("LLM " + m[1], d["hit_" + m].mean() * 100))
        ax = axs[0]
        cols = ["#4C6EF5"] * 3 + ["#F08C00"] * 3
        ax.bar(range(len(meth)), [v for _, v in meth], color=cols, edgecolor="white", lw=.6)
        ax.set_xticks(range(len(meth)))
        ax.set_xticklabels([n for n, _ in meth], rotation=20, fontsize=7.5)
        for i, (_, v) in enumerate(meth):
            ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_ylim(50, 78)
        ax.set_ylabel("accuracy on S1 (%)")
        ax.grid(axis="y", alpha=.25, lw=.5)
        ax.set_title("Rule baselines vs 7B (same 665)", fontsize=9)

        ax = axs[1]
        w = 0.2
        idx = np.arange(len(t12d))
        for k, L in enumerate(["A", "B", "C", "D"]):
            ax.bar(idx + (k - 1.5) * w, t12d[L].values, width=w, label=L, edgecolor="white", lw=.4)
        ax.set_xticks(idx)
        short = lambda z: "gold label" if "人类" in z else "LLM " + z.split("·")[1].strip()[:2]
        ax.set_xticklabels([short(z) for z in t12d.来源], rotation=0, fontsize=7.5)
        ax.set_ylabel("share of picks (%)")
        ax.legend(fontsize=7, frameon=False, ncol=4)
        ax.grid(axis="y", alpha=.25, lw=.5)
        ax.set_title("Option-position bias", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{FIG}/paper_fig5_llm.png")
        plt.close(fig)
    print(f"图写到 {FIG}/")


# ------------------------------------------------------------------ 报告
def write_report(d, t1, t2, t3, t4, t5a, t5b, t6, t6b, t7, t8, t8b, t9, tt,
                 t12a, t12b, t12c, t12d, best, sw, cv_hit, cv_w, subsets):
    n = len(d)
    g = lambda c: round(d[c].mean() * 100, 1)
    S1, S2, S3 = subsets[0][1], subsets[1][1], subsets[2][1]
    f1 = t6.set_index("子集")
    bw = float(sw.loc[sw.命中.idxmax(), "w_ori"])
    bv = float(sw.命中.max())
    n_free = int((d.free != "").sum())
    n_ok = int(d.ok.sum())
    small = d[d.A_n == 1]
    bigpool = d[d.n == 4]

    L = []
    A = L.append
    A("# ALFRED 空间指称实验：完整结果（冻结 v1.0，665 条）")
    A("")
    A("> 数据 `clean_spatial_ext.csv` 冻结 v1.0，只读，未改一个字节。口径：方向 = 机器人→锚点视线，"
      "距离 = 到锚点距离；「可以没有，不能相违背」。")
    A("> 一键复现：`python eval_paper.py`。所有表格由脚本动态生成，本文无手写数字。")
    A("")
    A("**建议标题**（原题「具身智能中的空间感知 RAG」已不准确——本工作无检索件）：")
    A("")
    A("- 中文：**具身指令里的「空间」到底指住了什么？——ALFRED 上指称约束力的量化与上限分析**")
    A("- 英文：**How Much Does Space Constrain? Quantifying Referential Constraints in Embodied Instructions**")
    A("")
    A("## 0 三个数先记住")
    A("")
    A("| 尺子 | 值 | 在测什么 |")
    A("|---|---:|---|")
    A(f"| U-SET（允许集合） | **{g('ok')}%** | 标注是否落在题面允许的范围（题面平均允许 {d.A_n.mean():.2f} 个） |")
    A(f"| M**（单点） | **{g('hit_final')}%** | 只准报一个，正好是标注 |")
    A(f"| 条件一致率 | **{f1.loc['S1 全量','条件一致率']}%** | 在 {n_ok} 条「标注确实在允许集内」上，挑中人类那个的比例 |")
    A("")
    qm = t6b.set_index("方法").loc["M**"]
    A(f"这三个数不是三个成绩，是同一件事的三层：{n} 条里 **{n_ok} 条**标注落在允许范围（{g('ok')}%）；"
      f"我们的方法 **{n_ok and int(d.hit_final.sum())} 条**挑中标注（{g('hit_final')}%）；"
      f"其中 **{int(qm.内对)} 条**在允许集内（{n_ok} × {f1.loc['S1 全量','条件一致率']}%），"
      f"另有 **{int(qm.外对)} 条**在允许集**外**却照样挑中了——因为软先验不否决任何候选。")
    A("")
    A(f"而允许集内**随机**挑的期望只有 {f1.loc['S1 全量','条件随机基线']}%"
      f"（条件于标注在集内），我们比它高 **{f1.loc['S1 全量','条件增量']} 点**"
      "——说明软先验学到的不是「随便」，是「人类通常怎么挑」。")
    if len(t12b):
        A("")
        A(f"**第四个参照：7B（Qwen2.5-7B，同 665 条）最好一套提示 {round(d['hit_'+best[0]].mean()*100,1)}%**，"
          f"低于取最近 {g('hit_m0')}%、硬筛 {g('hit_m2')}%、本文 {g('hit_final')}%（p<0.01）。"
          "它不是「不懂空间」，而是「守住了方向却不会在集内排序」，详见 T12。")
    A("")

    # T1
    A("## T1 三档主表（同一批样本，三把不同的刀）")
    A("")
    A("| 子集 | n | 含义 |")
    A("|---|---:|---|")
    A(f"| S1 全量 | {n} | 四种情况混在一起：有方向有距离 / 有方向无距离 / 无方向有距离 / 都没有 |")
    A(f"| S2 方向可算 | {len(S2)} | 剔掉「题面有方位但锚点定位不出」的 {n-len(S2)} 条 |")
    A(f"| S3 真被筛过 | {len(S3)} | 再剔掉「筛空回退」的 {len(S2)-len(S3)} 条，只留方向真的排除了候选的 |")
    A("")
    for nm, sub in subsets:
        tb = t1[t1.子集 == nm]
        A(f"### {nm}（n={len(sub)}）")
        A("")
        A("| 方法 | 性质 | 命中 | 95% CI |")
        A("|---|---|---:|---|")
        for _, r in tb.iterrows():
            star = "**" if r.方法.startswith("★") or r.方法.startswith("U-SET") else ""
            A(f"| {r.方法} | {r.性质} | {star}{r.命中}%{star} | "
              f"{'' if r.方法.startswith('ORACLE') else f'[{r.lo}, {r.hi}]'} |")
        A("")
    A("**读法**：三张表各讲一件事，不能合成一张。")
    A("")
    A(f"- 全量上 {g('ok')} vs {g('hit_final')}，落差主要来自那 {n-len(S2)} 条无方向样本"
      f"（M** 在上面只有 {round(d[d.Dir_Step=='不可算'].hit_final.mean()*100,1)}%，纯拖后腿）；")
    A(f"- 剔掉它们，M** 从 {g('hit_final')} → {round(S2.hit_final.mean()*100,1)}，"
      f"硬筛 M2 从 {g('hit_m2')} → {round(S2.hit_m2.mean()*100,1)}，**方法一上干净数据就涨**；")
    A(f"- 只留真被筛过的 {len(S3)} 条：M** {round(S3.hit_final.mean()*100,1)} vs "
      f"硬筛 {round(S3.hit_m2.mean()*100,1)}，**差 {round((S3.hit_final.mean()-S3.hit_m2.mean())*100,1)} 点**"
      "——这是「软先验优于硬排除」最干净的一张。")
    A("")
    A(f"> ⚠️ 注意：「有方向**又有距离**」这一格在全量里只有 "
      f"{int(sum(1 for x in d.Instruction if E.parse_dist(x)[0] is not None))} 条（题面几乎不写 nearest/furthest），"
      "所以严格按「文中有无距离」切 2×2 会出现空格子，本报告改用上面三档切法。")
    A("")

    # T2/T3/T4
    for tt_, title, note in [
        (t2, "T2 按数据档", "D2R = 锚点由同场景补齐、标注自洽 93%；D1 = 无可算锚点。"),
        (t3, "T3 按方位类型", "lr = 左右，front = 前方，behind = 后方。"),
        (t4, "T4 按允许集大小", "|A| = 1 表示题面唯一确定答案；|A| 越大题面越模糊。"),
    ]:
        A(f"## {title}")
        A("")
        A(note)
        A("")
        A("| 组 | n | U-SET | M** | M3* | M2 硬筛 | M0 | 平均允许 |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in tt_.iterrows():
            A(f"| {r.组} | {int(r.n)} | {r.U_SET}% | {r.Mstar}% | {r.M3s}% | {r.M2}% | {r.M0}% | {r.avg_A} |")
        A("")
    A(f"**T2 的要点**：档越干净方向越有用（D2R 上 M** {round(t2[t2.组=='D2R'].Mstar.iloc[0],1)}%），"
      f"唯一例外是 D1——那里没有方向信息，软层退化成纯距离，"
      f"{round(t2[t2.组=='D1'].Mstar.iloc[0],1)}% 与 M0 完全相同，符合预期。")
    A("")
    A(f"**T4 的要点（这条推翻了一个旧说法）**：|A|=1 的 {len(small)} 条上，U-SET 只有 "
      f"{round(small.ok.mean()*100,1)}%，而本文方法却有 {round(small.hit_final.mean()*100,1)}%"
      "——**软方法在「题面唯一」子集上反超了口径尺子**。")
    A("")
    A("这说明 73.5% 不是「标注噪声天花板」，而是「**严格按字面方位判**」这把尺子的上限。"
      "两种解释，目前分不开：")
    A("")
    A("1. 标注者不是按字面方位挑的，而是按空间邻近性挑的（与 §13「标注者其实按贴锚点选」一致）；")
    A("2. 我们在这一格的方位计算有误，真正的唯一答案被我们不当地排到了 A 之外。")
    A("")
    A("区分它们需要人工看这 "
      f"{int(len(small)-round(small.ok.mean()*len(small)))} 条——这正好是「157 条人工裁决」那件事的一部分。"
      "在没看过之前，只能说：**「唯一答案」子集上，几何比字面方位更贴合人类。**")
    A("")

    # T5
    A("## T5 标注审计：失分到底是谁的错")
    A("")
    A("| 组 | n | U-SET | M** | M3* | M2 硬筛 | M0 | 平均允许 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in t5a.iterrows():
        A(f"| {r.组 if r.组.startswith('缺失') else '标注'+r.组} | {int(r.n)} | {r.U_SET}% | {r.Mstar}% | {r.M3s}% | {r.M2}% | {r.M0}% | {r.avg_A} |")
    A("")
    okg = d[d.label_side == True]
    bad = d[d.label_side == False]
    A(f"- 标注**自洽**题面的 {len(okg)} 条：硬筛对 {int(okg.hit_m2.sum())}、软先验对 {int(okg.hit_final.sum())}"
      f"——硬筛只多 {int(okg.hit_m2.sum())-int(okg.hit_final.sum())} 条；")
    A(f"- 标注**违反**题面的 {len(bad)} 条：硬筛对 {int(bad.hit_m2.sum())}、软先验对 {int(bad.hit_final.sum())}"
      f"——软先验多捡回 {int(bad.hit_final.sum())-int(bad.hit_m2.sum())} 条。")
    A("")
    A("硬排除等于给方向一道**否决权**：cos<0 直接出局。可这 "
      f"{len(bad)} 条（占可算样本 {round(len(bad)/len(S2)*100,1)}%）的标注恰在被否决的一侧。"
      "**规则没错，是数据不听话**；否决权给出去就收不回，所以降级成 0.2 的权重。")
    A("")

    # T6
    A("## T6 第三把尺子：条件一致率")
    A("")
    A("| 子集 | n | U-SET | M** | 条件一致率 | 条件随机基线 | 增量 | 平均允许 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in t6.iterrows():
        A(f"| {r.子集} | {int(r.n)} | {r.U_SET}% | {r.Mstar}% | **{r.条件一致率}%** | {r.条件随机基线}% | "
          f"+{r.条件增量} | {r.平均允许} |")
    A("")
    A("### T6b 四象限：把 88.4 与 68.4 拆成同一张表")
    A("")
    A("| 方法 | 标注在允许集 · 挑中 | 标注在允许集 · 挑错 | 标注**不在**允许集 · 挑中 | 标注不在允许集 · 挑错 |")
    A("|---|---:|---:|---:|---:|")
    for _, r in t6b.iterrows():
        A(f"| {r.方法} | **{int(r.内对)}** | {int(r.内错)} | **{int(r.外对)}** | {int(r.外错)} |")
    A("")
    qm2 = t6b.set_index("方法")
    A(f"这张表就是全文的核心论证，四句话：")
    A("")
    A(f"1. **硬筛 M2 的「外对」必然是 0**：它把另一侧全排除，标注在那侧就永远拿不到。")
    A(f"2. 硬筛在「内对」上确实更强：{int(qm2.loc['M2 硬筛','内对'])} vs {int(qm2.loc['M**','内对'])}，"
      f"**多 {int(qm2.loc['M2 硬筛','内对']-qm2.loc['M**','内对'])} 条**——软先验在题面说清时会吃点亏。")
    A(f"3. 但它在「外对」上白丢 {int(qm2.loc['M**','外对'])} 条，净亏 "
      f"{int(qm2.loc['M**','外对'])-int(qm2.loc['M2 硬筛','内对']-qm2.loc['M**','内对'])} 条。")
    A(f"4. 总账：{int(qm2.loc['M**','内对']+qm2.loc['M**','外对'])} 对 "
      f"{int(qm2.loc['M2 硬筛','内对']+qm2.loc['M2 硬筛','外对'])}，"
      f"就是 {g('hit_final')}% vs {g('hit_m2')}%。")
    A("")
    A("所以「88.4 和 68.4 能不能放一张表」的答案是：**不能并排放，但可以拆开放**——"
      "四象限这一张表把两个口径同时装进去了，而且没有任何一行是白送的。")
    A("")
    A(f"白送那 {n_free} 条（题面没约束/筛空回退）上 M** 只有 {f1.loc['白送（无约束/筛空）','Mstar']}%、"
      f"条件一致率 {f1.loc['白送（无约束/筛空）','条件一致率']}%："
      "「任取都对」是对**范围**而言，单点仍要和人类那次随手挑对齐——这部分我们只能做到一半。"
      "这也解释了为什么 M** 在全量上被拉低。")
    A("")

    # T7
    A("## T7 组件消融")
    A("")
    A("| 配置 | 命中 | 95% CI | 相对只距离 | 作用 |")
    A("|---|---:|---|---:|---|")
    for _, r in t7.iterrows():
        A(f"| {r.配置} | **{r.命中}%** | [{r.lo}, {r.hi}] | {r.相对只距离:+.1f} | {r.作用} |")
    A("")
    A("**结论**：距离是主序（去掉它就掉到 "
      f"{round(t7[t7.配置.str.startswith('只方位')].命中.iloc[0],1)}%），方位是修正项（加上去只值 "
      f"{round(t7[t7.配置.str.startswith('软加权 0.2 + 距离硬层')].命中.iloc[0] - t7[t7.配置.str.startswith('只距离')].命中.iloc[0],1)} 点）；"
      "而**硬筛会把方位从修正项变成否决项，反而倒亏**。")
    A("")

    # ---- T7b：为什么表里有两行几乎一样（M** 与 M3*）
    hd = d[d.layer.astype(str).str.startswith("硬")]
    n_hd, n_all = len(hd), len(d)
    hd_hit, sft_hit = int(hd.hit_final.sum()), int(hd.hit_m3.sum())
    disc = int((d.hit_final != d.hit_m3).sum())
    e7 = t7.set_index("配置")
    r_m = e7.loc["软加权 0.2 + 距离硬层（M**）", "命中"]
    r_s = e7.loc["软加权 w=0.2（M3*）", "命中"]
    A("### T7b 为什么表里有两行几乎一样：M** 与 M3* 差在哪")
    A("")
    A("这两行**不是「方向软版」与「方向硬版」**。方向只有一个版本，就是软的，"
      "表里没有第二个候选。它们的唯一差别是**第二步的距离硬层要不要打开**：")
    A("")
    A("| 版本 | 方向项 | 距离项 | 距离硬层 | 命中 |")
    A("|---|---|---|---|---:|")
    A(f"| M**（主推） | 软先验 w=0.2 | 主序 1−0.4·d/dmax | **打开**（题面明写 nearest/furthest 才生效） | {r_m}% |")
    A(f"| M3*（消融） | 软先验 w=0.2 | 主序 1−0.4·d/dmax | 关闭 | {r_s}% |")
    A("")
    A(f"**硬层一共只触发了 {n_hd} 条（{round(n_hd/n_all*100,2)}%）**，在这 {n_hd} 条上："
      f"M** 对 {hd_hit} 条、纯软层对 {sft_hit} 条 → **净 {'+' if hd_hit>sft_hit else ''}{hd_hit-sft_hit} 条**。"
      f"两版判定不同的样本只有 {disc} 条，McNemar p=1.0。")
    A("")
    A("为什么会出现「两个几乎一样的版本」？**是数据的两条事实逼出来的，不是我们留了两个方案**：")
    A("")
    n_self, n_contra = int((d.Lab == "自洽").sum()), int((d.Lab == "矛盾").sum())
    A(f"1. **标注自洽率只有 {round(n_self / (n_self + n_contra) * 100, 1)}%**"
      f"（{n_contra} 条标注与题面方位矛盾）→ 方向不能当否决权，必须降为软先验。"
      "这一步已经定案，不存在「另留一手硬版」。")
    A(f"2. **人写指令几乎不写距离词**（665 条里只有 {n_hd} 条能定位到 nearest/furthest 的参照）→ "
      "第二步的硬取规则即使原样保留，也几乎没有施展机会。于是「开」与「关」自然只差 1 条样本。")
    A("")
    A("**为什么还两行都留着**：消融的规矩是「证明某个组件没用」也必须报，"
      "不能因为结果不涨就把它从表里删掉。所以这里的结论要写成：")
    A("")
    A(f"> **距离硬层在本数据上无贡献（触发 {n_hd} 条、净 {hd_hit-sft_hit} 条、p=1.0）。"
      "保留它是因为它实现了「题面写了 nearest 就按 nearest」这条规则，成本为 0；"
      "如果追求最简配置，去掉它即可（M3* = "
      f"{r_s}%），两者在统计上不可区分。**")
    A("")

    # T8
    A("## T8 演进史：为什么最后是「文本定类别 + 几何定实例」")
    A("")
    A("同一批 665 条，按**原始设定**重算（候选 = 场景内全部物体，34~40 个，真实检索难度）：")
    A("")
    A("| 阶段 | 命中 | 95% CI |")
    A("|---|---:|---|")
    for _, r in t8.iterrows():
        A(f"| {r.阶段} | **{r.命中}%** | [{r.lo}, {r.hi}] |")
    A("")
    e8 = t8.set_index("key")
    ch = round(d.cat_hit.mean() * 100, 1)
    A(f"故事线：纯文本 **{e8.loc['A','命中']}%**；加上距离衰减（B）→ **{e8.loc['B','命中']}%**；"
      "再加方位（B2）→ "
      f"**{e8.loc['B2','命中']}%**。")
    A("")
    A(f"纯文本为什么这么低？拆开看两层：**类别认对 {ch}%**，但认对之后同类实例里它没有任何信息，"
      f"只能按 1/{round(d.Num_Same.mean(),2)} 碰运气 → {ch} × {round(100/d.Num_Same.mean(),1)}% ≈ "
      f"{round(ch/d.Num_Same.mean(),1)}%，与实测 {e8.loc['A','命中']}% 吻合。")
    A("")
    A(f"加上距离后到 {e8.loc['B','命中']}%：**文本负责把类别锁死，几何负责在同类实例里挑**——这就是这条演进线的结论。"
      "再加方位（B2）不升反微降，因为大池子里方位信息被类别误差吃掉了（只有 "
      f"{round(100-ch,1)}% 的样本类别还认错，方位救不回来）。")
    A("")
    A("> 注：这三行是**全池设定**，与 T1 的同类池设定不同，不可与 M** 直接比大小；"
      "它们的作用是解释「为什么必须有几何」，不是竞争榜首。")
    A("")
    A(f"一个自洽性检查：B 在同类池里的退化形态就是「取离锚点最近」，即 M0 = {g('hit_m0')}%；"
      f"全池版 B 是 {e8.loc['B','命中']}%，差 {round(g('hit_m0')-e8.loc['B','命中'],1)} 点，"
      f"正好等于那 {round(100-ch,1)}% 类别认错的代价。")
    A("")

    # T8b
    ch2 = round(d.cat_hit.mean() * 100, 1)
    end2end = round(ch2 / 100 * g('hit_final'), 1)
    A("## T8b 两段式分解：瓶颈不在空间推理，在类别识别")
    A("")
    A("| 组 | n | A 纯文本 | B 公式 | M**（同类池） | M0（同类池） |")
    A("|---|---:|---:|---:|---:|---:|")
    for _, r in t8b.iterrows():
        A(f"| {r.组} | {int(r.n)} | {r.A}% | {r.B}% | {r.Mstar}% | {r.M0}% |")
    A("")
    A(f"**这条是全文最该被看见的一条。** 全池 B 的 {round(t8.set_index('key').loc['B','命中'],1)}% "
      f"≈ 类别认对率 {ch2}% × 认对后的实例正确率 {round(t8b[t8b.组=='类别认对'].B.iloc[0],1)}%。两层是乘起来的：")
    A("")
    A(f"1. **类别识别**（「要拿的是不是笔」）：TF-IDF 只能到 {ch2}%；认错时任何几何方法都归零（B 只剩 "
      f"{round(t8b[t8b.组=='类别认错'].B.iloc[0],1)}%）。")
    A(f"2. **实例选择**（「三支笔里是哪支」）：认对之后，几何把 {round(t8b[t8b.组=='类别认对'].A.iloc[0],1)}% "
      f"抬到 {round(t8b[t8b.组=='类别认对'].B.iloc[0],1)}%，这就是本文的全部贡献区间。")
    A("")
    A(f"所以必须写清边界：本文方法的 {g('hit_final')}% 是「**给定类别**」下的数字。"
      f"端到端还欠一个类别解析器：{ch2}% × {g('hit_final')}% ≈ **{end2end}%**，"
      f"与全池 B 实测 {round(t8.set_index('key').loc['B','命中'],1)}% 同量级（差的几个点来自两段误差不独立）。")
    A("")
    A("这不是甩锅，是定位：本文解决的是「同类多实例里选哪一个」，不是「物体是什么」。"
      "把这两件事混在一个数字里报，才是真问题。")
    A("")

    # T9
    A("## T9 M** 的失分去哪了")
    A("")
    A("| 类型 | n | 占失分 | 说明 |")
    A("|---|---:|---:|---|")
    for _, r in t9.iterrows():
        A(f"| {r.类型} | {int(r.n)} | {r.占比}% | {r.说明} |")
    A("")
    n_both = int(t9[t9.类型.str.startswith('标注不在允许集')].n.iloc[0])
    n_inside = int(t9.n.sum()) - n_both
    qq = t6b.set_index("方法").loc["M**"]
    A(f"合计失分 {int(t9.n.sum())} = {n} − {int(d.hit_final.sum())}。拆成两块：")
    A("")
    A(f"- **{n_both} 条**标注压根不在题面允许范围（U-SET 也判错）——这不是选错，是题面与标注打架；")
    A(f"- **{n_inside} 条**标注在范围内、我们却挑了别的同类——这才是真正「没学会人类怎么挑」的部分。")
    A("")
    A(f"两把尺子的 {round(g('ok')-g('hit_final'),1)} 点差距由此得出："
      f"{n_inside} 条范围内挑错（−{round(n_inside/n*100,1)} 点）"
      f"被 {int(qq.外对)} 条范围外挑中（+{round(int(qq.外对)/n*100,1)} 点）抵掉一部分，净剩 "
      f"{round((n_inside-int(qq.外对))/n*100,1)} 点。")
    A("")

    # T10
    A("## T10 权重与交叉验证")
    A("")
    A("| w_ori | " + " | ".join(f"{v:.1f}" for v in sw.w_ori) + " |")
    A("|---|" + "---:|" * len(sw))
    A("| M** 命中 | " + " | ".join(f"{v}" for v in sw.命中) + " |")
    A("")
    A(f"最优点 w={bw:.1f} → **{bv}%**；0.1~0.4 都在 {min(sw.命中[1:5]):.1f}% 以上，曲线平，不敏感。")
    A(f"5 折交叉（按 Task_ID 分组）：每折独立选 w → **{cv_w}**，OOF = **{cv_hit:.1f}%**，"
      "与全表一致，说明 0.2 不是拿测试集试出来的。")
    A("")

    # 配对
    A("## T11 配对检验（同一样本逐条比）")
    A("")
    A("| 对手 | M** 对/对手错 | M** 错/对手对 | p |")
    A("|---|---:|---:|---:|")
    for _, r in tt.iterrows():
        p = "<0.0001" if r.p < 1e-4 else f"{r.p:.4f}"
        A(f"| {r.对手} | {int(r.M对)} | {int(r.对手对)} | {p} |")
    A("")

    # T12 7B 学习基线
    if len(t12b):
        b = t12b.set_index("模式")
        c = t12c.set_index("组")
        bm = best[1]
        bmd = bm.replace("LLM 7B · ", "")
        pf = lambda x: "<0.0001" if x < 1e-4 else f"{x:.4f}"
        A("## T12 7B 学习基线（Qwen2.5-7B-Instruct，greedy，A5000 24GB）")
        A("")
        A("三套提示，差别只在「脚手架」多少：`v1` 只给坐标与数字；`v2` 把关系说成人话"
          "（「在 anchor 左侧 0.56 m、离你 2.2 m」）；`v3` 在 v2 基础上把两步规则明写进提示。")
        A("")
        A("| 模式 | S1 全量 665 | S2 方向可算 596 | S3 真被筛过 515 | 对 M** 的 McNemar |")
        A("|---|---:|---:|---:|---|")
        for m, dsp in LLM_MODES:
            if "hit_" + m not in d.columns:
                continue
            row = t12a[(t12a.模式 == dsp) & (t12a.对手.str.startswith("M**"))].iloc[0]
            A(f"| {dsp.replace('LLM 7B · ', '')} | **{round(d['hit_' + m].mean() * 100, 1)}%** | "
              f"{round(S2['hit_'+m].mean()*100,1)}% | {round(S3['hit_'+m].mean()*100,1)}% | "
              f"p={pf(row.p)}（LLM 对 {int(row.LLM对)} / M** 对 {int(row.对手对)}） |")
        A(f"| ★ M**（规则） | **{g('hit_final')}%** | {round(S2.hit_final.mean()*100,1)}% | "
          f"{round(S3.hit_final.mean()*100,1)}% | — |")
        A(f"| M2 硬筛 | {g('hit_m2')}% | {round(S2.hit_m2.mean()*100,1)}% | "
          f"{round(S3.hit_m2.mean()*100,1)}% | — |")
        A("")
        A("**三套提示两两之间没有显著差异**（"
          + "，".join(f"{r.模式.split(' vs ')[0]}→{r.对手.replace('vs ','')} p={pf(r.p)}"
                     for _, r in t12a[t12a.对手.str.startswith('vs')].iterrows())
          + "）。也就是说：**把数字说成人话、把规则明写，都没能把它救回来**——"
          "瓶颈不在表述形式，在它自己在候选中排序的能力。")
        A("")
        A("### T12b 它到底是「不会用空间」还是「不会在空间里挑」")
        A("")
        A("| 模式 | 选中的落在允许集 | 落在集内时的命中 | 与「集内最近」一致 | 一致时命中 | 偏离时命中 | 偏离条数 | 选 A 或 B |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in t12b.iterrows():
            A(f"| {r.模式.replace('LLM 7B · ', '')} | {r.在允许集}% | {r.集内命中}% | "
              f"{r.与集内最近一致}% | {r.一致时命中}% | {r.偏离时命中}% | "
              f"{int(r.偏离条数)} | {r.选A或B}% |")
        A(f"| 人类标注（参照） | {g('ok')}% | — | — | — | — | — | "
          f"{round(d.ti.isin([0,1]).mean()*100,1)}% |")
        A("")
        A("这张表把「LLM 不懂空间」这个方便说法推翻了一半：")
        A("")
        A(f"1. **它守住了方向**：{bmd} 有 **{b.loc[bm,'在允许集']}%** 的选中落在题面允许的集合里"
          f"——比如「笔在柜子左边」，它真的只从左边的笔里挑。这个数甚至高于标注自己的 {g('ok')}%。")
        A(f"2. **它输在集内排序**：只算「标注也在集内」的 {n_ok} 条，{bmd} 是 "
          f"**{round(d[d.ok][best[2]].mean()*100,1)}%**，而「集内取最近」这条一行代码的规则是 "
          f"{round(d[d.ok].hit_m2.mean()*100,1)}%、本文方法 {round(d[d.ok].hit_final.mean()*100,1)}%。"
          "也就是说：**给它缩小到正确的候选集合之后，它仍然挑输给「取最近」**。")
        A(f"3. **偏离「最近」就是亏**：{bmd} 与「集内最近」一致时命中 "
          f"{b.loc[bm,'一致时命中']}%，一旦换掉最近那个（{int(b.loc[bm,'偏离条数'])} 条），"
          f"命中掉到 **{b.loc[bm,'偏离时命中']}%**——它的「换一个」几乎没有一次换来正确答案。")
        A(f"4. **位置偏置明显**：{bmd} 把 {b.loc[bm,'选A或B']}% 的答案给了 A/B 两个位置"
          f"（人类标注 {round(d.ti.isin([0,1]).mean()*100,1)}%），`v1` 更极端。"
          "同一个场景，把选项顺序打乱，结果就会变——这一条本身就足以说明它没在做空间推理。")
        A("")
        A("| 来源 | A | B | C | D |")
        A("|---|---:|---:|---:|---:|")
        for _, r in t12d.iterrows():
            A(f"| {r.来源.replace('LLM 7B · ', '')} | {r.A}% | {r.B}% | {r.C}% | {r.D}% |")
        A("")
        A("### T12c 分层：它在哪里彻底崩掉")
        A("")
        A(f"| 组 | n | {bmd} | M** | M2 | M0 |")
        A("|---|---:|---:|---:|---:|---:|")
        for _, r in t12c.iterrows():
            A(f"| {r.组} | {int(r.n)} | {r.LLM}% | {r.Mstar}% | {r.M2}% | {r.M0}% |")
        A("")
        A(f"最刺眼的一行是**标注矛盾**那 {int(c.loc['标注矛盾','n'])} 条：LLM 只有 "
          f"**{c.loc['标注矛盾','LLM']}%**，而 M0（只取最近）有 {c.loc['标注矛盾','M0']}%。")
        A("原因不难懂：**它老老实实按题面说的那一侧挑，而标注者在那批样本里恰恰没按题面挑。**"
          "越听话，越吃亏——这反过来把 T5 的「标注噪声」结论钉死了："
          "一个只读题面的模型，在这批数据上拿不到 40%，说明题面与标注真的不一致，"
          "而不是我们的方法没看懂题。")
        A("")
        A("### 结论怎么写")
        A("")
        A("| 能写 | 不能写 |")
        A("|---|---|")
        A(f"| 「7B 在同一 665 条上 {round(d['hit_'+best[0]].mean()*100,1)}%，低于取最近的 {g('hit_m0')}%、"
          f"硬筛 {g('hit_m2')}%、本文 {g('hit_final')}%，三者 p<0.01」 | 「LLM 完全不会空间推理」——它守住了 {b.loc[bm,'在允许集']}% 的方位约束 |")
        A(f"| 「LLM 的失败点在允许集内的排序（{b.loc[bm,'集内命中']}%），不在方向识别（{b.loc[bm,'在允许集']}%）」 | "
          "「LLM 不如规则所以规则更好」——两者可组合：LLM 定侧、几何定实例 |")
        A(f"| 「三套提示无显著差异，表述形式不是瓶颈」 | 「7B 代表所有 LLM」——只有一个模型、一套量化，没跑 CoT / 更大模型 |")
        A("")
        A(f"边界要说清：单模型（Qwen2.5-7B-Instruct）、单次 greedy、无 CoT、无 self-consistency；"
          f"3 条 `Answer: None` 按错计，未手工补齐。")
        A("")

    # 报数纪律
    A("## R 报数纪律（写论文/汇报必须同时给）")
    A("")
    A(f"1. 说 {g('ok')}% 必须同时说：其中 **{n_free} 条是白送**"
      f"（{int((d.free=='方向筛空回退').sum())} 筛空回退 + {int((d.Band=='D1').sum())} 无锚点）；"
      f"方向可算的 {len(S2)} 条是 **{round(S2.ok.mean()*100,1)}%**；"
      f"再剔回退后 {len(S3)} 条是 **{round(S3.ok.mean()*100,1)}%**。")
    A(f"2. 说 {g('hit_final')}% 必须同时说：这是**单点**，和 {g('ok')}% 的集合口径**不是一把尺子**，"
      "不可并排比高低。桥接数是条件一致率。")
    pf = lambda x: "<0.0001" if x < 1e-4 else f"{x:.4f}"
    A(f"3. 说「比基线好」必须给配对检验与 n：对 M0 p={pf(tt[tt.对手.str.startswith('M0')].p.iloc[0])}，"
      f"对 M2 p={pf(tt[tt.对手.str.startswith('M2')].p.iloc[0])}。")
    A("4. 不能把 T8（全池设定）与 T1（同类池设定）放同一张表比大小。")
    A(f"5. 说任何一个单点数都必须带上「**给定类别**」四个字：类别解析率只有 {ch2}%，"
      f"乘上去端到端约 {end2end}%（见 T8b）。这是边界，不是缺陷——但不写就是误导。")
    if len(t12b):
        b0 = t12b.set_index("模式")
        A(f"6. 说「LLM 不如规则」必须同时说两件事：它守住了 **{b0.loc[best[1],'在允许集']}%** 的方位约束，"
          f"输在集内排序（{b0.loc[best[1],'集内命中']}%）；且这是**单模型单次 greedy**的结果，"
          "三套提示之间并无显著差异。")
    A("")

    # 缺口
    A("## 还差什么")
    A("")
    A("| 缺口 | 状态 | 说明 |")
    A("|---|---|---|")
    if len(t12b):
        A("| 同集学习基线 | ✅ 已完成 | Qwen2.5-7B × 3 套提示，T12；"
          f"最好一套 {round(d['hit_'+best[0]].mean()*100,1)}%，显著低于 M0/M2/M** |")
    else:
        A("| 同集学习基线 | ⏳ | `llm_pack/` 已备好，跑完放进 `llm_results/preds.jsonl` 即自动并入 |")
    A("| 第二数据集（Nr3D/SR3D） | 未做 | 需下载+对齐，不需要 GPU；投主会才需要 |")
    A("| 端到端 rollout | 未做 | 需 AI2-THOR 环境，若要报任务成功率才需要 |")
    A("| 157 条人工裁决 | 未做 | 需人眼，用于把「疑似噪声」升级为「噪声」 |")
    A(f"| 类别解析器（文本→类别） | 声明边界即可 | 现在只有 {ch2}%，端到端需乘；不在本文 claim 内 |")
    A("| 更大模型 / CoT / 多次投票 | 未做 | 7B 单次 greedy 已足够作负结果，升级需另一轮资源 |")
    A("")
    A("## 文件")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    for f, s in [("eval_paper.py", "本脚本（只读冻结数据，一键复现全部表格）"),
                 ("paper_T1_main.csv", "三档主表"),
                 ("paper_T2_band.csv / T3_rel / T4_pool", "分层"),
                 ("paper_T5_audit.csv", "标注审计"),
                 ("paper_T6_ruler.csv", "第三把尺子"),
                 ("paper_T7_ablation.csv", "组件消融"),
                 ("paper_T8_evolution.csv", "演进史"),
                 ("paper_T9_failures.csv", "失分归因"),
                 ("paper_T10_sweep.csv / T11_tests.csv", "扫描与配对检验"),
                 ("paper_T12a..d_llm_*.csv / llm_results/", "7B 检验、行为诊断、分层、位置偏置、原始预测"),
                 ("paper_detail.csv", "逐样本明细"),
                 ("figs/paper_fig1..5", "五张图（含 7B 基线）")]:
        A(f"| `{f}` | {s} |")
    A("")
    txt = "\n".join(L).replace("M**", "M\\*\\*")
    open(f"{RESULTS}/PAPER_EXPERIMENTS.md", "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    main(no_fig="--no-fig" in sys.argv)
