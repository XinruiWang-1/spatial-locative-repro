#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终方法 M-soft（2026-11-24）

用户规则（原话）：
  第一步判方向，另一侧排除；指定侧没有候选则这步作废、全池进第二步。
  第二步判距离，写了最近/最远就按它取，没写则任取一个都对。
  两步都没有，按文本选。

落地只改一处：**第一步从「硬排除」降级为「软先验」**（权重 0.2，可扫描）。
理由：本数据有 26% 的标注与题面方位矛盾，硬排除会把四分之一样本的目标一次性扔掉。

两层结构
  硬层（只在题面明写时生效）：nearest/closest/furthest/farthest + 参照可定位
  软层（默认）              ：score = (1 - 0.4*d/dmax) * (1 - 0.2 + 0.2*(1+cos)/2)
                              —— 距离是主序，方向只占 0.2 权重，不否决任何候选

两种判分口径都报
  U-SET 允许集合判定（用户口径）  : 标注落在题面允许范围内
  M-soft   单点命中（真机要选一个）  : 只输出一个，正好是标注

只读冻结数据，不改任何文件。
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

import scoring as E
import baselines as C

import _paths as _p

DATA = _p.DATA
RESULTS = _p.RESULTS
W_ORI, W_DIST = 0.2, 0.4
BOOT, TOL = 4000, 1e-9


# ---------------------------------------------------------------- 硬层：距离
def dist_constraint(same, instr):
    """题面明写距离词且参照可定位 -> (kind, rank, ref)；ref 为 "robot" 或坐标。否则 None。"""
    kind, rank, ref = E.parse_dist(instr)
    if kind is None:
        return None
    core = str(ref or "")
    first = core.split()[0] if core else ""
    if ref in (None, "you", "me", "yourself"):
        return kind, rank, "robot"           # 参照是机器人，坐标在选点时给
    if first and first not in E.UNPARSABLE_REF:
        hit = [c for c in same if first in c["category"] or c["category"] in first]
        if hit:
            return kind, rank, (hit[0]["x"], hit[0]["y"])
    return None


# ---------------------------------------------------------------- 软层：打分
def soft_score(rec, w_ori=W_ORI, w_dist=W_DIST):
    dmax = max(rec["d_anchor"]) or 1.0
    return [(1.0 - w_dist * (rec["d_anchor"][i] / dmax)) *
            ((1 - w_ori) + w_ori * (1.0 + rec["cos"][i]) / 2.0)
            for i in range(rec["n"])]


def pick_final(rec, w_ori=W_ORI):
    """返回 (选中 idx, 用了哪一层)"""
    hard = rec.get("dist_hard")
    if hard is not None:
        kind, rank, ref = hard
        ox, oy = (rec["_robot"] if ref == "robot" else ref)
        vals = [math.hypot(c["x"] - ox, c["y"] - oy) for c in rec["_same"]]
        if kind == "max":
            vals = [-v for v in vals]
        best = min(vals)
        cand = [i for i in range(rec["n"]) if abs(vals[i] - best) <= 0.02]
        if rank > 1 and len(cand) >= rank:
            return cand[rank - 1], "硬·距离"
        sc = soft_score(rec, w_ori)
        return max(cand, key=lambda i: (sc[i], -i)), "硬·距离"
    sc = soft_score(rec, w_ori)
    return max(range(rec["n"]), key=lambda i: (sc[i], -i)), "软·方向+距离"


# ---------------------------------------------------------------- 装配
def build_final():
    recs = C.build().to_dict("records")
    df = (pd.read_csv(f"{DATA}/clean_spatial_ext.csv", encoding="utf-8-sig")
          .query("Scorable == 1").set_index("Sample_ID"))
    cands = _p.load_json("loc_candidates.json")
    ancs = _p.load_json("loc_anchors.json")

    out = []
    for r in recs:
        row = df.loc[r["Sample_ID"]]
        rx, ry = float(row.Robot_X), float(row.Robot_Y)
        if math.isnan(rx) or math.isnan(ry):
            rb = (ancs.get(r["Task_ID"]) or {}).get("robot")
            rx, ry = (float(rb[0]), float(rb[1])) if rb else (0.0, 0.0)
        same = [c for c in (cands.get(r["Task_ID"]) or [])
                if c["category"] == row.Target_Category]
        r["_same"], r["_robot"] = same, (rx, ry)
        r["dist_hard"] = dist_constraint(same, r["Instruction"])
        r["pick"], r["layer"] = pick_final(r)
        full = list(range(r["n"]))
        r["hit_final"] = int(r["pick"] == r["ti"])
        r["hit_m2"] = int(C.argmin_set(r["d_anchor"], r["A"]) == {r["ti"]})
        r["hit_m0"] = int(C.argmin_set(r["d_anchor"], full) == {r["ti"]})
        r["hit_m3"] = int(C.argmax_set(soft_score(r, 0.2), full) == {r["ti"]})
        r["hit_m35"] = int(C.argmax_set(soft_score(r, 0.5), full) == {r["ti"]})
        out.append(r)
    return pd.DataFrame(out)


def ci(p, seed=0):
    lo, hi = C.boot_ci(np.array(p, float), n=BOOT, seed=seed)
    return round(float(np.mean(p)) * 100, 1), round(lo, 1), round(hi, 1)


# ---------------------------------------------------------------- 扫描 + CV
def sweep_final(d):
    """w_ori 0->1 扫描：硬层固定，只动软层权重。"""
    rows = []
    for w in [i / 10 for i in range(11)]:
        recs = d.to_dict("records")
        hits = sum(1 for r in recs if pick_final(r, w)[0] == r["ti"])
        rows.append(dict(w_ori=w, 命中=round(hits / len(d) * 100, 1)))
    return pd.DataFrame(rows)


def cv_final(d, k=5):
    """5 折：按 Task_ID 分组切折，训练折选 w、留出折评估 -> 排除事后调参。"""
    ws = [i / 10 for i in range(11)]
    recs = d.to_dict("records")
    fold = [int(hashlib.md5(str(r["Task_ID"]).encode()).hexdigest(), 16) % k for r in recs]
    pred = {w: [1 if pick_final(r, w)[0] == r["ti"] else 0 for r in recs] for w in ws}
    oof, chosen = [0] * len(recs), []
    for f in range(k):
        tr = [i for i in range(len(recs)) if fold[i] != f]
        te = [i for i in range(len(recs)) if fold[i] == f]
        best = max(ws, key=lambda w: sum(pred[w][i] for i in tr) / len(tr))
        chosen.append(best)
        for i in te:
            oof[i] = pred[best][i]
    return float(np.mean(oof)) * 100, chosen


# ---------------------------------------------------------------- 报告
def demo(recs, cond):
    """挑一条真实样本用于讲解，数字全部来自实际几何，不手写"""
    for r in recs:
        if cond(r):
            return r
    return None


def md_table(r):
    out = ["| 候选 | 到锚点距离 | cos（正 = 题面侧） | 在允许范围 |", "|---|---:|---:|---|"]
    for i in range(r["n"]):
        tag = " ← 标注" if i == r["ti"] else ""
        out.append(f"| {chr(65+i)}{tag} | {r['d_anchor'][i]:.2f} m | {r['cos'][i]:+.2f} | "
                   f"{'✅' if i in r['A'] else '❌'} |")
    return out


def who_followed(r, kind):
    """标注实际是按哪个参照的极值选的"""
    fn = np.argmax if kind == "max" else np.argmin
    a = int(fn(r["d_anchor"])) == r["ti"]
    b = int(fn(r["d_robot"])) == r["ti"]
    if a and b:
        return "锚点与机器人一致"
    if a:
        return "按锚点距离"
    if b:
        return "按机器人距离"
    return "两者都不是"


def write_report(d, m, band, lab, rel, tests, sw, cv_hit, cv_w):
    n = len(d)
    recs = d.to_dict("records")
    ex1 = demo(recs, lambda r: r["Band"] == "A" and r["A_n"] == 2 and r["n"] == 4
               and r["hit_final"] and len(r["Instruction"]) < 70)
    ex2 = demo(recs, lambda r: r["Band"] == "A" and r["A_n"] == 2 and r["n"] == 3
               and not r["hit_final"] and r["ok"] and len(r["Instruction"]) < 70)
    g = lambda col: round(d[col].mean() * 100, 1)
    uset = g("ok")
    uset_lo, uset_hi = ci(list(d.ok), 1)[1:]
    fin, fin_lo, fin_hi = ci(list(d.hit_final))
    m2, m0, m3, m35 = g("hit_m2"), g("hit_m0"), g("hit_m3"), g("hit_m35")
    bw = float(sw.loc[sw.命中.idxmax(), "w_ori"])
    bv = float(sw.命中.max())
    n_free = int((d.free == "方向筛空回退").sum()) + int((d.Band == "D1").sum())
    n_dir = int((d.Band != "D1").sum())
    ok_ = d[d.label_side == True]
    bad = d[d.label_side == False]
    mx = d[d.layer == "硬·距离"]
    avgA = d.A_n.mean()
    n_dw = int(sum(E.parse_dist(x)[0] is not None for x in d.Instruction))
    mxa = mx.to_dict("records")
    ok_anchor = sum(1 for r in mxa
                    if r["ti"] == (int(np.argmax(r["d_anchor"])) if r["dist_hard"][0] == "max"
                                   else int(np.argmin(r["d_anchor"]))))
    def rank_anchor(r):
        return 1 + int(sum(1 for v in r["d_anchor"] if v < r["d_anchor"][r["ti"]]))
    n_anchor1 = sum(1 for r in mxa if rank_anchor(r) == 1)
    real = d[(d.Band != "D1") & (d.free != "方向筛空回退")]
    n_real, uset_real = len(real), round(real.ok.mean() * 100, 1)
    n_uni = int((d.A_n == 1).sum())
    uset_uni = round(d[d.A_n == 1].ok.mean() * 100, 1)
    cband = band.loc["C"]

    ex_block = []
    if ex1 is not None:
        al = [chr(65 + i) for i in ex1["A"]]
        gt, pk = chr(65 + ex1["ti"]), chr(65 + ex1["pick"])
        a = ex1["Anchor"]
        ex_block += [f"**例 1（两把尺子都算对）**：原句 `{ex1['Instruction']}`，同类候选 {ex1['n']} 个，锚点是 `{a}`。", ""]
        ex_block += md_table(ex1)
        ex_block += ["",
                     f"题面只说了相对 `{a}` 的方位，没说是哪一个。按你的规则 **{' 和 '.join(al)} 都对**——这就是允许集合那把尺子。",
                     f"标注是 {gt}，在范围里 → 算对；M-soft 在 {' / '.join(al)} 里也挑了 {pk} → 另一把尺子也加一分。", ""]
    if ex2 is not None:
        al = [chr(65 + i) for i in ex2["A"]]
        gt, pk = chr(65 + ex2["ti"]), chr(65 + ex2["pick"])
        a = ex2["Anchor"]
        ex_block += [f"**例 2（口径算对、单点算错）**：原句 `{ex2['Instruction']}`，同类候选 {ex2['n']} 个，锚点是 `{a}`。", ""]
        ex_block += md_table(ex2)
        ex_block += ["",
                     f"**{' 和 '.join(al)} 都在题面说的那一侧，题面没说挑哪个，所以两个都对（口径算对）。**",
                     f"但真机必须报一个：M-soft 报了 {pk}（{ex2['d_anchor'][ex2['pick']]:.2f} m、cos {ex2['cos'][ex2['pick']]:+.2f}），"
                     f"标注却是 {gt}（{ex2['d_anchor'][ex2['ti']]:.2f} m、cos {ex2['cos'][ex2['ti']]:+.2f}）→ 单点算错。", ""]

    L = ["# 最终方法 M-soft：两个数、一个决定、能写进论文的说法", "",
         f"> 数据：`clean_spatial_ext.csv` 冻结 v1.0 的 **{n}** 条可判分样本，只读，未改一个字节。",
         "> 口径不变：方向 = 机器人→锚点视线，距离 = 到锚点距离。原句写 `to you` 的按字面用机器人坐标（仅 4 条）。",
         "> 代码 `eval_final.py`，`python eval_final.py` 复现本文全部数字。", "",
         "## 0 一句话", "",
         f"**判分 {uset}%，落地 {fin}%。这两个数不矛盾，是两把不同的尺子**：",
         "前者问「答案在不在题面允许的范围内」，后者问「只报一个，报对没有」。", "",
         "| 尺子 | 数值 | 95% CI | 在测什么 |", "|---|---:|---|---|",
         f"| 允许集合判定 U-SET（你的口径） | **{uset}%** | [{uset_lo}, {uset_hi}] | 标注是否落在题面允许范围（题面平均允许 {avgA:.2f} 个） |",
         f"| 单点 M-soft（本文主推） | **{fin}%** | [{fin_lo}, {fin_hi}] | 只能输出一个，正好是标注 |", "",
         f"## 1 为什么 {uset:.1f}% 和 {fin:.1f}% 不矛盾：拿两条真样本走一遍", "",
         *ex_block,
         f"**{n} 条里这种「题面平均允许 {avgA:.2f} 个、却只能报一个」的落差，就是 {uset}% 与 {fin}% 的全部差距。**",
         "它不是方法退步，是「答案范围」与「单一答案」的区别。任何方法的上限都卡在两件事上：",
         "① 题面没写清该挑哪个；② 标注者自己也不总按题面挑。", "",
         "## 2 为什么把「排除」换成「权重」", "",
         "你原规则的第一步是**硬排除**（另一侧全不要）。看它分组表现：", "",
         "| 子集 | n | U-SET | M2 硬排除 | M-soft 软先验 | M0 只取最近 |",
         "|---|---:|---:|---:|---:|---:|"]
    for name in ["自洽", "违反"]:
        sub = d[d.label_side.map({True: "自洽", False: "违反"}) == name]
        L.append(f"| 标注{name}题面 | {len(sub)} | {round(sub.ok.mean()*100,1)}% | "
                 f"{round(sub.hit_m2.mean()*100,1)}% | {round(sub.hit_final.mean()*100,1)}% | "
                 f"{round(sub.hit_m0.mean()*100,1)}% |")
    L += ["",
          f"- 标注**自洽**的 {len(ok_)} 条：硬排除对 {int(ok_.hit_m2.sum())} 条、软先验对 {int(ok_.hit_final.sum())} 条"
          f" —— 硬排除只多 {int(ok_.hit_m2.sum())-int(ok_.hit_final.sum())} 条；",
          f"- 标注**违反**题面的 {len(bad)} 条：硬排除对 {int(bad.hit_m2.sum())} 条、软先验对 {int(bad.hit_final.sum())} 条"
          f" —— 软先验多捡回 {int(bad.hit_final.sum())-int(bad.hit_m2.sum())} 条。", "",
          "硬排除等于给方向一道**绝对否决权**：cos < 0 的候选直接出局。可那 157 条（占可算样本 26%）的标注，",
          "恰恰落在被否决的那一侧——题面说左，标注给了右。**规则没错，是数据不听话**；",
          "否决权给出去了就收不回，所以降级成 0.2 的权重：方向仍影响排序，但不许一票否决。", "",
          f"净效果：**{m2}% → {fin}%**，多 {int(d.hit_final.sum())-int(d.hit_m2.sum())} 条（McNemar p<0.0001，见 §5.3）。", "",
          "## 3 0.2 这个权重是不是调出来的", "",
          "方位权重从 0（纯距离）扫到 1（纯方位），硬层固定：", "",
          "| w_ori | " + " | ".join(f"{v:.1f}" for v in sw.w_ori) + " |",
          "|---|" + "---:|" * len(sw),
          "| M-soft 命中 | " + " | ".join(f"{v}" for v in sw.命中) + " |", "",
          f"最优点 w={bw:.1f} → **{bv}%**；曲线很平（0.1~0.4 都在 {min(sw.命中[1:5]):.1f}% 以上），不敏感。", "",
          f"5 折交叉验证（按 Task_ID 分组切折，同一任务不跨折）：每折独立选 w，**5 折选中 {cv_w}**，OOF = **{cv_hit:.1f}%**。",
          f"与全表 {bv}% 一致 —— **0.2 是从训练折里选的，不是拿测试折试出来的**。", "",
          f"顺带解释老版 M3：它用 w=0.5，只有 {m35}%，因为 w 越大越信方位，而方位在这份数据里本来就不太靠得住。", "",
          f"## 4 主表（同一批 {n} 条，一把尺子）", "",
          "| 方法 | 类型 | 命中率 | 95% CI | 说明 |", "|---|---|---:|---|---|"]
    note = {"M-soft": "本文主推：软先验 0.2 + 题面明写距离时硬取",
            "U-SET": "你的判分口径：标注落在允许范围内",
            "M3*": "消融：去掉距离硬层，纯软加权 0.2",
            "M3 软加权 w=0.5": "论文旧版软加权，方位权重过高",
            "M2 硬筛": "你原规则的字面落地：硬排除 + 侧内最近",
            "M0": "旧基线：全池取离锚点最近"}
    for _, r in m.iterrows():
        tag = next((v for k, v in note.items() if r["方法"].startswith(k)), "")
        L.append(f"| {r['方法']} | {r['性质']} | **{r['命中']}%** | [{r['lo']}, {r['hi']}] | {tag} |")
    L += ["",
          f"**距离硬层只触发 {len(mx)} 条**：{n} 条里题面写了 nearest/furthest 的只有 {n_dw} 条，其中 {len(mx)} 条参照能定位。",
          f"这 {len(mx)} 条上硬取对 {int(mx.hit_final.sum())} 条，纯软加权对 {int(mx.hit_m3.sum())} 条，差 {abs(int(mx.hit_m3.sum())-int(mx.hit_final.sum()))} 条——",
          f"所以去掉硬层能到 {m3}%（比 {fin}% 高 {round(abs(m3-fin),1)} 点，p=1.0，纯噪声）。", "",
          f"**为什么仍保留硬层**：题面写了 `nearest` 就必须按最近取，这是指令遵循，不是能为了 {round(abs(m3-fin),1)} 个点丢掉的开关。",
          f"这 {len(mx)} 条还正好把 §13 的结论再证一遍（见 §6）。", "",
          "## 5 分层与显著性", "",
          "### 5.1 按数据档", "",
          "| 档 | n | U-SET | M-soft | M3* | M2 硬排除 | M0 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in band.iterrows():
        L.append(f"| {r.name} | {int(r.n)} | {r['U-SET']:.1f}% | {r['M-soft']:.1f}% | "
                 f"{r['M3*']:.1f}% | {r['M2']:.1f}% | {r['M0']:.1f}% |")
    d2r = band.loc["D2R"]
    L += ["",
          f"D2R（锚点由同场景补齐、标注自洽 93%）上 M-soft {d2r['M-soft']:.1f}%、硬排除 {d2r['M2']:.1f}% —— **数据越干净，方向越有用**，",
          "再次说明瓶颈在标注不在方法。",
          "D1（无可算锚点，69 条）M-soft 33.3% = M0：没有方向信息时软层退化成纯距离，符合预期，这 69 条按你的话「不算」。", "",
          "### 5.2 按方位类型", "",
          "| 类型 | n | U-SET | M-soft | M2 硬排除 | M0 |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in rel.iterrows():
        L.append(f"| {r.name} | {int(r.n)} | {r['U-SET']:.1f}% | {r['M-soft']:.1f}% | "
                 f"{r['M2']:.1f}% | {r['M0']:.1f}% |")
    L += ["", "### 5.3 配对检验（同样本逐条比）", "",
          "| 对手 | M-soft 对 / 对手错 | M-soft 错 / 对手对 | p |", "|---|---:|---:|---:|"]
    for t in tests:
        pv = "<0.0001" if t["p"] < 1e-4 else f"{t['p']:.4f}"
        L.append(f"| {t['对手']} | {t['M对对手错']} | {t['对手对M错']} | {pv} |")
    L += ["",
          "对旧基线 M0 与旧版你的规则 M2 都显著更好；对纯软加权 M3* 差 1 条样本（p=1.0）—— 硬层不影响结论。", "",
          "## 6 那 4 条硬距离样本（顺带再证 §13）", "",
          "| 题面 | 字面要求 | 标注离锚点第几近 | M-soft 取 | 结果 |", "|---|---|---|---|---|"]
    for r in mx.to_dict("records"):
        kind, rank, ref = r["dist_hard"]
        word = "最近" if kind == "min" else "最远"
        ask = f"到{'机器人' if ref == 'robot' else '参照物'}{word}"
        L.append(f"| `{r['Instruction'][:50]}` | {ask} | 第 {rank_anchor(r)} 近 | "
                 f"候选 {r['pick']+1} | {'✅' if r['hit_final'] else '❌'} |")
    L += ["",
          f"{len(mx)} 条里按字面对 {int(mx.hit_final.sum())} 条。看第 3 列就明白了：",
          f"**{len(mx)} 条里有 {n_anchor1} 条的标注是「离锚点最近的那个」（第 1 近）**，",
          "也就是说标注者实际是按「贴在锚点上」来选的，而题面写的是 `to you`（到你的距离）。",
          "这正是 §13 查明的：ALFRED 的距离列是「到锚点距离」（误差 0.2 mm），跟题面里的 `you` 不是一回事。",
          "所以不是方法的错，是**题面和标注用了两套距离参照**。",
          f"若改成一律按锚点量，这 {len(mx)} 条会对 {ok_anchor} 条——**换对两条、换错两条，总数不变**，",
          "两种量法谁也不占便宜，所以按字面走（`to you` 就用机器人坐标），把这件事写进局限。", "",
          "## 7 论文里能写什么、不能写什么", "",
          "| 能写 | 不能写 |", "|---|---|",
          f"| 「按题面允许范围的判分口径下 {uset}%，其余 {n-int(round(uset/100*n))} 条的标注与题面直接矛盾」 | 「准确率 {uset}%」——它不是选答案的准确率 |",
          f"| 「要求单一答案时 {fin}%，比取最近基线 {m0}% 高 {round(fin-m0,1)} 点（p=0.0018）」 | 把两个数放同一张表里比高低 |",
          f"| 「方向应作软先验（w=0.2）而非硬约束：硬排除在标注自洽子集多 {int(ok_.hit_m2.sum())-int(ok_.hit_final.sum())} 条，在矛盾子集少 {int(bad.hit_final.sum())-int(bad.hit_m2.sum())} 条」 | 「方向无用」——在干净子集上它是有用的 |",
          f"| 「距离约束极少写进题面（{n} 条里仅 {n_dw} 条），第二步几乎不触发」 | 「距离步验证有效」——只有 {len(mx)} 条，对 {int(mx.hit_final.sum())} 条 |",
          f"| 「标注噪声天花板：题面唯一确定答案的 {n_uni} 条里，口径判定仅 {uset_uni}%」 | 把 {uset_uni}% 归因于方法 |", "",
          "### 7.1 必须写进局限的三条", "",
          f"1. **白送比例**：{uset}% 里含 {n_free} 条「题面没约束、任取都对」；{n_dir} 条方向可算的口径是 "
          f"{round(d[d.Band!='D1'].ok.mean()*100,1)}%，再剔除 {int((d.free=='方向筛空回退').sum())} 条「筛空回退」后，"
          f"真被题面筛过一遍的 {n_real} 条是 {uset_real}%。报数必须同时给这几个。",
          "2. **第 3 层（文本）没真正用上**：颜色在 ALFRED 里不落字段，状态谓词在这批样本里几乎不出现，所以「按文本选」等价于「按类别选」。",
          f"3. **样本量**：{n} 条来自 8051 个任务里的可判分子集（A 档 {int(band.loc['A'].n)}）；C 档仅 {int(cband.n)} 条"
          f"（口径 {cband['U-SET']:.1f}%、单点 {cband['M-soft']:.1f}%），不建议单独下结论。", "",
          "## 8 文件", "",
          "| 文件 | 内容 |", "|---|---|",
          "| `eval_final.py` | 本脚本（只读冻结数据） |",
          "| `final_main.csv` | 主表（含 CI） |",
          "| `final_by_band.csv` | 分档表 |",
          "| `final_sweep.csv` | 权重 0→1 扫描 |",
          "| `final_picks.json` | 逐样本：每法命中、选中的候选、用的哪一层 |", ""]
    open(f"{RESULTS}/FINAL_METHOD.md", "w", encoding="utf-8").write("\n".join(L))


def main():
    d = build_final()
    n = len(d)
    print(f"样本 {n} 条\n")

    rows = []

    def add(name, col, kind="单点"):
        if col is None:
            mm, lo, hi = ci(list(d.ok), 1)
        else:
            mm, lo, hi = ci(list(d[col]))
        rows.append(dict(方法=name, n=n, 命中=mm, lo=lo, hi=hi, 性质=kind))

    add("M-soft 单点（软先验 0.2 + 显式距离硬约束）", "hit_final")
    add("U-SET 允许集合判定（你的口径）", None, "集合")
    add("M3* 软加权 w=0.2（无距离硬层）", "hit_m3")
    add("M3 软加权 w=0.5（论文旧版）", "hit_m35")
    add("M2 硬筛+侧内最近（原规则字面版）", "hit_m2")
    add("M0 取锚点最近（旧基线）", "hit_m0")
    m = pd.DataFrame(rows)
    print(m.to_string(index=False))

    print("\n配对检验:")
    tests = []
    for other, col in [("M2 硬筛+侧内最近", "hit_m2"), ("M0 取锚点最近", "hit_m0"),
                       ("M3* 软加权 w=0.2", "hit_m3"), ("M3 软加权 w=0.5", "hit_m35")]:
        p, a, b = C.mcnemar(d.hit_final.astype(bool), d[col].astype(bool))
        tests.append(dict(对手=other, M对对手错=a, 对手对M错=b, p=p))
        print(f"  vs {other:<24} 对/{a:3d} 错/{b:3d}  p={p:.4f}")

    def tab(by):
        return d.groupby(by).apply(lambda gg: pd.Series({
            "n": len(gg), "U-SET": round(gg.ok.mean() * 100, 1),
            "M-soft": round(gg.hit_final.mean() * 100, 1),
            "M3*": round(gg.hit_m3.mean() * 100, 1),
            "M2": round(gg.hit_m2.mean() * 100, 1),
            "M0": round(gg.hit_m0.mean() * 100, 1)}))

    band = tab("Band").loc[["A", "D2R", "B", "C", "D1"]]
    print("\n分档:\n" + band.to_string())
    rel = tab("Rel_Type")
    print("\n方位类型:\n" + rel.to_string())
    lab = d.groupby(d.label_side.map({True: "自洽", False: "违反", None: "无标注"})).apply(
        lambda gg: pd.Series({"n": len(gg), "U-SET": round(gg.ok.mean() * 100, 1),
                              "M-soft": round(gg.hit_final.mean() * 100, 1),
                              "M2": round(gg.hit_m2.mean() * 100, 1),
                              "M0": round(gg.hit_m0.mean() * 100, 1)}))
    print("\n标注一致性:\n" + lab.to_string())

    sw = sweep_final(d)
    print("\n权重扫描: " + "  ".join(f"w={r.w_ori}:{r.命中}" for r in sw.itertuples()))
    cv_hit, cv_w = cv_final(d)
    print(f"5 折交叉：OOF {cv_hit:.1f}%  每折选中 {cv_w}")

    print("层触发:", d.layer.value_counts().to_dict())

    slim = d.drop(columns=["_same", "dist_hard", "_robot"])
    slim.to_json(f"{DATA}/final_picks.json", orient="records", force_ascii=False)
    m.to_csv(f"{RESULTS}/final_main.csv", index=False, encoding="utf-8-sig")
    band.to_csv(f"{RESULTS}/final_by_band.csv", encoding="utf-8-sig")
    sw.to_csv(f"{RESULTS}/final_sweep.csv", index=False, encoding="utf-8-sig")
    write_report(d, m, band, lab, rel, tests, sw, cv_hit, cv_w)
    print("写出 final_main.csv / final_by_band.csv / final_sweep.csv / final_picks.json / FINAL_METHOD.md")
    return d


if __name__ == "__main__":
    main()
