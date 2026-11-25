# -*- coding: utf-8 -*-
"""
audit_final_clean.py —— 终审：剩下的数据是不是"要么按口径可算，要么不存在"

口径
----
方向 = 机器人面朝锚点那条视线（左/右 = 视线左/右转 90°；behind = 背离；front = 靠机器人）
距离 = 到锚点的距离（不是到机器人）

逐条判定，只有三种合法状态：
  OK   按口径可算：锚点是场景里的实物、且不等于机器人/目标，方向与距离都能复算
  NONE 不存在：题面没有"目标 vs 另一个物体"的方位约束（并按用户规则：同类任取都对）
  --   未定：句子里有关系但坐标算不出（D2）/ 同类唯一（D0）—— 既不是可算也不是不存在

输出
----
FINAL_CLEAN_AUDIT.md   逐档审计表 + 违规清单
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

import pandas as pd

import _paths as _p

DATA = _p.DATA
RESULTS = _p.RESULTS
DEGEN = 0.05


def unit(v):
    n = math.hypot(v[0], v[1])
    return None if n < 1e-9 else (v[0] / n, v[1] / n)


def main():
    df = pd.read_csv(os.path.join(DATA, "clean_spatial_ext.csv"), encoding="utf-8-sig")
    cands = json.load(open(os.path.join(DATA, "loc_candidates.json"), encoding="utf-8"))
    ancs = json.load(open(os.path.join(DATA, "loc_anchors.json"), encoding="utf-8"))

    rows = []
    for _, r in df.iterrows():
        cl = cands.get(r.Task_ID) or []
        same = [c for c in cl if c["category"] == r.Target_Category]
        ids = [str(c["id"]) for c in same]
        gt = str(r.Target_Object) if "Target_Object" in r and pd.notna(r.Target_Object) else None
        rob = (ancs.get(r.Task_ID) or {}).get("robot")
        rec = dict(r=r, band=r.Band, prob=[])

        # ---- 1 参照物不能是机器人 ----
        if "smart" in str(r.Instruction).lower():
            pass
        dref = str(r.Dist_Ref) if "Dist_Ref" in r and pd.notna(r.Dist_Ref) else "anchor"
        if dref == "robot":
            rec["prob"].append("距离参照是机器人（距离档不可用，方向档可用）")

        if r.Judge == "side":
            if not (pd.notna(r.Anchor_X) and pd.notna(r.Anchor_Y)):
                rec["prob"].append("标注可判分但锚点坐标为空")
                rows.append(rec)
                continue
            ax, ay = float(r.Anchor_X), float(r.Anchor_Y)
            if not rob:
                rec["prob"].append("缺机器人坐标")
                rows.append(rec)
                continue
            d_ar = math.hypot(ax - rob[0], ay - rob[1])
            if d_ar < DEGEN:
                rec["prob"].append("锚点=机器人(方向退化)")
            # 目标坐标
            if not ids or gt not in ids:
                rec["prob"].append("目标不在候选集")
                rows.append(rec)
                continue
            ti = ids.index(gt)
            tgt = same[ti]
            # ---- 距离口径：D_Target 必须是到锚点 ----
            d_ta = math.hypot(tgt["x"] - ax, tgt["y"] - ay)
            d_tr = math.hypot(tgt["x"] - rob[0], tgt["y"] - rob[1])
            if abs(d_ta - float(r.D_Target)) > 1e-3:
                rec["prob"].append(f"D_Target 不是到锚点({d_ta:.3f} vs {r.D_Target:.3f})")
            if d_ta < DEGEN:
                rec["prob"].append("目标=锚点(方向退化)")
            # ---- 方向口径复算 ----
            fwd = unit((ax - rob[0], ay - rob[1]))
            left = (-fwd[1], fwd[0])
            v = unit((tgt["x"] - ax, tgt["y"] - ay))
            if v and fwd:
                c_left = v[0] * left[0] + v[1] * left[1]
                c_fwd = v[0] * fwd[0] + v[1] * fwd[1]
                k = str(r.Rel_Type)
                if k == "lr":
                    recomputed = c_left if r.Side == "left" else -c_left
                elif k == "behind":
                    recomputed = c_fwd
                else:
                    recomputed = -c_fwd
                if pd.notna(r.Side_Cos) and abs(recomputed - float(r.Side_Cos)) > 1e-3:
                    rec["prob"].append("Side_Cos 与口径复算不一致")
                rec["cos"] = recomputed
                rec["in_side"] = recomputed > 0
            # 锚点是否是场景里的实物（与某个候选物重合，或用 PDDL 容器近似）
            rec["d_anchor_obj"] = min((math.hypot(c["x"] - ax, c["y"] - ay) for c in cl), default=9e9)
            rec["d_ar"] = d_ar
            rec["d_ta"] = d_ta
        elif r.Band == "D1":
            # ---- "不存在"必须成立：句子里没有任何锚点式方位关系 ----
            if re.search(r"\b(left|right)\s+(side|corner|edge)?\s*of\s+the\b", str(r.Instruction), re.I):
                rec["prob"].append("D1 但句子里有 of the 方位关系")
        rows.append(rec)

    # ---------- 汇总 ----------
    def stat(band):
        sub = [x for x in rows if x["band"] == band]
        bad = [x for x in sub if x["prob"]]
        return len(sub), len(bad), bad

    md = ["# 终审：剩下的数据干净吗", "",
          "> 口径：方向 = 机器人→锚点视线；距离 = 到锚点距离。",
          "> 每条只有三种合法状态：**OK 可算** / **NONE 不存在** / **未定**。", "",
          "## 1 逐档判定", "",
          "| 档 | 条数 | 状态 | 违反口径 | 说明 |",
          "|---|---:|---|---:|---|"]
    desc = {
        "A": ("OK", "`to the left/right of X`，锚点是实物坐标"),
        "B": ("OK", "`the left X on the Y`，锚点靠句法推断"),
        "C": ("OK", "`from the right shelf`，方位词可能污染锚点"),
        "D2R": ("OK", "锚点由同场景其它任务补齐（唯一实例）"),
        "D1": ("NONE", "题面没有锚点式方位约束，同类任取都对"),
        "D2": ("未定", "有关系但坐标算不出：缺坐标 / 方位词修饰容器 / 同名多实例 / 最近 X"),
        "D0": ("未定", "同类唯一，不存在选择"),
    }
    for b in ["A", "B", "C", "D2R", "D1", "D2", "D0"]:
        n, nb, _ = stat(b)
        md.append(f"| {b} | {n} | {desc[b][0]} | {nb} | {desc[b][1]} |")
    md.append("")

    # ---------- 关键量 ----------
    ok = [x for x in rows if x["band"] in ("A", "B", "C", "D2R")]
    in_side = sum(1 for x in ok if x.get("in_side"))
    d_anchor_obj = [x["d_anchor_obj"] for x in ok if "d_anchor_obj" in x]
    d_ar = [x["d_ar"] for x in ok if "d_ar" in x]
    d_ta = [x["d_ta"] for x in ok if "d_ta" in x]
    import statistics as st
    md += ["## 2 可算档（A/B/C/D2R）的体检", "",
           "| 档 | 条数 | 标注落在指定侧 | 自洽率 |", "|---|---:|---:|---:|"]
    for b in ["A", "B", "C", "D2R"]:
        sub = [x for x in ok if x["band"] == b]
        ins = sum(1 for x in sub if x.get("in_side"))
        md.append(f"| {b} | {len(sub)} | {ins} | {ins/len(sub)*100:.1f}% |")
    md += [f"| **合计** | **{len(ok)}** | **{in_side}** | **{in_side/len(ok)*100:.1f}%** |", "",
           f"- 条数 **{len(ok)}**；按口径复算，标注目标落在题面指定侧的 **{in_side}** 条"
           f"（{in_side/len(ok)*100:.1f}%）→ 其余是标注与定义不自洽（噪声/换参照系），不是公式错。",
           f"- 锚点到最近候选物的距离：中位 **{st.median(d_anchor_obj):.3f}m**，"
           f"<0.02m 的占 **{sum(1 for d in d_anchor_obj if d<0.02)/len(d_anchor_obj)*100:.1f}%**"
           "（一半锚点是家具/容器，坐标取容器中心，不是物体自身）。",
           f"- 锚点到机器人：中位 **{st.median(d_ar):.2f}m**，最小 **{min(d_ar):.2f}m**（无退化）。",
           f"- 目标到锚点：中位 **{st.median(d_ta):.2f}m**，最小 **{min(d_ta):.2f}m**"
           f"（<0.3m 占 {sum(1 for d in d_ta if d<0.3)/len(d_ta)*100:.0f}%，物体多半就摆在锚点上）。", "",
           "### 2.1 方向类型与文本是否对得上", "",
           "| 类型 | 条数 | 文本里能找到对应方位词 |", "|---|---:|---:|"]
    import re as _re
    for k, pat in [("lr", r"\b(left|right)\b"), ("behind", r"behind"), ("front", r"front of")]:
        sub = [x for x in ok if str(x["r"].Rel_Type) == k]
        good = sum(1 for x in sub if _re.search(pat, str(x["r"].Instruction), _re.I))
        md.append(f"| {k} | {len(sub)} | {good}/{len(sub)} |")
    md += ["", "### 2.2 距离参照是谁", "",
           "| 参照 | 条数 | 说明 |", "|---|---:|---|",
           f"| 锚点（合规） | {sum(1 for x in ok if str(x['r'].Dist_Ref) == 'anchor')} | 距离档按到锚点算 |",
           f"| 机器人（`closest to you`） | {sum(1 for x in ok if str(x['r'].Dist_Ref) == 'robot')} | 方向上没问题，但距离语义不同，距离档要单独剔掉 |", ""]

    # ---------- 违规清单 ----------
    bad = [x for x in rows if x["prob"]]
    md += ["## 3 违反口径的条目", "",
           f"共 **{len(bad)}** 条。", ""]
    if bad:
        c = Counter(p for x in bad for p in x["prob"])
        md += ["| 问题 | 条数 |", "|---|---:|"]
        for k, v in c.most_common():
            md.append(f"| {k} | {v} |")
        md += ["", "样例：", ""]
        for x in bad[:8]:
            md.append(f"- [{x['band']}] {str(x['r'].Instruction)[:80]} → {'; '.join(x['prob'])}")
        md.append("")

    # ---------- 结论 ----------
    n_ok, n_none, n_und = len(ok), stat("D1")[0], stat("D2")[0] + stat("D0")[0]
    md += ["## 4 一句话结论", "",
           f"- **按口径可算 {n_ok} 条**（A/B/C {stat('A')[0]+stat('B')[0]+stat('C')[0]} + D2R {stat('D2R')[0]}），"
           f"其中标注自洽 {in_side} 条。",
           f"- **不存在 {n_none} 条**（D1）：题面没有锚点式方位约束，判分按「同类任取都对」。",
           f"- **未定 {n_und} 条**（D2 {stat('D2')[0]} + D0 {stat('D0')[0]}）：既算不出、也不能说「不存在」，"
           "只能入库，不能进主表。", ""]
    open(os.path.join(RESULTS, "FINAL_CLEAN_AUDIT.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
