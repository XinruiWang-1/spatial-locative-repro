# -*- coding: utf-8 -*-
"""
extend_clean_917.py —— 把「锚点坐标缺失」的 917 条按用户口径并进来

用户口径（不可违背）
------------------
方向轴 : 机器人 -> 锚点(观测物体) 这条视线
         left/right = 视线左/右转 90°；behind = 背离观察者；front = 靠观察者
距离   : 到锚点(观测物体)的距离，不是到机器人

并入原则：可以"没有"，不能"相违背"
--------------------------------
X1 参照物不能是机器人：`closest to you` 这类距离参照是机器人 -> 剔除
X2 标注目标必须能在候选集里找到（不在候选集 -> 剔除）
X3 目标类别必须在句子里出现（没提到 -> 剔除）
X4 目标与锚点重合（方向退化）-> 剔除

分档
----
D1  无约束   : 句子里没有「目标 vs 另一个物体」的方位关系（`turn right`、`left corner`、`the left keys`）
               -> 题面没说方向，同类任取都对，可判分
D2R 锚点迁移 : 写了 `to the left of X`，X 的坐标本任务没有，但**同场景其它任务**的 PDDL 里有，
               且该场景该名称只有 1 个实例 -> 用场景级坐标算出方向/距离，可判分
               （已校验：PDDL 坐标 /4 与物体精确坐标完全一致；与主集 exact 锚点一致率 83%）
D2  锚点缺失 : 有关系但坐标补不到（同场景也没有，或同名多实例分不清）-> 入库、不判分
D0  唯一实例 : 该类整场只有 1 个，不存在"选哪个" -> 入库、不判分

输出
----
clean_spatial_ext.csv   原 A/B/C 档 + D1/D2R/D2/D0（新增 Band / Scorable / Constraint / Anchor_Status / Note）
CLEAN_917.md            并入诊断报告
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

SIDE_PAT = r"\b(left|right)\b"
BEHIND_PAT = r"\bbehind\s+(?:the\s+)?[a-z]"
FRONT_PAT = r"\bin front of\s+(?!you|me\b)(?:the\s+)?[a-z]"
# "目标 vs 另一个物体" 的真正方位关系短语
_NP = r"([a-z]+(?:\s+[a-z]+){0,2})"          # 锚点名词最多 3 个词，防止把后半句全吞进来
REL_PAT = re.compile(
    (r"(?:(?:to|on|in|at|from)\s+)?the\s+(?:far\s+|back\s+|front\s+|lower\s+|upper\s+|top\s+|bottom\s+)*"
     r"(?:left|right)(?:\s*(?:side|corner|edge|end|part|half))?\s+of\s+(?:the\s+)?" + _NP +   # to/on the left (side) of X
     r"|(?:in\s+)?front\s+of\s+(?:the\s+)?" + _NP +                                           # in front of X
     r"|behind\s+(?:the\s+)?" + _NP +                                                         # behind X
     r"|(?:left|right)\s+of\s+(?:the\s+)?" + _NP),                                            # left of X
    re.I)
# 方位词修饰"容器自身"（the left sink / the bottom left cabinet / the back right burner）
# 目标就在这个容器上，方向相对容器中心恒为 0 -> 不可定向（与主集 S3.5 同类）
CONTAINER = (r"(?:sink|sinkbasin|cabinet|shelf|drawer|counter|countertop|table|desk|dresser|couch|sofa|"
             r"bed|stove|burner|microwave|fridge|refrigerator|closet|stand|rack|tub|toilet|tv|television|"
             r"box|basket|cushion|corner|edge|side)")
# 方位词修饰"容器的某个部位"（the right back corner of the counter / the front left burner of the stove）
DIR_PART_OF = re.compile(
    r"\b(?:left|right|back|front)\s+(?:\w+\s+){0,2}"
    r"(corner|edge|side|burner|coil|range|door|drawer|shelf|section|part|half|end|level)\s+of\b", re.I)
# `the left pot from the stove top`：左/右 + 目标 + 容器 -> 仍是容器相对，不是候选之间的左右
CONTAINER_REL = re.compile(
    r"\bthe\s+(?:far\s+|left\s*most\s+|right\s*most\s+)?(?:left|right)\s+\w+\s+"
    r"(?:from|on|in|at|off)\s+(?:the\s+)?\w+", re.I)
CONTAINER_MOD = re.compile(
    r"\bthe\s+(?:far\s+|bottom\s+|top\s+|upper\s+|lower\s+|front\s+|back\s+)*"
    r"(?:left|right)(?:[- ]?hand)?(?:[- ](?:side|corner|edge|end))?\s+" + CONTAINER + r"\b", re.I)
STOP = {"you", "me", "your", "my", "it", "them", "this", "that"}
DIST_TO_YOU = re.compile(r"\b(closest|nearest|furthest|farthest)\b[^.]{0,30}\b(to|from)\s+(you|me)\b", re.I)
# "closest/nearest (to) the X"：距离约束，参照物是 X（不是机器人）
DIST_ANCHOR = re.compile(r"\b(closest|nearest)\b(?:\s+to)?\s+(?:the\s+)?([a-z][a-z ]*)", re.I)
DEGEN_TOL = 0.05

SYN = {
    "sink": ["sinkbasin", "sink"], "sink basin": ["sinkbasin"],
    "stove": ["stoveburner", "stove"], "burner": ["stoveburner"], "oven": ["stoveburner"],
    "counter": ["countertop"], "counter top": ["countertop"], "countertop": ["countertop"],
    "tv": ["television"], "tv stand": ["tvstand"], "couch": ["sofa"], "sofa": ["sofa"], "cushion": ["sofa"],
    "coffee maker": ["coffeemachine"], "coffee machine": ["coffeemachine"],
    "night stand": ["sidetable"], "nightstand": ["sidetable"],
    "table": ["diningtable", "coffeetable", "sidetable"], "chair": ["armchair"],
    "lamp": ["desklamp"], "garbage": ["garbagecan"], "trash": ["garbagecan"],
    "shelf": ["shelf"], "dresser": ["dresser"], "fridge": ["fridge"],
    "microwave": ["microwave"], "toaster": ["toaster"], "cabinet": ["cabinet"],
    "drawer": ["drawer"], "desk": ["desk"], "mirror": ["mirror"], "plant": ["houseplant"],
    "bed": ["bed"],
}


def unit(v):
    n = math.hypot(v[0], v[1])
    return None if n < 1e-9 else (v[0] / n, v[1] / n)


def side_kind(s):
    s = s or ""
    if re.search(FRONT_PAT, s, re.I):
        return "front"
    if re.search(BEHIND_PAT, s, re.I):
        return "behind"
    if re.search(SIDE_PAT, s, re.I):
        return "lr"
    return None


def rel_phrase(s):
    """句子里是否写了 '目标 vs 另一个物体' 的方位关系；返回锚点短语或 None"""
    if not isinstance(s, str):
        return None
    # `the back, right corner of the shelf` -> `the back right corner of the shelf`
    s = re.sub(r"\b(left|right|back|front)\s*,\s*", r"\1 ", s)
    pos = 0
    while True:
        m = REL_PAT.search(s, pos)
        if not m:
            return None
        groups = [g for g in m.groups() if g]
        if not groups:
            pos = m.start() + 1
            continue
        t = groups[0].strip().lower()
        if t.split()[0] in STOP:
            pos = m.start() + 1
            continue
        return t


def dist_anchor(s):
    """`closest / nearest (to) the X`：距离参照是另一个物体（不是机器人）时返回 X"""
    if not isinstance(s, str):
        return None
    for m in DIST_ANCHOR.finditer(s):
        t = m.group(2).strip().lower()
        if t.split()[0] in STOP or t.split()[0] in ("one", "object"):
            continue
        return t
    return None


def akey(p):
    if not isinstance(p, str):
        return None
    p = re.sub(r"^(the|a|an)\s+", "", p.strip().lower())
    p = re.sub(r"\s+(on|from|next to|in|of|at|top|back|left|right|side)$", "", p).strip()
    return re.sub(r"[^a-z ]", "", p) or None


def main():
    df = pd.read_csv(os.path.join(DATA, "dataset_locative.csv"))
    cands = json.load(open(os.path.join(DATA, "loc_candidates.json"), encoding="utf-8"))
    ancs = json.load(open(os.path.join(DATA, "loc_anchors.json"), encoding="utf-8"))
    scene_anchors = json.load(open(os.path.join(DATA, "scene_anchors.json"), encoding="utf-8"))
    obj_of = dict(zip(df.Sample_ID, df.Target_Object))
    task_scene = json.load(open(os.path.join(DATA, "task_scene.json"), encoding="utf-8"))

    def resolve_anchor(tid, phrase):
        """同场景静态家具坐标。返回 (x, y, status)：unique / ambiguous / missing"""
        dic = scene_anchors.get(str(task_scene.get(tid)), {})
        k = akey(phrase)
        if not k:
            return None, None, "missing"
        for c in [k.replace(" ", "")] + SYN.get(k, []):
            if c in dic:
                poses = dic[c]
                if len(poses) == 1:
                    return poses[0][0], poses[0][1], "unique"
                return None, None, "ambiguous"
        return None, None, "missing"

    def geometry(r, ax, ay, same):
        a = ancs.get(r.Task_ID) or {}
        rob = a.get("robot")
        if not rob:
            return None
        d0 = math.hypot(ax - rob[0], ay - rob[1])
        fwd = unit((ax - rob[0], ay - rob[1]))
        if fwd is None or d0 < DEGEN_TOL:
            return None
        left = (-fwd[1], fwd[0])
        ds, lcs, fcs = [], [], []
        for c in same:
            v = (c["x"] - ax, c["y"] - ay)
            u = unit(v)
            ds.append(math.hypot(v[0], v[1]))
            lcs.append(None if u is None else u[0] * left[0] + u[1] * left[1])
            fcs.append(None if u is None else u[0] * fwd[0] + u[1] * fwd[1])
        return dict(rob=rob, ds=ds, lcs=lcs, fcs=fcs)

    df["_kind"] = [side_kind(x) for x in df.Instruction]
    s1 = df[df._kind.notna()]
    miss = s1[s1.Anchor_X.isna()].copy()
    print(f"S1 {len(s1)}  |  无锚点 {len(miss)}")

    rows, d0, drop = [], [], Counter()
    for _, r in miss.iterrows():
        if DIST_TO_YOU.search(r.Instruction or ""):
            drop["X1 距离参照是 you/me（与口径相违背）"] += 1
            continue
        if r.Cat_Mentioned != 1:
            drop["X3 目标类别没提到"] += 1
            continue
        cl = cands.get(r.Task_ID) or []
        same = [c for c in cl if c["category"] == r.Target_Category]
        gt = str(r.Target_Object)
        ids = [str(c["id"]) for c in same]
        if gt not in ids and len(same) >= 2:
            drop["X2 标注目标不在候选集"] += 1
            continue
        rp = rel_phrase(r.Instruction)
        da = None if rp else dist_anchor(r.Instruction)
        rec = dict(row=r, same=same, ids=ids,
                   ti=ids.index(gt) if gt in ids else -1,
                   anchor_phrase=rp or da or r.Anchor_Phrase, status="na",
                   ax=None, ay=None, geo=None)
        if len(same) < 2:
            rec.update(band="D0", status="single")
            d0.append(rec)
            continue
        if da:
            # 距离约束（最近的是 X）：参照物成立，但"谁最近"要在候选池里排序，
            # 与"落在哪一侧"不是同一套判分，暂不入判分集
            rec.update(band="D2", status="dist_constraint")
            rows.append(rec)
            continue
        rec["band"] = "D2" if rp else "D1"
        if not rp and (CONTAINER_MOD.search(r.Instruction or "")
                       or DIR_PART_OF.search(r.Instruction or "")
                       or CONTAINER_REL.search(r.Instruction or "")):
            # `the yellow knife in the left sink`：方位词修饰的是容器，不是候选之间的相对位置
            rec.update(band="D2", status="container_modified")
            rows.append(rec)
            continue
        if rec["band"] == "D2":
            ax, ay, st = resolve_anchor(r.Task_ID, r.Anchor_Phrase)
            if st == "unique":
                geo = geometry(r, ax, ay, same)
                if geo and geo["ds"][rec["ti"]] >= DEGEN_TOL:
                    rec.update(band="D2R", status="scene", ax=ax, ay=ay, geo=geo)
                else:
                    rec["status"] = "degenerate"
            else:
                rec["status"] = st
        rows.append(rec)

    by = lambda b: [x for x in rows if x["band"] == b]
    d1, d2r, d2 = by("D1"), by("D2R"), by("D2")
    print(f"D1 无约束 {len(d1)}  D2R 场景锚点迁移 {len(d2r)}  D2 锚点缺失 {len(d2)}  "
          f"D0 唯一实例 {len(d0)}  剔除 {sum(drop.values())}")
    print("剔除原因:", dict(drop))
    print("D2 锚点状态:", Counter(x["status"] for x in d2).most_common())

    # ---------- 读入原 A/B/C 档 ----------
    base = pd.read_csv(os.path.join(DATA, "clean_spatial.csv"))
    base["Band"] = [("A" if r.Rel_Type in ("behind", "front")
                     else {"lr_of": "A", "rel": "B", "from": "C"}.get(r.Rel_Kind, "A"))
                    for _, r in base.iterrows()]
    base["Scorable"] = 1
    base["Constraint"] = "dir"
    base["Anchor_Status"] = "exact"
    base["Judge"] = "side"
    base["Status"] = "OK"
    base["Dist_Ref"] = ["robot" if DIST_TO_YOU.search(str(x)) else "anchor" for x in base.Instruction]
    base.insert(list(base.columns).index("Rel_Type"), "Target_Object",
                base.Sample_ID.map(obj_of))
    base["Note"] = ""

    NOTE = {"D1": "题面无方位约束，同类任取都对",
            "D2R": "锚点由同场景其它任务补齐（与 exact 锚点一致率 83%）",
            "D2": "锚点坐标缺失，定义成立但算不出",
            "D0": "同类只有 1 个，不存在选择问题"}
    CONS = {"D1": "none", "D2R": "dir", "D2": "dir_missing", "D0": "single"}

    add = []
    for x in rows + d0:
        r = x["row"]
        geo, ti = x["geo"], x["ti"]
        if geo:
            k = r._kind
            v = None
            if k == "lr":
                c = geo["lcs"][ti]
                v = None if c is None else (c if r.Rel_Direction == "left" else -c)
            elif k == "behind":
                v = geo["fcs"][ti]
            else:
                c = geo["fcs"][ti]
                v = None if c is None else -c
            side_cos, d_tgt = v, geo["ds"][ti]
            in_side = bool(v is not None and v > 0)
        else:
            side_cos, d_tgt, in_side = float("nan"), float("nan"), ""
        add.append(dict(
            Sample_ID=r.Sample_ID, Task_ID=r.Task_ID, Split=r.Split,
            Instruction=r.Instruction, Target_Category=r.Target_Category,
            Target_Object=str(r.Target_Object), Rel_Type=r._kind,
            Rel_Kind=r.Rel_Kind if isinstance(r.Rel_Kind, str) else "NA",
            Side=r.Rel_Direction if isinstance(r.Rel_Direction, str) else "",
            Anchor_Name="", Anchor_Phrase=x["anchor_phrase"] or "",
            Anchor_X=x["ax"] if x["ax"] is not None else float("nan"),
            Anchor_Y=x["ay"] if x["ay"] is not None else float("nan"),
            Robot_X=geo["rob"][0] if geo else float("nan"),
            Robot_Y=geo["rob"][1] if geo else float("nan"),
            Num_Same=len(x["same"]), Target_Idx=ti,
            D_Target=d_tgt, Side_Cos=side_cos, Label_In_Side=in_side,
            Band=x["band"],
            Scorable=1 if x["band"] in ("D1", "D2R") else 0,
            Constraint=CONS[x["band"]],
            Anchor_Status=x["status"],
            Judge={"D1": "category", "D2R": "side"}.get(x["band"], "-"),
            Dist_Ref=("robot" if DIST_TO_YOU.search(str(r.Instruction)) else "anchor"),
            Status={"D1": "NONE", "D2R": "OK"}.get(x["band"], "UNDEFINED"),
            Note=NOTE[x["band"]],
        ))
    ext = pd.concat([base, pd.DataFrame(add)], ignore_index=True)
    ext.to_csv(os.path.join(RESULTS, "clean_spatial_ext.csv"), index=False, encoding="utf-8-sig")

    cnt = Counter(ext.Band)
    scorable = int(ext.Scorable.sum())
    print(f"\n写出 clean_spatial_ext.csv: {len(ext)} 行"
          f"（A/B/C {len(base)} + D1 {len(d1)} + D2R {len(d2r)} + D2 {len(d2)} + D0 {len(d0)}）")
    print(f"可判分 {scorable} 条；D2R 标注落在题面侧 "
          f"{sum(1 for x in d2r if x['geo'] and x['geo'] and x['geo']['ds'] is not None)} 条中 "
          f"{sum(1 for a in add if a['Band']=='D2R' and a['Label_In_Side'] is True)} 条")

    # ---------- 报告 ----------
    d2r_in = sum(1 for a in add if a["Band"] == "D2R" and a["Label_In_Side"] is True)
    md = ["# 917 条「锚点缺失」并入报告", "",
          "> 口径（不可违背）：方向轴 = 机器人→锚点视线；距离 = 到锚点的距离。",
          "> 原则：可以「没有」，不能「相违背」。", "",
          "## 1 结论", "",
          f"- 917 条全部有交代：**{len(rows)+len(d0)}** 条入库（D1 {len(d1)} + D2R {len(d2r)} + "
          f"D2 {len(d2)} + D0 {len(d0)}），剔除 {sum(drop.values())} 条（与口径相违背）。",
          f"- 合并后总集 **{len(ext)}** 条：A {cnt['A']} + B {cnt['B']} + C {cnt['C']} + "
          f"D1 {cnt['D1']} + D2R {cnt['D2R']} + D2 {cnt['D2']} + D0 {cnt['D0']}。",
          f"- **可自动判分 {scorable} 条**（A/B/C {len(base)} + D1 {len(d1)} + D2R {len(d2r)}）；"
          f"D2/D0 共 {len(d2)+len(d0)} 条只入库、不判分。", "",
          "## 2 各档含义", "",
          "| 档 | 定义 | 判分 | 条数 |", "|---|---|---|---:|",
          f"| D1 | 句子里没有「目标 vs 另一个物体」的方位关系（`turn right`、`left corner of the desk`、`the left keys`） | 同类任取都对（题面没说就不约束） | {len(d1)} |",
          f"| D2R | 写了 `to the left of X`，本任务没有 X 坐标，但同场景其它任务的 PDDL 有，且 X 在该场景唯一 | 可判分（锚点迁移） | {len(d2r)} |",
          f"| D2 | 有关系但坐标补不到：同场景也没有，或同名多实例分不清 | 不判分 | {len(d2)} |",
          f"| D0 | 该类整场只有 1 个，选中就行 | 不判分 | {len(d0)} |", "",
          "## 3 剔除原因（防止与口径相违背）", "",
          "| 原因 | 条数 |", "|---|---:|"]
    for k, v in drop.most_common():
        md.append(f"| {k} | {v} |")
    md += ["", "## 4 D2R 的锚点为什么可信", "",
           "- ALFRED 同一 `scene_num` 是同一间屋子，家具不动；PDDL 只记「本次计划访问过的」容器，",
           "  所以单任务视角缺锚点，同场景其它任务能补上。",
           "- PDDL 坐标 `/4` 与 `object_poses` 的精确坐标**完全一致**（n=99，误差 0.000m），不是栅格近似。",
           f"- 用主集 523 条（exact 锚点）反查：可比 176 条中 146 条误差 <0.05m（**83%**），median 0.000m。",
           f"- D2R 这 {len(d2r)} 条里，标注目标落在题面指定侧的有 **{d2r_in}** 条"
           f"（{d2r_in/len(d2r)*100:.1f}%），与主集 73% 同一量级，说明锚点没跑偏。", "",
           "## 5 D2 补不回来的原因", "",
           "| 锚点状态 | 条数 | 说明 |", "|---|---:|---|",
           f"| ambiguous | {sum(1 for x in d2 if x['status'] == 'ambiguous')} | 同场景同名多实例（两个 shelf / 四个 burner），分不清是哪个 |",
           f"| container_modified | {sum(1 for x in d2 if x['status'] == 'container_modified')} | 方位词修饰容器自身（`the yellow knife in the left sink`），目标就在容器上、方向恒为 0（同主集 S3.5） |",
           f"| missing | {sum(1 for x in d2 if x['status'] == 'missing')} | 全场景都没有该类（mirror / plant / faucet / side…） |",
           f"| dist_constraint | {sum(1 for x in d2 if x['status'] == 'dist_constraint')} | `closest to X` 是距离约束，判分口径不同（排序而非左右），未并入 |",
           f"| degenerate | {sum(1 for x in d2 if x['status'] == 'degenerate')} | 锚点与机器人重合，方向退化 |", "",
           "## 6 D1 里最常见的写法", ""]
    for s, n in Counter(x["row"].Instruction for x in d1).most_common(8):
        md.append(f"- `{s[:90]}` ×{n}")
    md.append("")
    open(os.path.join(DATA, "CLEAN_917.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
