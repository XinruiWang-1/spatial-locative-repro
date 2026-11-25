#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_locative.py —— A 方案:从 high_descs 重建"空间定位型"实例级数据集

================================================================
【为什么要重建】
旧的 dataset_gt_all.csv 用整条 task_desc 当指令,问题有两个:
  1. ALFRED 的 task_desc 里 74% 的空间词是"放置目的地"
     ("put the bread in the fridge" 的 fridge 是终点,不告诉你面包在哪),
     这类样本算距离对"找目标"毫无意义;
  2. task_desc 本身很少编码目标位置,真正能验证"空间消歧"的样本
     全量只有 19 条 (0.27%)。

而 high_descs 是【分步指令】,每一步与 plan.high_pddl 严格逐条对齐:
  high_descs[i]  <->  high_pddl[i]
  "Pick up the pillow furthest from you on the right"
      <-> PickupObject, objectId=Pillow|-00.44|+00.74|+01.64
这样每一步都能拿到【精确的目标实例 + 该步的空间表达】,
天然构成"用空间线索在多实例中锚定目标"的样本。

【对齐关系已核验】
  len(high_descs) == len(high_pddl) - 1   (末尾是 NoOp/End)
  只有 PickupObject / ToggleObject 步带 objectId,格式 名字|x|y|z

【目标实例解析】
  objectId 里的名字是类别名(Pillow),而 object_poses 里是实例名(Pillow_568ecdf0)。
  同一实例名可能出现在多个坐标(=同一预制体的多个副本),所以用
  (类别前缀, x, z) 三重匹配回唯一实例,容差 TOL。

【输出】
  dataset_locative.csv    每条 = 一个"步骤级"样本
  loc_candidates.json     Task_ID -> 该场景可移动物体实例列表
  loc_anchors.json        Task_ID -> {锚点名: [x, z]}  (含 PDDL 家具 + robot)
  excluded_locative.txt   排除样本及原因
================================================================
"""

import os
import re
import json
import glob
import math
import argparse
from collections import Counter, defaultdict

TOL = 0.20          # 坐标匹配容差(米)
PLANNER_SCALE = 4.0  # 规划器坐标 -> 世界坐标的除数(= 1/gridSize,gridSize=0.25)
ARTICLES = {"the", "a", "an"}

# 关系模式(按优先级)。返回 (kind, anchor_phrase, direction, polarity)
#   polarity = +1 越近越可能, -1 越远越可能
REL_PATTERNS = [
    # 1) 相对某个物体:to the left of the plant
    (re.compile(r"\b(?:to the\s+)?(left|right)(?:\s+side)?\s+of\s+(?:the\s+|a\s+|an\s+)?([a-z]+(?:\s+[a-z]+){0,2})"),
     "lr_of", "lr"),
    # 2) 在某个物体上/里/旁:in the cabinet / on the counter / next to the sofa
    (re.compile(r"\b(on top of|inside of|inside|on|in|under|beneath|next to|near|close to|beside|in front of|behind|across from)\s+(?:the\s+|a\s+|an\s+)?([a-z]+(?:\s+[a-z]+){0,2})"),
     "rel", "none"),
    # 3) 相对机器人:furthest from you / closest to you
    (re.compile(r"\b(furthest|farthest|most distant)\s+(?:away\s+)?from\s+(?:you|me)\b"), "far_agent", "none"),
    (re.compile(r"\b(closest|nearest)\s+(?:to\s+)?(?:you|me)\b"), "near_agent", "none"),
    # 4) 相对机器人的方位:on the right / to the left
    (re.compile(r"\b(?:on|to)\s+the\s+(left|right)\b"), "lr_agent", "lr"),
    # 5) 从某处取:pick up the bread from the counter
    (re.compile(r"\bfrom\s+(?:the\s+|a\s+|an\s+)?([a-z]+(?:\s+[a-z]+){0,2})"), "from", "none"),
]

# "turn on the lamp" / "switch on the light" 里的 on 是动词短语,不是空间关系
TURN_ON_GUARD = re.compile(
    r"\b(?:turn|turning|switch|switching|flip|flipping)\b(?:\s+\w+){0,3}?\s+on\b", re.I)

# 这些词做参照物没有意义
STOP_REFS = {
    "light", "hand", "it", "one", "same", "middle", "center", "room", "you",
    "left", "right", "side", "top", "there", "them", "this", "that", "me",
    "your", "my", "front", "back", "end", "way", "around", "over",
}

FURN_WORDS = {
    "stove", "counter", "countertop", "sink", "fridge", "refrigerator", "microwave",
    "table", "desk", "dresser", "cabinet", "shelf", "shelves", "bed", "sofa",
    "couch", "armchair", "chair", "lamp", "desklamp", "floorlamp", "toilet",
    "bathtub", "toaster", "coffeemachine", "garbagecan", "drawer", "ottoman",
    "sidetable", "tvstand", "safe", "washingmachine", "dishwasher", "stairs",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def cat_of(name: str) -> str:
    return str(name).split("_")[0].strip().lower()


def clean_phrase(p: str) -> str:
    return " ".join(w for w in str(p).split() if w not in ARTICLES)


# ALFRED 内部类别名与人类指令用词不一致(keychain 说 "keys"、cloth 说 "rag"…)。
# 下表从 train 上自动挖掘 + 人工剔除家具词(desk/bed/counter 是锚点不是同义词)。
# 目的:让【文本基线】不被不公平地削弱。若某同义词恰好也是本样本的候选类别,
# 由调用方剔除,避免制造新的歧义。
from synonyms import SYNONYMS   # 表定义在 synonyms.py，避免两份漂移


def cat_mentioned(instr: str, cat: str) -> bool:
    """指令里是否用词面线索指到了该类别(含同义词)——用于标注数据质量"""
    ni = norm(instr)
    nc = norm(cat)
    if nc and nc in ni:
        return True
    for w in re.findall(r"[a-z]+", str(instr).lower()):
        if len(w) >= 3 and w in nc:
            return True
    low = str(instr).lower()
    for syn in SYNONYMS.get(cat, []):
        if re.search(r"\b" + re.escape(syn), low):
            return True
    return False


def find_anchor(phrase: str, anchors: dict):
    """把参照物短语匹配到锚点名(去空格后双向子串,长度≥3);返回 (name, xy) 或 (None, None)"""
    comp = norm(clean_phrase(phrase))
    if not comp or comp in STOP_REFS or len(comp) < 3:
        return None, None
    best = None
    for a in anchors:
        if a == "robot" or len(a) < 3 or a in STOP_REFS:
            continue
        if comp == a or comp in a or a in comp:
            if best is None or len(a) > len(best):
                best = a
    if best:
        return best, tuple(anchors[best])
    return None, None


def parse_relation(text: str, anchors: dict):
    """
    解析空间表达。返回 dict 或 None:
      kind / anchor_name / direction / polarity / anchor_phrase / resolved
    """
    raw = str(text).lower()
    guarded = TURN_ON_GUARD.sub(" ", raw)

    for pat, kind, dtype in REL_PATTERNS:
        for m in pat.finditer(guarded):
            direction = None
            polarity = 1
            if kind == "lr_of":            # 2 组:(left|right) + 参照物
                direction = m.group(1)
                phrase = m.group(2)
            elif kind == "lr_agent":       # 1 组:(left|right)
                direction = m.group(1)
                phrase = "you"
            elif kind == "far_agent":      # 1 组:furthest/farthest
                polarity, phrase = -1, "you"
            elif kind == "near_agent":     # 1 组:closest/nearest
                polarity, phrase = 1, "you"
            elif kind == "from":           # 1 组:参照物
                phrase = m.group(1)
            else:                          # rel:2 组,关系词 + 参照物
                phrase = m.group(2)

            if kind in ("far_agent", "near_agent", "lr_agent"):
                # 相对机器人 → 用 robot 锚点
                if "robot" in anchors:
                    return {"kind": kind, "anchor_name": "robot",
                            "direction": direction, "polarity": polarity,
                            "anchor_phrase": phrase, "resolved": True}
                continue

            name, _xy = find_anchor(phrase, anchors)
            if name:
                return {"kind": kind, "anchor_name": name, "direction": direction,
                        "polarity": polarity, "anchor_phrase": clean_phrase(phrase),
                        "resolved": True}
            # 解析到但拿不到坐标:记下第一个,便于统计"为什么失败"
            if kind == "lr_of":
                return {"kind": kind, "anchor_name": None, "direction": direction,
                        "polarity": polarity, "anchor_phrase": clean_phrase(phrase),
                        "resolved": False}
    return None


def build_scene(d: dict):
    """返回 (instances, anchors)
    instances: 去重后的可移动物体实例列表 [{'id','category','x','y'}]
    anchors:   {归一化名: [x, z]}  ← 可移动物体 + PDDL 家具包围盒 + robot
    """
    seen = {}
    instances = []
    for o in d["scene"].get("object_poses", []):
        name = o.get("objectName")
        if not name:
            continue
        p = o.get("position", {})
        x, z = round(float(p.get("x", 0.0)), 3), round(float(p.get("z", 0.0)), 3)
        key = (name, x, z)
        if key in seen:
            continue
        seen[key] = True
        instances.append({"id": f"{name}|{x:+.2f}|{z:+.2f}",
                          "name": name, "category": cat_of(name), "x": x, "y": z})

    anchors = {}
    for inst in instances:
        anchors.setdefault(norm(inst["category"]), [inst["x"], inst["y"]])
    # 家具:PDDL 坐标是【规划器坐标系】,必须换算才能和 object_poses 对齐:
    #   1) 数值顺序是 [X, X, Z, Z, Y, Y](不是 XYZ,是 X/Z/高度Y)
    #   2) 需要除以 gridSize(=0.25),即乘 4;已用 400 个任务核验,中位误差 6mm
    for h in d.get("plan", {}).get("high_pddl", []):
        pa = h.get("planner_action", {}) or {}
        for key in ("coordinateReceptacleObjectId", "coordinateObjectId"):
            v = pa.get(key)
            if isinstance(v, list) and len(v) == 2 and isinstance(v[1], list) and len(v[1]) == 6:
                v6 = v[1]
                anchors.setdefault(norm(v[0]), [round(v6[0] / PLANNER_SCALE, 3),
                                                round(v6[2] / PLANNER_SCALE, 3)])
    # 机器人初始位置
    ia = d["scene"].get("init_action", {}) or {}
    if "x" in ia and "z" in ia:
        anchors["robot"] = [round(float(ia["x"]), 3), round(float(ia["z"]), 3)]
    return instances, anchors


def resolve_target(oid: str, instances: list):
    """objectId='Pillow|-00.44|+00.74|+01.64' -> 匹配到唯一实例"""
    parts = oid.split("|")
    if len(parts) != 4:
        return None
    cat = cat_of(parts[0])
    try:
        tx, tz = float(parts[1]), float(parts[3])
    except ValueError:
        return None
    best, best_d = None, 1e9
    for inst in instances:
        if inst["category"] != cat:
            continue
        dist = max(abs(inst["x"] - tx), abs(inst["y"] - tz))
        if dist < best_d:
            best, best_d = inst, dist
    if best is not None and best_d <= TOL:
        return best
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="json_2.1.0")
    ap.add_argument("--splits", default="train,valid_seen,valid_unseen")
    ap.add_argument("--out-dataset", default="dataset_locative.csv")
    ap.add_argument("--out-cand", default="loc_candidates.json")
    ap.add_argument("--out-anchor", default="loc_anchors.json")
    ap.add_argument("--out-excluded", default="excluded_locative.txt")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    records, cand_map, anchor_map = [], {}, {}
    excluded = defaultdict(list)
    rel_kind = Counter()
    step_type = Counter()

    for sp in splits:
        files = sorted(glob.glob(os.path.join(args.root, sp, "**", "traj_data.json"),
                                 recursive=True))
        print(f"[{sp}] 轨迹 {len(files)} 个")
        for f in files:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            tid = d.get("task_id")
            if not tid:
                continue
            anns = d.get("turk_annotations", {}).get("anns", [])
            if not anns:
                excluded["无 turk_annotations"].append(tid)
                continue
            descs = anns[0].get("high_descs") or []
            plan = (d.get("plan") or {}).get("high_pddl") or []
            if not descs or not plan:
                excluded["无 high_descs 或 high_pddl"].append(tid)
                continue

            instances, anchors = build_scene(d)
            if len(instances) < 2:
                excluded["场景可移动物体 < 2"].append(tid)
                continue
            cand_map[tid] = [{"id": i["id"], "category": i["category"],
                              "x": i["x"], "y": i["y"]} for i in instances]
            anchor_map[tid] = anchors

            scene = d.get("scene", {}).get("floor_plan", "")
            for idx, (desc, step) in enumerate(zip(descs, plan)):
                pa = step.get("planner_action", {}) or {}
                act = pa.get("action", "")
                step_type[act] += 1
                if act not in ("PickupObject", "ToggleObject"):
                    continue
                oid = pa.get("objectId")
                if not oid:
                    excluded["步骤无 objectId"].append(f"{tid}#{idx}")
                    continue
                tgt = resolve_target(oid, instances)
                if tgt is None:
                    excluded[f"目标实例无法按坐标匹配({cat_of(oid.split('|')[0])})"].append(
                        f"{tid}#{idx}")
                    continue

                rel = parse_relation(desc, anchors)
                if rel and rel["resolved"]:
                    rel_kind[f"{rel['kind']}_p{rel['polarity']}"] += 1
                elif rel:
                    rel_kind[f"{rel['kind']}_unresolved"] += 1

                n_same = sum(1 for i in instances if i["category"] == tgt["category"])
                ax, ay = (anchors[rel["anchor_name"]]
                          if rel and rel["resolved"] else (float("nan"), float("nan")))

                # ---- 标注:该空间描述是否【唯一指向】目标(而非同名实例都对得上)----
                d_target = d_second = float("nan")
                rank_same = 0
                if rel and rel["resolved"]:
                    same = [i for i in instances if i["category"] == tgt["category"]]
                    ds = sorted((math.hypot(i["x"] - ax, i["y"] - ay), i["id"]) for i in same)
                    d_target = round(ds[0][0], 3)
                    for k, (dd, iid) in enumerate(ds, 1):
                        if iid == tgt["id"]:
                            rank_same = k
                            d_target = round(dd, 3)
                            break
                    rest = [dd for dd, iid in ds if iid != tgt["id"]]
                    d_second = round(rest[0], 3) if rest else float("nan")
                    if rel["polarity"] < 0:      # "furthest": 越远越可能,排名要反过来
                        ds_rev = list(reversed(ds))
                        for k, (dd, iid) in enumerate(ds_rev, 1):
                            if iid == tgt["id"]:
                                rank_same = k
                                break
                        d_target = round(ds_rev[0][0], 3)
                        d_second = round(ds_rev[1][0], 3) if len(ds_rev) > 1 else float("nan")

                records.append({
                    "Sample_ID": f"{tid}#{idx}",
                    "Task_ID": tid,
                    "Split": sp,
                    "Step_Idx": idx,
                    "Step_Action": act,
                    "Instruction": desc,
                    "Scene": scene,
                    "Target_Category": tgt["category"],
                    "Target_Object": tgt["id"],
                    "Target_X": tgt["x"],
                    "Target_Y": tgt["y"],
                    "Num_Same_Cat": n_same,
                    "Num_Cand": len(instances),
                    "Has_Rel": int(bool(rel and rel["resolved"])),
                    "Rel_Kind": rel["kind"] if rel else "",
                    "Rel_Polarity": rel["polarity"] if rel else "",
                    "Rel_Direction": (rel["direction"] or "") if rel else "",
                    "Anchor_Phrase": rel["anchor_phrase"] if rel else "",
                    "Anchor_Name": (rel["anchor_name"] or "") if rel else "",
                    "Anchor_X": ax,
                    "Anchor_Y": ay,
                    # 判别力标注:排名/目标到锚点距离/次优同类实例距离
                    "Rank_Same": rank_same,
                    "D_Target": d_target,
                    "D_Second": d_second,
                    # 指令里是否出现目标类别(含同义词由 eval 处理)
                    "Cat_Mentioned": int(cat_mentioned(desc, tgt["category"])),
                })

    # ---------------- 写出 ----------------
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(args.out_dataset, index=False, encoding="utf-8-sig")
    with open(args.out_cand, "w", encoding="utf-8") as fh:
        json.dump(cand_map, fh, ensure_ascii=False)
    with open(args.out_anchor, "w", encoding="utf-8") as fh:
        json.dump(anchor_map, fh, ensure_ascii=False)
    with open(args.out_excluded, "w", encoding="utf-8") as fh:
        fh.write("# 被排除的步骤及原因(可人工复核)\n")
        for reason, items in sorted(excluded.items(), key=lambda x: -len(x[1])):
            fh.write(f"\n[{reason}]  共 {len(items)} 条\n")
            for it in items[:200]:
                fh.write(f"  {it}\n")

    # ---------------- 统计 ----------------
    print("\n" + "=" * 68)
    print(f"样本总数: {len(df)}   任务数: {df['Task_ID'].nunique()}")
    print("=" * 68)
    print("\n按 split:")
    for sp, g in df.groupby("Split"):
        print(f"  {sp:14s} {len(g):5d} 条 | 场景 {g['Scene'].nunique():3d} "
              f"| 任务 {g['Task_ID'].nunique():4d} | 带空间线索 {int(g['Has_Rel'].sum())}")
    print("\n步骤类型:", dict(step_type))
    print(f"\n带可解析空间线索: {int(df['Has_Rel'].sum())} "
          f"({df['Has_Rel'].mean()*100:.1f}%)")
    print("关系类型分布:")
    for k, v in rel_kind.most_common(20):
        print(f"  {k}: {v}")
    print("\n多实例(目标类别≥2 个实例):")
    multi = df[df["Num_Same_Cat"] >= 2]
    print(f"  {len(multi)} 条 ({len(multi)/len(df)*100:.1f}%)")
    print(f"  其中带空间线索: {int(multi['Has_Rel'].sum())} 条  << 核心实验集")
    print("\n>> 输出:", args.out_dataset, args.out_cand, args.out_anchor, args.out_excluded)


if __name__ == "__main__":
    main()
