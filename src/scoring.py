#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三层级联判分（用户口径，2026-11-24）

规则：
  1 方向：题面给了"锚点 + 方位" -> 只留指定侧候选，另一侧排除。
          指定侧一个都没有 -> 这步作废，全部候选进下一步。
  2 距离：题面明确写了 closest/nearest/furthest/farthest -> 池内取最近/最远；
          没写 -> 池内任取一个都对。
  3 文本：方向、距离都没有 -> 按文本选（类别 + 可验证属性）。
          文本只说了类别时，同类任取即符合文本。

判分对象：clean_spatial_ext.csv 中 Scorable==1 的 665 条（冻结 v1.0）。

  python eval_cascade.py
"""
from __future__ import annotations

import json
import math
import re

import pandas as pd

import _paths as _p

DATA = _p.DATA
RESULTS = _p.RESULTS
TOL_TIE = 0.02
UNPARSABLE_REF = {"edge", "center", "centre", "corner", "middle", "side", "end",
                  "back", "front", "left", "right", "top", "bottom", "you", "me"}

DIST_WORD = re.compile(r"\b(?:(second|third|fourth)\s+)?(closest|nearest|furthest|farthest)\b", re.I)
RANK = {"second": 2, "third": 3, "fourth": 4}

# 文本属性：颜色在 ALFRED 数据里不落字段（objectName 无颜色），只能靠状态谓语验证
STATE_ATTR = {"clean", "dirty", "heated", "cooled", "sliced", "toggled", "broken", "filled"}
COLOR_ATTR = {"white", "black", "blue", "brown", "red", "green", "yellow", "gray",
              "grey", "orange", "purple", "pink"}


def unit(v):
    n = math.hypot(v[0], v[1])
    return None if n < 1e-9 else (v[0] / n, v[1] / n)


def side_cos_of(cands, ax, ay, rx, ry, kind, side):
    fwd = unit((ax - rx, ay - ry))
    if fwd is None:
        return [None] * len(cands)
    left = (-fwd[1], fwd[0])
    out = []
    for c in cands:
        u = unit((c["x"] - ax, c["y"] - ay))
        if u is None:
            out.append(None)
        elif kind == "lr":
            v = u[0] * left[0] + u[1] * left[1]
            out.append(v if side == "left" else -v)
        elif kind == "behind":
            out.append(u[0] * fwd[0] + u[1] * fwd[1])
        else:
            out.append(-(u[0] * fwd[0] + u[1] * fwd[1]))
    return out


def parse_dist(instr):
    """返回 (kind, rank, ref)；kind in {min,max,None}，ref=None 表示默认以机器人为参照"""
    s = instr or ""
    m = DIST_WORD.search(s)
    if not m:
        return None, 0, None
    rank = RANK.get((m.group(1) or "").lower(), 1)
    kind = "min" if m.group(2).lower() in ("closest", "nearest") else "max"
    t = s[m.end():]
    # "farthest right" / "furthest left" 是最右/最左，属方位不属距离
    if re.match(r"\s+(right|left|back|front)\b", t, re.I):
        return None, 0, None
    mm = re.match(r"\s+most\b", t, re.I)
    if mm:
        t = t[mm.end():]
    if re.match(r"\s+(?:away\s+)?(?:to\s+|from\s+)?(you|me)\b", t, re.I):
        return kind, rank, "you"
    m2 = re.match(r"\s+(?:away\s+)?(?:to|from)\s+([a-z][a-z ']*?)"
                  r"(?=[,.;]|\s+(?:from|on|in|at|next|the)\b|$)", t, re.I)
    return kind, rank, (m2.group(1).strip().lower() if m2 else None)


def text_attrs(instr):
    toks = set(re.findall(r"[a-z]+", (instr or "").lower()))
    return toks & (STATE_ATTR | COLOR_ATTR)


def main():
    df = pd.read_csv(f"{DATA}/clean_spatial_ext.csv", encoding="utf-8-sig")
    cands = _p.load_json("loc_candidates.json")
    df = df[df.Scorable == 1].reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        same = [c for c in (cands.get(r.Task_ID) or []) if c["category"] == r.Target_Category]
        ids = [str(c["id"]) for c in same]
        gt = str(r.Target_Object)
        rec = dict(Sample_ID=r.Sample_ID, Band=r.Band, Instruction=r.Instruction,
                   Rel_Type=r.Rel_Type, Side=r.Side, Anchor=r.Anchor_Phrase,
                   Pool_N=len(same), Side_N=-1, Dir_Step="", Dist_Step="",
                   Text_Step="", Allowed_N=0, Target_In="", Fail="", FreePass="")
        if len(same) < 2 or gt not in ids:
            rec.update(Fail="no_pool")
            rows.append(rec)
            continue
        ti = ids.index(gt)

        # ---------- 1 方向 ----------
        has_dir = r.Band != "D1" and isinstance(r.Rel_Type, str)
        keep = list(range(len(same)))
        if has_dir:
            cs = side_cos_of(same, float(r.Anchor_X), float(r.Anchor_Y),
                             float(r.Robot_X), float(r.Robot_Y), r.Rel_Type, str(r.Side))
            in_side = [i for i, c in enumerate(cs) if c is not None and c > 0]
            rec["Side_N"] = len(in_side)
            if in_side:
                keep, rec["Dir_Step"] = in_side, "筛过"
            else:
                rec["Dir_Step"], rec["FreePass"] = "筛空→作废", "方向筛空回退"
        else:
            rec["Dir_Step"] = "不可算"

        # ---------- 2 距离 ----------
        kind, rank, ref = parse_dist(r.Instruction)
        if rec["Dir_Step"] == "不可算" and kind is None:
            # ---------- 3 文本 ----------
            at = text_attrs(r.Instruction)
            rec["Text_Step"] = ("有属性词:" + ",".join(sorted(at))) if at else "仅类别"
            rec["Dist_Step"] = "无"
            allowed = list(range(len(same)))
            rec["FreePass"] = rec["FreePass"] or "仅按类别选"
        elif kind is None:
            rec["Dist_Step"] = "无（池内任取）"
            allowed = keep
        else:
            ok_ref, ox, oy = False, None, None
            core = re.sub(r"^(the|a|an)\s+", "", ref or "").strip()
            first = core.split()[0] if core else ""
            if ref in (None, "you", "me", "yourself"):
                ok_ref, ox, oy = True, float(r.Robot_X), float(r.Robot_Y)
            elif first and first not in UNPARSABLE_REF:
                hit = [c for c in same if first in c["category"] or c["category"] in first]
                if hit:
                    ok_ref, ox, oy = True, float(hit[0]["x"]), float(hit[0]["y"])
            coord_ok = not (math.isnan(ox) or math.isnan(oy)) if ok_ref else False
            if not ok_ref or not coord_ok:
                rec["Dist_Step"] = (f"参照不可算({ref})→作废" if not ok_ref
                                    else "参照坐标缺失→作废")
                allowed = keep
            else:
                pool = [same[i] for i in keep]
                ds = [math.hypot(c["x"] - ox, c["y"] - oy) for c in pool]
                order = sorted(range(len(pool)), key=lambda i: ds[i])
                k = min(rank, len(order)) - 1
                dk = ds[order[k]]
                allowed = [keep[i] for i in order if abs(ds[i] - dk) <= TOL_TIE]
                rec["Dist_Step"] = f"{'最近' if kind == 'min' else '最远'}第{rank}（参照 {ref or 'you'}）"

        rec["Allowed_N"] = len(allowed)
        rec["Target_In"] = ti in allowed
        if not rec["Target_In"]:
            if rec["Dir_Step"] == "筛过" and ti not in keep:
                rec["Fail"] = "方向步骤排除"
            elif rec["Dist_Step"].startswith(("最近", "最远")):
                rec["Fail"] = "距离步骤排除"
            else:
                rec["Fail"] = "其他"
        rows.append(rec)

    det = pd.DataFrame(rows)

    def case(x):
        if x.Dir_Step == "不可算":
            if x.Dist_Step.startswith(("最近", "最远")):
                return "④ 无方向·有距离 → 距离+文本"
            return "⑤ 无方向·无距离 → 按文本选"
        if x.Dist_Step.startswith(("最近", "最远")):
            return "② 方向+距离 两级叠加"
        if x.Dist_Step.startswith("参照不可算"):
            return "③ 方向可算·距离参照不可算"
        return "① 方向可算·无距离 → 侧内任取"

    det["Case"] = det.apply(case, axis=1)
    det.to_csv(f"{RESULTS}/cascade_detail.csv", index=False, encoding="utf-8-sig")
    sc = det[det.Fail != "no_pool"].copy()
    sc["Free"] = sc.FreePass != ""

    out = []
    for k in sorted(sc.Case.unique()):
        g = sc[sc.Case == k]
        real = g[~g.Free]
        out.append(dict(情形=k, n=len(g), 命中=int(g.Target_In.sum()),
                        命中率=round(g.Target_In.mean() * 100, 1),
                        白送=int((g.Free & g.Target_In).sum()),
                        真筛过n=len(real),
                        真筛过命中率=round(real.Target_In.mean() * 100, 1) if len(real) else None,
                        平均池=round(g.Pool_N.mean(), 1), 平均允许=round(g.Allowed_N.mean(), 1)))
    real_all = sc[~sc.Free]
    out.append(dict(情形="合计", n=len(sc), 命中=int(sc.Target_In.sum()),
                    命中率=round(sc.Target_In.mean() * 100, 1),
                    白送=int((sc.Free & sc.Target_In).sum()),
                    真筛过n=len(real_all),
                    真筛过命中率=round(real_all.Target_In.mean() * 100, 1),
                    平均池=round(sc.Pool_N.mean(), 1), 平均允许=round(sc.Allowed_N.mean(), 1)))
    tbl = pd.DataFrame(out)
    tbl.to_csv(f"{RESULTS}/cascade_main.csv", index=False, encoding="utf-8-sig")

    fail = sc[~sc.Target_In].Fail.value_counts().to_dict()
    dir_s = sc[sc.Band != "D1"]
    side_empty = int((dir_s.Dir_Step == "筛空→作废").sum())
    one = int((dir_s.Side_N == 1).sum())
    n_dist = int(sc.Dist_Step.str.startswith(("最近", "最远")).sum())
    n_word = int(sum(parse_dist(x)[0] is not None for x in sc.Instruction))
    write_report(tbl, sc, dir_s, fail, side_empty, one, n_dist, n_word)

    print(tbl.to_string(index=False))
    print("\n失分归因:", fail)
    print(f"方向可算 {len(dir_s)}：筛空回退 {side_empty}、侧内唯一 {one}、距离真生效 {n_dist}")
    print("写出 cascade_detail.csv / cascade_main.csv / CASCADE_REPORT.md")


def write_report(tbl, sc, dir_s, fail, side_empty, one, n_dist, n_word):
    total, hit = len(sc), int(sc.Target_In.sum())
    free = int((sc.Free & sc.Target_In).sum())
    real = sc[~sc.Free]
    L = ["# 三层级联判分报告（方向 → 距离 → 文本）", "",
         "> 数据：`clean_spatial_ext.csv`（冻结 v1.0）中 `Scorable==1` 的 665 条。",
         "> 规则：第一步方向筛池（筛空则作废），第二步距离（没说则池内任取），第三步文本（前两步都没有时才用）。", "",
         "## 1 结论", "",
         f"- 可判分 **{total}** 条，按你的规则标注目标落在允许集合 **{hit} 条 = {hit/total*100:.1f}%**。",
         f"- 其中 **{free} 条是「白送」**：方向筛空后回退全池（{side_empty} 条）或只剩类别可选的"
         f"（{free-side_empty} 条），这些不是方法赢的，是题面没给约束。",
         f"- 扣掉白送，**真正被题面筛过一次的 {len(real)} 条命中率 "
         f"{real.Target_In.mean()*100:.1f}%** —— 这才是方法在起作用的区间。", "",
         "| 情形 | n | 命中 | 命中率 | 其中白送 | 真筛过 n | 真筛过命中率 | 平均池 | 平均允许集 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in tbl.iterrows():
        rk = "—" if pd.isna(r["真筛过命中率"]) else f"{r['真筛过命中率']}%"
        L.append(f"| {r['情形']} | {r['n']} | {r['命中']} | {r['命中率']}% | {r['白送']} | "
                 f"{r['真筛过n']} | {rk} | {r['平均池']} | {r['平均允许']} |")

    L += ["", "## 2 三步各有多少条真的生效", "",
          f"**第一步 方向**：可算 {len(dir_s)} 条（D2R 补齐锚点的也算在内）。",
          f"- 真的筛出了候选：{len(dir_s)-side_empty} 条（{((len(dir_s)-side_empty)/len(dir_s)*100):.1f}%）",
          f"- **筛空作废**：{side_empty} 条 —— 题面说的那一侧一个都没有，方向这步白说",
          f"- 筛完只剩 1 个：{one} 条（方向一步定死答案）",
          f"- 筛完平均剩 {dir_s.Allowed_N.mean():.1f} 个（原池 {dir_s.Pool_N.mean():.1f} 个）", "",
          f"**第二步 距离**：题面写 closest/nearest/furthest/farthest 的共 **{n_word} 条**"
          f"（{n_word/total*100:.1f}%），其中参照物能落到坐标、真正生效的 **{n_dist} 条**"
          f"（{n_dist/total*100:.2f}%），其余 {n_word-n_dist} 条参照是 edge/center 或颜色名，落不到坐标 → 作废。",
          "  - 也就是说 **98% 以上的题面根本没提距离**，按你的规则第二步不触发，侧内任取都对；",
          "  - ALFRED 的「最近」是标注者的习惯，不是写在题面里的约束——这是这份数据的固有特点。", "",
          "**第三步 文本**：只有前两步都没有时才用，共 "
          f"{int((sc.Case=='⑤ 无方向·无距离 → 按文本选').sum())} 条（D1 档）。",
          "  - 文本里除类别外，**颜色在 ALFRED 数据里不可验证**（`objectName` 不含颜色），"
          "状态（clean/heated/sliced）才可验证；这几条基本都是纯类别 → 按类别选，同类任取。", "",
          "## 3 失分在哪", "",
          "| 原因 | 条数 | 说明 |", "|---|---:|---|"]
    names = {"方向步骤排除": "标注目标落在题面**反侧**，第一步就排掉了（标注与定义矛盾，非方法错）",
             "距离步骤排除": "目标在指定侧，但不是题面要的最近/最远"}
    for k, v in fail.items():
        L.append(f"| {k} | {v} | {names.get(k,'')} |")

    # 2.1 数字对账
    dir_hit = int(sc[sc.Band != "D1"].Target_In.sum())
    dist_excl = int((sc.Fail == "距离步骤排除").sum())
    in_side = dir_hit - side_empty + dist_excl
    L += ["", "### 3.1 三个数字怎么对上（防止以后自己看不懂）", "",
          "| 环节 | 条数 | 说明 |", "|---|---:|---|",
          f"| 有方向约束 | {len(dir_s)} | A/B/C/D2R |",
          f"| 目标本来就在题面那一侧 | {in_side} | 方向筛过之后还活着（= 旧口径的 73.7%） |",
          f"| 目标在反侧、且该侧筛空 → 回退白送 | {side_empty} | 白送，进全池自动命中 |",
          f"| 方向真的把目标排掉 | {int((sc.Fail=='方向步骤排除').sum())} | 判错 |",
          f"| 距离把目标排掉 | {dist_excl} | 判错 |",
          f"| **级联合计命中（含 D1）** | **{int(sc.Target_In.sum())} / {total}** | {hit/total*100:.1f}% |", "",
          f"**对账**：{in_side} + {side_empty} - {dist_excl} = {dir_hit}（有方向部分的命中），"
          f"再加 D1 的 69 条按文本全中 = {int(sc.Target_In.sum())}。"
          "88.4% 里含 150 条白送，**别拿它和 73.7% 直接比**。", "",
          "## 4 与你原话的逐条对应", "",
          "| 你的说法 | 实现 | 数据里的实际情况 |", "|---|---|---|",
          "| 「椅子左边的距离最近的杯子，右边全排除」 | 第一步按机器人→锚点视线只留指定侧 | 两级同时生效只有 4 条 |",
          "| 「左边有杯子→进第二步；没有→这步作废，全部杯子进第二步」 | 侧池非空用侧池，空则回退全池 | 回退 %d 条 |" % side_empty,
          "| 「第二步有限制选最近/最远」 | 只在题面出现距离词时生效 | 只有 %d 条有距离词 |" % n_word,
          "| 「没限制就随便拿一个都对」 | 允许集合 = 整个侧池 | 588 条走这条 |",
          "| 「距离和方向都不存在就按文本选」 | 类别 + 可验证属性；只说了类别则同类任取 | 69 条，其中颜色不可验证 |", ""]

    # 4.1 走完两级的那几条
    comp = sc[sc.Case == "② 方向+距离 两级叠加"]
    L += ["### 4.1 走完两级的那 4 条（你说的「椅子左边最近的杯子」就是这种）", "",
          "| 题面 | 侧池 | 允许集 | 标注命中 | 卡在哪 |", "|---|---:|---:|---|---|"]
    for _, x in comp.iterrows():
        L.append(f"| `{str(x.Instruction)[:64]}` | {x.Side_N} | {x.Allowed_N} | "
                 f"{'✅' if x.Target_In else '❌'} | "
                 f"{x.Fail if isinstance(x.Fail, str) else '—'} |")
    L += ["", "这 4 条里 2 条标注不在允许集内：一条方向就错（最远的那颗番茄），"
          "一条方向对但不是最近（`nearest you` 那条）。真实数据里两级叠加只有 4 条，"
          "**别用它当主结论**。", "",
          "## 5 一个必须说清的问题：D1 的 69 条不是「没给约束」", "",
          "这 69 条我全查了，**100% 都带方位/位置词**，只是锚点算不出来：", "",
          "| 类型 | 条数 | 例子 |", "|---|---:|---|",
          "| 容器内方位（left most / back right / top left…） | 28 | `Pick up keys from far right couch cushion` |",
          "| 相对机器人（on your right） | 7 | `Pick up the sponge on your right` |",
          "| 修饰容器或画面（on the edge of the bed / turn right） | 34 | `Turn right and pick up the book that is on the edge of the bed` |", "",
          "按你「可以没有、但不能相违背」的原则，这三类更接近**未定（有约束但算不出）**，"
          "而不是「无约束」。若把它们当「无约束」白送 100%，会把 69 条虚假正确算进分母。", "",
          "**建议**：报告主表只用「方向可算」的 %d 条（真筛过 %d 条，%0.1f%%），"
          "D1 单独列为「题面有方位但锚点不可算，不判分」。" % (len(dir_s), len(real), real.Target_In.mean()*100), "",
          "## 6 落地文件", "",
          "| 文件 | 内容 |", "|---|---|",
          "| `cascade_detail.csv` | 逐样本：池/侧池大小、方向步结果、距离步结果、文本步、允许集合、对错、白送标记、失分环节 |",
          "| `cascade_main.csv` | 上面那张分层表 |", ""]
    open(f"{RESULTS}/CASCADE_REPORT.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
