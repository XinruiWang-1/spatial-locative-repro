# -*- coding: utf-8 -*-
"""
build_scene_anchors.py —— 按【场景】汇总家具/容器坐标（跨任务共享）

为什么可以这样合并
----------------
ALFRED 里同一 scene_num 是同一间屋子，家具(sink / stove / shelf / couch …)位置固定不动；
但 PDDL 只记录"本次计划里访问过"的容器，所以单任务视角下大量锚点缺失。
把同一场景所有任务的 PDDL 容器坐标汇总，就能互相补齐。

坐标精度
--------
已用"拾取物"校验：PDDL 坐标 /4 与 object_poses 的精确坐标**完全一致**（误差 0.000m，n=99），
所以不是 0.25m 栅格近似，可以直接用。

输出
----
scene_anchors.json : scene_num -> { 归一化名称: [[x,z], ...] }   （同名多实例会有多个坐标）
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re

import _paths as _p

DATA = _p.DATA
ROOT = os.path.join(DATA, "json_2.1.0")
OUT = os.path.join(DATA, "scene_anchors.json")


def norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).split("_")[0].lower())


def main():
    files = glob.glob(os.path.join(ROOT, "**", "traj_data.json"), recursive=True)
    print(f"traj_data.json: {len(files)}")
    scene = collections.defaultdict(lambda: collections.defaultdict(set))
    n_plan = 0
    for p in files:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        sn = d.get("scene", {}).get("scene_num")
        if sn is None:
            continue
        hp = d.get("plan", {}).get("high_pddl") or []
        if hp:
            n_plan += 1
        for h in hp:
            pa = h.get("planner_action", {}) or {}
            for key in ("coordinateReceptacleObjectId", "coordinateObjectId"):
                v = pa.get(key)
                if isinstance(v, list) and len(v) == 2 and isinstance(v[1], list) and len(v[1]) == 6:
                    scene[sn][norm(v[0])].add((round(v[1][0] / 4.0, 3), round(v[1][2] / 4.0, 3)))
    out = {str(k): {n: sorted(list(s)) for n, s in v.items()} for k, v in scene.items()}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    sizes = [len(v) for v in out.values()]
    print(f"有 plan 的任务 {n_plan}；场景 {len(out)}；每场景容器数 中位 {sorted(sizes)[len(sizes)//2]}")
    print(f"写出 {OUT}")


if __name__ == "__main__":
    main()
