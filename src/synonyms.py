# -*- coding: utf-8 -*-
"""
synonyms.py —— 物体类别的同义词表（单一真源）

为什么单独成文件：eval_locative.py 只需要这一张表，却因为
`from build_dataset_locative import SYNONYMS` 而把整个建库脚本（含 json_2.1.0
的依赖）拉进来 — 换机器跑评测时会莫名其妙 ImportError。

表的作用：ALFRED 的指令里常出现"钥匙/钥匙链""杯子/马克杯"这类换词，
如果纯文本基线只认物体名原文，会把本该命中的候选算成不命中，
从而不公平地削弱文本基线（实测影响 13.6% 样本）。
"""

SYNONYMS = {
    "cloth": ["rag", "towel"],
    "handtowel": ["towel"],
    "keychain": ["keys", "key"],
    "ladle": ["spoon"],
    "glassbottle": ["vase"],
    "tissuebox": ["tissues", "tissue"],
    "toiletpaper": ["roll"],
    "bread": ["loaf"],
    "mug": ["cup"],
    "cup": ["glass"],
    "laptop": ["computer"],
    "cellphone": ["phone", "smartphone", "mobile"],
    "remotecontrol": ["remote"],
    "television": ["tv"],
    "garbagecan": ["trash", "bin"],
    "lightswitch": ["switch"],
    "alarmclock": ["clock", "alarm"],
    "desklamp": ["lamp"],
    "floorlamp": ["lamp"],
    "basketball": ["ball"],
    "baseballbat": ["bat"],
    "creditcard": ["card"],
    "spraybottle": ["spray"],
    "statue": ["trophy", "figurine"],
    "pillow": ["cushion"],
    "spatula": ["turner"],
    "pot": ["saucepan"],
    "pan": ["skillet"],
    "bowl": ["dish"],
    "plate": ["dish"],
    "book": ["notebook"],
    "soapbottle": ["lotion"],
    "soapbar": ["soap"],
}
