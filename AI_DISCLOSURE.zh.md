# AI 使用声明

本项目在开发过程中使用了 AI 编程助手（WorkBuddy，基于大语言模型）。为了研究诚信，以下是它参与的确切范围。

## AI 助手做了什么

| 方面 | 程度 |
|---|---|
| 为建立本仓库而做的代码重构 | 把脚本整理进 `src/`、把硬编码路径改为 `src/_paths.py`、重命名模块、编写 `scripts/*.sh` |
| 文档 | 起草本 README、`data/README.md`、`AI_DISCLOSURE.md`、YAML 配置文件，以及搬迁过程中补充的代码注释 |
| 排错 | 定位评测代码中的缺陷（例如误用 `Series.mode`，导致对象 repr 泄进结果表） |
| 语言 | 把中文工作笔记翻译成本 README 的英文表述 |

## AI 助手没有做什么

* 没有设计方法、没有选择基线、没有决定要跑哪些实验。
* 没有产出、修改或润色过任何一个报告数字。`results/` 里的每个数都由 `src/scoring.py` 和 `src/evaluate.py` 从冻结数据算出，可用 `bash scripts/run_all.sh` 重跑。
* 没有标注数据。标签是 ALFRED 的原始标注。

## 作者本人做过的验证

* 重构后的仓库与重构前的表格**逐字节一致**（`results/paper_T1_main.csv`、`paper_T7_ablation.csv`、`paper_T11_tests.csv`、`paper_detail.csv` 均与原始版本做过 `diff`）。
* `data/clean_spatial_ext.csv` 未被改动：sha256 与 `data/DATA_LOCK.json` 一致。
* 作者能解释每一个提交的脚本，并能重跑每一个实验。

## 第三方代码与模型

`src/` 中未复制任何第三方源代码。以下内容作为库、数据或模型使用，归属信息见 `README.md` 与 `LICENSE`：

* scikit-learn（`TfidfVectorizer`、余弦相似度）—— BSD-3-Clause
* pandas、numpy、matplotlib —— BSD-3-Clause / PSF
* ALFRED 数据集 —— MIT（https://github.com/askforalfred/alfred）
* Qwen2.5-7B-Instruct —— Apache-2.0（https://huggingface.co/Qwen/Qwen2.5-7B-Instruct）
