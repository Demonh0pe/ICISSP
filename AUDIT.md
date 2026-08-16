# 会议版结果核查

对 `main` 分支上的 notebook 与 `result/*.csv` 的逐项核查,对照 ICISSP 会议版
`ICISSP_2026_178_CR.pdf`。

复跑核查:

```bash
python analysis/recompute_metrics.py --repo <main 分支的 checkout>
```

**所有结论都可从仓库内的文件复现。** 下面每一条都标注了证据位置。区分了三类:
「已证实」= 从代码或数据直接读出;「已量化」= 由数据反推且通过一致性校验;
「待查」= 有迹象但需要你确认。

---

## 摘要

| # | 问题 | 性质 | 影响 |
|---|---|---|---|
| 1 | 主指标不是 Macro-F1,是 FIXED 类的 binary F1 | 已证实 | 论文全部 F1 数值 |
| 2 | 修正为真 Macro-F1 后,核心显著性结论不成立 | 已量化 | 论文的中心论点 |
| 3 | 统计检验代码在仓库中不存在,p=0.026 无法复现 | 已证实 | 可复现性 |
| 4 | Table 5 的 Cumulative 一行与数据不符 | 已证实 | Discussion 4.1 的论证 |
| 5 | Table 4 声称用 37 窗口交集,实际报的是各方法自身全窗均值 | 已证实 | 方法间可比性 |
| 6 | Hybrid-CASR 有三个结果文件,论文用了最高的 | 待查 | 效应量可信度 |
| 7 | 季度窗口的基线 = Hybrid-CASR 双月的成绩 | 已证实 | 方法的贡献度 |
| 8 | **Replay-3P 和 OLoRA 每窗口重建适配器,不是持续学习** | 已证实 | Discussion 4.2 两个核心论断 |
| 9 | LB-CL 的实现里没有任何类别加权 | 已证实 | Table 2、方法描述 |
| 10 | Hybrid-CASR 的 70/30 缓冲区划分在代码里不存在 | 已证实 | 方法描述 |
| 11 | **语料以 PHP 为主(30%),C/C++ 仅占 24%** | 已证实 | 2.2.1、4.4 的语料描述 |

---

## 1. 主指标不是 Macro-F1 【已证实】

十个 notebook **无一例外**:

```python
label_map = {"VULNERABLE": 0, "FIXED": 1}      # 每个 notebook 都有
...
"f1": f1_score(labels, preds),                  # 没有 average= 参数
```

sklearn 的 `f1_score` 默认是 `average='binary', pos_label=1`。因为 `FIXED` 被映射成 1,
**记录下来的 `f1` 是 FIXED 类(已修复代码)的 F1**。

两层偏差:

1. 不是 Macro-F1——没有做任何跨类平均,而论文 2.4.1 节给出了公式
   `Macro-F1 = ½(F1₀ + F1₁)` 并声称这是主指标
2. 报的是**反类**——漏洞检测论文的头条数字,衡量的是模型识别"已修复代码"的能力

论文 2.2.2 节写的是 `Vulnerable instances: (f_pre, y = 1)`,**与代码里的映射恰好相反**。

证据:`2month-Hybrid_casr.py:68`、`2month_zero_shot.py:89`、`2month__LB_CL.py:134`、
`2month-Replay3p.py:100`、`2month_OLoRA.py:156`、`vuln_LLM(phi2)*.py` 各处。
另见 `2month-Hybrid_casr.py:415` 的列名归一化 `"macro_f1":"f1"` —— 错误标签正是从这里传播的。

### 数值修正

类别计数是数据属性,与方法无关,可从 Hybrid-CASR 那份 CSV 取得,再反推混淆矩阵。
**反推经过校验**:用重建的混淆矩阵重算 accuracy,与日志中的 accuracy 逐行比对,
除 zero-shot 外全部残差 ≤ 1.7e-16(浮点精度)。

| 方法 | 论文报的(FIXED 类 F1) | VULNERABLE 类 F1 | 真 Macro-F1 | 差 |
|---|---|---|---|---|
| Hybrid-CASR | 0.6669 | 0.6041 | **0.6355** | −0.031 |
| Cumulative | 0.6604 | 0.6002 | 0.6303 | −0.030 |
| Replay-1P | 0.6594 | 0.6003 | 0.6299 | −0.030 |
| CASR | 0.6591 | 0.5985 | 0.6288 | −0.030 |
| LB-CL | 0.6509 | 0.5978 | 0.6244 | −0.027 |
| Window-only | 0.6505 | 0.6008 | 0.6257 | −0.025 |
| Replay-3P | 0.6216 | 0.5542 | 0.5879 | −0.034 |
| OLoRA | 0.5997 | 0.5308 | 0.5653 | −0.034 |
| Zero-shot | 0.5039 | 0.4195 | 0.4617 | −0.042 |

方法排序基本稳定,仅 LB-CL 与 Window-only 互换(两者本就相差 0.0004)。

---

## 2. 修正指标后,核心显著性不成立 【已量化】

论文的中心论点是 Hybrid-CASR 相对 window-only 有**统计显著**的提升。

配对 Wilcoxon 检验,同窗口配对:

| 指标 | Δ | p | 结论 |
|---|---|---|---|
| FIXED 类 F1(论文实际用的) | +0.0164 | 0.0067 | 显著 |
| **真 Macro-F1**(论文声称用的) | +0.0098 | **0.1701** | **不显著** |
| **VULNERABLE 类 F1** | +0.0033 | **0.9735** | **完全无效应** |

最后一行值得单独看:**在漏洞检测真正关心的那个类上,Hybrid-CASR 相对基线的提升是
+0.003,p=0.97——即没有任何可检测的效应。**

Cumulative 也一样(p 从 0.025 变成 0.405)。负向结论不受影响:Replay-3P、OLoRA、
zero-shot 显著劣于基线,两种口径下都成立。

---

## 3. 统计检验代码不存在 【已证实】

在 `main` 分支全部文件(含 notebook 的 markdown 单元与输出)中检索:

```
wilcoxon   → 0 处
cliff      → 0 处
scipy      → 0 处
"0.026"    → 0 处
"0.103"    → 0 处
```

论文的 `p = 0.026` 与 `Cliff's δ = 0.103` 在提交的代码里没有出处。

我用 CSV 复现,在各种窗口集合和检验参数下得到的最接近值:

- 37 窗口交集(论文声称的口径):**p = 0.0159**
- 41 个共有窗口:**p = 0.0061**

都 < 0.05,所以"在论文口径下显著"这个结论方向一致,但 **0.026 这个具体数值复现不出来**。
统计分析可能在未提交的 notebook 里做过——需要你确认它的口径,尤其是用的是哪一列 F1。

---

## 4. Table 5 的 Cumulative 一行与数据不符 【已证实】

从 CSV 直接算出的 IBR:

| 方法 | | @1 | @3 | @5 | @6 |
|---|---|---|---|---|---|
| Replay-1P | 数据 / 论文 | 0.791 ✓ | 0.747 ✓ | 0.734 ✓ | 0.729 ✓ |
| Hybrid-CASR | 数据 / 论文 | 0.741 ✓ | 0.726 ✓ | 0.716 ✓ | 0.710 ✓ |
| CASR | 数据 / 论文 | 0.734 ✓ | 0.719 ✓ | 0.707 ✓ | 0.706 ✓ |
| Window-only | 数据 / 论文 | 0.713 ✓ | 0.701 ✓ | 0.689 ✓ | 0.693 ✓ |
| LB-CL | 数据 | 0.718 | 0.705 | 0.702 | 0.700 |
| | 论文 | 0.718 ✓ | 0.703 | 0.691 | 0.687 |
| Replay-3P | 数据 | 0.702 | **0.635** | **0.627** | **0.627** |
| | 论文 | 0.702 ✓ | 0.688 | 0.676 | 0.673 |
| OLoRA | 数据 | 0.612 | 0.606 | 0.608 | 0.598 |
| | 论文 | 0.612 ✓ | 0.598 | 0.587 | 0.584 |
| Zero-shot | 数据 | 0.493 | 0.491 | 0.486 | 0.486 |
| | 论文 | 0.493 ✓ | 0.493 | 0.493 | 0.493 |
| **Cumulative** | 数据 | **0.717** | **0.717** | **0.713** | **0.716** |
| | 论文 | **0.661** | 0.661 | 0.661 | 0.661 |

**IBR@1 对所有方法都精确吻合**,说明表是从真实数据起手的。但:

- **Cumulative 整行错了。** 论文填的 0.661 恰好等于它自己的**前向** F1(Table 4 的 0.661),
  真实的反向数值是 ~0.716。看起来是把前向数字误填进了反向行。
- Replay-3P 在 @3/@5/@6 上偏差最大(0.635 → 0.688)。

**这直接影响 Discussion 4.1 的论证。** 论文在那里构建了一个"悖论":cumulative
稳定性完美但绝对保持率(0.661)低于 replay 类方法。按真实数据,cumulative 的
IBR@1 = 0.717,高于 window-only(0.713)、Replay-3P(0.702)、OLoRA(0.612),
处于中游而非垫底。悖论的前提消失了。

---

## 5. Table 4 的窗口集合 【已证实】

论文 3.2 节:

> "The analysis focuses on the intersection of windows where all methods complete
> training" / "Table 4 reports ... on the 37 bi-monthly windows where all methods succeed"

37 这个数字是对的——九个方法的窗口交集确实是 37。但**表里报的数不是交集上的均值**:

| 方法 | 自身全窗均值 | 37 窗交集均值 | 论文 |
|---|---|---|---|
| Hybrid-CASR | 0.6671 | 0.6654 | **0.667** |
| Cumulative | 0.6608 | 0.6575 | **0.661** |
| LB-CL | 0.6511 | 0.6490 | **0.651** |
| Window-only | 0.6508 | 0.6492 | **0.651** |
| Replay-3P | 0.6219 | 0.6206 | **0.622** |
| Zero-shot | 0.5038 | 0.5121 | **0.504** |

论文的数字逐个对应「自身全窗均值」。各方法的窗口数并不相同(37 到 42 不等),
**即方法是在不同的评测集合上比较的**。窗口难度差异很大(F1 跨度 0.464–0.728),
所以这不是可以忽略的细节。

差值本身很小(≤0.003),改用交集不会改变排序——但论文的表述与实际计算不一致,
而且"排除失败窗口"这件事本身是非随机的:方法在哪些窗口上崩,与窗口难度相关。

---

## 6. 三份 Hybrid-CASR 结果 【待查】

| 文件 | 前向窗口数 | 均值 |
|---|---|---|
| `方法结果/metrics_log_hybrid_casr.csv` | 40 | 0.6544 |
| `metrics_log_hybrid_casr.csv`(根目录) | 39 | 0.6640 |
| `result/..._balanced__RUN_20250821-220210.csv` | 41 | **0.6671** ← 论文用的 |

跨文件极差 **0.0127**,而论文声称的提升是 **+0.016**。两者同量级。

如果这三个是同一配置的重复运行,那么**运行间噪声几乎等于效应本身**,单次运行得出的
显著性没有意义。若取 0.6544 那份,Hybrid-CASR 会低于 Replay-1P(0.659)和 CASR(0.659)。

需要你确认:这三份是同一配置的不同次运行,还是不同配置(比如加/不加 class balancing)?
**这决定了扩展版必须跑多少个种子。**

---

## 7. 季度窗口白拿了方法的全部收益 【已证实】

| 来源 | 配置 | F1 |
|---|---|---|
| Table 4 | Hybrid-CASR @ 双月 | 0.667 |
| Table 3 | window-only @ **季度** | 0.667 |
| Table 3 | window-only @ 6 月 / 12 月 | 0.669 |

已从 CSV 复核:`metrics_log_3month.csv` 均值 0.6672,`halfyear` 0.6694,`wholeyear` 0.6689。

**把重训频率从双月改成季度,不用任何方法,就能拿到和 Hybrid-CASR 相同的成绩,而且更省。**
Contribution 2 和 Contribution 3 报的是同一个数字 0.667。

论文现有的防守是"granularity affects which vulnerabilities are detected rather than how many",
但无数据支撑。

---

## 8. Replay-3P 和 OLoRA 结构上不是持续学习 【已证实】

统计每个 notebook 里适配器的处理方式:

| 方法 | `PeftModel.from_pretrained`(加载上一窗口) | 实际行为 |
|---|---|---|
| Hybrid-CASR | 1 | 继承适配器 ✓ |
| LB-CL | 1 | 继承适配器 ✓ |
| window-only 系列 | 1 | 继承适配器 ✓ |
| **Replay-3P** | **0** | **每窗口 `get_peft_model` 新建** |
| **OLoRA** | **0** | **每窗口 `get_peft_model` 新建** |

代码里的注释自己写明了:

```python
# 2month_OLoRA.py:182
# Initialize fresh base model for each window (not continual)
model = get_peft_model(base_model, peft_config)
```

```python
# 2month-Replay3p.py:133
# Load fresh base model for each window
model = get_peft_model(base_model, peft_config)
```

**这两个方法从不携带任何跨窗口学到的参数。** 每个窗口都从随机初始化的适配器重新开始。
OLoRA 只是把这个新适配器对历史方向做了正交化,Replay-3P 只是多喂了两个窗口的原始数据。

### 影响:Discussion 4.2 的两个论断都是误归因

论文写道:

> "Replay-3P (F1 = 0.622) underperforms despite retaining more historical data.
> This contradicts the assumption that larger buffers monotonically improve performance
> and suggests that, in rapidly evolving domains, controlled forgetting can be more useful
> than comprehensive retention."

> "OLoRA's relatively poor performance (F1 = 0.599) raises questions about the applicability
> of strict orthogonality constraints ... they appear overly rigid when vulnerability types
> overlap across time."

两个"反直觉发现"都有一个平凡得多的解释:**这两个方法把每个窗口学到的东西全扔了。**
它们垫底不是因为缓冲区太大或正交约束太死,而是因为它们根本没在做持续学习。

而且 Replay-3P 与 Hybrid-CASR 的对比同时混淆了两个变量:

| | 适配器 | 回放预算 |
|---|---|---|
| Hybrid-CASR | 继承 | 125 条采样 |
| Replay-3P | **重建** | 两个完整窗口 |

论文把差异全部归给"回放预算",但"适配器是否继承"这个变量没有被控制,而且几乎肯定是主因。

**这是扩展版必须做的一个实验**,而且很便宜:把这两个方法改成继承适配器再跑一遍。
`experiments/train.py --method olora --adapter inherit` 就能跑。三种可能的结果,
每一种都是可发表的发现:

- OLoRA 继承后追上来了 → 原结论错误,正交约束本身没问题
- 仍然垫底 → 原结论成立,但**这次有了对照**,论证才站得住
- 介于两者之间 → 可以把两个因素的贡献拆开

---

## 9. LB-CL 没有类别加权 【已证实】

论文 Table 2 和 2.3.3 节:

> "LB-CL (Label-Balanced Continual LoRA) modifies the training objective to account for
> class imbalance ... Class-weighted cross-entropy with weights inversely proportional to
> class frequency within each window is applied."

在 `2month__LB_CL.py` 中检索 `class_weight` / `CrossEntropyLoss` / `weight=` / `compute_loss`:
**命中 0 处。** 没有任何自定义损失函数,用的是 `Trainer` 的默认交叉熵。

实际实现的是 **QR 分解正交初始化的 LoRA**:

```python
# 2month__LB_CL.py:69
# === Custom LB-CL LoRA module with orthogonal initialization ===
# ... QR decomposition gives us orthogonal matrix Q
# 148: Main training loop (no replay, using LB-CL orthogonal LoRA)
```

也就是说 **LB-CL 和 OLoRA 都是正交类方法**,而论文把它们描述成两条不同的技术路线
(类别加权 vs 正交约束)。Table 2 的"Memory requirements: No extra memory beyond model
parameters"倒是碰巧对的,但机制描述完全不符。

---

## 10. Hybrid-CASR 的 70/30 划分不存在 【已证实】

论文 2.3.3 节:

> "The replay buffer is partitioned such that 70% of slots are filled by high-uncertainty
> examples selected by these CASR criteria, and the remaining 30% are drawn uniformly to
> maintain coverage."

实际代码(`select_topk_uncertain_balanced`)是:按类别各取熵最高的 `k//2` 条。
随机抽样**只在某一类样本数不足 `k//2` 时才触发**作为补足手段。正常情况下
100% 由熵选出,不存在 70/30。

论文 2.3.3 还提到 CASR 用 `τ = 0.7` 的置信度阈值;实现里用的是熵排序取 top-k,
没有阈值。

这一条不影响结论(方法确实是"不确定性 + 类别均衡"),但方法描述需要按实现改写,
否则别人复现不出来。

---

## 11. 语料以 PHP 为主,不是 C/C++ 【已证实】

从重建的 patch 统计文件路径扩展名(`data/splits_hunk/build_stats.json` 的
`languages_seen`,基于已抓取的样本):

| 语言 | hunk 数 | 占比 |
|---|---|---|
| **php** | 7733 | **30.3%** |
| c | 5338 | 20.9% |
| unknown(无扩展名/配置/文档) | 4568 | 17.9% |
| java | 1970 | 7.7% |
| javascript | 1700 | 6.7% |
| python | 1394 | 5.5% |
| ruby | 1188 | 4.7% |
| c++ | 802 | 3.1% |
| go / typescript / c# / scala / rust / objective-c / swift | 846 | 3.3% |

**C + C++ 合计 6140,占 24.0%。PHP 是最大的单一语言。**

论文 2.2.1:

> "In this study we focus on function-level instances from the dominant languages
> in the corpus (primarily C/C++), filtering out samples from other languages to
> keep the setting homogeneous."

论文 4.4:

> "The CVEfixes-based dataset predominantly covers C/C++ and Java, which may bias
> results towards these languages."

两处都与数据不符:

1. 语料**不是**以 C/C++ 为主,而是以 PHP 为主
2. **过滤根本没有发生**——第 9 条已证实代码里没有任何语言过滤,所有语言的样本都进了训练集

也就是说,实验是在一个 PHP 占三成的多语言混合体上做的,论文描述的却是同质的 C/C++ 设定。
4.4 节把"偏向 C/C++"列为效度威胁,而真实的威胁方向恰好相反。

**对扩展版的影响:**

- 不要加语言过滤。原实验没有,加了就不是复现;而且过滤到 C/C++ 只剩四分之一的数据。
- 2.2.1 和 4.4 的语料描述按实际分布重写。多语言本身不是缺点——**跨语言的持续学习反而是更有意思的设定**,只是必须如实说。
- `unknown` 占 17.9% 值得单独查一下:里面混着配置文件、文档、构建脚本。原管线把它们和代码一起塞进了样本(第 12 条的 markdown 例子就是这么来的)。

---

## 建议的处理顺序

**扩展版必须做的:**

1. **改用真 Macro-F1(或明确声明口径),重出所有表。** 不需要重跑训练——
   `analysis/recompute_metrics.py` 已经能从现有 CSV 算出。同时报 VULNERABLE 类的 F1,
   这才是漏洞检测该看的数。
2. **重做统计检验并把代码提交进仓库。** 按修正后的指标,Hybrid-CASR 的优势
   不显著(p=0.17)——这个结果要如实写。
3. **修正 Table 5 的 Cumulative 行**,并相应重写 Discussion 4.1 的"悖论"论证。
4. **跑多种子。** 三份结果文件的离散度说明单次运行不足以支撑 +0.016 的结论。
5. **补季度/半年粒度下的全方法对比**,正面回应第 7 条。这是审稿人最可能提的问题,
   而且你已有基础设施。
6. **把 Replay-3P 和 OLoRA 改成继承适配器重跑**(第 8 条),并据此重写 Discussion 4.2。
   这是现有发现里最便宜、回报最高的一个实验。
7. **按实际实现重写 Table 2 和 2.3.3 节的方法描述**(第 9、10 条)。LB-CL 不是类别加权,
   Hybrid-CASR 没有 70/30,CASR 没有 τ=0.7 阈值。

**关于会议版已发表的 p=0.026:** 它是既成事实,扩展版引用时不改。但扩展版**不应
在新分析里延续这个错误口径**——正确做法是说明两版口径不同,并给出修正后的数字。

**如果修正后 Hybrid-CASR 的优势确实不显著,这不等于论文没有价值。** 现有的诚实结论仍然成立:
选择性回放在计算效率上有真实优势(F1/min 高 24%,且远优于 cumulative 的 15.9 倍开销),
反向保持率也确实更高。把论点从"更准"调整为"同等精度下更省,且更抗遗忘",是站得住的。
