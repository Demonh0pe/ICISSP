# 5090 操作手册

从零到跑出结果。每一步都有验收标准——**不通过就停下,不要往下走**,
按量计费的机器上带着坏环境或错数据往下跑,只会烧钱买一堆废结果。

---

## 步骤 1:环境(约 10 分钟)

```bash
cd /root/autodl-tmp
git clone https://github.com/Demonh0pe/ICISSP.git && cd ICISSP
git checkout claude/autodl-experiment-migration-3ak8my
bash env/setup.sh
```

**验收:最后一行是 `READY`。**

5090 是 sm_120。cu124 及更早的 torch wheel 能 import、`cuda.is_available()` 返回 True,
然后在第一次真正的 kernel launch 时炸 `no kernel image is available for execution on the
device`——那时候机时已经烧掉了。自检强制跑了真实的 matmul、bf16 反传和 SDPA,
把这个错误提前到第一分钟。

`download.pytorch.org` 慢或不通就换源:

```bash
TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cu128 bash env/setup.sh
```

---

## 步骤 2:确认数据集身份(约 2 分钟)

**这一步回答"我手里这份还是不是当初那份"。**

```bash
python data/verify_dataset.py --data-dir <你的 temporal_splits_by_time 目录>
```

原理:Hybrid-CASR 那次运行把每个窗口的类别计数写进了 CSV,而类别计数只取决于数据。
脚本用 notebook 里一模一样的过滤条件(`len(prompt) > 10`、
`response in {VULNERABLE, FIXED}`)重新数一遍,逐窗口比对。

**三种结果:**

| 输出 | 含义 | 该怎么做 |
|---|---|---|
| `matches ... on every reference window` | 就是当初那份 | 直接往下,新结果可与会议版对比 |
| `mismatched: <窗口列表>` | 同名窗口内容不同 | 很可能去重或语言过滤改过了。**新跑的一切都是新实验,不是复现**,论文里要写明 |
| `missing: <窗口列表>` | 缺文件 | 同上 |

数据丢了也不是死路——`data/prepare_cvefixes.py` 能从 `CVEfixes.db` 重建。
但重建出来的**几乎肯定不会完全一致**(CVEfixes 自己在更新),所以那种情况下
扩展版的定位就是"新数据上的独立验证",而不是"会议版的延续"。

---

## 步骤 3:冒烟测试(约 5 分钟,不碰 GPU)

**在开任何正式训练之前跑这个。** 用一个极小的模型走完整条链路:数据加载、
回放采样、训练、前向/反向评测、CSV 落盘。

```bash
python experiments/test_common.py       # 纯逻辑,42 项断言
python experiments/train.py --method hybrid-casr \
  --data-dir <数据目录> --out runs/smoke --smoke
```

**验收:`runs/smoke/metrics.csv` 生成,且每个窗口都有 `forward` 行。**

冒烟测试用 `hf-internal-testing/tiny-random-gpt2`(几 MB),所以数字毫无意义——
它验证的是管线不崩,不是效果。

---

## 步骤 4:单窗口试跑(约 15 分钟,真模型)

```bash
python experiments/train.py --method window-only \
  --data-dir <数据目录> --out runs/probe --limit-windows 3
```

**验收:**

- 不 OOM(5090 是 32GB,phi-2 + LoRA + FP32 batch 32 应该够;不够就降 `--batch-size 16`)
- `macro_f1` 落在 0.55–0.70 区间——跟会议版量级对得上
- 记下单窗口耗时,乘以 41 就是单次完整运行的预估

**这一步决定了你跑不跑得完。** 如果单窗口要 5 分钟,那 41 窗 × 9 方法 × 3 种子
= 92 小时,三天做不完,必须砍。

---

## 步骤 5:正式跑

按价值排序。**从上往下跑,余额或时间不够就在任意一行停下**——
前面跑完的部分已经能支撑扩展版了。

### 5.1 优先级最高:修正 OLoRA 和 Replay-3P(约 2 次运行)

`AUDIT.md` 第 8 条:这两个方法在会议版里每窗口重建适配器,根本不是持续学习。
论文却据此得出了两个"反直觉发现"。

```bash
D=<数据目录>
python experiments/train.py --method olora     --data-dir $D --out runs/main --adapter inherit
python experiments/train.py --method replay-3p --data-dir $D --out runs/main --adapter inherit
```

**为什么排第一:两次运行就能定性地修正 Discussion 4.2 的两个核心论断。**
性价比高于其他任何实验。

### 5.2 多种子重跑(9 方法 × 3 种子)

三份 Hybrid-CASR 结果文件的极差是 0.0127,而论文声称的提升是 +0.016——
**噪声和效应同量级**。单次运行的显著性没有意义。

```bash
for seed in 42 43 44; do
  for m in window-only replay-1p casr hybrid-casr lbcl olora replay-3p; do
    python experiments/train.py --method $m --data-dir $D --out runs/main --seed $seed
  done
done
```

`cumulative` 单独跑,它慢 15.9 倍:

```bash
python experiments/train.py --method cumulative --data-dir $D --out runs/main --seed 42
```

`zero-shot` 不训练,几分钟:

```bash
python experiments/train.py --method zero-shot --data-dir $D --out runs/main
```

### 5.3 粒度对比(回应 AUDIT 第 7 条)

季度窗口的 window-only 基线(0.667)等于 Hybrid-CASR 双月的成绩(0.667)。
审稿人只要把 Table 3 和 Table 4 并排看就会问。

```bash
for g in 3 6; do
  for m in window-only hybrid-casr; do
    python experiments/train.py --method $m --granularity $g --data-dir $D --out runs/gran
  done
done
```

### 5.4 换模型(优先级最低)

原计划的 Qwen 换模型排在最后,理由在 `PLAN.md`:它回应的是"单一架构"这个次要
limitation,而上面几项回应的是**结论本身是否成立**。

**注意别写错动机:** 论文 4.4 说 phi-2 有预训练污染风险(2023 年 12 月发布,
评测覆盖 2018–2023)。Qwen2.5 的知识截止**更晚**,换过去污染只会更重。
换模型的正当理由是架构多样性,不是污染。

```bash
python experiments/train.py --method hybrid-casr --model Qwen/Qwen2.5-Coder-1.5B \
  --data-dir $D --out runs/qwen
```

---

## 步骤 6:出结果

```bash
python analysis/recompute_metrics.py --repo <main 分支的 checkout>   # 会议版旧数据重算
```

新跑的 `runs/main/metrics.csv` 已经直接包含 `macro_f1`、`f1_vulnerable`、`f1_fixed`,
不需要反推。`f1_binary_pos1_LEGACY` 一列保留了会议版的口径,便于新旧对齐。

---

## 长跑注意

```bash
# SSH 断开不会杀掉训练
nohup python experiments/train.py ... > runs/log_hybrid.txt 2>&1 &
tail -f runs/log_hybrid.txt
```

- **每个窗口都会立即 flush 写盘**,中途挂掉前面的结果不丢
- 默认只保留上一个窗口的适配器(续训需要),要全留加 `--keep-adapters`,但吃磁盘
- AutoDL 按**开机时间**计费,不是按 GPU 利用率。**跑完立刻关机**
- 盯余额。跑到一半停机比跑失败更麻烦,checkpoint 可能是半截的

---

## 一致性纪律

**同一次对比里的所有运行,必须同卡、同超参、同 dtype。**

- `--dtype fp32` 是会议版的设置(论文 Table 1 写明 FP32),默认值
- `bf16` 更快,但数值不同。**要用就整块都用**,不要在一次对比里混
- A100 → 5090 这个硬件变更要在论文里注明,并说明"块内一致、块间不做绝对数值直接比较"
