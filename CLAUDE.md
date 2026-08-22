# CLAUDE.md — Python 脚本编写与修改优化规则

> 本文件是仓库内所有 Python 脚本的**通用强制编写规范**，适用于任何使用 CPU 数值计算（仿真、实验、数据分析）的项目。
> 目标：在无 GPU 的环境下，通过分层优化策略将计算速度提升 10-100 倍，同时**保证既有结果的可复现性不被破坏**。
>
> 本规范只约束两类动作：
> - **新增脚本**：遵守第 1-5 章（类型识别、优化、编码规范）
> - **修改既有脚本**：遵守第 1-5 章 + 第 6 章安全阀门
>
> 动手前必须先按第 1 章识别脚本类型；修改既有脚本前必须先执行第 6.1 节预检查；否则不得动手。

---

## 1. 核心原则：Python 做胶水，计算下沉编译层

```
┌─────────────────────────────────────────────┐
│  上层 Python（胶水层，禁止密集数值循环）       │
│  · 参数遍历、实验调度、数据集管理              │
│  · 统计、绘图、CSV/JSON 输出、日志             │
├─────────────────────────────────────────────┤
│  底层编译层（所有 CPU 密集数值循环必须在此）   │
│  · Numba @njit + NumPy 向量化                │
│  · 超大阵列时可扩展 Rust/PyO3（需评估）        │
└─────────────────────────────────────────────┘
```

**下沉判断标准**（须同时满足全部条件；任一不满足则保持 Python 原生写法，禁止加 `@njit`、禁止强行向量化）：

1. 循环体**仅**含数值运算与 NumPy 数组操作；
2. 循环体**不含**第 2.2 节"@njit 禁用调用清单"中的任何调用；
3. 循环规模 N > 100，或含时序依赖（如波形迭代）无法向量化；
4. 该函数不属于 FIG（绘图）/ ML（机器学习）类脚本。

调度循环、IO 循环、绘图循环、ML 训练循环天然豁免，保持 Python 原生写法。

---

## 2. 脚本类型识别与优化规则矩阵

动手前先识别脚本类型，写入脚本头部 docstring 的 `Type:` 字段。类型决定哪些优化规则适用。

### 2.1 类型判定

| 类型 | 判定信号 | 优化定位 |
|------|----------|----------|
| **CORE** | 被其他脚本 `import` 且含物理模型/核心算法公式 | 优化优先级最高，@njit 必加；修改影响向下游传播，受第 6 章严格约束 |
| **PAPER** | 输出 CSV/JSON 被论文表格/图直接引用 | 核心数值循环可加 @njit，调度层不可；修改受第 6 章严格约束 |
| **FIG** | 仅 `import matplotlib` 读 CSV/解析式画图，无仿真循环 | **禁用 @njit 与多进程** |
| **ML** | `import torch` / `import sklearn` / `import tensorflow` | **禁用 @njit**；多进程仅用于独立 MC trial 外层，不可包裹训练循环 |
| **EXPLORE** | 以上都不匹配 | 按需优化，无强制要求 |

信号冲突时按 **ML > CORE > PAPER > FIG > EXPLORE** 优先级判定，并在 docstring 注明混合性质；无法判定时停止并向用户询问。

### 2.2 @njit 禁用调用清单

循环体出现以下任意一项，立即放弃 @njit（保留 Python 原生，或将纯数值部分抽离为独立 @njit 函数）：

| 类别 | 禁用调用 |
|------|----------|
| 机器学习 | `sklearn.*`、`torch.*`、`tensorflow.*`、`xgboost.*`、`lightgbm.*` |
| 绘图 | `matplotlib.*`、`seaborn.*`、`plotly.*`、`bokeh.*` |
| 数据框 | `pandas.*` |
| IO 与序列化 | `open()`、`csv.*`、`json.*`、`pickle.*`、`np.savez`、`Path.*` |
| 动态容器 | 原生 `dict`、循环内动态扩张的 `list.append` |
| 字符串 | 拼接、`f-string`、`str.format`、正则 `re.*` |
| 高阶函数 | `lambda`、闭包捕获外部变量、`map`/`filter` 套 callable |
| 调试/动态分发 | `print`、`logging.*`、`pdb`、`getattr`、`eval`、`exec` |

> 例外：`scipy.special` 的部分 ufunc（`expit`、`gammaln`）numba 支持；`scipy.stats`、`scipy.optimize` 不支持。

### 2.3 规则适用矩阵

| 规则 | CORE | PAPER | FIG | ML | EXPLORE |
|------|:----:|:-----:|:---:|:--:|:-------:|
| @njit（核心数值循环） | ✅ 必加 | ✅ 视情况 | ❌ 禁 | ❌ 禁 | ⚪ 可选 |
| NumPy 向量化 | ✅ 必加 | ✅ 必加 | ⚪ 可选 | ⚪ 可选 | ⚪ 可选 |
| multiprocessing.Pool（独立 MC trial） | N/A | ✅ 必加 | ❌ 禁 | ⚠ 仅外层 | ⚪ 可选 |
| 自适应 MC 运行次数 | N/A | ✅ 必加 | N/A | ⚪ 可选 | ⚪ 可选 |
| in-place / 预计算 | ✅ 必加 | ✅ 必加 | ❌ 不适用 | ❌ 不适用 | ⚪ 可选 |
| `if __name__ == '__main__':` | ⚪ 可选 | ✅ 必加 | ✅ 必加 | ✅ 必加 | ✅ 必加 |
| numba try/except 降级 | ✅ 必加 | ✅ 必加 | ❌ 不需要 | ❌ 不需要 | ⚪ 可选 |
| 防缓冲输出（第 4.5 节） | ⚪ 可选 | ✅ 必加 | ⚪ 可选 | ✅ 必加 | ✅ 必加 |

图例：✅ 必须遵守 · ❌ 禁止使用 · ⚠ 有条件使用 · ⚪ 推荐但非强制 · N/A 不适用

---

## 3. 性能优化优先级（从低成本到高成本，按序执行）

### P0 — Numba JIT（收益最大，改动最小）

**前置安全检查（强制）**：加 @njit 前扫描循环体，任一命中即放弃：

1. 调用第 2.2 节清单中的任何对象？
2. 捕获外部闭包变量（非参数传入）？
3. 对原生 dict/list 做动态增删？
4. 该函数属于 FIG / ML 类脚本？

四项全否才可加 @njit；否则将纯数值部分抽离为独立 @njit 函数，外层保留 Python 调度。

```python
from numba import njit

@njit(parallel=True, fastmath=True, cache=True)
def evolve_waveform(x, tau, alpha, n_pulses, decay_pw, decay_dt):
    for _ in range(n_pulses):          # 时序依赖，无法向量化 → numba 编译
        x = x + alpha * (1.0 - x)
        x = x * decay_pw
        x = x * decay_dt
    return x
```

**规则**：
- 核心数值计算函数必须加 `@njit(fastmath=True, cache=True)`；可并行的循环加 `parallel=True`。
- `@njit` 函数内部只能用数值、numpy 数组运算、纯数学函数；外层实验调度代码（调用 sklearn、绘图、CSV、PyTorch）不加 `@njit`。
- 首次运行有编译开销（约 2 秒），`cache=True` 缓存到 `__pycache__`，后续直接加载。
- `numba` 为可选依赖，必须 `try/except` 降级保护：

```python
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):  # 无 numba 时退化为普通函数
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f
```

### P1 — NumPy 向量化（消灭内层小循环）

- 逐个器件、逐个时间步的 for 循环必须改成数组批量运算（矩阵广播，下沉到 C 实现）。
- 循环内重复计算的常数（`np.exp(-dt/tau)`、衰减系数等）**提前预计算**为数组，循环内只做数组乘加。
- 已向量化的函数（`_vec` 后缀）保持现状，**不要再加 `@njit`**（会冲突）。

```python
# 错误 ✗
for i in range(N_DEV):
    waveform[i] = alpha * (1 - x[i]) * np.exp(-dt / tau[i])

# 正确 ✓（预计算 + 广播）
decay = np.exp(-dt / tau)           # (N_DEV,) 预计算一次
waveform = alpha * (1 - x) * decay  # 向量化乘法
```

### P2 — 多进程并行（独立实验组）

**适用**：Monte Carlo 多次运行、多组参数对照实验。
**禁用**：FIG 绘图循环；ML 训练循环内部；单次运行 < 5 秒（进程启动开销大于收益）。

```python
from multiprocessing import Pool, cpu_count

def run_single_mc(args):
    """单个 MC trial，必须是顶层函数（可 pickle）。"""
    task_dts, tau0_vals, mode, run_idx = args
    return run_mode(task_dts, tau0_vals, mode, None, run_idx)

with Pool(min(cpu_count(), n_runs)) as pool:
    results = pool.map(run_single_mc, all_args)
```

**规则**：
- MC 的 `for r in range(n_runs)` 循环必须用 `Pool.map` 并行（每个进程独立 GIL）。
- 并行函数必须是顶层函数（可 pickle），不能是 lambda 或闭包。
- 进程间传大数组有拷贝开销 → 尽量在进程内部生成数据（用 `run_idx` 作种子），只返回标量结果。
- `n_proc = min(cpu_count(), n_runs)`；Windows 下必须用 `if __name__ == '__main__':` 保护入口。

### P3 — 代码逻辑优化（无依赖，纯重构）

- **缓存复用**：预迭代波形、τ 分布、衰减系数、基础矩阵只计算一次，循环复用。
- **自适应 MC 运行次数**：不影响统计显著性前提下，小规模 15 次、中规模 10 次、大规模 5 次。
- **批量 IO**：CSV 一次性写入，禁止循环内逐行 `writer.writerow`；二进制中间存储用 `np.savez_compressed`。
- **连续内存**：大数组传给 numba/numpy 前用 `np.ascontiguousarray(arr)`。
- **in-place 操作**：用 `arr += delta`、`np.clip(x, 0, 1, out=x)` 减少内存拷贝。

### P4 — 环境与重度重构（一次性配置 / 按需）

- 使用 Python 3.11+（Faster CPython），导出 `requirements.txt` 保证环境可复现。
- P0-P3 仍不满足性能时再考虑 Cython / Rust+PyO3（释放 GIL，上层 Python 不变）。
- **禁止 GPU 方案**（无显卡环境）。

---

## 4. 强制编码规范

### 4.1 文件结构

```python
#!/usr/bin/env python3
"""
脚本标题 — 一句话描述
=============================================================================
Type:           PAPER | CORE | FIG | ML | EXPLORE
Paper §:        §4.11 (可选，论文类项目 PAPER 脚本建议填写)
Experiment:     <实验名称> (PAPER/EXPLORE 类必填)
实验目的 / 物理参数 / 输出文件 / 依赖注意
=============================================================================
"""
import sys, os, time
import numpy as np
from multiprocessing import Pool, cpu_count

# 1. 可选依赖（numba 降级保护，CORE/PAPER 类必加；FIG/ML 类可省略）
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]): return args[0]
        return lambda f: f

# 2. 全局参数 → 3. @njit 核心计算函数（底层） → 4. 调度函数（胶水层） → 5. 主函数
if __name__ == '__main__':
    main()
```

### 4.2 可复现性

- **随机种子**：每个 MC trial 用 `run_idx * scale + offset` 作种子，保证独立但可复现。禁止用全局 `np.random`。
- **CSV/JSON 输出**：实验结果输出 CSV + JSON，路径用 `Path(__file__).parent` 相对路径；JSON 必须记录全部实验参数。
- **PyTorch 脚本**：固定 `torch.manual_seed(seed)` 与 `torch.backends.cudnn.deterministic`（CPU 也写），并记录 torch 版本到 JSON。

### 4.3 命名约定

- 向量化函数以 `_vec` 结尾；`@njit` 函数以 `_nb` 结尾（或保持原名替代旧函数）。
- 单次 MC 运行函数以 `run_single_` / `run_one_` 开头；绘图函数以 `gen_` / `plot_` 开头（FIG 类强制）。

### 4.4 语言规范（强制英文）

所有技术文本必须英文：代码标识符、注释、docstring、CSV 列名、JSON 键、图表标签、print/日志、异常信息、文件名。例外：物理标准符号（τ、α、κ）可保留并附英文说明；既有脚本中的中文注释不强制翻译，但新增注释必须英文。所有文件以 UTF-8 保存。

### 4.5 输出与日志（防缓冲，强制）

> 背景：Python 非 TTY 环境（重定向/后台/管道）默认块缓冲（4-8KB），输出延迟数秒到数分钟，易被误判为卡死；多进程子进程独立缓冲，问题更严重。

- **脚本头部**（import 之后）设置无缓冲：

```python
import sys, os, time
os.environ.setdefault('PYTHONUNBUFFERED', '1')
if hasattr(sys.stdout, 'reconfigure'):          # Python 3.7+
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
```

- **多进程 worker 函数开头**必须重新设置无缓冲（子进程不继承父进程配置）：

```python
def run_single_mc(args):
    """Single MC trial, must be top-level (picklable)."""
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    # ... business logic
```

- **长任务进度**：MC / 训练 / 扫描循环每 10% 至少输出一次进度，`print(..., flush=True)`。
- **关键节点时间戳**：脚本开始、每个实验阶段、结束必须输出带时间戳的状态（如 `[HH:MM:SS] START/DONE: ...`）。
- **AI 运行长脚本（> 30 秒）**：禁止裸 `python script.py`，使用 `$env:PYTHONUNBUFFERED=1; python script.py`（环境变量被子进程继承，多进程场景最优）或 `python -u script.py`。

---

## 5. 禁止事项

| 禁止 | 原因 | 替代 |
|------|------|------|
| `@njit` 内调用 sklearn/matplotlib/PyTorch | numba 不支持 | 外层调度代码调用 |
| 对 FIG/ML 类函数加 `@njit` | 必然编译失败 | 保持 Python 原生 |
| 对绘图循环、ML 训练循环加 `Pool` | 冲突或无收益 | 串行执行 |
| PAPER 类 MC 循环串行执行 | 浪费多核 | `Pool.map` 并行 |
| 循环内逐行写 CSV | IO 瓶颈 | 收集后批量写 |
| 循环内重复计算 `np.exp(-dt/tau)` 等常数 | 冗余计算 | 预计算为数组 |
| 使用全局 `np.random.seed()` | 破坏可复现性 | 独立 `RandomState(seed)` |
| 使用 CuPy / torch.cuda | 无显卡环境 | numba CPU + 多进程 |
| `@njit` 内用 Python dict/list | numba 不支持 | numpy 数组或 typed.Dict |
| 修改既有脚本未做第 6 章预检查 | 可能污染已验证数据 | 先备份 + 预检查 |
| 代码/注释/输出列名/图表标签用中文 | 编码风险、规范冲突 | 统一英文（见 4.4） |
| 后台运行长脚本不带 `flush` / `-u` | 块缓冲误判卡死 | 见 4.5 防缓冲规范 |

---

## 6. 修改既有脚本的安全阀门（强制）

> 真实痛点：修改既有脚本改出问题，已验证的数据被破坏且难以察觉。本章重心在修改流程；新增脚本仅需遵守第 1-5 章。
> 修改既有脚本必须逐条执行 6.1-6.5；若被修改脚本是 CORE 类（被其他脚本 import），额外触发 6.3 的下游验证。

### 6.1 修改前预检查（必须）

1. **读取目标脚本全文**（不可只读片段），识别 docstring 的 `Type:` 字段。
2. **扫描 import 语句**，判断是否含 sklearn/torch/matplotlib/pandas。
3. **识别身份**：CORE 类 → 修改核心公式须用户明确授权；PAPER 类 → 触发 6.3 数值一致性验证。
4. **记录修改前不变量**：物理常数当前值、输出文件路径与列名（CSV 表头/JSON 键）、随机种子规则（scale/offset）、关键数值预期范围。
5. **备份**：`git stash` 或复制为 `<script>.bak.py`，确保可回滚。
6. **下游依赖扫描**（CORE 类强制，PAPER 类视情况）：grep 项目内所有 `import <module_name>`，列出依赖该脚本的下游清单。

预检查未通过（身份不明、缺 docstring、指令含糊、CORE 修改无授权）→ **停止并请求澄清**，不得猜测意图继续修改。

### 6.2 修改中约束（默认禁止，除非用户本次明确要求）

| 默认禁止 | 触发条件 |
|----------|----------|
| 修改物理常数、CORE 核心公式 | 任何情况 |
| 修改 CSV/JSON 输出列名、单位、顺序 | 任何情况 |
| 修改随机种子规则或 scale/offset | 任何情况 |
| 删除既有 docstring、`Type:` 字段、`if __name__` 保护、numba 降级保护 | 任何情况 |
| 给 FIG/ML 类函数加 `@njit` | 必然编译失败 |
| 给 ML 训练循环加 `Pool` | 与 PyTorch 冲突 |
| 引入 GPU 依赖 | 无显卡环境 |
| 把已向量化的 `_vec` 函数再加 `@njit` | 会冲突 |
| 把 PAPER 类多进程 MC 改成串行 | 性能回退 |

允许且鼓励：纯重构（变量改名、提取函数）、补注释、补 `Type:` 字段、按第 3 章优化未优化的循环（须通过前置安全检查）。

### 6.3 修改后验证（必须运行，不得仅声明"已修改"）

| 脚本类型 | 验证动作 |
|----------|----------|
| 所有类型 | `python -c "import ast; ast.parse(open('<script>').read())"` 语法检查 |
| CORE / PAPER / EXPLORE | 运行脚本（或 `--dry-run` 等价物），确认无 numba 编译错误、无运行时异常 |
| PAPER | 对比输出 CSV/JSON：列名集合完全一致；关键数值在修改前 ±0.5% 以内 |
| FIG | 运行生成 PDF，文件存在且非空 |
| ML | 运行 1 epoch（或减小 N_TRAIN）快速验证，训练能启动、loss 下降、无形状错误 |
| CORE（被 import） | 对 6.1 步骤 6 列出的每个下游脚本做 smoke test（`import` 成功 + 1 次最小参数运行），下游输出超 ±0.5% 即回滚并向用户报告影响范围 |

**回滚规则**：验证失败（编译错误、运行异常、数值超差、输出缺失）必须立即用备份回滚，**不得在未回滚状态下交付**，并向用户报告失败原因与建议。

### 6.4 红线（绝对禁止，须先书面提示风险并等待二次确认）

1. 删除或重命名既有 PAPER 脚本的输出 CSV/JSON 文件名（论文已引用）。
2. 修改 CORE 库中弛豫方程、噪声模型等核心公式。
3. 改变既有实验的随机种子规则（破坏 MC 可复现性）。
4. 把 `@njit` 加到调用 sklearn/matplotlib/PyTorch 的函数上。
5. 引入 GPU 依赖（CuPy / torch.cuda）。
6. 删除 numba `try/except` 降级保护（无 numba 环境会直接崩溃）。
7. 把 PAPER 类多进程 MC 改成串行（性能大幅回退）。

### 6.5 修改报告（AI 交付时必须包含）

每次修改既有脚本后，回复必须包含以下结构化报告，否则视为未完成：

```
修改报告
--------
目标脚本：xxx.py  (Type: PAPER)
修改意图：<一句话>
预检查：已读全文 / 已扫描 import / 已备份 .bak.py / 已记录不变量 / 已扫描下游依赖
修改内容：
  - <文件位置>：<改了什么>
不变量核对：
  - 物理常数 / 输出列名 / 随机种子规则：均未改 ✓
受影响的下游脚本（CORE 必填，其他写"无"）：
  - <下游脚本>：smoke test 通过 / 输出变化 ±X% / 需重新验证
验证：
  - 语法检查：通过
  - 运行结果：<关键数值> vs 修改前 <原数值>，差异 <±X%>
  - 数值一致性：<通过 / 超差已回滚>
是否触及红线：否
```

### 6.6 修改记录（强制）

- 每次修改后必须追加一条记录到 `review_workspace/modification_log_<scope>.md`（scope 如 `core`/`v11`/`scripts`），记录内容**必须用中文**书写：目标脚本、修改意图、修改前不变量、diff 摘要、验证结果、备份位置、是否触及红线。同 scope 的历史记录累积追加到同一文件，用时间戳分隔。
- 修改产生"结果性变更"（新增/删除脚本、Type 变更、输出文件新增/改名、物理参数变更）时，向用户报告建议整合清单，**经用户确认后**才可整合到 README（整合内容必须英文）；过程性变更（纯重构、补注释、性能优化）不整合。

---

## 7. 检查清单（提交脚本前自检）

**通用**（所有类型）
- [ ] docstring 已填 `Type:` 字段与依赖注意？
- [ ] 随机种子独立可复现（非全局 seed）？
- [ ] Windows 入口有 `if __name__ == '__main__':` 保护？
- [ ] CSV 批量写入（非循环内逐行）？
- [ ] 代码/注释/docstring/CSV 列名/图表标签全英文？
- [ ] 脚本头部与多进程 worker 已设无缓冲（4.5）？
- [ ] 长任务有进度输出且 `flush=True`？开始/阶段/结束有时间戳？

**CORE / PAPER 附加**
- [ ] 核心数值循环已加 `@njit(fastmath=True, cache=True)`（通过前置安全检查）？numba 有降级保护？
- [ ] MC 多次运行已用 `Pool` 并行？循环内常数已预计算？大数组已 `np.ascontiguousarray`？

**FIG / ML 附加**
- [ ] FIG 类确认未用 @njit 与多进程？
- [ ] ML 类确认未用 @njit、多进程仅包裹独立 trial、`torch.manual_seed` 已设置？

**修改既有脚本**
- [ ] 已按第 6 章完成预检查、备份、修改后验证、修改报告与记录？
