# 新算法研究计划 v1.0 草案
# Physics-Constrained Dual-Timescale Online Learning Architecture

> 本文档是新研究项目的权威计划（internal planning document，中文 + 英文术语，
> 与 CLAUDE.md §6.6 内部记录惯例一致；所有代码工件仍全英文）。
> 推进方式：每步 = 设计讨论 → 实现 → 运行 → 结果评审门（讨论门），逐步推进。
> 状态：v1.0（S0 决策已冻结，见 §6；D5 算法定名后议）。

---

## 0. 背景与功能目标（已锁定）

前提条件（用户已确认）：
- 无实验室、无流片条件；**全部权重/状态均为数字模拟**（Digital Model 级）。
- 计算环境 CPU-only（numba + 多进程，禁止 GPU）。
- 现有论文（Si₃N₄ pulse encoding，目标 NCE）按原样投出，不因本计划改动。
- 本计划定义**下一个项目**：一套全新的训练-推理一体的在线学习算法。

功能目标（对应用户原始三点想法）：

| 目标 | 内容 | 对应机制 |
|---|---|---|
| G1 训练-推理一体 | 无独立训练阶段；推理活动直接产生学习信号（eligibility），学习结果实时改变推理动力学 | M1/M2 |
| G2 从混沌中挖掘信息 | 衬底动力学可调至 edge of chaos（λ≈0），奖励机制纠正与巩固读出 | M2/M5 |
| G3 元数据与自然遗忘 | 每个单元携带慢时间尺度使用记录，指数式自然遗忘；拓扑连接互相影响处理结果 | M3/M4 |
| G4 全数字实现 | 全新的大模型优化算法：非反传、非 Transformer、局部规则、CPU 可完整验证 | 全部 |

诚实定位（防审稿人一句话毙稿）：
- 不宣称"替代 Transformer/大模型"；宣称生态位：**在线、低功耗、物理可实现、多时间尺度记忆**。
- 组件各有文献对应（three-factor rules / edge-of-chaos RC / Benna-Fusi 多时间尺度）；
  novelty 落在**组合 + 物理约束**：
  ① τ log-normal 谱 = 物理遗忘核（"用器件物理设计遗忘曲线"）；
  ② 慢迹/wear = 物理元数据（"degradation as memory"）；
  ③ α_eff 耦合 = 物理混沌旋钮。
- "大"体现在机制与表征的严格性，**不承诺参数规模**（CPU 约束下的诚实选择）。

---

## 1. 现有资产盘点（复用，禁止修改既有 CORE）

| 资产 | 内容 | 新项目中的角色 |
|---|---|---|
| `scripts/shallow_trap_array_simulator.py` | CORE：器件物理（τ、γ、α、注入-弛豫、CTIA 噪声、batch 采样） | 物理校准锚点；**只 import，不改** |
| `scripts/topology_comparison.py` | 8 种拓扑、α_eff 准静态耦合、100 MC、两任务族 | 拓扑机制原型；S1 将其时序化 |
| `scripts/experiment_phase3_new_topologies_100runs.py`、`experiment_new_topologies_100runs.py` | 拓扑消融的统计规范模板 | S7 对照数据来源 |
| `scripts/fair_esn_comparison.py` | ESN-256-hetero（leak rate 匹配 τ 分布的公平基线） | S9 最强对照 |
| `scripts/universal_frontend_v18_wear_leveling.py` | wear/退化与均衡 | 慢时间尺度元数据的物理依据 |
| `scripts/digital_twin_hybrid.py` | MLP surrogate 管线、10-seed CV | 实验管理与统计规范模板 |
| MC 框架（Pool + 独立种子） | 可复现统计 | 全程沿用 |

关键差距（必须新建）：
1. 现有拓扑耦合是**准静态**的：对整段脉冲序列取块均值电流，3 轮不动点迭代 α_eff，
   无逐脉冲时间递归 → 现有系统是强收缩的有序动力学（λ<0），到不了 edge of chaos。
2. 读出全部离线（RidgeClassifier，2000 train / 400 test）→ 无在线学习。
3. 无奖励/资格机制、无慢状态、无流式任务（漂移/上下文切换）评测。

---

## 2. 架构草案 v0.1（S0 讨论后冻结为 v1.0）

### 2.1 衬底（recurrent relaxation substrate）

N 个弛豫单元（= Si₃N₄ tile），逐脉冲更新（输入 = 脉冲间隔序列 Δt_t）：

```
x_i(t+1) = [ x_i(t) + α_eff,i(t) · (1 − x_i(t)) ] · exp(−Δt_t / τ_i)
```

两种耦合形式（S0 定主次，S1 消融）：
- **(B) α_eff 动态调制**（延续论文物理叙事，推荐为主）：
  `α_eff,i(t) = α0 · (1 + κ · g_i(x_j∈N(i), t))`，
  g 为邻居对比函数（ring / hub / lateral-inhibition 的逐脉冲版）。
- **(A) 加性状态循环**（教科书 ESN 式，作对照）：
  `x_i(t+1) += κ · Σ_j C_ij · φ(x_j(t))`。

其他物理参数：τ_i ~ log-normal(τ0, CV)（CV = 遗忘谱设计旋钮）；
读出电流 `I_i(t) = I_HRS · exp(γ · x_i(t))`。

### 2.2 双时间尺度状态（G3）

- **快态**：x_i(t)（trap occupancy，τ≈174 μs 等效）——记模式。
- **慢态（元数据）**：
  `m_i(t+1) = m_i(t) + β · r_i(t) − λ_m · m_i(t)`，
  r_i = 单元使用强度（如 |读出贡献| 或 |I_i − Ī|），λ_m << 1/τ ——记统计。

### 2.3 在线读出 + 奖励（G1、G2）

```
ŷ(t) = W · s(t) + b          （s(t) = 单元电流向量或子采样波形特征）

资格迹：  E_ij(t+1) = λ_e · E_ij(t) + s_i(t) · δ_j(t)
巩固：    W_ij(t+1) = W_ij(t) + η · R(t) · E_ij(t)
奖励：    R(t) = R_task(t) + κ_int · R_int(t)
```
R_int = 预测误差改进量 / 新颖度（内在动机，接 Friston free-energy 叙事）。

### 2.4 结构可塑性 + 混沌稳压器（G2、G3）

- **边演化（慢）**：低使用边剪枝、高奖励相关边增长（m_i、R̄ 门控）。
- **混沌稳压器**：在线监测分离度代理量 S(t)，调节全局 κ 使 S 维持在目标区
  （edge-of-chaos homeostat）——"奖励机制纠正模型"的结构级实现。

---

## 3. 新算法：五个机制（M1–M5）

| # | 机制 | 对应用户想法 | 实现步 | 文献最近邻 |
|---|---|---|---|---|
| M1 | 推理期资格积累（eligibility） | 训练推理一体 | S3 | three-factor rules（Pfister & Gerstner 2006） |
| M2 | 三因子奖励门控（task + intrinsic） | 奖励纠正 | S3/S4 | Hoerzel et al. PNAS 2014 |
| M3 | 双时间尺度元数据（x 快 / m 慢） | 元数据 + 自然遗忘 | S5 | Benna & Fusi 2016 |
| M4 | 结构可塑性（拓扑边演化） | 拓扑互扰 | S7 | evolving ESN 拓扑 |
| M5 | 混沌稳压器（κ homeostat） | 从混沌挖信息 | S6 | edge-of-chaos RC（Bertschinger 2004）+ intrinsic plasticity（Steil 2007） |

**工作名（待 S0 定名）**：REDEM —— REward-gated Dual-timescale Eligibility Mechanism。

算法整体定位：**局部规则在线优化算法**（无反传、无 BPTT、CPU 友好）。
明确排除任何 BPTT 变体（与"不用 Transformer 那一套"的初衷一致，也是 CPU 约束的正解）。

---

## 4. 分步计划（每步含讨论门；脚本命名遵循 CLAUDE.md §4.3）

| Step | 内容 | 新建脚本 (Type) | 验收标准 | 止损条件 |
|---|---|---|---|---|
| S0 | 规格冻结（本文档 §2/§3 + §6 决策） | 无（文档） | §6 五项决策落定，冻结 v1.0 | — |
| S1 | 循环衬底 + 混沌-容量相图 | `recurrent_substrate.py` (CORE) + `substrate_recurrence_characterization.py` (PAPER) | κ–MC–λ 相图；每拓扑找临界 κ*；κ=0 退化与旧并行系统对齐（±0.5%） | 若物理参数范围内到不了 λ≈0（如 α_eff clip [0.001,0.10] 过窄），扩范围或引入输出反馈，回报再议 |
| S2 | 在线读出底座（RLS）+ 流式任务 | `streaming_tasks.py` (CORE) + `online_readout_streaming.py` (PAPER) | 漂移流上稳定追踪（阈值讨论门可调）；在线 RLS vs 离线 Ridge 对比表 | 若 RLS 数值不稳（遗忘因子病态），换 APRLS/GD 在线版 |
| S3 | 三因子学习 v1 | `three_factor_online_readout.py` (PAPER) | ≥1 个指标明确优于 RLS（样本效率/稀疏奖励/适应速度），否则降级为消融并诚实记录 | 不优 → 讨论混合方案（RLS 主体 + 奖励门控偏差项） |
| S4 | 内在奖励 | `intrinsic_reward_hybrid.py` (EXPLORE→PAPER) | 无外部奖励段维持学习；κ_int 扫描曲线 | 内在奖励无增益 → 保留任务奖励，R_int 进消融 |
| S5 | 双时间尺度元数据 | `dual_timescale_metadata.py` (PAPER) | 长程统计任务上双尺度显著优于单尺度（配对统计，10 seeds）；解析遗忘曲线 M(t)=∫p(τ)e^{−t/τ}dτ 对照 Benna-Fusi | 双尺度无优势 → 项目核心卖点动摇，回到讨论门重估 |
| S6 | 混沌稳压器 | `chaos_regulator_homeostat.py` (PAPER) | 扰动（τ 漂移/断边/噪声抬升）后恢复时间 << 固定 κ 对照 | 无增益 → M5 降级为离线分析，不进系统 |
| S7 | 结构可塑性 | `structure_plasticity_topology.py` (PAPER) | 演化拓扑 ≥ 最优固定拓扑（复用 topology_comparison 8 拓扑对照），或给出可解释结构规律 | 演化不稳定 → 固定拓扑 + 仅边权学习 |
| S8 | 整合系统 + 消融矩阵 | `integrated_system_benchmark.py` (PAPER) | 全开 vs 每机制单独 on/off（6+1 组）；N=256 开发 / N=1024 终确认 | 消融显示某机制负贡献 → 从系统中移除并记录 |
| S9 | 基准对决 | `baseline_showdown.py` (ML) | vs ESN-256-hetero（复用）/ 离线 Ridge / 小 GRU / tiny Transformer（CPU 规模）；在线精度曲线、plasticity-stability、样本效率、遗忘曲线、计算成本代理 | 诚实报告：只报互补生态位，不搞替代叙事 |
| S10 | 论文打包（两篇策略） | 无 | Paper A（S1+S6 表征）→ NCE；Paper B（S2–S9 算法）→ NCE / Neural Networks / Frontiers | — |

各步要点补充：

- **S1**：有限时间 Lyapunov 指数用成对微扰轨迹（同输入、扰动初态）发散率；
  memory capacity MC(k) 用 Jaeger 定义（u_t = 归一化 Δt，回归 u_{t−k}）；
  分离度 = 不同输入类状态距离。扫描 κ × 拓扑 × CV × N，Pool 并行 MC。
- **S2**：带遗忘因子的 RLS；任务生成器含漂移脉冲流、NARMA-10、Mackey-Glass、
  上下文切换。此步是"训练推理一体"的最小可行版本。
- **S3**：先用监督误差作为奖励代理（最干净），S4 再换混合奖励。
- **S5**：理论钩子（log-normal τ 谱 → 拉伸指数/近幂律遗忘）可独立成论文小节。
- **S9**：ML 类脚本遵守规范——不加 @njit；多进程只包裹独立 trial 外层；
  `torch.manual_seed` + CPU；GRU/Transformer 用最小可比规模。

---

## 5. 规范约束（CLAUDE.md 合规清单）

- 新脚本 docstring 必填 `Type:`；CORE（`recurrent_substrate.py`）必加
  `@njit(fastmath=True, cache=True)` + numba try/except 降级保护。
- PAPER 类：MC 用 `Pool.map` 并行（顶层函数，可 pickle）；独立种子
  `run_idx*scale+offset`；CSV+JSON 双输出且 JSON 记录全部参数；
  防缓冲输出（头部 + worker 内 reconfigure）；时间戳日志；长任务进度 flush。
- **禁止修改既有 `shallow_trap_array_simulator.py`**（CORE，下游众多）；
  新模块只 import（I_HRS, gamma, tau0, gen_tau_vec, ... 等既有接口）。
- 禁止 GPU；禁止 BPTT 类学习规则。
- 全部代码/注释/输出列名英文；每次实现后过 CLAUDE.md §7 自检清单。

---

## 6. S0 待决策项（讨论门）

| # | 决策点 | 选项 | 推荐 |
|---|---|---|---|
| # | 决策点 | 选项 | 决定 |
|---|---|---|---|
| D1 | 任务族 | A 双任务族（现有脉冲任务 + NARMA-10/MG）/ B 仅脉冲 / C 仅经典基准 | **A 双任务族**（用户确认，S0 讨论门） |
| D2 | 奖励形式 | A 混合（逐脉冲内在 + 序列任务）/ B 仅任务 / C 仅内在 | **A 混合奖励**（用户确认，S0 讨论门） |
| D3 | 耦合主次 | A α_eff 动态调制为主、加性循环对照 / B 加性为主 / C 两者完整消融 | **A α_eff 动态调制为主**（用户确认，S0 讨论门） |
| D4 | 衬底规模 | A N=256 开发、1024 终确认 / B 全程 1024 / C 全程 256 | **A 256 开发 / 1024 终确认**（用户确认，S0 讨论门） |
| D5 | 算法定名 | REDEM（暂名）/ 用户自定 / S0 后再定 | 后议（不阻塞 S1） |

---

## 7. 里程碑与时间线（单作者兼职节奏的乐观估计）

| 里程碑 | 内容 | 步 | 估计 |
|---|---|---|---|
| M1 | 相图 + Paper A 骨架 | S0–S1 | 3–4 周 |
| M2 | 在线学习核心，算法 v1.0 | S2–S5 | 5–7 周 |
| M3 | 鲁棒性 + 基准，可写论文 | S6–S9 | 6–8 周 |
| M4 | 成文投稿 | S10 | 3–4 周 |

总计约 4–6 个月。

---

## 8. 风险登记册

| 风险 | 概率 | 缓解 |
|---|---|---|
| 三因子不如 RLS | 中 | 降级为混合规则；RLS 本身也是"训练推理一体"的合法实现 |
| α_eff 物理范围内到不了 λ≈0 | 中 | 输出反馈环兜底；或叙事改为"临界附近" |
| 灾难遗忘 | 高（若单尺度） | 双时间尺度是标配不是加分项；S5 即验证 |
| 审稿人"ESN 变体"质疑 | 中 | 物理遗忘谱 + 元数据 + 稳压器三重差异化；引 Benna-Fusi / 三因子文献正名 |
| CPU 算力不足 | 低 | N=256 + numba + Pool；S8 才上 1024 |
| 与现有论文抢时间 | 中 | 现有论文先投；本项目为下一篇 |

---

## 9. 变更记录

- v1.0 草案：初版（AI 起草，含资产盘点、架构草案、五机制算法、S0–S10 分步计划）。
- v1.0 冻结：S0 讨论门完成，D1–D4 全部采纳推荐方案（双任务族 / 混合奖励 /
  α_eff 动态调制为主 / 256 开发 1024 终确认）；D5 定名后议，不阻塞 S1。
  依据：代码勘察确认现有拓扑耦合为准静态（3 轮不动点 α_eff），
  逐脉冲递归须新建（见 §1 关键差距第 1 条）。
- S1 完成（v1 粗网格 550 runs + v2 细化网格 610 runs，各 10 seeds）：
  - 新建 `scripts/recurrent_substrate.py`（CORE，自检 4 项全过；κ=0 与旧 CORE
    逐位一致 ≤1e-15，满足"退化对齐"验收）。
  - 新建 `scripts/substrate_recurrence_characterization.py`（PAPER，v2）：
    FTLE（Benettin）+ Jaeger MC（新增 70/30 持出分割，防混沌区过拟合假象）
    + 输入分离度；输出 `data/substrate_phase_diagram_v2.csv/json`。
  - 新建 `scripts/gen_substrate_phase_diagram.py`（FIG）→
    `figures/substrate_phase_diagram_v2.png`。
  - 核心发现（详见讨论门记录）：
    ① mode-1 负反馈族（ring_bidir/lateral_ring/random_graph）临界 κ*∈(25,30)，
       持出 MC 峰值在临界前 κ≈20-25（random_graph +53%：9.07→13.91）；
    ② ring_unidir（mode 2）在 κ≈1-3 自限临界（λ≈−0.001，α_eff clip 稳定），
       分离度最大（inter 0.085）但线性记忆≈0——分离-记忆取舍；
    ③ 深混沌摧毁持出记忆（κ=50/100 → MC_te 1.85-4.4）；
       训练段 MC 的"爆炸"（30+）确认是过拟合假象；
    ④ 加性对照（mode 3）记忆永不超基线且饱和即死；hub_star 无反馈环冻结；
       物理约束的对比耦合完胜朴素 ESN 式加性——支撑"物理约束即科学内容"叙事；
    ⑤ M5 稳压器设计输入：最优工作点依任务而定（记忆→临界前，分离→临界处），
       不是简单把 λ 拉向 0。
  - 未做（按计划推迟）：CV 扫描、N=1024 终确认（S8）。
- S2 完成（v3：120 runs，10 seeds；RLS λ=0.999 单一设置）：
  - 新建 `scripts/streaming_tasks.py`（CORE：漂移二分类 / NARMA-10 / Mackey-Glass /
    上下文切换生成器，自测全过）。
  - 新建 `scripts/online_readout.py`（CORE：OnlineRLS 带遗忘因子 + Tikhonov 正则
    + 迹上限；在线协议=先预测后更新；ridge_fit 离线基线含截距中心化）。
  - 新建 `scripts/online_readout_streaming.py`（PAPER）+ `gen_s2_curves.py`（FIG）
    → `data/s2_online_readout_v1.csv/json/npz`、`figures/s2_online_readout_v1.png`。
  - 核心发现：
    ① 漂移任务：在线 RLS 全程跟踪（mean acc 0.974-0.982，交换后 225-616 脉冲恢复），
       冻结离线基线交换后系统性反转（0.00-0.02，永不恢复）——G1 训练推理一体的
       最小可行演示；
    ② 近临界耦合（random_graph κ=25）大幅提升回归任务：MG NMSE 0.0018 vs parallel
       0.090（50×），NARMA 0.431 vs 0.549（+21%）；
    ③ 在线 RLS ≥ 离线 ridge 全场景成立（自适应读出从不更差）；
    ④ 稳定-可塑发现：λ<0.999 在长流上 RLS 慢性发散（近临界特征病态协方差），
       自适应遗忘留给 S5（元数据/双时间尺度机制）；
    ⑤ 方法学教训：离线 ridge 必须先中心化目标（截距），否则 NMSE 虚高 20 倍。
  - 验收：漂移流稳定追踪 ✅；在线 vs 离线对比表 ✅。
- S3 完成（160 runs，10 seeds）：
  - 新建 `scripts/three_factor_online_readout.py`（PAPER）；`online_readout.py`
    新增 `ThreeFactorReadout`（reward 模式：资格迹 x·o + ±1 奖励门控；
    error 模式：输入迹 + δ 门控）。
  - 核心发现（诚实负结果，触发计划 §4 止损条款"三因子不优→降级"）：
    ① RMHL（纯 ±1 奖励，无类别信息）：预交换 0.89-0.94，交换后 0.06-0.10
       永不恢复——无误差信号的奖励无法在映射反转时完成信用分配（可发表负结果）；
    ② 误差门控三因子（LMS，η=1e-4）可跟踪漂移（mean 0.908-0.927）但数值脆弱
       （η≥1e-3 发散），全任务弱于二阶 RLS；
    ③ RLS（二阶）是稳健在线学习者：dense 0.983-0.991 / sparse（块末类别）
       0.958-0.961 均通过反转；
    ④ 近临界耦合的增益在 RLS 上依旧：MG 0.0018 vs 0.093（50×）、NARMA 0.431
       vs 0.549；
    ⑤ 设计修正：M1/M2 的差异化价值不在读出层（RLS 赢），转移到衬底层适应
       （M4 结构可塑性 / M5 稳压器）与慢元数据（M3）——本就是计划中的
       独特组件；读出层以 RLS 为主体（混合规则：RLS 主体 + 奖励门控偏差项）。
  - 验收：漂移稀疏奖励对比表 ✅；回归验证 ✅。
- S4 完成（160 runs，10 seeds；`scripts/intrinsic_reward_experiment.py` PAPER）：
  - 实验：RMHL + 新颖性内在奖励 × κ_int ∈ {0,0.1,0.5,2.0} × 任务奖励频率
    {每块, 每5块} × 双衬底。
  - 结论（系统性负结果，M2 叙事闭环）：
    ① 任务无关内在奖励（新颖度）无法提供类别方向信息 → 无法营救反转失败；
       最佳 post 0.42-0.59（随机水平）vs 无内在的 0.06-0.10——部分缓解但
       以摧毁预交换学习为代价（0.89→0.57），全程 mean 从不超基线（≤0.514）；
    ② κ_int 增大单调摧毁初始学习（random_graph pre 0.94→0.41）；
    ③ 奖励每 5 块 + 内在 = 完全混乱（0.49-0.50）；
    ④ 内在信号的正确定位：不是读出层的奖励救援，而是 M4 结构可塑性
       （边增删的探索信号——那里不需要类别方向，只需要"什么变了"）。
  - M2 定稿：读出层 = RLS（误差驱动、二阶）；内在动机 → 结构探索信号（M4）；
    自由能叙事保留为 S4 负结果的解释框架（无误差信号的奖励 = 无信用分配）。
- S5 完成（100 runs，10 seeds；`scripts/dual_timescale_metadata.py` PAPER）：
  - 任务重设计：稀有事件率 regime 切换（3 区间共享相同脉冲间隔分布，仅
    "长间隔事件"概率 p ∈ {0.12,0.20,0.28} 不同——单脉冲不可分，长窗率估计才可分）。
  - 慢元数据 = 快特征（电流比）的逐单元 EMA（τ ∈ {200,1000} 脉冲）。
  - 结果（配对统计，10 seeds）：
    ① overall：dual 0.996/0.994-0.995 vs fast-only 0.983/0.973，
       差 +1.3pp（parallel，t=19.6，p<0.0001）/+2.1pp（random_graph，
       t=20.2，p<0.0001）；
    ② **边界适应：dual 3-12 脉冲 vs fast 66-105 脉冲（9-20× 快）**；
    ③ slow-only ≈ dual（regime 任务以统计量为主），但 dual 略优；
    ④ τ=200 与 τ=1000 相当（率估计在两种尺度都足够）。
  - 验收"双尺度显著优于单尺度（配对统计）"✅（核心卖点 M3 验证通过）。
  - 待办：解析遗忘曲线 M(t)=∫p(τ)e^{−t/τ}dτ 对照 Benna-Fusi（理论小节，
    归入 Paper A 写作阶段）。
- S6 完成（200 runs，10 seeds；`scripts/chaos_regulator.py` PAPER +
  `recurrent_substrate.py` 新增 `run_trajectory_kappa_nb`）：
  - Part 1（扰动下 MC-κ 景观）：噪声伤记忆（MC 10.05 vs 13.98）；edge_prune
    反而升记忆（16.53）；tau_drift 最优 κ 右移（25→40 区间更平）；活动代理量
    无扰动不变的最优目标 → 固定 S* 稳压器失败（首版实验确认后废弃）。
  - Part 2（λ-homeostat，Benettin FTLE 每 1000 脉冲估计一次，λ 目标 −0.02）：
    稳压器恒定settle在 κ≈26-27，**扰动后持出 MC 全面 +8~18%**：
    none +7.6%、tau_drift +18%、edge_prune +7.9%、noise +11.8%；
    任务级 MG NMSE 与 fixed 持平（RLS 读出补偿任务差异，衬底质量由 MC 探针
    揭示）；恢复时间部分场景更快（none 752 vs 3310 脉冲）。
  - 结论：M5 有正增益（MC +8-18%），保留进系统；读出补偿掩盖任务级差异
    → 稳压器的价值是衬底级记忆质量，S8 整合时用元数据读出放大。
  - 验收：扰动鲁棒性 ✅（MC 探针 +8-18%）；恢复时间部分达标（诚实标注）。
- S7 完成（70 runs，10 seeds；`scripts/structure_plasticity.py` PAPER +
  `recurrent_substrate.py` 新增 `adjacency_to_csr`；`online_readout.py` 新增
  `memory_capacity_heldout` 并修正 corr² 平方约定）：
  - 规则迭代：随机增长（负）、相关性引导（负）、多样性反规则（负）、
    温和搅动（正）——插桩修了两个 bug（mask 上下三角不对称、washout 双重裁剪）。
  - 结果（同协议持出 MC）：fixed_ring 11.54；fixed_random_sp（同密度随机）13.52；
    **evolve_ring 温和（5% 搅动）12.43（+7.8% vs 起点 ring）**；激进 20% 搅动 8.88
    （−23%，失稳）；**repair_damaged 12.68→14.11（+11.3%，修复损伤）**；
    剪密的随机图反而升记忆（644 边 12.68 > 全 1062 边 10.37）——去同质化提升多样性。
  - 可解释规律：① 慢速结构可塑性有效（5% 温和 > 20% 激进，符合生物慢适应叙事）；
    ② 相关性引导剪枝去冗余 → 多样性 → 记忆；③ 结构比密度更重要（稀疏随机 > 环）。
  - 诚实边界：演化未超过最优固定拓扑（12.43 < 13.52）；M4 保留为"温和相关
    引导重连"，S8 整合；激进搅动作为负对照记录。
- S8 完成（56 runs；N=256 10 seeds 消融 + N=1024 3 seeds 终确认；
  `scripts/integrated_benchmark.py` PAPER）：
  - 系统：RLS 读出 + 双时间尺度元数据（τ=500）+ λ-homeostat + 温和结构可塑性。
  - N=256 消融（regime 任务 overall acc / 适应时间）：full 0.996 / 3p；
    baseline（仅 RLS 快特征）0.973 / 105p；no_metadata 0.988 / 36p；
    no_homeostat 0.996 / 3p；no_plasticity 0.994 / 10p。
  - **配对检验：full vs baseline +2.28pp（t=19.4，p<0.0001）**；机制边际贡献：
    元数据 +0.78pp（最大，匹配 S5）、可塑性 +0.21pp、homeostat 本任务≈0
    （homeostat 价值在扰动场景，regime 任务无扰动——S6 已证）；
  - **N=1024 终确认：full 0.998 vs baseline 0.976（+2.1pp，3 seeds），
    优势在规模上保持并略扩大**。
  - 验收：消融矩阵 ✅；1024 终确认 ✅。
  - 实验阶段（S1-S8）全部完成。剩余：S9 基准对决（ESN-256-hetero / 小 GRU /
    tiny Transformer，CPU 规模）→ S10 两篇论文。
- S9 完成（70 runs；`scripts/baseline_showdown.py` ML 类，torch CPU）：
  - 系统：redem（耦合衬底+RLS 在线）/ esn（ESN-256-hetero+RLS 在线）/
    gru（GRU-64，前 30% 离线训练后冻结）/ trans（tiny transformer d=64
    context 256，前 30% 冻结）。修了 3 个评估 bug（GRU 末位广播、transformer
    窗口切片、GRU 输入维度）。
  - drift_binary（mean acc / 交换前 / 交换后）：esn 0.998 / 1.000 / 0.996；
    **redem 0.991 / 1.000 / 1.000**；gru 0.371 / 0.923 / 0.070；trans 0.351 /
    1.000 / 0.000。
  - mackey_glass NMSE：esn 0.0002；redem 0.0018；trans 1.07；gru 1.30。
  - 结论（诚实生态位，不搞替代叙事）：
    ① 在线 vs 冻结的决定性分界：漂移任务在线系统 0.99 vs 冻结批量 0.35-0.37
       （学到预交换、交换后系统性反转永不恢复）——"训练推理一体"的核心价值；
    ② **ESN 在两个任务上略胜 REDEM**（0.998 vs 0.991；0.0002 vs 0.0018）——
       通用调优储层（tanh、谱半径 0.9）很难在标准任务上被超越，如实记录；
    ③ REDEM 的差异化价值不在标准任务数值：物理可实现性（Si₃N₄ 叙事）、
       双时间尺度元数据（regime 任务，S5/S8 已证 ESN 无此结构）、扰动鲁棒性
       （S6）、无需 BPTT 的局部在线规则；
    ④ 论文定位：Paper B 不宣称"击败 ESN"，而宣称"物理基底 + 元数据 + 鲁棒性
       机制；标准任务上与 ESN 竞争、物理叙事与鲁棒性提供差异化"。
- 下一步：S10（论文打包：Paper A = 相图+容量表征+遗忘曲线理论（S1/S5/S6 表征）；
  Paper B = REDEM 算法+基准（S2-S9）；引用现有论文作 substrate calibration）。

---

## 10. 会话状态（实验阶段收官）

S1-S9 全部完成（14 个新脚本、12 组实验数据、`NEW_ALGORITHM_PLAN.md` 全程更新）。
会话暂停于此，S10 论文写作从新会话开始。新会话入口提示：
- 所有实验结果数据在 `data/`（substrate_phase_diagram_v2、s2_online_readout_v1、
  s3_three_factor_v1、s4_intrinsic_reward_v1、s5_dual_timescale_v1、
  s6_chaos_regulator_v1、s7_structure_plasticity_v1、s8_integrated_v1、
  s9_baseline_showdown_v1 等 .csv/.json）。
- 图表在 `figures/`（substrate_phase_diagram_v2.png、s2_online_readout_v1.png）。
- 写 Paper A 时补做：log-normal τ 遗忘曲线 M(t)=∫p(τ)e^{−t/τ}dτ 的解析推导
  （对照 Benna-Fusi），可加一个 CORE 分析脚本。
- S10 进行中（本次会话）：已完成理论件 `scripts/forgetting_curve_theory.py`
  （EXPLORE）→ `data/forgetting_curve_theory.csv` + `figures/forgetting_curve_theory.png`。
  结论：1/e 视野 ~12-16 脉冲（CV 无关，由中位 τ0 决定，与 N_eff≈17 吻合）；
  尾部斜率随 CV 增大变缓（−60→−7：宽谱=重尾，但未达幂律 −0.5，达幂律需极宽谱）；
  **S1 实测 MC 曲线与理论核 Pearson r=0.97**（lag 10 几乎重合：0.520 vs 0.523）。
  待写：Paper A/B 骨架（仓库顶层 PAPER_A_sketch.md / PAPER_B_sketch.md）。
- S10 本会话进度：
  - 遗忘曲线理论 ✅（见上）。
  - Paper A 骨架 ✅：标题/摘要草稿/8 节大纲/图表清单（Fig 2-3 现成，
    Fig 4 新生成）/关键数字表。
  - Paper B 骨架 ✅：标题/摘要草稿/6 节大纲（含 S3/S4 负结果设计论证）/
    图表清单（Fig 2 现成，Fig 3-6 新生成）/关键数字表。
  - 新图 4 张：`figures/paperA_fig4_robustness.png`、
    `paperB_fig3_metadata.png`、`paperB_fig5_ablation.png`、
    `paperB_fig6_showdown.png`（`scripts/gen_paper_figures.py` FIG 类）。
  - 剩余（下会话）：论文正文写作（引言文献、方法、讨论成文）、Fig 1 示意图、
    命名决定（D5）、期刊定稿。
- S10 本会话完成：
  - **Paper A 完整初稿**：`PAPER_A_draft.md`（~2500 词全文散文：引言文献、
    衬底模型、表征方法、相图、遗忘理论、λ-homeostat、讨论、结论 + 参考清单）。
  - **Paper B 完整初稿**：`PAPER_B_draft.md`（~3000 词全文散文：引言、架构、
    任务指标、五组结果含负结果、讨论、结论 + 参考清单）。
  - **README_REDEM.md**：REDEM 项目 README（概览、架构表、16 脚本清单、
    数据/图表清单、复现命令、S1-S9 结果表、路线图、待办）。
  - 两篇草稿均使用全部真实数据数字；骨架文档保留为图表清单参考。
  - 剩余：正文润色 + 文献补全 + Fig 1 示意图 + 命名（D5）+ 期刊定稿。
- S10 收尾会话完成：
  - 文献核实（web 检索）：Hoerzel 2014 修正为 **Cerebral Cortex 24(3):677-690**
    （原误标 PNAS）；Benna-Fusi 页码 1697-1706；其余关键文献（Maass 2002、
    Jaeger 2001、Bertschinger 2004、Pfister-Gerstner 2006、Friston 2010、
    Pathak 2018、Rao-Ballard 1999、McClelland 1995、Parisi 2019）全部核实
    并写回两篇草稿。
  - Fig 1 示意图 ×2：`figures/paperA_fig1_substrate.png`（衬底框图）、
    `figures/paperB_fig1_redem.png`（REDEM 系统框图）（`gen_architecture_schematic.py`）。
  - **任务级 CV 扫描**（`scripts/cv_sweep.py`，90 runs）：未耦合 MC 随 CV 升
    （2.54→3.37，+33%，符合遗忘核理论）；耦合近临界 MC 随 CV 降
    （random_graph 12.93→7.33，窄谱最优）——CV 旋钮方向依赖工作区间，
    已写入 Paper A §5.2/§7。
  - **ESN+元数据转移实验**（`scripts/esn_metadata_comparison.py`，30 runs）：
    esn_dual 0.998 ≈ esn_fast 0.996 ≈ redem-full 0.994——元数据是"均等器"、
    可迁移（ESN 0.996→0.998、REDEM-baseline 0.973→0.994），裸 ESN 的 tanh
    慢模态已部分覆盖 regime 任务；Paper B 摘要/§4.5/§5 已据实修正（替换
    "混合后会缩小差距"的推测声明）。
  - README_REDEM.md 已同步（新脚本/数据/图/结果行）。
  - 剩余：正文最终润色、期刊定稿、命名决定（D5）。
- S10 修订会话完成（用户 8 条诊断 S1-S5/T1-T3 评审 + 期刊拆分定稿）：
  - **期刊拆分定稿**：Paper A → IJBC/Chaos（动力学理论刊）；Paper B → Neural Networks（算法刊）。
  - **Paper B**（`PAPER_B_draft.md`）：
    - S1：标题去掉 "physics-constrained"，改为 "REDEM: Training-Inference Unified
      Learning with Meta-Adaptation and Structural Plasticity for Non-Stationary
      Environments"；物理叙事保留在正文作背景（材料谱=超参数不是自由超参）。
    - S2：摘要重写为"差异化前置"（训练推理一体、全局部信号、自调节、抗扰、
      ESN 标准任务对比放最后一句），未夸大。
    - S3：负结果（RMHL/内在奖励）压缩为 Table 1（4 行：RMHL/RMHL+novelty/LMS/RLS）
      + 3 句"教训→设计决策"；数据未删。
    - T1：元数据结论改述——元数据是**衬底无关的可迁移机制**（ESN 0.996→0.998，
      REDEM 0.973→0.994），REDEM 差异化 = 机制集（自调节、抗扰、局部稀疏耦合、
      材料谱）而非机制本身；**并纠正原方案的事实错误**："ESN 需要 BPTT" 为误
      （ESN 本身就是无 BPTT 方法），差异化清单已修正。
    - T3：§2 开头加衬底模型交叉引用（"The substrate follows the model of the
      companion theory paper (Paper A, §2)…"）。
  - **Paper A**（`PAPER_A_draft.md`）：
    - S4：任务级 CV 扫描从 §5.2 正文移入 **Supplementary Note 1 + Fig. S1**
      （新 FIG 脚本 `gen_paperA_supp_figures.py`）；正文保留一句摘要式结论。
    - S5：新增 **Appendix A 完整推导**（A.1 热激活谱→对数正态性（σ_E≈5.1 meV）；
      A.2 两步更新与序无关性；A.3 遗忘核 Gauss-Hermite 求积 + 鞍点中位数锚定
      + 尾部斜率渐近式 dlnM/dlnt≈−(lnt−μ)/σ²（t=200 预测 −64.8 vs 实测 −60.6）；
      A.4 Benettin 详细迭代步骤；A.5 MC(k) 估计器 + 线性储层闭式解
      MC(k)=c_kᵀΣ_x⁻¹c_k 与 ΣMC≤N 容量界（Dambre 2012）——非线性衬底如实标注为
      经验估计，未虚构闭式解）。
    - 正文各节加附录交叉引用；参考文献补 [12] companion Paper B。
  - **Fig 1（Paper B）**（`gen_architecture_schematic.py`，已备份 .bak.py）：
    - T2：新增 M4↔M5 耦合环（紫色点线双向箭头，标签 "kappa gates rewiring;
      structure feeds lambda"）；"physics substrate" 标签改 "relaxation substrate"；
      修复输出框越界与零长度箭头；像素校验通过（紫/红/绿/蓝元素齐全）。
  - README_REDEM.md 已同步（目标期刊、论文状态、Fig S1、新脚本、开放项）。
  - 剩余：正文最终润色、`.tex` 转换、命名决定（D5）。
- S10 后续会话：**`.tex` 投稿稿转换**（本机 MiKTeX，`pdflatex ×2` 编译验证通过
  ——latexmk 需 Perl 未安装；编译日志零 Overfull/零未定义引用/零重复锚点）：
  - `PAPER_A.tex`：IJBC/Chaos 格式（article 类可独立编译，投稿时换 ws-ijbc.cls
    或 revtex4-2）；含 4 幅图 + 2 表（相图汇总表、homeostat 增益表）、附录 A
    完整推导（A1-A5 编号公式）、Supplementary Note 1 + Fig. S1、12 条文献。
  - `PAPER_B.tex`：Neural Networks 格式（投稿时换 elsarticle.cls）；含 6 幅图
    + 2 表（负结果表 Table 1、标准基准表 Table 2）、机制方程 Eq.1-4
    （RLS 更新 / 元数据 EMA / homeostat / 相关性矩阵）、11 条文献。
  - 作者/单位/邮箱为占位符，投稿前填写。
  - 剩余：正文最终润色、命名决定（D5）、投稿系统注册与上传。
- S10 仓库整理会话：新仓库 **REDEM（私有）** 建立并上传（gh 账号 huyamingc）：
  - 删除旧论文（Si₃N₄ pulse encoding）文件：manuscript.tex、旧 README.md、
    iopjournal.cls、cover_letter.txt、orcid.pdf、titlepage.html、back/、
    spice_simulations/、旧 scripts 83 个、旧 data 207 个、旧 figures 21 个（PDF）。
  - 保留两个被新脚本依赖的 CORE 模块：`shallow_trap_array_simulator.py`（11 个
    新脚本 import）、`fair_esn_comparison.py`（S9/S10 ESN 类）；import 冒烟测试
    与 recurrent_substrate.self_test（4/4）通过。
  - 重构：`paper_a/`、`paper_b/` 子文件夹（.tex/.pdf/draft/sketch 各 4 件），
    .tex graphicspath 改为 `../figures/` 并从子目录重新编译验证
    （A 12 页 / B 9 页，零告警）。
  - 根 README.md 改为项目总览（两篇姊妹论文 + 仓库结构 + code availability）。
  - 本地 .git 重新初始化（旧仓库历史完整保留在
    github.com/huyamingc/Si3N4-Pulse-Encoding）；全量备份：
    `D:\work\papers\testpynew_preclean_20260822.zip`（6.4 MB）。
  - 新仓库：github.com/huyamingc/REDEM（私有，评审期，录用后转公开）。
- 补强实验会话（E3/E4/O4 全量运行 + 论文整合）：
  - **E3 扰动链**（`scripts/s11_disturbance_chain.py`，PAPER，10 seeds）：
    三轮连续扰动（τ-drift → edge-prune → noise）后 regulated 臂 MC 8.47 vs
    fixed 6.41（+32%）；κ 从 26.15 漂至 28.51（主动补偿）。
  - **E4 λ_target 扫描**（`scripts/s12_lambda_target_sweep.py`，PAPER，5 seeds
    × 4λ × 3CV × 2 arms = 120 runs）：λ_target=0（edge of chaos）在所有 CV 下
    最优——CV=0.1 +25%、CV=0.2 +19%、CV=0.4 +5% over fixed；MC 随 λ→0
    单调递增。更新手调 -0.02 为经验最优 0。
  - **O4 因果审计**（`scripts/s13_causal_audit.py`，PAPER，3 seeds × 7 arms）：
    所有 leak 臂与 normal Δ < 0.02pp → 因果干净；causal_split 协议 Δ=0.01pp。
  - **CORE 优化**：`scripts/online_readout.py` RLS update 下沉 numba @njit
    （预分配 workspace、in-place 修改 P/W），单次 O4 运行 604s→107.5s（5.6×）。
  - **论文整合**：Paper B §4.4 新增 E3/E4/O4 三段 + Table 2（λ sweep）+ 摘要/
    结论更新；Paper A §5.2 新增 E4 引用；README/README_REDEM 脚本清单 22→25、
    headline results + roadmap 同步。两篇 pdflatex ×2 编译零错误（A 13p / B 10p）。
- O4 复核会话（2026-02-19，s13 审查修复）：
  - **修复 bug**：`scripts/s13_causal_audit.py` 原 `leak_plasticity` 臂未实现
    （`LEAK_FRAC_PLASTICITY=0.10` 定义后从未被引用，该臂与 normal 逐位相同，
    已由数据核对证实）。补实现：C 混入 10% 下一块相关性（从当前态 x_cur +
    当前物理跑未来 PLASTICITY_EVERY 脉冲轨迹求 C_fut），与 leak_metadata
    同构；N_SEEDS 3→10；Pool(4)→min(cpu_count(), n_runs)。
  - **重跑结果**（7 臂 × 10 seeds，217.8s）：所有 leak 臂因果干净——
    最大偏移 0.012pp（leak_rls），可塑性 leak −0.01pp（最负，说明 M4
    不利用未来相关性）；causal_split 0.9963 vs normal 0.9963。
  - **论文更新**：Paper B §4.4 O4 段数字/seed 数更新（10 seeds，
    0.9964–0.9962 vs 0.9963，≤0.02pp）；README/README_REDEM O4 行同步。
    备份 scripts/s13_causal_audit.bak.py。
- S9 复核会话（2026-02-19，ML 脚本审查修复）：
  - **修复 3 处 + 1 处文档**：`scripts/baseline_showdown.py` ① ESN 特征加
    z-score（与 REDEM 同款 30% 拟合窗口，消除预处理不对称）；② TinyTransformer
    传因果掩码（原 `causal_mask()` 定义了从未使用，256 窗口内双向注意力可偷看
    未来，开发者注释 "no ->" 已自认）；③ `torch.manual_seed(0)` 全局 → 逐 trial
    `seed_idx*101+17`（NN 基线不再跨 seed 共享初始化）；④ `N_SEEDS_TRANSFORMER`
    5→10。`scripts/s18_llm_drift_gate.py` JSON 注释 steady*1.10→1.5（对齐代码）。
  - **重跑结果**（80 runs，436.7s）：REDEM 不变（0.991 / MG 0.0018）；ESN
    修正后 1.000 / MG 3.6e-5（z-score 移除劣势，诚实领先叙事更强）；GRU
    0.371→0.394（pre 0.923→0.860，post 0.070→0.145）；trans drift 0.351
    不变、MG 1.07→0.71；trans 现为 10 seeds。
  - **论文/文档更新**：PAPER_B.tex 4 处（摘要/批基线段/表/结论）+ sketch
    3 处 + README/README_REDEM S9 行；paperB_fig6_showdown.pdf 重生成；
    PAPER_B 重编译（10 页，零警告）。备份 scripts/baseline_showdown.bak.py。
- 投稿物料会话（2026-02-19）：
  - **Paper D 标题修订**：删除 "Foundation Model Architecture"（toy 原型
    过声称，用户挂起门关闭），改为 "A State-Space Architecture with Native
    Online Learning, Meta-Adaptation, and Structural Plasticity"。
    PAPER_D.tex（注释+\title）+ paper_d/README + sketch + 根 README（bullet+
    布局）+ README_REDEM 开放项 5 处同步；PAPER_D 重编译 11 页零警告。
  - **Cover Letter ×4**：paper_a/b/c/d/COVER_LETTER.md（真实作者：Yaming Hu,
    ORCID 0009-0003-1406-0485, Independent Researcher Guiyang, 64687555@qq.com;
    A→IJBC/Chaos，B→Neural Networks，C→Neurocomputing，D→arXiv/workshop 摘要
    注记；均含原创性声明 + 代码可用性 + 诚实定位）。
  - **投稿清单更新**：A/B/C README 作者占位项勾选完成 + 引用 COVER_LETTER；
    paper_d/README 新增 Submission checklist（标题/作者/编译/证据/诚实范围
    已勾；可选 related-work 段 + 机制消融表留给用户）。
