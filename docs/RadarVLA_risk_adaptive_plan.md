# RadarVLA：Doppler-Grounded Risk-Adaptive Radar-Language-Action

> **暂定题目**：**RadarVLA: Doppler-Grounded Risk-Adaptive Radar-Language-Action Modeling for Autonomous Driving**

## 1. 核心问题

现有自动驾驶 VLA 可以从感知输入与驾驶指令直接生成轨迹，但动态物理状态和安全风险通常隐式存在于视觉 token 中。

对于高速接近、前车急刹、cut-in、遮挡等场景，规划真正关心的是：

- 目标在哪里；
- 目标如何运动；
- 如果 ego 不及时干预，未来几秒是否会形成冲突；
- 当前应该优先进行紧急安全响应，还是进行更长期的路线规划。

毫米波 Radar 对这一问题具有天然优势：

- Range 直接提供 metric distance；
- Doppler 直接提供 radial relative velocity；
- 连续时序 Radar 可以进一步恢复完整运动状态；
- 距离与相对速度正是碰撞风险和紧急制动判断的关键物理量。

因此本文研究：

> **能否利用连续 Range-Azimuth Power-Doppler Radar，在生成驾驶轨迹之前显式估计运动学风险，并把风险状态作为条件输入 LLM，使模型根据当前安全状态自适应调整推理与规划策略？**

核心链条为：

$$
\boxed{ \text{Radar Observation} \rightarrow \text{Physical State} \rightarrow \text{Kinematic Risk} \rightarrow \text{Risk-Adaptive Reasoning} \rightarrow \text{Ego Action} }
$$

---

## 2. 输入

单帧 Radar 为 Range-Azimuth 双通道表示：

$$
X_t\in\mathbb{R}^{R\times A\times2},
$$

其中：

$$
X_t(:,:,1)=Power,\qquad X_t(:,:,2)=Doppler.
$$

连续 $N$ 帧输入为：

$$
X_{t-N+1:t}\in\mathbb{R}^{N\times R\times A\times2}.
$$

模型同时接收 ego state：

$$
E_t= [v_{ego},a_{ego},\dot\psi],
$$

以及导航/驾驶语言指令 $I$。

---

## 3. 总体框架

整体保持三个主体模块：

```text
Multi-frame RA Radar
 Power + Doppler
        │
        ▼
 Shared RA Radar Encoder
        │
        ▼
   Compact Radar Tokens
        │
        ├─────────────────────┐
        │                     │
        │                     ▼
        │          Doppler-Grounded
        │           Risk Estimator
        │                     │
        │                     ▼
        │            Kinematic Risk State
        │          Pcol / dmin / areq
        │                     │
        └──────────────┬──────┘
                       ▼
             LLM + Instruction
                       │
                       ▼
          Risk-Adaptive Structured
                 Generation
                       │
             ┌─────────┴─────────┐
             │                   │
       safety-prioritized   deliberative
          response          long-horizon plan
```

这里不使用 Risk Head 对两个独立模型进行硬路由。

Risk State 只是 LLM 的显式条件变量：

$$
\boxed{ P(Y|Z_{radar},Z_{risk},I) }
$$

同一个 LLM 根据当前风险状态决定应该优先关注紧急安全响应，还是完整的长期结构化规划。

---

## 4. Radar Encoder

### 4.1 RA 双通道联合输入

Power 与 Doppler 是同一个 $(r,\theta)$ cell 的两个物理属性，因此主模型直接联合编码：

$$
X_t(r,\theta) = [P(r,\theta),D(r,\theta)].
$$

Power 和 Doppler 分别归一化后组成：

$$
\tilde X_t = [\tilde P_t,\tilde D_t].
$$

其中 Power 可以进行 log compression：

$$
P'=\log(1+\alpha P).
$$

主模型不采用 Power/Doppler 双 backbone；双分支只作为 ablation。

### 4.2 Shared Spatial Backbone

每帧 RA map 经过共享二维 Radar backbone：

$$
F_t=f_{radar}(\tilde X_t),
$$

得到：

$$
F_t\in\mathbb R^{R'\times A'\times C}.
$$

Backbone 可采用轻量 CNN / ConvNeXt-style blocks。

### 4.3 Polar-Temporal Positional Encoding

RA feature 位于物理极坐标：

$$
(r,\theta).
$$

加入：

$$
PE(r,\theta,\Delta t) = MLP([ \tilde r, \sin\theta, \cos\theta, \Delta t ]).
$$

得到：

$$
F'_{t,r,a} = F_{t,r,a} + PE(r,\theta,\Delta t).
$$

### 4.4 Temporal Tokenization

使用固定数量的 learnable Radar queries：

$$
Q=\{q_1,\dots,q_M\},
$$

对连续多帧 RA feature 做 cross-attention：

$$
Z_{radar} = TemporalTokenizer(Q,F'_{1:N}),
$$

最终得到：

$$
Z_{radar}\in\mathbb R^{M\times C}.
$$

这个模块同时实现：

$$
\boxed{ Temporal\ Aggregation + Token\ Compression }
$$

最后通过 MLP projector 映射到 LLM hidden space。

因此 Radar Encoder 主线为：

$$
\boxed{ RA(Power,Doppler) \rightarrow Shared\ Spatial\ Encoding \rightarrow Polar+Temporal\ Encoding \rightarrow Compact\ Radar\ Tokens }
$$

---

## 5. Doppler-Grounded Risk Estimator

### 5.1 设计原则

不直接训练一个人为定义的：

$$
\rho\in[0,1].
$$

绝对危险分数往往依赖人为函数、权重和阈值，难以保证其物理含义。

本文定义一个可自动监督的：

$$
\boxed{ \textbf{Kinematic Risk State (KRS)} }
$$

由少量具有明确驾驶意义的物理风险量构成：

$$
\boxed{ R_t= [ P_{col}^{1s}, P_{col}^{2s}, P_{col}^{3s}, d_{\min}, a_{\text{req}} ] }
$$

其中：

- $P_{col}^{1s/2s/3s}$：未来不同时间尺度内发生安全冲突的概率；
- $d_{\min}$：在当前不增加安全干预时，预测未来最小间距；
- $a_{\text{req}}$：避免潜在冲突所需的最小纵向减速度。

这些量比单一 TTC 或人工 risk score 更直接地描述：

> **危险是否存在、多久以后出现、以及需要多强的响应。**

### 5.2 网络结构

Risk Estimator 不引入额外大 backbone。

在 Radar tokens 上增加一个 learnable risk query：

$$
q_{risk}\in\mathbb R^C.
$$

通过 cross-attention 聚合与风险最相关的 Radar 信息：

$$
h_{risk} = CrossAttn( q_{risk}, Z_{radar}, Z_{radar} ).
$$

同时编码 ego state：

$$
h_{ego} = MLP(E_t).
$$

融合后：

$$
h = MLP([h_{risk},h_{ego}]).
$$

再通过轻量 prediction head 输出：

$$
\hat R_t = [ \hat P_{col}^{1s}, \hat P_{col}^{2s}, \hat P_{col}^{3s}, \hat d_{\min}, \hat a_{\text{req}} ].
$$

```text
Radar Tokens
     │
     ▼
 <Risk Query>
 Cross-Attention
     │
     ▼
 Risk Feature
     │
  + Ego State
     │
     ▼
   Small MLP
     │
 ┌───┼───────────────┐
 ▼   ▼               ▼
Pcol dmin            areq
```

因此 Risk Estimator 本质上只是 Radar Encoder 后的一个轻量 planning-facing perception head。

---

## 6. Risk Supervision：Counterfactual No-Intervention Risk

风险监督不需要人工标注“危险/安全”。

核心思想是：

$$
\boxed{ \textbf{Counterfactual No-Intervention Risk} }
$$

即：

> **如果 ego 从当前时刻开始不采取新的安全干预，未来几秒会有多危险？**

不能直接用真实 ego future 计算风险，因为真实驾驶员可能已经提前减速或避障，从而把原始风险消除了。

### 6.1 Nominal Ego Rollout

根据当前 ego state 构造短期 nominal trajectory：

$$
\tilde T_{ego} = f_{nominal} ( p_t, v_t, \psi_t, \dot\psi_t ).
$$

第一版可以采用 constant velocity / CTRV rollout。

### 6.2 Future Collision Labels

利用未来 GT agent trajectory / bounding boxes：

$$
T_i^{GT} = \{ B_i^{t+1},\dots,B_i^{t+H} \},
$$

计算 nominal ego box：

$$
\tilde B_{ego}^{t+\tau}.
$$

定义：

$$
y_{col}^{h} = \mathbf 1 [ \exists \tau\le h: SafetyViolation( \tilde B_{ego}^{t+\tau}, B_i^{t+\tau} ) ].
$$

得到：

$$
y_{col}^{1s}, \quad y_{col}^{2s}, \quad y_{col}^{3s}.
$$

SafetyViolation 可以使用 bounding-box overlap 或带 safety margin 的 envelope overlap。

### 6.3 Minimum Future Distance

计算：

$$
d_{\min} = \min_{i,\tau} d( \tilde B_{ego}^{t+\tau}, B_i^{t+\tau} ).
$$

它提供连续的 safety margin 监督。

### 6.4 Required Deceleration

进一步定义：

$$
a_{\text{req}}.
$$

其含义是：

> **从当前状态开始，为避免未来安全冲突所需要的最小纵向减速度。**

训练标签可通过离线数值搜索生成：

$$
a_{\text{req}} = \min_a a
$$

subject to：

$$
d_{\min} ( T_{ego}(a), T_{agents}^{GT} ) \ge d_{safe}.
$$

对于简单纵向场景，它与 DRAC 类似：

$$
a_{\text{req}} \approx \frac{(\Delta v)^2}{2d}.
$$

但训练标签生成阶段采用 rollout search 更通用。

---

## 7. Risk Loss

Risk Estimator 的损失保持简单：

$$
\boxed{ \mathcal L_{risk} = \mathcal L_{col} + \lambda_d\mathcal L_{dist} + \lambda_a\mathcal L_{brake} + \lambda_m\mathcal L_{mono} }
$$

### Collision Risk

$$
\mathcal L_{col} = \sum_{h\in\{1,2,3\}} FocalLoss( \hat P_{col}^{h}, y_{col}^{h} ).
$$

### Minimum Distance

$$
\mathcal L_{dist} = Huber( \hat d_{\min}, d_{\min}^{GT} ).
$$

### Required Deceleration

$$
\mathcal L_{brake} = Huber( \hat a_{\text{req}}, a_{\text{req}}^{GT} ).
$$

### Temporal Monotonicity

理论上：

$$
P_{col}^{1s} \le P_{col}^{2s} \le P_{col}^{3s}.
$$

因此增加：

$$
\mathcal L_{mono} = ReLU( P_{col}^{1s}-P_{col}^{2s} ) + ReLU( P_{col}^{2s}-P_{col}^{3s} ).
$$

---

## 8. Risk Token

KRS 不直接作为文本字符串输入 LLM。

将连续风险状态编码为：

$$
Z_{risk} = MLP_{risk}( \hat R_t ) \in \mathbb R^{d_{LLM}}.
$$

最终 LLM 输入为：

$$
\boxed{ [ Z_{radar}; Z_{risk}; Z_{instruction} ] }
$$

Risk State 不是独立 router，而是 LLM 的显式 conditioning variable。

---

## 9. Risk-Adaptive Structured Generation

LLM 学习：

$$
P(Y| Z_{radar}, Z_{risk}, I).
$$

核心不是固定地把场景分成两个不同模型，而是让同一个 LLM 根据 Risk State 改变生成策略。

### 9.1 低风险场景

风险较低时，模型可以进行完整的结构化长期规划：

$$
\boxed{ Road \rightarrow Agents \rightarrow AgentFuture \rightarrow EgoFuture }
$$

主要关注：

- route / instruction following；
- 长期 trajectory quality；
- comfort；
- smoothness。

### 9.2 高风险场景

风险明显升高时，模型优先关注：

$$
\boxed{ Critical\ Dynamics \rightarrow Immediate\ Safety\ Action }
$$

减少不必要的长规划推理，重点生成短时域安全轨迹。

这里的“快/慢”不是两套网络，而是：

$$
\boxed{ \textbf{Risk-Adaptive Reasoning Budget} }
$$

即同一个模型根据 KRS 自适应调整安全优先级、规划时域和输出长度。

---

## 10. Structured Physical Generation

正常长期规划仍保持原有结构化输出。

### Road

$$
Y_{road} = \{ C_{center}, w_{road} \}.
$$

使用少量 spline control points 表示局部可行驶走廊。

### Agents

第 $i$ 个 agent：

$$
A_i = \{ x_i,y_i, v_{r,i}, v_{x,i},v_{y,i} \}.
$$

### Agent Future

$$
\hat T_i = \{ c_i^1,\dots,c_i^K \}.
$$

由固定 spline/interpolation decoder 恢复完整 trajectory。

### Ego Future

$$
T_{ego} = \{ e^1,\dots,e^K \}.
$$

完整 ego trajectory 同样由确定性 spline decoder 恢复。

---

## 11. Physical Tokenization

连续物理量使用 discrete physical tokens。

例如：

$$
x \rightarrow \langle X_k\rangle, \qquad v \rightarrow \langle V_k\rangle.
$$

空间坐标采用 signed-log companding：

$$
z' = sign(z)\log(1+\alpha|z|),
$$

使近距离区域获得更高分辨率。

训练时使用 Gaussian soft target：

$$
q_j \propto \exp \left( -\frac{(z_j-z_{gt})^2} {2\sigma^2} \right).
$$

---

## 12. Physics-Consistent Loss

完整 supervised loss 为：

$$
\boxed{ \mathcal L = \mathcal L_{token} + \lambda_r\mathcal L_{risk} + \lambda_d\mathcal L_{doppler} + \lambda_k\mathcal L_{kin} + \lambda_t\mathcal L_{traj} }
$$

### Doppler Consistency

$$
\hat v_r = \hat v_x\cos\theta + \hat v_y\sin\theta.
$$

$$
\mathcal L_{doppler} = Huber( \hat v_r-v_r^{radar} ).
$$

### Kinematic Consistency

$$
\mathcal L_{kin} = \sum_t \| \hat p_{t+\Delta t} - ( \hat p_t+\hat v_t\Delta t ) \|_1.
$$

### Trajectory Loss

$$
\mathcal L_{traj} = \mathcal L_{agent} + \beta\mathcal L_{ego} + \gamma\mathcal L_{smooth}.
$$

其中：

$$
\mathcal L_{smooth} = \sum_t\|a_t\|^2 + \rho \sum_t\|j_t\|^2.
$$

---

## 13. Training Strategy

### Stage 1：Physical + Risk Grounding

首先训练：

$$
Radar \rightarrow PhysicalState
$$

以及：

$$
Radar \rightarrow KinematicRiskState.
$$

重点监督：

- Agent position；
- Doppler / full velocity；
- Agent Future；
- $P_{col}^{1/2/3s}$；
- $d_{\min}$；
- $a_{\text{req}}$。

目标是先证明：

$$
\boxed{ Radar \rightarrow Reliable\ Physical\ Risk }
$$

成立。

### Stage 2：Risk-Conditioned Radar-Language-Action SFT

加入：

- Radar Tokens；
- predicted Risk Token；
- language instruction；
- Ego Future。

训练：

$$
\boxed{ Radar + Risk + Instruction \rightarrow StructuredReasoning \rightarrow EgoAction }
$$

训练初期可混合 GT Risk 与 predicted Risk，随后逐渐过渡到 predicted Risk，减少训练/推理分布差异。

### Stage 3：Adaptive Reasoning RFT（可选）

如果希望进一步学习“什么时候应该缩短推理、什么时候值得做完整规划”，可进行闭环 reinforcement fine-tuning。

$$
R = w_sR_{safety} + w_pR_{progress} + w_cR_{comfort} + w_iR_{instruction} - w_lR_{latency}.
$$

该阶段作为增强，不是基础方案成立的必要条件。

---

## 14. 核心实验

### Q1：Doppler 是否改善物理动态理解？

比较：

- Power only；
- Power + Doppler；
- single-frame；
- multi-frame。

指标：

- position error；
- radial velocity MAE；
- full velocity RMSE；
- Agent ADE/FDE。

### Q2：Radar 是否能够可靠预测 Kinematic Risk？

评估：

$$
P_{col}^{1/2/3s}, \quad d_{\min}, \quad a_{\text{req}}.
$$

指标：

- AUROC / AUPRC；
- Brier Score / ECE；
- $d_{\min}$ MAE；
- $a_{\text{req}}$ MAE。

比较：

- Power only；
- Power + Doppler；
- single-frame；
- multi-frame；
- without ego state。

### Q3：Risk State 是否真正改善 Action？

比较：

1. RadarVLA without Risk Token；
2. RadarVLA + predicted Risk Token；
3. RadarVLA + GT Risk Token（oracle）。

指标：

- collision rate；
- minimum TTC；
- brake onset time；
- hard-brake frequency；
- jerk；
- route completion；
- Ego ADE/FDE。

### Q4：Action 是否 Grounded in Doppler and Risk？

固定：

- Power；
- 目标位置；
- language instruction。

只改变：

$$
v_r=-2,-5,-8\ {\rm m/s}.
$$

观察：

$$
P_{col}, d_{\min}, a_{\text{req}}, AgentFuture, EgoFuture.
$$

希望得到：

$$
|v_r|\uparrow \Rightarrow Risk\uparrow \Rightarrow Earlier\ Deceleration.
$$

### Q5：Risk-Adaptive Reasoning 是否有效？

比较：

- always long structured reasoning；
- always short response；
- risk-conditioned adaptive reasoning。

评估：

- safety；
- trajectory quality；
- instruction following；
- average output tokens；
- inference latency。

---

## 15. 数据与标签生成

### 真实 Radar 数据

主要用于：

- Power-Doppler grounding；
- velocity estimation；
- temporal motion modeling；
- Agent Future；
- adverse-weather robustness。

如果提供未来 object trajectory / tracking annotation，可以离线生成：

$$
d_{\min}, P_{col}, a_{\text{req}}.
$$

如果没有完整 future GT，可以利用 LiDAR / tracking teacher 离线构造 risk label。

### CARLA / Closed-loop Simulator

适合自动生成：

- future agent boxes；
- nominal ego rollout；
- collision labels；
- minimum distance；
- required deceleration。

同时用于：

- sudden brake；
- cut-in；
- crossing；
- collision avoidance；
- risk-conditioned planning；
- closed-loop evaluation。

真实恶劣天气 claim 仍应主要由真实 Radar 数据支撑。

---

## 16. 核心贡献

### Contribution 1：Doppler-Grounded Radar Tokenization

针对连续 Range-Azimuth Power-Doppler 双通道输入，采用共享 Radar backbone 联合建模距离、反射与径向运动，并通过 temporal tokenization 得到紧凑 Radar tokens。

### Contribution 2：Physically Supervised Kinematic Risk State

提出 Doppler-Grounded Risk Estimator，不学习人为危险分数，而通过 counterfactual no-intervention rollout 自动构造监督，显式预测：

$$
P_{col}^{1/2/3s}, d_{\min}, a_{\text{req}}.
$$

### Contribution 3：Risk-Adaptive Radar-Language-Action

将 Kinematic Risk State 编码为 Risk Token，与 Radar Tokens 和语言指令共同输入同一个 LLM，使模型根据环境安全状态自适应调整推理预算和规划重点，并生成安全、平滑的 ego trajectory。

---

## 17. 与相关工作的区别

### AutoVLA

AutoVLA 证明单一 VLA 可以学习 fast / slow reasoning，并使用强化微调减少不必要推理。

RadarVLA 的区别是：

$$
\boxed{ \text{Reasoning adaptation is explicitly conditioned on physically supervised Radar risk.} }
$$

### Risk-Aware Occupancy

Risk-Aware Occupancy 证明显式 planning-facing risk representation 可以改善安全规划。

RadarVLA 不构造复杂 dense occupancy risk map，而利用 Radar 最擅长的 range / Doppler，预测紧凑 Kinematic Risk State。

### DRiF

DRiF 指出人工绝对风险分数依赖人为函数、系数和阈值。

因此 RadarVLA 不把人工定义的 $\rho$ 作为监督目标，而预测具有物理含义和自动标签的：

$$
P_{col}, d_{\min}, a_{\text{req}}.
$$

---

## 18. 最终论文主线

整篇论文围绕：

$$
\boxed{ Power+Doppler \rightarrow Kinematics \rightarrow KinematicRisk \rightarrow RiskAdaptiveReasoning \rightarrow EgoAction }
$$

其中：

- **Power + Range** 提供目标位置与空间结构；
- **Doppler** 提供直接 closing-motion 先验；
- **Temporal Radar** 恢复完整运动状态；
- **Risk Estimator** 回答“如果现在不干预，未来有多危险”；
- **Risk Token** 告诉 LLM 当前安全优先级；
- **LLM** 根据风险状态和语言指令自适应生成短时安全响应或长期结构化规划；
- **Ego Future** 是最终 action representation。

最终定位：

> **RadarVLA 将连续 Range-Azimuth Power-Doppler Radar 转化为结构化物理状态和可监督的 Kinematic Risk State，并以风险状态显式条件化 LLM 的驾驶推理，使 Radar 原生的距离与速度信息不仅参与运动理解，还直接影响推理策略和最终驾驶 action。**

---

## 19. 参考文献建议

1. Zhou et al., **AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning**, NeurIPS 2025 / arXiv:2506.13757.
2. Chen et al., **Risk-Aware Occupancy for Safety-Oriented End-to-End Autonomous Driving**, arXiv:2609.21470, 2026.
3. Tian et al., **Data-Driven Risk Fields for Safer End-to-End Autonomous Driving**, arXiv:2609.10377, 2026.
4. Han et al., **Radar4D-VLM: Proposal-Grounded Temporal 4D Radar Reasoning Across Frozen Language Models**, arXiv:2608.04130, 2026.
5. Kung et al., **STAR-VLM: Spatiotemporal Grounding Vision-Language Models for Motion and Velocity Estimation via Automotive Radar Supervision**, arXiv:2608.01535, 2026.
6. Paek et al., **K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions**, 2022.
