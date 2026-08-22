# Paper A — First Draft

## Memory and chaos in a physics-constrained relaxation substrate: phase diagram, multi-timescale forgetting, and disturbance robustness

*(Working title. Target: *International Journal of Bifurcation and Chaos* (World Scientific) or *Chaos: An Interdisciplinary Journal of Nonlinear Science* (AIP). Single author. Companion algorithm paper: Paper B, "REDEM: Training-Inference Unified Learning…" (target: *Neural Networks*).)*

---

## Abstract

Reservoir computing exploits the fading-memory dynamics of a physical substrate, yet the memory–chaos trade-off is usually studied on abstract recurrent networks with hand-tuned leak rates. We characterize a physics-constrained relaxation substrate inspired by Si$_3$N$_4$ shallow-trap charge storage: $N$ units with a log-normal time-constant spectrum (median $\tau_0 \approx 174\,\mu$s, CV $=0.20$) coupled through a per-pulse, topology-dependent modulation of the injection coefficient. Sweeping the coupling strength $\kappa$ across five topology families and measuring the finite-time Lyapunov exponent $\lambda$, the held-out Jaeger memory capacity MC, and input separation, we find a sharp order–chaos transition at $\kappa^\ast \in (25,30)$, with the held-out memory capacity peaking 24–53% above the uncoupled baseline just before the transition, and deep chaos destroying memory. The decay of linear memory follows an analytically derived forgetting kernel $M(t)=\int p(\tau)e^{-t/\tau}\,d\tau$ over the log-normal trap spectrum (Pearson $r=0.97$ against the measured memory-capacity curve): the $1/e$ horizon is pinned near $\tau_0/\bar{\Delta t}\approx 16$ pulses regardless of spectrum width, while the width CV controls the tail weight. Finally, a homeostatic regulator that estimates $\lambda$ online and adjusts $\kappa$ to a near-critical target improves post-disturbance held-out memory by 8–18% under temperature drift, edge damage, and readout noise.

---

## 1. Introduction

Physical reservoir computing proposes that a complex dynamical substrate — a spiking neural microcircuit [1], an optoelectronic system [2], a memristive array [3], or a charge-trap device [4] — can serve as a nonlinear, fading-memory kernel whose instantaneous state is linearly read out by a trained network. The theoretical foundations are well established: echo state networks [5] and liquid state machines [1] rely on the *echo state property* (contracting, input-forgetting dynamics), and the *edge of chaos* maximizes the computational capability of driven recurrent systems [6]. Dambre et al. [7] formalized the total information-processing capacity of dynamical systems, showing a universal trade-off between linear memory and nonlinear processing. On the memory side, multi-timescale synaptic models [8, 9] show that memory retention is dramatically extended by a *spectrum* of time constants: cascade models achieve near power-law forgetting from many exponentially decaying components.

For a physical reservoir, the time-constant spectrum is not a free hyperparameter — it is a *material property*. Shallow-trap charge storage in Si$_3$N$_4$ exhibits thermally activated relaxation with a log-normal spread of trap time constants; our prior work [4] established that this substrate encodes pulse-interval patterns through exponential relaxation with $\tau_0 \approx 174\,\mu$s at 300 K and a device-to-device spread of CV $=0.20$, achieving 79.6% four-class hybrid encoding accuracy in a purely numerical (Digital-Model) setting. That work, however, treated the substrate as a *parallel* array of independent devices; the coupling between devices — the element that could push the substrate toward richer, near-critical dynamics — was only implemented as a quasi-static modulation of the effective injection coefficient across blocks.

This paper closes that gap. We introduce the *per-pulse temporal generalization* of the topology coupling: every pulse, each unit's injection coefficient is modulated by a topology-dependent contrast against its neighbors' current ratios, making the substrate a genuinely recurrent, tunable dynamical system with a physical chaos knob $\kappa$. Our contributions are:

1. **A memory–chaos phase diagram** for a physics-constrained relaxation substrate: the finite-time Lyapunov exponent $\lambda$, the held-out Jaeger memory capacity MC, and the input separation, all as functions of the coupling strength $\kappa$, across five topology families (bidirectional ring, unidirectional ring, hub-and-spoke, lateral-inhibition ring, random graph) and two coupling sign conventions (negative-feedback contrast and positive-feedback contrast), with an additive state-coupling control.
2. **An analytic forgetting kernel** $M(t)=\int p(\tau)e^{-t/\tau}\,d\tau$ for the log-normal trap spectrum, validated against the measured memory-capacity curve (Pearson $r=0.97$), which turns the device-physics spread CV into a *design knob for the forgetting curve*.
3. **A homeostatic chaos regulator** that estimates $\lambda$ online from Benettin twin trajectories and adjusts $\kappa$ to a near-critical target, restoring 8–18% of held-out memory after environmental disturbances (temperature drift, edge damage, readout noise) where fixed coupling cannot.

Throughout, we use a *held-out* memory-capacity protocol: ridge fits are trained on the first 70% of a driving stream and evaluated on the last 30% (with a $k_{\max}$-length leakage buffer). We show that the common in-sample Jaeger convention dramatically overstates memory in the chaotic regime, which is a methodological caution for the field.

## 2. Substrate model

### 2.1 Single-unit relaxation

Each unit models a Si$_3$N$_4$ shallow-trap population with effective trap occupancy $x_i \in [0,1]$. A write pulse of width $p_w=1\,\mu$s injects charge with coefficient $\alpha$; the occupancy then relaxes exponentially with the unit's trap time constant $\tau_i$ both during the pulse width and during the inter-pulse interval $\Delta t_t$. The per-pulse update is

$$x_i(t+1) = \Big[x_i(t) + \alpha_{\text{eff},i}(t)\big(1-x_i(t)\big)\Big]\, e^{-p_w/\tau_i} e^{-\Delta t_t/\tau_i}, \quad x_i \in [0,1], \tag{1}$$

and the observable current ratio is $i_i = e^{\gamma x_i}$ with $\gamma=\ln 100$ (a 100:1 ON/OFF ratio at $I_{\text{HRS}}=50$ fA). The trap time constants are drawn from a log-normal distribution with median $\tau_0=174\,\mu$s (from $E_a=0.55$ eV, $\nu=10^{13}$ s$^{-1}$, $T=300$ K) and CV $=0.20$: $\tau_i \sim \text{Lognormal}(\mu,\sigma)$ with $\sigma^2=\ln(1+\text{CV}^2)$; the log-normal form follows from a Gaussian spread of activation energies (Appendix A.1).

### 2.2 Per-pulse topology coupling

The injection coefficient of unit $i$ at pulse $t$ is modulated by a topology-dependent contrast between its own current ratio and a weighted mean of its neighbors' current ratios:

$$\alpha_{\text{eff},i}(t) = \text{clip}\Big(\alpha_0\big(1 + \kappa\, g_i(t)\big),\, \alpha_{\min}, \alpha_{\max}\Big), \tag{2}$$

with two contrast conventions — *negative feedback* (mode 1, "v4 style"): $g_i=(\bar{i}_{\mathcal N(i)} - i_i)/i_i$; and *positive feedback* (mode 2, "Phase-3/hub style"): $g_i = (i_i - \bar{i}_{\mathcal N(i)})/\bar{i}_{\mathcal N(i)}$ — where $\bar{i}_{\mathcal N(i)}$ is the weighted mean of neighbor current ratios (row-normalized weights). The physical clip range $[\alpha_{\min},\alpha_{\max}]=[0.001,0.10]$ bounds the injection coefficient. A control condition replaces the contrast coupling with an additive state drive $x_i \mathrel{+}= \kappa\,\bar{x}_{\mathcal N(i)}$ after the standard update (the non-physical, "textbook ESN" style). All coupling is computed two-pass (contrasts from the pre-update state, then all units updated), making the step order-independent (Appendix A.2). At $\kappa=0$ the model reduces exactly to the parallel array of [4] (verified to machine precision).

Five topologies over the $N=256$ units are used: **ring_bidir** (each unit watches its two neighbors), **ring_unidir** (each unit watches its predecessor), **hub_star** (satellites watch a fixed hub), **lateral_ring** (distance-decayed neighbors within radius 4), and **random_graph** (Erdős–Rényi, mean degree 8, fixed structure across the sweep). The drive is an i.i.d. uniform interval stream $\Delta t_t \sim U[2,20]\,\mu$s, the fast-drive regime in which the nominal memory window is $N_{\text{eff}} = \tau_0/\bar{\Delta t} \approx 17$ pulses.

## 3. Characterization methods

**Finite-time Lyapunov exponent.** For a driven system the relevant quantity is sensitivity to initial conditions under identical drive. We evolve a twin trajectory perturbed by $\epsilon=10^{-8}$ per unit, renormalize the separation to $\epsilon\sqrt{N}$ every 10 pulses (Benettin scheme), and report $\lambda = \langle \ln(d/\epsilon\sqrt{N})\rangle$ per pulse, averaged over the full stream (detailed iteration steps in Appendix A.4).

**Held-out memory capacity.** Following Jaeger, the linear memory of lag $k$ is the squared Pearson correlation between the ridge-decoded value of the driving interval $\Delta t_{t-k}$ and its true value, decoded from the current-ratio observables at time $t$. We fit the ridge readout (regularization $\lambda_r=1$, features standardized with training-segment statistics) on the first 70% of the collected stream, hold out a $k_{\max}=50$-step leakage buffer, and evaluate on the last 30%; $\text{MC}_{\text{total}} = \sum_{k=1}^{50} r_k^2$. The held-out protocol is essential: the in-sample convention inflates memory in the chaotic regime by 2–3× (Section 4.3). The estimator derivation and the linear-reservoir closed form that anchors the theory are given in Appendix A.5.

**Input separation.** The RMS distance between the state trajectories produced by two different i.i.d. drives from the same initial condition, averaged over the collected window — a proxy for how far different input histories push the state.

**Activity proxy.** The running standard deviation of the population-mean current ratio over a 200-pulse window (used in the homeostat analysis).

## 4. Phase diagram

All quantities below are means over 10 independent seeds (paired $\tau$ draws across $\kappa$); the full table is in Supplementary Table S1 (data/substrate_phase_diagram_v2.csv).

### 4.1 Negative-feedback family: stabilization then a sharp transition

For the mode-1 (negative-feedback) topologies, weak-to-moderate coupling *stabilizes*: $\lambda$ becomes more negative ($-0.065 \to -0.090$ as $\kappa$ grows to 10 for the ring) and the mean absolute contrast $|g|$ shrinks ($0.15 \to 0.01$), i.e., the coupling actively homogenizes neighboring states. Memory rises modestly in this regime (ring_bidir held-out MC $9.1 \to 10.6$ at $\kappa=10$). At $\kappa \in (25, 30)$ the system crosses a sharp order–chaos transition ($\lambda$ jumps from $\approx -0.05$ to $>0$) for all three mode-1 topologies; the transition is accompanied by the onset of $\alpha_{\text{eff}}$ clipping (clip fraction 0.3–0.6), i.e., the physical bounds start to bind. **The held-out memory capacity peaks in the last ordered configuration before the transition**: random_graph at $\kappa=25$ reaches MC $=13.91$ vs $9.07$ uncoupled ($+53\%$), ring_bidir at $\kappa=20$ $+30\%$, lateral_ring at $\kappa=20$ $+24\%$.

### 4.2 Positive-feedback family: self-limited criticality, separation without linear memory

The mode-2 (positive-feedback) topologies behave very differently. The **unidirectional ring** self-limits at criticality: $\lambda$ approaches $\approx -0.002$ (within our FTLE resolution of zero) over a full decade of coupling strength $\kappa\in[1,10]$, held there by the $\alpha_{\text{eff}}$ clip (clip fraction 0.5–1.0) interacting with the $(1-x)$ saturation. This regime shows the *largest* input separation of the entire study (inter-stream RMS 0.054–0.085 vs 0.022 baseline) but *zero* held-out linear memory (MC $0.12$–$0.39$, despite in-sample MC of 18–20). The exponential current nonlinearity makes the saturated critical states information-rich but linearly undecodable — a clean demonstration of the separation–memory trade-off: separation-rich critical states require nonlinear readout to be exploited. The **hub-and-spoke** topology freezes instead: with no feedback loop through the hub, satellites saturate into bistable above/below-hub states (clip fraction 1.0) and memory never exceeds the baseline.

### 4.3 Chaos destroys memory; in-sample MC is misleading

In the chaotic regime ($\kappa=50,100$), the held-out MC collapses to 1.85–4.4 (a 3–7× drop from the peak). The in-sample (train-segment) MC, by contrast, *explodes* to 30–35 — an overfitting artifact of ridge decoding the huge-amplitude chaotic state; the held-out protocol reveals the true destruction of memory. We therefore report held-out MC throughout and recommend the practice to the field.

### 4.4 The additive control: why physics constraints matter

The additive state-coupling control (mode 3, textbook ESN style) reaches its best in-sample MC (17–22) near its saturation threshold ($\kappa\approx 0.08$–0.1) and then dies: the state pins at the saturation boundary and the FTLE estimate collapses numerically (a signature of degenerate dynamics). Its held-out MC never exceeds the uncoupled baseline meaningfully (max 9.1), and the operating window is a narrow band before saturation. The physics-constrained contrast coupling, by contrast, remains well-behaved across three decades of $\kappa$ and achieves the memory gains above — the physical clip acts as a built-in stability regulator.

## 5. Multi-timescale forgetting theory

### 5.1 The kernel

The linear memory of a single trap for an input at lag $t$ decays as $e^{-t/\tau}$. The substrate's population forgetting kernel is the average over the log-normal spectrum:

$$M(t) = \int_0^\infty p(\tau)\, e^{-t/\tau}\, d\tau, \qquad p(\tau)=\text{Lognormal}(\tau_0,\text{CV}). \tag{3}$$

We evaluate $M(t)$ by Gauss–Hermite quadrature (160 nodes, exact to machine precision; derivation in Appendix A.3). Three properties follow (Fig. 3):

1. **The $1/e$ horizon is median-pinned.** $M(\tau_0/\bar{\Delta t}) \approx e^{-1}$ regardless of CV; the horizon is 12–16 pulses for CV $\in[0.02,1.0]$, in agreement with the nominal window $N_{\text{eff}}\approx 17$. The *median* time constant sets where memory is; the width sets how it decays.
2. **The tail steepens with narrowing spectrum.** The log-log slope $d\ln M/d\ln t$ at long lags is $-60.6$ at CV $=0.2$ and rises to $-7.1$ at CV $=1.0$ (and $-\infty$ in the CV$\to0$ single-exponential limit). A wider trap spread yields a heavier tail — slower forgetting of old information.
3. **Comparison to cascade ideals.** The log-Gaussian tail $\ln M(t)\sim -(\ln t-\mu)^2/(2\sigma^2)$ is *slower* than a single exponential but *steeper* than the near power-law retention of multi-timescale cascade models [8,9]. Reaching power-law-like retention would require an unrealistically wide spectrum; the physical substrate therefore occupies a specific point in the forgetting design space set by CV.

### 5.2 Validation against the measured memory capacity

The measured parallel-substrate curve $\sqrt{\text{MC}(k)}$ (held-out, S1) tracks $M(k\cdot\bar{\Delta t})$ at CV $=0.20$ with **Pearson $r=0.97$**; at lag 10 the two agree almost exactly (0.520 vs 0.523), with systematic damping at short lags (readout regression attenuation for a random drive). The device-physics spectrum thus predicts the substrate's empirical memory curve from first principles.

**Implication.** CV — a fabrication-controllable spread parameter — is a *design knob for the forgetting curve*. This is the quantitative basis for the "physics as a structured state-space kernel" view: the substrate realizes a specific multi-timescale kernel with a tunable tail.

**The coupled-substrate caveat (task-level CV sweep).** The kernel theory is single-unit; at the task level the CV knob's direction is operating-regime-dependent — uncoupled operation follows the theory (wider CV, heavier tail, more retained memory), while coupled near-critical operation prefers narrow spectra. The full sweep (CV ∈ {0.1, 0.2, 0.4} × {uncoupled, random_graph κ=25, ring_bidir κ=20}, 10 seeds) is given in Supplementary Note 1 (Fig. S1); the kernel-level stability across CV ∈ [0.02, 1.0] is in Fig. 3 and Appendix A.3.

## 6. Disturbance robustness and the λ-homeostat

### 6.1 The disturbed memory landscape

We subject the random-graph substrate ($\kappa=25$, the nominal optimum) to three abrupt disturbances at $t=10$k in an online Mackey–Glass forecasting task (RLS readout): **temperature drift** (all $\tau$ scaled by 1.5), **edge damage** (40% of coupling edges removed), and **readout noise** ($\sigma=0.1\times$ feature std). The held-out MC–$\kappa$ landscape shifts with the disturbance: noise reduces memory everywhere (peak 10.05 vs 13.98 nominal), edge damage *improves* memory (16.53 — removing homogenizing edges), and temperature drift moves the optimum to larger $\kappa$. Consequently, any regulator anchored to a *fixed activity target* (a common homeostatic design) fails: the target's optimum is not disturbance-invariant (this negative design result motivated the λ-homeostat).

### 6.2 The λ-homeostat

The homeostat estimates $\lambda$ directly: every 1000 pulses it runs a 400-pulse Benettin pair at the current $\kappa$ (20% computational overhead) and steps

$$\kappa \leftarrow \text{clip}\big(\kappa + \eta\,\text{clip}(\lambda_{\text{target}} - \hat\lambda,\, -1, 1),\, [1,60]\big), \qquad \lambda_{\text{target}} = -0.02, \tag{4}$$

the slightly-ordered side of the transition where held-out MC peaks (Section 4.1). The regulated system settles at $\kappa \approx 26$–27 under all conditions and achieves **post-disturbance held-out MC gains of +7.6% (none), +18% (τ-drift), +7.9% (edge-prune), +11.8% (noise)** over the fixed coupling, at task-level NMSE parity (the RLS readout absorbs task-level differences; the substrate-level memory improvement is revealed by the readout-independent MC probe). The λ-homeostat is disturbance-type-agnostic because it tracks the dynamical quantity itself rather than a proxy whose optimum shifts.

## 7. Discussion

**The physical parameter range bounds the accessible computation.** Three material/design quantities jointly define the substrate's computational envelope: the clip range $[\alpha_{\min},\alpha_{\max}]$ (which bounds coupling and, in the positive-feedback family, self-limits the system at criticality), the coupling threshold $\kappa^\ast$ (which separates the memory-rich ordered side from the memory-destructive chaotic side), and the trap spectrum (which fixes both the $1/e$ memory horizon and the forgetting tail). For the neuromorphic-design question "what can this device compute", the answer is: a bounded but substantial region of the memory–chaos plane, fully characterized here.

**Separation without linear memory.** The positive-feedback family reaches true criticality (|λ|≈0) with maximal separation yet zero held-out linear memory. This is a caution for the common practice of tuning reservoirs to the edge of chaos: at criticality the *linear* readout loses the input trace, and the separation must be exploited nonlinearly (a companion work uses these states for structure-plasticity signals). For linear readouts, the optimal operating point is the *last ordered configuration before the transition* — a slightly-subcritical target (κ = 20–25, λ ≈ −0.05), exactly where the homeostat parks the system.

**The forgetting kernel as design language.** $M(t)$ connects device physics to the state-space-model literature: the substrate is a hardware-realizable structured state-space kernel whose decay spectrum is set by fabrication (CV). The r=0.97 validation shows the connection is quantitative, not metaphorical.

**Limitations.** All results are numerical (Digital-Model level; no fabricated device); the task-level CV sweep reveals an operating-regime-dependent optimum (Supplementary Note 1); the homeostat's gains are substrate-level (task-level NMSE is readout-compensated); FTLE estimates are finite-time and resolution-limited near zero.

## 8. Conclusion

A physics-constrained relaxation substrate with per-pulse contrast coupling possesses a well-defined memory–chaos phase diagram: negative-feedback coupling buys 24–53% more held-out memory up to a sharp transition, positive-feedback coupling self-limits at criticality with maximal separation, chaos destroys linear memory, and the log-normal trap spectrum fixes a forgetting kernel that predicts the measured memory curve (r=0.97) with a fabrication-tunable tail. A λ-homeostat restores 8–18% of memory after disturbances. These results turn a device-physics spread parameter into a computational design knob and provide the substrate-level theory for the REDEM online-learning architecture (companion paper [12]).

---

## Appendix A. Derivation details

### A.1 The trap spectrum is log-normal (Section 2.1)

A shallow trap escapes by thermally activated emission with rate $\nu e^{-E/(kT)}$, so a single trap's time constant is $\tau(E)=\nu^{-1}e^{E/(kT)}$. With median activation energy $E_a$, $\tau_0=\nu^{-1}e^{E_a/(kT)}$; for $E_a=0.55$ eV, $\nu=10^{13}$ s$^{-1}$, $T=300$ K ($kT\approx 25.85$ meV): $\tau_0=10^{-13}e^{0.55/0.02585}\,\text{s}\approx 174\,\mu$s. If the activation energy varies across devices as $E_i=E_a+\Delta E_i$ with $\Delta E_i\sim\mathcal{N}(0,\sigma_E^2)$, then

$$\ln\tau_i=\ln\nu^{-1}+(E_a+\Delta E_i)/(kT)\sim\mathcal{N}\big(\ln\tau_0,\ \sigma_E^2/(kT)^2\big),$$

i.e., $\tau_i$ is log-normal with $\sigma^2=\sigma_E^2/(kT)^2$ and $\text{CV}^2=e^{\sigma^2}-1$. The CV $=0.20$ used throughout corresponds to a spread $\sigma_E=kT\sqrt{\ln(1+0.20^2)}\approx 5.1$ meV — a few-meV fabrication spread — which motivates treating CV as a physically plausible design knob rather than a free hyperparameter.

### A.2 Per-pulse update and the order independence of coupling (Section 2.2)

Over a write pulse of width $p_w$, the injection term drives occupancy toward 1 at rate $\alpha_{\text{eff},i}$; over the inter-pulse interval $\Delta t_t$ both the injected and the pre-existing occupancy relax with the trap time constant. The discrete per-pulse map (Eq. 1) is the composition of one injection step and one relaxation step:

$$x_i(t+1)=\Big[x_i(t)+\alpha_{\text{eff},i}(t)\big(1-x_i(t)\big)\Big]\,e^{-(p_w+\Delta t_t)/\tau_i}.$$

Two-pass order independence: the contrast $g_i(t)$ is evaluated from the *pre-update* state $x(t)$ alone (neighbor current ratios at time $t$), and all $N$ units are then updated simultaneously from the same pre-update state. The map is therefore a well-defined function of the state at $t$ — a parallel map — and the order in which individual unit updates are computed is irrelevant. At $\kappa=0$, $\alpha_{\text{eff},i}=\alpha_0$ for all $i$ and the coupling term vanishes; the map reduces exactly to the parallel array of [4] (verified to $\le 10^{-15}$ in the implementation's self-test).

### A.3 The forgetting kernel and its quadrature (Section 5.1)

For a single trap the linear memory of an input at lag $t$ decays as $e^{-t/\tau}$; averaging over the log-normal spectrum gives Eq. (3). Substituting $z=(\ln\tau-\mu)/(\sqrt{2}\sigma)$, with $d\tau=\sqrt{2}\,\sigma e^{\mu+\sqrt{2}\sigma z}dz$ and $p(\tau)d\tau=\pi^{-1/2}e^{-z^2}dz$:

$$M(t)=\pi^{-1/2}\int_{-\infty}^{\infty}e^{-z^2}\exp\!\big(-t\,e^{-(\mu+\sqrt{2}\sigma z)}\big)\,dz \;\approx\; \pi^{-1/2}\sum_{i=1}^{n} w_i\,\exp\!\big(-t\,e^{-(\mu+\sqrt{2}\sigma z_i)}\big),$$

where $(z_i,w_i)$ are the $n$-point Gauss–Hermite nodes and weights ($n=160$ used; the quadrature is exact to machine precision for $t$ up to $10^4$ pulses).

**Median pinning of the $1/e$ horizon.** The integrand's maximum is at $\tau^*(t)$ solving $d/d\tau\,[\ln p(\tau)-t/\tau]=0$, i.e. $t/\tau^*=1+(\ln\tau^*-\mu)/\sigma^2$. At $t=\tau_0$ the solution is $\tau^*=\tau_0$ up to a weak CV-dependent correction, so $M(\tau_0)\approx e^{-1}$: the $1/e$ horizon is pinned at $\approx\tau_0/\bar{\Delta t}\approx 16$–17 pulses regardless of CV (numerically 12–16 pulses for CV $\in[0.02,1.0]$, Fig. 3).

**Tail asymptotics.** For $t\gg\tau_0$ the saddle point moves to $\tau^*\approx t\,\sigma^2/\ln(t/\tau_0)$, and $\ln M(t)\approx-(\ln\tau^*-\mu)^2/(2\sigma^2)-t/\tau^*\approx-(\ln t-\mu)^2/(2\sigma^2)$ to leading order. The log–log slope therefore asymptotes to

$$\frac{d\ln M}{d\ln t}\approx -\frac{\ln t-\mu}{\sigma^2}.$$

At $t=200$ pulses, CV $=0.2$ ($\sigma^2=0.0392$; $\mu=\ln(\tau_0/\bar{\Delta t})\approx2.76$ in pulse units), this predicts $-64.8$, in agreement with the measured long-lag slope $-60.6$ (the residual difference is the sub-leading $t/\tau^*$ term). Narrowing the spectrum (CV$\to0$) makes $\sigma^2\to0$ and the slope $\to-\infty$: the single-exponential limit.

### A.4 Benettin algorithm (Section 3)

1. Evolve the main trajectory $x(t)$ under the fixed drive stream for the full collection window.
2. At each renormalization point $t_m=m\,T_{\text{ren}}$ ($T_{\text{ren}}=10$ pulses), create the twin $\tilde{x}(t_m)=x(t_m)+\delta$ with $\delta_i\sim U(-\epsilon,\epsilon)$ per unit, $\epsilon=10^{-8}$.
3. Evolve $\tilde{x}$ in parallel with $x$ under the *identical* drive sequence for $T_{\text{ren}}$ pulses.
4. Measure the separation $d_m=\|\tilde{x}(t_{m+1})-x(t_{m+1})\|_2$, accumulate $\ln(d_m/(\epsilon\sqrt{N}))$, and renormalize the twin onto the reference direction: $\tilde{x}\leftarrow x+(\epsilon\sqrt{N})(\tilde{x}-x)/d_m$.
5. After $M$ renormalization blocks, $\lambda\approx(MT_{\text{ren}})^{-1}\sum_{m=1}^{M}\ln(d_m/(\epsilon\sqrt{N}))$ per pulse.

The estimate is finite-time and resolution-limited near zero ($|\lambda|\lesssim10^{-3}$ is indistinguishable from 0); the perturbation $\epsilon$ is small enough that the linearized dynamics dominates within one block.

### A.5 Memory-capacity estimator and the linear-reservoir closed form (Section 3)

For lag $k$ the target is $y_k(t)=\Delta t_{t-k}$ and the features are the standardized current ratios plus bias, $\phi_t$. Ridge regression on the first 70% of the stream solves $W_k=(\Phi^\top\Phi+\lambda_r I)^{-1}\Phi^\top Y_k$ (targets mean-centered); evaluation is on the last 30% after a $k_{\max}=50$-step leakage buffer, so no evaluation target references training-segment inputs. The per-lag capacity is the squared multiple correlation $MC(k)=r_k^2=\text{corr}(y_k,\hat{y}_k)^2$, $MC_{\text{total}}=\sum_{k=1}^{50}r_k^2$.

*Linear closed form.* Linearize the substrate (clip inactive, $\kappa$ small): $x_{t+1}=A x_t+b u_{t+1}$ with $u_t=\Delta t_t$ and $A$ contractive. Then $x_t=\sum_{j\ge0}A^j b\,u_{t-j}$; the input at lag $k$ contributes the component $A^{k-1}b\,u_{t-k}$, whose covariance with $x_t$ is $c_k=A^{k-1}b\,\text{Var}(u)$. The best linear predictor achieves the multiple correlation

$$MC(k)=c_k^\top\,\Sigma_x^{-1}\,c_k,\qquad \Sigma_x=\text{Var}(x_t)=\sum_{j\ge0}A^j bb^\top(A^\top)^j\,\text{Var}(u),$$

and Dambre et al. [7] show the total linear capacity of an $N$-dimensional linear system is bounded by the state dimension: $\sum_k MC(k)\le N$, with equality attained by orthogonally normalized dynamics. The measured nonlinear-substrate curve is estimated with the held-out protocol of Section 3; the closed form anchors the theory at the linear limit.

---

## Supplementary Material

**Supplementary Note 1. Task-level CV sweep (Fig. S1).** At the task level (held-out MC at the memory-optimal operating points, CV ∈ {0.1, 0.2, 0.4}, 10 seeds), the uncoupled substrate follows the single-unit kernel theory: MC rises 2.54 → 3.37 (+33%) as CV widens from 0.1 to 0.4 — the heavier tail retains more old lags. The coupled substrate at the near-critical optimum behaves *oppositely*: random_graph κ=25 MC falls 12.93 → 7.33 as CV widens (narrow spectrum best), and ring_bidir κ=20 falls 9.55 → 7.86. The collective near-critical state prefers homogeneous timescales — heterogeneous relaxation rates disrupt the synchronization of the contrast-feedback loop. The CV knob's direction therefore depends on the operating regime: wide spectra for parallel (kernel-dominated) operation, narrow spectra for coupled near-critical operation. (Fig. S1 is prepared from `data/s10_cv_sweep_v1.csv`.)

---

## References (to complete)

1. Maass, W., Natschläger, T., Markram, H. Real-time computing without stable states: a new framework for neural computation based on perturbations. *Neural Computation* 14(11), 2531–2560 (2002). https://doi.org/10.1162/089976602760407955
2. Larger, L. et al. Photonic information processing beyond Turing: an optoelectronic implementation of reservoir computing. *Opt. Express* 20, 3241–3249 (2012).
3. Du, C., Cai, F., Zidan, M. A., Ma, W., Lee, S. H., Lu, W. D. Reservoir computing using dynamic memristors for temporal information processing. *Nature Communications* 8, 2204 (2017).
4. Author's prior NCE manuscript: Si3N4 pulse-encoding (substrate calibration source).
5. Jaeger, H. The "echo state" approach to analysing and training recurrent neural networks. GMD Report 148 (2001). https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf
6. Bertschinger, N., Natschläger, T. Real-time computation at the edge of chaos in recurrent neural networks. *Neural Computation* 16(7), 1413–1436 (2004). https://doi.org/10.1162/089976604323057443
7. Dambre, J., Verstraeten, D., Schrauwen, B., Massar, S. Information processing capacity of dynamical systems. *Scientific Reports* 2, 514 (2012).
8. Fusi, S., Drew, P. J., Abbott, L. F. Cascade models of synaptically stored memories. *Neuron* 46(4), 599–609 (2005).
9. Benna, M. K., Fusi, S. Computational principles of synaptic memory consolidation. *Nature Neuroscience* 19(12), 1697–1706 (2016). https://doi.org/10.1038/nn.4401
10. Boyd, S., Chua, L. O. Fading memory and the problem of approximating nonlinear operators with Volterra series. *IEEE Trans. Circuits Syst.* 32(11), 1151–1161 (1985).
11. Pathak, J., Hunt, B., Girvan, M., Lu, Z., Ott, E. Model-free prediction of large spatiotemporally chaotic systems from data: a reservoir computing approach. *Phys. Rev. Lett.* 120, 024102 (2018). https://doi.org/10.1103/PhysRevLett.120.024102
12. Author's companion Paper B: REDEM — training-inference unified learning with meta-adaptation and structural plasticity (target: *Neural Networks*).
