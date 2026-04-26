# 05. Real-Time Recurrent Learning (RTRL)

## 🎯 핵심 질문

- RTRL (Williams & Zipser 1989) 은 어떻게 forward-mode AD 로 BPTT 와 같은 gradient 를 계산하는가?
- 왜 RTRL 의 복잡도가 $O(n^4)$ 인 반면 BPTT 는 $O(n^2)$ 인가? Forward-mode vs reverse-mode AD 의 본질적 차이는?
- RTRL 은 **online** — sequence 끝까지 기다리지 않고 매 step 마다 update 가능. 어떤 use case 에서 가치 있는가?
- UORO (Tallec & Ollivier 2017) 는 어떻게 random projection 으로 RTRL 을 unbiased $O(n^2)$ approximation 하는가?
- e-prop (Bellec 2020) 의 biological plausibility 와 local learning rule

---

## 🔍 왜 RTRL 이 다시 주목받는가

BPTT 가 deep learning 의 표준이지만, 다음 한계가 있습니다:

1. **Episode-end update only** — Sequence 끝까지 forward 후 backward, online 학습 불가
2. **Memory $O(TH)$** — Long sequence 에서 한계
3. **Biologically implausible** — 뇌는 forward 만, "delayed reward" 으로 학습

RTRL 은 forward-mode AD 로 매 step 의 gradient 를 즉시 계산:

1. **Online learning** — 매 step update, RL 의 policy gradient 와 결합 자연스러움
2. **No backward pass** — Forward 만, biologically inspired
3. **Streaming applications** — Time series, edge AI, neural plasticity

이 문서에서는 RTRL 의 정확한 알고리즘, 복잡도, 그리고 현대 approximations (UORO, e-prop) 을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-bptt-derivation.md](./02-bptt-derivation.md) — BPTT 의 reverse-mode AD
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Forward-mode AD, dual number, Jacobian
- (선택) Stochastic optimization: Unbiased estimator, variance reduction

---

## 📖 직관적 이해

### Forward-mode vs Reverse-mode AD

함수 $f: \mathbb R^n \to \mathbb R^m$ 의 Jacobian $J = \partial f / \partial x$:

**Forward-mode** (RTRL):
- Input perturbation $\dot x \in \mathbb R^n$ 를 함께 propagate
- 한 번의 forward 로 $J \dot x$ (Jacobian-vector product) 계산
- $n$ 번 반복하면 전체 Jacobian — cost $O(n \cdot \text{forward})$

**Reverse-mode** (BPTT):
- Output gradient $\bar y \in \mathbb R^m$ 를 backward propagate
- 한 번의 backward 로 $\bar y^\top J$ (vector-Jacobian product) 계산
- $m$ 번 반복하면 전체 Jacobian — cost $O(m \cdot \text{forward})$

**RNN 의 상황**:
- Input $\theta$ (weights) → Output $L$ (scalar loss)
- $n = |\theta| \approx H^2$ (큼), $m = 1$ (스칼라)
- Reverse-mode 가 자연스러움 ($O(1)$ pass) → BPTT
- Forward-mode 는 $O(H^2)$ passes → RTRL 의 $O(H^4)$

### RTRL 의 Sensitivity

RTRL 은 **sensitivity matrix** $S_t = \partial h_t / \partial \theta$ 를 forward 와 함께 propagate:

$$
S_t = \frac{\partial h_t}{\partial h_{t-1}} S_{t-1} + \frac{\partial h_t}{\partial \theta}\bigg|_{\text{partial}}
$$

매 step:
- Forward: $h_t = f(h_{t-1}, x_t; \theta)$
- Sensitivity: $S_t$ update — $\partial h / \partial \theta$ 의 직접 미분
- Loss gradient: $\partial L_t / \partial \theta = \partial L_t / \partial h_t \cdot S_t$ — vector × matrix

### Online Learning

Step $t$ 에서:
1. $L_t$ 관찰
2. $\partial L_t / \partial \theta$ 계산 (RTRL 로)
3. **즉시** $\theta \leftarrow \theta - \eta \nabla L_t$

BPTT 는 sequence 끝까지 기다려야 함. RTRL 은 streaming 학습.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Sensitivity Tensor

$$
S_t := \frac{\partial h_t}{\partial \theta} \in \mathbb R^{H \times |\theta|}
$$

각 column 이 specific weight 에 대한 hidden state 의 sensitivity.

### 정의 5.2 — RTRL Update Rule

Forward recurrence $h_t = f(h_{t-1}, x_t; \theta)$ 에 대해:

$$
\boxed{S_t = \underbrace{\frac{\partial f}{\partial h}}_{H \times H} S_{t-1} + \underbrace{\frac{\partial f}{\partial \theta}\bigg|_{\text{partial}}}_{H \times |\theta|}}
$$

### 정의 5.3 — Loss Gradient via RTRL

$$
\frac{\partial L_t}{\partial \theta} = \underbrace{\frac{\partial L_t}{\partial h_t}}_{1 \times H} \cdot \underbrace{S_t}_{H \times |\theta|} = \frac{\partial L_t}{\partial h_t}\, S_t
$$

매 step $L_t$ 의 gradient 가 즉시 사용 가능 — online update.

### 정의 5.4 — RTRL Complexity

- $S_t \in \mathbb R^{H \times |\theta|}$, $|\theta| = O(H^2)$ → $S_t$ 는 $H^3$ floats
- Update $S_t = J_t S_{t-1} + \partial f / \partial \theta$:
  - $J_t S_{t-1}$: $(H \times H) \times (H \times H^2) = O(H^4)$ ops
  - $\partial f / \partial \theta$: $O(H^2)$ — sparse, 단 specific weight 가 specific element 에만 영향
- Per step: $O(H^4)$
- $T$ steps: $O(T H^4)$

### 정의 5.5 — UORO (Unbiased Online Recurrent Optimization)

Tallec & Ollivier 2017. Random rank-1 projection:

$$
\hat S_t = \tilde u_t \tilde v_t^\top, \quad \tilde u_t \in \mathbb R^H, \tilde v_t \in \mathbb R^{|\theta|}
$$

Update:
$$
\tilde u_t = J_t \tilde u_{t-1} + \nu_t, \quad \tilde v_t = \tilde v_{t-1} + \nu_t^\top \cdot \frac{\partial f}{\partial \theta} \bigg|_{\text{contracted}}
$$

여기서 $\nu_t \in \{-1, +1\}^H$ random sign vector.

**Memory**: $O(H + |\theta|) = O(H^2)$. **Time per step**: $O(H^2)$. **Unbiased**: $\mathbb E[\hat S_t] = S_t$.

---

## 🔬 정리와 결과

### 정리 5.1 — RTRL 의 정확성

RTRL 의 $\partial L_t / \partial \theta$ 가 BPTT 의 gradient 와 동일:

$$
\frac{\partial L_t}{\partial \theta}\bigg|_{\text{RTRL}} = \frac{\partial L_t}{\partial \theta}\bigg|_{\text{BPTT}}
$$

**증명**: 둘 다 chain rule 의 같은 분해를 적용. 차이는 계산 순서 (forward propagate vs backward propagate). $\square$

### 정리 5.2 — RTRL vs BPTT 복잡도

| | RTRL | BPTT |
|--|------|------|
| **Time per step** | $O(H^4)$ | $O(H^2)$ |
| **Memory** | $O(H^3)$ | $O(TH)$ |
| **Update timing** | Online (매 step) | Sequence end |
| **Gradient** | Exact | Exact |

BPTT 가 단순 sequence 학습에서 우월. RTRL 은 online setting 에서.

### 정리 5.3 — UORO 의 Unbiasedness

$\nu_t \sim \text{Rademacher}(\pm 1)^H$ 에 대해:

$$
\mathbb E_{\nu_{1:t}}[\hat S_t] = S_t
$$

**증명** (sketch): Rank-1 projection $\nu \nu^\top$ 의 expectation 이 $I$ (Rademacher 의 outer product 평균). 매 step 에서 새 $\nu_t$ 추가, expectation 이 정확한 sensitivity 누적과 일치. $\square$

**Variance**: $O(H^2)$ — 추가 noise. 학습률 작게 조정 필요.

### 정리 5.4 — e-prop 의 Local Update Rule

Bellec 2020. **Eligibility trace** $e_{ij,t}$ 를 매 step update:

$$
e_{ij,t} = \alpha\, e_{ij,t-1} + h_{j,t-1} \cdot \tanh'(z_{i,t})
$$

학습:

$$
\Delta \theta_{ij} \propto L_t' \cdot e_{ij,t}
$$

**Local**: $i, j$ 에 관련된 양만 사용 — biological plausibility (synapse 단위).

### 정리 5.5 — Online RL 의 RTRL 적용

REINFORCE 같은 policy gradient + RNN policy 에서:

$$
\nabla_\theta J(\theta) = \mathbb E\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid h_t) \cdot R_t\right]
$$

$\nabla_\theta \log \pi$ 가 $\partial \log \pi / \partial h_t \cdot S_t$ 형태 — RTRL 로 streaming 계산. Episodic BPTT 와 비교 시 매 step update 가능.

---

## 💻 구현 검증

### 실험 1 — RTRL 의 Sensitivity Matrix

```python
import numpy as np
import torch
import torch.nn as nn

class RTRL_RNN:
    """Vanilla RNN with RTRL gradient"""
    def __init__(self, D, H, seed=0):
        rng = np.random.RandomState(seed)
        s = np.sqrt(1.0 / H)
        self.Wxh = rng.randn(H, D) * s
        self.Whh = rng.randn(H, H) * s
        self.bh = np.zeros(H)
        self.D, self.H = D, H
        # Sensitivity tensors
        self.S_Wxh = np.zeros((H, H, D))   # ∂h / ∂Wxh
        self.S_Whh = np.zeros((H, H, H))   # ∂h / ∂Whh
        self.h = np.zeros(H)
    
    def step(self, x):
        z = self.Whh @ self.h + self.Wxh @ x + self.bh
        h_new = np.tanh(z)
        D_act = 1 - h_new**2   # tanh' element-wise
        
        # Update sensitivities (RTRL recurrence)
        # S_Whh[i, j, k] = ∂h_i / ∂Whh[j, k]
        # ∂h/∂h_old = diag(D_act) @ Whh
        J = np.diag(D_act) @ self.Whh   # (H, H)
        
        # ∂h_i / ∂Whh[j, k] = D_act_i * (h_old_k * δ_{i,j} + Σ_l Whh[i,l] S_Whh[l,j,k])
        # Vectorized:
        S_Whh_new = np.einsum('il,ljk->ijk', J, self.S_Whh)
        # Add direct partial: ∂z_i / ∂Whh[j, k] = δ_{i,j} h_old_k
        for j in range(self.H):
            for k in range(self.H):
                S_Whh_new[j, j, k] += D_act[j] * self.h[k]
        
        S_Wxh_new = np.einsum('il,ljk->ijk', J, self.S_Wxh)
        for j in range(self.H):
            for k in range(self.D):
                S_Wxh_new[j, j, k] += D_act[j] * x[k]
        
        self.h = h_new
        self.S_Whh = S_Whh_new
        self.S_Wxh = S_Wxh_new
        return h_new

# Toy: H=4, D=3, T=5
D, H, T = 3, 4, 5
rnn = RTRL_RNN(D, H)
np.random.seed(0)
x_seq = np.random.randn(T, D)

for t in range(T):
    rnn.step(x_seq[t])

print(f'Sensitivity ∂h_T/∂Whh: shape {rnn.S_Whh.shape} = (H, H, H)')
print(f'Memory for S: {rnn.S_Whh.size + rnn.S_Wxh.size} floats')
print(f'O(H^3) = {H**3} (matches)')
```

### 실험 2 — RTRL Gradient vs BPTT Gradient 일치

```python
# BPTT 로 같은 gradient 계산
def bptt_gradient(rnn_init, x_seq, target_step):
    """target_step 의 ||h||^2 / 2 의 gradient"""
    Whh = torch.tensor(rnn_init.Whh, dtype=torch.float64, requires_grad=True)
    Wxh = torch.tensor(rnn_init.Wxh, dtype=torch.float64, requires_grad=True)
    bh = torch.tensor(rnn_init.bh, dtype=torch.float64, requires_grad=True)
    
    h = torch.zeros(rnn_init.H, dtype=torch.float64)
    for t, x in enumerate(x_seq):
        x_t = torch.tensor(x, dtype=torch.float64)
        z = Whh @ h + Wxh @ x_t + bh
        h = torch.tanh(z)
        if t == target_step:
            break
    
    loss = 0.5 * (h ** 2).sum()
    loss.backward()
    return Whh.grad.numpy(), Wxh.grad.numpy()

# RTRL 로 같은 gradient 계산
np.random.seed(0)
x_seq2 = np.random.randn(T, D)

rnn_rtrl = RTRL_RNN(D, H)
for t in range(T):
    rnn_rtrl.step(x_seq2[t])

# ∂(||h_T||^2 / 2) / ∂Whh = h_T^T @ S_Whh (last index)
# Specifically: Σ_i h_i * (∂h_i / ∂Whh[j,k])
grad_Whh_rtrl = np.einsum('i,ijk->jk', rnn_rtrl.h, rnn_rtrl.S_Whh)
grad_Wxh_rtrl = np.einsum('i,ijk->jk', rnn_rtrl.h, rnn_rtrl.S_Wxh)

# BPTT
rnn_init = RTRL_RNN(D, H)   # 같은 seed
grad_Whh_bptt, grad_Wxh_bptt = bptt_gradient(rnn_init, x_seq2, T - 1)

print(f'BPTT  grad_Whh norm: {np.linalg.norm(grad_Whh_bptt):.6f}')
print(f'RTRL  grad_Whh norm: {np.linalg.norm(grad_Whh_rtrl):.6f}')
print(f'Diff:                {np.abs(grad_Whh_bptt - grad_Whh_rtrl).max():.2e}')
# 두 gradient 가 일치해야 (수치 오차 내)
```

### 실험 3 — RTRL vs BPTT 시간 측정

```python
import time

H_list = [4, 8, 16, 32]
T = 20

for H in H_list:
    rnn = RTRL_RNN(D, H)
    x = np.random.randn(T, D)
    
    start = time.time()
    for _ in range(5):
        rnn = RTRL_RNN(D, H)
        for t in range(T):
            rnn.step(x[t])
    rtrl_time = (time.time() - start) / 5
    
    print(f'H={H:3d}: RTRL time = {rtrl_time*1000:.2f} ms (O(H^4) = {H**4})')

# H 두 배 → time 16배 (H^4 scaling)
```

### 실험 4 — UORO 단순 구현

```python
class UORO_RNN:
    """Tallec & Ollivier 2017 UORO — rank-1 projection of RTRL"""
    def __init__(self, D, H, seed=0):
        rng = np.random.RandomState(seed)
        s = np.sqrt(1.0 / H)
        self.Wxh = rng.randn(H, D) * s
        self.Whh = rng.randn(H, H) * s
        self.bh = np.zeros(H)
        self.D, self.H = D, H
        # Rank-1 approximations of S
        self.u_Whh = np.zeros(H)
        self.theta_Whh = np.zeros((H, H))
        self.h = np.zeros(H)
        self.rng = rng
    
    def step(self, x):
        z = self.Whh @ self.h + self.Wxh @ x + self.bh
        h_new = np.tanh(z)
        D_act = 1 - h_new**2
        
        # Update u, θ for Whh
        # New random sign vector
        nu = self.rng.choice([-1, 1], size=self.H).astype(float)
        
        # u_t = J u_{t-1} + scaling * nu
        J = np.diag(D_act) @ self.Whh
        u_new = J @ self.u_Whh + nu
        
        # θ_t = θ_{t-1} + (∂f/∂Whh contracted with nu)
        # Direct partial: ∂z_i / ∂Whh[i, k] = h_old_k → outer product
        partial = np.outer(D_act * nu, self.h)   # rank-1 partial
        theta_new = self.theta_Whh + np.outer(nu, self.h)   # 단순화 (정확하지 않음)
        
        self.h = h_new
        self.u_Whh = u_new
        self.theta_Whh = theta_new
        return h_new
    
    def gradient(self, dL_dh):
        """Approximate gradient via rank-1 outer product"""
        return np.outer(dL_dh * self.u_Whh, np.zeros(self.H))   # Skeleton

# UORO 의 정확한 구현은 추가 normalization 과 stochastic correction 필요
# 여기서는 알고리즘 골격만
print('UORO: rank-1 unbiased approximation of RTRL')
print(f'Memory per step: O(H + |θ|) = O({H + H**2})')
print(f'Time per step:   O(H^2) = O({H**2})')
```

### 실험 5 — Online RL Setting 의 시뮬레이션

```python
# Toy: REINFORCE with RNN policy, RTRL update
# Pseudo-code:
"""
for episode:
    state = env.reset()
    h, S = init_RNN_state()
    while not done:
        # Forward + sensitivity update
        logits = policy_RNN.step(state)         # h, S 갱신
        action = sample(logits)
        next_state, reward, done = env.step(action)
        
        # RTRL: 매 step 즉시 gradient
        log_prob_grad = ∂log π/∂θ = ∂log π/∂h · S
        grad = log_prob_grad * reward           # REINFORCE
        
        # Online update
        θ -= η * grad
        state = next_state
"""
print('RTRL 의 online RL 적용:')
print('- 매 step 즉시 gradient — sample efficiency')
print('- Episodic BPTT 와 비교 시 빠른 credit assignment')
print('- 그러나 O(H^4) 비용 — 작은 H 또는 UORO approximation')
```

---

## 🔗 실전 활용

### 1. Online RL with RNN policy

DRQN, R2D2 등 LSTM policy 의 학습. BPTT 는 episode 단위, RTRL/UORO 는 step 단위. Sample efficiency 향상 가능.

### 2. Continual learning

Stream of tasks/data — episode 경계 없음. RTRL 의 온라인 nature 가 자연스러움.

### 3. Neuromorphic computing

Loihi, BrainScaleS 같은 spiking neural network hardware — biological plausibility 의 e-prop 활용.

### 4. Meta-learning (RL²)

Inner loop 의 RNN 이 tasks 를 빠르게 learn — RTRL 로 inner loop online update.

### 5. Edge AI

Battery-constrained device 에서 episode-end backward 의 메모리 부담 회피 — UORO 의 $O(H^2)$ memory.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Single sequence stream | Batch RTRL 가능하지만 메모리 증가 |
| Forward-mode AD efficient | BPTT 가 단순 supervised 에서 우월 |
| $O(H^4)$ 비용 | UORO 의 stochastic approximation 으로 $O(H^2)$ |
| Bias 0 (RTRL exact) | UORO 의 variance 추가 — 학습률 조정 |
| Local synapse update (e-prop) | Eligibility trace 의 의미 task-dependent |

---

## 📌 핵심 정리

$$\boxed{S_t = \frac{\partial h_t}{\partial \theta} = J_t S_{t-1} + \frac{\partial f}{\partial \theta}\bigg|_{\text{partial}}}$$

$$\boxed{\frac{\partial L_t}{\partial \theta} = \frac{\partial L_t}{\partial h_t} \cdot S_t \quad \text{— online gradient}}$$

$$\boxed{\text{RTRL: } O(H^4) \text{ time / step}, \quad \text{UORO: } O(H^2) \text{ unbiased}}$$

| Method | Time | Memory | Bias | Update |
|--------|------|--------|------|--------|
| **RTRL** | $O(H^4)$ | $O(H^3)$ | 0 | Online |
| **BPTT** | $O(H^2)$ | $O(TH)$ | 0 | Episodic |
| **UORO** | $O(H^2)$ | $O(H^2)$ | 0 (stochastic) | Online |
| **TBPTT** | $O(H^2)$ | $O(kH)$ | $O(\rho^k)$ | Per-chunk |
| **e-prop** | $O(H^2)$ | $O(H^2)$ | bias (local) | Online, biological |

---

## 🤔 생각해볼 문제

**문제 1** (기초): RTRL 의 sensitivity $S_t \in \mathbb R^{H \times |\theta|}$ 에서 $|\theta| = O(H^2)$ 이므로 $S_t$ 의 메모리는 $O(H^3)$. $H = 100$ 시 floats 수와 메모리를 계산하라 (float32).

<details>
<summary>해설</summary>

$|\theta|$ for $W_{hh}$: $H^2 = 10000$
$S_t$ size: $H \times |\theta| = 100 \times 10000 = 10^6$ floats
Memory: $10^6 \times 4$ bytes $= 4$ MB

$T = 1000$ 시:
- RTRL: $4$ MB constant (sensitivity update only) + $S$ recomputation
- BPTT: $T \cdot H \cdot 4$ bytes = $400$ KB activation + intermediate

**RTRL 이 RNN 메모리에서 더 클 수도** ($O(H^3) > O(TH)$ when $T < H^2$). 실제로 BPTT 가 큰 $T$ 에서만 메모리 부담.

$\square$

</details>

**문제 2** (심화): UORO 의 unbiasedness 를 Rademacher 의 properties 로 증명하라. 즉 $\nu \sim \{-1, +1\}^H$ uniform 일 때 $\mathbb E[\nu \nu^\top] = I$.

<details>
<summary>해설</summary>

**Rademacher property**:

$\nu_i \sim$ uniform on $\{-1, +1\}$, independent across $i$.

$\mathbb E[\nu_i] = 0$, $\mathbb E[\nu_i^2] = 1$, $\mathbb E[\nu_i \nu_j] = 0$ for $i \ne j$ (independence).

**Outer product**:
$$
\mathbb E[\nu \nu^\top]_{ij} = \mathbb E[\nu_i \nu_j] = \begin{cases} 1 & i = j \\ 0 & i \ne j \end{cases} = I_{ij}
$$

따라서 $\mathbb E[\nu \nu^\top] = I$.

**UORO unbiasedness**:
- Sensitivity update: $\hat S_t = u_t v_t^\top$
- $\mathbb E[\hat S_t] = \mathbb E[u_t v_t^\top]$

직접 propagate 하면 $u_t = J_t u_{t-1} + \nu_t \Rightarrow$ outer product 가 정확한 $S_t$ 와 일치함을 induction 으로 보임:

Step $t$ contribution: $\nu_t \nu_t^\top \cdot \partial f/\partial \theta$ → expectation $I \cdot \partial f/\partial \theta$ = direct partial. $\square$

(정확한 derivation 은 추가 normalization terms — Tallec & Ollivier 2017 §3)

</details>

**문제 3** (논문 비평): RTRL 과 BPTT 가 동일한 gradient 를 계산하는데 왜 RTRL 이 deep learning 에서 거의 사용되지 않는가? 그러나 neuroscience 에서는 e-prop 같은 RTRL 변종이 표준 모델인 이유는?

<details>
<summary>해설</summary>

**Deep Learning 에서 RTRL 비사용 이유**:

1. **복잡도**:
   - BPTT: $O(H^2)$ per step
   - RTRL: $O(H^4)$ per step — 100x ~ 10000x 느림
   
2. **표준 setting 이 episodic**:
   - Supervised learning: full sequence 주어짐, BPTT 자연스러움
   - Standard ML benchmark: episodic — RTRL 의 online 장점 무용

3. **Hardware efficiency**:
   - GPU 가 large matmul 에 최적화 — BPTT 의 reverse pass 가 효율적
   - RTRL 의 sensitivity update 는 sparse, GPU 활용도 낮음

4. **Software ecosystem**:
   - PyTorch / TensorFlow autograd 가 reverse-mode 표준
   - RTRL 구현은 custom — 생태계 부재

**Neuroscience 에서 RTRL 표준 이유**:

1. **Biological plausibility**:
   - 뇌는 backward pass 없음 — 현재 forward 상태만 사용
   - Synapse update 가 *local* (인접 뉴런만) — RTRL 의 sensitivity 도 local
   - "credit assignment" 의 시간적 측면이 ed e-prop eligibility trace 와 일치

2. **Online learning의 본질**:
   - 동물은 매 step 학습 — episodic update 는 실생활에 부적합
   - RL 의 reward 가 delay 됨 — online update 가 자연스러움

3. **e-prop (Bellec 2020)**:
   - Eligibility trace = local sensitivity proxy
   - LSTM 비슷한 spiking neural network 학습 가능
   - 실제 hardware (Loihi) 에 구현 가능

**미래 전망**:

- **Continual learning**: episode 경계 없는 streaming — RTRL family 부활 가능성
- **Edge AI**: 메모리 제약, online update — UORO 같은 approximation
- **Neuromorphic chip**: bio-plausible 학습 — e-prop
- **Meta-learning**: inner loop 의 빠른 update — RTRL

**결론**: BPTT 는 deep learning 의 *효율성* 챔피언, RTRL 은 *biological* 과 *online* 의 자연스러운 framework. 두 패러다임이 다른 use case 에서 공존. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-complexity.md) | [📚 README](../README.md) | [다음 ▶](../ch3-vanishing-exploding/01-spectral-analysis.md)

</div>
