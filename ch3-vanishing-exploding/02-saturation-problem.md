# 02. 왜 $\rho = 1$ 유지가 어려운가 — Saturation 문제

## 🎯 핵심 질문

- $\rho(W_{hh}) = 1$ 으로 정확히 init 해도 왜 학습 중 effective ρ 가 1 이하로 감쇠하는가?
- $\tanh$ 의 saturation ($\sigma'(z) \to 0$ as $|z|$ 증가) 이 어떻게 vanishing 을 거의 필연적으로 만드는가?
- ReLU RNN 의 unbounded activation 이 왜 exploding 위험을 증가시키는가? Le 2015 의 IRNN 동기는?
- Spectral radius 와 activation derivative 의 product 가 어떻게 effective contraction 을 결정하는가?
- 학습 중 weight update 가 어떻게 spectral radius 를 변화시키며, 이것이 학습 dynamics 의 stability 에 어떤 영향을 주는가?

---

## 🔍 왜 ρ = 1 유지가 RNN 의 가장 어려운 문제인가

Pascanu 2013 (Ch3-01) 의 분석으로 **이상적인 RNN 은 $\rho_{\text{eff}} = 1$** 정확히 유지. 그러나 실전에서 이는 거의 불가능한 trade-off:

1. **Saturation의 압박** — $\tanh' \le 1$ 가 ρ 를 추가 감쇠, 평균 $\sigma' \approx 0.5$ 시 effective ρ = 0.5 ρ_W
2. **Activation 의 instability** — ReLU 는 unbounded → exploding
3. **학습 중 weight drift** — Gradient update 가 ρ 를 부지불식간 변화
4. **Edge of chaos** — ρ = 1 은 dynamical system 의 critical point, 작은 perturbation 도 instability

이 문제를 해결하려는 시도들:
- **Orthogonal init** (Ch3-04): ρ = 1 정확히 init
- **Identity init / IRNN** (Ch3-05): ReLU + identity init
- **Gating** (LSTM, Ch4): saturation 우회
- **Linear RNN** (SSM, Ch7-04): activation 제거

이 문서는 saturation 의 정확한 분석과 ReLU 와 tanh 의 trade-off 를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-spectral-analysis.md](./01-spectral-analysis.md) — Pascanu 정리, spectral radius
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): 활성화 함수, derivative bounds
- 미적분: $\tanh'(z) = 1 - \tanh^2(z)$, sigmoid $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- (선택) Dynamical systems: edge of chaos, Lyapunov exponent

---

## 📖 직관적 이해

### Saturation 시각화

```
tanh(z):
   1 ┤───────────────╱────────────
     │            ╱
     │         ╱
   0 ┤──────╱
     │   ╱
     │╱
  -1 ┤─────────────────────────────
     -3       0       3       z

tanh'(z):
   1 ┤        ╱╲
     │      ╱   ╲
     │    ╱       ╲
   0 ┤──────────────────
     -3   0   3       z
```

**Saturation 영역** ($|z| > 2$): $\tanh'(z) \approx 0$ — gradient 가 거의 사라짐.

### Pre-activation 분포의 진화

학습 초기 $z = W_{hh} h + W_{xh} x + b$ 의 분포가 $\mathcal N(0, \sigma^2)$ 정도면 대부분 $\tanh$ 의 linear region. 학습 진행 시 weight 가 커지면 $z$ 도 커지고 saturation 진입.

### ReLU vs Tanh 의 Trade-off

**Tanh**:
- $|h_t| < 1$ — bounded → exploding 약함
- 그러나 $\tanh' \le 1$ — vanishing 강함

**ReLU**:
- $h_t \ge 0$, unbounded → exploding 위험
- 그러나 $\sigma'(z) \in \{0, 1\}$, identity 영역에서 perfect propagation

**IRNN** (Le 2015, Ch3-05): ReLU 의 장점 + identity init 으로 stability.

### Edge of Chaos

Dynamical system 에서 $\rho = 1$ 은 critical point:
- $\rho < 1$: ordered phase (모든 trajectory 가 하나의 attractor 로 수렴)
- $\rho > 1$: chaotic phase (작은 perturbation 이 exponential 증폭)
- $\rho = 1$: edge of chaos — 정보 보존, 그러나 unstable

학습이 이 edge 를 walking — 작은 weight update 도 ordered/chaotic 으로 push.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Activation Derivative Bounds

$$
\sigma' \in [0, \sigma'_{\max}], \quad \sigma'_{\max} = \begin{cases} 1 & \tanh \\ 1/4 & \text{sigmoid} \\ 1 & \text{ReLU (positive region)} \\ 1 & \text{linear (no activation)} \end{cases}
$$

### 정의 2.2 — Saturation Region

$$
\mathcal S_{\tanh} := \{z : \tanh'(z) < \epsilon\}
$$

$\epsilon = 0.01$ 시 $\mathcal S_{\tanh} = \{z : |z| > \mathrm{arctanh}(\sqrt{0.99}) \approx 2.6\}$.

### 정의 2.3 — Effective Spectral Radius

학습 중 instantaneous effective rate:

$$
\rho_{\text{eff}}(t) = \rho(W_{hh}) \cdot \mathbb E[\sigma'(z_t)]
$$

이는 forward dynamics 와 weight 의 결합.

### 정의 2.4 — Lyapunov Exponent

RNN 의 long-term gradient growth rate:

$$
\Lambda = \lim_{T \to \infty} \frac{1}{T} \log \|\partial h_T / \partial h_0\|
$$

- $\Lambda < 0$: vanishing
- $\Lambda = 0$: edge of chaos
- $\Lambda > 0$: exploding (chaotic)

### 정의 2.5 — Edge of Chaos

Dynamical system 의 critical phase transition: $\Lambda = 0$ 의 manifold.

---

## 🔬 정리와 결과

### 정리 2.1 — Tanh RNN 의 Saturation Inevitability

Random init $W_{hh} \sim \mathcal N(0, \sigma^2/H)$ 에서 학습 진행 시 $\|z_t\|$ 가 증가하는 경향:

$$
\mathbb E[\|z_t\|^2] \approx \sigma^2 \mathbb E[\|h_{t-1}\|^2] + \mathbb E[\|x_t\|^2] \cdot d
$$

학습으로 weight magnitude 증가 → $\|z\|$ 증가 → saturation → effective $\sigma'$ 감소.

**증명** (sketch): Initialization variance 가 $\sigma^2$ 일 때 $\mathbb E[\|z\|^2] = \sigma^2 H$ for $\|h\|^2 = O(1)$ (tanh bound). $\sigma$ 가 충분히 크면 $\|z\|$ 가 saturation 영역. Learning 이 weight magnitude 증가시키므로 $\sigma_{\text{eff}}$ 가 시간에 따라 증가. $\square$

### 정리 2.2 — ReLU RNN 의 Unboundedness

ReLU 활성화 시:

$$
h_t = \max(0, W_{hh} h_{t-1} + W_{xh} x_t + b)
$$

만약 $\rho(W_{hh}) > 1$ 이고 $h_t$ 가 positive orthant 에 있으면:

$$
h_{t+1} \ge W_{hh} h_t \implies \|h_{t+1}\| \ge \rho \|h_t\|
$$

$\|h\|$ 가 unbounded 증가 → numerical overflow.

**증명**: ReLU 가 positive orthant 에서 identity, $h_{t+1} \ge W_{hh} h_t$ in element-wise sense (positive entries 가정). Spectral radius 가 multiplicative 증가. $\square$

### 정리 2.3 — Lyapunov Exponent 의 Forward Estimate

Long-term gradient norm:

$$
\Lambda = \lim_T \frac{1}{T} \sum_{t=1}^T \log \|J_t\|
$$

Stationary distribution 에서:

$$
\Lambda = \log \rho(W_{hh}) + \mathbb E_{\text{stationary}}[\log \sigma'(z)]
$$

**$\Lambda = 0$** 의 조건이 edge of chaos.

### 정리 2.4 — Glorot / Xavier Init 의 ρ 추정

$W_{hh} \sim \mathcal N(0, 1/H)$ 시:
- $\sigma_{\max}(W_{hh})$ 의 expectation $\approx 2$ (Marchenko-Pastur)
- $\rho(W_{hh}) \approx 1$ (asymptotic)
- 그러나 $\sigma_{\max} \ne \rho$ — finite-time growth 가능

따라서 random init 가 spectral radius 측면에서 *대략* edge of chaos 에 있음.

### 정리 2.5 — Spectral Drift During Training

Gradient update $\Delta W_{hh} = -\eta \partial L / \partial W_{hh}$ 가 $\rho(W_{hh})$ 를 변화시킴. Perturbation theory:

$$
\delta \rho \approx u^\top (\Delta W_{hh}) v
$$

$u, v$ 는 dominant eigenvector. Random gradient 시 ρ 의 random walk → 학습이 길어질수록 ρ 가 1 에서 멀어짐.

---

## 💻 NumPy 실험

### 실험 1 — Saturation 진입의 시간축 추적

```python
import numpy as np
import matplotlib.pyplot as plt

H, D = 50, 10
T = 200
W_hh = np.random.randn(H, H) * (1 / np.sqrt(H))   # Glorot
# Spectral radius 정규화
W_hh /= max(abs(np.linalg.eigvals(W_hh)))   # ρ = 1

W_xh = np.random.randn(H, D) * (1 / np.sqrt(H))
b_h = np.zeros(H)

h = np.zeros(H)
z_norms = []
sigma_prime_avgs = []

for t in range(T):
    x_t = np.random.randn(D) * 0.5
    z = W_hh @ h + W_xh @ x_t + b_h
    h = np.tanh(z)
    
    z_norms.append(np.linalg.norm(z))
    sigma_prime_avgs.append((1 - h**2).mean())

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(z_norms); ax[0].set_title('||z_t|| over time')
ax[0].set_xlabel('t'); ax[0].set_ylabel('||z||')
ax[1].plot(sigma_prime_avgs); ax[1].set_title("avg σ'(z_t) over time")
ax[1].set_xlabel('t'); ax[1].set_ylabel("avg σ'")
plt.tight_layout()
plt.savefig('saturation_dynamics.png', dpi=120)
print(f'Initial avg σ\' = {sigma_prime_avgs[0]:.3f}')
print(f'Final avg σ\' = {sigma_prime_avgs[-1]:.3f}')
print(f'Effective ρ change: 1.0 → {1.0 * sigma_prime_avgs[-1]:.3f}')
```

### 실험 2 — Random Init 의 Spectral Radius 분포

```python
H_list = [10, 50, 100, 500]
n_samples = 50

for H in H_list:
    rhos = []
    for _ in range(n_samples):
        W = np.random.randn(H, H) * (1 / np.sqrt(H))
        rhos.append(max(abs(np.linalg.eigvals(W))))
    print(f'H={H:4d}: ρ ~ {np.mean(rhos):.3f} ± {np.std(rhos):.3f} (Glorot init)')

# Glorot init 의 expected ρ ≈ 1 (asymptotic Marchenko-Pastur for symmetric, 
# circular law for general), 그러나 large H 에서 안정
```

### 실험 3 — ReLU vs Tanh Bounded-ness

```python
def simulate_rnn(activation, T, H=50, init_scale=1.0):
    """다른 activation 의 ||h_t|| 시간축"""
    W = np.random.randn(H, H) * (init_scale / np.sqrt(H))
    W /= max(abs(np.linalg.eigvals(W)))   # ρ = 1
    W *= 1.05   # 약간 above 1
    
    h = np.random.randn(H) * 0.1
    norms = [np.linalg.norm(h)]
    for t in range(T):
        z = W @ h
        if activation == 'tanh':
            h = np.tanh(z)
        elif activation == 'relu':
            h = np.maximum(0, z)
        norms.append(np.linalg.norm(h))
    return norms

np.random.seed(0)
n_tanh = simulate_rnn('tanh', T=100)
np.random.seed(0)
n_relu = simulate_rnn('relu', T=100)

print(f'After T=100, ρ = 1.05:')
print(f'  Tanh ||h_T||: {n_tanh[-1]:.4f}  (bounded by 1·√H ≈ 7)')
print(f'  ReLU ||h_T||: {n_relu[-1]:.4e}  (unbounded → exploding)')
```

### 실험 4 — Spectral Radius Drift During Training

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.cell = nn.RNNCell(D, H, nonlinearity='tanh')
    def forward(self, x_seq):
        h = torch.zeros(x_seq.size(1), self.cell.hidden_size)
        for t in range(x_seq.size(0)):
            h = self.cell(x_seq[t], h)
        return h

D, H = 5, 20
torch.manual_seed(0)
model = SimpleRNN(D, H)

# Spectral radius before training
W_init = model.cell.weight_hh.detach().numpy()
rho_init = max(abs(np.linalg.eigvals(W_init)))

# Train on random task
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for step in range(500):
    x = torch.randn(20, 8, D)
    target = torch.randn(8, H)
    h = model(x)
    loss = ((h - target)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

W_final = model.cell.weight_hh.detach().numpy()
rho_final = max(abs(np.linalg.eigvals(W_final)))

print(f'ρ(W_hh) before training: {rho_init:.4f}')
print(f'ρ(W_hh) after  training: {rho_final:.4f}')
print(f'Drift: {abs(rho_final - rho_init):.4f}')
# 학습이 ρ 를 random walk → drift
```

### 실험 5 — Lyapunov Exponent 추정

```python
def estimate_lyapunov(W, activation, T=1000):
    """Forward dynamics 에서 ||δh|| 의 average log-growth"""
    H = W.shape[0]
    h = np.random.randn(H) * 0.1
    delta = np.random.randn(H); delta /= np.linalg.norm(delta)
    
    log_norms = []
    for t in range(T):
        z = W @ h
        if activation == 'tanh':
            h = np.tanh(z)
            sigma_prime = 1 - h**2
        elif activation == 'relu':
            h = np.maximum(0, z)
            sigma_prime = (z > 0).astype(float)
        
        # Tangent dynamics
        delta = (sigma_prime[:, None] * W) @ delta
        norm = np.linalg.norm(delta)
        log_norms.append(np.log(norm))
        delta /= norm   # Renormalize
    
    return np.mean(log_norms)

W = np.random.randn(50, 50) * (1 / np.sqrt(50))
W /= max(abs(np.linalg.eigvals(W)))   # ρ = 1

for rho_target in [0.8, 1.0, 1.2]:
    W_scaled = W * rho_target
    Lambda_tanh = estimate_lyapunov(W_scaled, 'tanh', T=2000)
    print(f'ρ={rho_target}, tanh: Λ = {Lambda_tanh:.4f}')
# Λ < 0: vanishing, Λ ≈ 0: edge of chaos, Λ > 0: chaotic
```

---

## 🔗 실전 활용

### 1. Initialization recipes

- Tanh RNN: Glorot/Xavier $\sigma^2 = 1/H$, 그러나 ρ drift 위험
- ReLU RNN: He init $\sigma^2 = 2/H$, 그러나 exploding 위험
- Best: **Orthogonal init** (Ch3-04) — ρ = 1 정확히

### 2. Activation 선택

- Tanh: 표준, bounded, 그러나 vanishing prone
- ReLU: 빠른 학습, exploding 주의
- LeakyReLU / GELU: ReLU 의 dead unit 회피
- LSTM/GRU: gating 으로 saturation 우회

### 3. Spectral monitoring

학습 중 매 epoch 후 $\rho(W_{hh})$ 측정 — drift 가 클 시 weight clipping 또는 spectral normalization (Miyato 2018) 적용.

### 4. Layer normalization / Batch normalization

Pre-activation $z$ 의 분포를 normalize → saturation 회피. LSTM 에서도 효과적 (Ba 2016).

### 5. Dropout

활성화 분포를 더 sparse 하게 → effective dynamics 변화. RNN 에서는 variational dropout (Gal 2016) 표준.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Random gradient direction | 학습된 RNN 은 specific direction에 align |
| Stationary input distribution | Real data 는 non-stationary |
| Symmetric $W_{hh}$ | 일반 RNN 은 non-symmetric → spectral radius 와 operator norm 차이 |
| Single hidden layer | Stacked RNN 은 layer 별 dynamics |
| Constant activation | Activation 변경 시 (다른 RNN variants) bound 다름 |

---

## 📌 핵심 정리

$$\boxed{\rho_{\text{eff}}(t) = \rho(W_{hh}) \cdot \mathbb E[\sigma'(z_t)] \le \rho(W_{hh})}$$

$$\boxed{\Lambda = \log \rho(W_{hh}) + \mathbb E[\log \sigma'(z)] \quad \text{Lyapunov exponent}}$$

$$\boxed{\rho = 1 \text{ 정확히 유지: 거의 불가능 (drift, saturation)} \implies \text{LSTM, gating 필요}}$$

| Activation | $\sigma'_{\max}$ | $h$ bound | Vanishing | Exploding |
|-----------|-----------------|-----------|-----------|-----------|
| **Tanh** | 1 | $|h| < 1$ | 강함 (saturation) | 약함 |
| **Sigmoid** | 1/4 | $h \in [0,1]$ | 매우 강함 | 거의 없음 |
| **ReLU** | 1 | unbounded | 약함 (positive region) | 강함 |
| **Linear** | 1 | unbounded | $\rho < 1$ 시 | $\rho > 1$ 시 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\tanh$ activation 에서 $z = 2.5$ 시 $\sigma'(z)$ 값을 계산하라. 이 값을 가진 RNN 의 effective $\rho$ ($\rho_W = 1$ 가정) 는?

<details>
<summary>해설</summary>

$\tanh(2.5) \approx 0.987$
$\tanh'(2.5) = 1 - 0.987^2 \approx 0.026$

**Effective $\rho$**: $1.0 \times 0.026 = 0.026$ — 매우 작음.

**의미**: 한 번 saturation 진입하면 effective $\rho \to 0$, gradient 가 거의 흐르지 않음. **One bad step → vanishing**.

**LSTM 의 우회**:
- Cell state 의 forget gate $f_t$ 는 sigmoid ($\sigma'(z) \le 1/4$ 이지만 $f_t \in [0,1]$ direct)
- 더 중요: $\partial c_t / \partial c_{t-1} = f_t$ — saturation 무관
- $f_t \approx 1$ 유지하면 vanishing 회피 (Ch4-03 CEC)

$\square$

</details>

**문제 2** (심화): Glorot init $W \sim \mathcal N(0, 1/H)$ 의 spectral radius 가 $H \to \infty$ 시 1 로 수렴하는 이유 (circular law) 를 설명하고, $H = 100$ 에서 finite-size 보정의 크기를 추정하라.

<details>
<summary>해설</summary>

**Circular Law** (Tao-Vu 2010): $W \sim \mathcal N(0, 1/H)^{H \times H}$ 의 고유값 분포가 $H \to \infty$ 시 unit disk 의 uniform.

따라서:
- $\rho(W) \to 1$ asymptotically
- $\rho(W)$ 의 distribution: **Tracy-Widom** for largest eigenvalue

**Finite-size correction**:

$\mathbb E[\rho(W)] = 1 + O(H^{-1/2})$

$H = 100$ 시 보정 $\approx 1 / \sqrt{100} = 0.1$ — $\rho$ 가 $1.0 \pm 0.1$ 정도 fluctuation.

**Variance of $\rho$**:
$\mathrm{Var}(\rho) \sim H^{-2/3}$ (Tracy-Widom scaling)

$H = 100$: std $\approx 100^{-1/3} \approx 0.21$

**Practical**: random init 의 $\rho$ 가 정확히 1 이 아님, $1 \pm 0.2$ 범위. 이 randomness 가 학습 dynamics 의 첫 단계에 영향.

**Spectral normalization** (Miyato 2018): 매 step 후 $W \leftarrow W / \sigma_{\max}(W)$ 로 강제 — GAN discriminator 표준. RNN 에서도 적용 가능. $\square$

</details>

**문제 3** (논문 비평): "$\rho = 1$ 정확히 유지" 가 이론적으로 이상적이지만, 실전에서 LSTM 이 $\rho$ 무관하게 작동하는 이유는? Gating 이 어떻게 spectral 분석을 *우회* 하는가?

<details>
<summary>해설</summary>

**$\rho = 1$ 의 이상화**:
- Pascanu 정리: gradient 의 무한한 보존 위해 $\rho = 1$
- 그러나:
  1. ρ drift (학습 중 변화)
  2. Saturation (effective ρ ≪ ρ)
  3. Non-normal matrix 의 transient instability

**LSTM 의 우회 메커니즘**:

1. **Cell state separation**:
   - Hidden $h_t = o_t \odot \tanh(c_t)$ — saturation 문제 동일
   - Cell $c_t = f_t c_{t-1} + i_t \tilde c_t$ — **linear in $c$**
   - Gradient flow 가 cell path 위주

2. **Linear cell update**:
   $$
   \frac{\partial c_t}{\partial c_{t-1}} = f_t \quad (\text{element-wise scalar})
   $$
   - Matrix multiplication 이 element-wise multiplication 으로 단순화
   - Spectral radius 무관, **per-element decay rate**

3. **Adaptive forget**:
   - $f_t = \sigma(W_f [h, x] + b_f)$ — input-dependent
   - 학습이 $f_t$ 를 task 에 맞춰 조정
   - "이 정보는 보존, 저 정보는 forget" 의 selective memory

4. **Gradient bound**:
   $$
   \frac{\partial c_T}{\partial c_0} = \prod_t f_t
   $$
   $f_t \approx 1$ 시 보존 (Constant Error Carousel)

**Spectral 분석의 한계**:
- Plain RNN 의 Jacobian 은 *matrix* product → spectral radius
- LSTM 의 cell Jacobian 은 *element-wise* product → 각 dim 독립
- 후자는 더 robust: 각 dim 이 독자적으로 forget rate 결정

**그러나**: LSTM 의 hidden state $h$ 는 여전히 saturation 영향. **부분적 해결**:
- Cell 은 long-range
- Hidden 은 short-range
- 두 경로의 결합

**결론**: $\rho = 1$ 의 이론적 이상은 *plain RNN* 의 한계. LSTM 은 spectral 분석을 우회하는 architectural innovation — vanishing 의 *root cause* (matrix product) 를 제거. Transformer 는 더 극단적 — product 자체를 attention 으로 대체. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-spectral-analysis.md) | [📚 README](../README.md) | [다음 ▶](./03-gradient-clipping.md)

</div>
