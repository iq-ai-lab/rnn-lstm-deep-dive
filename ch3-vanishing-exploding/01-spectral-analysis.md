# 01. Gradient 의 Spectral 분석 (Pascanu 2013)

## 🎯 핵심 질문

- $\partial h_t / \partial h_k = \prod_{j=k+1}^t W_{hh}^\top \mathrm{diag}(\sigma'(z_j))$ 의 곱의 spectral radius 가 어떻게 vanishing/exploding 의 결정자가 되는가?
- **정리 (Pascanu 2013)**: $\rho(W_{hh}) \cdot \max \sigma' < 1 \Rightarrow$ vanishing exponential, $> 1 \Rightarrow$ exploding 의 정확한 증명 단계는?
- SVD 분해 $W_{hh} = U \Sigma V^\top$ 의 singular value 분포가 학습 dynamics 를 어떻게 결정하는가?
- $\tanh' \le 1$ 의 saturation 효과로 effective spectral radius 가 어떻게 추가로 감쇠되는가?
- $T = 100$ step 후 gradient norm 의 exponential 감쇠를 NumPy 로 측정하고 spectral radius 와 일치 검증

---

## 🔍 왜 이 분석이 RNN 의 근본 한계를 정의하는가

Pascanu 2013 의 *On the Difficulty of Training Recurrent Neural Networks* 는 RNN 의 학습 어려움을 처음으로 엄밀히 정량화한 논문입니다. 이 분석은 다음 모든 결과의 기반:

1. **Vanishing/exploding gradient 의 진단** — 단순한 "gradient 가 작아진다" 가 아니라 spectral radius 의 정확한 조건
2. **LSTM 의 동기** (Ch4-01) — Cell state 의 additive update 가 곱셈적 누적을 우회하는 정확한 메커니즘
3. **Orthogonal initialization** (Ch3-04) — $\rho = 1$ 정확히 유지하는 spectral 조건의 만족
4. **Gradient clipping 의 정당화** (Ch3-03) — Exploding 의 norm-based 대응
5. **Identity init / IRNN** (Ch3-05) — $W_{hh} = I$ 의 spectral 의미

이 문서에서는 Pascanu 의 정리를 한 단계씩 증명하고 NumPy 로 spectral radius 와 gradient 감쇠율의 정확한 관계를 측정합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch2-02 BPTT 의 완전 유도](../ch2-bptt/02-bptt-derivation.md) — Jacobian 곱 형태
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): **Spectral radius**, eigendecomposition, SVD, **submultiplicative norm**
- 함수 해석학: Operator norm $\|A\|_2 = \sigma_{\max}(A)$, Gelfand 의 spectral radius formula
- 미적분: Activation derivative bounds ($\sigma' \le 1$ for $\tanh$, $\sigma'(0) \le 1/4$ for sigmoid)

---

## 📖 직관적 이해

### Jacobian 곱의 누적

BPTT 의 핵심 항 (Ch2-02 정리 2.3):

$$
\prod_{j=k+1}^{t} J_j^\top, \quad J_j = \mathrm{diag}(\sigma'(z_j))\, W_{hh}
$$

이 곱이 vanishing/exploding 의 모든 정보를 담고 있습니다.

### Spectral Radius 의 의미

$W$ 의 **spectral radius** $\rho(W) = \max_i |\lambda_i(W)|$ — 가장 큰 고유값의 magnitude.

- $\rho < 1$: $W^k \to 0$ as $k \to \infty$ (contraction)
- $\rho = 1$: $W^k$ bounded (neutral)
- $\rho > 1$: $W^k \to \infty$ (expansion)

RNN 에서 $\prod J_j$ 가 비슷한 행동.

### Saturation 효과

$\tanh'(z) = 1 - \tanh^2(z) \in [0, 1]$. 특히:
- $z \approx 0$: $\tanh'(0) = 1$ — 최대
- $|z|$ 큼: $\tanh'(z) \to 0$ — saturation

따라서 effective spectral radius $\rho_{\text{eff}} = \rho(W_{hh}) \cdot \max \sigma' \le \rho(W_{hh})$ — saturation 이 추가 감쇠.

### Vanishing 의 시각적 직관

```
||δ_T||  →  step T-1  ←  step T-2  ←  ...  ←  step 0
   1.0       0.7          0.49           ...     0.7^T

T = 100, ρ = 0.9: 0.9^100 ≈ 2.7 × 10^-5  ← vanishing
T = 100, ρ = 1.1: 1.1^100 ≈ 1.4 × 10^4   ← exploding
```

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Spectral Radius

$$
\rho(A) := \max_i |\lambda_i(A)|
$$

여기서 $\lambda_i$ 는 $A$ 의 모든 고유값.

### 정의 1.2 — Operator Norm (2-norm)

$$
\|A\|_2 := \sup_{\|x\|=1} \|Ax\| = \sigma_{\max}(A)
$$

(largest singular value)

### 정의 1.3 — Spectral Radius Formula (Gelfand)

$$
\rho(A) = \lim_{k \to \infty} \|A^k\|^{1/k}
$$

특히, normal matrix ($AA^* = A^*A$) 시 $\|A\| = \rho(A)$. 일반 행렬은 $\rho(A) \le \|A\|$.

### 정의 1.4 — Pascanu Spectral Condition

RNN 의 Jacobian $J_j = \mathrm{diag}(\sigma'(z_j))\, W_{hh}$ 에 대해:

$$
\rho(W_{hh}) \cdot \max_j \|\sigma'(z_j)\|_\infty
$$

이 양이 $1$ 보다 작으면 **vanishing**, 크면 **exploding** 의 sufficient condition.

### 정의 1.5 — Gradient Norm Ratio

$$
r(t, k) := \frac{\|\partial L_t / \partial W_{hh}^{(k)}\|}{\|\partial L_t / \partial W_{hh}^{(t)}\|}
$$

$r$ 의 $t - k$ 에 대한 행동이 vanishing/exploding 진단.

---

## 🔬 정리와 증명

### 정리 1.1 — Pascanu 의 Vanishing Theorem

**Theorem** (Pascanu, Mikolov, Bengio 2013): RNN 의 hidden Jacobian $J_j = W_{hh}^\top \mathrm{diag}(\sigma'(z_j))$ 에 대해, **충분조건**:

$$
\rho(W_{hh}) < \frac{1}{\max_z \sigma'(z)} \implies \left\|\prod_{j=k+1}^{t} J_j\right\| \to 0 \;\; \text{as} \;\; t - k \to \infty
$$

**증명**:

**Step 1**: Submultiplicative property:

$$
\left\|\prod_{j=k+1}^{t} J_j\right\| \le \prod_{j=k+1}^{t} \|J_j\|
$$

**Step 2**: 각 $J_j$ 의 norm:

$$
\|J_j\| = \|W_{hh}^\top \mathrm{diag}(\sigma'(z_j))\| \le \|W_{hh}^\top\| \cdot \|\mathrm{diag}(\sigma'(z_j))\| = \|W_{hh}\| \cdot \max_i |\sigma'(z_{j,i})|
$$

**Step 3**: $\sigma' \le \sigma'_{\max}$ uniformly:

$$
\|J_j\| \le \|W_{hh}\| \cdot \sigma'_{\max}
$$

**Step 4**: Product:

$$
\left\|\prod_{j=k+1}^{t} J_j\right\| \le (\|W_{hh}\| \cdot \sigma'_{\max})^{t-k}
$$

**Step 5**: For symmetric $W_{hh}$, $\|W_{hh}\| = \rho(W_{hh})$. General case 에서는 $\|W_{hh}\| \ge \rho(W_{hh})$ (정의 1.3 의 Gelfand 부등식). 충분조건:

$$
\|W_{hh}\| \cdot \sigma'_{\max} < 1 \implies \text{exponential decay}
$$

비대칭 행렬에서는 spectral radius 만으로 부족하지만, **Gelfand limit** 을 이용한 long-term behavior 는:

$$
\lim_{n} \|W_{hh}^n\|^{1/n} = \rho(W_{hh})
$$

따라서 충분히 큰 $t - k$ 에서 effective rate 가 $\rho(W_{hh}) \sigma'_{\max}$. $\square$

### 정리 1.2 — Exploding Condition

$$
\rho(W_{hh}) > \frac{1}{\max_z \sigma'(z)} \;\; \text{and} \;\; \sigma'(z_j) \approx \sigma'_{\max} \;\;\forall j \implies \left\|\prod J_j\right\| \to \infty
$$

(Saturation 안 된 영역에서 학습되는 경우)

**증명** (sketch): $\sigma'(z_j) \approx \sigma'_{\max} = 1$ for $\tanh$ near origin. Product 의 dominant eigenvalue 가 $\rho(W_{hh})^{t-k}$, 발산. $\square$

### 정리 1.3 — Tanh 의 Saturation 효과

$\tanh' = 1 - \tanh^2 \in [0, 1]$. Average activation 이 saturated region 에 있으면:

$$
\mathbb E[\sigma'(z_j)] \ll 1 \implies \rho_{\text{eff}} \ll \rho(W_{hh})
$$

따라서 $\rho(W_{hh}) = 1$ 이라도 saturation 이 vanishing 을 야기.

### 정리 1.4 — SVD Decomposition 과 Vanishing Direction

$W_{hh} = U \Sigma V^\top$ 일 때, gradient $\delta$ 의 propagation:

$$
\prod J_j \cdot \delta = U_\Sigma \cdot (\text{singular value combinations}) \cdot V^\top \delta
$$

작은 singular value 방향의 $\delta$ component 가 vanishing, 큰 방향이 exploding. **Mode 별로 다른 dynamics**.

### 정리 1.5 — Pascanu's Lower Bound on Long-term Information

가능한 long-term task (의존 길이 $\tau$) 의 gradient signal:

$$
\left\|\frac{\partial L_T}{\partial h_0}\right\| \ge c \cdot \rho_{\text{eff}}^T
$$

$\rho_{\text{eff}} < 1$ 이면 정보가 exponentially 손실 — 학습 불가. $\rho_{\text{eff}} = 1$ 정확히 유지가 필수.

---

## 💻 NumPy 구현 검증

### 실험 1 — Spectral Radius 의 Gradient 감쇠율 직접 측정

```python
import numpy as np
import matplotlib.pyplot as plt

def measure_gradient_decay(W_hh, T, n_samples=20):
    """∏ W_hh^T diag(σ') 의 norm 시간축"""
    H = W_hh.shape[0]
    norms_avg = np.zeros(T+1)
    
    for sample in range(n_samples):
        # Random initial gradient
        delta = np.random.randn(H)
        delta /= np.linalg.norm(delta)
        
        norms = [1.0]
        for t in range(T):
            # Random pre-activation z (simulate through forward dynamics)
            z = np.random.randn(H) * 0.5   # Near 0 → σ'(z) ≈ 1
            sigma_prime = 1 - np.tanh(z)**2
            
            J = np.diag(sigma_prime) @ W_hh.T   # J^T for backward
            delta = J @ delta
            norms.append(np.linalg.norm(delta))
        norms_avg += np.array(norms)
    return norms_avg / n_samples

# 다양한 spectral radius 의 W
H = 50
np.random.seed(0)

results = {}
for rho_target in [0.5, 0.9, 1.0, 1.1, 1.5]:
    W = np.random.randn(H, H)
    u, s, vt = np.linalg.svd(W)
    W_normalized = u @ np.diag(np.ones_like(s) * rho_target) @ vt
    norms = measure_gradient_decay(W_normalized, T=100)
    results[rho_target] = norms
    print(f'ρ={rho_target}: ||δ_100|| / ||δ_0|| = {norms[-1]/norms[0]:.6e}')

# Theoretical: (ρ * σ'_max)^T
# σ'_max ≈ 1 (z ~ 0 가정)
# 0.5^100 ≈ 8e-31, 0.9^100 ≈ 2.7e-5, 1.1^100 ≈ 1.4e4, 1.5^100 ≈ 4e17
```

### 실험 2 — Plot Vanishing/Exploding

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for rho_target, norms in results.items():
    plt.semilogy(norms, label=f'ρ={rho_target}')
plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Time step t')
plt.ylabel('||δ_t|| (log scale)')
plt.title('Gradient norm decay/growth — Pascanu 2013')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gradient_decay.png', dpi=120, bbox_inches='tight')
print('Saved: gradient_decay.png')
```

### 실험 3 — SVD 분포의 영향

```python
# 같은 spectral radius 라도 singular value 분포가 다르면 dynamics 다름
H = 50
np.random.seed(0)

# Case A: Uniform singular values (모두 1)
W_uniform = np.random.randn(H, H)
u, _, vt = np.linalg.svd(W_uniform)
W_uniform = u @ np.eye(H) @ vt   # singular = 1

# Case B: Heavy-tailed (1 큰, 나머지 작은)
sigma_heavy = np.zeros(H)
sigma_heavy[0] = 1.0
sigma_heavy[1:] = 0.1
W_heavy = u @ np.diag(sigma_heavy) @ vt

# Spectral radius (largest eig magnitude)
print(f'Uniform: ρ = {max(abs(np.linalg.eigvals(W_uniform))):.4f}')
print(f'Heavy : ρ = {max(abs(np.linalg.eigvals(W_heavy))):.4f}')

# Gradient decay
n_uniform = measure_gradient_decay(W_uniform, T=50)
n_heavy = measure_gradient_decay(W_heavy, T=50)

print(f'\\nT=50 후 ||δ||:')
print(f'  Uniform (σ=1):       {n_uniform[-1]:.4e}')
print(f'  Heavy (σ_1=1, rest=0.1): {n_heavy[-1]:.4e}')
# Heavy 의 random direction 은 작은 σ 방향이 dominant → 더 빠른 vanishing
```

### 실험 4 — Saturation 시 effective spectral radius 측정

```python
def measure_with_saturation(W, T, z_scale):
    """z 의 scale 에 따른 vanishing rate"""
    H = W.shape[0]
    delta = np.random.randn(H); delta /= np.linalg.norm(delta)
    
    rates = []
    for t in range(T):
        z = np.random.randn(H) * z_scale   # 큰 z → saturated
        sigma_prime = 1 - np.tanh(z)**2
        avg_sigma_prime = sigma_prime.mean()
        rates.append(avg_sigma_prime)
        
        J = np.diag(sigma_prime) @ W.T
        delta = J @ delta
    
    return delta, rates

W_test = np.eye(H) + 0.1 * np.random.randn(H, H)   # ρ ≈ 1
W_test = W_test / max(abs(np.linalg.eigvals(W_test)))   # ρ = 1 정확히

for z_scale in [0.1, 1.0, 3.0]:
    np.random.seed(0)
    delta_final, rates = measure_with_saturation(W_test, T=50, z_scale=z_scale)
    avg_sigma_prime = np.mean(rates)
    final_norm = np.linalg.norm(delta_final)
    print(f'z_scale={z_scale}: avg σ\' = {avg_sigma_prime:.3f}, ||δ_50|| = {final_norm:.4e}')
# 큰 z_scale → saturation → effective spectrum 감쇠 → vanishing
```

### 실험 5 — Spectral Radius 추정과 실험 일치

```python
def gelfand_limit_estimate(W, k_max=100):
    """Gelfand: ρ(W) = lim ||W^k||^{1/k}"""
    Wk = np.eye(W.shape[0])
    norms = []
    for k in range(1, k_max+1):
        Wk = Wk @ W
        norms.append(np.linalg.norm(Wk) ** (1/k))
    return norms

W = np.random.randn(H, H)
W /= max(abs(np.linalg.eigvals(W)))   # ρ = 1
W *= 0.9                                 # ρ = 0.9

rates = gelfand_limit_estimate(W, k_max=50)
true_rho = max(abs(np.linalg.eigvals(W)))
print(f'True ρ = {true_rho:.4f}')
print(f'Gelfand estimates (last 10): {rates[-10:]}')
# Estimates → true ρ as k 증가 (Gelfand 정리)
```

---

## 🔗 실전 활용

### 1. Pre-training initialization 분석

새로운 RNN 변종 (custom gating) 의 학습 dynamics 를 spectral radius 로 사전 예측. Random init 의 ρ 측정.

### 2. Mid-training diagnostics

`gradient_norm` 을 epoch 별 추적 — vanishing/exploding 진단. PyTorch hook 으로 layer-wise gradient norm 측정.

### 3. Architectural decision

LSTM/GRU 의 forget gate 가 effective ρ 를 task-adaptive 하게 조정. RNN 변종 비교 시 effective spectral radius 측정.

### 4. Long-range modeling

특정 task 의 의존성 길이 $\tau$ 가 known 이면 $\rho_{\text{eff}}^\tau \ge \epsilon$ 의 minimum $\rho$ 추정.

### 5. Mamba / SSM 의 stability

State space matrix $A$ 의 eigenvalue 가 unit disk 내부 — 같은 spectral 분석 framework.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Spectral radius bound 가 tight | Non-normal $W$ 시 finite $T$ behavior 가 다를 수 있음 — Gelfand 는 asymptotic |
| $\sigma' \le 1$ uniform | $\sigma'(z)$ 가 $z$ 에 따라 변동 — task 별 average 측정 |
| Random gradient propagation | 학습된 RNN 은 specific direction 에 align — empirical 측정 |
| Fixed $W$ during analysis | $W$ 가 학습됨 — dynamic spectral radius |
| Tanh activation | ReLU RNN 은 unbounded $\sigma'$ — different bound 필요 |

---

## 📌 핵심 정리

$$\boxed{\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} W_{hh}^\top \mathrm{diag}(\sigma'(z_j)) \quad \text{— Jacobian 곱}}$$

$$\boxed{\rho(W_{hh}) \cdot \max \sigma' < 1 \implies \|\prod J_j\| \to 0 \;\; \text{(vanishing)}}$$

$$\boxed{\rho(W_{hh}) \cdot \max \sigma' > 1 \implies \|\prod J_j\| \to \infty \;\; \text{(exploding)}}$$

| Spectral 조건 | 결과 | 학습 가능성 |
|--------------|------|--------------|
| $\rho \cdot \sigma'_{\max} < 1$ | Vanishing | Long-range 학습 불가 |
| $\rho \cdot \sigma'_{\max} = 1$ | Neutral | 이상적 (orthogonal init Ch3-04) |
| $\rho \cdot \sigma'_{\max} > 1$ | Exploding | Clipping (Ch3-03) 필요 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\rho(W_{hh}) = 0.95$ 인 RNN 에서 $\tanh$ 의 average $\sigma' \approx 0.5$ 라 가정. 100 step 후 gradient norm 의 decay rate 는?

<details>
<summary>해설</summary>

**Effective rate**: $\rho_{\text{eff}} = 0.95 \times 0.5 = 0.475$

**100 step decay**:
$$
\|\delta_{100}\| / \|\delta_0\| \approx (0.475)^{100} \approx 1.6 \times 10^{-33}
$$

**해석**: 사실상 0 — long-range dependency 학습 거의 불가.

**비교**:
- $\sigma' = 1$ (no saturation): $0.95^{100} \approx 0.006$ — 여전히 작지만 측정 가능
- $\sigma' = 0.1$ (heavy saturation): $0.095^{100} \approx 10^{-100}$ — 완전 vanishing

**결론**: Saturation 이 vanishing 을 dominate — $\tanh$ RNN 의 경우 ρ 만 키워도 (1.0) 부족, **gating** 이 필요. $\square$

</details>

**문제 2** (심화): Non-normal matrix $W$ 에서 spectral radius 와 operator norm 이 다를 수 있다. $W = \begin{pmatrix} 1 & 10 \\ 0 & 1 \end{pmatrix}$ 의 ρ 와 $\|W\|_2$ 를 계산하라. Finite-time behavior 가 어떤가?

<details>
<summary>해설</summary>

**Spectral radius**: 고유값 $\lambda_{1,2} = 1, 1$ (반복 root) → $\rho(W) = 1$

**Operator norm**:
$$
W^\top W = \begin{pmatrix} 1 & 10 \\ 10 & 101 \end{pmatrix}, \quad \det(W^\top W - \lambda I) = (1-\lambda)(101-\lambda) - 100 = \lambda^2 - 102\lambda + 1
$$

$\lambda = (102 \pm \sqrt{102^2 - 4})/2 \approx 102, 0.01$

$\sigma_{\max}(W) = \sqrt{102} \approx 10.1$

**$\|W\|_2 \approx 10.1 \gg \rho(W) = 1$**

**Finite-time**:
- Gelfand: $\lim \|W^k\|^{1/k} = \rho = 1$ — long-term neutral
- Short-term: $\|W^k\| \sim 10 k$ — **선형 증가** (Jordan block 의 영향)

**RNN 적용**:
- $\rho = 1$ 으로 init 해도 finite $T$ 에서 norm 폭발 가능
- Pascanu 의 spectral 정리는 asymptotic — practical 학습은 short-term dynamics 도 중요
- **Orthogonal init** (Ch3-04) 가 우월: $\sigma_{\max} = \sigma_{\min} = 1$ → finite-time 도 안정 $\square$

</details>

**문제 3** (논문 비평): Pascanu 2013 은 "$\rho < 1$ vanishing, $\rho > 1$ exploding" 의 **충분조건** 이라 명시. 필요조건은 아닐 수 있다. 이 차이가 어떻게 LSTM 의 동기와 연결되는가?

<details>
<summary>해설</summary>

**Sufficient vs Necessary**:

**Pascanu 정리 (sufficient)**:
- $\rho \cdot \sigma' < 1 \Rightarrow$ vanishing
- 그러나 $\rho \cdot \sigma' \ge 1$ 이라도 vanishing 가능 (specific direction 에서)

**왜 충분조건만**:
- Spectral radius 는 average 행동, individual direction 은 $\rho_{\text{eff}}^{\text{dir}}$ 가 다양
- Non-normal matrix 에서 transient growth 가능
- $\sigma'$ 가 각 step 에서 다름 — 평균이 아닌 product 의 행동

**LSTM 의 동기**:

LSTM 은 Pascanu 의 spectral 분석을 *우회* 합니다:

1. **Cell state 의 additive update** (Ch4-03):
   $$
   \frac{\partial c_t}{\partial c_{t-1}} = f_t \quad (\text{element-wise, not matrix})
   $$
   Jacobian 곱이 **scalar product** $\prod f_t$ 로 단순화 — spectral radius 무관

2. **Forget gate $f_t \in [0, 1]$**:
   - $f_t \approx 1$: gradient 보존 (CEC)
   - $f_t \approx 0$: 의도적 forget
   - **Adaptive control** of effective decay rate

3. **Direct path**:
   - Cell state $c$ 가 hidden state $h$ 와 분리
   - Gradient flow 가 $c$ 의 path 를 따라 흐름
   - $h$ 의 path 는 여전히 vanishing 하지만 *less critical*

**결론**: Pascanu 의 충분조건은 RNN 의 *유일한* 한계가 아니라 *대표적* 한계. LSTM 은 이 한계를 벗어나는 architecture 변경. 그러나 LSTM 도 완전 해결은 아님 — *partial* solution. Transformer 는 곱 자체를 제거 (attention 이 direct connection). $\square$

</details>

---

<div align="center">

[◀ 이전](../ch2-bptt/05-rtrl.md) | [📚 README](../README.md) | [다음 ▶](./02-saturation-problem.md)

</div>
