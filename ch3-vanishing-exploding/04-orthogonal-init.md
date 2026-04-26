# 04. Orthogonal Initialization (Saxe 2014)

## 🎯 핵심 질문

- Saxe 2014 의 *Exact Solutions to the Nonlinear Dynamics of Learning* 가 제안한 orthogonal initialization 이 어떻게 $\rho = 1$ 정확히 달성하는가?
- Random orthogonal matrix $W$ 의 $W^\top W = I$ 가 모든 singular value $= 1$ 을 보장하는 이유?
- 왜 orthogonal init 이 finite-time gradient norm 을 보존 (depth-independent dynamics) 하는가?
- PyTorch `nn.init.orthogonal_` 의 implementation — QR decomposition 의 활용
- Vanilla RNN 의 Adding problem (long dependency) 에서 orthogonal init 이 학습 가능성을 어떻게 증명하는가?

---

## 🔍 왜 orthogonal init 이 RNN 의 spectral 문제 해법인가

Pascanu 2013 (Ch3-01) 의 요구: $\rho = 1$ 정확히 유지. Random Gaussian init 은:
- Asymptotic 으로 $\rho \approx 1$ (circular law)
- 그러나 finite-time variance, non-normal matrix 의 transient instability

**Orthogonal init** 은 이를 정확히 해결:

1. **모든 singular value $= 1$** — operator norm = spectral radius = 1
2. **Normal matrix** — Gelfand 의 spectral 식이 점잖게 작동
3. **Finite-time stable** — $\|W^k\|_2 = 1$ for all $k$
4. **Linear regime 에서 perfect** — saturation 없는 RNN 은 정확히 정보 보존

이 문서는 orthogonal init 의 정확한 정의, theoretical guarantee, PyTorch implementation 을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [03-gradient-clipping.md](./03-gradient-clipping.md) — Exploding 대응
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): **Orthogonal matrix**, QR decomposition, SVD
- 정의: $W$ orthogonal $\Leftrightarrow$ $W^\top W = I$ $\Leftrightarrow$ all singular values $= 1$
- (선택) Random matrix theory: Haar measure on orthogonal group $O(n)$

---

## 📖 직관적 이해

### Orthogonal Matrix 의 기하학적 의미

```
         |
   y     | rotation by 30°
   ↑     |
   |     |       ↗ Wy
   ●━━━━━┃━━━━━●
   |     |   ↗
   ●━━━━━●  ↗  Wx
         | ↗
         |↗
   ──────●────────→ x
```

Orthogonal $W$: rotation + reflection 만, scaling 없음. **거리 보존**: $\|Wx\| = \|x\|$.

### 모든 Singular Value = 1

$W^\top W = I$ ⇔ $W$ 의 행 (또는 열) 들이 서로 수직, 단위 norm.

SVD: $W = U \Sigma V^\top$ 시 $W^\top W = V \Sigma^2 V^\top = I$ → $\Sigma = I$ → all $\sigma_i = 1$.

**의미**:
- $\|W\|_2 = \sigma_{\max} = 1$
- $\rho(W) = $ largest $|\lambda|$ — orthogonal 의 eigenvalue 는 unit circle 위 → $\rho = 1$
- 모든 방향에서 distance preservation

### RNN 의 Forward 와 Backward Stability

Linear RNN $h_t = W h_{t-1}$ (no activation):
- Forward: $h_t = W^t h_0$, $\|h_t\| = \|h_0\|$ — 정보 보존
- Backward: $\delta_0 = (W^\top)^t \delta_t$, $\|\delta_0\| = \|\delta_t\|$ — gradient 보존

활성화 함수의 saturation 만 vanishing 의 원인 → Linear regime 에서 perfect.

### QR Decomposition 으로 Sample

Random orthogonal matrix sampling:

```
1. M = randn(N, N)        # random matrix
2. Q, R = qr(M)            # QR decomposition
3. d = sign(diag(R))       # sign correction (Haar measure)
4. W = Q · diag(d)          # orthogonal, uniform on O(n)
```

PyTorch `nn.init.orthogonal_` 가 이 알고리즘을 구현.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Orthogonal Matrix

$W \in \mathbb R^{n \times n}$ 이 **orthogonal** ⇔ $W^\top W = W W^\top = I$.

성질:
- $W^{-1} = W^\top$
- $\det(W) = \pm 1$
- 모든 singular value $= 1$
- 모든 eigenvalue $|\lambda| = 1$ (unit circle 위)

### 정의 4.2 — Orthogonal Initialization

$W_{hh}$ 를 **uniform Haar measure** on $O(n)$ 에서 샘플:

$$
W_{hh} \sim \text{Uniform}(O(n))
$$

QR decomposition 으로 sampling: $W_{hh} = Q \cdot \mathrm{diag}(\mathrm{sgn}(\mathrm{diag}(R)))$.

### 정의 4.3 — Scaled Orthogonal

ReLU 같은 unbounded activation 시 scale factor $g > 1$:

$$
W = g \cdot Q
$$

PyTorch `nn.init.orthogonal_(W, gain=g)` 가 이를 적용.

### 정의 4.4 — Weight Matrix Decomposition

$W = (1 - \alpha) I + \alpha Q$ — identity 와 orthogonal 의 mixture (variants).

### 정의 4.5 — Forward/Backward Norm Preservation

Linear regime 에서:

$$
\|h_t\| = \|W^t h_0\| = \|h_0\|, \quad \|\delta_0\| = \|(W^\top)^T \delta_T\| = \|\delta_T\|
$$

---

## 🔬 정리와 결과

### 정리 4.1 — Orthogonal Matrix 의 Spectral Properties

$W$ orthogonal 이면:
1. 모든 singular value $\sigma_i(W) = 1$
2. $\|W\|_2 = 1$
3. 모든 eigenvalue 는 unit circle 위: $|\lambda_i(W)| = 1$
4. $\rho(W) = 1$

**증명**:

(1) $W^\top W = I$ 의 eigenvalue 는 모두 1, 즉 $\sigma_i^2 = 1 \Rightarrow \sigma_i = 1$.

(2) $\|W\|_2 = \sigma_{\max}(W) = 1$.

(3) Schur decomposition $W = U T U^*$ ($U$ unitary, $T$ upper triangular). $W$ orthogonal $\Rightarrow$ $W^\top W = I$. Diagonal of $T$ 가 eigenvalue, $|\det W| = \prod |\lambda_i| = 1$. $W$ 의 norm = 1 + eigenvalue magnitudes.

좀 더 직접: $W^\top W = I$ ⇒ $W$ 가 length-preserving ($\|Wx\|^2 = x^\top W^\top W x = \|x\|^2$). Eigenvalue $\lambda$, eigenvector $v$ 에 대해 $\|Wv\| = |\lambda| \|v\| = \|v\|$ (length preservation) ⇒ $|\lambda| = 1$.

(4) Direct from (3). $\square$

### 정리 4.2 — Linear RNN 의 Information Preservation

Linear RNN $h_t = W h_{t-1}$, $W$ orthogonal:

$$
\|h_t\| = \|h_0\| \quad \forall t
$$

**증명**: $\|W^t h_0\| = \|h_0\|$ by repeated length preservation. $W^t$ 도 orthogonal (orthogonal group 이 closed under multiplication). $\square$

**의미**: 모든 시간에서 정보 손실 없음 — 이상적인 long-range modeling 의 가능성.

### 정리 4.3 — Saxe 2014 의 Layer-wise Dynamics

Deep linear network (orthogonal init) 의 학습 dynamics:

$$
\frac{dW^{(\ell)}}{dt} = \text{(coupling terms across layers)}
$$

Saxe 2014 의 핵심: orthogonal init 에서 **각 layer 의 gradient norm 이 depth 에 무관**. 따라서 모든 layer 가 동시에 학습 — **plateau 회피**.

**결과**: ResNet 이전에 deep network 학습 가능성 입증.

### 정리 4.4 — Random Orthogonal Sampling

QR decomposition 의 $Q$ 가 Haar measure on $O(n)$ 을 sample 하려면 sign correction 필요:

$$
W = Q \cdot \mathrm{diag}(\mathrm{sgn}(\mathrm{diag}(R)))
$$

**증명** (sketch): QR 의 $Q$ 자체는 Haar 와 다른 distribution (대각선 부호 dependence). Sign correction 후 uniform on $O(n)$. $\square$

### 정리 4.5 — Orthogonal Init 의 Edge of Chaos

Random orthogonal $W$ 의 Lyapunov exponent (linear regime):

$$
\Lambda = \log \rho(W) + \mathbb E[\log \sigma'(z)] = 0 + 0 = 0
$$

(Linear: $\sigma' = 1$). 정확히 edge of chaos — 정보 보존, 학습 가능.

비선형 regime (tanh) 에서:

$$
\Lambda = 0 + \mathbb E[\log \sigma'(z)] \le 0
$$

여전히 saturation 영향. 그러나 random Gaussian 보다 좋음.

---

## 💻 NumPy / PyTorch 검증

### 실험 1 — Orthogonal Sampling

```python
import numpy as np

def orthogonal_init(n, seed=0):
    """QR decomposition based orthogonal sampling"""
    rng = np.random.RandomState(seed)
    M = rng.randn(n, n)
    Q, R = np.linalg.qr(M)
    # Sign correction
    d = np.sign(np.diag(R))
    Q = Q * d
    return Q

W = orthogonal_init(50)
print(f'W^T W = I? {np.allclose(W.T @ W, np.eye(50))}')
print(f'All singular values = 1? {np.allclose(np.linalg.svd(W, compute_uv=False), 1)}')
print(f'ρ(W) = {max(abs(np.linalg.eigvals(W))):.6f}')
print(f'Eigenvalues on unit circle: {np.allclose(np.abs(np.linalg.eigvals(W)), 1)}')
```

### 실험 2 — Forward Norm Preservation (Linear)

```python
H = 50
W = orthogonal_init(H)

h = np.random.randn(H)
norms = [np.linalg.norm(h)]
for t in range(100):
    h = W @ h
    norms.append(np.linalg.norm(h))

print(f'||h_0||  = {norms[0]:.6f}')
print(f'||h_50|| = {norms[50]:.6f}')
print(f'||h_100||= {norms[100]:.6f}')
print(f'Max deviation: {max(abs(np.array(norms) - norms[0])):.2e}')
# 정확히 보존 (numerical precision 내)
```

### 실험 3 — Gaussian vs Orthogonal Init 비교

```python
def measure_decay(W, T):
    h = np.random.randn(W.shape[0]); h /= np.linalg.norm(h)
    for t in range(T):
        h = np.tanh(W @ h)   # tanh 활성화
    return np.linalg.norm(h)

T = 100
for trial in range(3):
    np.random.seed(trial)
    
    # Gaussian
    W_gauss = np.random.randn(H, H) * (1 / np.sqrt(H))
    rho_gauss = max(abs(np.linalg.eigvals(W_gauss)))
    
    # Orthogonal
    W_orth = orthogonal_init(H, seed=trial)
    
    norm_gauss = measure_decay(W_gauss, T)
    norm_orth = measure_decay(W_orth, T)
    
    print(f'Trial {trial}: Gauss ρ={rho_gauss:.3f}, ||h||={norm_gauss:.4e}'
          f'  |  Orth ρ=1.0, ||h||={norm_orth:.4e}')
# Orthogonal 이 일관되게 1 근처 유지 (saturation 영향만 있음)
```

### 실험 4 — PyTorch nn.init.orthogonal_ 사용

```python
import torch
import torch.nn as nn

rnn = nn.RNN(10, 50, batch_first=False, nonlinearity='tanh')
print(f'Default init ρ: {max(abs(torch.linalg.eigvals(rnn.weight_hh_l0).numpy())):.4f}')

# Orthogonal init
nn.init.orthogonal_(rnn.weight_hh_l0)
W = rnn.weight_hh_l0.detach().numpy()
print(f'After orthogonal_: ρ = {max(abs(np.linalg.eigvals(W))):.4f}')
print(f'σ_max = {max(np.linalg.svd(W, compute_uv=False)):.4f}')
print(f'σ_min = {min(np.linalg.svd(W, compute_uv=False)):.4f}')
# σ_max = σ_min = 1 정확히
```

### 실험 5 — Adding Problem (Long Dependency Test)

```python
# Le 2015, Saxe 2014 등 standard benchmark
# 두 random number 의 합을 sequence 끝에서 출력
# Sequence length 가 길수록 long-range dependency 학습 능력 측정

def adding_problem(T):
    """Generate one sample"""
    seq = np.zeros((T, 2))
    seq[:, 0] = np.random.uniform(0, 1, T)
    # Two positions marked with 1 (in second feature)
    pos = np.random.choice(T, 2, replace=False)
    seq[pos, 1] = 1.0
    target = seq[pos, 0].sum()
    return seq, target

class AddingProblemRNN(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.rnn = nn.RNN(2, H, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
    def reset_orthogonal(self):
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

# Test: T = 50 sequence with default vs orthogonal init
def train_adding(use_orthogonal, T_seq=50, n_steps=200):
    H = 100
    model = AddingProblemRNN(H)
    if use_orthogonal:
        model.reset_orthogonal()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    losses = []
    for step in range(n_steps):
        # Batch
        seqs, targets = [], []
        for _ in range(32):
            s, t = adding_problem(T_seq)
            seqs.append(s); targets.append(t)
        x = torch.tensor(np.stack(seqs), dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
        pred = model(x)
        loss = ((pred - y)**2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses

torch.manual_seed(0); np.random.seed(0)
losses_default = train_adding(use_orthogonal=False)
torch.manual_seed(0); np.random.seed(0)
losses_orth = train_adding(use_orthogonal=True)

print(f'Default init  final loss: {np.mean(losses_default[-20:]):.4f}')
print(f'Orthogonal init final loss: {np.mean(losses_orth[-20:]):.4f}')
# Orthogonal 이 더 빠르게 수렴 (long dependency 학습 가능)
```

---

## 🔗 실전 활용

### 1. PyTorch RNN/LSTM 의 표준 init

```python
nn.init.orthogonal_(model.weight_hh_l0)
```
LSTM 의 4 gate weight 도 동일하게.

### 2. Transformer 의 query/key projection

Self-attention 의 $W_Q, W_K$ 도 orthogonal init 시 stable training. (실제로는 Xavier/Glorot 가 표준)

### 3. Image classification 의 ResNet

Conv layer 도 orthogonal-style init 가능 (`he_orthogonal_`). Saxe 2014 가 vanilla deep net 의 학습 가능성 입증.

### 4. Spectral normalization 의 출발점

Miyato 2018 의 GAN spectral norm 이 orthogonal 의 generalization — 매 step 후 $\sigma_{\max} = 1$ 유지.

### 5. Mamba / SSM 의 stability

State matrix $A$ 의 eigenvalue 가 unit disk 안 — orthogonal/diagonal init 의 일반화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Square matrix $H \times H$ | Non-square 의 경우 partial orthogonal (`semi_orthogonal`) |
| Linear regime | Tanh saturation 영향 여전 — gating 필요 |
| Initial condition | 학습 진행 시 ρ drift — periodic re-orthogonalization 가능 |
| Single matrix | Multi-matrix interaction (LSTM 4 gates) 에서 효과 제한 |
| Random direction | Task-specific structure 학습은 별도 |

---

## 📌 핵심 정리

$$\boxed{W^\top W = I \implies \sigma_i(W) = 1 \;\forall i \implies \rho(W) = 1}$$

$$\boxed{\|h_t\| = \|h_0\| \text{ in linear RNN} \quad \text{(perfect information preservation)}}$$

$$\boxed{\text{QR decomp} + \text{sign correction} = \text{Haar measure on } O(n)}$$

| Init Type | $\sigma_{\max}$ | $\rho$ | Variance | Use |
|-----------|-----------------|--------|----------|-----|
| **Gaussian** $\mathcal N(0, 1/H)$ | ~2 (asymp) | ~1 | High | Default 그러나 RNN 위험 |
| **Glorot/Xavier** | ~1 (대략) | ~1 | Medium | MLP 표준 |
| **He** | ~$\sqrt{2}$ | ~$\sqrt{2}$ | Medium | ReLU MLP |
| **Orthogonal** | 1 정확히 | 1 정확히 | 0 (deterministic σ) | RNN/LSTM 권장 |
| **Identity** $I$ | 1 정확히 | 1 정확히 | 0 | IRNN (Ch3-05) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $2 \times 2$ orthogonal matrix 의 일반 형태를 구하라. Rotation 과 reflection 의 구분은?

<details>
<summary>해설</summary>

$W^\top W = I$ 의 일반 해:

$$
W = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \quad (\det W = +1, \text{ rotation})
$$

또는

$$
W = \begin{pmatrix} \cos\theta & \sin\theta \\ \sin\theta & -\cos\theta \end{pmatrix} \quad (\det W = -1, \text{ reflection})
$$

**Eigenvalues**:
- Rotation: $e^{\pm i\theta}$ — unit circle 위
- Reflection: $\pm 1$ — also unit magnitude

**검증**:

Rotation 예: $\theta = 30°$:
$$
W = \begin{pmatrix} \sqrt{3}/2 & -1/2 \\ 1/2 & \sqrt{3}/2 \end{pmatrix}
$$

$W^\top W = I$, $\det = 1$. $|\lambda| = 1$. $\square$

</details>

**문제 2** (심화): QR decomposition 후 sign correction 이 왜 필요한가? Sign 없는 경우의 distribution 이 Haar 와 어떻게 다른가?

<details>
<summary>해설</summary>

**QR decomposition**: $M = QR$
- $Q$ orthogonal
- $R$ upper triangular with $R_{ii} > 0$ (보통의 convention)

**$Q$ 의 distribution**:
- $M$ 이 random Gaussian matrix
- $R_{ii} > 0$ convention 이 $Q$ 의 column 부호를 *constraint*
- 따라서 $Q$ 의 distribution 이 sign 에 *bias* 가 있음

**Haar measure 의 정의**:
- $O(n)$ 위의 uniform distribution
- 어떤 transformation 에도 invariant

**Sign correction 의 필요성**:

만약 $R_{ii} > 0$ constraint 없으면 $Q$ 의 column 이 $\pm$ 두 가지 가능 — uniform.
$R_{ii} > 0$ 강제 시 $Q$ 의 column 부호가 fix 됨.

$Q' = Q \cdot \mathrm{diag}(\mathrm{sgn}(\mathrm{diag}(R)))$ 로 sign 을 random 으로 set:

$$
Q' \sim \text{Uniform}(O(n))
$$

**왜 중요한가**:
- Without correction: $Q$ 의 distribution 에 systematic bias
- With correction: 정확한 Haar — random matrix theory results 적용 가능

**실용**:
- Most ML 적용에서 차이 미세 — 다른 random 변동에 묻힘
- 그러나 정확한 random orthogonal 이 필요한 setting (random matrix theory 검증, advanced spectral methods) 에서 중요
- PyTorch `nn.init.orthogonal_` 가 sign correction 포함 ✓

$\square$

</details>

**문제 3** (논문 비평): Orthogonal init 이 vanilla RNN 에 좋다면, LSTM 도 orthogonal init 이어야 하는가? LSTM 의 4 gate weight 들에 대한 init 권장과 그 이유는?

<details>
<summary>해설</summary>

**LSTM 의 4 weight matrices**:
- $W_f$ (forget): $\sigma$ activation
- $W_i$ (input): $\sigma$
- $W_g$ (candidate): $\tanh$
- $W_o$ (output): $\sigma$

**각각의 dynamics**:

1. **Cell state $c_t = f_t c_{t-1} + i_t g_t$**:
   - $\partial c_t / \partial c_{t-1} = f_t$ — element-wise scalar
   - **Spectral radius 무관**: matrix product 가 element-wise product 로 단순화
   - Forget gate 자체의 init 이 더 중요

2. **Hidden $h_t = o_t \tanh(c_t)$**:
   - 단일 step transformation, recurrence 에 직접 안 들어감
   - Output gate 의 init 영향 적음

**권장 init**:

1. **Forget gate $W_f, b_f$** — *가장 중요*:
   - $b_f = 1$ (Jozefowicz 2015) — 초기에 cell state 보존 (Ch4-04)
   - $W_f$: orthogonal 또는 Glorot

2. **Input gate $W_i$**:
   - 일반 init (Glorot, He)

3. **Candidate gate $W_g$**:
   - $\tanh$ activation — orthogonal 또는 Glorot

4. **Output gate $W_o$**:
   - 일반 init

**$W_{hh}$ 의 4 stack**:

PyTorch 의 `weight_hh` 는 4 gate weight 의 stacked matrix $\in \mathbb R^{4H \times H}$. Orthogonal init:

```python
nn.init.orthogonal_(rnn.weight_hh_l0)   # 전체를 orthogonal 로
```

그러나 4 gate 가 각각 $H \times H$ — block-wise orthogonal 이 더 나을 수도:

```python
W = rnn.weight_hh_l0.data
for i in range(4):
    nn.init.orthogonal_(W[i*H:(i+1)*H])
```

**Empirical**:
- Vanilla LSTM 에서 orthogonal vs Glorot 의 차이 미세 (gating 이 main effect)
- $b_f = 1$ 초기화가 더 큰 영향 (Adding problem 에서 확인 가능)

**결론**: Orthogonal init 이 LSTM 에 *해롭지 않음*, 그러나 LSTM 의 *vanishing 문제는 architecture* 로 해결 — init 의 marginal benefit. RNN/IRNN 에서는 init 이 critical, LSTM 에서는 forget bias 가 critical. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-gradient-clipping.md) | [📚 README](../README.md) | [다음 ▶](./05-irnn.md)

</div>
