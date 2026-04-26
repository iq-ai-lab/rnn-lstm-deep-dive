# 05. Identity Initialization 과 IRNN (Le 2015)

## 🎯 핵심 질문

- Le 2015 의 *A Simple Way to Initialize Recurrent Networks of Rectified Linear Units* 가 어떻게 ReLU + $W_{hh} = I$ init 으로 long-range dependency 학습을 가능하게 했는가?
- $h_t = h_{t-1} + \mathrm{ReLU}(\ldots)$ 형태가 왜 residual connection 의 RNN 버전이며 gradient 보존을 자연스럽게 만드는가?
- ReLU 의 unbounded 출력이 왜 IRNN 에서 exploding 위험이 되며 어떻게 gradient clipping (Ch3-03) 으로 완화하는가?
- IRNN 의 Adding Problem (T=300, T=1000) 에서 LSTM 과 비등한 성능 — 어떻게 성취했는가?
- IRNN 이 ResNet 과 Transformer 의 residual connection 정신과 어떻게 연결되는가?

---

## 🔍 왜 IRNN 이 RNN 의 simple but effective 해법인가

LSTM (Ch4) 이 RNN 의 vanishing 을 가장 일반적으로 해결하지만 4 gates × matrix multiplication 의 복잡성. Le 2015 는 더 단순한 해법:

1. **$W_{hh} = I$ 로 초기화** — Identity 가 정보 보존의 perfect starting point
2. **ReLU 활성화** — Linear region 에서 $\sigma'(z) = 1$, gradient 보존
3. **Vanilla RNN architecture** — LSTM 의 복잡성 없음

이 단순한 변경으로:
- Adding Problem $T = 300, 1000$ 에서 LSTM 과 동등 성능
- Sequential MNIST $T = 784$ 에서 90%+ 정확도
- 파라미터 수 LSTM 의 1/4

이 문서는 IRNN 의 이론적 정당성과 ReLU + Identity init 의 메커니즘을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [04-orthogonal-init.md](./04-orthogonal-init.md) — Orthogonal init, $\rho = 1$
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): ReLU 의 derivative, dead unit
- 정의: ReLU $\sigma(z) = \max(0, z)$, $\sigma'(z) = \mathbb 1[z > 0] \in \{0, 1\}$

---

## 📖 직관적 이해

### Identity Init 의 효과

$W_{hh} = I$, ReLU activation:

$$
h_t = \mathrm{ReLU}(W_{hh} h_{t-1} + W_{xh} x_t) = \mathrm{ReLU}(h_{t-1} + W_{xh} x_t)
$$

$h_{t-1}$ 가 positive 면 ReLU 가 identity → $h_t = h_{t-1} + W_{xh} x_t$ — **Residual update**.

### Residual Connection 비교

ResNet:
$$
h^{(\ell+1)} = h^{(\ell)} + f(h^{(\ell)})
$$

IRNN (positive $h$):
$$
h_t = h_{t-1} + W_{xh} x_t   \quad (\text{ReLU 가 trivial 한 region})
$$

**같은 정신**: $h$ 의 *변화* 만 학습, base 는 보존.

### Gradient Flow

$$
\frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}(\mathbb 1[h_{t-1} + W_{xh} x_t > 0]) \cdot I = \mathrm{diag}(\mathbb 1[\cdot]) \cdot I
$$

대부분의 dimension 에서 $> 0$ 이면 Jacobian $\approx I$ — **gradient 1 차원 곱셈** (vanishing 안 됨!).

### Tanh vs ReLU 의 RNN Trade-off

| | Tanh | ReLU (IRNN) |
|--|------|-------------|
| **$\sigma'$** | $\le 1$ (saturation) | $\in \{0, 1\}$ |
| **$h$ bound** | $|h| < 1$ | unbounded |
| **Vanishing** | 강함 | 약함 (positive region 에서 perfect) |
| **Exploding** | 약함 | 강함 (특히 ρ > 1) |
| **Dead unit** | 없음 | 가능 (모두 negative) |

---

## ✏️ 엄밀한 정의

### 정의 5.1 — IRNN

$$
h_t = \mathrm{ReLU}(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h)
$$

**Initialization**:
- $W_{hh} = I$ (identity matrix)
- $W_{xh} \sim \mathcal N(0, 0.001)$ — small to avoid saturation
- $b_h = 0$

### 정의 5.2 — np-RNN (np = nonparametric, Le 2015 §3)

Identity init 의 variant:
$$
W_{hh} = (1 - \alpha) I + \alpha M, \quad M \sim \mathcal N(0, \sigma^2/H)
$$

$\alpha$ 가 작으면 거의 identity, 큰 random 추가 가능.

### 정의 5.3 — Linear Region of ReLU RNN

$\Omega_+ := \{(h, x) : h_{t-1} + W_{xh} x_t > 0\text{ element-wise}\}$.

이 영역에서:
$$
h_t = h_{t-1} + W_{xh} x_t \quad (\text{ReLU 가 identity})
$$

### 정의 5.4 — Dead Unit

$h_{t,i} = 0$ for all $t$ (한 dimension 이 항상 negative pre-activation). ReLU 의 known issue.

### 정의 5.5 — Gradient Through ReLU

$$
\sigma'(z) = \mathbb 1[z > 0] \in \{0, 1\}^H
$$

- $z > 0$: gradient 1 (perfect propagation)
- $z \le 0$: gradient 0 (block)

---

## 🔬 정리와 결과

### 정리 5.1 — IRNN 의 Initial Gradient Preservation

$W_{hh} = I$, ReLU, $b_h = 0$ init 시 forward $h_t = h_{t-1}$ ($x_t$ contribution 무시 시). Backward:

$$
\frac{\partial h_t}{\partial h_0} \approx \prod_{j=1}^{t} \mathrm{diag}(\mathbb 1[h_j > 0]) \cdot I
$$

$h_j$ 가 positive 인 dimensions 에서 gradient 정확히 보존, negative 에서 0.

**증명**: Jacobian $\partial h_j / \partial h_{j-1} = \mathrm{diag}(\sigma'(z_j)) \cdot W_{hh} = \mathrm{diag}(\mathbb 1[z_j > 0]) \cdot I$. Repeated product = element-wise AND of indicators. $\square$

**의미**: Active neurons 에서 perfect gradient — vanishing 의 root cause (matrix product 의 곱셈적 누적) 제거.

### 정리 5.2 — Residual Connection 의 RNN 등가성

IRNN 에서 ReLU 가 trivial 인 region:

$$
h_t = h_{t-1} + W_{xh} x_t = h_0 + \sum_{s=1}^{t} W_{xh} x_s
$$

이는 **ResNet 의 RNN 버전**. 정보 손실 없음.

### 정리 5.3 — Le 2015 의 Adding Problem 결과

$T = 300$ Adding Problem (random 두 숫자 합):
- LSTM: ~95% 정확도
- IRNN: ~95% 정확도
- Vanilla RNN (Glorot init, tanh): ~50% (chance level)

**해석**: IRNN 이 단순한 4 init 변경으로 LSTM 동등 성능 — vanishing 의 architectural 해결의 *대안*.

### 정리 5.4 — IRNN 의 Sequential MNIST

$T = 784$ pixel sequence, 10-class classification:
- LSTM: ~98.5%
- IRNN: ~95-97% (configurations 따라)
- Vanilla RNN: training 못 함

T = 784 long dependency 에서 IRNN 의 효과 입증.

### 정리 5.5 — Exploding 위험과 Clipping 필수

ReLU 는 unbounded → $W_{hh}$ 가 학습으로 ρ > 1 으로 drift 시 exploding. **Gradient clipping** (Ch3-03) 이 필수.

Le 2015 의 권장: $\theta = 1.0 \sim 10.0$.

---

## 💻 PyTorch 구현 검증

### 실험 1 — IRNN 구현

```python
import torch
import torch.nn as nn

class IRNN(nn.Module):
    """Le 2015 IRNN — ReLU + Identity init"""
    def __init__(self, D, H, init_input_scale=0.001):
        super().__init__()
        self.D, self.H = D, H
        self.W_xh = nn.Linear(D, H, bias=False)
        self.W_hh = nn.Linear(H, H, bias=True)
        self._init_weights(init_input_scale)
    
    def _init_weights(self, input_scale):
        # Identity init for W_hh
        with torch.no_grad():
            self.W_hh.weight.copy_(torch.eye(self.H))
            self.W_hh.bias.zero_()
        # Small init for W_xh
        nn.init.normal_(self.W_xh.weight, mean=0, std=input_scale)
    
    def forward(self, x_seq, h0=None):
        T, B, _ = x_seq.shape
        h = torch.zeros(B, self.H, device=x_seq.device) if h0 is None else h0
        hs = []
        for t in range(T):
            z = self.W_hh(h) + self.W_xh(x_seq[t])
            h = torch.relu(z)
            hs.append(h)
        return torch.stack(hs)

# Toy
torch.manual_seed(0)
irnn = IRNN(D=2, H=50)
x = torch.randn(20, 4, 2) * 0.5
hs = irnn(x)
print(f'Hidden states: {hs.shape}')
print(f'||h_0|| = {hs[0].norm(dim=1).mean():.4f}')
print(f'||h_19||= {hs[-1].norm(dim=1).mean():.4f}')
# 초기에 h ≈ 0, 점차 누적
```

### 실험 2 — Adding Problem on IRNN

```python
import numpy as np

def adding_problem(T, B):
    seqs = np.zeros((T, B, 2))
    seqs[:, :, 0] = np.random.uniform(0, 1, (T, B))
    pos1 = np.random.randint(0, T // 2, B)
    pos2 = T // 2 + np.random.randint(0, T // 2, B)
    targets = np.zeros(B)
    for b in range(B):
        seqs[pos1[b], b, 1] = 1.0
        seqs[pos2[b], b, 1] = 1.0
        targets[b] = seqs[pos1[b], b, 0] + seqs[pos2[b], b, 0]
    return torch.tensor(seqs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

class AddingIRNN(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.rnn = IRNN(D=2, H=H)
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        hs = self.rnn(x)
        return self.fc(hs[-1]).squeeze(-1)

class AddingTanhRNN(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.rnn = nn.RNN(2, H, nonlinearity='tanh')
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        hs, _ = self.rnn(x)
        return self.fc(hs[-1]).squeeze(-1)

class AddingLSTM(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.rnn = nn.LSTM(2, H)
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        hs, _ = self.rnn(x)
        return self.fc(hs[-1]).squeeze(-1)

def train_adding(model_cls, T, n_steps=200, H=100, lr=1e-3):
    torch.manual_seed(42); np.random.seed(0)
    model = model_cls(H)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        x, y = adding_problem(T, B=64)
        pred = model(x)
        loss = ((pred - y)**2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return np.mean(losses[-20:])

T = 100
print(f'Adding Problem T={T}:')
for cls, name in [(AddingTanhRNN, 'Vanilla RNN (tanh)'), (AddingIRNN, 'IRNN'), (AddingLSTM, 'LSTM')]:
    final = train_adding(cls, T)
    print(f'  {name:20s}: final loss = {final:.4f}')
# IRNN 과 LSTM 이 vanilla RNN 보다 훨씬 낮은 loss
```

### 실험 3 — Identity Init 의 첫 step Gradient

```python
def measure_gradient_through_T(model_cls, T, H=50):
    torch.manual_seed(0)
    model = model_cls(H)
    
    # Set h_0 to require_grad
    if isinstance(model, AddingIRNN):
        # Manual forward to get h_T as function of h_0
        x = torch.randn(T, 1, 2)
        h0 = torch.zeros(1, H, requires_grad=True)
        h = h0
        for t in range(T):
            z = model.rnn.W_hh(h) + model.rnn.W_xh(x[t])
            h = torch.relu(z)
        # Norm of dh_T / dh_0
        h.sum().backward()
        return h0.grad.norm().item()
    return None

for T in [10, 50, 100, 200]:
    grad = measure_gradient_through_T(AddingIRNN, T)
    print(f'IRNN T={T:3d}: ||∂h_T / ∂h_0|| = {grad:.4f}')
# Identity init 으로 처음에는 gradient 1 근처 유지
```

### 실험 4 — ReLU Dead Unit 측정

```python
def measure_dead_units(model, x):
    """Dead unit: h_t = 0 for all t in some dim"""
    with torch.no_grad():
        hs = model.rnn(x)   # (T, B, H)
        # Each (t, b) 에서 active 인 units
        active = (hs > 0).float()
        per_unit_active_rate = active.mean(dim=(0, 1))   # (H,)
        dead_units = (per_unit_active_rate < 0.01).sum().item()
    return dead_units, per_unit_active_rate

torch.manual_seed(0)
irnn = AddingIRNN(H=100)
x = torch.randn(50, 32, 2)

dead, rates = measure_dead_units(irnn, x)
print(f'Dead units (active < 1%): {dead}/100')
print(f'Avg active rate: {rates.mean():.4f}')
print(f'Min active rate: {rates.min():.4f}')

# 학습 후
opt = torch.optim.Adam(irnn.parameters(), lr=1e-3)
for _ in range(50):
    x, y = adding_problem(50, B=32)
    pred = irnn(x)
    loss = ((pred - y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

dead_after, rates_after = measure_dead_units(irnn, x)
print(f'After training: Dead units = {dead_after}/100')
# 학습 후 dead unit 비율 변화
```

### 실험 5 — IRNN 의 Spectral Drift

```python
def measure_spectral_radius(W):
    return max(abs(np.linalg.eigvals(W.detach().numpy()))).item() \
           if hasattr(W, 'detach') else max(abs(np.linalg.eigvals(W)))

torch.manual_seed(42)
irnn = AddingIRNN(H=50)
opt = torch.optim.Adam(irnn.parameters(), lr=1e-3)

rhos = [measure_spectral_radius(irnn.rnn.W_hh.weight)]
for step in range(100):
    x, y = adding_problem(50, B=32)
    pred = irnn(x)
    loss = ((pred - y)**2).mean()
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(irnn.parameters(), 1.0)
    opt.step()
    if step % 10 == 0:
        rhos.append(measure_spectral_radius(irnn.rnn.W_hh.weight))

print(f'ρ trajectory: {[f"{r:.3f}" for r in rhos[:5]]} ... {[f"{r:.3f}" for r in rhos[-5:]]}')
print(f'ρ initial: {rhos[0]:.4f}, final: {rhos[-1]:.4f}')
# Identity init (ρ=1) 이 학습 진행으로 drift
```

---

## 🔗 실전 활용

### 1. Long-document classification

T > 1000 token document 에서 IRNN 이 LSTM 과 비등. 단순한 architecture 의 장점 (memory, speed).

### 2. Time series with regular patterns

Periodicity 가 강한 task — IRNN 의 residual update 가 linear trend 학습 자연스러움.

### 3. Reinforcement learning policy

DRQN 같은 setting — LSTM 의 4x parameters 보다 IRNN 의 lean architecture 선호 가능.

### 4. ResNet 의 RNN 일반화

IRNN 이 ResNet 의 정신을 RNN 에 적용. Highway Network (Srivastava 2015) 도 같은 정신.

### 5. LSTM 의 alternative 평가

새로운 RNN architecture 제안 시 baseline 으로 IRNN — LSTM 보다 단순하지만 강력.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| ReLU positive region 에서 perfect | Dead unit 발생 시 학습 stuck |
| $W_{hh} = I$ 유지 | 학습 drift 로 ρ 변화 — clipping 필수 |
| Linear regime | Non-linear modeling 능력 LSTM 보다 약할 수도 |
| Single layer | Stacked IRNN 은 별도 분석 필요 |
| Small input scale | $W_{xh}$ 너무 크면 saturation 또는 instability |

---

## 📌 핵심 정리

$$\boxed{W_{hh} = I, \;\; \text{ReLU activation} \implies h_t \approx h_{t-1} + W_{xh} x_t \;\; (\text{positive region})}$$

$$\boxed{\frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}(\mathbb 1[z_t > 0]) \cdot I \quad (\text{element-wise gradient preservation})}$$

$$\boxed{\text{IRNN} \approx \text{LSTM on Adding Problem with 1/4 parameters}}$$

| RNN Variant | $W_{hh}$ Init | Activation | Vanishing 대응 |
|------------|--------------|-----------|-------------|
| **Vanilla (tanh)** | Glorot | $\tanh$ | 약함 |
| **Vanilla (orthogonal)** | Orthogonal $\rho=1$ | $\tanh$ | 중간 |
| **IRNN (Le 2015)** | Identity $I$ | ReLU | 강함 (residual) |
| **np-RNN** | $(1-\alpha)I + \alpha M$ | ReLU | 강함 |
| **LSTM** | Glorot/Orthogonal | $\sigma, \tanh$ + gates | 가장 강함 (CEC) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): IRNN 에서 모든 $h_{t-1}$ component 가 positive 인 경우, $T$ step 후 $h_T$ 의 정확한 형태를 구하라.

<details>
<summary>해설</summary>

**가정**: $h_{t-1,i} + (W_{xh} x_t)_i > 0$ for all $i, t$ — ReLU 가 항상 identity.

**Forward**:
$$
h_t = h_{t-1} + W_{xh} x_t
$$

**Unrolled**:
$$
h_T = h_0 + W_{xh}(x_1 + x_2 + \ldots + x_T) = h_0 + W_{xh} \sum_{s=1}^T x_s
$$

**의미**:
- 모든 input 이 동일 weight 로 합산
- Positional information 손실 (모든 $x_s$ symmetric)
- Residual sum 의 형태

**비교**:
- Tanh RNN: $h_T$ 가 $\tanh(W \cdots \tanh(W x_1) \cdots)$ — 복잡 nonlinear
- IRNN (linear region): 단순 linear sum
- Real IRNN: ReLU 의 boundary 가 nonlinear, partial linear

**Position-aware 가 중요한 task** (LM) 에서는 IRNN 만으로 부족 — positional encoding 또는 explicit time index 필요. $\square$

</details>

**문제 2** (심화): IRNN 의 $W_{xh}$ 가 작은 scale (0.001) 로 init 되는 이유를 explain 하고, large scale 시 어떤 문제가 발생하는가?

<details>
<summary>해설</summary>

**Small $W_{xh}$ 의 동기**:

1. **Non-linearity 회피**:
   - $z_t = h_{t-1} + W_{xh} x_t$ 의 $W_{xh} x_t$ contribution 이 작으면
   - $h_{t-1}$ 의 sign 이 $z_t$ 의 sign 결정
   - ReLU 의 boundary 거의 안 만남 → linear regime 유지

2. **Gradient 보존**:
   - Linear regime 에서 $\partial h_t / \partial h_{t-1} \approx I$
   - $W_{xh}$ 에 의한 perturbation 이 작으면 stability

3. **Initial dynamics**:
   - $h_0 = 0$, $h_1 = W_{xh} x_1 \approx 0$
   - Slow build-up — gradient 이 충분히 backward 흐를 시간

**Large $W_{xh}$ 의 문제**:

1. **ReLU dead units**:
   - $W_{xh} x_t$ 가 $-h_{t-1}$ 보다 크면 negative → $h_t = 0$
   - 한 번 dead 되면 다시 살아나기 어려움

2. **Gradient blocking**:
   - Dead 한 dimension 의 ReLU 가 gradient 차단
   - $\partial h_t / \partial h_{t-1}$ 의 일부 차원이 0

3. **Non-stationary dynamics**:
   - 매 step 큰 input perturbation → $h$ 의 분포 변화
   - 학습 dynamics 가 transient 에 dominate

**실용 권장 (Le 2015)**:
- $W_{xh} \sim \mathcal N(0, 0.001^2)$ — 매우 작음
- 학습으로 점차 증가 — gradient signal 따라 자연스럽게

**비교 (orthogonal init RNN)**:
- $W_{hh}$ 와 $W_{xh}$ 가 같은 scale 적합 (둘 다 perturbation)
- IRNN 은 $W_{hh}$ 가 identity → $W_{xh}$ 가 *유일한* nonlinearity source — 작아야 함

**결론**: IRNN 의 small $W_{xh}$ 가 핵심 — gradient flow 와 dead unit 의 trade-off. $\square$

</details>

**문제 3** (논문 비평): IRNN 이 LSTM 과 비등한 성능을 낸다면 LSTM 의 복잡성이 정당화되는가? 두 architecture 의 trade-off 와 use case 를 비교하라.

<details>
<summary>해설</summary>

**IRNN vs LSTM 의 trade-off**:

**IRNN 장점**:
1. **Simplicity**: vanilla RNN + 다른 init — 1/4 parameters
2. **Speed**: 4 gates vs 1 RNN cell — ~4x faster forward/backward
3. **Memory**: 1 hidden state vs hidden + cell
4. **Theoretical clarity**: residual connection 정신 명확

**LSTM 장점**:
1. **Robust to hyperparameters**: gating 이 자동 조정 — IRNN 의 small $W_{xh}$ 같은 fragility 없음
2. **Long-term selectivity**: forget gate 가 *selective* memory — IRNN 의 indiscriminate sum 보다 정교
3. **Stability**: bounded sigmoid + tanh gates — exploding 회피
4. **Empirical**: 대부분의 task 에서 약간 우월

**Use case 별 추천**:

1. **Lean architecture, fast prototyping** → IRNN
   - Resource-constrained edge AI
   - Simple sequence task (counting, sorting)

2. **Robust production model** → LSTM
   - Variable input domain
   - Long-form text generation
   - Speech recognition

3. **Long-range, sparse update** → LSTM 또는 Transformer
   - 의존성 길이 > 1000
   - Multi-modal sequence

4. **State Space Model** → S4, Mamba (Ch7-04)
   - 가장 긴 sequence (LRA, audio)

**실제 historical 변천**:
- 2015: IRNN 논문 출현, LSTM 대안 제시
- 2017-2018: LSTM 이 NMT 표준, IRNN 은 specialized use
- 2017+: Transformer 부상, RNN 전반 감소
- 2023: Mamba 등 SSM 부상, RNN-like 부활

**결론**: IRNN 은 *minimalist* 해법 — LSTM 의 complexity 가 항상 필요하지 않음을 증명. 그러나 LSTM 의 *robustness* 가 production 에서 가치. 둘 다 ResNet/Transformer 의 attention 우월 시대 이후 secondary 위치, 그러나 SSM 의 부활로 IRNN 정신 (linear recurrence + identity skip) 이 다시 주목. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-orthogonal-init.md) | [📚 README](../README.md) | [다음 ▶](../ch4-lstm/01-lstm-motivation.md)

</div>
