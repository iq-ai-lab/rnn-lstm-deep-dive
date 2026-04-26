# 04. LSTM 의 Gradient Flow 분석

## 🎯 핵심 질문

- LSTM 의 전체 gradient 경로 — cell state $c$ 통한 long-range vs hidden state $h$ 통한 short-range — 가 어떻게 통합되는가?
- **Jozefowicz 2015** 의 핵심 발견: forget bias $b_f = 1$ 초기화의 정확한 효과는?
- $\sigma(W_f \cdot [h, x] + 1) \approx 0.73$ 이 왜 cell state 의 초기 보존을 보장하는가?
- Adding Problem $T = 200$ 에서 $b_f = 0$ vs $b_f = 1$ 학습 곡선의 차이를 정량화
- PyTorch `nn.LSTM` 의 default $b_f = 0$ 와 명시적 $b_f = 1$ 적용법

---

## 🔍 왜 forget bias 초기화가 LSTM 학습의 결정적 요소인가

CEC (Ch4-03) 의 정리는 *수학적* 으로 LSTM 의 gradient 보존을 보장합니다. 그러나 **학습이 시작될 때** 가 critical:

- $b_f = 0$ init: $f_t \approx \sigma(0) = 0.5$ → 매 step gradient 50% 감쇠 → 초기에는 plain RNN 과 비슷
- $b_f = 1$ init: $f_t \approx \sigma(1) = 0.73$ → 매 step 73% 보존 → 학습 초기부터 long-range gradient 흐름

Jozefowicz et al. 2015 의 *An Empirical Exploration of Recurrent Network Architectures* 가 이 단순한 변경이 학습 성공의 핵심임을 입증:
- $T = 100$ Adding Problem: $b_f = 0$ 시 학습 5x 느림
- 일부 task 에서는 $b_f = 0$ 시 학습 *불가*

이 문서는 forget bias 의 정확한 효과를 분석하고, LSTM 의 전체 gradient flow 를 추적합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [03-cec-proof.md](./03-cec-proof.md) — CEC 정리, $\partial c_t / \partial c_{t-1} = f_t$
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Sigmoid bias 의 effect, learning dynamics
- 정의: $\sigma(z + b) = \sigma(z) \cdot \sigma(b) / \sigma(0)$ 같은 transformation properties

---

## 📖 직관적 이해

### Forget Bias 의 직관

Sigmoid 의 input shift:

```
σ(z + b)

b = 0:  z = 0 → σ = 0.5  (no preference)
b = 1:  z = 0 → σ = 0.73 (prefer "remember")
b = 2:  z = 0 → σ = 0.88 (strongly prefer "remember")
```

Random init weights 시 $W_f \cdot [h, x] \approx 0$ (mean) → bias 가 default behavior 결정.

### 학습 초기의 Cell State Dynamics

**$b_f = 0$ 초기**:
- $f_t \approx 0.5$ → $c_t \approx 0.5 c_{t-1} + 0.5 \tilde c_t$
- 매 step cell state 가 절반 잊고 절반 update
- $T$ step 후: $\partial c_T / \partial c_0 \approx 0.5^T$ (e.g., $T = 100$ → $10^{-30}$)
- **Plain RNN 과 비슷한 vanishing**

**$b_f = 1$ 초기**:
- $f_t \approx 0.73$ → $c_t \approx 0.73 c_{t-1} + 0.27 \tilde c_t$
- $T = 100$: $0.73^{100} \approx 10^{-13}$ — 여전히 작지만 1000x 더 큼
- $b_f = 2$: $0.88^{100} \approx 3 \times 10^{-6}$
- **Long-range gradient 가 측정 가능 수준**

### 학습 후에는 어떻게?

학습이 진행되면 $W_f$ 가 update 되고 $f_t$ 가 task 에 맞춰 분화:
- 보존해야 할 info: $f_t \to 1$
- Forget 해야 할 info: $f_t \to 0$

그러나 *학습 시작* 시점에 gradient 가 0 에 가까우면 학습 자체가 stuck — **cold start problem**.

$b_f = 1$ 가 cold start 회피 — 처음부터 long-range signal 이 흐름.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Forget Bias Initialization

LSTM 의 forget gate bias $b_f$ 의 초기화:

- **Default (PyTorch)**: $b_f = 0$ → $\sigma(0) = 0.5$
- **Jozefowicz recommendation**: $b_f = 1$ → $\sigma(1) \approx 0.7311$
- **Aggressive**: $b_f = 2 \sim 5$ — 매우 long-range task

### 정의 4.2 — Sigmoid Bias 의 Closed-form Effect

$$
\sigma(z + b) = \frac{1}{1 + e^{-z-b}}
$$

$z = 0$ 시:

$$
\sigma(b) = \frac{e^b}{1 + e^b}
$$

$b = 1: 0.7311$
$b = 2: 0.8808$
$b = 3: 0.9526$
$b = 5: 0.9933$

### 정의 4.3 — Initial Cell State Decay Rate

학습 초기 ($W_f \cdot [h, x] \approx 0$):

$$
\alpha_{\text{init}} = \sigma(b_f)
$$

$T$ step 후:

$$
\frac{\partial c_T}{\partial c_0} \approx \alpha_{\text{init}}^T
$$

### 정의 4.4 — Cold Start Problem

Gradient 가 너무 작아 ($\alpha^T < \epsilon_{\text{machine}}$) 학습이 시작되지 못하는 현상.

$\epsilon_{\text{machine}} \approx 10^{-8}$ for float32. $T = 100$ 시:
- $\alpha = 0.5$: $10^{-30} \ll \epsilon_{\text{machine}}$ — cold start
- $\alpha = 0.73$: $10^{-13}$ — 여전히 작지만 측정 가능
- $\alpha = 0.95$: $10^{-2.2}$ — 안정 학습

### 정의 4.5 — Effective Memory Length

Cell state 가 정보 보존하는 평균 step 수:

$$
\tau_{\text{eff}} = \frac{1}{1 - \mathbb E[f_t]}
$$

- $\mathbb E[f_t] = 0.5$: $\tau_{\text{eff}} = 2$ — 매우 짧음
- $\mathbb E[f_t] = 0.73$: $\tau_{\text{eff}} \approx 3.7$
- $\mathbb E[f_t] = 0.95$: $\tau_{\text{eff}} = 20$
- $\mathbb E[f_t] = 0.99$: $\tau_{\text{eff}} = 100$

---

## 🔬 정리와 결과

### 정리 4.1 — Forget Bias 1 의 Gradient Boost

$b_f = 0$ vs $b_f = 1$ 의 long-range gradient 비율:

$$
\frac{(\partial c_T / \partial c_0)_{b_f=1}}{(\partial c_T / \partial c_0)_{b_f=0}} = \left(\frac{0.7311}{0.5}\right)^T = 1.4622^T
$$

$T = 100$: $1.4622^{100} \approx 7 \times 10^{16}$ — $10^{16}$ 배.

**의미**: $b_f = 1$ 이 학습 초기에 long-range gradient 를 $10^{16}$ 배 amplify — cold start 회피.

### 정리 4.2 — Adding Problem 의 학습 시간

$T$ step Adding Problem 에서 $b_f = 1$ 의 학습 시간 (epoch to convergence):

| $T$ | $b_f = 0$ | $b_f = 1$ | $b_f = 2$ |
|-----|----------|-----------|-----------|
| 50  | ~50 ep   | ~10 ep    | ~10 ep    |
| 100 | ~200 ep  | ~30 ep    | ~20 ep    |
| 200 | ❌ 학습 X | ~80 ep    | ~50 ep    |
| 500 | ❌        | ❌ (어려움) | ~150 ep   |

(Empirical, Jozefowicz 2015 + 후속 실험)

### 정리 4.3 — Trained vs Initial Forget Distribution

학습 진행 시 $f_t$ 의 분포:
- 초기: $f_t \approx \sigma(b_f)$ uniform
- 후기: bimodal — task 의 *retention* (높은 $f$) vs *forget* (낮은 $f$)

학습이 task 에 맞춰 *각 dimension 마다 다른 forget rate* 학습.

### 정리 4.4 — Multi-step Gradient through Forget Gate

$T$ step gradient norm 의 expected value (random $f_t$ 가정):

$$
\mathbb E\left[\prod_{t=1}^{T} f_t\right] = \mathbb E[f_1]^T \quad \text{(if } f_t \text{ iid)}
$$

(실제로는 $f_t$ 가 sequential 의존, 그러나 좋은 approximation)

### 정리 4.5 — Forget Bias 와 Information Capacity

Cell 의 information capacity $H$ floats. $f_t \to 1$ 유지 시 capacity 가 *모든* past info 로 차야 함 → finite capacity 의 한계로 *eventually* forget 필요.

**Trade-off**: 짧은 의존성 task 는 $b_f$ 작아도 OK, 매우 긴 의존성 (>1000) 은 $b_f = 5+$ 필요.

---

## 💻 PyTorch 구현 검증

### 실험 1 — PyTorch nn.LSTM 의 Forget Bias 설정

```python
import torch
import torch.nn as nn

D, H = 10, 20
lstm = nn.LSTM(D, H, batch_first=False)

# PyTorch 의 LSTM bias 구조:
# weight_ih_l0.bias: 4H = [b_i, b_f, b_g, b_o]   (in this order)
# Default: 모두 0

print('Default biases:')
print(f'  bias_ih_l0[H:2H] (forget): {lstm.bias_ih_l0[H:2*H].mean():.4f}')

# Forget bias = 1 설정
def set_forget_bias_to_1(lstm_module, hidden_size):
    """LSTM 의 forget bias 를 1로 설정"""
    with torch.no_grad():
        for layer_idx in range(lstm_module.num_layers):
            bias_ih = getattr(lstm_module, f'bias_ih_l{layer_idx}')
            bias_hh = getattr(lstm_module, f'bias_hh_l{layer_idx}')
            # PyTorch order: i, f, g, o
            bias_ih[hidden_size:2*hidden_size].fill_(1.0)
            bias_hh[hidden_size:2*hidden_size].fill_(0.0)   # b_ih + b_hh sum = 1

set_forget_bias_to_1(lstm, H)
print('After setting forget bias = 1:')
print(f'  bias_ih_l0[H:2H]: {lstm.bias_ih_l0[H:2*H].mean():.4f}')
print(f'  bias_hh_l0[H:2H]: {lstm.bias_hh_l0[H:2*H].mean():.4f}')
print(f'  Effective forget bias: {(lstm.bias_ih_l0[H:2*H] + lstm.bias_hh_l0[H:2*H]).mean():.4f}')
```

### 실험 2 — Initial Forget Gate 분포 측정

```python
import numpy as np

def measure_initial_forget(b_f_value, n_samples=100):
    torch.manual_seed(0)
    lstm = nn.LSTM(D, H, batch_first=False)
    set_forget_bias_to_1(lstm, H) if b_f_value == 1.0 else None
    if b_f_value == 0.0:
        with torch.no_grad():
            lstm.bias_ih_l0[H:2*H].zero_()
            lstm.bias_hh_l0[H:2*H].zero_()
    
    # Random input 으로 forget gate 측정
    f_values = []
    for _ in range(n_samples):
        x = torch.randn(20, 1, D) * 0.5
        h, c = torch.zeros(1, 1, H), torch.zeros(1, 1, H)
        for t in range(20):
            # Manual extraction (PyTorch internal 직접 접근 어려움 — manual cell 사용)
            xh_in = lstm.bias_ih_l0 + x[t, 0] @ lstm.weight_ih_l0.T
            xh_h = lstm.bias_hh_l0 + h[0, 0] @ lstm.weight_hh_l0.T
            z = xh_in + xh_h
            i = torch.sigmoid(z[0:H])
            f = torch.sigmoid(z[H:2*H])
            g = torch.tanh(z[2*H:3*H])
            o = torch.sigmoid(z[3*H:4*H])
            c = (f * c.squeeze() + i * g).unsqueeze(0).unsqueeze(0)
            h = (o * torch.tanh(c.squeeze())).unsqueeze(0).unsqueeze(0)
            f_values.append(f.detach().numpy())
    
    return np.concatenate(f_values)

# Compare distributions
f_default = measure_initial_forget(0.0)
f_jozefowicz = measure_initial_forget(1.0)

print(f'b_f = 0:  mean f = {f_default.mean():.4f}, std = {f_default.std():.4f}')
print(f'b_f = 1:  mean f = {f_jozefowicz.mean():.4f}, std = {f_jozefowicz.std():.4f}')
# b_f = 1 이 평균 0.73 근처
```

### 실험 3 — Adding Problem Long-range 학습 비교

```python
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

class AddingLSTM(nn.Module):
    def __init__(self, H=128, b_f_init=0.0):
        super().__init__()
        self.lstm = nn.LSTM(2, H)
        self.fc = nn.Linear(H, 1)
        self.H = H
        if b_f_init != 0:
            with torch.no_grad():
                self.lstm.bias_ih_l0[H:2*H].fill_(b_f_init)
                self.lstm.bias_hh_l0[H:2*H].fill_(0.0)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[-1]).squeeze(-1)

def train_adding_track(b_f_init, T_seq, n_steps=100, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AddingLSTM(H=128, b_f_init=b_f_init)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(n_steps):
        x, y = adding_problem(T_seq, 64)
        pred = model(x)
        loss = ((pred - y)**2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses

T = 100
print(f'Adding Problem T={T}, 100 steps:')
losses_0 = train_adding_track(0.0, T)
losses_1 = train_adding_track(1.0, T)
losses_2 = train_adding_track(2.0, T)

print(f'  b_f=0 final loss: {np.mean(losses_0[-10:]):.4f}')
print(f'  b_f=1 final loss: {np.mean(losses_1[-10:]):.4f}')
print(f'  b_f=2 final loss: {np.mean(losses_2[-10:]):.4f}')
# b_f = 1, 2 가 훨씬 빠른 수렴
```

### 실험 4 — 학습 진행에 따른 Forget Gate 분포 변화

```python
def track_forget_distribution(model, x, t):
    """학습 중인 model 의 forget gate 분포"""
    model.eval()
    with torch.no_grad():
        # Manual forward to extract f_t
        h, c = torch.zeros(1, 1, model.H), torch.zeros(1, 1, model.H)
        b_ih = model.lstm.bias_ih_l0
        b_hh = model.lstm.bias_hh_l0
        for tau in range(t):
            z = b_ih + x[tau, 0:1] @ model.lstm.weight_ih_l0.T \
                + b_hh + h[0] @ model.lstm.weight_hh_l0.T
            i = torch.sigmoid(z[:, 0:model.H])
            f = torch.sigmoid(z[:, model.H:2*model.H])
            g = torch.tanh(z[:, 2*model.H:3*model.H])
            o = torch.sigmoid(z[:, 3*model.H:])
            c = (f * c.squeeze(0) + i * g).unsqueeze(0)
            h = (o * torch.tanh(c.squeeze(0))).unsqueeze(0)
        return f.numpy().flatten()

# 학습 단계별 forget gate 분포 추적
torch.manual_seed(0); np.random.seed(0)
model = AddingLSTM(H=128, b_f_init=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

x_test, _ = adding_problem(50, 1)
print('Forget gate distribution evolution:')

dist_0 = track_forget_distribution(model, x_test, 25)
print(f'Step 0:    f mean={dist_0.mean():.3f}, near 1: {(dist_0>0.9).mean()*100:.0f}%')

for step in range(100):
    x, y = adding_problem(50, 64)
    loss = ((model(x) - y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

dist_100 = track_forget_distribution(model, x_test, 25)
print(f'Step 100:  f mean={dist_100.mean():.3f}, near 1: {(dist_100>0.9).mean()*100:.0f}%, near 0: {(dist_100<0.1).mean()*100:.0f}%')
# 학습이 진행되면 bimodal 분포 (선택적 forget)
```

### 실험 5 — Long-range Gradient 직접 측정

```python
def measure_long_range_gradient(model, T):
    """∂h_T / ∂c_0 의 norm 측정"""
    H = model.H
    # h_0, c_0 require_grad
    h_0 = torch.zeros(1, 1, H, requires_grad=True)
    c_0 = torch.zeros(1, 1, H, requires_grad=True)
    
    x = torch.randn(T, 1, 2)
    out, (h_T, c_T) = model.lstm(x, (h_0, c_0))
    
    # Loss = ||c_T||^2
    loss = c_T.pow(2).sum()
    loss.backward()
    
    return c_0.grad.abs().mean().item()

torch.manual_seed(0)
for b_f in [0.0, 1.0, 2.0, 5.0]:
    model = AddingLSTM(H=128, b_f_init=b_f)
    g_t = measure_long_range_gradient(model, T=100)
    print(f'b_f = {b_f:.1f}: ||∂c_100 / ∂c_0|| avg = {g_t:.4e}')
# b_f 클수록 long-range gradient 큼
```

---

## 🔗 실전 활용

### 1. 모든 LSTM 학습의 표준 init

```python
# Custom LSTM init
def init_lstm_forget_bias(lstm_module, value=1.0):
    for layer_idx in range(lstm_module.num_layers):
        bias_ih = getattr(lstm_module, f'bias_ih_l{layer_idx}')
        bias_hh = getattr(lstm_module, f'bias_hh_l{layer_idx}')
        H = bias_ih.size(0) // 4
        with torch.no_grad():
            bias_ih[H:2*H].fill_(value)
            bias_hh[H:2*H].fill_(0.0)

lstm = nn.LSTM(input_size, hidden_size)
init_lstm_forget_bias(lstm, 1.0)
```

### 2. Long-range task 의 specialized init

매우 긴 의존성 (음악, long document) 시 $b_f = 2 \sim 5$ — initial cell state retention 강화.

### 3. Continual learning 의 forget rate

새 task 학습 시 catastrophic forgetting 회피 — $b_f$ 를 task 별 조정.

### 4. Empirical default

PyTorch 의 default 가 0 이지만, modern best practice 는 명시적 1 설정. fastai, transformers 라이브러리 는 종종 자동 설정.

### 5. Architecture-specific tuning

ConvLSTM, BiLSTM, Stacked LSTM 모두 동일 원리 — 각 layer 의 forget bias = 1.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Random init weights $\to z \approx 0$ | 학습 후 $z$ 가 변동, $b_f$ 영향 감소 |
| Single layer LSTM | Stacked LSTM 의 layer 별 dynamics |
| Forward 학습 시작에 critical | Pre-trained model 에는 영향 적음 |
| Task-agnostic 1.0 | Task 별 최적값 다름 |
| Static $b_f$ | Adaptive forget bias (learnable per-step) variant 가능 |

---

## 📌 핵심 정리

$$\boxed{b_f = 1 \implies \sigma(b_f) \approx 0.7311, \quad \alpha_{\text{init}} \approx 0.73}$$

$$\boxed{T \text{ step} \text{ initial gradient}: \alpha^T \quad b_f=1 \text{ 이 } b_f=0 \text{ 의 } 1.46^T \text{ 배}}$$

$$\boxed{\text{Cold start 회피: } \alpha^T > 10^{-8} \implies T < \log(10^{-8}) / \log \alpha}$$

| $b_f$ | Init $\sigma(b_f)$ | $T = 100$ gradient | Use case |
|-------|---------------------|---------------------|----------|
| **-2** | 0.119 | $10^{-92}$ | 비추천 |
| **0** | 0.500 | $10^{-30}$ | PyTorch default — cold start |
| **1** | 0.731 | $10^{-13}$ | Jozefowicz 권장 — 표준 |
| **2** | 0.881 | $10^{-5.5}$ | Long-range task |
| **5** | 0.993 | $0.5$ | 매우 long-range |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $b_f = 1$ 시 sequence 길이 $T = 50$ 의 initial gradient norm 비율 (vs $b_f = 0$) 을 계산하라.

<details>
<summary>해설</summary>

**Forget gate values**:
- $b_f = 0$: $\sigma(0) = 0.5$
- $b_f = 1$: $\sigma(1) = 0.7311$

**$T = 50$ 후 gradient**:
- $b_f = 0$: $0.5^{50} \approx 8.88 \times 10^{-16}$
- $b_f = 1$: $0.7311^{50} \approx 7.4 \times 10^{-7}$

**비율**:
$$
\frac{0.7311^{50}}{0.5^{50}} = \left(\frac{0.7311}{0.5}\right)^{50} = 1.4622^{50} \approx 8.3 \times 10^8
$$

— 약 **$10^9$ 배** 더 큰 gradient.

**의미**: $b_f = 0$ 시 $10^{-16}$ — float32 precision 한계 ($10^{-8}$) 이하 → **gradient 가 사실상 0**, 학습 stuck.
$b_f = 1$ 시 $10^{-7}$ — 측정 가능, 학습 가능.

이것이 Jozefowicz 의 단순한 변경이 학습 가능성의 *binary* difference 를 만드는 이유. $\square$

</details>

**문제 2** (심화): 학습이 진행되어 $W_f$ 가 update 되면 $f_t$ 가 더 이상 $\sigma(b_f)$ 가 아니다. $b_f$ 의 영향이 학습 후에도 남는가?

<details>
<summary>해설</summary>

**학습 진행 시 forget gate 의 변화**:

$f_t = \sigma(W_f [h_{t-1}; x_t] + b_f)$

학습이 $W_f$ 와 $b_f$ 둘 다 update.

**Linear approximation 영역**:
- 학습 초기: $W_f \cdot [h, x] \approx 0$ → $f_t \approx \sigma(b_f)$
- 학습 후기: $W_f \cdot [h, x]$ 가 task-specific signal — input-dependent gating

**$b_f$ 의 sustained influence**:

1. **초기 학습 trajectory**:
   - $b_f = 1$ 가 long-range gradient 를 학습 가능한 수준으로 유지
   - 한 번 학습 시작되면 $W_f$ 가 task 에 맞춰 update
   - 만약 $b_f = 0$ 이면 학습 *시작* 자체가 stuck

2. **수렴된 모델의 $b_f$**:
   - $b_f$ 도 학습되므로 task 에 맞춰 변화
   - $W_f$ 가 강한 signal 학습하면 $b_f$ 의 marginal 영향 감소
   - 그러나 $b_f$ 가 default behavior 의 baseline 역할

3. **Implicit regularization**:
   - $b_f = 1$ init 이 "forget less" prior — 학습이 이 prior 에서 출발
   - 학습이 $b_f$ 를 줄이려면 강한 evidence 필요 → "remember more" bias 유지

**실험적 증거**:
- 학습 후 $b_f$ 측정: 0.5 → -1 (감소) 또는 1 → 0.5 (감소) 의 trend
- 그러나 *initial* gradient flow 가 critical — 학습 trajectory 결정

**결론**: $b_f = 1$ 은 학습 시작의 *catalyst* — 한 번 학습이 진행되면 marginal effect 감소, 그러나 trajectory 가 영구적으로 다름. **Cold start problem 해결이 핵심 contribution**. $\square$

</details>

**문제 3** (논문 비평): Jozefowicz 2015 가 forget bias = 1 을 발견한 후 PyTorch 의 default 가 여전히 0 인 이유는? Best practice 가 빠르게 표준화되지 않는 이유는?

<details>
<summary>해설</summary>

**역사적 맥락**:

- 1997: LSTM 도입, forget gate 없음 (original)
- 2000: Forget gate 추가 (Gers)
- 2014: Cho 의 GRU
- **2015: Jozefowicz et al. 의 ablation study — forget bias = 1 의 효과 입증**
- 2016+: 일부 framework 가 default 변경, 일부는 그대로

**왜 PyTorch 가 default 0**:

1. **Backward compatibility**:
   - 기존 model 의 reproducibility
   - Default 변경 시 model behavior 미세 변화

2. **Library 일관성**:
   - cuDNN 의 LSTM 이 zero bias default
   - PyTorch 가 cuDNN wrap → default 일관성

3. **사용자 명시성**:
   - "Magical default" 보다 사용자가 의식적으로 선택
   - Documentation 에 권장 init 명시

4. **Empirical 변동성**:
   - $b_f = 1$ 이 *일반적* 으로 좋지만 *모든* task 에서 그렇지는 않음
   - Some specialized tasks 에서 $b_f = 0$ 이 더 좋을 수도

**Best practice 의 분산**:

- **fastai**: $b_f = 1$ default
- **HuggingFace transformers**: 자동 설정
- **PyTorch nn.LSTM**: $b_f = 0$ (사용자 manual 설정 필요)
- **Keras / TensorFlow**: 일부 model class 에서 자동 1

**Modern view**:

- LSTM 자체가 Transformer 에 밀려 NLP 에서 secondary
- 새 architecture (Mamba 등) 에서 이런 init detail 다시 문제
- 일반화: **architectural detail 의 best practice 가 항상 default 되는 것은 아님** — historical, software engineering, conservatism 의 영향

**Lesson**:

1. **Paper 권장이 default 가 되기까지 시간**: Jozefowicz 2015 → 일부 framework 채택 2017+
2. **사용자 책임**: Best practice 를 manually 적용하는 습관 필요
3. **Documentation**: PyTorch 의 nn.LSTM doc 도 forget bias = 1 권장 명시 (recent updates)

**결론**: Jozefowicz 발견은 강력하지만 default 변경은 sluggish. **사용자가 의식적으로 적용** 하는 것이 modern ML engineering 의 책임 — best practice 가 자동화될 때까지. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-cec-proof.md) | [📚 README](../README.md) | [다음 ▶](./05-gru.md)

</div>
