# 03. Cell State 와 Constant Error Carousel

## 🎯 핵심 질문

- **정리 (CEC)** $\partial c_t / \partial c_{t-1} = f_t$ — element-wise scalar, 잔여항 없음 — 의 정확한 증명?
- 왜 Jacobian 의 곱셈적 누적이 element-wise scalar 곱 $\prod f_t$ 로 단순화되는가?
- $f_t \approx 1$ 시 $\partial c_T / \partial c_0 \approx 1$ 가 vanishing 의 정확한 해법인 이유?
- Plain RNN 의 matrix $\prod W_{hh}^\top \mathrm{diag}(\sigma')$ 와 LSTM 의 element-wise $\prod f_t$ 의 차이가 long-range 학습에 어떤 의미인가?
- Hidden state path $\partial h_T / \partial h_0$ 가 여전히 vanishing 인 이유와 cell state path 와의 관계

---

## 🔍 왜 CEC 증명이 LSTM 의 핵심인가

LSTM 의 4 gate 수식 (Ch4-02) 만으로는 *왜* vanishing 이 해결되는지 알 수 없습니다. CEC 의 정확한 증명이:

1. **Vanishing 의 root cause 제거** 입증 — Matrix Jacobian 곱이 scalar element-wise 곱으로 단순화
2. **Forget gate 의 역할 정량화** — $f_t \in [0, 1]$ 가 per-dimension decay rate
3. **LSTM 의 long-range 능력의 정량적 bound** — $\prod f_t$ 의 lower bound 가 학습 가능 dependency 길이
4. **Modern variants 의 비교 baseline** — ResNet, Highway, Transformer 모두 비슷한 "곱이 아닌 합" 정신

이 문서는 CEC 정리를 한 단계씩 증명하고, hidden vs cell path 의 gradient flow 를 정확히 분석합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-lstm-equations.md](./02-lstm-equations.md) — LSTM 4 gate 수식
- [Ch3-01 Spectral 분석](../ch3-vanishing-exploding/01-spectral-analysis.md) — Plain RNN 의 vanishing
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Partial derivative, multi-variable chain rule
- 정의: Element-wise (Hadamard) product 의 Jacobian

---

## 📖 직관적 이해

### Plain RNN vs LSTM Cell 의 Gradient Path

**Plain RNN**:
```
h_t = tanh(W h_{t-1} + ...)
        │           ▲
        ▼           │
   ∂h_t/∂h_{t-1} = diag(σ') · W   ← matrix
   
   곱 = diag(σ'_1)·W · diag(σ'_2)·W · ... 
       = matrix product (eigenvalue 의 거듭제곱)
```

**LSTM Cell**:
```
c_t = f_t · c_{t-1} + (input contribution)
      │      ▲
      ▼      │
   ∂c_t/∂c_{t-1} = diag(f_t)   ← only diagonal
   
   곱 = diag(f_1) · diag(f_2) · ... = diag(∏ f_t)
       = element-wise scalar product
```

핵심: matrix product → element-wise scalar product.

### "잔여항 없음" 의 의미

$c_t = f_t c_{t-1} + i_t \tilde c_t$ 에서 $\partial / \partial c_{t-1}$:

- 첫 항: $\partial (f_t c_{t-1}) / \partial c_{t-1} = f_t$ (자명)
- 두 번째 항: $\partial (i_t \tilde c_t) / \partial c_{t-1}$
  - $i_t = \sigma(W_i [h_{t-1}; x_t] + b)$ — $h_{t-1}$ 의존, $c_{t-1}$ 직접 의존 *없음*
  - $\tilde c_t = \tanh(W_c [h_{t-1}; x_t] + b)$ — 마찬가지
  - 따라서 두 번째 항의 $c_{t-1}$ 에 대한 *direct* partial = 0

**결과**: $\partial c_t / \partial c_{t-1} = f_t$ (element-wise, **direct partial only**).

⚠️ Total derivative 는 다를 수 있음 — $h_{t-1}$ 통해 indirect 영향. 그러나 cell state path 위에서 *direct* 만 고려하면 element-wise scalar.

### 시각적 비교

$T = 100$ step 의 gradient norm:

```
              gradient magnitude
            
ρ=0.9 RNN   ●(1.0) → ── 0.9 → ── 0.81 → ... → ●(2.7e-5)  ← vanishing
            t=0      t=1      t=2          t=99

f=0.99 LSTM ●(1.0) → ── 0.99 → ── 0.98 → ... → ●(0.366)  ← 보존
            t=0      t=1       t=2          t=99
```

LSTM 이 약 $10^4$ 배 더 큰 gradient.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Direct Partial Derivative

$f(x_1, x_2, \ldots, x_n)$ 의 $x_i$ 에 대한 **direct** partial:

$$
\frac{\partial f}{\partial x_i}\bigg|_{\text{direct}} := \frac{\partial f}{\partial x_i}\bigg|_{x_j \text{ fixed for } j \ne i}
$$

다른 변수의 implicit dependency 무시.

### 정의 3.2 — LSTM Cell Recursion 의 Direct Form

$c_t = f_t c_{t-1} + i_t \tilde c_t$ 에서 $f_t, i_t, \tilde c_t$ 모두 $h_{t-1}$ 의 함수, $h_{t-1}$ 도 $c_{t-1}$ 의 함수 (transitively). 그러나 $c_{t-1}$ 에 *직접* 의존하는 것은 첫 항 $f_t c_{t-1}$ 뿐.

### 정의 3.3 — Cell-to-Cell Direct Jacobian

$$
\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{direct}} = f_t \in [0, 1]^H
$$

(Element-wise, diagonal matrix interpretation $\mathrm{diag}(f_t)$)

### 정의 3.4 — Cell Path Gradient

$T$ step 의 cell-only gradient:

$$
\frac{\partial c_T}{\partial c_0}\bigg|_{\text{cell path}} = \prod_{t=1}^{T} f_t = \mathrm{diag}\left(\bigodot_{t=1}^{T} f_t\right)
$$

여기서 $\bigodot$ 는 element-wise product across $t$.

### 정의 3.5 — Total Gradient

Indirect path (through $h_{t-1} \to f_t, i_t, \tilde c_t$) 포함:

$$
\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{total}} = f_t + (\text{indirect terms via } h_{t-1})
$$

---

## 🔬 정리와 증명

### 정리 3.1 — Constant Error Carousel (Direct Partial)

LSTM cell update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$ 에서, direct partial:

$$
\boxed{\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{direct}} = f_t}
$$

(Element-wise; matrix form: $\mathrm{diag}(f_t)$)

**증명**:

$f_t, i_t, \tilde c_t$ 가 $c_{t-1}$ 에 directly depend 하는지 검토:

1. $f_t = \sigma(W_f [h_{t-1}; x_t] + b_f)$:
   - $h_{t-1}, x_t$ 의 함수
   - $c_{t-1}$ direct 의존 없음
   - $\Rightarrow \partial f_t / \partial c_{t-1}|_{\text{direct}} = 0$

2. $i_t, \tilde c_t$: 마찬가지 — $h_{t-1}, x_t$ 만 직접 의존

3. $f_t \odot c_{t-1}$:
   - $c_{t-1}$ 에 *직접* 의존 (chain rule 적용 가능)
   - $\partial / \partial c_{t-1} = f_t$ (element-wise scalar 곱)

4. $i_t \odot \tilde c_t$:
   - $c_{t-1}$ 직접 의존 없음
   - $\partial / \partial c_{t-1}|_{\text{direct}} = 0$

따라서:

$$
\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{direct}} = f_t + 0 = f_t \quad \text{element-wise}
$$

$\square$

### 정리 3.2 — Cell Path Gradient Product

$T$ step 의 direct cell path:

$$
\boxed{\frac{\partial c_T}{\partial c_0}\bigg|_{\text{cell path}} = \prod_{t=1}^{T} f_t \quad \text{(element-wise)}}
$$

**증명**: Induction on $T$.

**Base** $T = 1$: $\partial c_1 / \partial c_0 = f_1$ (정리 3.1).

**Inductive step**: Assume $\partial c_{T-1} / \partial c_0 = \prod_{t=1}^{T-1} f_t$. 그러면:

$$
\frac{\partial c_T}{\partial c_0} = \frac{\partial c_T}{\partial c_{T-1}} \cdot \frac{\partial c_{T-1}}{\partial c_0} = f_T \cdot \prod_{t=1}^{T-1} f_t = \prod_{t=1}^{T} f_t
$$

(Element-wise multiplication 의 associativity)

$\square$

### 정리 3.3 — CEC 의 Lower Bound on Long-Range Gradient

$f_t \ge \alpha$ for all $t$ ($\alpha \in (0, 1]$):

$$
\left\|\frac{\partial c_T}{\partial c_0}\right\|_\infty \ge \alpha^T
$$

특히 $\alpha = 1$ 시 (CEC 정확) $\partial c_T / \partial c_0 = 1$.

**증명**: Element-wise product $\prod f_t \ge \alpha^T$ component-wise. $\square$

**의미**: 학습이 $f_t$ 를 1 에 가깝게 유지하면 임의로 긴 의존성 학습 가능 (gradient 보존).

### 정리 3.4 — Plain RNN vs LSTM Gradient Comparison

Plain RNN: $\|\partial h_T / \partial h_0\| \le (\rho \sigma')^T$ — exponential.
LSTM cell: $\|\partial c_T / \partial c_0\|_\infty \ge \alpha^T$ where $\alpha = \min_t \min_i f_{t,i}$.

**비교**:
- Plain RNN ($\rho = 1, \sigma' = 0.5$): $0.5^{100} \approx 8 \times 10^{-31}$
- LSTM ($\alpha = 0.95$): $0.95^{100} \approx 0.006$
- LSTM ($\alpha = 0.99$): $0.99^{100} \approx 0.366$

LSTM 이 $10^{27} \sim 10^{30}$ 배 큰 gradient.

### 정리 3.5 — Hidden State Path 의 Vanishing 잔존

Hidden state $h_t = o_t \odot \tanh(c_t)$. 

$$
\frac{\partial h_t}{\partial h_{t-1}}
$$

는 multi-path:
1. $h_{t-1} \to f_t, i_t, \tilde c_t \to c_t \to h_t$ (gates 통한 indirect)
2. $h_{t-1} \to o_t \to h_t$

각 path 가 matrix multiplication → spectral radius 의 곱 → vanishing 가능.

**그러나**: 정보의 *long-term* part 는 cell 통해 (정리 3.2), hidden 의 vanishing 영향 제한.

**증명** (sketch): Multi-path chain rule 의 모든 항 합산. 각 항이 $W_f, W_i, W_c, W_o$ 의 곱 → matrix product → spectral radius. $\square$

---

## 💻 NumPy / PyTorch 검증

### 실험 1 — CEC 정리의 직접 검증

```python
import torch
import torch.nn as nn
import numpy as np

class LSTMCellManual(nn.Module):
    """수식 직접 구현 — gradient flow 추적"""
    def __init__(self, D, H):
        super().__init__()
        self.D, self.H = D, H
        self.W_f = nn.Linear(D + H, H)
        self.W_i = nn.Linear(D + H, H)
        self.W_c = nn.Linear(D + H, H)
        self.W_o = nn.Linear(D + H, H)
        # Forget bias = 1
        with torch.no_grad():
            self.W_f.bias.fill_(1.0)
    
    def forward(self, x, h, c):
        xh = torch.cat([x, h], dim=-1)
        f = torch.sigmoid(self.W_f(xh))
        i = torch.sigmoid(self.W_i(xh))
        g = torch.tanh(self.W_c(xh))
        o = torch.sigmoid(self.W_o(xh))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new, (f, i, g, o)

torch.manual_seed(0)
D, H, T = 3, 10, 50
cell = LSTMCellManual(D, H)

# c_0 require_grad
c0 = torch.zeros(1, H, requires_grad=True)
h = torch.zeros(1, H)
c = c0
forget_history = []

x_seq = torch.randn(T, 1, D) * 0.1   # small input
for t in range(T):
    h, c, (f, i, g, o) = cell(x_seq[t], h, c)
    forget_history.append(f.detach())

# Loss = sum(c_T)
c.sum().backward()
print(f'∂(sum c_T) / ∂c_0:')
print(f'  Empirical (autograd): {c0.grad.norm():.6f}')

# Theoretical: ∏ f_t
prod_f = torch.ones(1, H)
for f in forget_history:
    prod_f *= f
theoretical = prod_f.norm().item()
print(f'  Theoretical (∏f_t):    {theoretical:.6f}')
print(f'  Match? {torch.allclose(c0.grad, prod_f.sum(0) if False else prod_f.squeeze(0), atol=1e-5)}')
# Direct partial derivative 정리 검증
```

### 실험 2 — Forget Gate 분포의 효과

```python
# 다양한 forget bias 로 ∂c_T / ∂c_0 비교
def measure_cec_gradient(forget_bias, T=50):
    torch.manual_seed(0)
    cell = LSTMCellManual(3, 10)
    with torch.no_grad():
        cell.W_f.bias.fill_(forget_bias)
    
    c0 = torch.zeros(1, 10, requires_grad=True)
    h = torch.zeros(1, 10); c = c0
    for t in range(T):
        x_t = torch.randn(1, 3) * 0.1
        h, c, _ = cell(x_t, h, c)
    
    c.sum().backward()
    return c0.grad.abs().mean().item()

print('Forget bias 별 ∂c_T / ∂c_0 (T=50):')
for b_f in [-2.0, 0.0, 1.0, 2.0]:
    grad = measure_cec_gradient(b_f, T=50)
    sigma_b = 1 / (1 + np.exp(-b_f))   # σ(b_f)
    print(f'  b_f = {b_f:+.1f}: σ(b_f) ≈ {sigma_b:.3f}, |∂c_T/∂c_0| avg = {grad:.4e}')
# b_f = 1 (default) 가 long-range gradient 보존 우수
```

### 실험 3 — Hidden Path vs Cell Path Gradient

```python
def measure_path_gradients(T=50):
    torch.manual_seed(0)
    cell = LSTMCellManual(3, 10)
    
    # Path 1: c_0 → c_T (only cell path)
    c0_isolated = torch.zeros(1, 10, requires_grad=True)
    h0_fixed = torch.zeros(1, 10)
    h, c = h0_fixed, c0_isolated
    for t in range(T):
        x_t = torch.randn(1, 3) * 0.1
        h, c, _ = cell(x_t, h, c)
    c.sum().backward()
    grad_cell_path = c0_isolated.grad.abs().mean().item()
    
    # Path 2: h_0 → h_T (only hidden path)
    cell.zero_grad()
    h0_isolated = torch.zeros(1, 10, requires_grad=True)
    c0_fixed = torch.zeros(1, 10)
    h, c = h0_isolated, c0_fixed
    torch.manual_seed(0)
    for t in range(T):
        x_t = torch.randn(1, 3) * 0.1
        h, c, _ = cell(x_t, h, c)
    h.sum().backward()
    grad_hidden_path = h0_isolated.grad.abs().mean().item()
    
    return grad_cell_path, grad_hidden_path

for T in [10, 30, 50, 100]:
    g_cell, g_hidden = measure_path_gradients(T)
    print(f'T={T:3d}: cell path = {g_cell:.4e}, hidden path = {g_hidden:.4e}')
# Cell path 가 hidden path 보다 훨씬 robust
```

### 실험 4 — Plain RNN vs LSTM Direct Comparison

```python
class PlainRNNCell(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.W = nn.Linear(D + H, H)
    def forward(self, x, h):
        xh = torch.cat([x, h], dim=-1)
        return torch.tanh(self.W(xh))

def measure_rnn_gradient(T):
    torch.manual_seed(0)
    cell = PlainRNNCell(3, 10)
    h0 = torch.zeros(1, 10, requires_grad=True)
    h = h0
    for t in range(T):
        x_t = torch.randn(1, 3) * 0.1
        h = cell(x_t, h)
    h.sum().backward()
    return h0.grad.abs().mean().item()

print('Plain RNN vs LSTM gradient at h_0:')
for T in [10, 30, 50, 100]:
    g_rnn = measure_rnn_gradient(T)
    g_cell, _ = measure_path_gradients(T)
    print(f'T={T:3d}: Plain RNN = {g_rnn:.4e},  LSTM cell = {g_cell:.4e},  ratio = {g_cell/g_rnn:.0f}x')
# LSTM cell path 가 plain RNN 보다 long-range 에서 훨씬 큰 gradient
```

### 실험 5 — Synthetic Adding Problem 결과

```python
# T=200 Adding Problem 에서 LSTM 의 long-range 학습 입증
def adding_problem_data(T, B):
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

class AddingModel(nn.Module):
    def __init__(self, cell_type='lstm', H=128):
        super().__init__()
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(2, H, batch_first=False)
        else:
            self.rnn = nn.RNN(2, H, nonlinearity='tanh', batch_first=False)
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[-1]).squeeze(-1)

def train_adding(model, T_seq, n_steps=300):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(n_steps):
        x, y = adding_problem_data(T_seq, 64)
        pred = model(x)
        loss = ((pred - y)**2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return np.mean(losses[-30:])

torch.manual_seed(0); np.random.seed(0)
T_seq = 100
print(f'Adding Problem T={T_seq}:')
for cell_type in ['rnn', 'lstm']:
    model = AddingModel(cell_type)
    final = train_adding(model, T_seq)
    print(f'  {cell_type.upper()}: final MSE = {final:.4f}')
# LSTM 이 RNN 보다 훨씬 낮은 MSE
```

---

## 🔗 실전 활용

### 1. PyTorch autograd 가 자동 처리

`loss.backward()` 가 CEC 와 hidden path 의 모든 contribution 정확히 계산. NumPy 구현은 *이해* 위한 것.

### 2. Forget bias 초기화

CEC 의 직접 결과: $b_f = 1$ → 초기 $f_t \approx 0.73$ → long-range gradient 보존. **표준 권장** (Ch4-04).

### 3. Long-range modeling 의 진단

Cell state norm 의 시간축 관찰 — exponential decay 면 forget gate 가 과도, increase 면 적절.

### 4. ResNet, Highway Network 의 정신

ResNet $h^{l+1} = h^l + f(h^l)$, Highway $h^{l+1} = T(h^l) \odot f(h^l) + (1 - T) \odot h^l$ — 같은 *additive update* 정신, vanishing 의 architectural 해결.

### 5. State Space Model (Mamba, Ch7-04)

$h_t = A h_{t-1} + B x_t$ — linear update, $A$ 가 element-wise diagonal 시 cell state 의 일반화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Direct partial only | Indirect path 도 학습 dynamics 영향 |
| $f_t \approx 1$ 유지 가능 | Saturation 시 binary $\{0, 1\}$, soft control 손실 |
| Element-wise scalar | Matrix mixing 부재 — multi-head attention 같은 cross-dim 부족 |
| Cell state path dominant | Hidden path 도 short-term 에 중요 |
| Random gradient | Task 별 specific direction 학습 |

---

## 📌 핵심 정리

$$\boxed{\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{direct}} = f_t \quad \text{— element-wise scalar}}$$

$$\boxed{\frac{\partial c_T}{\partial c_0}\bigg|_{\text{cell path}} = \prod_{t=1}^{T} f_t \quad \text{(no matrix product!)}}$$

$$\boxed{f_t \approx 1 \implies \text{constant error carousel} — \text{vanishing 정확히 해결}}$$

| Path | Gradient | Decay rate |
|------|----------|-----------|
| **Plain RNN $h$** | Matrix product $\prod J$ | Spectral radius $\rho^T$ |
| **LSTM cell $c$** | Element-wise $\prod f_t$ | $\alpha^T, \alpha \approx 1$ |
| **LSTM hidden $h$** | Multi-path matrix | Mixed |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 정리 3.1 의 증명에서 "$f_t, i_t, \tilde c_t$ 가 $c_{t-1}$ 에 direct 의존 없음" 이 핵심이다. Peephole connection (Gers 2002) 이 이를 어떻게 변경하는가?

<details>
<summary>해설</summary>

**Peephole connection**: gate 들이 cell state $c_{t-1}$ 도 본다:

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}; x_t; c_{t-1}] + b_f) \\
i_t &= \sigma(W_i [h_{t-1}; x_t; c_{t-1}] + b_i) \\
o_t &= \sigma(W_o [h_{t-1}; x_t; c_t] + b_o)
\end{aligned}
$$

(Output gate 은 *현재* cell state $c_t$ 를 본다)

**CEC 영향**:

$\partial c_t / \partial c_{t-1}$ 의 direct partial 이 더 이상 단순 $f_t$ 가 아님:

$$
\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{direct}} = f_t + \frac{\partial f_t}{\partial c_{t-1}} \cdot c_{t-1} + \frac{\partial i_t}{\partial c_{t-1}} \cdot \tilde c_t
$$

각 추가 항이:
- $\partial f_t / \partial c_{t-1} = \sigma'(...) \cdot W_f^{(c)}$ (weight 의 cell 부분)
- 이는 *matrix product* 형태 — vanishing 위험 부분 복귀

**Trade-off**:
- **장점**: Cell state 의 직접 정보로 더 정확한 gating (precision timing 같은 task)
- **단점**: CEC 의 element-wise simplicity 부분적 손실, vanishing 위험

**Greff 2017 의 ablation**:
- Peephole 의 marginal benefit (~1%)
- Vanilla LSTM 이 robust

**결론**: Peephole 이 CEC 의 element-wise scalar 형태를 살짝 깨뜨림. 표준 LSTM 의 단순한 $\partial c_t / \partial c_{t-1} = f_t$ 가 long-range gradient 의 핵심 — peephole 이 이를 약간 trade-off. $\square$

</details>

**문제 2** (심화): LSTM cell path 가 vanishing 을 해결하지만 hidden path 는 여전히 vanishing 한다. 그럼에도 LSTM 이 long-range task 에서 효과적인 이유를 정확히 설명하라.

<details>
<summary>해설</summary>

**두 path 의 역할**:

**Cell path ($c_0 \to c_T$)**:
- Long-term storage
- 정보 보존이 1차 목적
- Gradient flow 가 강 (CEC)

**Hidden path ($h_0 \to h_T$)**:
- Short-term context
- 매 step 의 prediction 에 사용
- Gradient flow 가 약 (vanishing)

**효과적인 이유**:

1. **분업 학습**:
   - 학습이 long-range info 를 cell 에 저장하도록 *유도*
   - Output gate $o_t$ 가 *현재 step 에 필요한* 정보만 hidden 으로 release
   - Hidden 은 매 step "renewed" — vanishing 이 *오히려* freshness 를 보장

2. **Hidden vanishing 의 영향 제한**:
   - Hidden 은 short-term context 만 표현
   - Vanishing 으로 100 step 전 hidden 가 영향 없는 것은 *desirable* (long-term 은 cell 통해)

3. **Information bottleneck 의 분리**:
   - Cell: long-term, capacity $H$ floats
   - Hidden: short-term, capacity $H$ floats
   - 둘 다 $H$ 차원이지만 *정보 공간 분리*

4. **Empirical evidence (Karpathy 2015)**:
   - Cell unit 이 specific concept 추적 (인용 깊이, 코드 vs 코멘트)
   - Hidden unit 은 보통 dense, 해석 어려움

**한계**:
- 매우 긴 sequence (>1000) 에서는 cell 도 어려움 (forget gate drift)
- Hidden vanishing 으로 mid-range (10-100) dependency 는 약할 수도
- → ResNet, Transformer 의 attention 이 추가 해법

**Modern view**:
- LSTM 의 cell path = short-cut for long-range
- Transformer 의 attention = direct connection for any range
- Mamba 의 selective SSM = adaptive cell path

**결론**: LSTM 의 *부분적* 해결이 충분히 강력 — cell 의 long-term + hidden 의 short-term 의 분업으로 most practical task (의존성 < 500) 에서 작동. 이 design pattern 이 modern architecture 의 inspiration. $\square$

</details>

**문제 3** (논문 비평): CEC 가 LSTM 의 핵심이라면, 왜 ResNet (CV) 과 Transformer (NLP) 가 LSTM 을 대체했는가? 세 architecture 의 "additive update" 정신이 어떻게 다른가?

<details>
<summary>해설</summary>

**세 architecture 의 additive update 비교**:

**LSTM**:
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t
$$
- Time 축 additive
- Element-wise $f_t$ 가 selective decay
- Gating 이 input/output control

**ResNet**:
$$
h^{(\ell+1)} = h^{(\ell)} + f(h^{(\ell)})
$$
- Depth (layer) 축 additive
- 단순 identity skip (gating 없음)
- Conv $f$ 가 residual learning

**Transformer**:
$$
h^{(\ell+1)} = h^{(\ell)} + \mathrm{Attention}(h^{(\ell)}) + \mathrm{FFN}(h^{(\ell)})
$$
- Layer 축 additive (residual)
- Attention 이 *cross-position* mixing — RNN 의 sequence path 를 attention 으로 대체
- 두 sub-block (attention, FFN) 각각 residual

**왜 LSTM 이 대체되었는가**:

1. **Sequence parallelism**:
   - LSTM: $c_t = f(c_{t-1})$ — sequential
   - Transformer: 모든 position 동시 attention — parallel
   - GPU utilization 이 결정적

2. **Long-range modeling**:
   - LSTM: cell path 통한 indirect — 의존성 길이 ~500
   - Transformer: direct attention — 의존성 길이 ~10000 (with proper PE)

3. **Information access**:
   - LSTM: $c_t$ 가 모든 history 를 single vector 로 압축
   - Transformer: 각 position 이 모든 다른 position 을 직접 access

4. **Empirical**:
   - 2017+ 모든 NLP benchmark 에서 Transformer 우월
   - LSTM 은 streaming / online setting 에서만 잔존

**ResNet 의 sustained relevance**:
- CV 에서는 (ViT 가 부상하지만) ResNet 표준
- Image 의 *spatial locality* 가 conv + residual 에 적합
- Recently: ConvNeXt 등 conv-based 가 다시 경쟁력

**Modern resurgence (Mamba, etc.)**:
- Linear RNN (cell path 만, hidden 없음) 의 부활
- Transformer 의 $O(T^2)$ 비용 회피
- LSTM 의 *core insight* (selective additive update) 의 minimal form

**결론**: LSTM 의 CEC 가 *additive update* 의 inspiration 이지만, sequence parallelism 의 한계로 Transformer 에 자리. ResNet 은 비슷한 정신을 depth 축에 적용 — 다른 차원의 같은 idea. Modern Mamba 가 LSTM 의 정수 (linear recurrence) 를 parallel scan 으로 부활. **CEC 의 idea 가 architecture history 의 backbone**. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-lstm-equations.md) | [📚 README](../README.md) | [다음 ▶](./04-gradient-flow.md)

</div>
