# 02. BPTT 의 완전 유도

## 🎯 핵심 질문

- $\partial L / \partial W_{hh}$ 가 왜 $\sum_t \sum_{k \le t} \big(\prod_{j=k+1}^t \partial h_j / \partial h_{j-1}\big) \cdot \partial h_k / \partial W_{hh}$ 형태로 분해되는가?
- Jacobian $\partial h_j / \partial h_{j-1} = W_{hh}^\top \mathrm{diag}(\sigma'(z_j))$ 가 어떻게 vanishing/exploding 의 핵심 항이 되는가?
- BPTT 의 backward sweep 에서 **delta** $\delta_t = \partial L / \partial h_t$ 가 어떻게 시간 역순으로 전파되는가?
- Forward 와 backward 의 정확한 대칭성은 무엇이며 메모리 보존 요구는?
- 4-step toy RNN 의 BPTT 를 손으로 유도하고 PyTorch autograd 와 일치 검증

---

## 🔍 왜 BPTT 의 정확한 유도가 필수인가

BPTT 를 "이름으로 안다" 는 것과 "Jacobian 곱의 누적이 vanishing/exploding 의 spectral 분석 (Ch3-01) 의 출발점이 됨" 을 안다는 것은 다릅니다. BPTT 의 정확한 형태가 다음 모든 결과의 기반:

1. **Pascanu 2013 의 spectral 분석** (Ch3-01) — $\prod \partial h_j / \partial h_{j-1}$ 의 spectral radius 분석
2. **LSTM 의 CEC** (Ch4-03) — Cell state 의 $\partial c_t / \partial c_{t-1} = f_t$ 가 BPTT 의 곱셈적 누적을 단순화
3. **Truncated BPTT** (Ch2-03) — 마지막 $k$ step 만 합산하는 근사
4. **RTRL** (Ch2-05) — 같은 gradient 를 forward-mode AD 로 계산

이 문서는 BPTT 의 모든 항을 한 단계씩 유도하고 NumPy 로 PyTorch autograd 와 일치 검증합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-unrolled-graph.md](./01-unrolled-graph.md) — Unrolling, shared weight gradient 합산
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Chain rule, multi-variable Jacobian
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix-vector product, transpose, $\mathrm{diag}$ 연산
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Backpropagation, computational graph

---

## 📖 직관적 이해

### BPTT 는 "시간축으로 펼친 backprop"

Standard MLP backprop:

```
x → h^{(1)} → h^{(2)} → ... → h^{(L)} → y → L
                                         ↑
                  ← δ^{(L)} ← δ^{(L-1)} ← ...
```

BPTT 는 같은 구조이지만 **layer 가 time step 으로** 대체:

```
x_1, x_2, x_3 →   h_1   →   h_2   →   h_3   → y_3 → L
                                                ↑
                  ← δ_1 ← δ_2 ← δ_3 ←
```

차이: weight 가 모든 step 에서 동일 ($W_{hh}, W_{xh}$ 공유) → gradient **합산**.

### Loss 의 분해

매 step 에서 loss 가 발생하면:

$$
L = \sum_{t=1}^{T} L_t
$$

선형성으로:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}
$$

각 $\partial L_t / \partial W_{hh}$ 가 **시간을 거슬러 가는 gradient path** 로 다시 분해됨.

### Path 의 시각화

$L_3$ 가 $W_{hh}$ 에 어떻게 의존?

$$
L_3 \leftarrow y_3 \leftarrow h_3 \leftarrow \begin{cases} h_2 \leftarrow \begin{cases} h_1 \leftarrow W_{hh} \\ W_{hh} \end{cases} \\ W_{hh} \end{cases}
$$

3개의 path: $W_{hh} \to h_1 \to h_2 \to h_3 \to L_3$, $W_{hh} \to h_2 \to h_3 \to L_3$, $W_{hh} \to h_3 \to L_3$.

각 path 가 chain rule 로 합쳐져 Jacobian 곱이 됨.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Forward Recurrence

$$
z_t = W_{hh} h_{t-1} + W_{xh} x_t + b_h, \qquad h_t = \tanh(z_t), \qquad y_t = W_{hy} h_t + b_y
$$

### 정의 2.2 — Loss

각 step 의 loss $L_t$ (예: $L_t = \|y_t - y_t^*\|^2 / 2$ 또는 cross-entropy):

$$
L = \sum_{t=1}^{T} L_t
$$

### 정의 2.3 — Hidden Jacobian

$$
J_t := \frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}(\tanh'(z_t)) \, W_{hh} \in \mathbb R^{H \times H}
$$

또는 transposed for left-multiplication: $J_t^\top = W_{hh}^\top \mathrm{diag}(\tanh'(z_t))$.

### 정의 2.4 — Delta (Backward Signal)

$$
\delta_t := \frac{\partial L}{\partial h_t}
$$

이는 **모든 future loss term** 의 누적 gradient.

---

## 🔬 정리와 증명

### 정리 2.1 — BPTT 의 Recursive Form

$$
\boxed{\delta_t = \frac{\partial L_t}{\partial h_t} + J_{t+1}^\top \, \delta_{t+1}}
$$

(Boundary: $\delta_T = \partial L_T / \partial h_T$, $\delta_{T+1} = 0$)

**증명**: $h_t$ 가 영향을 주는 곳: (1) 즉시 $L_t$, (2) 다음 step $h_{t+1}$ 을 통해 모든 미래 $L_{t+1}, \ldots, L_T$. Chain rule:

$$
\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \frac{\partial L}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_t}
$$

두 번째 항: $\delta_{t+1} \cdot J_{t+1}$ — vector-Jacobian product. Transpose 표기로 $J_{t+1}^\top \delta_{t+1}$. $\square$

### 정리 2.2 — $\partial L / \partial W_{hh}$ 의 명시적 형태

$$
\boxed{\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} (\delta_t \odot \tanh'(z_t)) \, h_{t-1}^\top}
$$

**증명**: $z_t = W_{hh} h_{t-1} + \ldots$ 이므로 $\partial z_t / \partial W_{hh}$ 는 outer product 형태. Chain rule:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_t \frac{\partial L}{\partial z_t} \cdot \frac{\partial z_t}{\partial W_{hh}}
$$

$\partial L / \partial z_t = \delta_t \odot \tanh'(z_t)$ (element-wise), $\partial z_t / \partial W_{hh}$ 는 $h_{t-1}^\top$ outer product. 결과는 outer product 합. $\square$

### 정리 2.3 — Jacobian 곱 누적의 명시적 형태 (Pascanu 2013)

$$
\boxed{\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^{t} \left(\prod_{j=k+1}^{t} J_j^\top\right) \frac{\partial L_t}{\partial h_t} \, \frac{\partial h_k}{\partial W_{hh}}\bigg|_{\text{partial}}}
$$

(여기서 $\partial h_k / \partial W_{hh}|_{\text{partial}}$ 는 $W_{hh}$ 의 $k$-th usage 에 대한 partial — 다른 step 의 $W_{hh}$ 는 fix)

**증명**: $L_t \leftarrow h_t \leftarrow h_{t-1} \leftarrow \ldots \leftarrow h_k \leftarrow W_{hh}$ 의 모든 path 합산. Chain rule on each:

$$
\frac{\partial L_t}{\partial W_{hh}^{(k)}} = \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_{t-1}} \cdots \frac{\partial h_{k+1}}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}^{(k)}} = \frac{\partial L_t}{\partial h_t} \prod_{j=k+1}^{t} J_j \cdot \frac{\partial h_k}{\partial W_{hh}^{(k)}}
$$

모든 $k$ (1부터 $t$ 까지) 에 대해 합산. Shared weight 의 자동 합산 (정리 1.2). $\square$

**의미**: Jacobian 의 곱 $\prod_{j=k+1}^t J_j$ 가 핵심 항. Spectral radius $\rho(\prod J_j)$ 가 vanishing/exploding 결정 (Ch3-01).

### 정리 2.4 — $\partial L / \partial W_{xh}$ 와 $\partial L / \partial b_h$

대칭으로:

$$
\frac{\partial L}{\partial W_{xh}} = \sum_t (\delta_t \odot \tanh'(z_t)) \, x_t^\top, \quad \frac{\partial L}{\partial b_h} = \sum_t (\delta_t \odot \tanh'(z_t))
$$

### 정리 2.5 — Output Layer Gradient

$y_t = W_{hy} h_t + b_y$ 이므로:

$$
\frac{\partial L}{\partial W_{hy}} = \sum_t \frac{\partial L_t}{\partial y_t} h_t^\top, \quad \frac{\partial L}{\partial b_y} = \sum_t \frac{\partial L_t}{\partial y_t}
$$

(Loss 가 $L_t$ 에만 직접 영향, recurrent path 없음)

---

## 💻 NumPy 구현 검증

### 실험 1 — BPTT 바닥부터

```python
import numpy as np

class RNN_BPTT:
    def __init__(self, D, H, O, seed=0):
        rng = np.random.RandomState(seed)
        s = np.sqrt(1.0 / H)
        self.Wxh = rng.randn(H, D) * s
        self.Whh = rng.randn(H, H) * s
        self.Why = rng.randn(O, H) * s
        self.bh = np.zeros(H); self.by = np.zeros(O)
        self.D, self.H, self.O = D, H, O
    
    def forward(self, x_seq, targets):
        T = len(x_seq)
        H = self.H
        hs = np.zeros((T+1, H))
        zs = np.zeros((T, H))
        ys = np.zeros((T, self.O))
        L_total = 0.0
        for t in range(T):
            zs[t] = self.Whh @ hs[t] + self.Wxh @ x_seq[t] + self.bh
            hs[t+1] = np.tanh(zs[t])
            ys[t] = self.Why @ hs[t+1] + self.by
            L_total += 0.5 * np.sum((ys[t] - targets[t])**2)
        return L_total, hs, zs, ys
    
    def bptt(self, x_seq, targets, hs, zs, ys):
        T = len(x_seq)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh  = np.zeros_like(self.bh)
        dby  = np.zeros_like(self.by)
        
        # delta from future: δ_{t+1} 누적
        dh_next = np.zeros(self.H)
        
        for t in reversed(range(T)):
            # Output gradient
            dy = ys[t] - targets[t]                      # ∂L_t / ∂y_t
            dWhy += np.outer(dy, hs[t+1])
            dby  += dy
            
            # δ_t = ∂L_t/∂h_t + J_{t+1}^T δ_{t+1}
            #     = W_hy^T dy + δ_t (from future)
            dh = self.Why.T @ dy + dh_next               # δ_t
            
            # ∂L/∂z_t = δ_t ⊙ tanh'(z_t)
            dz = dh * (1 - np.tanh(zs[t])**2)
            
            # 누적 to weights
            dbh  += dz
            dWxh += np.outer(dz, x_seq[t])
            dWhh += np.outer(dz, hs[t])                  # h_{t-1}
            
            # δ_t backward to δ_{t-1}: J_t^T δ_t = W_hh^T (δ_t ⊙ tanh'(z_t))
            dh_next = self.Whh.T @ dz
        
        return dWxh, dWhh, dWhy, dbh, dby

# Toy 검증
D, H, O, T = 3, 5, 2, 4
rnn = RNN_BPTT(D, H, O, seed=42)
np.random.seed(0)
x_seq   = np.random.randn(T, D)
targets = np.random.randn(T, O)

L, hs, zs, ys = rnn.forward(x_seq, targets)
dWxh, dWhh, dWhy, dbh, dby = rnn.bptt(x_seq, targets, hs, zs, ys)
print(f'Loss: {L:.6f}')
print(f'dW_hh shape: {dWhh.shape}, ||dW_hh||: {np.linalg.norm(dWhh):.4f}')
```

### 실험 2 — PyTorch autograd 와 일치 검증

```python
import torch
import torch.nn as nn

# 같은 weight 로 PyTorch RNN 구성
class TorchRNN(nn.Module):
    def __init__(self, D, H, O):
        super().__init__()
        self.cell = nn.RNNCell(D, H, nonlinearity='tanh')
        self.out  = nn.Linear(H, O)
    def forward(self, x_seq, targets):
        T = len(x_seq)
        h = torch.zeros(self.cell.hidden_size)
        L = 0.0
        for t in range(T):
            h = self.cell(x_seq[t], h)
            y = self.out(h)
            L = L + 0.5 * ((y - targets[t])**2).sum()
        return L

torch_rnn = TorchRNN(D, H, O)
with torch.no_grad():
    torch_rnn.cell.weight_ih.copy_(torch.tensor(rnn.Wxh, dtype=torch.float32))
    torch_rnn.cell.weight_hh.copy_(torch.tensor(rnn.Whh, dtype=torch.float32))
    torch_rnn.cell.bias_ih.copy_(torch.tensor(rnn.bh, dtype=torch.float32))
    torch_rnn.cell.bias_hh.zero_()
    torch_rnn.out.weight.copy_(torch.tensor(rnn.Why, dtype=torch.float32))
    torch_rnn.out.bias.copy_(torch.tensor(rnn.by, dtype=torch.float32))

x_t = torch.tensor(x_seq, dtype=torch.float32)
t_t = torch.tensor(targets, dtype=torch.float32)
L_torch = torch_rnn(x_t, t_t)
L_torch.backward()

dWhh_torch = torch_rnn.cell.weight_hh.grad.numpy()
dWxh_torch = torch_rnn.cell.weight_ih.grad.numpy()
dWhy_torch = torch_rnn.out.weight.grad.numpy()

print(f'Loss diff:  {abs(L - L_torch.item()):.2e}')
print(f'dWhh diff:  {np.abs(dWhh - dWhh_torch).max():.2e}')
print(f'dWxh diff:  {np.abs(dWxh - dWxh_torch).max():.2e}')
print(f'dWhy diff:  {np.abs(dWhy - dWhy_torch).max():.2e}')
# 모두 < 1e-5 이면 BPTT 유도 정확
```

### 실험 3 — Jacobian 곱 가시화

```python
def jacobian_chain(rnn, zs, k, t):
    """∏_{j=k+1}^{t} J_j^T 계산"""
    H = rnn.H
    prod = np.eye(H)
    for j in range(k+1, t+1):
        Jj = rnn.Whh.T @ np.diag(1 - np.tanh(zs[j-1])**2)
        prod = Jj @ prod
    return prod

# t=3 에서 k=0 까지의 chain
chain_3_0 = jacobian_chain(rnn, zs, k=0, t=3)
print(f'||∏ J^T from k=0 to t=3||_F = {np.linalg.norm(chain_3_0):.4f}')

# Spectral radius
eigvals = np.linalg.eigvals(chain_3_0)
print(f'Spectral radius of chain product = {np.abs(eigvals).max():.4f}')
# < 1: vanishing 신호, > 1: exploding 신호 (Ch3-01)
```

### 실험 4 — Loss term 별 contribution 분해

```python
def loss_term_gradient(rnn, x_seq, targets, t_target):
    """L_t 만의 ∂L_t / ∂W_hh 계산 (다른 step 의 loss 무시)"""
    # 단일 step loss 만 활성화
    targets_zero = np.zeros_like(targets)
    targets_zero[t_target] = targets[t_target]
    # forward 는 같음, backward 시 다른 step gradient 무시
    # 간단히: 위 BPTT 에서 dy 를 0 으로 (t != t_target)
    T = len(x_seq)
    L_t, hs, zs, ys = rnn.forward(x_seq, targets_zero)
    
    # 수정된 BPTT (target_t 외 zero loss)
    dWhh = np.zeros_like(rnn.Whh)
    dh_next = np.zeros(rnn.H)
    for t in reversed(range(T)):
        if t == t_target:
            dy = ys[t] - targets[t]
        else:
            dy = np.zeros(rnn.O)
        dh = rnn.Why.T @ dy + dh_next
        dz = dh * (1 - np.tanh(zs[t])**2)
        dWhh += np.outer(dz, hs[t])
        dh_next = rnn.Whh.T @ dz
    return dWhh

# 각 t 의 contribution
for t in range(T):
    dW_t = loss_term_gradient(rnn, x_seq, targets, t)
    print(f'∂L_{t} / ∂W_hh norm = {np.linalg.norm(dW_t):.4f}')
# 합산 = 전체 ∂L / ∂W_hh
```

### 실험 5 — Gradient norm 시간축

```python
# δ_t 의 norm 을 시간축으로 plot
def get_deltas(rnn, x_seq, targets):
    T = len(x_seq)
    L, hs, zs, ys = rnn.forward(x_seq, targets)
    deltas = np.zeros((T, rnn.H))
    dh_next = np.zeros(rnn.H)
    for t in reversed(range(T)):
        dy = ys[t] - targets[t]
        dh = rnn.Why.T @ dy + dh_next
        deltas[t] = dh
        dz = dh * (1 - np.tanh(zs[t])**2)
        dh_next = rnn.Whh.T @ dz
    return deltas

T = 50
x_long = np.random.randn(T, D)
y_long = np.random.randn(T, O)
deltas = get_deltas(rnn, x_long, y_long)
norms = np.linalg.norm(deltas, axis=1)
print(f'||δ_t|| at t=0:  {norms[0]:.4f}')
print(f'||δ_t|| at t=49: {norms[-1]:.4f}')
print(f'Ratio (early / late) = {norms[0] / norms[-1]:.4f}')
# Vanishing: early ≪ late, exploding: early ≫ late
```

---

## 🔗 실전 활용

### 1. PyTorch autograd 가 이 모든 것을 자동 수행

`loss.backward()` 한 줄이 위의 reverse iteration 을 정확히 실행. NumPy 구현은 *이해* 를 위한 것.

### 2. Custom RNN cell 구현 시

새로운 gating 메커니즘 (LSTM, GRU, Mamba) 의 BPTT 를 유도해야 cell 의 gradient flow 분석 가능.

### 3. Gradient 분석 도구

`torch.autograd.grad` 로 specific weight 의 gradient norm 추적, vanishing 진단.

### 4. Synthetic gradient (Jaderberg 2017)

BPTT 를 근사하여 module-level local update 가능. 분산 학습에서 step lag 줄임.

### 5. Differentiable computing

Neural ODE, NTM, differentiable rendering 등이 모두 BPTT 의 일반화 — chain rule on dynamic computation graph.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 모든 forward activation 보존 | Memory $O(TH)$ — checkpointing |
| 정확한 gradient | Truncated BPTT 근사 (Ch2-03) |
| Forward-mode 보다 효율 | $O(n^2)$ vs RTRL $O(n^4)$ — output 수가 적을 때만 우월 |
| Sequence-internal sequential | 병렬화 불가 → Transformer |
| Single trajectory | RTRL, UORO 같은 forward-mode 대안 필요 시 |

---

## 📌 핵심 정리

$$\boxed{\delta_t = \frac{\partial L_t}{\partial h_t} + J_{t+1}^\top \delta_{t+1}, \quad J_t = \mathrm{diag}(\tanh'(z_t))\, W_{hh}}$$

$$\boxed{\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} (\delta_t \odot \tanh'(z_t)) \, h_{t-1}^\top}$$

$$\boxed{\frac{\partial L_t}{\partial W_{hh}} = \sum_{k \le t} \big(\prod_{j=k+1}^{t} J_j^\top\big) \frac{\partial L_t}{\partial h_t} \, h_{k-1}^\top \tanh'(z_k) \quad \text{— Pascanu 형태}}$$

| 경로 | 수식 | Computational |
|------|------|---------------|
| **Forward** | $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$ | $O(TH^2)$ |
| **$\delta$ backward** | $\delta_t = \partial L_t/\partial h_t + J_{t+1}^\top \delta_{t+1}$ | $O(TH^2)$ |
| **Weight gradient** | $\nabla W_{hh} = \sum (\delta_t \odot \tanh') h_{t-1}^\top$ | $O(TH^2)$ |
| **Memory** | All $h_t, z_t$ | $O(TH)$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T = 2$, $H = 1$ 인 super-simple RNN ($w_{hh}, w_{xh}, w_{hy} \in \mathbb R$, no bias) 에서 $L = (y_2 - y_2^*)^2 / 2$ 만 사용. $\partial L / \partial w_{hh}$ 를 손으로 유도하라.

<details>
<summary>해설</summary>

**Forward**:
- $z_1 = w_{hh} h_0 + w_{xh} x_1$, $h_1 = \tanh(z_1)$
- $z_2 = w_{hh} h_1 + w_{xh} x_2$, $h_2 = \tanh(z_2)$
- $y_2 = w_{hy} h_2$
- $L = \frac{1}{2}(y_2 - y_2^*)^2$

**Backward**:
- $\partial L / \partial y_2 = y_2 - y_2^* =: dy$
- $\partial L / \partial h_2 = w_{hy} \cdot dy$
- $\partial L / \partial z_2 = w_{hy} dy \cdot (1 - h_2^2) =: dz_2$
- $\partial L / \partial h_1 = w_{hh} \cdot dz_2$
- $\partial L / \partial z_1 = w_{hh} dz_2 \cdot (1 - h_1^2) =: dz_1$

**$\partial L / \partial w_{hh}$**:
- $w_{hh}$ 가 $z_2$ 에 ($h_1$ 곱) 와 $z_1$ 에 ($h_0$ 곱) 두 번 사용:

$$
\frac{\partial L}{\partial w_{hh}} = dz_2 \cdot h_1 + dz_1 \cdot h_0
$$

전개:

$$
= w_{hy} dy (1 - h_2^2) h_1 + w_{hh} w_{hy} dy (1 - h_2^2)(1 - h_1^2) h_0
$$

**$h_0 = 0$** 이면 두 번째 항 사라짐. $\square$

</details>

**문제 2** (심화): 정리 2.3 의 Jacobian 곱 $\prod_{j=k+1}^{t} J_j^\top$ 를 사용해, $\rho(W_{hh}) < 1$ 이고 $\sigma' \le 1$ 인 경우 $\|\partial L_t / \partial W_{hh}^{(k)}\|$ 가 $t - k$ 에 대해 어떻게 행동하는지 보이라.

<details>
<summary>해설</summary>

**Spectral 분석**:

$$
J_j^\top = W_{hh}^\top \, \mathrm{diag}(\tanh'(z_j))
$$

Spectral radius:

$$
\rho(J_j^\top) \le \rho(W_{hh}^\top) \cdot \max_i |\tanh'(z_j)|_i \le \rho(W_{hh}) \cdot 1 < 1
$$

(Submultiplicative property of spectral radius — 정확히는 $\sigma_{\max}(J) \le \sigma_{\max}(W) \cdot \max \sigma'$)

**Product**:

$$
\left\|\prod_{j=k+1}^{t} J_j^\top\right\| \le \prod_{j=k+1}^{t} \|J_j^\top\| \le (\rho(W_{hh}))^{t-k}
$$

(Operator norm 이 spectral radius 의 supremum 으로 bounded for normal matrices)

**Vanishing**:
$$
\|\partial L_t / \partial W_{hh}^{(k)}\| \lesssim (\rho(W_{hh}))^{t-k} \cdot \|h_{k-1}\|
$$

$\rho < 1$ 이면 $t - k \to \infty$ 시 **exponential decay**. 학습이 long-range dependency 를 잡지 못함.

**$\rho > 1$**: exploding 으로 발산.

$\square$ — 이는 Pascanu 2013 의 핵심 정리 (Ch3-01 에서 자세히)

</details>

**문제 3** (논문 비평): BPTT 와 RTRL 은 같은 gradient 를 다른 방식으로 계산한다. RTRL 이 forward-mode AD 로 $O(n^4)$ vs BPTT 의 $O(n^2)$ 인 이유를 설명하고, online learning 시 RTRL 이 다시 주목받는 맥락을 논하라.

<details>
<summary>해설</summary>

**복잡도 비교**:

**BPTT (reverse-mode AD)**:
- Forward: $T$ steps, 각 $O(H^2)$ — total $O(TH^2)$
- Backward: $T$ steps, 각 $O(H^2)$ — total $O(TH^2)$
- Per step: $O(H^2)$

**RTRL (forward-mode AD)**:
- Forward 와 함께 propagate: $S_t = \partial h_t / \partial \theta \in \mathbb R^{H \times |\theta|}$
- $|\theta| = O(H^2)$ for $W_{hh}$ → $S_t$ 가 $H \times H^2 = H^3$ 차원
- Update $S_t$ 는 $H^4$ ops
- Per step: $O(H^4)$

**왜 차이?**:
- Reverse-mode: gradient 가 scalar loss 에서 backward — output 차원에 의존
- Forward-mode: Jacobian 전체 propagate — input 차원에 의존
- Loss is scalar, $\theta$ is high-dim → **reverse 가 우월**

**RTRL 의 부활 맥락**:
1. **Online learning**: BPTT 는 sequence 끝까지 forward 후 backward — episode-end update only. RTRL 은 매 step update 가능.
2. **Memory**: BPTT 는 $O(TH)$ — long sequence 에서 메모리 한계. RTRL 은 $O(H^3)$ state 만 유지.
3. **Approximations**: 
   - **UORO** (Tallec 2017): $O(H^2)$ unbiased estimator, RTRL 의 random projection
   - **e-prop** (Bellec 2020): Biological plausibility, local online learning
4. **Continual learning**: BPTT 의 episode 경계가 없는 streaming 학습에 RTRL/UORO 적합

**결론**: Production 에서는 BPTT (truncated) 가 표준이지만, online RL, edge computing, neural plasticity 연구에서 RTRL family 가 재조명. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-unrolled-graph.md) | [📚 README](../README.md) | [다음 ▶](./03-truncated-bptt.md)

</div>
