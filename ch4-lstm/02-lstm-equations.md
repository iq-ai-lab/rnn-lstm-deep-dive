# 02. LSTM 의 4개 Gate 수식

## 🎯 핵심 질문

- LSTM 의 4개 gate (forget, input, candidate, output) 의 정확한 수식은?
- 왜 forget/input/output 은 sigmoid, candidate 은 tanh 인가? 각 활성화 함수의 역할은?
- Cell update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$ 와 hidden update $h_t = o_t \odot \tanh(c_t)$ 의 정확한 의미는?
- NumPy 로 바닥부터 LSTM 을 구현하고 PyTorch `nn.LSTM` 과 출력이 정확히 일치하는지 검증
- 4 gate weight 의 stacked representation $W \in \mathbb R^{4H \times (D+H)}$ — PyTorch 의 effective implementation

---

## 🔍 왜 LSTM 의 정확한 수식이 중요한가

LSTM 을 "쓴다" 와 "이해한다" 의 차이는 다음에서 드러납니다:

1. **Gate 의 역할 구분** — 각 gate 가 어떤 task 의 어떤 측면을 담당하는지
2. **Forward 의 정확한 메모리 lookup** — Karpathy 의 LSTM 시각화는 정확한 수식 이해 위에
3. **Backprop 시 gradient flow** — 어떤 weight 가 어떤 신호로부터 update 받는지
4. **Variant 설계** — Peephole, Coupled, ConvLSTM 등이 표준 수식의 어떤 부분을 변경하는지
5. **PyTorch implementation 효율** — 4 gate weight stack 의 single matmul

이 문서는 LSTM 의 모든 수식을 한 줄씩 derive 하고 NumPy 로 PyTorch 와 일치 검증합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-lstm-motivation.md](./01-lstm-motivation.md) — CEC 비전, additive update
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Sigmoid, tanh, element-wise operations
- (선택) Vector calculus: Element-wise product (Hadamard) 와 matrix-vector product

---

## 📖 직관적 이해

### 4 Gate 의 역할 비유

```
       ┌────────────┐
       │   Forget   │   "이전 기억 얼마나 유지?"
       │   gate     │   sigmoid → [0, 1]
       └─────┬──────┘
             │
             ▼
        c_{t-1} ⊙ f_t  ──── + ─── c_t  →  output
                          ▲
                          │
                          i_t ⊙ ĉ_t
                       ┌──┴────┐
                       │       │
                  ┌────────┐ ┌─────────┐
                  │ Input  │ │ Cand.   │
                  │ gate   │ │ ĉ       │
                  │ σ      │ │ tanh    │
                  └────────┘ └─────────┘
                  "새 정보 얼마나?"  "어떤 새 정보?"
```

```
        c_t  →  tanh(c_t)
                    │
                    ▼
                    ⊙
                    ▲
                    │
              ┌──────────┐
              │  Output  │   "cell 정보 외부 공개?"
              │  gate    │   sigmoid
              └──────────┘
                    │
                    ▼
                   h_t
```

### 활성화 함수의 분업

- **Sigmoid** (forget, input, output): $\in [0, 1]$ — *얼마나* (degree of openness)
- **Tanh** (candidate, output transform): $\in [-1, 1]$ — *어떤 값* (signed magnitude)

이 분업은 정보의 "양 (gate)" 과 "내용 (value)" 을 분리.

### 4 Gate 가 모두 같은 구조

```
gate = sigmoid(W_gate · [h_{t-1}; x_t] + b_gate)
```

또는 candidate:

```
ĉ_t = tanh(W_c · [h_{t-1}; x_t] + b_c)
```

같은 input $[h; x]$ 에서 4 different linear transformation + activation. **Stacked matmul** 이 자연스러움.

### Stacked Weight Matrix

PyTorch 표현:
- $W \in \mathbb R^{4H \times D}$ — input projection (input 부분)
- $U \in \mathbb R^{4H \times H}$ — hidden projection
- $b \in \mathbb R^{4H}$

각각 4 gate 의 weight 가 row-wise stacked. Single matmul:

$$
[\hat f; \hat i; \hat g; \hat o] = W x_t + U h_{t-1} + b
$$

그리고 component-wise activation 적용.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — LSTM Equations (Standard)

Time step $t$ 에서, $h_{t-1}, c_{t-1} \in \mathbb R^H$ given:

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}; x_t] + b_f) & \text{forget gate} \\
i_t &= \sigma(W_i [h_{t-1}; x_t] + b_i) & \text{input gate} \\
\tilde c_t &= \tanh(W_c [h_{t-1}; x_t] + b_c) & \text{candidate} \\
o_t &= \sigma(W_o [h_{t-1}; x_t] + b_o) & \text{output gate} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t & \text{cell state update} \\
h_t &= o_t \odot \tanh(c_t) & \text{hidden state}
\end{aligned}
$$

여기서 $[h_{t-1}; x_t] \in \mathbb R^{H+D}$ 는 concatenation, $\odot$ 는 element-wise product.

### 정의 2.2 — Stacked Form (PyTorch)

$$
\begin{pmatrix} \hat f_t \\ \hat i_t \\ \hat g_t \\ \hat o_t \end{pmatrix} = \underbrace{W_x}_{4H \times D} x_t + \underbrace{W_h}_{4H \times H} h_{t-1} + b
$$

$$
f_t = \sigma(\hat f_t), \quad i_t = \sigma(\hat i_t), \quad \tilde c_t = \tanh(\hat g_t), \quad o_t = \sigma(\hat o_t)
$$

### 정의 2.3 — Parameter Count

$|\theta_{\text{LSTM}}| = 4 \times (H \cdot (D + H) + H) = 4H(D + H + 1)$.

비교: Plain RNN $|\theta| = H(D + H + 1)$ — LSTM 은 4x parameters.

### 정의 2.4 — Initialization

표준:
- $W_x, W_h$: Xavier/Glorot 또는 orthogonal
- $b_f$: **1** (Jozefowicz 2015, Ch4-04 에서 자세히)
- $b_i, b_c, b_o$: 0

### 정의 2.5 — Forward Computation Cost

Per step:
- Matmul $W_x x_t$: $O(4HD)$
- Matmul $W_h h_{t-1}$: $O(4H^2)$
- Activations: $O(H)$
- Element-wise: $O(H)$

Total: $O(4H(H + D)) \approx O(H^2)$ for $H \gg D$. Plain RNN 의 4배.

---

## 🔬 정리와 결과

### 정리 2.1 — LSTM 의 Output Range

$h_t \in [-1, 1]^H$ for all $t$.

**증명**: $h_t = o_t \odot \tanh(c_t)$. $o_t \in [0, 1]$, $\tanh(c_t) \in [-1, 1]$. 곱: $[-1, 1]$. $\square$

### 정리 2.2 — Cell State 의 Unboundedness

$c_t$ 는 unbounded:

$$
c_t = f_t c_{t-1} + i_t \tilde c_t
$$

$f_t, i_t \in [0, 1]$, $\tilde c_t \in [-1, 1]$. 그러나 $c_{t-1}$ 자체가 unbounded.

**Example**: $f_t = 1, i_t = 1, \tilde c_t = 1$ for all $t$ → $c_t = c_{t-1} + 1$ → $c_T = T$.

**의미**: 학습 중 $c_t$ 가 매우 커질 수 있음 → $\tanh(c_t)$ 가 saturated. $h_t$ 의 information 이 손실 가능.

**해결**: Layer norm (Ba 2016) — cell normalization.

### 정리 2.3 — 4 Gate 의 함수적 분리

각 gate 가 independent linear transformation:

$$
\hat g_k = W_k [h; x] + b_k, \quad k \in \{f, i, c, o\}
$$

**Independence**: 다른 gate 의 weight 가 같은 input 에 대해 다른 mapping 을 학습 — *modular* representation.

### 정리 2.4 — LSTM 과 GRU 의 정보 흐름 차이

LSTM: cell state $c$ + hidden state $h$ separated.
GRU (Ch4-05): single hidden state $h$ with update gate $z$.

LSTM 의 separation 이 더 표현력 있지만 GRU 의 unified state 가 학습 효율적인 경우 있음.

### 정리 2.5 — PyTorch nn.LSTM 의 정확한 식

PyTorch 의 LSTM 은 위 정의와 *almost* 같지만 bias 가 두 개:

```python
# PyTorch nn.LSTMCell 의 정확한 식
i = sigmoid(W_ii x + b_ii + W_hi h + b_hi)
f = sigmoid(W_if x + b_if + W_hf h + b_hf)
g = tanh(W_ig x + b_ig + W_hg h + b_hg)
o = sigmoid(W_io x + b_io + W_ho h + b_ho)
```

`b_ii, b_hi` 분리는 historical (Caffe convention). 학습 때 둘 다 update — 결과는 동일.

---

## 💻 NumPy 구현 검증

### 실험 1 — NumPy LSTM Cell 바닥부터

```python
import numpy as np

class NumPyLSTM:
    def __init__(self, D, H, seed=0):
        rng = np.random.RandomState(seed)
        s = np.sqrt(1.0 / H)
        # Stacked weight: 4H × (D + H)
        self.W_x = rng.randn(4*H, D) * s
        self.W_h = rng.randn(4*H, H) * s
        # Bias: forget gate = 1, others = 0
        self.b = np.zeros(4*H)
        self.b[:H] = 1.0   # Forget bias = 1 (Jozefowicz 2015)
        self.D, self.H = D, H
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x_seq, h0=None, c0=None):
        T = len(x_seq)
        H = self.H
        h = np.zeros(H) if h0 is None else h0
        c = np.zeros(H) if c0 is None else c0
        
        history = []
        for t in range(T):
            # Stacked pre-activation
            z = self.W_x @ x_seq[t] + self.W_h @ h + self.b   # (4H,)
            
            # Split into 4 gates
            f = self.sigmoid(z[0:H])
            i = self.sigmoid(z[H:2*H])
            g = np.tanh(z[2*H:3*H])
            o = self.sigmoid(z[3*H:4*H])
            
            # Cell update
            c = f * c + i * g
            # Hidden update
            h = o * np.tanh(c)
            
            history.append({'h': h.copy(), 'c': c.copy(),
                           'f': f, 'i': i, 'g': g, 'o': o})
        
        return h, c, history

# Toy
D, H, T = 4, 8, 5
lstm = NumPyLSTM(D, H, seed=42)
x_seq = np.random.randn(T, D)
h_T, c_T, hist = lstm.forward(x_seq)

print(f'h_T = {h_T[:4]}')
print(f'c_T = {c_T[:4]}')
print(f'Forget gate (avg over T): {np.mean([h["f"].mean() for h in hist]):.4f}')
print(f'  → Around σ(1) ≈ 0.73 due to b_f = 1')
```

### 실험 2 — PyTorch nn.LSTMCell 과 일치 검증

```python
import torch
import torch.nn as nn

# 같은 weight 로 PyTorch LSTMCell
torch_lstm = nn.LSTMCell(D, H)

# Weight 동기화
with torch.no_grad():
    # PyTorch 의 weight order: i, f, g, o
    # 우리: f, i, g, o (Hochreiter convention)
    # 변환 필요!
    
    # 우리의 stacked: [f|i|g|o]
    # PyTorch: weight_ih = [i|f|g|o], weight_hh = [i|f|g|o]
    
    Wx_torch = np.zeros_like(lstm.W_x)
    Wh_torch = np.zeros_like(lstm.W_h)
    b_torch = np.zeros(4*H)
    
    # Permutation: f→1, i→0, g→2, o→3 (우리) vs PyTorch i→0, f→1
    Wx_torch[0:H]   = lstm.W_x[H:2*H]    # i (PyTorch) = i (ours)
    Wx_torch[H:2*H] = lstm.W_x[0:H]      # f (PyTorch) = f (ours)
    Wx_torch[2*H:3*H] = lstm.W_x[2*H:3*H]  # g
    Wx_torch[3*H:4*H] = lstm.W_x[3*H:4*H]  # o
    
    Wh_torch[0:H]   = lstm.W_h[H:2*H]
    Wh_torch[H:2*H] = lstm.W_h[0:H]
    Wh_torch[2*H:3*H] = lstm.W_h[2*H:3*H]
    Wh_torch[3*H:4*H] = lstm.W_h[3*H:4*H]
    
    b_torch[0:H]   = lstm.b[H:2*H]
    b_torch[H:2*H] = lstm.b[0:H]
    b_torch[2*H:3*H] = lstm.b[2*H:3*H]
    b_torch[3*H:4*H] = lstm.b[3*H:4*H]
    
    torch_lstm.weight_ih.copy_(torch.tensor(Wx_torch, dtype=torch.float32))
    torch_lstm.weight_hh.copy_(torch.tensor(Wh_torch, dtype=torch.float32))
    torch_lstm.bias_ih.copy_(torch.tensor(b_torch, dtype=torch.float32))
    torch_lstm.bias_hh.zero_()

# Forward
x_t = torch.tensor(x_seq, dtype=torch.float32)
h, c = torch.zeros(1, H), torch.zeros(1, H)
for t in range(T):
    h, c = torch_lstm(x_t[t].unsqueeze(0), (h, c))

print(f'NumPy h_T:   {h_T[:4]}')
print(f'PyTorch h_T: {h.squeeze().numpy()[:4]}')
print(f'Match? {np.allclose(h_T, h.squeeze().numpy(), atol=1e-5)}')
```

### 실험 3 — Stacked Matmul 의 효율성

```python
import time

H, D, T = 256, 64, 100
torch.manual_seed(0)

# Method 1: 4 separate matmul
class LSTM_Separate(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.W_f = nn.Linear(D + H, H)
        self.W_i = nn.Linear(D + H, H)
        self.W_c = nn.Linear(D + H, H)
        self.W_o = nn.Linear(D + H, H)
    def forward(self, x_seq, h, c):
        for t in range(x_seq.size(0)):
            xh = torch.cat([x_seq[t], h], dim=-1)
            f = torch.sigmoid(self.W_f(xh))
            i = torch.sigmoid(self.W_i(xh))
            g = torch.tanh(self.W_c(xh))
            o = torch.sigmoid(self.W_o(xh))
            c = f * c + i * g; h = o * torch.tanh(c)
        return h, c

# Method 2: Stacked matmul
class LSTM_Stacked(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.W = nn.Linear(D + H, 4 * H)
        self.H = H
    def forward(self, x_seq, h, c):
        for t in range(x_seq.size(0)):
            xh = torch.cat([x_seq[t], h], dim=-1)
            z = self.W(xh)   # (B, 4H)
            f = torch.sigmoid(z[:, :self.H])
            i = torch.sigmoid(z[:, self.H:2*self.H])
            g = torch.tanh(z[:, 2*self.H:3*self.H])
            o = torch.sigmoid(z[:, 3*self.H:])
            c = f * c + i * g; h = o * torch.tanh(c)
        return h, c

x = torch.randn(T, 32, D)
h0, c0 = torch.zeros(32, H), torch.zeros(32, H)

m1 = LSTM_Separate(D, H)
m2 = LSTM_Stacked(D, H)

start = time.time()
for _ in range(10):
    m1(x, h0, c0)
print(f'Separate: {(time.time()-start)*100:.2f} ms / call')

start = time.time()
for _ in range(10):
    m2(x, h0, c0)
print(f'Stacked:  {(time.time()-start)*100:.2f} ms / call')
# Stacked 가 ~3x 빠름 (single matmul 의 throughput 우위)
```

### 실험 4 — Gate Activation Analysis

```python
# 학습 후 gate activation 분포 분석
torch.manual_seed(0)
class LSTMforAnalysis(nn.Module):
    def __init__(self, D, H, V):
        super().__init__()
        self.lstm = nn.LSTM(D, H, batch_first=True)
        self.embed = nn.Embedding(V, D)
        self.out = nn.Linear(H, V)
    def forward(self, x):
        e = self.embed(x)
        h, _ = self.lstm(e)
        return self.out(h)

V, D, H = 100, 32, 64
model = LSTMforAnalysis(D, H, V)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train on random data
for _ in range(50):
    x = torch.randint(0, V, (16, 30))
    target = torch.randint(0, V, (16, 30))
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, V), target.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

# Manual forward to extract gates
e = model.embed(torch.randint(0, V, (1, 50)))   # (1, 50, D)
W_ih = model.lstm.weight_ih_l0
W_hh = model.lstm.weight_hh_l0
b_ih = model.lstm.bias_ih_l0
b_hh = model.lstm.bias_hh_l0

h = torch.zeros(1, H); c = torch.zeros(1, H)
gate_history = {'i': [], 'f': [], 'g': [], 'o': []}
for t in range(e.size(1)):
    z = e[:, t, :] @ W_ih.T + h @ W_hh.T + b_ih + b_hh
    gates = z.split(H, dim=-1)
    i = torch.sigmoid(gates[0])
    f = torch.sigmoid(gates[1])
    g = torch.tanh(gates[2])
    o = torch.sigmoid(gates[3])
    c = f * c + i * g; h = o * torch.tanh(c)
    
    gate_history['i'].append(i.detach().numpy())
    gate_history['f'].append(f.detach().numpy())
    gate_history['g'].append(g.detach().numpy())
    gate_history['o'].append(o.detach().numpy())

for name, vals in gate_history.items():
    arr = np.concatenate(vals).flatten()
    print(f'{name} gate: mean={arr.mean():.3f}, std={arr.std():.3f}, sat<0.1={(arr<0.1).mean()*100:.1f}%, sat>0.9={(arr>0.9).mean()*100:.1f}%')
```

### 실험 5 — Cell State Magnitude Tracking

```python
torch.manual_seed(0)
lstm = nn.LSTM(D, H, batch_first=False)

x = torch.randn(200, 1, D)
out, (h_T, c_T) = lstm(x)

# c 의 norm tracking 위해 manual forward
h = torch.zeros(1, 1, H)
c = torch.zeros(1, 1, H)
c_norms = []
for t in range(x.size(0)):
    out_t, (h, c) = lstm(x[t:t+1], (h, c))
    c_norms.append(c.norm().item())

print(f'c norm 시간축:')
print(f'  t=0:   {c_norms[0]:.4f}')
print(f'  t=50:  {c_norms[50]:.4f}')
print(f'  t=100: {c_norms[100]:.4f}')
print(f'  t=199: {c_norms[-1]:.4f}')
# Cell state 가 bounded 하지 않음 — 학습 전 random init 도 점진적 증가
```

---

## 🔗 실전 활용

### 1. PyTorch `nn.LSTM` 표준 사용

```python
lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
output, (h_T, c_T) = lstm(x)
```

### 2. Custom LSTM cell

표준에서 벗어난 modification (e.g., layer norm) 시 nn.LSTMCell 사용:

```python
cell = nn.LSTMCell(input_size, hidden_size)
h, c = torch.zeros(B, H), torch.zeros(B, H)
for t in range(T):
    h, c = cell(x[t], (h, c))
```

### 3. Bidirectional LSTM

`bidirectional=True` 로 양방향 — NER, POS tagging 표준.

### 4. Stacked LSTM

`num_layers=2` 또는 그 이상 — Google NMT 의 8-layer LSTM.

### 5. Variational dropout (Gal 2016)

매 step 의 dropout mask 를 sequence 내에서 fix — RNN 의 표준 dropout.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 4 gates 충분 | More gates (peephole) 로 약간 향상, marginal |
| Sigmoid + tanh 표준 | 다른 activation (e.g., GELU) 가능, 거의 사용 안 함 |
| Cell unbounded | Layer norm 으로 정규화 가능 |
| Single hidden state | Multi-head LSTM 가능하지만 attention 으로 대체 |
| Sequential | Transformer 의 병렬성 부족 |

---

## 📌 핵심 정리

$$\boxed{\begin{aligned} f_t &= \sigma(W_f [h_{t-1}; x_t] + b_f) \\ i_t &= \sigma(W_i [h_{t-1}; x_t] + b_i) \\ \tilde c_t &= \tanh(W_c [h_{t-1}; x_t] + b_c) \\ o_t &= \sigma(W_o [h_{t-1}; x_t] + b_o) \\ c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t \\ h_t &= o_t \odot \tanh(c_t) \end{aligned}}$$

| Gate | Activation | Range | Role |
|------|-----------|-------|------|
| **Forget $f$** | sigmoid | [0,1] | "이전 cell 보존?" |
| **Input $i$** | sigmoid | [0,1] | "새 정보 받아들임?" |
| **Candidate $\tilde c$** | tanh | [-1,1] | "어떤 새 정보?" |
| **Output $o$** | sigmoid | [0,1] | "cell 의 어떤 부분 노출?" |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $H = 100$, $D = 50$ LSTM 의 parameter 수를 계산하라. 같은 hidden size 의 vanilla RNN 과 비교.

<details>
<summary>해설</summary>

**LSTM**:
- $W_x \in \mathbb R^{4H \times D}$: $4 \times 100 \times 50 = 20000$
- $W_h \in \mathbb R^{4H \times H}$: $4 \times 100 \times 100 = 40000$
- $b \in \mathbb R^{4H}$: $400$
- Total: **60,400**

**Vanilla RNN**:
- $W_{xh}$: $100 \times 50 = 5000$
- $W_{hh}$: $100 \times 100 = 10000$
- $b$: $100$
- Total: **15,100**

**비율**: LSTM ≈ 4x parameters of vanilla RNN.

**비교 (GRU, Ch4-05)**:
- 3 gates (no separate cell), 약 3x of vanilla
- ~75% of LSTM parameters

**Trade-off**: LSTM 의 4x parameter 가 long-range modeling 능력을 위한 cost. 작은 task 에서는 LSTM 이 over-parameterized 가능. $\square$

</details>

**문제 2** (심화): LSTM 에서 forget gate 와 input gate 가 **coupled** ($f + i = 1$, Greff 2017 의 변종) 인 경우 parameter 수와 표현력의 변화는?

<details>
<summary>해설</summary>

**Coupled forget-input** (CIFG):

$f_t + i_t = 1 \Rightarrow i_t = 1 - f_t$

수식:
$$
c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde c_t
$$

이는 **convex combination** — 이전과 새 정보의 mixing.

**Parameter 변화**:
- Forget gate weight 만 필요 (input gate weight 제거)
- $4H \to 3H$ stack
- Parameters: $0.75 \times$ LSTM

**표현력**:
- 정보 보존 vs 새 정보의 trade-off 명시적 (zero-sum)
- 모든 update 가 weighted average — 단조 행동
- 그러나 표현력 제한: $f$ 와 $i$ 가 *독립적* 일 수 없음

**Use case**:
- Streaming 학습 (information bottleneck 명시)
- Lean architecture (parameter 절약)
- GRU 와 비슷한 구조 — coupled 가 GRU 의 핵심 idea

**GRU 와의 차이**:
- CIFG: cell state 와 hidden 분리 유지
- GRU: cell + hidden 통합

**Greff 2017 의 ablation 결과**:
- CIFG vs vanilla LSTM: marginal performance loss (~1%)
- Parameter 절약과의 trade-off 합리적

**결론**: Coupling 이 단순화된 LSTM, parameter 절약 + interpretability 향상, 약간의 표현력 손실. $\square$

</details>

**문제 3** (논문 비평): PyTorch 의 `nn.LSTM` 이 bias 를 `bias_ih` 와 `bias_hh` 두 개로 분리한다. 수학적으로 이는 single bias 와 동치이다. 왜 분리되어 있는가?

<details>
<summary>해설</summary>

**Mathematical equivalence**:

$$
z = W_{ih} x + W_{hh} h + b_{ih} + b_{hh} = W_{ih} x + W_{hh} h + (b_{ih} + b_{hh})
$$

학습 시 $b_{ih}$ 와 $b_{hh}$ 가 단지 $b = b_{ih} + b_{hh}$ 의 두 components → effectively single bias.

**왜 두 개로 분리?**:

1. **Historical (Caffe convention)**:
   - 초기 deep learning library 가 dense layer 를 분리 ($W x + b$, $V h + b'$)
   - Concat 후 add 하면 두 separate bias

2. **Implementation parallel**:
   - GPU 에서 $W_{ih} x$ 와 $W_{hh} h$ 를 separate kernel 로 실행 가능
   - 각각 자체 bias add — 결과 동일

3. **CuDNN compatibility**:
   - NVIDIA 의 cuDNN LSTM kernel 이 두 bias 를 expect
   - PyTorch 의 nn.LSTM 이 cuDNN 을 wrap

4. **Custom modification**:
   - 두 bias 를 분리하면 input bias 와 hidden bias 를 다르게 init 가능
   - 일부 paper 에서 input bias 만 zero, hidden bias 는 1 (forget) 등 모듈화

**PyTorch 의 contiguous memory**:
- `bias_ih_l0` 와 `bias_hh_l0` 가 별도 tensor — separate optimizer state
- Adam 의 first/second moment 가 각 bias 별로 추적
- 학습 dynamics 는 single bias 와 다를 수 있음 (마찰 수준)

**결론**: 수학적으로 동치, 그러나:
- Historical: legacy
- Practical: cuDNN, GPU efficiency
- Research: bias 별 분리된 modification 가능

**대부분의 사용자**: 두 bias 의 차이 무시, single bias 처럼 작동. 그러나 forget bias = 1 init 시 `bias_ih_l0` 의 forget 부분에 1, `bias_hh_l0` 의 forget 부분에 0 으로 set (sum 이 1) — 이것이 PyTorch 의 일반적 방식. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-lstm-motivation.md) | [📚 README](../README.md) | [다음 ▶](./03-cec-proof.md)

</div>
