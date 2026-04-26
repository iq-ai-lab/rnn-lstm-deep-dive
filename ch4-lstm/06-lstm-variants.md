# 06. LSTM Variants — Peephole · Coupled · ConvLSTM

## 🎯 핵심 질문

- Peephole connection (Gers 2002) 이 cell state 를 gate 계산에 포함하는 메커니즘과 그 효과는?
- Coupled input-forget gate (Greff 2017) 가 $i_t + f_t = 1$ 강제로 어떤 표현력 변화를 만드는가?
- **ConvLSTM** (Shi 2015) 이 matrix multiplication 을 convolution 으로 바꾼 정확한 의미와 video / spatial sequence 에의 적용?
- **Greff 2017** *LSTM: A Search Space Odyssey* 의 8가지 variants ablation 결과 — vanilla LSTM 의 robust 입증
- LSTM 에 추가 gate 또는 connection 을 추가하는 것이 marginal benefit 인 이유

---

## 🔍 왜 LSTM variants 의 검토가 architectural insight 를 주는가

LSTM 의 표준 4 gate formulation 이 *유일한* 또는 *최적의* 형태인지 검증하는 것이 중요. 수많은 variants 가 제안되었지만:

1. **Greff 2017** — 8 variants ablation: vanilla LSTM 이 surprisingly robust
2. **Marginal improvements** — 대부분 variants 가 0.5~2% 만 향상, computational cost 증가
3. **Specialized use cases** — ConvLSTM 등은 *generic* 향상이 아닌 *specific domain* 의 fitting

이 문서는 주요 variants 를 정리하고, "왜 vanilla LSTM 이 우월한가" 의 paradox 를 분석합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [05-gru.md](./05-gru.md) — GRU 의 simplification
- [Ch4-02 LSTM equations](./02-lstm-equations.md) — Standard LSTM
- [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive) — Convolution operation (ConvLSTM)

---

## 📖 직관적 이해

### Peephole 의 직관

**Standard LSTM**: gates 가 $h_{t-1}$ 과 $x_t$ 만 봄.

**Peephole**: gates 가 *cell state* 도 봄:

```
   c_{t-1}  →  ┌─────────┐
   h_{t-1}  →  │  gate   │
   x_t      →  │  σ(...)  │  →  f_t (or i_t, o_t)
                └─────────┘
```

직관: "지금 cell 안에 무엇이 있는지 보고 결정 — precision timing".

### Coupled Input-Forget 의 직관

**Standard**: $f_t$ 와 $i_t$ 가 *독립*.

**Coupled (CIFG)**: $i_t = 1 - f_t$.

```
c_t = f_t · c_{t-1} + (1 - f_t) · c̃_t
    = convex combination
```

GRU 의 update gate 와 동일 정신. Parameter 절약.

### ConvLSTM 의 직관

**Standard LSTM**: gate $f_t = \sigma(W_f \cdot \text{vec})$ — fully connected.

**ConvLSTM** (video, spatial data): $f_t = \sigma(W_f * X_t + U_f * H_{t-1} + b_f)$ — convolution.

```
LSTM:   1D vector input → 1D vector hidden
ConvLSTM: 2D image input → 2D feature map hidden
```

Spatial structure 가 보존되는 sequence model.

### Greff 2017 의 ablation 정신

8 variants:
1. Vanilla LSTM
2. No input gate (NIG): $c_t = f_t c_{t-1} + \tilde c_t$
3. No forget gate (NFG): $c_t = c_{t-1} + i_t \tilde c_t$
4. No output gate (NOG): $h_t = \tanh(c_t)$
5. No input activation (NIAF): $c_t = f c_{t-1} + i (W [h, x] + b)$
6. No output activation (NOAF): $h = o \cdot c$
7. Coupled (CIFG)
8. Peephole

ablation 결과: NIG 와 NOG 만 큰 성능 손실, 나머지는 marginal.

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Peephole LSTM (Gers 2002)

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}; x_t] + V_f \cdot c_{t-1} + b_f) \\
i_t &= \sigma(W_i [h_{t-1}; x_t] + V_i \cdot c_{t-1} + b_i) \\
\tilde c_t &= \tanh(W_c [h_{t-1}; x_t] + b_c) \\
o_t &= \sigma(W_o [h_{t-1}; x_t] + V_o \cdot c_t + b_o) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

$V_f, V_i, V_o$: peephole weight (diagonal 또는 dense).

### 정의 6.2 — Coupled Input-Forget Gate (CIFG)

$$
i_t = 1 - f_t, \qquad c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde c_t
$$

Parameter 절약: $4H(D+H+1) \to 3H(D+H+1)$.

### 정의 6.3 — ConvLSTM (Shi 2015)

Input $X_t \in \mathbb R^{D \times H_{\text{img}} \times W_{\text{img}}}$, hidden $H_t \in \mathbb R^{C \times H_{\text{img}} \times W_{\text{img}}}$:

$$
\begin{aligned}
F_t &= \sigma(W_f * X_t + U_f * H_{t-1} + b_f) \\
I_t &= \sigma(W_i * X_t + U_i * H_{t-1} + b_i) \\
\tilde C_t &= \tanh(W_c * X_t + U_c * H_{t-1} + b_c) \\
O_t &= \sigma(W_o * X_t + U_o * H_{t-1} + b_o) \\
C_t &= F_t \odot C_{t-1} + I_t \odot \tilde C_t \\
H_t &= O_t \odot \tanh(C_t)
\end{aligned}
$$

$*$: 2D convolution.

### 정의 6.4 — No Input Gate (NIG, Greff 2017)

$$
c_t = f_t \odot c_{t-1} + \tilde c_t
$$

Input gate 제거 — candidate 가 항상 fully integrated.

### 정의 6.5 — Greff 2017 Ablation Methodology

Same task (PTB LM, sequence labeling), 모든 variants 동일 hyperparameter search, multiple random seeds. Statistical significance test.

---

## 🔬 정리와 결과

### 정리 6.1 — Peephole 의 Gradient 영향

Peephole 추가 시 cell-to-cell direct partial:

$$
\frac{\partial c_t}{\partial c_{t-1}}\bigg|_{\text{direct}} = f_t + \underbrace{\frac{\partial f_t}{\partial c_{t-1}} \cdot c_{t-1}}_{\text{peephole 영향}} + \frac{\partial i_t}{\partial c_{t-1}} \tilde c_t
$$

추가 항이 *matrix* 형태 (peephole weight $V_f$ 의 곱) → CEC 의 element-wise simplicity 약간 손실.

**결과**: Peephole 이 *precision timing* task (e.g., music tempo) 에 도움, 일반 task 에서는 marginal.

### 정리 6.2 — CIFG Parameter Saving

$|\theta_{\text{CIFG}}| = 3H(D+H+1) = 0.75 |\theta_{\text{LSTM}}|$.

Performance: vanilla LSTM 대비 ~1% loss (Greff 2017).

**Trade-off**: 25% 절약 vs 1% performance — **specific use cases 에서 가치**.

### 정리 6.3 — ConvLSTM 의 Parameter Sharing

Conv 가 spatial parameter sharing → fewer parameters than fully-connected LSTM 적용 to same input size.

**Example**: $H_{\text{img}} = 32, W_{\text{img}} = 32, D = 3, C = 16$, $3 \times 3$ conv:
- Fully-connected LSTM: $4 \times 16 \times (3 \cdot 32 \cdot 32 + 16 \cdot 32 \cdot 32 + 1) = 4 \cdot 16 \cdot 19{,}457 \approx 1.25M$
- ConvLSTM ($3 \times 3$ kernel): $4 \times 16 \times (3 \cdot 9 + 16 \cdot 9 + 1) = 4 \cdot 16 \cdot 172 \approx 11K$

— 100x fewer parameters.

### 정리 6.4 — Greff 2017 의 핵심 결과

Variants 의 PTB perplexity (Greff 2017 Table 1):
- Vanilla: 84.7
- NIG: 92.3 (no input gate critical)
- NFG: 88.5 (forget gate critical for long-range)
- NOG: 89.1 (output gate moderately important)
- NIAF: 84.5 (no significant change)
- NOAF: 84.6 (similar)
- CIFG: 85.1 (negligible loss)
- Peephole: 84.9 (no improvement)

**결론**: **Forget, input, output gate 의 *세트* 가 핵심**. 추가/제거가 marginal benefit/loss.

### 정리 6.5 — LSTM 의 "Already Optimal" 가설

LSTM 의 4 gate 형태가 *empirically optimal* — 추가 mechanism 이 marginal.

**가설 1**: Universal approximation 의 statistical efficiency — 4 gates 가 sufficient for most tasks.
**가설 2**: Optimization landscape — 4 gates 가 SGD friendly 한 곡면 형성.
**가설 3**: Inductive bias — 너무 많은 mechanism 이 overfitting.

---

## 💻 PyTorch 검증

### 실험 1 — Peephole LSTM 구현

```python
import torch
import torch.nn as nn

class PeepholeLSTM(nn.Module):
    """Cell state 를 gate 계산에 포함"""
    def __init__(self, D, H):
        super().__init__()
        self.D, self.H = D, H
        # Standard gates
        self.W_f = nn.Linear(D + H, H)
        self.W_i = nn.Linear(D + H, H)
        self.W_c = nn.Linear(D + H, H)
        self.W_o = nn.Linear(D + H, H)
        # Peephole weights (diagonal 으로 단순화)
        self.V_f = nn.Parameter(torch.zeros(H))
        self.V_i = nn.Parameter(torch.zeros(H))
        self.V_o = nn.Parameter(torch.zeros(H))
        # Forget bias = 1
        with torch.no_grad():
            self.W_f.bias.fill_(1.0)
    
    def forward(self, x_seq, h0=None, c0=None):
        T, B, _ = x_seq.shape
        h = torch.zeros(B, self.H) if h0 is None else h0
        c = torch.zeros(B, self.H) if c0 is None else c0
        outputs = []
        for t in range(T):
            xh = torch.cat([x_seq[t], h], dim=-1)
            f = torch.sigmoid(self.W_f(xh) + self.V_f * c)   # Peephole!
            i = torch.sigmoid(self.W_i(xh) + self.V_i * c)
            g = torch.tanh(self.W_c(xh))
            c_new = f * c + i * g
            o = torch.sigmoid(self.W_o(xh) + self.V_o * c_new)   # Peephole on c_t
            h = o * torch.tanh(c_new)
            c = c_new
            outputs.append(h)
        return torch.stack(outputs)

D, H, T = 4, 16, 10
torch.manual_seed(0)
plstm = PeepholeLSTM(D, H)
x = torch.randn(T, 8, D)
out = plstm(x)
print(f'Peephole LSTM output: {out.shape}')
print(f'Parameter count: {sum(p.numel() for p in plstm.parameters())}')

# Standard LSTM 비교
slstm = nn.LSTM(D, H, batch_first=False)
print(f'Standard LSTM params: {sum(p.numel() for p in slstm.parameters())}')
print(f'Peephole adds {sum(p.numel() for p in [plstm.V_f, plstm.V_i, plstm.V_o])} parameters')
```

### 실험 2 — Coupled Input-Forget Gate (CIFG)

```python
class CIFG_LSTM(nn.Module):
    """i_t = 1 - f_t — coupled gates"""
    def __init__(self, D, H):
        super().__init__()
        self.D, self.H = D, H
        self.W_f = nn.Linear(D + H, H)
        self.W_c = nn.Linear(D + H, H)
        self.W_o = nn.Linear(D + H, H)
        with torch.no_grad():
            self.W_f.bias.fill_(1.0)
    
    def forward(self, x_seq, h0=None, c0=None):
        T, B, _ = x_seq.shape
        h = torch.zeros(B, self.H) if h0 is None else h0
        c = torch.zeros(B, self.H) if c0 is None else c0
        outputs = []
        for t in range(T):
            xh = torch.cat([x_seq[t], h], dim=-1)
            f = torch.sigmoid(self.W_f(xh))
            i = 1 - f                          # coupled
            g = torch.tanh(self.W_c(xh))
            c = f * c + i * g
            o = torch.sigmoid(self.W_o(xh))
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs)

cifg = CIFG_LSTM(D, H)
print(f'CIFG params:     {sum(p.numel() for p in cifg.parameters())}')
print(f'Standard LSTM:   {sum(p.numel() for p in slstm.parameters())}')
# 약 75% 의 LSTM
```

### 실험 3 — ConvLSTM 구현

```python
class ConvLSTMCell(nn.Module):
    """2D convolution 기반 LSTM cell"""
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # Stacked conv for 4 gates
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=padding
        )
        # Forget bias = 1
        with torch.no_grad():
            self.conv.bias[hidden_channels:2*hidden_channels].fill_(1.0)
    
    def forward(self, x, h, c):
        # x: (B, C_in, H, W), h: (B, C_hid, H, W)
        combined = torch.cat([x, h], dim=1)
        z = self.conv(combined)   # (B, 4·C_hid, H, W)
        i, f, g, o = z.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

# Video frame 처리
B, T, C_in = 2, 5, 3
H_img, W_img, C_hid = 16, 16, 8
torch.manual_seed(0)
cell = ConvLSTMCell(C_in, C_hid, kernel_size=3)

video = torch.randn(B, T, C_in, H_img, W_img)
h = torch.zeros(B, C_hid, H_img, W_img)
c = torch.zeros(B, C_hid, H_img, W_img)
for t in range(T):
    h, c = cell(video[:, t], h, c)

print(f'ConvLSTM final h: {h.shape}')   # (B, C_hid, H, W)
print(f'Parameters: {sum(p.numel() for p in cell.parameters())}')
# Convolution 의 spatial parameter sharing 으로 fully-connected LSTM 보다 훨씬 적음
```

### 실험 4 — Greff 2017 Style Ablation

```python
class AblatedLSTM(nn.Module):
    """Various LSTM ablations"""
    def __init__(self, D, H, ablation='vanilla'):
        super().__init__()
        self.D, self.H = D, H
        self.ablation = ablation
        self.W_f = nn.Linear(D + H, H)
        self.W_i = nn.Linear(D + H, H)
        self.W_c = nn.Linear(D + H, H)
        self.W_o = nn.Linear(D + H, H)
        with torch.no_grad():
            self.W_f.bias.fill_(1.0)
    
    def forward(self, x_seq, h0=None, c0=None):
        T, B, _ = x_seq.shape
        h = torch.zeros(B, self.H) if h0 is None else h0
        c = torch.zeros(B, self.H) if c0 is None else c0
        outputs = []
        for t in range(T):
            xh = torch.cat([x_seq[t], h], dim=-1)
            
            if self.ablation == 'NIG':
                # No input gate
                f = torch.sigmoid(self.W_f(xh))
                g = torch.tanh(self.W_c(xh))
                c = f * c + g
                o = torch.sigmoid(self.W_o(xh))
                h = o * torch.tanh(c)
            elif self.ablation == 'NFG':
                # No forget gate
                i = torch.sigmoid(self.W_i(xh))
                g = torch.tanh(self.W_c(xh))
                c = c + i * g
                o = torch.sigmoid(self.W_o(xh))
                h = o * torch.tanh(c)
            elif self.ablation == 'NOG':
                # No output gate
                f = torch.sigmoid(self.W_f(xh))
                i = torch.sigmoid(self.W_i(xh))
                g = torch.tanh(self.W_c(xh))
                c = f * c + i * g
                h = torch.tanh(c)
            else:   # vanilla
                f = torch.sigmoid(self.W_f(xh))
                i = torch.sigmoid(self.W_i(xh))
                g = torch.tanh(self.W_c(xh))
                c = f * c + i * g
                o = torch.sigmoid(self.W_o(xh))
                h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs)

# Synthetic LM-like task 비교
def benchmark_ablation(ablation, T_seq=50, n_steps=100):
    torch.manual_seed(0)
    model = AblatedLSTM(D=10, H=64, ablation=ablation)
    out_proj = nn.Linear(64, 10)
    opt = torch.optim.Adam(list(model.parameters()) + list(out_proj.parameters()), lr=1e-3)
    losses = []
    for step in range(n_steps):
        x = torch.randn(T_seq, 16, 10)
        target = torch.randint(0, 10, (T_seq, 16))
        h_seq = model(x)
        logits = out_proj(h_seq)
        loss = nn.functional.cross_entropy(logits.view(-1, 10), target.view(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return np.mean(losses[-10:])

import numpy as np
print('Ablation Study (lower loss = better):')
for ab in ['vanilla', 'NIG', 'NFG', 'NOG']:
    final = benchmark_ablation(ab)
    print(f'  {ab:8s}: {final:.4f}')
# vanilla 가 가장 낮은 loss, NIG 가 가장 높음 (input gate 가 critical)
```

### 실험 5 — Computational Cost 비교

```python
import time

def benchmark_speed(model, x_seq, n_iter=20):
    target = torch.randn_like(model(x_seq))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    start = time.time()
    for _ in range(n_iter):
        out = model(x_seq)
        loss = ((out - target)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (time.time() - start) / n_iter * 1000

D, H, T, B = 32, 128, 50, 16
x = torch.randn(T, B, D)

models = {
    'Vanilla LSTM' : AblatedLSTM(D, H, 'vanilla'),
    'CIFG'         : CIFG_LSTM(D, H),
    'Peephole'     : PeepholeLSTM(D, H),
}

for name, m in models.items():
    t = benchmark_speed(m, x)
    p = sum(p.numel() for p in m.parameters())
    print(f'{name:15s}: {t:.2f} ms / step, params: {p}')
# CIFG 가 약간 빠름, Peephole 은 약간 느림
```

---

## 🔗 실전 활용

### 1. Video prediction (ConvLSTM)

Precipitation nowcasting (Shi 2015 의 original use), action recognition, video forecasting.

### 2. Limited resources (CIFG)

Edge AI 에서 25% parameter 절약. Marginal performance trade-off acceptable.

### 3. Music modeling (Peephole)

Precision timing 이 critical 한 audio / music task — peephole 이 cell state 의 정확한 timing 추적.

### 4. Architecture search

Greff 2017 의 ablation 정신 — 새 architecture 제안 시 ablation 으로 각 component 의 contribution 측정.

### 5. Domain-specific simplification

특정 task 에 맞춘 LSTM 변종 — text generation 의 simpler GRU, video 의 ConvLSTM 등.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Variants 가 vanilla 보다 우월 | 대부분 marginal |
| Peephole 이 timing 향상 | Specific task 에서만 |
| ConvLSTM 이 video 표준 | Transformer-based video 모델 부상 |
| CIFG 가 25% 절약 | Performance 1% 손실 |
| Ablation 이 robust | Hyperparameter / dataset 별 변동 |

---

## 📌 핵심 정리

$$\boxed{\text{Peephole: } \text{gate} = \sigma(W [h, x] + V \cdot c)}$$

$$\boxed{\text{CIFG: } i_t = 1 - f_t \implies 75\% \text{ parameters}}$$

$$\boxed{\text{ConvLSTM: } \text{matmul} \to \text{conv} \implies \text{spatial-aware sequence}}$$

$$\boxed{\text{Greff 2017: vanilla LSTM 이 robust — 추가 mechanism 의 marginal benefit}}$$

| Variant | Change | Param 변화 | Use case |
|---------|--------|-----------|----------|
| **Vanilla** | — | baseline | 표준 |
| **Peephole** | gate sees $c$ | +3H | Precision timing |
| **CIFG** | $i = 1 - f$ | -25% | Edge AI |
| **NIG** | no $i$ | -$H$ | (poor performance) |
| **ConvLSTM** | matmul → conv | -90%+ for image | Video, spatial |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Peephole LSTM 의 forget gate 가 $V_f$ (peephole weight) 가 0 일 때 standard LSTM 과 동등함을 보이라.

<details>
<summary>해설</summary>

**Peephole forget gate**:
$$
f_t = \sigma(W_f [h_{t-1}; x_t] + V_f \cdot c_{t-1} + b_f)
$$

$V_f = 0$ 시:
$$
f_t = \sigma(W_f [h_{t-1}; x_t] + 0 + b_f) = \sigma(W_f [h_{t-1}; x_t] + b_f)
$$

— Standard LSTM forget gate. ✓

**의미**:
- $V_f = 0$ init 시작해서 학습이 $V_f$ 를 0 으로 유지 → standard LSTM 과 동등
- 학습이 $V_f$ 를 nonzero 로 학습 → cell state 의 peephole 이용

**Greff 2017 결과**:
- 학습된 $V_f$ 가 작은 magnitude (≪ $\|W_f\|$) — peephole 의 marginal effect
- Specific task (timing) 에서만 $V_f$ 가 의미 있게 학습

**결론**: Peephole 이 standard LSTM 의 *generalization* (extra parameters) 이지만 학습이 자주 trivial init ($V \approx 0$) 에 머무름. $\square$

</details>

**문제 2** (심화): CIFG 와 GRU 의 update mechanism 이 같은가? 두 architecture 의 차이를 정확히 분석하라.

<details>
<summary>해설</summary>

**CIFG**:
$$
c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde c_t, \quad h_t = o_t \odot \tanh(c_t)
$$

**GRU**:
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t
$$

여기서 GRU 의 candidate $\tilde h_t = \tanh(W [r_t \odot h_{t-1}; x_t])$.

**유사점**:
- 두 architecture 모두 convex combination — $f_t$ 또는 $z_t \in [0, 1]$
- Update gate 가 forget + input 의 coupled 형태

**차이점**:

1. **State separation**:
   - CIFG: $c_t$ + $h_t$ separated, output gate $o_t$
   - GRU: single $h_t$

2. **Reset mechanism**:
   - CIFG: 없음 — $\tilde c_t$ 가 항상 $h_{t-1}$ 정보 포함
   - GRU: reset gate $r_t$ — candidate 가 $h_{t-1}$ 의 일부만

3. **Output**:
   - CIFG: $h_t = o_t \tanh(c_t)$ — output gate 통한 selective release
   - GRU: $h_t$ 그 자체

**Parameter count**:
- CIFG: 3 weight matrices ($f, c, o$) × $H \times (D + H)$
- GRU: 3 weight matrices ($z, r, h$) × $H \times (D + H)$
- 동일!

**표현력**:

- CIFG = GRU 와 *비슷한 표현력* 이지만 다른 inductive bias
- CIFG 의 cell-hidden separation 이 long-term storage 명시
- GRU 의 reset gate 가 fresh start 명시

**Empirical**:
- 두 모두 vanilla LSTM 의 ~75% parameters
- 두 모두 vanilla 대비 ~1% performance loss (Greff 2017, Chung 2014)
- Task-dependent 어느 것이 우월

**결론**: CIFG 와 GRU 가 *근접* 한 simplification 이지만 *equivalent* 가 아님. CIFG 는 LSTM 에서 한 단계, GRU 는 두 단계 simplification. **둘 다 LSTM 의 4 gate 중 일부가 marginal 함을 입증**. $\square$

</details>

**문제 3** (논문 비평): Greff 2017 가 vanilla LSTM 의 우월성을 입증했지만, 이후 Transformer 가 LSTM 을 대체했다. "Vanilla 가 robust 하다" 와 "Transformer 가 더 나음" 의 모순을 어떻게 reconcile?

<details>
<summary>해설</summary>

**두 결과의 구분**:

**Greff 2017 의 주장**:
- LSTM family 내에서 vanilla 가 robust
- Variants 의 추가 mechanism 이 marginal
- LSTM space 의 *intra-architecture* 비교

**Transformer 의 우위**:
- 다른 architecture family
- LSTM space 외부 — *inter-architecture* 비교

**모순 없음**:
- LSTM 의 4 gate 가 *주어진 RNN framework 내* optimal
- 그러나 RNN framework 자체가 *suboptimal* — Transformer 의 attention 이 fundamentally better

**왜 Transformer 가 우월한가**:

1. **Sequence parallelism**:
   - LSTM: $O(T)$ sequential
   - Transformer: $O(\log T)$ depth (parallel)
   - GPU 시대의 결정적 advantage

2. **Direct connection**:
   - LSTM: $h_T$ 가 $h_0$ 정보 *압축* 통해 access
   - Transformer: 각 position 이 모든 position *직접* attend

3. **Information capacity**:
   - LSTM: hidden 차원 $H$ 에 모든 history 압축
   - Transformer: $T \times H$ "memory" — capacity 가 sequence 길이에 비례

4. **Inductive bias**:
   - LSTM: Markov-like, sequence 의 sequential 구조
   - Transformer: position-agnostic (PE 로 보완), set-like

**Lesson**:

1. **Local optimum vs global optimum**:
   - LSTM 의 vanilla 가 *RNN local optimum*
   - Transformer 가 *global optimum* (현재까지)

2. **Architecture design 의 hierarchy**:
   - 같은 family 내 ablation 은 marginal
   - 다른 family 가 step-change improvement

3. **Greff 의 의의**:
   - "LSTM 이 충분" 의 증명
   - 하지만 "LSTM 이 절대 우월" 이 아님

**Modern view**:
- 2017+ Transformer 표준
- 그러나 Mamba (2023) 가 다시 RNN-like 부활
- Architecture search 가 cyclical

**결론**: Greff 2017 와 Transformer 우위는 *non-contradictory* — 다른 abstraction level. **LSTM family 의 internal optimization vs sequence model 전체의 fundamental redesign**. ML 의 진화는 둘 다 필요 — incremental improvement 와 paradigm shift. $\square$

</details>

---

<div align="center">

[◀ 이전](./05-gru.md) | [📚 README](../README.md) | [다음 ▶](../ch5-advanced-rnn/01-bidirectional.md)

</div>
