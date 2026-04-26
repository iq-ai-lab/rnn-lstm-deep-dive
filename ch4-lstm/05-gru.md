# 05. GRU (Gated Recurrent Unit, Cho 2014)

## 🎯 핵심 질문

- Cho 2014 의 GRU 가 어떻게 LSTM 의 4 gate 를 2 gate (update, reset) 로 단순화했는가?
- Cell state 와 hidden state 의 통합이 왜 가능하며 표현력에 어떤 영향을 주는가?
- GRU 의 update gate $z_t$ 가 LSTM 의 forget + input gate 의 coupled 형태인 이유는?
- **Chung 2014** 의 empirical 비교: GRU 가 LSTM 과 동등 또는 우월한 task 들 — Polyphonic music modeling 등
- LSTM 대비 GRU 의 ~25% parameter 절약과 ~30% 빠른 학습 — 실무 trade-off

---

## 🔍 왜 GRU 가 LSTM 의 자연스러운 대안인가

LSTM 의 4 gate 가 *모두* 필요한가? Cho 2014 는 이 질문에 대답:

1. **Cell-Hidden 통합** — 별도 storage 가 본질적으로 필요한가?
2. **Forget-Input coupling** — 두 gate 가 ($f, i$) 가 zero-sum 이면 충분한가?
3. **Reset gate** — Hidden state 의 일부를 reset 하는 새 메커니즘
4. **Update gate** — Forget + input 의 합쳐진 형태

결과:
- **75% parameters** — LSTM 의 1/4 절약
- **유사한 성능** — Chung 2014 의 systematic comparison
- **빠른 학습** — 적은 parameters → 빠른 SGD 수렴

이 문서는 GRU 의 정확한 수식, LSTM 과의 정량 비교, 그리고 empirical insights 를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [04-gradient-flow.md](./04-gradient-flow.md) — LSTM 의 gradient flow
- [Ch4-02 LSTM equations](./02-lstm-equations.md) — LSTM 의 4 gate
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Sigmoid + tanh 결합

---

## 📖 직관적 이해

### LSTM 4 → GRU 2 Gate 의 simplification

**LSTM**:
```
4 gates: f_t, i_t, c̃_t, o_t
Cell:    c_t = f_t · c_{t-1} + i_t · c̃_t
Hidden:  h_t = o_t · tanh(c_t)
```

**GRU**:
```
2 gates: z_t (update), r_t (reset)
Hidden:  h_t = (1 - z_t) · h_{t-1} + z_t · h̃_t
            (no separate cell state)
```

### Update Gate 의 직관

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t
$$

- $z_t = 0$: $h_t = h_{t-1}$ (full retention)
- $z_t = 1$: $h_t = \tilde h_t$ (full update)
- $z_t \in (0, 1)$: convex combination

이는 LSTM 의 $f_t = 1 - z_t$, $i_t = z_t$ 의 coupled 버전.

### Reset Gate 의 직관

Candidate $\tilde h_t$ 계산 시 이전 hidden 의 일부 reset:

$$
\tilde h_t = \tanh(W [r_t \odot h_{t-1}; x_t])
$$

- $r_t = 1$: 모든 $h_{t-1}$ 정보 사용 (LSTM 의 candidate 와 같음)
- $r_t = 0$: $h_{t-1}$ 무시, $x_t$ 만으로 candidate 생성 (현재 input 에 reset)
- $r_t \in (0, 1)$: partial reset

이는 LSTM 에 없는 메커니즘 — **memoryless candidate** 가능.

### Cell-Hidden 통합

LSTM 의 cell $c$ 와 hidden $h$ 가 다른 *capacity* 와 *role* (long vs short-term).

GRU 는 이를 single $h$ 로 통합:
- Long-term: $z_t \approx 0$ (retention)
- Short-term: $z_t \approx 1$ (update)
- 학습이 dimension 별로 결정

---

## ✏️ 엄밀한 정의

### 정의 5.1 — GRU Equations

$$
\begin{aligned}
z_t &= \sigma(W_z [h_{t-1}; x_t] + b_z) & \text{update gate} \\
r_t &= \sigma(W_r [h_{t-1}; x_t] + b_r) & \text{reset gate} \\
\tilde h_t &= \tanh(W [r_t \odot h_{t-1}; x_t] + b) & \text{candidate} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t & \text{hidden update}
\end{aligned}
$$

### 정의 5.2 — Parameter Count

GRU: 3 weight matrices each $H \times (D + H)$:

$$
|\theta_{\text{GRU}}| = 3 H (D + H) + 3H = 3H(D + H + 1)
$$

LSTM: $4H(D + H + 1)$.

**비율**: GRU = 0.75 × LSTM.

### 정의 5.3 — Hidden Update 의 Convex Combination

$h_t = (1 - z_t) h_{t-1} + z_t \tilde h_t$ 가 element-wise convex combination ($z_t \in [0, 1]^H$).

특성:
- $\|h_t\| \le \max(\|h_{t-1}\|, \|\tilde h_t\|)$
- $\|h_t\| \in [\min, \max]$ — bounded if $h_{t-1}, \tilde h_t$ bounded

### 정의 5.4 — GRU Cell Gradient

$$
\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + (\text{indirect terms via } z_t, r_t, \tilde h_t)
$$

Direct partial: $1 - z_t$. Total: matrix form.

### 정의 5.5 — LSTM-GRU Mapping

LSTM 의 $f_t = 1 - z_t$, $i_t = z_t$ (coupling), no separate cell, no output gate, but with reset gate.

---

## 🔬 정리와 결과

### 정리 5.1 — GRU 의 Parameter Saving

GRU = $0.75 \times$ LSTM parameters.

**증명**: LSTM 4 gates × $(D + H + 1) H$ = $4H(D+H+1)$. GRU 3 gates × $(D + H + 1) H$ = $3H(D+H+1)$. 비율 $3/4$. $\square$

**구체 예시** ($H = 100, D = 50$):
- LSTM: $4 \times 100 \times 151 = 60{,}400$
- GRU: $3 \times 100 \times 151 = 45{,}300$
- 절약: 15,100 parameters (25%)

### 정리 5.2 — GRU 의 Cell Gradient

Direct partial:

$$
\frac{\partial h_t}{\partial h_{t-1}}\bigg|_{\text{direct}} = 1 - z_t
$$

(Element-wise)

**증명**: $h_t = (1 - z_t) h_{t-1} + z_t \tilde h_t$. Direct partial of $(1 - z_t) h_{t-1}$ wrt $h_{t-1}$ = $1 - z_t$. Other terms 의 direct partial = 0 ($z_t, \tilde h_t$ 가 $h_{t-1}$ 에 *indirect* 의존). $\square$

**비교 LSTM**: $\partial c_t / \partial c_{t-1} = f_t$. GRU 의 $1 - z_t$ 가 effectively LSTM 의 forget gate.

### 정리 5.3 — Information Preservation Bound

$z_t \le 1 - \alpha$ for all $t$ ($\alpha > 0$) → information retention $\ge \alpha^T$:

$$
\|\partial h_T / \partial h_0\|_\infty \ge \alpha^T
$$

CEC 와 같은 mechanism.

### 정리 5.4 — Reset Gate 의 효과

Reset gate $r_t$ 는 candidate 생성 시 *hidden state 의 일부 무시*:

$$
\tilde h_t = \tanh(W \begin{pmatrix} r_t \odot h_{t-1} \\ x_t \end{pmatrix})
$$

- $r_t = 0$: $\tilde h_t = \tanh(W [0; x_t])$ — input-only candidate
- $r_t = 1$: $\tilde h_t = \tanh(W [h_{t-1}; x_t])$ — full context

**해석**: "이번 update 에서 이전 정보 얼마나 활용?". LSTM 에는 없는 mechanism.

### 정리 5.5 — Chung 2014 의 Empirical Comparison

Polyphonic music modeling, speech recognition 에서:
- GRU 와 LSTM 이 거의 동일 성능 (within noise)
- GRU 가 약간 빠른 수렴 (~10-30% fewer epochs)
- Task 별 우열 — universal 우열 없음

---

## 💻 구현 검증

### 실험 1 — GRU 바닥부터 NumPy 구현

```python
import numpy as np

class NumPyGRU:
    def __init__(self, D, H, seed=0):
        rng = np.random.RandomState(seed)
        s = np.sqrt(1.0 / H)
        # 3 weight matrices
        self.W_z = rng.randn(H, D + H) * s
        self.W_r = rng.randn(H, D + H) * s
        self.W_h = rng.randn(H, D + H) * s
        self.b_z = np.zeros(H)
        self.b_r = np.zeros(H)
        self.b_h = np.zeros(H)
        self.D, self.H = D, H
    
    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    
    def forward(self, x_seq, h0=None):
        T = len(x_seq)
        h = np.zeros(self.H) if h0 is None else h0
        history = []
        for t in range(T):
            xh = np.concatenate([x_seq[t], h])
            z = self.sigmoid(self.W_z @ xh + self.b_z)
            r = self.sigmoid(self.W_r @ xh + self.b_r)
            
            # Candidate with reset
            xh_reset = np.concatenate([x_seq[t], r * h])
            h_tilde = np.tanh(self.W_h @ xh_reset + self.b_h)
            
            # Update
            h = (1 - z) * h + z * h_tilde
            history.append({'h': h.copy(), 'z': z, 'r': r, 'h_tilde': h_tilde})
        return h, history

# Toy
D, H, T = 4, 8, 5
gru = NumPyGRU(D, H, seed=42)
x_seq = np.random.randn(T, D)
h_T, hist = gru.forward(x_seq)
print(f'h_T = {h_T[:4]}')
print(f'Avg z (update gate): {np.mean([h["z"].mean() for h in hist]):.4f}')
print(f'Avg r (reset gate):  {np.mean([h["r"].mean() for h in hist]):.4f}')
```

### 실험 2 — PyTorch nn.GRU 와 일치 검증

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
torch_gru = nn.GRU(D, H, batch_first=False)

# PyTorch 의 GRU bias 구조: weight_ih = [r, z, h], weight_hh = [r, z, h]
# 우리: z, r, h 순서. 변환 필요!

# 단순 검증을 위해 PyTorch 의 weight 를 우리 NumPy 로 복사
W_ih = torch_gru.weight_ih_l0.detach().numpy()   # (3H, D)
W_hh = torch_gru.weight_hh_l0.detach().numpy()   # (3H, H)
b_ih = torch_gru.bias_ih_l0.detach().numpy()
b_hh = torch_gru.bias_hh_l0.detach().numpy()

# PyTorch order: r, z, h
# 우리 NumPy: 통합된 W_z [H, D+H] format
# 직접 manual forward
def pytorch_gru_manual(x_seq, W_ih, W_hh, b_ih, b_hh, H):
    T = len(x_seq)
    h = np.zeros(H)
    for t in range(T):
        ih = W_ih @ x_seq[t] + b_ih   # (3H,)
        hh = W_hh @ h + b_hh           # (3H,)
        # Split
        i_r, i_z, i_n = np.split(ih, 3)
        h_r, h_z, h_n = np.split(hh, 3)
        
        r = 1 / (1 + np.exp(-(i_r + h_r)))
        z = 1 / (1 + np.exp(-(i_z + h_z)))
        n = np.tanh(i_n + r * h_n)
        h = (1 - z) * n + z * h
    return h

x_np = np.random.randn(T, D)
h_manual = pytorch_gru_manual(x_np, W_ih, W_hh, b_ih, b_hh, H)

x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
h0 = torch.zeros(1, 1, H)
out, h_T_torch = torch_gru(x_t, h0)
print(f'Manual h_T:  {h_manual[:4]}')
print(f'PyTorch h_T: {h_T_torch.squeeze().numpy()[:4]}')
print(f'Match? {np.allclose(h_manual, h_T_torch.squeeze().numpy(), atol=1e-5)}')
```

### 실험 3 — GRU vs LSTM Parameter Count

```python
def count_params(module):
    return sum(p.numel() for p in module.parameters())

H, D = 256, 64
lstm = nn.LSTM(D, H)
gru = nn.GRU(D, H)
rnn = nn.RNN(D, H)

print(f'Parameter count (D={D}, H={H}):')
print(f'  RNN  : {count_params(rnn):,}')
print(f'  GRU  : {count_params(gru):,} ({count_params(gru)/count_params(rnn):.2f}x of RNN)')
print(f'  LSTM : {count_params(lstm):,} ({count_params(lstm)/count_params(rnn):.2f}x of RNN)')
print(f'  GRU/LSTM ratio: {count_params(gru)/count_params(lstm):.2f}')
```

### 실험 4 — Chung 2014 Style Comparison (간단 task)

```python
import time

def benchmark_rnn(model, x, n_iter=50):
    """Forward+backward speed"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.randn_like(model(x)[0])
    
    times = []
    for _ in range(n_iter):
        start = time.time()
        out, _ = model(x)
        loss = ((out - target)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        times.append(time.time() - start)
    return np.median(times) * 1000   # ms

torch.manual_seed(0)
T, B = 100, 32
x = torch.randn(T, B, D)

models = {
    'RNN' : nn.RNN(D, H, nonlinearity='tanh'),
    'GRU' : nn.GRU(D, H),
    'LSTM': nn.LSTM(D, H),
}

print(f'\nBenchmark (T={T}, B={B}):')
for name, model in models.items():
    t = benchmark_rnn(model, x)
    print(f'  {name:5s}: {t:.2f} ms / step')
# RNN 가장 빠름, GRU 가 LSTM 보다 빠름, LSTM 가장 느림
```

### 실험 5 — Adding Problem 에서 GRU 성능

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

class AddingGRU(nn.Module):
    def __init__(self, H=128):
        super().__init__()
        self.rnn = nn.GRU(2, H)
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[-1]).squeeze(-1)

class AddingLSTM(nn.Module):
    def __init__(self, H=128):
        super().__init__()
        self.rnn = nn.LSTM(2, H)
        # forget bias = 1
        with torch.no_grad():
            self.rnn.bias_ih_l0[H:2*H].fill_(1.0)
            self.rnn.bias_hh_l0[H:2*H].fill_(0.0)
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[-1]).squeeze(-1)

def train(model_cls, T_seq, n_steps=100):
    torch.manual_seed(0); np.random.seed(0)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(n_steps):
        x, y = adding_problem(T_seq, 64)
        loss = ((model(x) - y)**2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return np.mean(losses[-10:])

print(f'Adding Problem T=100:')
print(f'  GRU  final MSE: {train(AddingGRU, 100):.4f}')
print(f'  LSTM final MSE: {train(AddingLSTM, 100):.4f}')
# 비슷한 성능, GRU 가 약간 빠른 수렴 가능
```

---

## 🔗 실전 활용

### 1. Resource-constrained 모델

Mobile, edge AI 에서 GRU 의 25% 절약이 critical. PyTorch Mobile, TensorFlow Lite 표준.

### 2. Smaller dataset

데이터가 적을 때 GRU 의 fewer parameters 가 overfitting 회피.

### 3. NMT (단순 구조)

ConvSeq2Seq, ConvolutionalSeq2Seq 의 GRU encoder/decoder. Attention 과 결합.

### 4. RL policy

DRQN 의 LSTM 변종으로 GRU 사용 — episodic update 의 빠른 학습.

### 5. Conditional generation

Audio (WaveNet 의 alternative), text (smaller LM) 에서 GRU 가 표준 baseline.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 2 gates 충분 | Long-range task 에서 LSTM 이 약간 우월 |
| Cell-hidden 통합 | Modular interpretability 손실 |
| Forget-input coupling | Independent control 불가 |
| Reset gate 가 추가 표현력 | Marginal benefit 일 수도 |
| Universal 우월 없음 | Task 별 LSTM vs GRU 선택 필요 |

---

## 📌 핵심 정리

$$\boxed{\begin{aligned} z_t &= \sigma(W_z [h_{t-1}; x_t] + b_z) \\ r_t &= \sigma(W_r [h_{t-1}; x_t] + b_r) \\ \tilde h_t &= \tanh(W [r_t \odot h_{t-1}; x_t] + b) \\ h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t \end{aligned}}$$

$$\boxed{|\theta_{\text{GRU}}| = 3H(D+H+1) = 0.75 \times |\theta_{\text{LSTM}}|}$$

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 4 (f, i, c̃, o) | 2 (z, r) + candidate |
| **States** | $h$ + $c$ | $h$ only |
| **Parameters** | $4H(D+H+1)$ | $3H(D+H+1)$ |
| **Speed** | Slower | ~30% faster |
| **Performance** | Slightly better long-range | Comparable shorter |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GRU 의 $z_t = 0$ 인 dimension 의 hidden update 는?

<details>
<summary>해설</summary>

$z_t = 0$ 시:

$$
h_t = (1 - 0) \cdot h_{t-1} + 0 \cdot \tilde h_t = h_{t-1}
$$

— 완벽한 정보 보존, $\tilde h_t$ 무시.

**LSTM 의 corresponding case**:
- $f_t = 1, i_t = 0$ (coupled in GRU as $z = 0$)
- $c_t = c_{t-1}$, $h_t$ 도 변화 없음 (output gate 영향 별도)

**의미**: GRU 의 $z = 0$ = "이번 step skip", 정보 그대로 통과. CEC 의 GRU 버전. $\square$

</details>

**문제 2** (심화): GRU 의 reset gate $r_t$ 가 LSTM 에 없는 mechanism 이다. Reset 이 학습에 어떤 unique effect 를 주는가?

<details>
<summary>해설</summary>

**Reset gate 의 역할**:

$$
\tilde h_t = \tanh(W [r_t \odot h_{t-1}; x_t] + b)
$$

- $r_t = 0$: candidate 가 $h_{t-1}$ 정보 *완전* 무시 → "fresh start with $x_t$"
- $r_t = 1$: 모든 history 포함

**LSTM 에서 가장 가까운 것**:
- LSTM 의 candidate $\tilde c_t = \tanh(W_c [h_{t-1}; x_t])$ — *항상* $h_{t-1}$ 사용
- 그러나 forget gate $f_t = 0$ 이면 cell 이 reset, candidate 가 dominant

**Reset 의 unique effect**:

1. **Topic shift in dialogue**:
   - 새 topic 시작 시 $r_t \approx 0$ — past context reset
   - LSTM 은 forget gate 로 cell 만 reset, hidden 은 점진적

2. **Phrase boundaries (NMT)**:
   - 문장 끝에서 candidate 가 next sentence 의 fresh start
   - Hierarchical structure 학습 도움

3. **Adversarial robustness**:
   - 잘못된 history 에 stuck 시 reset 가능

4. **Stateful resetting**:
   - Mid-sequence reset event 에 대응

**Empirical 가치**:
- Toy synthetic task 에서 reset 이 크게 중요한 경우 발견
- Real-world NLP 에서 marginal benefit
- Chung 2014 의 ablation: reset gate 의 *isolated* contribution 작음

**LSTM + Reset 의 hybrid**:
- "Reset LSTM" 같은 variant 가 시도되었지만 표준 안 됨
- 대부분 task 에서 LSTM 의 forget gate 만으로 충분

**결론**: Reset gate 는 GRU 의 *theoretical* contribution 이지만 *practical* impact 는 task-specific. 대부분의 경우 LSTM 의 forget gate 가 비슷한 functional role. 그러나 dialogue, hierarchical structure 같은 specific case 에서 reset 의 explicit 표현이 도움. $\square$

</details>

**문제 3** (논문 비평): Chung 2014 의 결론 "GRU 와 LSTM 이 거의 동등" 이 맞는데도 LSTM 이 더 인기인 이유는? 두 architecture 의 historical 과 cultural 측면을 논하라.

<details>
<summary>해설</summary>

**Empirical evidence** (Chung 2014 + 후속 연구):
- Polyphonic music modeling: comparable
- Penn Treebank LM: LSTM 약간 우월 (1-2% PP)
- 언어 모델링 일반: task-specific
- Speech recognition: LSTM 이 표준 (Graves 2013 의 영향)
- NMT: LSTM (Sutskever 2014, Bahdanau 2015)

**LSTM 의 인기 이유**:

1. **Historical priority**:
   - LSTM 1997 vs GRU 2014 — 17년 차이
   - 그동안 LSTM 의 ecosystem (papers, code, tutorials) 형성
   - "표준" 이 된 후 변경 어려움 (path dependence)

2. **Modular interpretability**:
   - 4 gate 가 각각 distinct role — 분석/시각화 용이 (Karpathy 2015)
   - GRU 의 2 gate 가 functional 으로 합쳐짐 — 분리된 설명 어려움

3. **Empirical 안정성**:
   - LSTM 이 long-range task 에서 약간 더 robust
   - Forget bias = 1 같은 known tricks
   - GRU 도 비슷한 tricks 있지만 less standardized

4. **Library standardization**:
   - cuDNN 이 LSTM 우선 최적화
   - Most papers' code 가 LSTM 기반
   - GRU 도 지원되지만 secondary

5. **Academic culture**:
   - Hochreiter & Schmidhuber 의 long lineage
   - LSTM 의 "Long Short-Term Memory" name 의 mnemonic 효과
   - GRU 의 "Gated Recurrent Unit" 이 generic

**GRU 의 상대적 사용처**:
- Mobile / edge AI (parameter 절약 critical)
- Simple baseline (less hyperparameter)
- Educational (simpler structure)
- 일부 paper authors 의 personal preference

**Modern context (2017+)**:
- Both LSTM and GRU 가 Transformer 에 밀림
- 둘 다 NLP 에서 secondary, 주로 RL / streaming setting
- Mamba (2023) 의 부상 — RNN family 자체의 부활, 그러나 LSTM/GRU 보다 simpler (linear)

**Lesson**:

1. **Empirical equivalence** ≠ **adoption equivalence**:
   - Inertia, ecosystem, marketing matter
   - First-mover advantage 강함

2. **Architecture choice 가 *technical* 이 아닌 *sociological***:
   - Citation 가 결정 (popular → more papers → more popular)

3. **Best architecture 가 항상 표준 되지 않음**:
   - VGG vs GoogLeNet, RMSProp vs Adam similar dynamics

**결론**: GRU 와 LSTM 의 empirical 동등에도 LSTM 의 우세는 *technical* 이 아닌 *historical + cultural*. Modern ML engineering 에서 **architecture 선택 시 popularity 가 아닌 task fit 을 봐야 함** — GRU 가 더 적합한 case 가 의외로 많음. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-gradient-flow.md) | [📚 README](../README.md) | [다음 ▶](./06-lstm-variants.md)

</div>
