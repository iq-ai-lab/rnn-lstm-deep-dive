# 04. RNN 의 동기와 정의

## 🎯 핵심 질문

- RNN 의 recurrent 구조 $h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b)$ 가 왜 가변 길이 sequence 처리를 가능하게 하는가?
- RNN 의 **parameter sharing across time** 이 왜 sequence length $T$ 와 무관한 model size 를 보장하는가?
- Elman network vs Jordan network 의 차이는 무엇이며, 왜 Elman 이 표준이 되었는가?
- RNN 이 어떻게 hidden state $h_t$ 에 **무한한 history** 를 (이론상) 압축할 수 있는가?
- Vanilla RNN 의 limitations (vanishing gradient, training instability) 는 어디서 오며 왜 LSTM (Ch4) 이 필요한가?

---

## 🔍 왜 RNN 이 sequence 학습의 핵심 도약인가

Neural LM (Ch1-03) 의 fixed window 한계를 극복하기 위해 RNN 은 **state machine 의 신경망 구현** 을 도입합니다:

1. **Recurrent connection** $h_{t-1} \to h_t$ — 과거 정보를 hidden state 로 압축, 무한한 lookback 가능
2. **Parameter sharing** $W_{hh}, W_{xh}$ — 모든 time step 에서 동일 weight, sequence length 에 무관한 model size
3. **Universal approximation for sequence functions** (Siegelmann & Sontag 1991) — Turing-complete (이론상)

이 세 가지 통찰은 finite automata, HMM, dynamical system 의 신경망 일반화입니다. 그러나 실제 학습은 vanishing/exploding gradient (Ch3) 와 saturation 으로 어려우며, 이것이 LSTM (Ch4) 으로 이어지는 동기가 됩니다.

이 문서는 RNN 의 정확한 정의와 forward dynamics, parameter sharing 의 이점을 정량화합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [03-neural-lm.md](./03-neural-lm.md) — Embedding, fixed-window 한계
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): MLP, 활성화 함수 ($\tanh$, $\sigma$, ReLU)
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix-vector product, recurrence relation
- (선택) Dynamical system: state transition, fixed point, Lyapunov stability

---

## 📖 직관적 이해

### Recurrent 구조의 핵심

```
        x₁           x₂           x₃         ...      x_T
         │            │            │                    │
         ▼            ▼            ▼                    ▼
       ┌───┐        ┌───┐        ┌───┐                ┌───┐
   h₀→ │RNN│ → h₁ → │RNN│ → h₂ → │RNN│ → h₃  ...  →  │RNN│ → h_T
       └───┘        └───┘        └───┘                └───┘
         │            │            │                    │
         ▼            ▼            ▼                    ▼
        y₁           y₂           y₃                   y_T
```

**같은 RNN cell** 이 매 time step 에서 호출됨 — weight sharing.

### Parameter Sharing 의 이점

Feed-forward 로 sequence 를 처리한다면:

```
[x₁, x₂, ..., x_T]  →  MLP_T  →  output
```

문제: MLP의 input 차원 = $T \times D$ → 파라미터 수가 $T$ 에 비례, 그리고 다른 $T'$ 에는 적용 불가.

RNN 은:

```
h_t = σ(W_hh h_{t-1} + W_xh x_t)   ← 모든 t 에서 같은 W_hh, W_xh
```

파라미터 수 $= O(H^2 + H D)$ — **$T$ 와 무관**. 학습된 RNN 을 임의 길이 sequence 에 적용 가능.

### Hidden State 의 의미

$h_t \in \mathbb{R}^H$ 는 시점 $t$ 까지의 **모든 정보의 압축**:

$$
h_t = \mathrm{compress}(x_1, x_2, \ldots, x_t)
$$

이상적으로는 $h_t$ 만 알면 미래 예측에 충분 (Markov property in the hidden state). 그러나 $H$ 차원 hidden 이 모든 history 를 잡아낼 수 있는가? — 이것이 RNN 의 **표현력 vs 학습성** 의 핵심 질문.

### Elman vs Jordan

```
Elman (1990):                       Jordan (1986):
       y_{t-1}                            y_{t-1}
                                             │ (output → context)
                                             ▼
   x_t →  RNN  → y_t                  x_t →  RNN  → y_t
          ↑                                    ↑
          │                                    │
       h_{t-1}                              y_{t-1}
       (hidden state)                       (previous output)
```

- **Elman**: hidden state $h_{t-1}$ 가 context — 표준
- **Jordan**: 이전 output $y_{t-1}$ 이 context — early speech recognition

Elman 이 표준이 된 이유: hidden state 가 더 풍부한 표현 (output dimension 에 제약되지 않음).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Vanilla RNN (Elman)

Hidden state $h_t \in \mathbb{R}^H$, input $x_t \in \mathbb{R}^D$, output $y_t \in \mathbb{R}^O$:

$$
\begin{aligned}
h_t &= \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h) \\
y_t &= W_{hy}\, h_t + b_y
\end{aligned}
$$

파라미터: $\theta = (W_{hh}, W_{xh}, W_{hy}, b_h, b_y)$. 초기 상태 $h_0$ 는 zero 또는 학습 가능 vector.

### 정의 4.2 — Pre-activation

$$
z_t = W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h, \quad h_t = \tanh(z_t)
$$

$z_t$ 는 BPTT 분석에서 자주 등장 (Ch2-02, Ch3-01).

### 정의 4.3 — Many-to-One vs Many-to-Many

- **Many-to-many synced**: $y_t = W_{hy} h_t$ 매 step
- **Many-to-one**: $y = W_{hy} h_T$ — 마지막만
- **Seq2Seq**: encoder $h_T^{\text{enc}}$ → decoder 초기 상태 (Ch6-01)

### 정의 4.4 — Parameter Count

$$
|\theta| = H^2 + H D + H O + H + O
$$

$T$ (sequence length) 에 **무관** — RNN 의 핵심 이점.

### 정의 4.5 — Computational Graph (Unrolled)

$T$ time step 의 RNN 을 펼치면 $T$-layer feed-forward network 와 동치 (단, weight 공유). Forward 연산:

$$
\text{Forward: } O(T \cdot (H^2 + HD)), \quad \text{Memory: } O(T \cdot H)
$$

(모든 $h_t$ 를 backward 에서 사용하기 위해 보존)

---

## 🔬 정리와 결과

### 정리 4.1 — Parameter Sharing 의 정량 이점

가변 길이 sequence batch $\{(x^{(n)}_{1:T_n}, y^{(n)})\}_{n=1}^N$ ($T_n$ 다양) 을 학습할 때, RNN 의 model size 는 sample 별 sequence length 와 무관.

**증명**: $|\theta| = H^2 + HD + HO + H + O$ 는 $T_n$ 에 의존 안 함. Forward 는 $T_n$ 길이만큼 cell 호출, backward 도 같음. Mini-batch 시 zero-padding + masking. $\square$

**의미**: 
- Train: 평균 길이 50, test: 평균 길이 200 sequence 도 동일 모델
- 비교: Bengio 2003 NLM 은 $n = 5$ 고정, 다른 length 에 적용 불가

### 정리 4.2 — Universality (Siegelmann & Sontag 1991)

**정리**: Polynomially-bounded computation 의 임의 함수가 polynomial-time RNN 으로 표현 가능. 즉 RNN 은 Turing-complete.

**증명** (sketch): RNN 의 hidden state 가 unbounded precision 의 rational number 를 표현할 수 있다고 가정하면, RNN 이 임의의 finite-state machine + counter 를 시뮬레이션. 수렴성은 $\sigma$ 의 saturation 분석에 의존. $\square$

**한계**:
- 무한 정밀도 가정 (실제로는 float32)
- 학습 가능성 (trainability) 는 다른 문제 — vanishing gradient (Ch3) 가 실제 한계

### 정리 4.3 — Hidden State 의 Information Bottleneck

$h_t$ 가 모든 $x_{1:t}$ 의 정보를 보존하려면 $H \ge \log_2 |\mathcal X|^t = t \log_2 |\mathcal X|$ — 시간에 따라 선형 증가 필요.

**증명**: Information theory — distinct sequence 수 $|\mathcal X|^t$ 를 distinct hidden state 로 매핑하려면 $\log_2 |\mathcal X|^t$ bits 필요. $h_t \in \mathbb R^H$ 의 finite precision 에서 표현 가능 sequence 수 $\propto 2^H$. $\square$

**의미**: 충분히 긴 sequence 에서 정보 손실 불가피 — RNN 은 **lossy compression**. 실용적으로 task 에 중요한 정보만 보존하도록 학습.

### 정리 4.4 — Forward Dynamics 의 Boundedness

$\tanh$ activation 시 $\|h_t\|_\infty \le 1$ 항상 보장.

**증명**: $\tanh: \mathbb R \to (-1, 1)$, element-wise → $\|h_t\|_\infty < 1$. $\square$

**대조**: ReLU RNN 은 $h_t$ unbounded → exploding 위험 (Le 2015, IRNN, Ch3-05).

### 정리 4.5 — Fixed Point Analysis

$x_t = x$ (constant input) 가정 시, fixed point $h^* = \tanh(W_{hh} h^* + W_{xh} x + b_h)$ 가 존재 (Brouwer's fixed point theorem, $\tanh$ 가 $[-1, 1]^H$ 에서 continuous mapping).

**Stability**: $h^*$ 가 attractor 이면 RNN 은 결국 $h^*$ 로 수렴 → 초기 정보 망각 (vanishing 의 dynamical system 해석).

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — Vanilla RNN 바닥부터

```python
import numpy as np

class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=0):
        rng = np.random.RandomState(seed)
        D, H, O = input_dim, hidden_dim, output_dim
        scale = np.sqrt(1.0 / H)
        self.Wxh = rng.randn(H, D) * scale
        self.Whh = rng.randn(H, H) * scale
        self.Why = rng.randn(O, H) * scale
        self.bh  = np.zeros(H)
        self.by  = np.zeros(O)
        self.D, self.H, self.O = D, H, O
    
    def forward(self, x_seq, h0=None):
        T = len(x_seq)
        if h0 is None:
            h0 = np.zeros(self.H)
        hs = np.zeros((T+1, self.H))
        ys = np.zeros((T, self.O))
        zs = np.zeros((T, self.H))
        hs[0] = h0
        for t in range(T):
            zs[t] = self.Whh @ hs[t] + self.Wxh @ x_seq[t] + self.bh
            hs[t+1] = np.tanh(zs[t])
            ys[t] = self.Why @ hs[t+1] + self.by
        return ys, hs, zs

# Toy
D, H, O, T = 4, 8, 2, 10
rnn = VanillaRNN(D, H, O)
x_seq = np.random.randn(T, D)
ys, hs, zs = rnn.forward(x_seq)
print(f'Outputs: {ys.shape}')   # (10, 2)
print(f'Hidden : {hs.shape}')   # (11, 8) — h_0 ~ h_T
print(f'||h_t||_inf max = {np.abs(hs).max():.4f}')   # ≤ 1 (tanh bound)
```

### 실험 2 — Parameter Count 검증

```python
total = (rnn.Wxh.size + rnn.Whh.size + rnn.Why.size 
         + rnn.bh.size + rnn.by.size)
expected = D*H + H*H + H*O + H + O
print(f'Param count: {total} = expected {expected}')

# Different sequence length, same model
for T_test in [5, 50, 500, 5000]:
    x_test = np.random.randn(T_test, D)
    ys, _, _ = rnn.forward(x_test)
    print(f'T={T_test}: output shape = {ys.shape}, model still works')
# 같은 RNN 이 다른 길이 sequence 처리
```

### 실험 3 — PyTorch nn.RNN 과 일치 검증

```python
import torch
import torch.nn as nn

torch_rnn = nn.RNN(D, H, batch_first=False, nonlinearity='tanh')
# weight 동기화
with torch.no_grad():
    torch_rnn.weight_ih_l0.copy_(torch.tensor(rnn.Wxh, dtype=torch.float32))
    torch_rnn.weight_hh_l0.copy_(torch.tensor(rnn.Whh, dtype=torch.float32))
    torch_rnn.bias_ih_l0.copy_(torch.tensor(rnn.bh, dtype=torch.float32))
    torch_rnn.bias_hh_l0.zero_()

x_t = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(1)   # (T, 1, D)
hs_torch, h_T_torch = torch_rnn(x_t)
hs_torch = hs_torch.squeeze(1).detach().numpy()

print(f'NumPy   h_T : {hs[1:][−1][:5]}')    # 마지막 hidden
print(f'PyTorch h_T : {hs_torch[-1][:5]}')
print(f'Match? {np.allclose(hs[1:], hs_torch, atol=1e-5)}')
```

### 실험 4 — Hidden State Bottleneck 시각화

```python
# 다른 input sequence 가 hidden state 에 어떻게 영향?
T = 50
x1 = np.random.randn(T, D)
x2 = x1.copy()
x2[0] = -x1[0]   # 첫 step 만 다름

_, hs1, _ = rnn.forward(x1)
_, hs2, _ = rnn.forward(x2)

dist = np.linalg.norm(hs1 - hs2, axis=1)
print(f'Hidden state 차이 시간축:')
for t in [0, 1, 5, 10, 25, 50]:
    print(f't={t:3d}: ||h^1_t - h^2_t|| = {dist[t]:.6f}')
# 초기 차이가 시간에 따라 감소 (vanishing 의 신호) 또는 증가 (exploding)
```

### 실험 5 — Many-to-One vs Many-to-Many 출력

```python
# Many-to-one: 마지막 hidden 만 사용
class M2O_RNN(nn.Module):
    def __init__(self, D, H, K):
        super().__init__()
        self.rnn = nn.RNN(D, H, batch_first=True)
        self.clf = nn.Linear(H, K)
    def forward(self, x):
        _, h_T = self.rnn(x)
        return self.clf(h_T.squeeze(0))

# Many-to-many: 모든 hidden 사용
class M2M_RNN(nn.Module):
    def __init__(self, D, H, K):
        super().__init__()
        self.rnn = nn.RNN(D, H, batch_first=True)
        self.clf = nn.Linear(H, K)
    def forward(self, x):
        hs, _ = self.rnn(x)
        return self.clf(hs)

m2o = M2O_RNN(D, H, K=3)
m2m = M2M_RNN(D, H, K=3)
x = torch.randn(2, T, D)   # B=2
print(f'Many-to-one  out: {m2o(x).shape}')   # (B, K)
print(f'Many-to-many out: {m2m(x).shape}')   # (B, T, K)
```

---

## 🔗 실전 활용

### 1. Time series forecasting

기상, 주가, 수요 예측. Vanilla RNN 은 짧은 lookback 에 적합, LSTM/GRU 가 표준 (Ch4).

### 2. Language modeling

Mikolov 2010 RNN-LM 이 PTB 에서 N-gram 능가. 그러나 vanishing 으로 long-range 한계 — LSTM-LM 이 표준 (Zaremba 2014).

### 3. Speech recognition

Acoustic model 의 frame classification — many-to-many synced. CTC loss 와 결합 (Graves 2006).

### 4. Sentiment classification

IMDB review → many-to-one. 마지막 $h_T$ 또는 attention pool 을 분류기에 입력.

### 5. RL 의 partial observability

POMDP 에서 belief state 를 RNN hidden 으로 표현 (DRQN, Hausknecht 2015).

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Discrete time step | 비균등 간격 → Time-LSTM, Neural ODE |
| $\tanh$ activation | Saturation 의 vanishing → ReLU (불안정), gating (LSTM) |
| Single hidden vector $h_t$ | Long-range 표현 한계 → cell state 분리 (LSTM) |
| Sequential 의존 | GPU 병렬화 불가 → Transformer (Ch7-01) |
| Dense matmul | 대규모 $H$ 에서 비용 → block-diagonal, low-rank |
| Vanilla 학습 어려움 | Vanishing/exploding (Ch3), gradient clipping |

---

## 📌 핵심 정리

$$\boxed{h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h), \quad y_t = W_{hy}\, h_t + b_y}$$

$$\boxed{|\theta| = H^2 + HD + HO + H + O \quad \text{— $T$ 와 무관}}$$

$$\boxed{\|h_t\|_\infty < 1 \;\; (\text{tanh bound}), \quad \text{forward } O(TH^2)}$$

| 측면 | RNN | NLM (Ch1-03) |
|------|-----|------|
| **Context length** | 무한 (이론) | Fixed $n-1$ |
| **Parameter** | $O(H^2 + HD)$ | $O(\|V\|d + nd H)$ |
| **Sequence length** | 가변 | 고정 |
| **Computational** | Sequential | Parallel |
| **표현력** | Turing-complete | Finite-state |
| **학습성** | Vanishing 문제 | 안정 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Vanilla RNN 의 forward 1 step 을 손으로 계산하라. $W_{hh} = \begin{pmatrix} 0.5 & 0.3 \\ -0.2 & 0.8 \end{pmatrix}$, $W_{xh} = \begin{pmatrix} 1.0 & 0.5 \\ -0.5 & 0.0 \end{pmatrix}$, $b_h = (0, 0)^\top$, $h_0 = (0, 0)^\top$, $x_1 = (1, 0)^\top$. $h_1$ 을 구하라.

<details>
<summary>해설</summary>

$$
z_1 = W_{hh} h_0 + W_{xh} x_1 + b_h = 0 + \begin{pmatrix} 1.0 & 0.5 \\ -0.5 & 0.0 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} + 0 = \begin{pmatrix} 1.0 \\ -0.5 \end{pmatrix}
$$

$$
h_1 = \tanh(z_1) = \begin{pmatrix} \tanh(1.0) \\ \tanh(-0.5) \end{pmatrix} = \begin{pmatrix} 0.7616 \\ -0.4621 \end{pmatrix}
$$

**검산**: $\|h_1\|_\infty = 0.7616 < 1$ ✓ (tanh bound). $\square$

</details>

**문제 2** (심화): RNN 의 hidden state 가 모든 history 를 보존하려면 $H \ge t \log_2 |\mathcal X|$ 가 필요하다는 정리 4.3 을 사용하여, $|\mathcal X| = 26$ (alphabet) 이고 $T = 100$ 인 sequence 를 lossless 압축하려면 최소 $H$ 는? Float32 precision 에서 가능한가?

<details>
<summary>해설</summary>

**Information-theoretic 하한**:

$$
H \ge T \log_2 |\mathcal X| = 100 \cdot \log_2 26 \approx 100 \cdot 4.7 = 470 \text{ bits}
$$

**Float32 precision**: 한 차원당 약 23 bits (mantissa). 470 / 23 ≈ **21 차원**.

그러나 이는 **이론적 lower bound** — 실제 RNN 은:
1. Float 의 distinct value 가 $2^{32}$ 가 아니라 effective precision 이 더 낮음 (gradient learning)
2. Activation $\tanh$ 의 bounded range 가 effective bits 를 제한
3. 학습 시 weight 가 information bottleneck 에 도달 못함

**실용**: $H = 100 \sim 500$ 이 일반적, 그러나 정보 손실 (lossy compression).

**RNN 의 본질**: lossless 가 아니라 **task-relevant 정보를 우선 보존** — 학습이 결정. $\square$

</details>

**문제 3** (논문 비평): RNN 의 universality (정리 4.2) 가 실제 학습 가능성을 보장하지 않는 이유를 설명하라. 표현력 vs 학습성의 차이가 deep learning 일반에서 어떻게 나타나는가?

<details>
<summary>해설</summary>

**표현력 vs 학습성**:
- **Universality**: "어떤 함수든 RNN 으로 *표현* 가능" — 존재성
- **Trainability**: "그 함수를 SGD 로 *찾을* 수 있는가" — 알고리즘적

RNN 의 경우:
- Siegelmann-Sontag 정리는 무한 정밀도 + 임의 weight 가정
- 실제 학습은 finite precision + SGD
- **Vanishing gradient** 가 long-range dependency 의 학습을 차단

**Deep learning 일반의 패턴**:
1. **MLP**: Universal approximation theorem (Hornik 1991) 는 hidden unit 수를 충분히 늘리면 임의 continuous function 표현 가능. 그러나 학습은 SGD 의 local minima, saddle point 에 갇힘.
2. **Transformer**: Universal approximator (Yun 2019) 이지만 in-context learning 같은 emergent property 는 데이터/scale 의존.
3. **GAN**: Generator 가 임의 분포 표현 가능하지만 mode collapse, unstable training.

**결론**: 표현력은 **upper bound** 이고 학습성은 **achievable bound**. 둘 사이의 gap 이 모든 architecture 의 핵심 도전:
- LSTM (Ch4): Vanilla RNN 의 학습성 향상
- Residual connection: Deep network 의 학습성
- Adam: SGD 의 수렴성

이 패턴이 ML 의 "no architecture is sufficient without trainability" 의 보편 진리. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-neural-lm.md) | [📚 README](../README.md) | [다음 ▶](../ch2-bptt/01-unrolled-graph.md)

</div>
