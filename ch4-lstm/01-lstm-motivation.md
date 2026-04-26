# 01. LSTM 의 설계 동기

## 🎯 핵심 질문

- Hochreiter & Schmidhuber 1997 의 *Long Short-Term Memory* 가 어떻게 vanishing gradient 의 root cause 를 진단했는가?
- **Constant Error Carousel (CEC)** 의 비전 — Linear self-loop unit 으로 gradient 가 무한히 보존되는 메커니즘
- 왜 단순한 "memory cell" 만으로는 부족하며 forget/input/output gate 가 추가되어야 했는가?
- LSTM 이 plain RNN 의 곱셈적 감쇠를 어떻게 *additive* update 로 변환하는가?
- LSTM 이 vanishing 을 해결하지만 *완전히* 해결하지는 않는 이유 — gate saturation 의 영향

---

## 🔍 왜 LSTM 이 RNN history 의 결정적 도약인가

Hochreiter 의 1991 thesis (vanishing gradient 의 첫 진단) 와 1997 LSTM 논문은 RNN 의 근본 한계와 그 해법을 동시에 제시했습니다. 이 흐름이 모든 modern sequence model 의 출발:

1. **Vanishing 의 root cause 식별** — Jacobian 곱의 곱셈적 누적
2. **Architectural 해법** — Linear self-loop 가 곱셈을 덧셈으로 변환
3. **Gating 메커니즘** — 정보의 selective preservation/erasure
4. **Long short-term memory** — short-term (hidden) + long-term (cell) 의 통합
5. **Modern descendants** — GRU, Highway, ResNet, Transformer 모두 같은 정신

이 문서는 LSTM 의 *motivational story* — 어떤 문제를 어떻게 해결하려 했는지 — 를 추적하고, 다음 문서 (02 의 정확한 수식) 로 이어집니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch3-01 Spectral 분석](../ch3-vanishing-exploding/01-spectral-analysis.md), [Ch3-05 IRNN](../ch3-vanishing-exploding/05-irnn.md)
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Chain rule, Jacobian
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Spectral radius, matrix product

---

## 📖 직관적 이해

### Vanishing 의 핵심 원인 재정리

Plain RNN 의 BPTT (Ch2-02):

$$
\frac{\partial L_t}{\partial W_{hh}^{(k)}} \propto \prod_{j=k+1}^{t} W_{hh}^\top \mathrm{diag}(\sigma'(z_j))
$$

이 *곱* 이 모든 문제의 원인:
- $\rho < 1$: exponential vanishing
- $\rho > 1$: exponential exploding
- $\rho = 1$: edge of chaos, 거의 유지 불가

**핵심 통찰**: 곱셈적 누적이 본질적 문제. *덧셈적 누적* 으로 바꾸면 해결.

### Linear Self-Loop 의 비전

이상적인 RNN 의 update:

$$
c_t = c_{t-1} + (\text{new info from } x_t)
$$

즉 *additive*. 그러면:

$$
\frac{\partial c_t}{\partial c_{t-1}} = 1
$$

곱이 아니라 단순 덧셈 — 어떤 $T$ 에서도 gradient 보존!

### Constant Error Carousel (CEC)

Hochreiter 의 비전: 정보가 "carousel" 처럼 cell 안을 돌며 유지. 외부에서 *write* 하거나 *read* 할 수 있지만, 단순 storage 는 perfect.

```
       ┌────────────────────┐
       │                    │
       │    cell c_t  ←──── + ←── new info
       │       │            │
       │       └────────────┤
       │           (x1 self-loop = identity)
       └────────────────────┘
```

### 왜 단순 Memory 만으로는 부족한가

Pure linear self-loop:
$$
c_t = c_{t-1} + W x_t
$$

문제:
1. **선택성 부족** — 모든 $x_t$ 가 동일하게 합산
2. **Forget 불가** — 무한히 누적
3. **Read 안 됨** — $c$ 자체가 output 이면 noise 까지 포함
4. **Saturation** — $c$ 가 unbounded 증가

**해법**: Gate 도입.

### Gate 의 역할

Sigmoid gate $\in [0, 1]$:
- $0$: 닫힘 (block)
- $1$: 열림 (pass)
- 학습으로 결정

3 gates:
- **Forget**: "이전 cell 정보를 얼마나 유지?"
- **Input**: "새 정보를 얼마나 받아들임?"
- **Output**: "cell 정보를 얼마나 외부에 노출?"

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Constant Error Carousel (Idealized)

가장 단순한 형태:

$$
c_t = c_{t-1} + i_t \tilde c_t
$$

$i_t$: input gate, $\tilde c_t$: candidate. **Linear self-loop**.

### 정의 1.2 — LSTM 의 진화 (1997 → modern)

1. **Original LSTM (1997)**: input gate, output gate, no forget gate
2. **Forget gate 추가 (Gers 2000)**: $c_t = f_t c_{t-1} + i_t \tilde c_t$ — controlled forgetting
3. **Peephole (Gers 2002)**: gate 가 cell state 를 봄
4. **Modern LSTM**: 4 gates (forget, input, candidate, output)

**현재 표준** = Gers 2000.

### 정의 1.3 — Gradient Path Analysis

Plain RNN 의 gradient path:
$$
h_T \to h_{T-1} \to \ldots \to h_0
$$

매 transition 이 matrix $W_{hh}$ 의 곱셈.

LSTM 의 cell gradient path:
$$
c_T \to c_{T-1} \to \ldots \to c_0
$$

매 transition 이 *element-wise* $f_t$ 의 곱셈 — *vector* 가 아님.

### 정의 1.4 — Element-wise vs Matrix Product

**Plain RNN**: $\partial h_t / \partial h_{t-1} = J_t \in \mathbb R^{H \times H}$ (matrix)
- 곱이 matrix product, spectral radius 의 $T$-거듭제곱
- Distinct dimensions 가 mixing

**LSTM cell**: $\partial c_t / \partial c_{t-1} = \mathrm{diag}(f_t) \in \mathbb R^{H \times H}$ (diagonal)
- 곱이 element-wise, $\prod f_t$
- 각 dimension 독립

---

## 🔬 정리와 결과

### 정리 1.1 — Plain RNN 의 곱셈적 누적

Plain RNN 에서 $T$ step 의 gradient:

$$
\frac{\partial h_T}{\partial h_0} = \prod_{j=1}^{T} W_{hh}^\top \mathrm{diag}(\sigma'(z_j))
$$

Spectral radius 의 $T$-거듭제곱 → exponential.

(Ch3-01 정리 1.1 의 reformulation)

### 정리 1.2 — LSTM Cell 의 덧셈적 누적

Cell update $c_t = f_t c_{t-1} + i_t \tilde c_t$ 에서:

$$
\frac{\partial c_T}{\partial c_0} = \prod_{j=1}^{T} f_j
$$

(Element-wise)

**증명**: $\partial c_t / \partial c_{t-1} = f_t$ — direct partial derivative. Chain rule 이 element-wise scalar product 로 단순화. $\square$

**의미**: $f_j \approx 1$ 시 product $\approx 1$ — vanishing 없음.

### 정리 1.3 — Constant Error Carousel 의 정확한 의미

$f_t = 1$ for all $t$ (no forget):

$$
c_t = c_{t-1} + i_t \tilde c_t \implies c_T = c_0 + \sum_{j=1}^{T} i_j \tilde c_j
$$

Backward:

$$
\frac{\partial c_T}{\partial c_0} = 1
$$

— 정확히 1, vanishing/exploding 없음. 정보의 *constant flow*.

### 정리 1.4 — LSTM 의 Hidden State 는 여전히 Vanishing

Hidden state $h_t = o_t \odot \tanh(c_t)$:

$$
\frac{\partial h_t}{\partial h_{t-1}} = (\text{multi-path through gates and } c_t)
$$

이는 matrix-form, spectral radius 의 곱 → vanishing 가능.

**결과**: LSTM 이 *cell state path* 의 vanishing 만 해결, hidden state path 는 여전히 영향.

### 정리 1.5 — Gate Saturation 의 역설

$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$. $W_f$ 가 학습으로 커지면 $f_t$ 가 saturated $\{0, 1\}$ 로 push:

- $f_t \approx 1$: perfect preservation, 그러나 selective forget 불가
- $f_t \approx 0$: 매 step reset, long-term memory 잃음

**Trade-off**: Sharp gating (saturated) vs soft gating (interpolation). 학습이 task 에 맞춰 결정.

---

## 💻 구현 검증

### 실험 1 — Plain RNN vs LSTM Cell 의 Gradient Flow

```python
import numpy as np
import matplotlib.pyplot as plt

T = 100

# Plain RNN: ρ = 0.9
def rnn_gradient_norm(T, rho=0.9, sigma_avg=0.5):
    """평균적인 vanishing rate"""
    return [(rho * sigma_avg) ** t for t in range(T+1)]

# LSTM cell: f ≈ 0.99 (거의 1)
def lstm_cell_gradient(T, f_gate=0.99):
    """∂c_T / ∂c_0 = ∏ f_t"""
    return [f_gate ** t for t in range(T+1)]

plt.figure(figsize=(10, 5))
plt.semilogy(rnn_gradient_norm(T, 0.9, 0.5), label='Plain RNN ρ=0.9, σ\'=0.5')
plt.semilogy(rnn_gradient_norm(T, 1.1, 0.5), label='Plain RNN ρ=1.1 (explode)')
plt.semilogy(lstm_cell_gradient(T, 0.95), label='LSTM cell f=0.95')
plt.semilogy(lstm_cell_gradient(T, 0.99), label='LSTM cell f=0.99')
plt.semilogy(lstm_cell_gradient(T, 1.00), label='LSTM cell f=1 (CEC)')
plt.xlabel('Time step')
plt.ylabel('||gradient|| (log)')
plt.title('Plain RNN multiplicative decay vs LSTM additive preservation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cec_comparison.png', dpi=120, bbox_inches='tight')
plt.close()

print('LSTM f=0.99 after T=100:', 0.99**100)
print('Plain RNN ρ=0.9, σ\'=0.5 after T=100:', (0.9*0.5)**100)
# LSTM: 0.366, Plain RNN: 1e-30
```

### 실험 2 — Element-wise vs Matrix Product

```python
H = 50

# Plain RNN Jacobian
W_hh = np.random.randn(H, H) * (1 / np.sqrt(H))
W_hh /= max(abs(np.linalg.eigvals(W_hh)))  # ρ = 1
J_rnn = W_hh   # simplified, σ' = 1

# LSTM cell Jacobian (forget gate)
f = np.random.uniform(0.9, 1.0, H)   # diagonal
J_lstm = np.diag(f)

T = 50
prod_rnn = np.eye(H)
prod_lstm = np.eye(H)
for _ in range(T):
    prod_rnn = J_rnn @ prod_rnn
    prod_lstm = J_lstm @ prod_lstm

print(f'Plain RNN ||J^T|| at T=50: {np.linalg.norm(prod_rnn):.4f}')
print(f'LSTM cell ||diag(f)^T||: {np.linalg.norm(prod_lstm):.4f}')

# Eigenvalue 분포
eig_rnn = np.linalg.eigvals(prod_rnn)
eig_lstm = np.diag(prod_lstm)   # diagonal 의 eigenvalue
print(f'Plain RNN eig magnitudes range: [{abs(eig_rnn).min():.4f}, {abs(eig_rnn).max():.4f}]')
print(f'LSTM cell diag values range:    [{abs(eig_lstm).min():.4f}, {abs(eig_lstm).max():.4f}]')
# Plain RNN: 모드별 mixing — gradient direction 변화
# LSTM: independent dimensions — selective preservation
```

### 실험 3 — CEC 의 정확한 검증 (No Forget Gate)

```python
import torch
import torch.nn as nn

class IdealCEC(nn.Module):
    """Forget gate 없는 단순 cell — Hochreiter 1997 original"""
    def __init__(self, D, H):
        super().__init__()
        self.D, self.H = D, H
        self.W_i = nn.Linear(D + H, H)   # input gate
        self.W_c = nn.Linear(D + H, H)   # candidate
        self.W_o = nn.Linear(D + H, H)   # output gate
    
    def forward(self, x_seq, c0=None, h0=None):
        T, B, _ = x_seq.shape
        c = torch.zeros(B, self.H) if c0 is None else c0
        h = torch.zeros(B, self.H) if h0 is None else h0
        cs = []
        for t in range(T):
            xh = torch.cat([x_seq[t], h], dim=-1)
            i = torch.sigmoid(self.W_i(xh))
            c_tilde = torch.tanh(self.W_c(xh))
            o = torch.sigmoid(self.W_o(xh))
            c = c + i * c_tilde   # ★ NO forget — pure CEC
            h = o * torch.tanh(c)
            cs.append(c)
        return torch.stack(cs), h

torch.manual_seed(0)
model = IdealCEC(D=2, H=50)

# c_T 가 c_0 의 함수로 differentiable
c0 = torch.zeros(1, 50, requires_grad=True)
x = torch.randn(100, 1, 2)
cs, h_T = model(x, c0=c0)
c_T = cs[-1]

# ∂||c_T||^2 / ∂c_0 의 norm
c_T.sum().backward()
print(f'∂c_T / ∂c_0 norm: {c0.grad.norm():.4f} (T=100)')
# CEC: gradient 정확히 보존 (ideally 1)
```

### 실험 4 — Modern LSTM 의 Gradient Flow

```python
class CompleteLSTM(nn.Module):
    """Forget gate 포함 standard LSTM"""
    def __init__(self, D, H):
        super().__init__()
        self.D, self.H = D, H
        self.W_f = nn.Linear(D + H, H)   # forget
        self.W_i = nn.Linear(D + H, H)
        self.W_c = nn.Linear(D + H, H)
        self.W_o = nn.Linear(D + H, H)
    
    def forward(self, x_seq, c0=None, h0=None):
        T, B, _ = x_seq.shape
        c = torch.zeros(B, self.H) if c0 is None else c0
        h = torch.zeros(B, self.H) if h0 is None else h0
        cs = []
        for t in range(T):
            xh = torch.cat([x_seq[t], h], dim=-1)
            f = torch.sigmoid(self.W_f(xh))
            i = torch.sigmoid(self.W_i(xh))
            c_tilde = torch.tanh(self.W_c(xh))
            o = torch.sigmoid(self.W_o(xh))
            c = f * c + i * c_tilde   # ★ forget gate
            h = o * torch.tanh(c)
            cs.append(c)
        return torch.stack(cs), h

torch.manual_seed(0)
model = CompleteLSTM(D=2, H=50)
c0 = torch.zeros(1, 50, requires_grad=True)
x = torch.randn(100, 1, 2)
cs, h_T = model(x, c0=c0)
cs[-1].sum().backward()
print(f'Modern LSTM ∂c_T / ∂c_0 norm (T=100): {c0.grad.norm():.4f}')
# Forget gate 가 partial decay — 그러나 plain RNN 보다 훨씬 강함
```

### 실험 5 — Gate Saturation 의 효과

```python
# 학습 후 forget gate 의 분포 측정
torch.manual_seed(0)
model = CompleteLSTM(D=2, H=50)

# Adding problem 학습 (간단 설정)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(100):
    x = torch.randn(50, 32, 2)
    target = torch.randn(32, 50)
    cs, h = model(x)
    loss = ((h - target)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# Inference 시 forget gate 분포
with torch.no_grad():
    x_test = torch.randn(50, 8, 2)
    h = torch.zeros(8, 50); c = torch.zeros(8, 50)
    f_values = []
    for t in range(50):
        xh = torch.cat([x_test[t], h], dim=-1)
        f = torch.sigmoid(model.W_f(xh))
        i = torch.sigmoid(model.W_i(xh))
        g = torch.tanh(model.W_c(xh))
        o = torch.sigmoid(model.W_o(xh))
        c = f * c + i * g; h = o * torch.tanh(c)
        f_values.append(f.numpy())

f_arr = np.array(f_values)   # (T, B, H)
print(f'Forget gate avg: {f_arr.mean():.4f}')
print(f'Forget gate std: {f_arr.std():.4f}')
print(f'Saturated near 0 ({(f_arr < 0.1).mean()*100:.1f}%)')
print(f'Saturated near 1 ({(f_arr > 0.9).mean()*100:.1f}%)')
print(f'Linear region    ({((f_arr > 0.1) & (f_arr < 0.9)).mean()*100:.1f}%)')
# 학습된 LSTM 의 forget gate 가 task 에 맞춰 분화
```

---

## 🔗 실전 활용

### 1. NLP/MT 의 표준 (2014~2017)

Sutskever 2014 의 Seq2Seq, Bahdanau 2015 의 Attention 모두 LSTM 기반. Vanishing 해결로 긴 문장 처리.

### 2. Speech recognition

Graves 2013 의 LSTM-CTC 가 음성 인식의 새 표준. TIMIT, WSJ 등에서 SOTA.

### 3. Music generation

LSTM 으로 구조화된 sequence (멜로디, 화음) 생성. Magenta, MidiNet.

### 4. Time series forecasting

LSTM 이 ARIMA 보다 우월한 forecasting. Stock prediction, demand forecasting.

### 5. Modern descendants

- **GRU** (Cho 2014): LSTM 의 단순화 (Ch4-05)
- **ResNet** (He 2015): Residual connection 의 정신
- **Highway Network** (Srivastava 2015): Gated residual
- **Transformer**: Attention 으로 대체

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Cell state path 가 dominant | Hidden path 도 학습, vanishing 영향 |
| Forget gate 가 selective | Saturated 시 binary, soft gating 손실 |
| 4 gates 충분 | More gates (peephole, ConvLSTM) 가능 |
| Element-wise gate | Multi-head attention 같은 cross-dim mixing 부족 |
| Sequential | Transformer 의 병렬성 부족 |

---

## 📌 핵심 정리

$$\boxed{\text{Plain RNN: } \frac{\partial h_t}{\partial h_{t-1}} = J_t \;\;(\text{matrix}), \quad \prod \text{ exponential}}$$

$$\boxed{\text{LSTM cell: } \frac{\partial c_t}{\partial c_{t-1}} = f_t \;\;(\text{element-wise}), \quad \prod = \prod f_t}$$

$$\boxed{\text{CEC vision: } f_t = 1 \implies \frac{\partial c_T}{\partial c_0} = 1 \;\;(\text{constant})}$$

| Aspect | Plain RNN | LSTM (cell path) | LSTM (hidden path) |
|--------|-----------|------------------|--------------------|
| **Update** | Multiplicative | Additive | Multiplicative through gates |
| **Jacobian** | Matrix | Diagonal (element-wise) | Matrix (multi-path) |
| **Vanishing** | Strong | Mitigated by $f \approx 1$ | Still possible |
| **Long-range** | Difficult | Possible | Limited |

---

## 🤔 생각해볼 문제

**문제 1** (기초): CEC ($f_t = 1$, no forget gate) RNN 에서 $T = 100$ step 후 $\partial c_T / \partial c_0$ 의 값은? Plain RNN ($\rho = 0.9$, $\sigma' \approx 0.5$) 와 비교.

<details>
<summary>해설</summary>

**CEC**:
$$
\frac{\partial c_T}{\partial c_0} = \prod_{t=1}^{T} f_t = 1^{100} = 1
$$

**Plain RNN**:
$$
\frac{\partial h_T}{\partial h_0} \approx (\rho \sigma')^T = (0.9 \times 0.5)^{100} = 0.45^{100} \approx 7.9 \times 10^{-36}
$$

**차이**: $10^{36}$ 배 — vanishing 의 catastrophic difference.

**의미**: CEC 가 long-range gradient 를 정확히 보존, plain RNN 은 사실상 0. **이것이 LSTM 이 long sequence 학습을 가능하게 한 정확한 이유**. $\square$

</details>

**문제 2** (심화): Forget gate $f_t = 0.99$ 인 LSTM 의 effective decay rate 는? 100 step 후 gradient norm 의 비율은?

<details>
<summary>해설</summary>

**Forget gate effect**:
$$
\frac{\partial c_T}{\partial c_0} = 0.99^{100} \approx 0.366
$$

— 약 37% 보존. CEC ($f = 1$) 의 100% 와 plain RNN 의 거의 0 사이의 sweet spot.

**해석**:
- $f = 1$: perfect preservation, 그러나 selective forget 불가
- $f = 0.99$: 99% gradient 보존 per step, 100 step 후 37%
- $f = 0.95$: 95% per step, 100 step 후 0.6%
- $f = 0.5$: random walk-like, 100 step 후 $10^{-30}$

**Task 별 적정 $f$**:
- 짧은 의존성 (5-10 step): $f \approx 0.5$ 적합
- 중간 (50-100): $f \approx 0.95-0.99$
- 매우 긴 (1000+): $f \approx 0.999+$ 또는 Mamba 같은 SSM

**LSTM 의 학습**: $f$ 가 input-dependent — task 에 맞춰 자동 조정. 이것이 LSTM 의 *adaptive long-term memory* 의 essence.

**Forget bias = 1 효과** (Jozefowicz 2015): 초기 $f \approx \sigma(1) = 0.73$ — Plain Glorot init 의 $f \approx 0.5$ 보다 long-range 친화적. (Ch4-04 에서 자세히)

$\square$

</details>

**문제 3** (논문 비평): LSTM 이 cell state 의 vanishing 만 해결하고 hidden state 의 vanishing 은 그대로다. 그럼에도 LSTM 이 long-range 학습에 효과적인 이유는?

<details>
<summary>해설</summary>

**LSTM 의 dual path**:

1. **Cell state $c$ path**:
   - $c_t = f_t c_{t-1} + i_t \tilde c_t$
   - Long-term storage
   - Gradient flow 가 element-wise scalar product → strong

2. **Hidden state $h$ path**:
   - $h_t = o_t \tanh(c_t)$
   - $h$ 가 다음 step 의 gate 계산에 사용 (recurrent)
   - Gradient flow 가 matrix product → vanishing

**왜 효과적인가**:

1. **Long-term info 가 cell 통해 전파**:
   - 핵심 기억은 $c$ 에 저장
   - $h$ 의 vanishing 영향 적음 (short-term context 제공자)

2. **Output gate 의 selectivity**:
   - $o_t \tanh(c_t)$ 가 *현재 시점에 필요한* 정보만 hidden 으로 release
   - Hidden 은 short-term context, cell 은 long-term

3. **학습된 분업**:
   - Cell: episodic memory (긴 거리 의존)
   - Hidden: working memory (현재 step 의 활용)
   - LSTM 은 두 path 를 *jointly* 학습 — 어떤 정보가 어디로 갈지 task 에 맞춰

**Empirical 증거**:
- Karpathy 2015 "Visualizing LSTM": cell unit 별로 specific concept 추적 (인용 깊이, 코멘트 vs 코드)
- Cell state 의 long-term retention 이 명확히 visualize

**Hidden vanishing 의 *부분적* 영향**:
- 매우 긴 seq (>1000) 에서는 LSTM 도 어려움
- Skip connection (ResNet 정신), Transformer attention 이 추가 해법

**결론**: LSTM 의 효과는 *완전한* vanishing 해결이 아니라 *분업 architecture*. Cell 이 long-term 을 책임지고 hidden 의 short-term 한계를 우회. **부분적 해결이지만 충분히 강력**. Transformer 가 attention 으로 더 근본적 해결을 했지만, LSTM 의 dual-path 정신은 modern architecture (e.g., Mamba 의 selective SSM) 까지 이어짐. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch3-vanishing-exploding/05-irnn.md) | [📚 README](../README.md) | [다음 ▶](./02-lstm-equations.md)

</div>
