# 03. Truncated BPTT

## 🎯 핵심 질문

- Truncated BPTT 는 BPTT 의 어떤 항을 잘라내며, 왜 그것이 메모리 절약을 가져오는가?
- Truncation length $k$ 의 선택이 학습 성능에 어떤 영향을 주는가? Karpathy char-RNN 의 $k = 25$ 의 근거는?
- $k$-truncation 의 **bias** 는 무엇이며 unbiased gradient (full BPTT) 와 어떻게 다른가?
- PyTorch 에서 `detach()` 로 어떻게 정확히 구현하는가?
- PTB language modeling 에서 $k$ 별 perplexity 측정 — bias-variance trade-off

---

## 🔍 왜 truncated BPTT 가 실전 표준인가

Full BPTT 는 정확하지만 다음 한계가 있습니다:

1. **메모리 $O(TH)$** — 길이 $T = 10000$ token 의 sequence 에서 GPU 메모리 한계
2. **학습 frequency 의 저하** — Sequence 끝까지 forward 후 backward, episode 길이만큼 한 번 update
3. **Gradient 의 불안정성** — 매우 긴 chain 에서 $\prod J_j$ 의 vanishing/exploding 이 학습 불안정

Truncated BPTT (TBPTT) 는 이를 해결:
- **Memory $O(kH)$**, $k \ll T$
- **매 $k$ step 마다 update** — 더 자주 학습
- **Long-range gradient 의 chain 길이 제한** — 상대적 안정

이 문서에서는 TBPTT 의 정확한 정의, $k$ 의 trade-off, PyTorch 구현을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-bptt-derivation.md](./02-bptt-derivation.md) — Full BPTT 의 정확한 형태
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Stochastic gradient descent, biased gradient
- (선택) 통계학: Estimator bias-variance trade-off

---

## 📖 직관적 이해

### Full BPTT 의 메모리 폭발

$T = 1000$ language modeling 에서:

```
Forward:  h_0 → h_1 → h_2 → ... → h_1000
                                    │
Backward: ← δ_0 ← δ_1 ← ... ← δ_1000
                                    │
                                    L
```

모든 1000 개 hidden state $h_t$ 와 pre-activation $z_t$ 를 보존 — $O(1000 \times H \times \text{floats})$ 메모리.

### Truncation 의 직관

TBPTT($k = 5$):

```
[h_0 ... h_5] ← detach ← [h_5 ... h_10] ← detach ← [h_10 ... h_15] ...
       │                       │                         │
   backward                backward                   backward
   (5 step)                (5 step)                  (5 step)
```

매 5 step 마다 backward, 그 후 hidden state 를 **detach** (gradient flow 차단) 하고 forward 계속.

### Bias 의 본질

Full BPTT 는 모든 path:
$$
\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^{t} (\text{path of length } t-k)
$$

TBPTT 는 마지막 $k$ path 만:
$$
\frac{\partial L_t}{\partial W_{hh}}\bigg|_{\text{TBPTT}} = \sum_{k'=t-k+1}^{t} (\text{path of length } t-k')
$$

**Bias**: 잃어버린 long-range path 가 systematic error.

**그러나**: Vanishing 으로 long-range path 가 어차피 작음 → bias 가 실제로는 작을 수 있음.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Truncated BPTT($k$)

Sequence $x_{1:T}$ 를 $\lceil T/k \rceil$ chunk 로 분할: $[1, k], [k+1, 2k], \ldots$. 각 chunk:

1. **Forward**: 이전 chunk 의 $h$ 를 초기 상태로 (단, **detached** — gradient 차단)
2. **Backward** chunk 내 loss 에 대해 $k$ step 만 BPTT
3. Update weights, 다음 chunk 로

### 정의 3.2 — Truncated Gradient

$$
\frac{\partial L}{\partial W_{hh}}\bigg|_{\text{TBPTT}_k} = \sum_{c} \sum_{t \in c} \sum_{k' = \max(1, t-k+1)}^{t} \big(\prod J_j\big) \cdot (\ldots)
$$

여기서 $c$ 는 chunk index. **Truncation**: $k' \ge t - k + 1$.

### 정의 3.3 — Detach 연산 (PyTorch)

`h.detach()`: $h$ 의 값은 동일하지만 backward 시 $\partial h / \partial \theta = 0$ 처리. Gradient flow 의 boundary.

### 정의 3.4 — Variants of TBPTT

- **TBPTT($k_1, k_2$)** (Williams 1990): 매 $k_2$ step 마다 forward 후, 마지막 $k_1$ step 에 대해 backward. $k_1 = k_2 = k$ 가 일반적.
- **Sliding TBPTT**: Window 가 $k_1$ overlap 하며 sliding — 모든 step 이 균등 update 받음.

---

## 🔬 정리와 결과

### 정리 3.1 — Memory Saving

TBPTT($k$) 의 메모리: $O(kH)$, full BPTT: $O(TH)$. 절약율 $T/k$.

**증명**: Backward 가 chunk 내 $k$ step 에 대해서만 실행되므로 $k$ 개 hidden state 만 보존. Chunk 간에는 detach 로 gradient flow 차단, 이전 chunk 의 activation 은 free 가능. $\square$

**예시**: $T = 1000, k = 25$ → 메모리 40x 절약.

### 정리 3.2 — Truncated Gradient 의 Bias

TBPTT 는 long-range dependency $> k$ 의 gradient 를 **무시**:

$$
\frac{\partial L}{\partial W_{hh}}\bigg|_{\text{full}} - \frac{\partial L}{\partial W_{hh}}\bigg|_{\text{TBPTT}_k} = \sum_{t} \sum_{k' < t-k} \big(\prod_{j=k'+1}^{t} J_j\big) \cdot (\ldots)
$$

**Bound** (under spectral $\rho < 1$):

$$
\|\text{bias}\| \le \sum_{t} \sum_{k' < t-k} \rho^{t - k'} \|\ldots\| = O(\rho^k)
$$

**의미**: Bias 가 $k$ 에 대해 exponentially 감소. $k$ 가 충분히 크면 bias 무시 가능.

### 정리 3.3 — Optimal $k$ Selection

$k$ 가 클수록 bias 작지만 memory/time cost 증가. Optimal $k$ 는:
- **Task 의 의존성 길이**: NER 짧음 ($k \approx 10$), language modeling 길음 ($k \approx 100$)
- **Vanishing rate** $\rho$ : $\rho$ 가 작을수록 작은 $k$ 충분
- **Hardware constraint**: GPU memory

**Karpathy 의 char-RNN ($k = 25$)** — Shakespeare 같은 char-level LM 에서 합리적 trade-off.

### 정리 3.4 — Sliding TBPTT 의 Unbiasedness (가까이 갈수록)

Sliding TBPTT (overlap $k_1 < k_2$, $k_2 = k$) 는 모든 step 이 평균 $k_1 / k$ 비율로 backward 받음 → bias 가 fixed-stride TBPTT 보다 작을 수 있음.

---

## 💻 PyTorch 구현 검증

### 실험 1 — Full BPTT vs TBPTT

```python
import torch
import torch.nn as nn

D, H, V = 32, 64, 1000
torch.manual_seed(0)

class CharRNN(nn.Module):
    def __init__(self, V, D, H):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.cell = nn.RNNCell(D, H, nonlinearity='tanh')
        self.out = nn.Linear(H, V)
    def init_hidden(self, B):
        return torch.zeros(B, self.cell.hidden_size)
    def forward(self, x_seq, h):
        # x_seq: (T, B) of indices
        outs = []
        for t in range(x_seq.size(0)):
            h = self.cell(self.emb(x_seq[t]), h)
            outs.append(self.out(h))
        return torch.stack(outs), h

model = CharRNN(V, D, H)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 가상 corpus
T_total = 200
B = 4
data = torch.randint(0, V, (T_total + 1, B))   # +1 for next-token target
inputs  = data[:-1]    # (T, B)
targets = data[1:]     # (T, B)
```

### 실험 2 — TBPTT($k = 25$) 학습 루프

```python
def train_tbptt(model, inputs, targets, k=25, epochs=3):
    losses = []
    h = model.init_hidden(inputs.size(1))
    for ep in range(epochs):
        ep_loss = 0.0
        for s in range(0, inputs.size(0), k):
            x_chunk = inputs[s:s+k]
            t_chunk = targets[s:s+k]
            
            # ★ Detach hidden — 이전 chunk 의 gradient 차단
            h = h.detach()
            
            logits, h = model(x_chunk, h)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, V), t_chunk.reshape(-1))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * x_chunk.size(0)
        losses.append(ep_loss / inputs.size(0))
        print(f'TBPTT(k={k}) Epoch {ep+1}: avg loss = {losses[-1]:.4f}')
    return losses

train_tbptt(model, inputs, targets, k=25, epochs=2)
```

### 실험 3 — Full BPTT (단일 sequence)

```python
def train_full_bptt(model, inputs, targets, epochs=3):
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        h = model.init_hidden(inputs.size(1))
        logits, _ = model(inputs, h)             # 전체 forward
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1))
        opt2.zero_grad()
        loss.backward()                          # 전체 backward
        opt2.step()
        print(f'Full BPTT Epoch {ep+1}: loss = {loss.item():.4f}')

# 짧은 sequence 에서만 가능
short_inputs = inputs[:100]
short_targets = targets[:100]
train_full_bptt(model, short_inputs, short_targets, epochs=2)
```

### 실험 4 — $k$ 별 Perplexity 측정

```python
def measure_ppl(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        h = model.init_hidden(inputs.size(1))
        total_loss, total_tok = 0, 0
        for s in range(0, inputs.size(0), 50):
            x_chunk = inputs[s:s+50]
            t_chunk = targets[s:s+50]
            logits, h = model(x_chunk, h)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, V), t_chunk.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tok += t_chunk.numel()
    model.train()
    return torch.exp(torch.tensor(total_loss / total_tok)).item()

for k in [5, 25, 100]:
    fresh = CharRNN(V, D, H)
    opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    train_tbptt(fresh, inputs, targets, k=k, epochs=3)
    ppl = measure_ppl(fresh, inputs, targets)
    print(f'k={k}: final PP = {ppl:.2f}')
# 일반적으로 k 증가 → PP 개선 (bias 감소), 그러나 메모리/시간 증가
```

### 실험 5 — Detach 의 Gradient Flow 시각화

```python
# detach 가 gradient 를 어떻게 차단하는지 확인
h = torch.randn(4, H, requires_grad=True)
h_detached = h.detach()

x = torch.randn(4, D)
emb = nn.Linear(D, H)

out1 = emb(x) + h           # h 가 graph 에 포함
out2 = emb(x) + h_detached  # h_detached 는 leaf, grad 없음

print(f'out1.requires_grad: {out1.requires_grad}')
print(f'out2.requires_grad: {out2.requires_grad}')
# 둘 다 True (emb 가 require_grad), 그러나 backward 시:

out1.sum().backward(retain_graph=True)
print(f'h.grad after out1: {h.grad is not None}')  # True

h.grad = None
out2.sum().backward()
print(f'h.grad after out2: {h.grad}')  # None — detach 로 차단됨
```

---

## 🔗 실전 활용

### 1. Karpathy char-RNN

`min-char-rnn.py` 에서 $k = 25$ TBPTT 로 Shakespeare 스타일 character-level 생성. 메모리 효율과 학습 속도의 trade-off.

### 2. PyTorch language model tutorial

`word_language_model` 예제가 TBPTT 표준 구현. `bptt = 35` 가 default — 35-token chunk 마다 update.

### 3. Online learning / streaming RL

Atari DQN 의 LSTM variant (DRQN) — episode 길이 제한으로 TBPTT 활용. Continual learning 에서도 chunk 단위 update.

### 4. Megatron-LM 의 sequence parallelism

Long sequence 를 여러 GPU 에 split — 각 GPU 가 부분 chunk 를 TBPTT. Inter-GPU gradient sync 가 chunk boundary 에서.

### 5. Mixed precision training

Float16 로 TBPTT 하면 메모리 추가 절약. `torch.cuda.amp` 와 함께 사용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 의존성 길이 $\le k$ | 긴 의존성 학습 불가 — $k$ 증가 필요 |
| Vanishing 으로 bias 작음 | LSTM/GRU 가 vanishing 완화 → 더 큰 effective $k$ |
| Detach 로 chunk 독립 | Cross-chunk gradient 무시 — sliding TBPTT 로 완화 |
| Static $k$ | 동적 $k$ (의존성 길이 추정) 가능하지만 표준 아님 |
| Mini-batch 길이 일정 | Variable $T$ 시 padding + masking 추가 |

---

## 📌 핵심 정리

$$\boxed{\text{TBPTT}(k):\; \text{forward } k \text{ steps, backward } k \text{ steps, detach, repeat}}$$

$$\boxed{\text{Memory } O(kH), \quad \text{Bias } O(\rho^k) \text{ exponentially small}}$$

| $k$ | Memory | Bias | Update freq | Typical use |
|------|--------|------|-------------|-------------|
| **5** | 매우 작음 | 큼 | 매우 자주 | Online RL |
| **25** | 작음 | 작음 | 자주 | Char-RNN (Karpathy) |
| **35** | 중간 | 매우 작음 | 보통 | PyTorch LM tutorial |
| **100~200** | 큼 | 거의 0 | 적음 | Long-form LM |
| **$T$ (full)** | $O(TH)$ | 0 | Episode end | 짧은 sequence |

---

## 🤔 생각해볼 문제

**문제 1** (기초): TBPTT($k = 10$) 으로 $T = 1000$ sequence 를 학습할 때 (1) chunk 수, (2) chunk 별 backward step 수, (3) memory 절약율을 계산하라.

<details>
<summary>해설</summary>

(1) **Chunk 수**: $T / k = 1000 / 10 = 100$ chunks
(2) **Backward step / chunk**: $k = 10$ steps
(3) **Memory 절약**:
   - Full BPTT: $O(1000 H)$
   - TBPTT(10): $O(10 H)$
   - 절약율: $1000 / 10 = 100\times$

**Wallclock 비교**:
- Forward time 동일: $O(TH^2)$
- Backward time: 둘 다 $O(TH^2)$ (각 chunk 의 backward 는 $k$ step, 100 chunks → $1000 \cdot H^2$)
- 차이: TBPTT 는 100 update, full BPTT 는 1 update — TBPTT 가 100x 더 자주 학습

$\square$

</details>

**문제 2** (심화): $\rho(W_{hh}) = 0.9$ 인 RNN 에서 TBPTT($k = 25$) 와 TBPTT($k = 100$) 의 bias 비율을 추정하라. Long-range dependency 학습 가능성 측면에서 비교.

<details>
<summary>해설</summary>

**Bias bound** (정리 3.2):

$$
\|\text{bias}\| \lesssim \rho^k = (0.9)^k
$$

- $k = 25$: $(0.9)^{25} \approx 0.0718$ — 7.2% bias
- $k = 100$: $(0.9)^{100} \approx 2.66 \times 10^{-5}$ — 0.003% bias

**비율**: $k = 100$ 의 bias 가 $k = 25$ 의 약 $0.0001$ 배 — 실질적으로 0.

**Long-range 학습**:
- 의존성 거리 $d$ 인 task 에서 $d \le k$ 이어야 학습 가능
- $\rho = 0.9$ + $k = 25$: 25-step dependency 까지 잡지만 그 이후는 무시
- $k = 100$: 100-step 까지

**그러나 vanishing**:
- Plain RNN 에서 $\rho = 0.9$ 면 25 step 후 gradient $0.07$ 배, 100 step 후 $2.6 \times 10^{-5}$ 배
- 즉 $k = 100$ 으로 늘려도 vanishing 으로 효과 없음
- **LSTM 에서는 다름**: CEC 로 $\rho \approx 1$ effective, $k = 100$ 까지 의미

**결론**: Plain RNN 에서 $k$ 를 키우는 것은 메모리만 낭비. LSTM/GRU 와 함께 $k$ 증가가 의미 있음. $\square$

</details>

**문제 3** (논문 비평): Sliding TBPTT (Williams 1990 의 $k_1 < k_2$ variant) 와 fixed-stride TBPTT 의 차이를 설명하라. 왜 PyTorch 표준이 fixed-stride 인가?

<details>
<summary>해설</summary>

**Fixed-stride TBPTT** (PyTorch 표준):
- Stride = window length = $k$
- Chunk 가 disjoint: $[1, k], [k+1, 2k], \ldots$
- 각 step 이 정확히 한 번 backward 받음

**Sliding TBPTT** ($k_2 = $ stride, $k_1 = $ backward window, $k_1 < k_2$):
- Chunk overlap: $[1, k_1], [k_2+1, k_2+k_1], \ldots$ wait, 실제로는:
  - Forward: 매 $k_2$ step
  - Backward: 마지막 $k_1$ step만
- $k_1 < k_2$ 면 overlap 없음, $k_1 > k_2$ 면 overlap

**차이**:
1. **Step coverage**:
   - Fixed: 각 step 정확히 1번
   - Sliding (k_1 > k_2): 각 step 약 $k_1 / k_2$ 번 backward
2. **Edge bias**:
   - Fixed: chunk boundary 의 step 이 short-range gradient 만 받음
   - Sliding: 모든 step 이 동일하게 처리
3. **Computational**:
   - Fixed: $O(T)$ backward step
   - Sliding ($k_1 = 2k_2$): $O(2T)$ backward — 2x 비용

**왜 PyTorch 표준이 fixed-stride**:
1. **단순성**: detach + chunk loop 가 간결
2. **속도**: Sliding 은 추가 비용
3. **실용성**: 실제 사용 시 fixed 도 충분 (LSTM 의 vanishing 완화로)
4. **메모리 일정**: chunk 마다 정확히 $k$ step

**Sliding 이 유용한 경우**:
- 매우 긴 의존성 (long-form LM)
- Edge bias 가 critical 한 task
- Research 환경 (속도 무관)

**결론**: 80% 의 경우 fixed-stride 가 충분, sliding 은 specialized use. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-bptt-derivation.md) | [📚 README](../README.md) | [다음 ▶](./04-complexity.md)

</div>
