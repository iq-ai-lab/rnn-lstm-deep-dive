# 01. Bidirectional RNN

## 🎯 핵심 질문

- Schuster & Paliwal 1997 의 Bidirectional RNN 이 어떻게 forward + backward 두 RNN 을 결합하는가?
- $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$ 의 concatenation 이 왜 양방향 context 를 정확히 표현하는가?
- BiLSTM 이 왜 NER, POS tagging 의 표준이 되었는가? 양방향 context 가 sequence labeling 에 결정적인 이유?
- BiLSTM 의 inference 시 **전체 sequence 가 필요** — online / streaming setting 에서의 한계
- **BiLSTM-CRF** (Lample 2016) 가 NER 의 SOTA 를 달성한 메커니즘 — RNN encoder + structured prediction

---

## 🔍 왜 Bidirectional 이 sequence labeling 의 fundamental tool 인가

Sequence labeling task (POS tagging, NER, chunking) 는 각 token 에 label 을 할당. 한 token 의 label 은:

1. **Past context**: 이전 단어들 — forward RNN
2. **Future context**: 이후 단어들 — backward RNN

예: "He saw her duck" — "duck" 가 noun (오리) vs verb (몸을 숙이다) 인지 결정하려면 *후속* 단어 필요. "He saw her duck under the table" 에서 "duck" = verb.

Unidirectional RNN 은 future context 를 못 봄 → BiRNN 이 자연스러운 해법.

이 문서는 BiRNN 의 정확한 정의, BiLSTM 의 NER 응용, 그리고 streaming 의 한계를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch4-06 LSTM Variants](../ch4-lstm/06-lstm-variants.md) — LSTM 표준
- [Graphical Models Deep Dive](https://github.com/iq-ai-lab/graphical-models-deep-dive) — CRF, HMM (BiLSTM-CRF 의 결합)
- 정의: Concatenation, parallel processing

---

## 📖 직관적 이해

### Forward + Backward 의 결합

```
Sequence: x_1  x_2  x_3  x_4  x_5

Forward RNN:
  →h_1 → →h_2 → →h_3 → →h_4 → →h_5
   ↑      ↑      ↑      ↑      ↑
   x_1    x_2    x_3    x_4    x_5

Backward RNN:
  ←h_1 ← ←h_2 ← ←h_3 ← ←h_4 ← ←h_5
   ↑      ↑      ↑      ↑      ↑
   x_1    x_2    x_3    x_4    x_5

Concatenated hidden:
  h_t = [→h_t; ←h_t]  ∈ ℝ^{2H}

Output:
  y_t = W h_t + b   (also at each step)
```

### POS Tagging 예시

문장: "He saw her duck"

Without future context: "duck" 가 noun 인지 verb 인지 모호.

With backward context:
- $\overleftarrow{h}_4$ ("duck" 의 backward hidden) 이 후속 정보 인코딩
- 만약 다음에 "under" 가 오면 → verb signal
- "ate" 같은 동사 다음이면 → noun signal

### NER 의 양방향 의존

"Apple is looking at buying U.K. startup for $1 billion"

- "Apple": 회사 (ORG) 인지 과일 (O) 인지?
- 뒤에 "is looking", "buying" 등 동사 → 회사 가능성
- 앞에 quantifier 또는 article 없음 → proper noun

양방향 context 가 모호성 해소.

### Inference 의 한계

Online streaming (real-time speech recognition) 시:
- Forward RNN: 현재까지의 input 만으로 update 가능 ✓
- Backward RNN: 전체 sequence 후 시작 — *future input 필요* ✗

BiRNN 은 본질적으로 **batch processing** — 전체 sentence 가 input 일 때만 사용.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Bidirectional RNN

Forward RNN 과 backward RNN 의 parallel application:

$$
\begin{aligned}
\overrightarrow{h}_t &= f_{\text{fwd}}(\overrightarrow{h}_{t-1}, x_t) \\
\overleftarrow{h}_t &= f_{\text{bwd}}(\overleftarrow{h}_{t+1}, x_t) \\
h_t &= [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb R^{2H}
\end{aligned}
$$

Forward 가 $1, 2, \ldots, T$ 순으로, backward 가 $T, T-1, \ldots, 1$ 순으로.

### 정의 1.2 — BiLSTM

Forward 와 backward 가 LSTM:

$$
\begin{aligned}
\overrightarrow{h}_t, \overrightarrow{c}_t &= \text{LSTM}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1}, \overrightarrow{c}_{t-1}) \\
\overleftarrow{h}_t, \overleftarrow{c}_t &= \text{LSTM}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1}, \overleftarrow{c}_{t+1}) \\
h_t &= [\overrightarrow{h}_t; \overleftarrow{h}_t]
\end{aligned}
$$

### 정의 1.3 — Output Layer

각 position 의 prediction:

$$
y_t = \text{softmax}(W_y h_t + b_y)
$$

(Sequence labeling) 또는 sentence-level pooling:

$$
y = \text{softmax}\left(\frac{1}{T} \sum_t h_t \cdot W_y\right)
$$

### 정의 1.4 — Parameter Count

BiLSTM = 2 × LSTM:

$$
|\theta_{\text{BiLSTM}}| = 2 \times 4H(D + H + 1) = 8H(D + H + 1)
$$

### 정의 1.5 — BiLSTM-CRF

BiLSTM 의 hidden 위에 CRF (Conditional Random Field):

$$
P(y_{1:T} | x_{1:T}) = \frac{1}{Z} \exp\left(\sum_t \psi(y_t, h_t) + \sum_t \phi(y_t, y_{t-1})\right)
$$

$\psi$: emission score (BiLSTM 출력), $\phi$: transition score (CRF), $Z$: partition function.

---

## 🔬 정리와 결과

### 정리 1.1 — Information Capacity 의 두 배

$h_t \in \mathbb R^{2H}$ — unidirectional RNN 의 두 배 capacity.

**증명**: Forward 와 backward 가 *independent* parameters. Concat 후 dimension = 2H. $\square$

**의미**: 동일 effective capacity 를 위해 unidirectional RNN 은 hidden $2H$ 필요 — BiRNN 이 더 효율적 (parameter 수 같음).

### 정리 1.2 — Bidirectional Context 의 표현력

각 $h_t$ 가 *모든* $x_{1:T}$ 의 정보를 (이론적으로) 포함:

- $\overrightarrow{h}_t$: $x_{1:t}$ 의 information
- $\overleftarrow{h}_t$: $x_{t:T}$ 의 information
- $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$: full sequence info (with overlap at $x_t$)

### 정리 1.3 — Sequence Labeling 에서의 우위

Conditional independence:
$$
P(y_t | x_{1:T}) \ne P(y_t | x_{1:t})
$$

특히 future context 가 informative 한 경우 ($P(y_t | x_{t:T}) \ne P(y_t)$). BiRNN 이 $P(y_t | x_{1:T})$ 직접 모델링.

### 정리 1.4 — Inference Latency

BiRNN inference 시:
- Forward: $O(T)$ time
- Backward: $O(T)$ time
- 총: $O(T)$ — 그러나 *전체 sequence 가 필요*

Streaming 에서: latency = sequence length × per-step time.

### 정리 1.5 — BiLSTM-CRF 의 Decoding

CRF decoding 은 Viterbi:

$$
y^* = \arg\max_y P(y_{1:T} | x_{1:T}) = \arg\max_y \sum_t \psi(y_t, h_t) + \sum_t \phi(y_t, y_{t-1})
$$

Dynamic programming, $O(T \cdot K^2)$ ($K$: label 수).

---

## 💻 PyTorch 구현 검증

### 실험 1 — BiLSTM 기본 구현

```python
import torch
import torch.nn as nn

D, H, T, B = 10, 32, 20, 4
torch.manual_seed(0)

# bidirectional=True 로 BiLSTM
bilstm = nn.LSTM(D, H, bidirectional=True, batch_first=False)
x = torch.randn(T, B, D)
out, (h_T, c_T) = bilstm(x)

print(f'Input:  {x.shape}')
print(f'Output: {out.shape}')              # (T, B, 2H) — 결합된 hidden
print(f'h_T:    {h_T.shape}')              # (2, B, H) — fwd + bwd 마지막
print(f'h_fwd[T-1] vs h_T[0]:  {torch.allclose(out[-1, :, :H], h_T[0])}')
print(f'h_bwd[0]   vs h_T[1]:  {torch.allclose(out[0, :, H:], h_T[1])}')
# Forward 의 마지막 = h_T[0], backward 의 마지막 (= 시간순 첫번째) = h_T[1]
```

### 실험 2 — Forward 와 Backward 의 분리 확인

```python
# Forward 만, Backward 만 LSTM 으로 BiLSTM 재구성
torch.manual_seed(0)
fwd_lstm = nn.LSTM(D, H)
bwd_lstm = nn.LSTM(D, H)

# Forward
out_fwd, _ = fwd_lstm(x)

# Backward: input 역순
x_rev = x.flip(0)
out_bwd_rev, _ = bwd_lstm(x_rev)
out_bwd = out_bwd_rev.flip(0)   # 다시 정상 순서로

# Concatenate
out_manual = torch.cat([out_fwd, out_bwd], dim=-1)   # (T, B, 2H)
print(f'Manual BiLSTM output: {out_manual.shape}')
# 이 manual 구성이 PyTorch 의 nn.LSTM(bidirectional=True) 와 동등 (weight 동일 시)
```

### 실험 3 — POS Tagging Toy Task

```python
# 가상 POS tagging: 모음 = noun, 자음 = verb (단순화)
def make_data(n_samples, T=10, V=26, K=2):
    """V=26 (단어 수), K=2 (POS 수)"""
    inputs = torch.randint(0, V, (T, n_samples))
    targets = torch.zeros(T, n_samples, dtype=torch.long)
    # 짝수 index = label 0, 홀수 = label 1 (toy task)
    for t in range(T):
        targets[t] = inputs[t] % 2
    return inputs, targets

class POSTagger(nn.Module):
    def __init__(self, V, D, H, K, bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.lstm = nn.LSTM(D, H, bidirectional=bidirectional, batch_first=False)
        out_dim = 2*H if bidirectional else H
        self.fc = nn.Linear(out_dim, K)
    def forward(self, x):
        e = self.emb(x)
        h, _ = self.lstm(e)
        return self.fc(h)

V, K = 100, 5
torch.manual_seed(0)
model_uni = POSTagger(V, D, H, K, bidirectional=False)
model_bi  = POSTagger(V, D, H, K, bidirectional=True)

print(f'Uni-LSTM params: {sum(p.numel() for p in model_uni.parameters())}')
print(f'Bi-LSTM  params: {sum(p.numel() for p in model_bi.parameters())}')
# BiLSTM 이 약간 많음 (Linear의 input 차원 차이)

# Train
def train_pos(model, n_steps=50):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(n_steps):
        x, t = make_data(32)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, K), t.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

loss_uni = train_pos(model_uni)
loss_bi  = train_pos(model_bi)
print(f'Uni-LSTM final loss: {loss_uni:.4f}')
print(f'Bi-LSTM  final loss: {loss_bi:.4f}')
# 이 toy task 는 future context 안 필요하므로 차이 적음
# 실제 NER, POS 에서는 BiLSTM 이 우월
```

### 실험 4 — 양방향 Context 가 critical 한 task

```python
# Last token = mid token 인지 예측 (future context 필요)
def context_dependent_task(n_samples, T=10, V=10):
    inputs = torch.randint(0, V, (T, n_samples))
    # Target: 마지막 token 과 mid token 이 같은가? (binary)
    targets = (inputs[T//2] == inputs[-1]).long()
    return inputs, targets

class SentenceClassifier(nn.Module):
    def __init__(self, V, D, H, bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.lstm = nn.LSTM(D, H, bidirectional=bidirectional, batch_first=False)
        out_dim = 2*H if bidirectional else H
        self.fc = nn.Linear(out_dim, 2)
    def forward(self, x):
        e = self.emb(x)
        h, _ = self.lstm(e)
        # Mid hidden 사용 (양방향 context 필요)
        T = x.size(0)
        return self.fc(h[T//2])

torch.manual_seed(0)
m_uni = SentenceClassifier(V, D, H, bidirectional=False)
m_bi  = SentenceClassifier(V, D, H, bidirectional=True)

def train_sc(model, n_steps=100):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(n_steps):
        x, t = context_dependent_task(32)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, t)
        opt.zero_grad(); loss.backward(); opt.step()
    # Test
    with torch.no_grad():
        x, t = context_dependent_task(200)
        acc = (model(x).argmax(-1) == t).float().mean().item()
    return loss.item(), acc

l_uni, a_uni = train_sc(m_uni)
l_bi, a_bi   = train_sc(m_bi)
print(f'Uni-LSTM: loss={l_uni:.4f}, accuracy={a_uni:.4f}')
print(f'Bi-LSTM:  loss={l_bi:.4f}, accuracy={a_bi:.4f}')
# Future context 필요한 task 에서 BiLSTM 이 명확히 우월
```

### 실험 5 — Inference Latency 비교

```python
import time

def measure_inference(model, x, n_iter=20):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        start = time.time()
        for _ in range(n_iter):
            model(x)
        return (time.time() - start) / n_iter * 1000

T_inference = 100
x_test = torch.randint(0, V, (T_inference, 1))   # B=1 streaming setting

uni = POSTagger(V, D, H, K, bidirectional=False)
bi  = POSTagger(V, D, H, K, bidirectional=True)

t_uni = measure_inference(uni, x_test)
t_bi  = measure_inference(bi, x_test)

print(f'Inference time (T={T_inference}, B=1):')
print(f'  Uni-LSTM: {t_uni:.2f} ms')
print(f'  Bi-LSTM:  {t_bi:.2f} ms (~2x because of fwd + bwd)')
print(f'  Streaming feasibility: Uni-LSTM = real-time, Bi-LSTM = sentence-level only')
```

---

## 🔗 실전 활용

### 1. Named Entity Recognition (NER)

BiLSTM-CRF (Lample 2016) 가 CoNLL-2003 의 SOTA. NER 의 ambiguity 가 양방향 context 로 해소.

### 2. Part-of-Speech Tagging

각 단어의 POS 가 양방향 의존. BiLSTM 이 표준 baseline.

### 3. Chunking / Shallow Parsing

Phrase boundary 식별 — past + future context 모두 활용.

### 4. ELMo (Peters 2018)

Pre-trained BiLSTM 의 contextualized word representation. Layer 별 활성화 가 다른 의미 (lower = syntactic, higher = semantic).

### 5. Speech Recognition (offline)

Frame classification 시 양방향 context — *online* 은 unidirectional, *offline* 은 BiLSTM.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 전체 sequence 사용 가능 | Streaming 불가 — Latency-critical 에 부적합 |
| Forward/backward 독립 | 두 RNN 의 *interaction* 학습 불가 — BiTransformer 가 더 강력 |
| 2x parameters | Edge AI 에서 비용 |
| Sequence-level pooling | $h_t$ 의 결합 방식 (concat vs sum) 의 trade-off |
| Sentence boundary 명확 | 긴 document 에서 boundary 정의 모호 |

---

## 📌 핵심 정리

$$\boxed{h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb R^{2H} \quad \text{— full sequence context}}$$

$$\boxed{\overrightarrow{h}_t = f(\overrightarrow{h}_{t-1}, x_t), \quad \overleftarrow{h}_t = f(\overleftarrow{h}_{t+1}, x_t)}$$

$$\boxed{\text{BiLSTM-CRF: emission (BiLSTM) + transition (CRF) for sequence labeling}}$$

| Task | Bidirectional 효과 |
|------|-------------------|
| **POS tagging** | 강함 (단어의 syntactic role) |
| **NER** | 강함 (entity boundary) |
| **Sentiment** | 보통 (sentence-level) |
| **Translation** | Encoder 에서 강함, decoder 는 unidirectional |
| **Streaming ASR** | 불가 (latency) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $H = 100$ BiLSTM 의 parameter 수와 동등 capacity 의 unidirectional LSTM 의 hidden size 를 계산하라.

<details>
<summary>해설</summary>

**BiLSTM** ($H = 100$):
$$
2 \times 4H(D + H + 1) = 8 \times 100 \times (D + 101) \approx 8.08 \times 10^4 (D + 101)
$$

**Unidirectional LSTM** with $H' = 200$ (matching output dim):
$$
4 H' (D + H' + 1) = 4 \times 200 \times (D + 201) = 8 \times 10^2 (D + 201) \approx 1.61 \times 10^5
$$

**비교**:
- BiLSTM: ~8.08 × 10^4 (D + 101) — D=50 시 12.2M
- Uni-LSTM ($H = 200$): ~1.61 × 10^5 — D=50 시 16.1M

BiLSTM 이 약간 적은 parameters 에 더 effective context capacity. 그러나 BiLSTM 은 future context 사용으로 *quality* 가 다름.

**Equivalent context capacity** vs **equivalent parameter count** 가 다름.

$\square$

</details>

**문제 2** (심화): BiLSTM-CRF 가 단순 BiLSTM + softmax 보다 NER 에서 우월한 이유를 explain 하라. CRF 의 transition score 의 역할은?

<details>
<summary>해설</summary>

**BiLSTM + softmax** (per-token classification):
$$
P(y_t | x) = \text{softmax}(W h_t + b)
$$

각 token 이 *독립* — label 간 의존 무시.

**문제**: NER 의 BIO encoding:
- B-PER: person 시작
- I-PER: person 내부
- O: outside
- "I-PER" 이 "B-PER" 또는 "I-PER" 뒤에만 valid (외부 token 뒤에 I-PER 불가)

Per-token softmax 는 "O O I-PER" 같은 invalid 구조 생성 가능.

**CRF transition**:
$$
P(y_{1:T} | x) = \frac{1}{Z} \exp\left(\sum_t \psi(y_t, h_t) + \sum_t \phi(y_t, y_{t-1})\right)
$$

$\phi(y_t, y_{t-1})$: label 간 transition score
- $\phi(\text{O}, \text{I-PER})$ 학습으로 강한 negative — invalid
- $\phi(\text{B-PER}, \text{I-PER})$ 강한 positive — valid

**Viterbi decoding**:
- 가장 likely *sequence* 선택 (per-token 이 아님)
- Constraint 자연스럽게 enforced

**Empirical**:
- Lample 2016: BiLSTM 만 CoNLL F1 = 89%, BiLSTM-CRF F1 = 91% (~2% 향상)
- 특히 *boundary* tagging (B-, I-) 에서 큰 차이

**현대 perspective**:
- Transformer + CRF 가 NER SOTA (2018+)
- LLM (GPT, BERT) 의 sequence tagging — soft attention 이 implicit transition

**결론**: CRF 가 *structured prediction* 의 핵심 — 단순 emission 만으로는 sequence consistency 학습 부족. BiLSTM 의 contextual emission + CRF 의 transition = NER 의 winning combination. $\square$

</details>

**문제 3** (논문 비평): BiLSTM 이 NER 표준이었지만 BERT 가 (also bidirectional) 가 BiLSTM 을 대체했다. BERT 의 bidirectional 이 BiLSTM 과 어떻게 다른가?

<details>
<summary>해설</summary>

**BiLSTM 의 bidirectional**:
- Two *independent* RNNs (forward, backward)
- 각 token 의 representation = concat of two unidirectional contexts
- Two contexts 의 *interaction* 은 이후 layer 에서만

**BERT (Devlin 2018) 의 bidirectional**:
- Single Transformer encoder
- Self-attention 이 *모든* position 을 *동시* 에 attend
- Bidirectional = attention masking 이 future 도 허용 (causal mask 없음)

**결정적 차이**:

1. **Interaction depth**:
   - BiLSTM: shallow concat, 두 path 가 마지막에 결합
   - BERT: deep — 매 layer 의 self-attention 이 양방향 mixing

2. **Information flow**:
   - BiLSTM: 각 path 가 sequential — 거리에 따른 vanishing
   - BERT: direct attention — distance-independent

3. **Pre-training**:
   - BiLSTM: task-specific 학습 (small data)
   - BERT: massive corpus (16GB+), MLM objective

4. **Transfer learning**:
   - BiLSTM: feature extraction (frozen) 또는 task-specific fine-tune
   - BERT: standard fine-tune paradigm

**ELMo (Peters 2018) 의 bridge**:
- Pre-trained BiLSTM (large corpus, LM objective)
- Contextualized embedding extraction
- BiLSTM 의 pre-training potential 입증
- BERT 가 같은 idea 를 Transformer 로 — 더 강력

**왜 BERT 가 BiLSTM 대체**:

1. **Better representation**: deep self-attention > shallow concat
2. **Pre-training scale**: Transformer 가 large data 에서 BiLSTM 보다 효율적 학습
3. **Fine-tuning**: 통일된 paradigm (한 모델, 다양한 task)
4. **Multi-task**: BERT 의 representations 이 NER, sentiment, QA 모두에 강력

**Modern view**:
- BiLSTM-CRF: 2018 이전 NER 표준
- BERT + CRF: 2018-2020 표준
- LLM (GPT-3+) + few-shot: 2020+ paradigm

**Architectural lesson**:

- "Bidirectional" 의 *implementation* 이 결정적 — concat vs deep attention
- Transformer 의 success 가 sequence model 의 reformulation 입증
- 그러나 BiLSTM 의 *idea* — 양방향 context 의 importance — 가 BERT 까지 이어짐

**결론**: BiLSTM 의 bidirectional 정신을 BERT 가 *deeper* 와 *pre-trained* 로 강화. 같은 핵심 idea 의 다른 implementation — 이것이 ML 진화의 패턴. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch4-lstm/06-lstm-variants.md) | [📚 README](../README.md) | [다음 ▶](./02-stacked-rnn.md)

</div>
