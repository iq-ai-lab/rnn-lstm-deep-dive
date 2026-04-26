# 01. Sequence 학습 문제의 정식화

## 🎯 핵심 질문

- Sequence 학습 문제는 input/output 길이에 따라 어떻게 분류되는가? **many-to-one**, **many-to-many synced**, **seq2seq** 의 차이는?
- 각 유형에서 손실 함수는 어떻게 정의되는가? Cross-entropy, sequence-level loss, **CTC loss** 의 사용 시점은?
- 가변 길이 sequence 를 mini-batch 로 처리할 때 **padding · masking · packing** 은 왜 필요한가?
- 평가 지표 (accuracy, perplexity, BLEU, ROUGE, WER) 는 task 별로 어떻게 다른가?
- Sequence 학습이 일반 supervised learning 과 다른 본질적 점은 무엇인가?

---

## 🔍 왜 이 정식화가 sequence 처리에 필수인가

Sequence 학습 모델 (RNN/LSTM/Transformer) 의 모든 설계 결정 — hidden state 의 차원, output 시점, 손실 함수, batch 구성 — 은 결국 **"어떤 형태의 input → output 을 학습하는가"** 에 따라 결정됩니다. 그러나 많은 실무자가 PyTorch `nn.LSTM` 을 사용하면서 다음을 정확히 구분하지 못합니다:

1. **언제 마지막 hidden state $h_T$ 만 쓰고 언제 모든 $h_{1:T}$ 를 쓰는가** — many-to-one (감정 분석) 은 $h_T$ → 분류기, many-to-many (POS tagging) 은 모든 $h_t$ → step-별 분류
2. **Padding 의 영향** — `pack_padded_sequence` 없이 padded sequence 를 LSTM 에 넣으면 padding 이 hidden state 를 오염시키고 backward 에서 gradient 가 padding 토큰까지 흐름
3. **Loss masking** — Padding 위치의 cross-entropy 를 합산하면 sequence length 가 짧은 sample 일수록 loss 가 작아짐 (편향)
4. **Seq2seq 의 두 phase** — Training (teacher forcing, label leak 가능성) vs Inference (autoregressive, exposure bias)

이 문서에서는 sequence 학습 문제를 **유형별로 엄밀히 분류**하고, 각 유형에 적합한 손실 함수와 배치 처리 방식을 정리합니다.

---

## 📐 수학적 선행 조건

- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Cross-entropy loss, softmax, supervised learning 정식화
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Stochastic gradient descent, mini-batch 구성
- 확률론: $p(y \mid x)$, joint distribution $p(x_{1:T}, y_{1:T})$, conditional independence
- (선택) 정보이론: Kullback-Leibler divergence, 엔트로피와 perplexity 의 관계

---

## 📖 직관적 이해

### Sequence 학습의 4가지 유형

```
유형 1. one-to-one         (일반 supervised — sequence 아님)
        x  →  y

유형 2. one-to-many        (image captioning)
        x  →  y₁ y₂ y₃ ... y_S

유형 3. many-to-one        (감정 분석, 시계열 → 한 값 예측)
        x₁ x₂ ... x_T  →  y

유형 4a. many-to-many synced  (POS tagging, frame classification)
        x₁ x₂ ... x_T
        ↓  ↓       ↓
        y₁ y₂ ... y_T

유형 4b. many-to-many unsynced (seq2seq — 번역, 요약)
        x₁ x₂ ... x_T  →  y₁ y₂ ... y_S    (T ≠ S)
```

### 손실 함수의 직관

- **many-to-one**: 마지막 시점에서만 예측 → **단일 cross-entropy**
- **many-to-many synced**: 매 시점 예측 → **시점별 cross-entropy 의 평균** (mask 적용)
- **seq2seq**: decoder 의 매 step 예측 → **decoder 시점별 cross-entropy 의 합/평균**
- **unaligned (음성 인식)**: input/output alignment 미상 → **CTC loss** (모든 가능한 alignment 에 대한 marginalize)

### Padding 의 문제와 Masking

가변 길이 sequence 들을 batch 로 묶을 때, 짧은 sequence 를 가장 긴 길이에 맞춰 0 으로 채웁니다 (padding). 손실 계산 시 padding 위치를 그대로 두면:

```
sample 1 (length 5):   x₁ x₂ x₃ x₄ x₅ |0 |0 |0   ← padding 3개
sample 2 (length 8):   x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈

loss = Σ CE(y_t, ŷ_t) over t = 1..8
       └ sample 1 의 t = 6, 7, 8 의 padding 까지 합산 ←  편향!
```

**Masking** 으로 padding 위치의 loss 를 0 으로 만들고, 분모 (시퀀스 실제 길이의 합) 도 함께 조정해야 정확합니다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Sequence Learning Problem

**Sequence learning problem** 은 다음 형태의 데이터에서 conditional distribution 을 학습하는 supervised learning:

$$
\mathcal D = \{(x^{(n)}_{1:T_n},\, y^{(n)}_{1:S_n})\}_{n=1}^N, \qquad x^{(n)}_t \in \mathcal X,\; y^{(n)}_s \in \mathcal Y
$$

여기서 $T_n$, $S_n$ 은 sample 별로 다를 수 있음. 학습 목표:

$$
\max_\theta \sum_{n=1}^N \log p_\theta(y^{(n)}_{1:S_n} \mid x^{(n)}_{1:T_n})
$$

### 정의 1.2 — Sequence Learning 의 4가지 유형

$T = |x|$, $S = |y|$ 라 할 때:

| 유형 | 조건 | 예시 |
|------|------|------|
| **one-to-one** | $T = S = 1$ | 일반 분류 (sequence 아님) |
| **one-to-many** | $T = 1, S > 1$ | Image captioning, music generation |
| **many-to-one** | $T > 1, S = 1$ | 감정 분석, 시계열 예측 |
| **many-to-many synced** | $T = S, t \mapsto y_t$ | POS tagging, NER, frame classification |
| **many-to-many unsynced (seq2seq)** | $T \ne S$ in general | 번역, 요약, dialogue, ASR (alignment 학습) |

### 정의 1.3 — Cross-Entropy Loss with Masking

가변 길이 batch $\{(x^{(n)}_{1:T_n}, y^{(n)}_{1:T_n})\}$ 에 대해 mask $m^{(n)}_t \in \{0, 1\}$ ($1$: 실제 토큰, $0$: padding):

$$
\mathcal L = -\frac{\sum_{n,t} m^{(n)}_t \log p_\theta(y^{(n)}_t \mid x^{(n)}_{1:t})}{\sum_{n,t} m^{(n)}_t}
$$

분모 normalization 으로 sequence length 편향 제거.

### 정의 1.4 — Seq2Seq Conditional Probability

Encoder $p_\theta(z \mid x_{1:T})$ 와 decoder $p_\theta(y_{1:S} \mid z)$ 로 구성:

$$
p_\theta(y_{1:S} \mid x_{1:T}) = \prod_{s=1}^{S} p_\theta(y_s \mid y_{<s}, x_{1:T})
$$

Autoregressive factorization — 매 step 의 output 이 다음 step 의 input 이 됨.

### 정의 1.5 — Perplexity

Language modeling 의 표준 평가 지표:

$$
\mathrm{PP}(W) = p_\theta(w_{1:T})^{-1/T} = \exp\!\left(-\frac{1}{T} \sum_{t=1}^T \log p_\theta(w_t \mid w_{<t})\right)
$$

낮을수록 좋음. **$\mathrm{PP} = 100$** 은 평균적으로 다음 단어를 100개 candidates 중 하나로 좁힘.

### 정의 1.6 — BLEU Score

Machine translation 평가, $n$-gram precision 의 기하평균에 brevity penalty:

$$
\mathrm{BLEU} = \mathrm{BP} \cdot \exp\!\left(\sum_{n=1}^{4} w_n \log p_n\right)
$$

$p_n$ 은 modified $n$-gram precision, $\mathrm{BP} = \min(1, e^{1 - r/c})$.

---

## 🔬 정리와 결과

### 정리 1.1 — Likelihood 분해의 일반성

Sequence learning 의 likelihood 는 항상 chain rule 로 분해 가능:

$$
p_\theta(y_{1:S} \mid x) = \prod_{s=1}^{S} p_\theta(y_s \mid y_{<s}, x)
$$

**증명**: 확률의 multiplicative chain rule (조건부 결합확률의 정의) 로부터 직접. $\square$

**의미**: Autoregressive 모델 (RNN, Transformer decoder) 이 본질적으로 이 분해를 학습. 한 번에 전체 $y_{1:S}$ 를 생성하는 모델 (non-autoregressive MT 등) 은 conditional independence 가정을 추가로 함.

### 정리 1.2 — Padding 에 의한 Loss 편향

Padding 위치의 cross-entropy 를 mask 없이 합산하면:

$$
\mathcal L_{\text{biased}} = -\sum_{n,t=1}^{T_{\max}} \log p(y^{(n)}_t \mid x^{(n)}_{1:t})
$$

여기서 $t > T_n$ 인 항은 padding 토큰의 loss 로, **짧은 sequence sample 이 더 큰 padding loss 를 가져 학습이 편향**됨.

**증명** (직관): $\mathbb E[\mathcal L_{\text{biased}}] \propto T_{\max}$ 가 되어, batch 별 평균에서 가장 긴 sequence 의 loss 가 지배. Mask normalize 시 $\mathbb E[\mathcal L] = $ per-token loss 로 수렴. $\square$

### 정리 1.3 — Perplexity 와 Cross-Entropy 의 관계

$$
\log_2 \mathrm{PP}(W) = -\frac{1}{T} \sum_{t=1}^T \log_2 p(w_t \mid w_{<t}) = H(p_{\text{data}}, p_\theta)
$$

즉 perplexity 의 로그는 데이터 분포와 모델 분포 사이의 cross-entropy. **$\mathrm{PP} \to 1$** ⇔ 모델이 다음 단어를 결정론적으로 맞춤 (over-fit 의 신호일 수도).

### 정리 1.4 — Teacher Forcing 의 Exposure Bias

Training: $p_\theta(y_s \mid y^{\text{true}}_{<s}, x)$ — 정답 prefix 사용
Inference: $p_\theta(\hat y_s \mid \hat y_{<s}, x)$ — 모델 예측 prefix 사용

**현상**: Training/inference distribution mismatch 로 모델이 본 적 없는 prefix 를 inference 에서 만나 오류 누적. **Scheduled sampling** (Bengio 2015) 으로 완화.

---

## 💻 NumPy / PyTorch 구현 검증

### 실험 1 — 4가지 유형의 모델 출력 차원

```python
import torch
import torch.nn as nn

# 공통 RNN
B, T, D, H, V = 4, 8, 16, 32, 100   # batch, time, in_dim, hidden, vocab
rnn = nn.LSTM(D, H, batch_first=True)

x = torch.randn(B, T, D)
hs, (h_T, c_T) = rnn(x)
print(f'전체 hidden hs:   {hs.shape}')      # (B, T, H) — many-to-many
print(f'마지막 hidden h_T: {h_T.shape}')    # (1, B, H) — many-to-one

# many-to-one 분류
clf_m2o = nn.Linear(H, 5)                   # 5-class
logits_m2o = clf_m2o(h_T.squeeze(0))
print(f'many-to-one logits: {logits_m2o.shape}')   # (B, 5)

# many-to-many synced (POS tagging)
clf_m2m = nn.Linear(H, 50)                  # 50 POS tags
logits_m2m = clf_m2m(hs)
print(f'many-to-many logits: {logits_m2m.shape}')  # (B, T, 50)
```

### 실험 2 — Padding 과 Masking 손실

```python
# 가변 길이 sample
lengths = torch.tensor([5, 8, 3, 7])   # B=4
T_max = lengths.max().item()

# random "logits" + "labels"
torch.manual_seed(0)
logits = torch.randn(4, T_max, V)
labels = torch.randint(0, V, (4, T_max))

# Mask 생성
mask = torch.arange(T_max).unsqueeze(0) < lengths.unsqueeze(1)
mask = mask.float()

# Naive (편향) loss
loss_naive = nn.CrossEntropyLoss(reduction='sum')(
    logits.reshape(-1, V), labels.reshape(-1)) / (B * T_max)

# Masked loss
log_probs = nn.functional.log_softmax(logits, dim=-1)
nll = -log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
loss_masked = (nll * mask).sum() / mask.sum()

print(f'Naive loss  : {loss_naive:.4f}')
print(f'Masked loss : {loss_masked:.4f}')
print(f'두 loss 가 다른 이유: padding 위치의 random logit 의 loss 가 합산되어 편향')
```

### 실험 3 — pack_padded_sequence 로 효율적 처리

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 길이 내림차순 정렬 (PyTorch 1.x 요구사항, 2.x 는 enforce_sorted=False)
sorted_len, idx = lengths.sort(descending=True)
x_sorted = x[idx]

packed = pack_padded_sequence(x_sorted, sorted_len.cpu(), batch_first=True)
out_packed, _ = rnn(packed)
out_padded, out_len = pad_packed_sequence(out_packed, batch_first=True)
print(f'Output (padded back): {out_padded.shape}')   # (B, T_max_in_batch, H)
print('packing 으로 padding 토큰의 LSTM 연산 자체를 skip — gradient 오염 방지')
```

### 실험 4 — Perplexity 계산

```python
# 가상 LM 결과
log_probs = torch.tensor([-1.2, -2.0, -0.8, -1.5, -1.0])   # 토큰별 log p
T = len(log_probs)
ce = -log_probs.mean().item()
ppl = torch.exp(-log_probs.mean()).item()
print(f'Cross-entropy: {ce:.4f} nats')
print(f'Perplexity   : {ppl:.4f}')
# PP = exp(CE) — base e 기준
```

### 실험 5 — Teacher Forcing 의 Exposure Bias 시뮬레이션

```python
# 이상적인 LM (실제로는 학습된 모델 사용)
def lm_step(prev_token):
    return torch.randn(V)   # logit

# Teacher forcing: ground truth 사용
def teacher_forcing_loss(seq):
    loss = 0
    for t in range(1, len(seq)):
        logit = lm_step(seq[t-1])              # 정답 prefix
        loss += nn.functional.cross_entropy(
            logit.unsqueeze(0), seq[t].unsqueeze(0))
    return loss / (len(seq) - 1)

# Free-running: 모델 예측 사용
def free_running_loss(seq):
    loss = 0
    pred = seq[0]
    for t in range(1, len(seq)):
        logit = lm_step(pred)                  # 모델 prefix
        loss += nn.functional.cross_entropy(
            logit.unsqueeze(0), seq[t].unsqueeze(0))
        pred = logit.argmax()
    return loss / (len(seq) - 1)

torch.manual_seed(0)
seq = torch.randint(0, V, (10,))
print(f'Teacher forcing loss: {teacher_forcing_loss(seq):.4f}')
print(f'Free running loss   : {free_running_loss(seq):.4f}')
# 일반적으로 free-running 이 더 큼 (exposure bias)
```

---

## 🔗 실전 활용

### 1. 감정 분석 (many-to-one)

IMDB review → positive/negative. `nn.LSTM` 의 마지막 $h_T$ 또는 attention pooling 을 분류기에 입력. Sequence length 를 256~512 로 truncation 하는 것이 표준.

### 2. POS Tagging / NER (many-to-many synced)

각 단어에 태그 부여. BiLSTM 으로 양방향 context 활용 (Ch5-01), CRF 헤드 추가로 transition 제약 (Lample 2016). CoNLL-2003 NER 표준 벤치마크.

### 3. Neural Machine Translation (seq2seq)

Encoder-Decoder + Attention (Ch6). **Inference**: beam search (beam size 4~10), length normalization 으로 짧은 가설 편향 보정.

### 4. Speech Recognition (seq2seq, alignment 미상)

Audio frames → text. **CTC loss** (Graves 2006) 가 모든 monotonic alignment 에 대해 marginalize. RNN-T 가 LSTM 기반 streaming ASR 의 표준.

### 5. Time Series Forecasting (many-to-one or many-to-many)

주가, 수요 예측 등. 표준 RNN/LSTM 외에 N-BEATS, Temporal Fusion Transformer, **Mamba** (Ch7-04) 가 최신 SOTA.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Discrete time step (균등 간격) | 비균등 간격 (의료 EHR) 에는 Time-LSTM, Neural ODE 필요 |
| Mark distribution 이 stationary | Concept drift, online learning, continual learning 필요 |
| Vocab 이 fixed and finite | OOV 처리 — BPE, SentencePiece subword tokenization |
| Teacher forcing 의 모델 = inference 모델 | Exposure bias — Scheduled sampling, MIXER (RL) |
| Sequence length 가 모델 capacity 에 적합 | Long sequence — Truncated BPTT, Attention, Mamba |

---

## 📌 핵심 정리

$$\boxed{p_\theta(y_{1:S} \mid x_{1:T}) = \prod_s p_\theta(y_s \mid y_{<s}, x_{1:T}) \quad \text{— autoregressive factorization}}$$

$$\boxed{\mathcal L = -\frac{\sum_{n,t} m^{(n)}_t \log p_\theta(y^{(n)}_t \mid \cdot)}{\sum_{n,t} m^{(n)}_t} \quad \text{— masked cross-entropy}}$$

$$\boxed{\mathrm{PP}(W) = \exp\!\Big(-\frac{1}{T}\sum_t \log p_\theta(w_t \mid w_{<t})\Big)}$$

| Task | 출력 형태 | 손실 | 표준 평가 |
|------|----------|------|----------|
| **many-to-one** | $h_T \to y$ | Cross-entropy | Accuracy / F1 |
| **many-to-many synced** | $\{h_t \to y_t\}$ | Sum CE (masked) | Per-token Acc / F1 |
| **seq2seq** | Encoder → Decoder | Sum CE (decoder) | BLEU / ROUGE |
| **Speech (unaligned)** | Audio → text | CTC / RNN-T | WER |
| **Language Model** | $h_t \to w_t$ | NLL | Perplexity |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 길이 $T = (5, 3, 8)$ 인 3개 sample 의 batch 에서 mask matrix 를 구성하라. 이 mask 로 padded loss 를 정확히 normalize 하는 식을 쓰시오.

<details>
<summary>해설</summary>

$T_{\max} = 8$, mask $m \in \{0, 1\}^{3 \times 8}$:

$$
m = \begin{pmatrix} 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 \\ 1 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{pmatrix}
$$

Loss:

$$
\mathcal L = \frac{\sum_{n=1}^{3} \sum_{t=1}^{8} m_{n,t} \cdot \mathrm{CE}(y_{n,t}, \hat y_{n,t})}{\sum_{n,t} m_{n,t}} = \frac{\text{sum CE}}{5+3+8 = 16}
$$

분모 $16$ 이 실제 토큰 수, batch 평균 per-token loss 로 정확한 추정. $\square$

</details>

**문제 2** (심화): Seq2seq 에서 teacher forcing 으로 학습한 모델이 inference 시 성능이 떨어지는 이유 (exposure bias) 를 distribution mismatch 관점에서 설명하라. Scheduled sampling 이 이를 어떻게 완화하는가?

<details>
<summary>해설</summary>

**Distribution mismatch**:
- Training: $p_\theta(y_s \mid y_{<s}^{\text{true}}, x) $ — prefix 가 training data 분포 $p_{\text{data}}$
- Inference: $p_\theta(\hat y_s \mid \hat y_{<s}, x)$ — prefix 가 모델 분포 $p_\theta$

오류가 누적되면 prefix 가 점점 $p_{\text{data}}$ 와 멀어지고, 모델이 본 적 없는 영역에서 generalization 불가 → exposure bias.

**Scheduled sampling (Bengio 2015)**: Training 중 매 step 에 확률 $\epsilon_s$ 로 ground truth $y^{\text{true}}_{s-1}$, $1 - \epsilon_s$ 로 모델 예측 $\hat y_{s-1}$ 을 prefix 로 사용. $\epsilon_s$ 를 학습 진행에 따라 1 → 0 으로 감소시켜 점진적 transition.

**한계**: Bias-variance trade-off — ground truth 비율이 줄어들수록 학습 신호의 variance 증가. **MIXER** (Ranzato 2016) 는 REINFORCE 로 sequence-level reward (예: BLEU) 를 직접 최적화하는 대안. $\square$

</details>

**문제 3** (논문 비평): CTC loss (Graves 2006) 는 alignment 가 미상인 음성 인식에서 사용된다. Cross-entropy (alignment 명시) 와 비교하여 CTC 의 동기와 단점을 설명하라. 왜 Transformer encoder 에 CTC head 를 붙이는 wav2vec 2.0 / Whisper 가 표준이 되었는가?

<details>
<summary>해설</summary>

**CTC 의 동기**:
- 음성 인식에서 audio frame ($T \approx 1000$) 와 text token ($S \approx 50$) 의 alignment 가 미리 주어지지 않음
- CTC 는 모든 monotonic alignment 에 대한 likelihood 를 marginalize: $p(y \mid x) = \sum_{a \in \mathcal A^{-1}(y)} p(a \mid x)$
- Forward-Backward 알고리즘으로 $O(TS)$ 계산

**CTC 의 단점**:
- **Conditional independence 가정**: $p(a \mid x) = \prod_t p(a_t \mid x)$ — output 토큰 간 의존성 모델링 못함 (autoregressive 가 아님)
- **Blank 토큰** 의 도입으로 vocab 확장
- 언어모델 외부 융합 (shallow fusion / deep fusion) 에 의존

**wav2vec 2.0 / Whisper 의 표준화**:
- wav2vec 2.0 — self-supervised pre-training + CTC fine-tuning 으로 데이터 효율성
- Whisper — encoder-decoder + cross-entropy 로 contextual modeling, 대규모 weakly-supervised 학습
- **두 패러다임 공존**: streaming ASR 은 RNN-T (CTC + autoregressive), 정확도 우선은 attention encoder-decoder

따라서 CTC 는 "alignment-free" 의 우아한 해법이지만 **conditional independence 한계** 로 cross-entropy 기반 autoregressive 와 hybrid 사용이 표준. $\square$

</details>

---

<div align="center">

[◀ 이전](../README.md) | [📚 README](../README.md) | [다음 ▶](./02-ngram-lm.md)

</div>
