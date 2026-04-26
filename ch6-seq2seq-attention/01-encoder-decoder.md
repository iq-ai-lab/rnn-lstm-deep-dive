# 01. Encoder-Decoder Framework (Sutskever 2014)

## 🎯 핵심 질문

- Sutskever 2014 의 *Sequence to Sequence Learning with Neural Networks* 가 어떻게 가변 길이 input → 가변 길이 output 을 처리하는가?
- Encoder LSTM 이 input sequence $x_{1:T}$ 를 fixed vector $v = h_T^{\text{enc}}$ 로 압축하는 메커니즘은?
- Decoder LSTM 이 $v$ 와 이전 output 으로 다음 token 을 *autoregressive* 하게 생성하는 방식?
- **Teacher forcing** vs **scheduled sampling** vs free-running 의 trade-off
- Sutskever 의 **reverse input trick** — input 을 역순으로 넣으면 BLEU 향상하는 이유?

---

## 🔍 왜 Seq2Seq 가 sequence-to-sequence 학습의 paradigm 인가

번역 (translation), 요약 (summarization), dialogue 등은 input 과 output 이 다른 길이 / 모달리티. Sutskever 2014 가 이를 통일된 framework 로:

1. **Encoder**: variable input → fixed vector
2. **Decoder**: fixed vector → variable output
3. **End-to-end**: encoder + decoder 가 하나의 model 로 학습

이는 NLP 의 paradigm shift:
- Pre-2014: Phrase-based MT, IBM Models, statistical alignment
- 2014+: Neural Seq2Seq, deep learning approach

이 문서는 Seq2Seq 의 정확한 architecture, 학습 (teacher forcing), inference (autoregressive generation), 그리고 reverse input trick 을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch5-04 Echo State Network](../ch5-advanced-rnn/04-esn.md), [Ch4 LSTM](../ch4-lstm/01-lstm-motivation.md)
- [Ch1-01 Sequence formulation](../ch1-sequence-basics/01-sequence-formulation.md) — Seq2seq 정식화
- 정의: Autoregressive factorization $p(y_{1:S} | x) = \prod_s p(y_s | y_{<s}, x)$

---

## 📖 직관적 이해

### Encoder-Decoder Architecture

```
Input:    x_1   x_2   x_3   x_4   x_5  <eos>
           ↓     ↓     ↓     ↓     ↓     ↓
Encoder LSTM:  h_1 → h_2 → h_3 → h_4 → h_5 → h_6
                                              │
                                              ▼
                                          context v
                                              │
                                              ▼
Decoder LSTM:  s_1 → s_2 → s_3 → s_4 → s_5 → ...
                ↓     ↓     ↓     ↓     ↓
              y_1   y_2   y_3   y_4   y_5

(Decoder 의 input 은 이전 output: <bos>, y_1, y_2, ...)
```

### Encoder 의 Compression

Encoder 가 *전체* input 을 *single* vector $v = h_T^{\text{enc}}$ 로 압축. 이는:
- **압축 능력**: $|V|^T$ possible inputs → $\mathbb R^H$ vector
- **Bottleneck**: 정보 loss 불가피 (Ch6-02 에서 자세히)

### Decoder 의 Autoregressive Generation

```
t=0:  s_0 = v (or g(v)),  y_0 = <bos>
t=1:  s_1, y_1 = LSTM(s_0, y_0), output(s_1)
t=2:  s_2, y_2 = LSTM(s_1, y_1), output(s_2)
...
t=T:  generate until <eos>
```

매 step 의 output 이 다음 step 의 input — **autoregressive**.

### Teacher Forcing (Training)

학습 시 *ground truth* prefix 사용:
```
Input:   <bos>  y_1*  y_2*  y_3*  ...  (정답)
Output:   y_1   y_2   y_3   y_4   ...  (예측)
```

장점: 학습 빠르고 안정.
단점: Inference 시 모델 prediction 사용 → distribution mismatch (exposure bias).

### Free-Running (Inference)

```
Input:   <bos>  ŷ_1   ŷ_2   ŷ_3   ...  (모델 예측)
Output:   ŷ_1   ŷ_2   ŷ_3   ŷ_4   ...
```

오류 누적 가능.

### Reverse Input Trick

Sutskever 2014: input 을 역순으로 넣으면 BLEU 향상.

```
Original:  x_1, x_2, ..., x_T → encoder → v → decoder → y
Reversed:  x_T, x_{T-1}, ..., x_1 → encoder → v → decoder → y
```

**왜 효과적**: encoder 의 마지막 hidden ($v$) 가 *처음* input $x_1$ 의 정보를 더 잘 보존 — long range dependency 단축.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Encoder-Decoder Seq2Seq

**Encoder**:
$$
h_t^{\text{enc}} = \text{LSTM}_{\text{enc}}(x_t, h_{t-1}^{\text{enc}}), \quad t = 1, \ldots, T
$$
$$
v := h_T^{\text{enc}} \in \mathbb R^H
$$

**Decoder**:
$$
s_0 = v \;\; (\text{or } g(v))
$$
$$
s_t, \ell_t = \text{LSTM}_{\text{dec}}(y_{t-1}, s_{t-1})
$$
$$
p(y_t | y_{<t}, x) = \mathrm{softmax}(W_o \ell_t + b_o)
$$

### 정의 1.2 — Likelihood

$$
p(y_{1:S} | x_{1:T}) = \prod_{s=1}^{S} p(y_s | y_{<s}, x_{1:T})
$$

Autoregressive factorization. $y_0 = $ `<bos>`, generation 은 $y_S = $ `<eos>` 시 종료.

### 정의 1.3 — Training Objective

Cross-entropy on each output position:

$$
\mathcal L = -\sum_{n=1}^{N} \sum_{s=1}^{S_n} \log p_\theta(y_s^{(n)} | y_{<s}^{(n)*}, x^{(n)})
$$

(Teacher forcing: $y_{<s}^*$ ground truth)

### 정의 1.4 — Inference (Greedy)

$$
\hat y_t = \arg\max_y p_\theta(y | \hat y_{<t}, x)
$$

매 step 의 max 선택. Free-running.

### 정의 1.5 — Inference (Beam Search)

Beam size $B$, 매 step 에서 top $B$ candidates 유지:

$$
\text{Beam}_t = \text{top-B}\left\{(\hat y_{1:t-1}, w \cdot p(y_t | \hat y_{<t}, x)) : w \in \text{Beam}_{t-1}, y_t \in V\right\}
$$

(Log-prob accumulation)

---

## 🔬 정리와 결과

### 정리 1.1 — Information Bottleneck

Encoder 의 fixed $v \in \mathbb R^H$ 가 모든 $x_{1:T}$ 정보를 인코딩하려면:

$$
H \cdot \log_2(\text{precision}) \ge T \cdot \log_2(|V|) - \log_2(\text{capacity overhead})
$$

$T \to \infty$ 시 정보 손실 불가피 — Ch6-02 에서 자세히.

### 정리 1.2 — Teacher Forcing 의 Distribution Mismatch

Training: $p(y_t | y_{<t}^*, x)$ — ground truth prefix.
Inference: $p(y_t | \hat y_{<t}, x)$ — model prefix.

$$
\mathbb E_{y^* \sim p_{\text{data}}}[\text{loss}_{\text{train}}] \ne \mathbb E_{\hat y \sim p_\theta}[\text{loss}_{\text{infer}}]
$$

이 gap 이 **exposure bias**.

### 정리 1.3 — Reverse Input Trick 의 효과

Reversed input: encoder 의 마지막 hidden $v$ 가 **처음** input $x_1$ 의 vanishing gradient 영향 받지 않음:

- Original: $x_1 \to h_1 \to h_2 \to \ldots \to h_T = v$, $x_1$ 와 $v$ 사이 거리 $T$
- Reversed: $x_T \to h_1$, $x_1 \to h_T = v$, $x_1$ 와 $v$ 사이 거리 $1$

**Decoder 의 $y_1$ 이 $x_1$ 와 가장 align** (translation 의 monotonic alignment) → reversed 가 short-range dependency.

**Empirical** (Sutskever 2014): Reversed 가 BLEU +4.7 (En→Fr).

### 정리 1.4 — Beam Search 의 Approximation

Beam search 는 exact MAP $\arg\max_y p(y|x)$ 의 approximation. $B = 1$ greedy, $B = |V|^S$ exact.

**Empirical**: $B = 4 \sim 10$ 가 BLEU 의 sweet spot — 더 큰 $B$ 는 marginal.

### 정리 1.5 — Length Normalization

Long sequence 가 더 작은 likelihood (각 token 의 곱). Length normalization:

$$
\text{score}(y) = \frac{1}{|y|^\alpha} \log p(y | x), \quad \alpha \in (0, 1]
$$

$\alpha = 0$: no normalization, $\alpha = 1$: full normalization.

---

## 💻 PyTorch 구현 검증

### 실험 1 — 단순 Seq2Seq Encoder/Decoder

```python
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, V_in, V_out, D, H):
        super().__init__()
        self.V_in, self.V_out = V_in, V_out
        self.emb_in = nn.Embedding(V_in, D)
        self.emb_out = nn.Embedding(V_out, D)
        self.encoder = nn.LSTM(D, H)
        self.decoder = nn.LSTM(D, H)
        self.fc = nn.Linear(H, V_out)
    
    def forward(self, x_seq, y_seq):
        """Teacher forcing training"""
        # Encoder
        x_emb = self.emb_in(x_seq)
        _, (h, c) = self.encoder(x_emb)   # h: (1, B, H)
        
        # Decoder with teacher forcing
        y_emb = self.emb_out(y_seq[:-1])    # input <bos>, y_1, ..., y_{S-1}
        out, _ = self.decoder(y_emb, (h, c))
        logits = self.fc(out)                # (S-1, B, V_out)
        return logits
    
    def generate(self, x_seq, max_len=20, bos_idx=0, eos_idx=1):
        """Greedy generation"""
        x_emb = self.emb_in(x_seq)
        _, (h, c) = self.encoder(x_emb)
        
        B = x_seq.size(1)
        y = torch.full((B,), bos_idx, dtype=torch.long)
        outputs = []
        for t in range(max_len):
            y_emb = self.emb_out(y).unsqueeze(0)
            out, (h, c) = self.decoder(y_emb, (h, c))
            logits = self.fc(out.squeeze(0))
            y = logits.argmax(-1)
            outputs.append(y.clone())
            if (y == eos_idx).all():
                break
        return torch.stack(outputs)

# Test
torch.manual_seed(0)
V_in, V_out, D, H = 100, 80, 32, 64
model = Seq2Seq(V_in, V_out, D, H)
x = torch.randint(0, V_in, (10, 4))   # T_in=10, B=4
y = torch.randint(0, V_out, (8, 4))    # T_out=8

# Training
logits = model(x, y)
print(f'Training logits: {logits.shape}')   # (S-1, B, V_out)

# Generation
gen = model.generate(x, max_len=15)
print(f'Generated:       {gen.shape}')      # (≤15, B)
```

### 실험 2 — Teacher Forcing vs Free-Running

```python
def train_teacher_forcing(model, x, y, opt):
    logits = model(x, y)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y[1:].reshape(-1)
    )
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

def train_free_running(model, x, y, opt, eos_idx=1):
    """학습 시 모델 prediction 사용 (no teacher forcing)"""
    x_emb = model.emb_in(x)
    _, (h, c) = model.encoder(x_emb)
    
    B = x.size(1)
    y_pred_prev = torch.full((B,), 0, dtype=torch.long)   # <bos>
    losses = []
    for t in range(y.size(0) - 1):
        y_emb = model.emb_out(y_pred_prev).unsqueeze(0)
        out, (h, c) = model.decoder(y_emb, (h, c))
        logits = model.fc(out.squeeze(0))
        loss = nn.functional.cross_entropy(logits, y[t+1])
        losses.append(loss)
        y_pred_prev = logits.argmax(-1)   # 모델 prediction
    
    total_loss = sum(losses) / len(losses)
    opt.zero_grad(); total_loss.backward(); opt.step()
    return total_loss.item()

# Compare
torch.manual_seed(0)
model_tf = Seq2Seq(V_in, V_out, D, H)
model_fr = Seq2Seq(V_in, V_out, D, H)
opt_tf = torch.optim.Adam(model_tf.parameters(), lr=1e-3)
opt_fr = torch.optim.Adam(model_fr.parameters(), lr=1e-3)

losses_tf, losses_fr = [], []
for step in range(30):
    x = torch.randint(0, V_in, (10, 16))
    y = torch.randint(0, V_out, (8, 16))
    losses_tf.append(train_teacher_forcing(model_tf, x, y, opt_tf))
    losses_fr.append(train_free_running(model_fr, x, y, opt_fr))

print(f'Teacher forcing  final loss: {losses_tf[-1]:.4f}')
print(f'Free running     final loss: {losses_fr[-1]:.4f}')
# TF 가 더 빠른 수렴 (학습 안정)
```

### 실험 3 — Reverse Input Trick

```python
# Reversed input 으로 학습
def train_reversed_input(model, x_rev, y, opt):
    logits = model(x_rev, y)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y[1:].reshape(-1)
    )
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

torch.manual_seed(0)
model_normal = Seq2Seq(V_in, V_out, D, H)
model_rev = Seq2Seq(V_in, V_out, D, H)
opt_n = torch.optim.Adam(model_normal.parameters(), lr=1e-3)
opt_r = torch.optim.Adam(model_rev.parameters(), lr=1e-3)

# 의존성: y의 첫 토큰이 x의 첫 토큰에 strong dependency
def make_dep_data(B, T_in=10, T_out=8):
    x = torch.randint(0, V_in, (T_in, B))
    y = torch.zeros(T_out, B, dtype=torch.long)
    y[0] = 0   # <bos>
    y[1] = x[0] % V_out   # y[1] depends on x[0]
    y[2:] = torch.randint(0, V_out, (T_out-2, B))
    return x, y

losses_n, losses_r = [], []
for step in range(30):
    x, y = make_dep_data(16)
    x_rev = x.flip(0)   # 역순
    losses_n.append(train_teacher_forcing(model_normal, x, y, opt_n))
    losses_r.append(train_reversed_input(model_rev, x_rev, y, opt_r))

print(f'Normal input  final loss: {losses_n[-1]:.4f}')
print(f'Reversed input final loss: {losses_r[-1]:.4f}')
# Reversed input 이 dependency 가 짧아져 더 잘 학습
```

### 실험 4 — Beam Search 단순 구현

```python
def beam_search(model, x_seq, beam_size=4, max_len=15, bos_idx=0, eos_idx=1):
    """Beam search inference"""
    x_emb = model.emb_in(x_seq)
    _, (h, c) = model.encoder(x_emb)
    B = x_seq.size(1)
    if B != 1:
        raise NotImplementedError("Single example beam search")
    
    # Initialize beams
    beams = [(torch.tensor([bos_idx]), 0.0, h, c)]   # (sequence, log_prob, hidden, cell)
    
    for step in range(max_len):
        new_beams = []
        for seq, log_p, h_b, c_b in beams:
            if seq[-1].item() == eos_idx:
                new_beams.append((seq, log_p, h_b, c_b))
                continue
            y_emb = model.emb_out(seq[-1:].unsqueeze(0))
            out, (h_new, c_new) = model.decoder(y_emb, (h_b, c_b))
            logits = model.fc(out.squeeze(0).squeeze(0))
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Top-k expansion
            top_log_probs, top_indices = log_probs.topk(beam_size)
            for log_prob, idx in zip(top_log_probs, top_indices):
                new_seq = torch.cat([seq, idx.unsqueeze(0)])
                new_log_p = log_p + log_prob.item()
                new_beams.append((new_seq, new_log_p, h_new, c_new))
        
        # Top-k beams overall
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        
        if all(b[0][-1].item() == eos_idx for b in beams):
            break
    
    return beams

x_single = torch.randint(0, V_in, (10, 1))
beams = beam_search(model_normal, x_single, beam_size=3)
print('Top 3 beams:')
for i, (seq, log_p, _, _) in enumerate(beams):
    print(f'  Beam {i+1}: log_prob = {log_p:.4f}, length = {len(seq)}')
```

### 실험 5 — Encoder Final State 의 Information Capacity

```python
# 다양한 input length 에서 encoder 의 v 가 reproducible 한 정보량 측정
def encode_and_decode(model, x_seq, decode_len=None):
    """encode then decode (autoencoder-like)"""
    if decode_len is None:
        decode_len = x_seq.size(0)
    
    x_emb = model.emb_in(x_seq)
    _, (h, c) = model.encoder(x_emb)
    
    # Decode (single token: <bos>)
    bos = torch.zeros(1, x_seq.size(1), dtype=torch.long)
    y_emb = model.emb_out(bos)
    out, _ = model.decoder(y_emb.expand(decode_len, -1, -1), (h, c))
    logits = model.fc(out)
    return logits.argmax(-1)

torch.manual_seed(0)
ae_model = Seq2Seq(V_in, V_in, D=32, H=64)   # Same V for autoencoder
opt = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

# Train as autoencoder
for step in range(50):
    T = torch.randint(5, 20, (1,)).item()
    x = torch.randint(0, V_in, (T, 16))
    # Use x as both source and target (autoencoder)
    logits = ae_model(x, x)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, V_in),
        x[1:].reshape(-1)
    )
    opt.zero_grad(); loss.backward(); opt.step()

# Test: longer sequence 의 reproduction accuracy
for T in [5, 10, 20, 50]:
    x = torch.randint(0, V_in, (T, 4))
    pred = encode_and_decode(ae_model, x, decode_len=T-1)
    acc = (pred == x[1:]).float().mean().item()
    print(f'T = {T}: reproduction accuracy = {acc:.3f}')
# T 클수록 정확도 감소 — encoder bottleneck (Ch6-02)
```

---

## 🔗 실전 활용

### 1. Neural Machine Translation (NMT)

Sutskever 2014 (En→Fr WMT'14 BLEU 30.6) — Seq2Seq 의 첫 실용 success.

### 2. Abstractive Summarization

Document → summary. Pointer-generator network (See 2017) 의 기반.

### 3. Speech Recognition

Listen-Attend-Spell (Chan 2016): audio → text Seq2Seq.

### 4. Dialogue Generation

Encoder-decoder for chatbot (Vinyals 2015).

### 5. Image Captioning

CNN encoder + LSTM decoder (Vinyals 2015 Show-and-Tell).

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Fixed encoder vector $v$ | Information bottleneck → Attention (Ch6-03) |
| Sequential generation | Non-autoregressive (Gu 2018) for speed |
| Teacher forcing | Exposure bias → Scheduled sampling, REINFORCE |
| Greedy/beam search | Sampling 도 가능 (top-k, nucleus) |
| Single encoder direction | BiLSTM encoder 표준 |

---

## 📌 핵심 정리

$$\boxed{\text{Encoder: } x_{1:T} \mapsto v = h_T^{\text{enc}}}$$

$$\boxed{\text{Decoder: } y_t \sim p(y | y_{<t}, v) \quad \text{(autoregressive)}}$$

$$\boxed{\text{Reverse input trick: } x_T, x_{T-1}, \ldots, x_1 \to v \text{ (short-range alignment)}}$$

| Strategy | Training | Inference | Use case |
|----------|----------|-----------|----------|
| **Teacher forcing** | Ground truth prefix | N/A | 표준 학습 |
| **Free running** | Model prefix | Match infer | Adversarial robust |
| **Scheduled sampling** | Mixed | Match | 점진적 transition |
| **Greedy** | N/A | Argmax | Fast inference |
| **Beam search** | N/A | Top-B | Better quality |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Encoder LSTM 의 last hidden $h_T$ 가 어떻게 input 의 모든 정보를 압축하는가? Capacity 의 정량적 limit 은?

<details>
<summary>해설</summary>

**Encoder hidden $h_T \in \mathbb R^H$**:
- Float precision 약 32 bits per dim
- Total information: $32 H$ bits

**Input information**:
- $T$ tokens, vocab $|V|$ → $T \log_2 |V|$ bits
- 예: $T = 50, |V| = 30000$ → $50 \times 15 = 750$ bits

**Capacity matching**:
- $H = 256$, float32: $32 \times 256 = 8192$ bits — 750 보다 훨씬 큼
- 이론상 모든 정보 인코딩 가능

**그러나**:
1. **Effective precision**: float32 의 32 bits 가 모두 distinct value 가 아님 — gradient learning 의 effective precision $\sim 10$ bits
2. **Distributed representation**: 비선형 압축, lossy
3. **Long sequences** ($T \to \infty$): 정보량 폭발

**Empirical**: $T > 30$ 시 정보 손실 명확 (Cho 2014b BLEU drop)

**해결**: Attention (Ch6-03) — encoder 의 *모든* $h_t$ 보존, fixed $v$ 의 bottleneck 회피.

$\square$

</details>

**문제 2** (심화): Teacher forcing 의 distribution mismatch 가 *왜* exposure bias 를 만드는지 distribution mismatch 관점에서 정확히 설명하라.

<details>
<summary>해설</summary>

**Training distribution**:
$$
p_{\text{train}}(y_{<t}) = p_{\text{data}}(y_{<t}) \quad (\text{ground truth})
$$

**Inference distribution**:
$$
p_{\text{infer}}(y_{<t}) = p_\theta(y_{<t} | x) \quad (\text{model output})
$$

**Loss minimization**:
$$
\theta^* = \arg\min_\theta \mathbb E_{x, y \sim p_{\text{data}}}[-\log p_\theta(y | x)]
$$

학습이 $p_{\text{data}}$ 분포 위에서 optimization. 그러나 inference 는 $p_\theta$ 위에서 동작.

**Mismatch 의 결과**:

1. **모델이 본 적 없는 prefix**:
   - Training: $y_{<t}^*$ ground truth (perfect)
   - Inference: $\hat y_{<t}$ — error 누적

2. **모델 예측이 정답 distribution 에서 멀어짐**:
   - 한 step error → 다음 step input 이 unusual
   - $p_\theta(y_t | unusual prefix)$ 의 generalization 약함

3. **Compounding errors**:
   - $\epsilon$ probability of error per step
   - $T$ steps: $1 - (1-\epsilon)^T \approx \epsilon T$ probability of *any* error
   - Long sequence 에서 catastrophic

**MIXER (Ranzato 2016)**:
- Sequence-level reward (BLEU) 직접 최적화
- REINFORCE 로 model output 사용
- Distribution mismatch 해결

**Scheduled sampling (Bengio 2015)**:
- Training 중 점진적으로 model output 사용
- $\epsilon_s$ probability 로 ground truth, $1 - \epsilon_s$ 로 model
- $\epsilon_s$ 가 epoch 따라 1 → 0

**현대 perspective**:
- Transformer + RL fine-tuning (RLHF) — exposure bias 의 modern approach
- Pre-training + fine-tuning paradigm 이 distribution shift 완화

**결론**: Exposure bias 는 *fundamental* training-inference mismatch — teacher forcing 의 conveniences 와 inference reality 의 trade-off. Modern solutions (RL fine-tuning) 가 이를 alleviate. $\square$

</details>

**문제 3** (논문 비평): Sutskever 2014 의 reverse input trick 이 +4.7 BLEU 향상 — 이 단순한 변경이 왜 그렇게 큰 효과를 내는가? Bahdanau attention (Ch6-03) 이 reverse trick 을 unnecessary 로 만든 이유는?

<details>
<summary>해설</summary>

**Reverse input trick 의 본질**:

**Original**:
- $x_1$ 이 encoder 의 첫 input
- $h_T$ (encoder 의 마지막) 가 $x_1$ 으로부터 $T$ steps away
- LSTM 의 vanishing gradient 로 $x_1$ 정보 weak in $h_T$

**Reversed**:
- $x_T, x_{T-1}, \ldots, x_1$ 순서로 encoder
- $h_T$ 가 (reversed sequence 의) $x_1$ — 즉 *original 의* $x_T$ 에서 1 step
- 그러나 더 중요: $h_T$ 가 *original 의* $x_1$ 에서 $T$ steps — 같은 거리

**Key insight**: 번역의 monotonic alignment

- En→Fr: 첫 번역 단어가 첫 source 단어와 가깝게 align
- $y_1$ 의 prediction 이 $x_1$ 정보 중요
- Decoder 의 첫 step 이 $h_T$ (encoder last) 에 의존

**Original**:
- $y_1$ ↔ $x_1$ : encoder dist $T$ + decoder dist $0$ = $T$ steps of LSTM
- Long-range: vanishing 위험

**Reversed**:
- $y_1$ ↔ $x_1$ : encoder dist $0$ (마지막 input) + decoder dist $0$ = $0$ steps directly
- Short-range: clear signal

**왜 큰 효과**:
- $y_1$ 이 잘 예측되면 (auto-regressive) 후속 $y_t$ 도 잘
- 첫 토큰의 quality 가 sequence 전체에 cascade

**Empirical**:
- Sutskever 2014: BLEU 25.9 (forward) → 30.6 (reversed)
- +4.7 — 매우 큰 NMT improvement (보통 2 이내)

**Bahdanau Attention 의 영향**:

Bahdanau 2015 (Ch6-03) 가 reverse trick 을 *unnecessary* 로 만듦:

1. **All encoder states accessible**:
   - $h_t^{\text{enc}}$ for all $t$ 가 attention 의 candidate
   - Decoder 가 *직접* $h_1^{\text{enc}}$ access 가능 (vanishing gradient 없음)

2. **Position-flexible**:
   - Reverse 든 forward 든 decoder 가 적절히 align 학습
   - Attention 의 alignment learning 이 reverse 의 manual 한 trick 대체

3. **Long sequence robust**:
   - Attention 이 distance-independent
   - Reverse 의 short-range trick 불필요

**Empirical (Bahdanau 2015)**:
- Attention RNN 이 vanilla Seq2Seq + reverse 보다 우월
- BLEU +5 ~ +10 향상
- Long sentence 에서 특히 강함

**Modern view**:
- Reverse trick = vanishing gradient workaround
- Attention = vanishing gradient bypass
- Transformer = attention-only, sequence-parallel
- ML 의 진화: workaround → fundamental fix → architectural rethink

**결론**: Reverse input trick 이 LSTM Seq2Seq 의 *clever hack*, 그러나 attention 이 etymological 해결. ML 의 *idea ladder* — 작은 trick 이 *문제 인식* 을 sharpen, 그것이 *fundamental solution* 으로 진화. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch5-advanced-rnn/04-esn.md) | [📚 README](../README.md) | [다음 ▶](./02-bottleneck.md)

</div>
