# 03. Bahdanau Attention (Additive)

## 🎯 핵심 질문

- Bahdanau 2015 의 *Neural Machine Translation by Jointly Learning to Align and Translate* 가 어떻게 attention 으로 information bottleneck (Ch6-02) 을 해결했는가?
- **Additive attention** $e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)$ 의 정확한 수식과 의미?
- $\alpha_{ij} = \mathrm{softmax}(e_{ij})$, $c_j = \sum_i \alpha_{ij} h_i$ — context vector 의 weighted sum
- **Alignment 가 학습 가능한 함수** — 고정 alignment (IBM model) 와의 차이
- WMT'14 En→Fr 에서 BLEU 28.5 → 36.2 — Seq2Seq 대비 attention 의 dramatic improvement

---

## 🔍 왜 Bahdanau Attention 이 paradigm shift 인가

Cho 2014b 의 진단 (Ch6-02): vanilla Seq2Seq 의 long sentence curse. Bahdanau 2015 의 해법:

1. **Encoder의 모든 hidden state 보존** — Fixed bottleneck 제거
2. **Decoder 가 매 step 에서 attend** — Position-specific information access
3. **Differentiable alignment** — End-to-end 학습

이는 NMT revolution 의 시작:
- Vanilla Seq2Seq (2014): BLEU 25-30
- Bahdanau attention (2015): BLEU 36+
- Transformer (2017): BLEU 40+

이 문서는 Bahdanau attention 의 정확한 수식, 학습된 alignment 의 시각화, 그리고 Transformer 까지 이어지는 계보를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-bottleneck.md](./02-bottleneck.md) — Information bottleneck 진단
- [Ch6-01 Encoder-Decoder](./01-encoder-decoder.md) — Seq2Seq baseline
- 정의: Softmax, weighted sum, learnable parameters

---

## 📖 직관적 이해

### Attention 의 핵심 idea

```
Encoder hiddens: h_1   h_2   h_3   h_4   h_5
                  │     │     │     │     │
                  └─────┴─────┴─────┴─────┘
                         │
                         ▼
                    Decoder step j 에서
                    각 h_i 의 relevance 계산
                         │
                         ▼
                    α_{1j}, α_{2j}, α_{3j}, α_{4j}, α_{5j}
                         │
                         ▼ weighted sum
                    c_j = Σ_i α_{ij} h_i
                         │
                         ▼
                    Decoder uses c_j (in addition to s_{j-1})
```

### Alignment 학습

번역 예: "I love cats" → "J'aime les chats"
- $y_1 = $ "J'" → most relevant: $x_1 = $ "I"
- $y_2 = $ "aime" → $x_2 = $ "love"
- $y_3 = $ "les" → no direct alignment (article addition)
- $y_4 = $ "chats" → $x_3 = $ "cats"

학습이 이 alignment 를 자동으로 발견.

### Additive vs Multiplicative

**Additive (Bahdanau)**:
$$
e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)
$$
- Two linear projections + tanh + projection to scalar
- More expressive (학습된 nonlinear similarity)

**Multiplicative (Luong, Ch6-04)**:
$$
e_{ij} = h_i^\top W s_j \text{ (general)} \quad \text{or} \quad h_i^\top s_j \text{ (dot)}
$$
- Single matrix product
- More efficient

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Bahdanau Additive Attention

Encoder hidden states $\{h_1, \ldots, h_T\}$, decoder hidden $s_j$:

$$
\begin{aligned}
e_{ij} &= v^\top \tanh(W_1 h_i + W_2 s_j) & \text{alignment score} \\
\alpha_{ij} &= \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{kj})} & \text{attention weight} \\
c_j &= \sum_{i=1}^{T} \alpha_{ij} h_i & \text{context vector}
\end{aligned}
$$

여기서 $v \in \mathbb R^{d_a}$, $W_1 \in \mathbb R^{d_a \times H}$, $W_2 \in \mathbb R^{d_a \times H}$ 는 학습 가능 parameters.

### 정의 3.2 — Decoder Update

Context $c_j$ 가 decoder 의 input:

$$
s_j = \text{LSTM}(s_{j-1}, [y_{j-1}; c_j])
$$

또는 output 에 사용:

$$
p(y_j | y_{<j}, x) = \mathrm{softmax}(W_o [s_j; c_j] + b_o)
$$

### 정의 3.3 — Attention Vector Alternatives

**Bahdanau-style** (concat input):
$$
\tilde s_j = \text{LSTM}(\tilde s_{j-1}, [y_{j-1}; c_j])
$$

**Output-side attention** (Luong-style):
$$
s_j = \text{LSTM}(s_{j-1}, y_{j-1}); \quad \tilde s_j = \tanh(W_c [c_j; s_j])
$$

### 정의 3.4 — Bidirectional Encoder + Attention

$h_i = [\overrightarrow{h}_i; \overleftarrow{h}_i]$ (BiLSTM, Ch5-01) — both directions context.

Attention 시 양방향 정보 모두 사용 가능.

### 정의 3.5 — Training Objective

Standard cross-entropy:
$$
\mathcal L = -\sum_{j=1}^{S} \log p(y_j | y_{<j}, x)
$$

Attention parameters $v, W_1, W_2$ 가 end-to-end 학습됨 (no separate alignment loss).

---

## 🔬 정리와 결과

### 정리 3.1 — Capacity Improvement

Vanilla Seq2Seq capacity: $H$ (single $v$).
Attention capacity: $T \cdot H$ (모든 $h_i$).

**증명**: Decoder 의 각 step 이 *모든* $\{h_i\}$ access. Information 이 weighted sum 으로 retrievable. $\square$

### 정리 3.2 — Bottleneck Bypass

Decoder step $j$ 의 information access 가 sentence length $T$ 에 *비례*. Long sentence 에서도 capacity 충분.

### 정리 3.3 — Differentiability

Softmax + weighted sum 모두 differentiable → end-to-end 학습.

**Implication**: Alignment 가 *implicit* 으로 학습 — explicit alignment label 없이.

### 정리 3.4 — Attention Weight Interpretation

학습된 $\alpha_{ij}$ 가 alignment matrix 시각화. Bahdanau 2015 의 visualization:
- Diagonal-dominant for monotonic alignment (e.g., En→Fr 일부 phrase)
- Off-diagonal for reordering (e.g., En→Ja word order 차이)

### 정리 3.5 — Empirical Improvement

WMT'14 En→Fr (Bahdanau 2015):
- Vanilla Seq2Seq + reverse: BLEU 28.5
- Bahdanau Attention RNN: BLEU 36.2
- **+7.7 BLEU** — major NMT advance

특히 long sentence 에서:
- Length 50+: 30+ improvement (Cho 2014b 의 catastrophic drop 회복)

---

## 💻 PyTorch 구현 검증

### 실험 1 — Bahdanau Attention 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, H_enc, H_dec, d_attn):
        super().__init__()
        self.W1 = nn.Linear(H_enc, d_attn, bias=False)
        self.W2 = nn.Linear(H_dec, d_attn, bias=False)
        self.v  = nn.Linear(d_attn, 1, bias=False)
    
    def forward(self, encoder_states, decoder_h):
        """
        encoder_states: (T_enc, B, H_enc)
        decoder_h: (B, H_dec)
        
        Returns:
          context: (B, H_enc)
          alpha: (T_enc, B)
        """
        T_enc = encoder_states.size(0)
        # W2 s_j (broadcast)
        decoder_proj = self.W2(decoder_h).unsqueeze(0)   # (1, B, d_attn)
        # W1 h_i
        encoder_proj = self.W1(encoder_states)           # (T_enc, B, d_attn)
        # Sum + tanh
        scores = self.v(torch.tanh(encoder_proj + decoder_proj))   # (T_enc, B, 1)
        scores = scores.squeeze(-1)                       # (T_enc, B)
        # Softmax over T_enc
        alpha = F.softmax(scores, dim=0)                  # (T_enc, B)
        # Weighted sum
        context = (alpha.unsqueeze(-1) * encoder_states).sum(0)   # (B, H_enc)
        return context, alpha

# Test
T_enc, B, H = 10, 4, 32
attn = BahdanauAttention(H, H, d_attn=16)
encoder_states = torch.randn(T_enc, B, H)
decoder_h = torch.randn(B, H)
context, alpha = attn(encoder_states, decoder_h)
print(f'Context: {context.shape}')
print(f'Alpha:   {alpha.shape}, sum: {alpha.sum(0)}')   # Each column sums to 1
```

### 실험 2 — Full Attention Seq2Seq

```python
class AttentionSeq2Seq(nn.Module):
    def __init__(self, V_in, V_out, D, H, d_attn=64):
        super().__init__()
        self.emb_in = nn.Embedding(V_in, D)
        self.emb_out = nn.Embedding(V_out, D)
        self.encoder = nn.LSTM(D, H, bidirectional=True)   # BiLSTM
        H_enc = 2 * H   # bidirectional
        self.decoder_cell = nn.LSTMCell(D + H_enc, H)
        self.attn = BahdanauAttention(H_enc, H, d_attn)
        self.fc = nn.Linear(H + H_enc, V_out)
        self.H = H
        self.H_enc = H_enc
    
    def forward(self, x_seq, y_seq):
        """Teacher forcing"""
        # Encoder
        x_emb = self.emb_in(x_seq)
        encoder_states, _ = self.encoder(x_emb)   # (T_enc, B, 2H)
        
        # Decoder with attention
        B = y_seq.size(1)
        h = torch.zeros(B, self.H, device=x_seq.device)
        c = torch.zeros(B, self.H, device=x_seq.device)
        
        outputs = []
        attentions = []
        for j in range(y_seq.size(0) - 1):
            y_in = self.emb_out(y_seq[j])
            context, alpha = self.attn(encoder_states, h)
            attentions.append(alpha)
            
            # Decoder step
            input_with_context = torch.cat([y_in, context], dim=-1)
            h, c = self.decoder_cell(input_with_context, (h, c))
            
            # Output
            output = self.fc(torch.cat([h, context], dim=-1))
            outputs.append(output)
        
        return torch.stack(outputs), torch.stack(attentions)

torch.manual_seed(0)
V_in, V_out, D, H = 100, 80, 32, 64
model = AttentionSeq2Seq(V_in, V_out, D, H)

x = torch.randint(0, V_in, (15, 4))
y = torch.randint(0, V_out, (10, 4))
logits, attns = model(x, y)
print(f'Logits:     {logits.shape}')   # (S-1, B, V_out)
print(f'Attentions: {attns.shape}')     # (S-1, T_enc, B)
print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')
```

### 실험 3 — Training on Toy Translation

```python
def make_translation_pair(B, T_in=10, V_in=100, V_out=80):
    """간단한 toy: input 이 output 의 reversal 또는 shifted"""
    x = torch.randint(0, V_in, (T_in, B))
    # Output: reversed input mod V_out
    y = (x.flip(0) % V_out)
    # Add <bos> at start
    y = torch.cat([torch.zeros(1, B, dtype=torch.long), y])
    return x, y

torch.manual_seed(0)
model = AttentionSeq2Seq(V_in=100, V_out=80, D=32, H=64)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for step in range(100):
    x, y = make_translation_pair(B=32)
    logits, _ = model(x, y)
    loss = F.cross_entropy(logits.reshape(-1, 80), y[1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
print(f'Initial loss: {losses[0]:.4f}')
print(f'Final loss:   {losses[-1]:.4f}')
```

### 실험 4 — Attention Heatmap 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

# 학습된 model 의 attention 시각화
def visualize_attention(model, x, max_len=10):
    model.eval()
    with torch.no_grad():
        x_emb = model.emb_in(x)
        encoder_states, _ = model.encoder(x_emb)
        
        B = x.size(1)
        h = torch.zeros(B, model.H)
        c = torch.zeros(B, model.H)
        y = torch.zeros(B, dtype=torch.long)
        
        attentions = []
        outputs = []
        for j in range(max_len):
            y_in = model.emb_out(y)
            context, alpha = model.attn(encoder_states, h)
            attentions.append(alpha[:, 0].numpy())   # First sample
            
            inp = torch.cat([y_in, context], dim=-1)
            h, c = model.decoder_cell(inp, (h, c))
            output = model.fc(torch.cat([h, context], dim=-1))
            y = output.argmax(-1)
            outputs.append(y[0].item())
        
        return np.stack(attentions), outputs

x_test, _ = make_translation_pair(1, T_in=10)
attentions, outputs = visualize_attention(model, x_test, max_len=10)

plt.figure(figsize=(8, 6))
plt.imshow(attentions, aspect='auto', cmap='Blues')
plt.xlabel('Encoder position')
plt.ylabel('Decoder step')
plt.title('Bahdanau Attention Heatmap (toy reversal task)')
plt.colorbar()
plt.savefig('bahdanau_attention.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved: bahdanau_attention.png')
print('Note: For reversal task, attention should be anti-diagonal')
```

### 실험 5 — Long Sentence Performance

```python
# 다양한 length 의 학습 결과 비교
def test_length(model, T_test, n=50):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n):
            x, y = make_translation_pair(1, T_in=T_test)
            logits, _ = model(x, y)
            preds = logits.argmax(-1)
            correct += (preds.squeeze(-1) == y[1:].squeeze(-1)).sum().item()
            total += T_test
    return correct / total

for T in [5, 10, 20, 30]:
    acc = test_length(model, T)
    print(f'T = {T:3d}: accuracy = {acc:.4f}')
# Attention 이 long sentence 에서 robust
```

---

## 🔗 실전 활용

### 1. Neural Machine Translation

Bahdanau 2015 가 WMT'14 En→Fr SOTA. 모든 후속 NMT 가 attention 사용.

### 2. Abstractive Summarization

Pointer-generator (See 2017) — attention + copy mechanism.

### 3. Image Captioning

Show-Attend-Tell (Xu 2015) — image regions 에 attend.

### 4. Speech Recognition

Listen-Attend-Spell (Chan 2016) — frame-to-character attention.

### 5. Question Answering

BiDAF (Seo 2016) — passage 와 question 사이의 attention.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Soft attention | Hard attention (RL) 가 일부 task 에서 우월 |
| Single attention head | Multi-head (Transformer) 가 더 강력 |
| Concat in tanh | Multiplicative (Luong) 가 더 효율 |
| RNN encoder | Transformer 가 sequence parallel |
| Position-blind attention | Positional encoding 필요 |

---

## 📌 핵심 정리

$$\boxed{e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j), \quad \alpha_{ij} = \mathrm{softmax}(e_{ij})}$$

$$\boxed{c_j = \sum_i \alpha_{ij} h_i \quad \text{— context vector}}$$

$$\boxed{\text{Capacity: } T \cdot H \text{ (vs vanilla Seq2Seq } H)}$$

| Component | Role |
|-----------|------|
| **$W_1 h_i$** | Encoder state projection |
| **$W_2 s_j$** | Decoder state projection |
| **$\tanh$** | Nonlinear similarity |
| **$v^\top$** | Score scalar projection |
| **softmax** | Probability normalization |
| **weighted sum** | Context retrieval |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Bahdanau attention 의 학습 가능 parameters 를 나열하라. $H = 256$, $d_{\text{attn}} = 64$ 시 attention layer 의 parameter 수는?

<details>
<summary>해설</summary>

**Parameters**:
- $W_1 \in \mathbb R^{d_{\text{attn}} \times H}$: $64 \times 256 = 16{,}384$
- $W_2 \in \mathbb R^{d_{\text{attn}} \times H}$: $64 \times 256 = 16{,}384$
- $v \in \mathbb R^{d_{\text{attn}}}$: $64$
- Total: $32{,}832$

**비교**:
- Encoder LSTM (1-layer, $H = 256$, $D = 32$): $4 \times 256 \times (256+32+1) = 296{,}960$
- Attention 추가 비용: ~10% of encoder

**의미**: Attention 이 *경제적* — 작은 추가 parameter 로 dramatic improvement.

$\square$

</details>

**문제 2** (심화): Bahdanau additive attention 이 왜 multiplicative 보다 *더 expressive* 한가? 그리고 *덜 efficient* 한 이유는?

<details>
<summary>해설</summary>

**Additive (Bahdanau)**:
$$
e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)
$$

**Multiplicative (Luong general)**:
$$
e_{ij} = h_i^\top W s_j
$$

**Expressiveness 비교**:

**Additive**:
- $\tanh$ 의 nonlinearity → arbitrary similarity learning
- 예: $h_i$ 와 $s_j$ 의 *non-linear* relationship 학습 가능
- $h_i + s_j$ 의 specific combinations 가 high score

**Multiplicative**:
- Bilinear form $h^\top W s$ — linear in $h$ and $s$ separately
- $W$ 가 $h, s$ 의 *linear* alignment 학습
- Cosine similarity 의 generalization

**구체적 예시**:
- Additive: "if $h_i$ is verb AND $s_j$ is noun (different categories), score high" — non-linear logic
- Multiplicative: "if $h_i$ and $s_j$ are similar in $W$-projected space" — linear similarity

**Efficiency**:

Additive 의 cost:
- $W_1 h_i$: $O(H \cdot d_{\text{attn}})$ per encoder state
- $W_2 s_j$: $O(H \cdot d_{\text{attn}})$ per decoder state
- $\tanh$ + sum + $v$: $O(d_{\text{attn}})$
- Total per pair: $O(H \cdot d_{\text{attn}} + d_{\text{attn}})$
- All $T \times S$ pairs: $O(T S H d_{\text{attn}})$

Multiplicative 의 cost:
- $h_i^\top W$: $O(H^2)$ pre-compute
- $\cdot s_j$: $O(H)$ per query
- All pairs: $O(T S H + T H^2)$ — 보통 $H^2 \ll T S H$ for long sequences

**Speed comparison**:
- Multiplicative: $\sim 30$-$50\%$ faster
- Vector reuse, single matmul

**왜 둘 다 사용**:
- Bahdanau: research-friendly, expressive, default choice
- Luong: production-friendly, efficient
- Modern Transformer: scaled dot-product (multiplicative variant)

**Empirical**:
- Most tasks 에서 거의 동일 BLEU
- Very specific tasks (음성 인식 의 pitch matching) 에서 additive 우월
- 일반 NMT 에서 multiplicative 표준

**결론**: Additive 의 expressiveness vs multiplicative 의 efficiency — 대부분 task 에서 efficiency 가 win. Transformer 가 efficient 한 multiplicative attention 으로 표준화. $\square$

</details>

**문제 3** (논문 비평): Bahdanau 2015 에서 "alignment 가 *implicit* 학습" 의 의미를 설명하라. 이전 NMT (IBM models) 의 explicit alignment 와 어떻게 다른가?

<details>
<summary>해설</summary>

**IBM Models (1990s)**:
- Statistical MT 의 표준
- Word alignment 가 *separate* component
- Alignment $\to$ phrase extraction $\to$ language model 의 pipeline

**IBM Model 1**:
$$
P(y, a | x) = \prod_j P(y_j | x_{a_j}) \cdot P(a_j)
$$
$a_j$: $y_j$ 가 어떤 $x_{a_j}$ 와 align 되는지 — *latent variable*.

**IBM Model 4-5**:
- 더 복잡한 distortion model, fertility (한 source word → 여러 target)
- EM algorithm 으로 alignment 학습
- *Separate* training stage

**Alignment 의 필요**:
- Phrase extraction 위한 word alignment (heuristic merge)
- Translation pair (phrase) 의 frequency table
- 이 모든 것이 **discrete, hard** alignment

**Bahdanau 의 implicit alignment**:

**Soft attention $\alpha_{ij} \in [0, 1]$**:
- 매 decoder step 의 *probability distribution* over encoder positions
- Hard alignment 의 generalization

**Implicit 의미**:
- *Alignment label 없이* 학습
- End-to-end training 의 부산물
- Translation quality 가 alignment 학습을 *driving*

**Soft alignment 의 장점**:
1. **Differentiable**: gradient-based learning 가능
2. **Multi-source**: 한 target word 가 여러 source word 의 weighted combination
3. **Continuous**: hard decision 의 brittleness 회피

**학습 dynamics**:
- 초기: $\alpha$ uniform (random)
- 학습 진행: 점진적으로 sharp align (specific positions)
- 최종: sparse, often diagonal-dominant

**Visualize**:
- Bahdanau 2015 Fig 3: En→Fr alignment heatmap
- Diagonal main, with reorderings (e.g., adjective-noun 순서)
- IBM Model 의 alignment 와 *유사하지만* 더 nuanced

**왜 implicit 가 우월**:

1. **No alignment supervision**:
   - IBM 은 explicit alignment annotation 필요 (또는 EM)
   - Bahdanau 은 parallel corpus 만

2. **Joint optimization**:
   - Alignment 와 translation 이 *서로 inform*
   - Better translation → better alignment → better translation

3. **Soft mixture**:
   - 정확한 1-to-1 alignment 가 모호한 case (idioms, function words)
   - Soft attention 이 더 자연스러움

4. **Gradient flow**:
   - 모든 component 가 backprop
   - Local minima 적음

**현대 perspective**:

- Transformer 의 self-attention: implicit context modeling
- BERT 의 masked LM: implicit alignment between mask and context
- LLM 의 in-context learning: implicit task understanding
- **Implicit learning** 이 모든 modern ML 의 핵심 정신

**결론**: Bahdanau 의 implicit alignment 가 *paradigm shift* — explicit pipeline 에서 end-to-end learning 으로. 이 정신이 모든 modern NLP architectures 의 기반. **Implicit learning 의 power** 가 deep learning 의 success 의 key. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-bottleneck.md) | [📚 README](../README.md) | [다음 ▶](./04-luong-attention.md)

</div>
