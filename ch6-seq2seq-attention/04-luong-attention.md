# 04. Luong Attention (Multiplicative)

## 🎯 핵심 질문

- Luong 2015 의 *Effective Approaches to Attention-based Neural Machine Translation* 의 3가지 scoring functions — dot, general, concat 의 차이?
- **General**: $e_{ij} = h_i^\top W s_j$ 가 BLEU 측면에서 최우수인 이유?
- **Global vs Local attention** — local 의 monotonic alignment 가정과 window 제한
- 이것이 어떻게 Transformer 의 **scaled dot-product attention** $\frac{QK^\top}{\sqrt{d}}$ 의 직계 조상인가?
- Vaswani 2017 가 Luong 의 general scoring 에 정확히 어떤 변경을 가했는가?

---

## 🔍 왜 Luong Attention 이 Transformer 의 prelude 인가

Bahdanau 2015 (Ch6-03) 의 additive attention 이 NMT revolution 의 시작이지만 **scoring function 의 선택지** 가 명확하지 않음. Luong 2015 가 이를 systematic 비교:

1. **3 scoring functions 비교** — dot, general, concat
2. **Global vs local attention** — efficiency 와 quality trade-off
3. **Empirical SOTA** — WMT'15 En→De 에서 BLEU 25.9

가장 중요한 contribution: **Multiplicative attention 이 additive 와 동등 또는 우월** 입증 → 효율성. 이는 Transformer 의 scaled dot-product 의 직접적 motivation.

이 문서는 Luong 의 3 scoring 비교, local attention 의 메커니즘, 그리고 Transformer 까지의 계보를 추적합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [03-bahdanau-attention.md](./03-bahdanau-attention.md) — Additive attention
- (선택) [Transformer Deep Dive](https://github.com/iq-ai-lab/transformer-deep-dive) — Scaled dot-product 의 final form

---

## 📖 직관적 이해

### 3가지 Scoring Functions

```
"How similar is decoder state s_j to encoder state h_i?"

1. Dot:           e_{ij} = h_i^T s_j           (단순)
2. General:       e_{ij} = h_i^T W s_j         (학습된 bilinear)
3. Concat:        e_{ij} = v^T tanh(W [h_i; s_j])  (Bahdanau-style)
```

### Global vs Local Attention

```
Global:   모든 encoder positions attended
          [α_{1j}, α_{2j}, ..., α_{Tj}]
          
Local:    Window around predicted position p_j
          [α_{p_j-D, j}, ..., α_{p_j+D, j}]
```

Local 이 efficiency 우위 (특히 long sequence), monotonic alignment 가정.

### Multiplicative 의 Computational 우위

```
Additive (Bahdanau):
  W_1 h_i   ← per encoder state (linear)
  W_2 s_j   ← per decoder state
  tanh + v  ← per (i, j) pair
  Cost: O(T·S·H·d_attn)

Multiplicative (Luong general):
  W h_i     ← per encoder state (pre-compute, reuse)
  · s_j     ← per (i, j) pair (matmul)
  Cost: O(T·H^2 + T·S·H)
```

Long sequence 에서 multiplicative 가 30-50% faster.

### Transformer 와의 연결

```
Luong general:        e_{ij} = h_i^T W s_j

Decompose W:          W = W_K^T W_Q  (low-rank)

Then:                 e_{ij} = h_i^T W_K^T W_Q s_j
                            = (W_K h_i)^T (W_Q s_j)
                            = K_i^T Q_j

Add scaling:          e_{ij} = (K_i^T Q_j) / √d_K   ← Transformer!
```

Luong general 의 직접적 generalization 이 Transformer 의 scaled dot-product.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Luong Scoring Functions

세 가지:

$$
e_{ij} = \begin{cases}
h_i^\top s_j & \text{dot} \\
h_i^\top W_a s_j & \text{general} \\
v^\top \tanh(W_a [h_i; s_j]) & \text{concat (= Bahdanau)}
\end{cases}
$$

### 정의 4.2 — Global Attention

모든 encoder positions:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{kj})}
$$

### 정의 4.3 — Local Attention

Predicted center $p_j$ 와 window size $D$:

$$
\alpha_{ij} = \begin{cases}
\frac{\exp(e_{ij})}{\sum_{k \in [p_j - D, p_j + D]} \exp(e_{kj})} \cdot \exp\left(-\frac{(i - p_j)^2}{2\sigma^2}\right) & i \in [p_j - D, p_j + D] \\
0 & \text{otherwise}
\end{cases}
$$

$p_j$ 는 학습 가능 함수 (e.g., $p_j = T \cdot \sigma(v^\top \tanh(W s_j))$).

### 정의 4.4 — Decoder Update (Luong-style)

Bahdanau 의 *input* attention 과 달리, Luong 은 *output* attention:

$$
\tilde s_j = \tanh(W_c [c_j; s_j]), \quad p(y_j) = \mathrm{softmax}(W_s \tilde s_j)
$$

$s_j$ 는 standard LSTM update (no attention input).

### 정의 4.5 — Scaled Dot-Product Attention (Transformer)

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_K}}\right) V
$$

여기서:
- $Q = h_i W_Q, K = h_j W_K, V = h_j W_V$ (linear projections)
- $\sqrt{d_K}$ scaling — gradient stability

---

## 🔬 정리와 결과

### 정리 4.1 — Luong 의 Empirical Comparison

WMT'15 En→De (Luong 2015 Table 4):
- Dot: BLEU 25.4
- General: **BLEU 25.9** (best!)
- Concat: BLEU 25.4

**General 이 우월** — additive (concat) 와 동등하거나 약간 우월.

### 정리 4.2 — Computational Efficiency

| Scoring | Cost per pair | Total ($T \times S$) |
|---------|--------------|---------------------|
| **Dot** | $O(H)$ | $O(T S H)$ |
| **General** | $O(H^2)$ pre + $O(H)$ | $O(T H^2 + T S H)$ |
| **Concat (Bahdanau)** | $O(d_a H + d_a)$ | $O(T S (d_a H + d_a))$ |

General 이 best balance — pre-compute reuse + linear pair cost.

### 정리 4.3 — Local Attention 의 Speed

Window $D$, total positions $T$:
- Local: $O(T S D)$ vs Global $O(T S T) = O(T^2 S)$
- $D \ll T$ 시 dramatic speedup

**Trade-off**: Local 이 BLEU 약간 손실 (~0.5), efficiency 큰 향상.

### 정리 4.4 — Luong → Transformer 연결

**Luong general**: $e = h^\top W s$
**Transformer**: $e = (h W_K)^\top (s W_Q) / \sqrt{d_K}$

차이:
1. $W$ 의 *low-rank decomposition* into $W_K, W_Q$
2. $\sqrt{d_K}$ scaling (gradient stability)
3. *Self*-attention (Q = K = V from same source)
4. Multi-head (parallel attention heads)

이 모든 변경이 Vaswani 2017.

### 정리 4.5 — Multi-Head 의 Generalization

Luong 의 single attention → Transformer 의 multi-head:

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

각 head 가 다른 $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ — 다양한 attention pattern 학습.

---

## 💻 PyTorch 구현 검증

### 실험 1 — Luong 3 Scoring Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    def __init__(self, H, scoring='general'):
        super().__init__()
        self.H = H
        self.scoring = scoring
        if scoring == 'general':
            self.W = nn.Linear(H, H, bias=False)
        elif scoring == 'concat':
            self.W = nn.Linear(2 * H, H)
            self.v = nn.Linear(H, 1, bias=False)
        # 'dot' 은 parameter 없음
    
    def forward(self, encoder_states, decoder_h):
        """
        encoder_states: (T, B, H)
        decoder_h: (B, H)
        """
        T, B, H = encoder_states.shape
        
        if self.scoring == 'dot':
            # h_i^T s_j
            scores = torch.bmm(
                encoder_states.transpose(0, 1),   # (B, T, H)
                decoder_h.unsqueeze(2)             # (B, H, 1)
            ).squeeze(-1)                          # (B, T)
            scores = scores.transpose(0, 1)        # (T, B)
        
        elif self.scoring == 'general':
            # h_i^T W s_j
            decoder_proj = self.W(decoder_h)       # (B, H)
            scores = torch.bmm(
                encoder_states.transpose(0, 1),    # (B, T, H)
                decoder_proj.unsqueeze(2)          # (B, H, 1)
            ).squeeze(-1)                          # (B, T)
            scores = scores.transpose(0, 1)
        
        elif self.scoring == 'concat':
            # v^T tanh(W [h_i; s_j])
            decoder_h_exp = decoder_h.unsqueeze(0).expand(T, -1, -1)
            combined = torch.cat([encoder_states, decoder_h_exp], dim=-1)
            scores = self.v(torch.tanh(self.W(combined))).squeeze(-1)   # (T, B)
        
        alpha = F.softmax(scores, dim=0)
        context = (alpha.unsqueeze(-1) * encoder_states).sum(0)
        return context, alpha

# Test all 3
torch.manual_seed(0)
T, B, H = 8, 4, 32
encoder_states = torch.randn(T, B, H)
decoder_h = torch.randn(B, H)

for scoring in ['dot', 'general', 'concat']:
    attn = LuongAttention(H, scoring=scoring)
    ctx, alpha = attn(encoder_states, decoder_h)
    print(f'{scoring:8s}: context {ctx.shape}, alpha sum {alpha.sum(0)[0]:.4f}')
    print(f'         params: {sum(p.numel() for p in attn.parameters())}')
```

### 실험 2 — Local Attention 구현

```python
class LocalAttention(nn.Module):
    def __init__(self, H, D=2, scoring='general'):
        super().__init__()
        self.H, self.D = H, D
        self.scoring = scoring
        if scoring == 'general':
            self.W = nn.Linear(H, H, bias=False)
        # Center prediction
        self.W_p = nn.Linear(H, H)
        self.v_p = nn.Linear(H, 1)
    
    def forward(self, encoder_states, decoder_h, T_enc=None):
        if T_enc is None:
            T_enc = encoder_states.size(0)
        B = decoder_h.size(0)
        
        # Predict center p_j (Luong 2015 §3.2 local-p)
        p_j = T_enc * torch.sigmoid(self.v_p(torch.tanh(self.W_p(decoder_h)))).squeeze(-1)   # (B,)
        
        # Compute scores for all (or window)
        if self.scoring == 'general':
            decoder_proj = self.W(decoder_h)
            scores = torch.bmm(
                encoder_states.transpose(0, 1),
                decoder_proj.unsqueeze(2)
            ).squeeze(-1).transpose(0, 1)   # (T, B)
        
        # Apply Gaussian window centered at p_j
        positions = torch.arange(T_enc, dtype=torch.float).unsqueeze(1)   # (T, 1)
        gaussian = torch.exp(-((positions - p_j.unsqueeze(0))**2) / (2 * (self.D / 2)**2))
        
        # Mask outside window [p_j - D, p_j + D]
        mask = (torch.abs(positions - p_j.unsqueeze(0)) <= self.D).float()
        masked_scores = scores * mask + (1 - mask) * (-1e9)
        
        alpha = F.softmax(masked_scores, dim=0) * gaussian
        alpha = alpha / (alpha.sum(0) + 1e-9)
        context = (alpha.unsqueeze(-1) * encoder_states).sum(0)
        return context, alpha, p_j

local_attn = LocalAttention(H, D=2)
ctx, alpha, p = local_attn(encoder_states, decoder_h)
print(f'Local context: {ctx.shape}')
print(f'Predicted centers p_j: {p}')
print(f'Alpha (first sample): {alpha[:, 0].numpy()}')
```

### 실험 3 — Speed Comparison: Additive vs Multiplicative

```python
import time

def benchmark_attention(attn, encoder_states, decoder_h, n_iter=100):
    # Warmup
    for _ in range(5):
        attn(encoder_states, decoder_h)
    start = time.time()
    for _ in range(n_iter):
        attn(encoder_states, decoder_h)
    return (time.time() - start) / n_iter * 1000

T, B, H = 100, 32, 256
encoder_states = torch.randn(T, B, H)
decoder_h = torch.randn(B, H)

attn_dot = LuongAttention(H, 'dot')
attn_gen = LuongAttention(H, 'general')
attn_cat = LuongAttention(H, 'concat')

print(f'Attention speed (T={T}, B={B}, H={H}):')
print(f'  Dot:     {benchmark_attention(attn_dot, encoder_states, decoder_h):.2f} ms')
print(f'  General: {benchmark_attention(attn_gen, encoder_states, decoder_h):.2f} ms')
print(f'  Concat:  {benchmark_attention(attn_cat, encoder_states, decoder_h):.2f} ms')
# Concat (additive) 가 가장 느림 (per-pair tanh)
```

### 실험 4 — General → Scaled Dot-Product 변환

```python
class ScaledDotProductAttention(nn.Module):
    """Transformer 의 attention"""
    def __init__(self, H, d_K):
        super().__init__()
        self.W_Q = nn.Linear(H, d_K, bias=False)
        self.W_K = nn.Linear(H, d_K, bias=False)
        self.W_V = nn.Linear(H, H, bias=False)
        self.d_K = d_K
    
    def forward(self, query_state, key_states, value_states=None):
        if value_states is None:
            value_states = key_states
        Q = self.W_Q(query_state)        # (B, d_K)
        K = self.W_K(key_states)         # (T, B, d_K)
        V = self.W_V(value_states)       # (T, B, H)
        
        # Score: Q^T K / √d_K
        scores = torch.bmm(
            K.transpose(0, 1),           # (B, T, d_K)
            Q.unsqueeze(2)               # (B, d_K, 1)
        ).squeeze(-1).transpose(0, 1)    # (T, B)
        scores = scores / (self.d_K ** 0.5)
        
        alpha = F.softmax(scores, dim=0)
        context = (alpha.unsqueeze(-1) * V).sum(0)
        return context, alpha

sdp = ScaledDotProductAttention(H=H, d_K=64)
ctx_sdp, alpha_sdp = sdp(decoder_h, encoder_states)
print(f'Scaled dot-product: context {ctx_sdp.shape}')
print(f'Alpha sum: {alpha_sdp.sum(0)[0]:.4f}')
```

### 실험 5 — Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """Transformer multi-head attention"""
    def __init__(self, H, n_heads):
        super().__init__()
        self.H, self.n_heads = H, n_heads
        self.d_K = H // n_heads
        self.W_Q = nn.Linear(H, H, bias=False)
        self.W_K = nn.Linear(H, H, bias=False)
        self.W_V = nn.Linear(H, H, bias=False)
        self.W_O = nn.Linear(H, H, bias=False)
    
    def forward(self, query_state, key_states):
        B = query_state.size(0)
        T = key_states.size(0)
        
        Q = self.W_Q(query_state).view(B, self.n_heads, self.d_K)    # (B, n_h, d_K)
        K = self.W_K(key_states).view(T, B, self.n_heads, self.d_K)
        V = self.W_V(key_states).view(T, B, self.n_heads, self.d_K)
        
        # Score per head
        scores = torch.einsum('btnd,bnd->btn', K.transpose(0, 1), Q) / (self.d_K ** 0.5)   # (B, T, n_h)
        alpha = F.softmax(scores, dim=1)
        context = torch.einsum('btn,btnd->bnd', alpha, V.transpose(0, 1))   # (B, n_h, d_K)
        context = context.reshape(B, self.H)
        return self.W_O(context)

mha = MultiHeadAttention(H=H, n_heads=4)
ctx_mha = mha(decoder_h, encoder_states)
print(f'Multi-head attention: {ctx_mha.shape}')
print(f'Parameters: {sum(p.numel() for p in mha.parameters())}')
# 4 heads × d_K=8 = H=32, with extra W_O
```

---

## 🔗 실전 활용

### 1. NMT 의 표준 attention

Luong general 이 Bahdanau additive 보다 자주 사용 — efficiency.

### 2. Transformer 의 self-attention

Vaswani 2017 의 핵심: Luong 의 idea 를 self-attention 으로 확장.

### 3. BERT, GPT 의 attention

Modern LLM 의 attention 이 Luong 의 직계 후손.

### 4. Vision Transformer (ViT)

Image patches 사이의 attention — Luong 의 정신을 image 로.

### 5. Cross-modal attention

Image-to-text, video-to-language 등 multi-modal attention.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Single attention head | Multi-head 가 더 강력 |
| Sequential decoder | Transformer 가 모든 step 병렬 |
| Local attention 의 Gaussian | Different window shapes 가능 |
| RNN encoder | Self-attention 만으로 충분 (Transformer) |
| BLEU 최적화 | Diversity 부족 — 다양한 sampling |

---

## 📌 핵심 정리

$$\boxed{\text{Luong scoring: dot, general, concat } (h^\top W s \text{ 가 best})}$$

$$\boxed{\text{General → Transformer scaled dot-product: } \frac{K^\top Q}{\sqrt{d_K}}}$$

$$\boxed{\text{Local attention: window around predicted center } p_j}$$

| Scoring | Formula | Speed | BLEU (Luong 2015) |
|---------|---------|-------|-------------------|
| **Dot** | $h^\top s$ | Fastest | 25.4 |
| **General** | $h^\top W s$ | Fast | **25.9** |
| **Concat** | $v^\top \tanh(W [h; s])$ | Slowest | 25.4 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Luong general 의 $W$ matrix 를 low-rank 분해 $W = W_K^\top W_Q$ 로 표현하라. 이것이 어떻게 Transformer 의 query/key separation 으로 이어지는가?

<details>
<summary>해설</summary>

**Luong general**:
$$
e_{ij} = h_i^\top W s_j
$$

**Low-rank decomposition** (rank $d_K \le H$):
$$
W = W_K^\top W_Q, \quad W_K \in \mathbb R^{d_K \times H}, W_Q \in \mathbb R^{d_K \times H}
$$

Then:
$$
e_{ij} = h_i^\top W_K^\top W_Q s_j = (W_K h_i)^\top (W_Q s_j) = K_i^\top Q_j
$$

**Transformer 의 정의**:
- Key: $K_i = W_K h_i$
- Query: $Q_j = W_Q s_j$
- Score: $K^\top Q$

이는 *exactly* Luong general 의 low-rank 형태!

**왜 separation**:

1. **Conceptual**:
   - $K$ = "내가 무엇을 제공하는가" (encoder)
   - $Q$ = "내가 무엇을 찾는가" (decoder)
   - 명시적 query/key separation

2. **Computational**:
   - Pre-compute $K_i$ for all encoder positions
   - $Q_j$ 만 매 decoder step 계산
   - $T \cdot d_K + S \cdot d_K$ projections vs $T \cdot S \cdot H^2$ direct multiplication

3. **Multi-head**:
   - 각 head 가 다른 $W_K, W_Q$
   - 다양한 attention pattern 동시 학습
   - Single $W$ 보다 표현력 강함

4. **Self-attention**:
   - Same input 으로 Q, K, V 모두 생성
   - Sequence 내부 의존성 학습

**Transformer 의 추가**:
- **Value $V = W_V h$**: Attention 의 *output* 도 transform
- **$\sqrt{d_K}$ scaling**: gradient stability
- **Multi-head**: $h$ heads, each $d_K = H / h$

**Lesson**: Luong general 의 *implicit* idea (query/key similarity) 가 Transformer 에서 *explicit* architectural primitive 로 진화. **Decomposition 이 design pattern 의 generalization 을 enable**. $\square$

</details>

**문제 2** (심화): Local attention 의 Gaussian window 가 hard window 보다 우월한 이유? 학습 dynamics 측면에서.

<details>
<summary>해설</summary>

**Hard window**:
$$
\alpha_{ij} = \begin{cases}
\frac{\exp(e_{ij})}{\sum_{k \in [p_j - D, p_j + D]} \exp(e_{kj})} & i \in [p_j - D, p_j + D] \\
0 & \text{otherwise}
\end{cases}
$$

**Gaussian window**:
$$
\alpha_{ij} \propto \exp(e_{ij}) \cdot \exp\left(-\frac{(i - p_j)^2}{2\sigma^2}\right)
$$

**Gaussian 의 우월점**:

1. **Smooth gradient**:
   - Hard window: $i = p_j + D + 1$ 에서 abrupt cut → discontinuous gradient w.r.t. $p_j$
   - Gaussian: smooth, $\partial \alpha / \partial p_j$ well-defined

2. **Center prediction 학습**:
   - $p_j$ 의 학습이 hard window 에서 어려움 (binary in/out)
   - Gaussian 은 *어느 방향* 으로 $p_j$ 이동해야 score 향상 명시

3. **Soft uncertainty**:
   - 정확한 $p_j$ 가 모호한 case (idiom, function word)
   - Gaussian 의 spread 가 soft commitment

4. **Empirical**:
   - Luong 2015 §3.2: Gaussian 이 hard 보다 BLEU +0.5-1

**Hard window 의 장점**:
- Sparser computation (window 외부 무시)
- Speed 우위 (실제 0 entries skip 가능)
- Inference 시 deterministic

**Trade-off**:
- Gaussian: training stable, inference slightly slower
- Hard: training fragile, inference faster

**Modern**:
- Sliding window attention (Longformer): hard window with overlap
- Linear attention (Performer): kernel-based, soft
- Sparse attention (Big Bird): structured sparsity

**결론**: Gaussian window 의 *differentiability* 가 학습에 critical — 이는 "soft attention" 의 기본 정신과 일치. Hard 는 efficiency 의 last resort, soft 가 default. $\square$

</details>

**문제 3** (논문 비평): Luong 2015 가 multiplicative attention 의 우월성을 입증했지만 Transformer 가 더 우월. Luong → Transformer 의 architectural changes 를 정리하고, 각 change 의 motivation 을 분석하라.

<details>
<summary>해설</summary>

**Luong general**:
$$
\begin{aligned}
e_{ij} &= h_i^\top W s_j \\
\alpha_{ij} &= \mathrm{softmax}(e_{ij}) \\
c_j &= \sum_i \alpha_{ij} h_i
\end{aligned}
$$

**Transformer multi-head scaled dot-product self-attention**:
$$
\begin{aligned}
Q, K, V &= h W_Q, h W_K, h W_V \\
\mathrm{head}_i &= \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_K}}\right) V_i \\
\mathrm{MHA}(Q, K, V) &= \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W_O
\end{aligned}
$$

**Architectural changes**:

**1. Q/K/V separation** (low-rank decomposition):
- Luong: Single $W$
- Transformer: $W_Q, W_K, W_V$ 분리
- **Motivation**: Query, key, value 의 *role* 명시화. Value 도 transform 가능.

**2. $\sqrt{d_K}$ scaling**:
- Luong: 없음
- Transformer: 분모에 $\sqrt{d_K}$
- **Motivation**: Large $d_K$ 에서 dot product 가 큰 magnitude → softmax 가 sharp → gradient vanishing. Scaling 으로 magnitude 정규화.

**3. Multi-head**:
- Luong: Single attention
- Transformer: $h$ heads, parallel
- **Motivation**: 한 head 가 한 type 의 relationship 학습 — 다양한 head 가 다양한 pattern (syntactic, semantic, etc.)

**4. Self-attention**:
- Luong: Decoder-encoder cross attention
- Transformer: Self attention (within encoder, within decoder), + cross
- **Motivation**: Sequence 내부 의존성 학습 — RNN 의 sequential dependency 대체

**5. No RNN encoder**:
- Luong: BiLSTM encoder + RNN decoder
- Transformer: Pure self-attention + FFN
- **Motivation**: Sequence parallelism — RNN 의 $O(T)$ sequential bottleneck 회피

**6. Positional encoding**:
- Luong: RNN 이 position 자동 인코딩
- Transformer: Sin/cos PE 또는 learned PE 추가
- **Motivation**: Self-attention 이 position-agnostic — 순서 정보 명시 필요

**7. Layer norm + residual**:
- Luong: 표준 LSTM
- Transformer: Pre-LN 또는 Post-LN + residual
- **Motivation**: Deep network 의 stable training

**8. Feed-forward layer**:
- Luong: 없음 (RNN cell 자체가 nonlinearity)
- Transformer: 매 attention 후 2-layer FFN
- **Motivation**: Position-wise nonlinear transformation, parameter capacity 추가

**Quantitative impact** (BLEU 향상):

| Step | Change | BLEU gain |
|------|--------|-----------|
| Bahdanau (2015) | Additive attention | +5 (vs Seq2Seq) |
| Luong (2015) | Multiplicative + global | +0.5 |
| Q/K/V separation | Better representation | +1 |
| $\sqrt{d_K}$ | Stable gradient | +0.5 |
| Multi-head | Diverse patterns | +1 |
| Self-attention | RNN replacement | +2 |
| Positional encoding | Order info | +0 (necessary) |
| Layer norm + residual | Deeper training | +1 |
| FFN | Capacity | +0.5 |
| **Total Transformer (2017)** | | **+10 vs Bahdanau** |

**Lesson**:

1. **Incremental improvements**:
   - Each change 가 specific limitation 해결
   - Cumulative effect 가 dramatic

2. **Decomposition 의 가치**:
   - Single $W$ → Q/K/V/O separation
   - Single attention → multi-head
   - 모든 step 이 *modularization*

3. **Sequence parallelism 의 importance**:
   - Self-attention 의 가장 큰 contribution
   - RNN 의 sequential 한계를 fundamental 로 해결

4. **Empirical engineering**:
   - $\sqrt{d_K}$ 같은 detail 이 학습 stability 결정
   - Layer norm placement, residual 등의 micro-architecture matters

**현대 (2024)**:
- Transformer 의 attention 이 *standard*
- Variants: Linear attention, Performer, FlashAttention (efficiency)
- Mamba (2023): non-attention RNN-like 의 부활
- LLM 의 success 는 Transformer architecture 의 culmination

**결론**: Luong → Transformer 가 *paradigm shift* 의 *incremental refinement*. **Each change 가 well-motivated 하고 cumulatively dramatic**. 이것이 ML architecture 진화의 ideal pattern — careful engineering + theoretical insight 의 결합. $\square$

</details>

---

<div align="center">

[◀ 이전](./03-bahdanau-attention.md) | [📚 README](../README.md) | [다음 ▶](./05-coverage-pointer.md)

</div>
