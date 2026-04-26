# 02. Information Bottleneck Problem

## 🎯 핵심 질문

- Seq2Seq 의 fixed vector $v$ 가 왜 긴 sentence 에서 정보 병목이 되는가?
- $\dim(v) = H$ (예: 1024) 가 50+ word sentence 의 모든 정보를 정량적으로 인코딩 가능한가?
- **Cho 2014b** 의 "long sentence curse" — BLEU 가 sentence length 에 따라 단조 감소하는 실증적 발견
- Bottleneck 의 information-theoretic 한계와 Bahdanau attention 의 해결 메커니즘
- Decoder 의 모든 step 이 같은 $v$ 만 보는 한계 — input 의 specific position 에 attend 불가

---

## 🔍 왜 Information Bottleneck 이 attention 의 동기인가

Sutskever 2014 의 Seq2Seq 가 NMT 에 첫 NN 접근이지만 한계 명확:
- Encoder 의 *마지막* hidden 만 사용
- 모든 input 정보가 *fixed-size* vector 에 압축
- Long sentence 에서 catastrophic loss

**Cho 2014b** *On the Properties of Neural Machine Translation: Encoder-Decoder Approaches* 가 이 한계를 정량화:
- 30+ word sentence 에서 BLEU 급감
- 이 발견이 attention (Ch6-03) 의 직접적 동기

이 문서는 bottleneck 의 정확한 분석과 attention 의 motivational basis 를 제공합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-encoder-decoder.md](./01-encoder-decoder.md) — Seq2Seq architecture
- (선택) 정보이론: Channel capacity, mutual information
- (선택) Compression theory: Rate-distortion

---

## 📖 직관적 이해

### Fixed Vector 의 압축 한계

```
Input (50 words):  "The cat sat on the mat ... happy ever after."
                          ↓
                    Encoder LSTM
                          ↓
                    v ∈ ℝ^1024 (one vector!)
                          ↓
                    Decoder LSTM
                          ↓
Output: "Le chat ... heureux pour toujours."
```

50 words 의 정보를 1024-dim vector 에 압축 — quantitatively 가능?

### 정보량 계산

- Vocab $|V| = 30{,}000$
- Sentence $T = 50$
- Total info: $T \log_2 |V| = 50 \times 15 = 750$ bits

- Vector $v \in \mathbb R^{1024}$ (float32)
- Theoretical max: $1024 \times 32 = 32{,}768$ bits
- Effective precision (gradient learning): ~10 bits per dim
- Effective: $\sim 10{,}000$ bits

**얼핏 capacity 충분**. 그러나 *학습된* model 의 effective bits 는 훨씬 작음 (학습 dynamics 의 한계).

### Empirical Evidence (Cho 2014b)

```
BLEU vs sentence length (English-French):

         BLEU
         30 ┤ ●●●  
            │   ●●●
         25 ┤      ●●  
            │        ●●
         20 ┤          ●●  
            │            ●●●
         15 ┤               ●●●
            │                  ●●●
         10 ┤                     ●●●●●
            └──────────────────────────────→
            10    20    30    40    50    60    sentence length
```

30+ word 에서 급감 — fixed vector 의 한계.

### Decoder 의 한계

각 decoder step $s_t$ 가 같은 $v$ 만 봄:
- $s_1$ 는 input 의 *first* word 정보 필요
- $s_5$ 는 input 의 *fifth* word 정보 필요
- 그러나 둘 다 *같은* $v$ 에서 추출

이는 sub-optimal — *position-specific* information access 부재.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Information Bottleneck

Encoder $f_{\text{enc}}: \mathcal X^T \to \mathbb R^H$ 가 압축. **Bottleneck capacity**:

$$
I(X; V) \le H(V) \le H \cdot \log_2(\text{precision})
$$

Mutual information 의 channel capacity bound.

### 정의 2.2 — Long Sentence Curse

$T$ 가 증가함에 따라 BLEU 가 monotone decrease:

$$
\frac{d \text{BLEU}}{dT} < 0 \quad \text{(empirically)}
$$

특히 $T > 30$ 에서 가속.

### 정의 2.3 — Effective Capacity

학습된 encoder 의 *effective* capacity:

$$
H_{\text{eff}}(V) = \mathbb E_{x \sim p_{\text{data}}}[\text{var of } V | X = x]
$$

— 학습된 representation 의 distinguishable values 수.

### 정의 2.4 — Attention 의 Capacity Multiplier

Attention 이 모든 $h_1, \ldots, h_T$ 보존 → capacity:

$$
H_{\text{attention}} = T \cdot H \quad \text{(모든 encoder hidden)}
$$

Sentence length 에 *비례* 한 capacity.

### 정의 2.5 — Decoder Position-Wise Information

Decoder step $t$ 의 *ideal* information access:

$$
I_t^{\text{ideal}} = \text{aligned}(y_t, x_{:T})
$$

NMT 에서 monotonic alignment 가정 시 $I_t \approx h_{a(t)}^{\text{enc}}$, $a(t)$ = alignment.

---

## 🔬 정리와 결과

### 정리 2.1 — Cho 2014b 의 Empirical Finding

WMT'14 En→Fr Seq2Seq + LSTM:
- Sentence length 0-10: BLEU ~30
- Length 20-30: BLEU ~25
- Length 40-50: BLEU ~17
- Length 60+: BLEU ~10

**Curve fitting**: BLEU $\approx a - b \cdot T^c$ for $c > 0$.

### 정리 2.2 — Information-Theoretic Lower Bound on Loss

Channel capacity argument:
$$
\mathbb E[\text{loss}] \ge -\frac{I(Y; X|V)}{T} \ge -\frac{H(V)}{T}
$$

(Information lost in compression manifests as inevitable loss)

### 정리 2.3 — Hidden State Capacity 의 학습 dynamics

학습 중 $H(V)$ effective:
- Random init: $\sim H \log_2(\text{init scale})$
- 학습 후: data 의 *relevant* information capacity

학습이 task 에 important 한 정보를 prioritize — 그러나 long sequence 에서 어떤 정보를 keep / discard 결정 어려움.

### 정리 2.4 — Attention 의 Capacity Improvement

Attention 의 effective capacity:

$$
H_{\text{attention}}^{\text{eff}} = T \cdot H
$$

Linear in $T$. Sentence length 에 따른 capacity scaling.

**비교 vanilla Seq2Seq**: $H$ (constant in $T$).

### 정리 2.5 — Soft vs Hard Alignment

Soft attention (Bahdanau): $\alpha_{ij} \in [0, 1]$, $\sum_i \alpha_{ij} = 1$.
Hard attention: $\alpha_{ij} \in \{0, 1\}$ (one-hot).

Soft 의 capacity = $T \cdot H$ (모든 information mixed), hard 의 capacity = $H$ (single position selected).

**Differentiable**: Soft 만, hard 는 RL 또는 Gumbel-softmax.

---

## 💻 PyTorch 검증

### 실험 1 — Information Loss Empirical Measurement

```python
import torch
import torch.nn as nn
import numpy as np

def measure_information_preservation(model, V_in, T_test, n_samples=200):
    """Encoder 가 input 정보를 얼마나 보존하는지 측정"""
    # Random sequences
    sequences = torch.randint(0, V_in, (T_test, n_samples))
    
    # Encode
    with torch.no_grad():
        x_emb = model.emb_in(sequences)
        _, (h_T, c_T) = model.encoder(x_emb)
        v = h_T.squeeze(0)   # (n_samples, H)
    
    # 첫 token 의 reconstruction (autoencoder 만)
    # 또는 직접 측정: distinct sequences 가 distinct v 를 만드는지
    
    # Pairwise distance in v space vs original sequences
    v_dists = torch.cdist(v, v)
    seq_dists = torch.cdist(sequences.float().T, sequences.float().T)   # naive
    
    correlation = torch.corrcoef(torch.stack([v_dists.flatten(), seq_dists.flatten()]))[0, 1]
    return correlation.item()

# Train Seq2Seq autoencoder briefly
class SimpleSeq2Seq(nn.Module):
    def __init__(self, V, D=32, H=64):
        super().__init__()
        self.emb_in = nn.Embedding(V, D)
        self.emb_out = nn.Embedding(V, D)
        self.encoder = nn.LSTM(D, H)
        self.decoder = nn.LSTM(D, H)
        self.fc = nn.Linear(H, V)
    
    def forward(self, x, y):
        x_emb = self.emb_in(x)
        _, (h, c) = self.encoder(x_emb)
        y_emb = self.emb_out(y[:-1])
        out, _ = self.decoder(y_emb, (h, c))
        return self.fc(out)

torch.manual_seed(0)
V_in = 50
ae = SimpleSeq2Seq(V_in, D=32, H=64)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)

# Train as autoencoder
for step in range(100):
    T = torch.randint(5, 30, (1,)).item()
    x = torch.randint(0, V_in, (T, 16))
    logits = ae(x, x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, V_in), x[1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

# Measure preservation at different lengths
for T in [5, 10, 20, 30, 50]:
    corr = measure_information_preservation(ae, V_in, T)
    print(f'T = {T:3d}: correlation between v-distance and seq-distance: {corr:.4f}')
# T 클수록 correlation 감소 (information loss)
```

### 실험 2 — Reconstruction Accuracy vs Length

```python
def autoencode_accuracy(model, V_in, T_test, n_samples=100):
    """Length T 의 sequence 를 얼마나 정확히 재구성하는가"""
    correct = 0
    total = 0
    for _ in range(n_samples):
        x = torch.randint(0, V_in, (T_test, 1))
        with torch.no_grad():
            x_emb = model.emb_in(x)
            _, (h, c) = model.encoder(x_emb)
            
            # Decode
            B = 1
            y = torch.zeros(B, dtype=torch.long)   # <bos>
            preds = []
            for t in range(T_test - 1):
                y_emb = model.emb_out(y).unsqueeze(0)
                out, (h, c) = model.decoder(y_emb, (h, c))
                logits = model.fc(out.squeeze(0))
                y = logits.argmax(-1)
                preds.append(y.clone())
            
            preds = torch.stack(preds).squeeze(-1)
            target = x[1:].squeeze(-1)
            correct += (preds == target).sum().item()
            total += T_test - 1
    return correct / total

print('Reconstruction accuracy by length:')
for T in [5, 10, 20, 30, 50]:
    acc = autoencode_accuracy(ae, V_in, T)
    print(f'  T = {T:3d}: {acc:.4f}')
# T 클수록 accuracy 감소 — bottleneck
```

### 실험 3 — Compression Ratio Analysis

```python
# 정량적으로: input 정보량 vs vector capacity
V_in = 50
H = 64

import math

print('Input vs vector capacity:')
print(f'Vector v ∈ R^{H} (float32):')
print(f'  Theoretical max bits: {H * 32}')
print(f'  Effective bits (heuristic): {H * 8:.0f}')

for T in [5, 10, 20, 50, 100]:
    input_bits = T * math.log2(V_in)
    print(f'  T={T:3d}, |V|={V_in}: input bits = {input_bits:.1f}')

# T > 50 시 input bits > effective vector bits
```

### 실험 4 — Information Bottleneck Visualization

```python
import matplotlib.pyplot as plt

# 다양한 T 의 BLEU-like 측정 (random task simulation)
T_range = list(range(5, 60, 5))
accs = []
for T in T_range:
    acc = autoencode_accuracy(ae, V_in, T, n_samples=50)
    accs.append(acc)

plt.figure(figsize=(8, 5))
plt.plot(T_range, accs, 'o-')
plt.xlabel('Sequence length T')
plt.ylabel('Reconstruction accuracy')
plt.title('Bottleneck Effect: Accuracy decreases with length')
plt.grid(True, alpha=0.3)
plt.savefig('bottleneck.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved: bottleneck.png')
```

### 실험 5 — Attention 의 Capacity (Sketch)

```python
# Attention 시 *모든* encoder state 보존 — capacity 가 T 에 비례
class AttentionSeq2Seq(nn.Module):
    def __init__(self, V, D=32, H=64):
        super().__init__()
        self.emb_in = nn.Embedding(V, D)
        self.emb_out = nn.Embedding(V, D)
        self.encoder = nn.LSTM(D, H)
        self.decoder = nn.LSTMCell(D + H, H)
        self.attn = nn.Linear(2 * H, 1)
        self.fc = nn.Linear(H, V)
        self.H = H
    
    def attention(self, encoder_states, decoder_h):
        """Bahdanau-style additive attention"""
        T_enc = encoder_states.size(0)
        # Repeat decoder_h for all encoder positions
        decoder_h_exp = decoder_h.unsqueeze(0).expand(T_enc, -1, -1)
        # Concatenate
        combined = torch.cat([encoder_states, decoder_h_exp], dim=-1)
        scores = self.attn(combined).squeeze(-1)   # (T_enc, B)
        alpha = torch.softmax(scores, dim=0)
        context = (alpha.unsqueeze(-1) * encoder_states).sum(0)
        return context, alpha

# 더 자세한 attention 구현은 Ch6-03 에서
print('Attention 의 capacity = T × H (encoder의 모든 state 보존)')
print('Vanilla bottleneck = H (단일 vector)')
```

---

## 🔗 실전 활용

### 1. Vanilla Seq2Seq 의 한계 인식

새 NMT 모델 설계 시 *반드시* attention 또는 transformer 사용. Vanilla 는 toy task 만.

### 2. Encoder Bidirectionality

BiLSTM encoder 가 fixed vector 의 capacity 를 두 배 ($2H$). 부분적 보완.

### 3. Hierarchical Encoder

Sentence → paragraph → document 의 multi-level encoding (Yang 2016). Bottleneck 의 layer-wise reduction.

### 4. Document QA / Summarization

긴 document 처리 시 chunk-based 또는 retrieval-augmented 접근.

### 5. Modern transformer

Attention 이 표준 — bottleneck 자체가 사라짐. 그러나 long context 에서 다시 issue (Ch7-03 의 linear attention 동기).

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Single vector $v$ 압축 | Attention 이 T × H 로 확장 |
| Float precision 충분 | Gradient learning 의 effective bits 작음 |
| Static capacity | Dynamic capacity (e.g., variable hidden size) 가능 |
| Sentence-level | Document level 은 더 큰 bottleneck |
| Encoder-decoder split | Joint encoder-decoder (Transformer) 가 더 flexible |

---

## 📌 핵심 정리

$$\boxed{\text{Vanilla Seq2Seq capacity: } H \text{ (constant)}}$$

$$\boxed{\text{Attention capacity: } T \cdot H \text{ (linear in length)}}$$

$$\boxed{\text{Long sentence curse: BLEU} \downarrow \text{ as } T \uparrow \text{ (Cho 2014b)}}$$

| Architecture | Capacity scaling | Long sentence |
|--------------|-----------------|----------------|
| **Vanilla Seq2Seq** | $O(H)$ | Catastrophic |
| **BiLSTM encoder** | $O(2H)$ | 약간 개선 |
| **Attention (Bahdanau)** | $O(T \cdot H)$ | Robust |
| **Transformer** | $O(T \cdot H)$ | 매우 robust |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T = 100$, $|V| = 50000$, $H = 1024$ 인 Seq2Seq 의 input 정보량 vs vector capacity 를 비교하라.

<details>
<summary>해설</summary>

**Input 정보**:
$$
T \log_2 |V| = 100 \times \log_2 50000 = 100 \times 15.6 = 1560 \text{ bits}
$$

**Vector $v \in \mathbb R^{1024}$ (float32)**:
- Theoretical max: $1024 \times 32 = 32{,}768$ bits
- Effective (gradient learning): $\sim 1024 \times 8 = 8192$ bits

**비교**: 8192 > 1560 — **이론적으로 capacity 충분**.

**그러나 empirical**:
- 학습된 model 의 effective capacity 는 *학습 dynamics*, *task complexity* 에 의존
- 모든 capacity 가 task-relevant 정보로 사용되는 것은 아님
- Cho 2014b 의 BLEU drop 이 T = 30 에서 시작 — capacity 는 충분하지만 *학습이 어려움*

**Lesson**:
- Capacity ≠ usable capacity
- Architecture (attention) 이 capacity 를 더 효율적으로 사용
- Long sequence: bottleneck *information theoretic* 보다 *optimization theoretic*

$\square$

</details>

**문제 2** (심화): Information bottleneck 이 *information-theoretic* 한계인가 *optimization* 한계인가? 구분하여 분석하라.

<details>
<summary>해설</summary>

**Information-theoretic 한계**:
- Channel capacity: $H \cdot \log_2(\text{precision})$ bits
- Sufficient statistic 으로 $V$ 가 input 의 모든 *task-relevant* info 보존 가능 시 OK
- 문제: 무엇이 task-relevant 인지 학습이 결정

**Optimization 한계**:
- Gradient-based learning 의 capacity utilization 효율
- 학습된 representation 의 *effective* dimensionality
- Local minima, saddle points 의 영향

**Empirical 분석**:

1. **BLEU 가 T=30 에서 drop**:
   - Vector capacity (1024 × 32 bits = 32K bits) >> input info (30 × 15 bits = 450 bits)
   - 이론적으로 capacity 충분
   - 즉 **information-theoretic 한계 아님**

2. **학습 dynamics 의 영향**:
   - Long sequence 의 BPTT vanishing gradient
   - Encoder 가 *처음* tokens 의 정보를 *끝* hidden 에 보존하기 어려움
   - **Optimization 한계**

3. **Attention 의 효과**:
   - 같은 hidden size $H$, 그러나 모든 $h_t$ access
   - Capacity 가 $T H$ 로 증가 (information-theoretic)
   - **그리고** vanishing gradient 우회 (optimization)
   - 두 한계 모두 해결

4. **Transformer 의 추가 개선**:
   - Attention 이 capacity 를 늘리고 vanishing 도 우회
   - Layer norm, residual 이 optimization 더 stable
   - Sequence parallelism 으로 training efficient

**구분**:

| Aspect | Information-theoretic | Optimization |
|--------|----------------------|---------------|
| **Vanilla Seq2Seq bottleneck** | Marginal (capacity 충분) | **Major (vanishing)** |
| **Long-document handling** | Critical | Critical |
| **Attention 의 효과** | Capacity 확장 | Vanishing 우회 |
| **Transformer 의 추가** | 같은 capacity | + parallelism |

**결론**: Bottleneck 은 *주로* optimization 한계 — capacity 자체는 충분하나 학습이 모든 capacity 를 효과적으로 사용 못함. **Attention 이 capacity + optimization 둘 다 해결**. 이것이 attention 이 NMT revolution 의 결정적 idea 인 이유. $\square$

</details>

**문제 3** (논문 비평): Cho 2014b 가 long sentence curse 를 *진단* 했지만 해법은 제시 안 함. Bahdanau 2015 가 attention 으로 해결 — 두 paper 의 *role distinction* 과 ML research 의 problem-solution dynamics 를 논하라.

<details>
<summary>해설</summary>

**Cho 2014b 의 contribution**:
- Empirical finding: BLEU vs length 의 monotone decrease
- Architecture analysis: encoder의 fixed-vector bottleneck 진단
- **No solution proposed** — 한계만 명시

**Bahdanau 2015 의 contribution**:
- Cho 2014b 의 진단을 *acknowledge*
- Attention mechanism 제안: $e_{ij} = v^T \tanh(W_1 h_i + W_2 s_j)$
- Empirical: long sentence 에서 dramatic improvement
- **Solution to identified problem**

**Role distinction**:

1. **Diagnostic paper (Cho 2014b)**:
   - Problem identification
   - Empirical evidence 제공
   - Future research direction 제안

2. **Solution paper (Bahdanau 2015)**:
   - Architectural innovation
   - 진단된 문제의 직접 해결
   - Quantitative validation

**ML Research Dynamics**:

이 패턴이 ML 진화의 핵심:

1. **Problem awareness**:
   - Empirical observation (e.g., Cho 의 BLEU drop)
   - Theoretical limitation 분석 (e.g., Pascanu 2013 의 vanishing)
   - Community 가 *문제* 인식

2. **Multiple solution attempts**:
   - 여러 group 이 다른 approach 시도
   - Bahdanau 의 attention 외에도: hierarchical encoding, copy mechanism, etc.
   - **Best solution** 이 emerge

3. **Standard 화**:
   - Effective solution 이 표준
   - Community adopts
   - Future research 의 base

**Cho-Bahdanau pair 의 흥미로운 점**:

- 둘 다 Cho 가 co-author!
- Cho 가 *문제 인식* 후 *직접 해결* 시도
- Same researcher 가 disease 와 cure 모두 제시 — 강력한 paper pair

**다른 problem-solution pairs**:

| Problem (paper) | Solution (paper) |
|-----------------|------------------|
| Vanishing gradient (Bengio 1994, Pascanu 2013) | LSTM (Hochreiter 1997), gradient clipping (Pascanu 2013) |
| Over-smoothing in GCN (Li 2018) | DropEdge (Rong 2020), APPNP (Klicpera 2019) |
| Mode collapse in GAN (Goodfellow 2014) | WGAN (Arjovsky 2017), progressive GAN (Karras 2017) |

**Lesson**:

1. **진단의 가치**:
   - Empirical observation 이 theoretical analysis 만큼 가치
   - Cho 2014b 가 attention 의 motivational basis
   - **문제 인식이 첫 단계**

2. **Solution 의 timing**:
   - Bahdanau 2015 가 Cho 2014b 의 *직후* — community 의 sense of urgency
   - Open problem 의 명확한 정의 가 빠른 해결 inspire

3. **Iterative refinement**:
   - Bahdanau attention → Luong attention (Ch6-04) → Transformer (2017)
   - 각 step 이 previous 의 한계 해결
   - **Cumulative progress**

**현대 perspective**:

- Diagnostic + solution paper pair 가 ML 의 standard pattern
- Modern: scaling laws (Kaplan 2020) → emergence (Wei 2022) → fine-tuning (RLHF) — 비슷한 dynamics
- "Identifying the next bottleneck" 이 research 의 core skill

**결론**: Cho 2014b 와 Bahdanau 2015 가 *paradigmatic* problem-solution pair. **진단의 명확함이 빠른 해결을 enable**. ML 의 progress 는 *정확한 문제 정의* 와 *creative architectural solution* 의 iteration. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-encoder-decoder.md) | [📚 README](../README.md) | [다음 ▶](./03-bahdanau-attention.md)

</div>
