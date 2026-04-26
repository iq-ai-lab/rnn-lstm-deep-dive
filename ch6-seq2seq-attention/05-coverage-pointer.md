# 05. Coverage Mechanism 과 Pointer Network

## 🎯 핵심 질문

- Tu 2016 의 *Modeling Coverage for Neural Machine Translation* 가 NMT 의 under-/over-translation 문제를 어떻게 해결하는가?
- **Coverage vector** $c_j^{\text{cov}} = \sum_{j' < j} \alpha_{ij'}$ 가 어떻게 attention 이력을 추적하는가?
- Vinyals 2015 의 **Pointer Network** 가 attention 자체를 output distribution 으로 사용하는 메커니즘?
- $p(y_t = i) = \mathrm{softmax}(e_{ti})$ — combinatorial optimization (TSP, sorting) 응용
- See 2017 의 **Pointer-Generator Network** — copy mechanism + abstractive summarization

---

## 🔍 왜 Coverage 와 Pointer 가 attention 의 자연스러운 확장인가

Bahdanau/Luong attention (Ch6-03, 04) 의 한계:
1. **Repetition**: 같은 source position 을 반복 attend → over-translation
2. **Skipping**: 일부 position 을 ignore → under-translation
3. **OOV problem**: Target vocab 외 단어 (proper noun, rare term) 를 generate 못함
4. **Combinatorial structure**: Output 이 input 의 selection / permutation 일 때 attention 활용 어려움

해법:
1. **Coverage** (Tu 2016): Attention history tracking → repetition penalty
2. **Pointer Network** (Vinyals 2015): Attention as output distribution → selection task
3. **Pointer-Generator** (See 2017): Hybrid generation + copying

이 문서는 두 mechanism 의 정확한 정의, NMT/QA/summarization 의 응용, 그리고 modern descendants (RAG, retrieval) 와의 관계를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [04-luong-attention.md](./04-luong-attention.md) — Luong attention 의 다양한 형태
- (선택) Combinatorial optimization: TSP, sorting

---

## 📖 직관적 이해

### Coverage 의 직관

NMT 의 over-translation 예:
```
Source: "I love you"
Target: "Je t'aime, je t'aime, je t'aime"   ← 반복!
```

원인: $\alpha_{ij}$ 가 매 step 같은 source position 에 attend.

해결: $c_j^{\text{cov}} = \sum_{j' < j} \alpha_{ij'}$ 로 *이미 attended* 인 position 추적, scoring 시 페널티.

### Pointer Network 의 직관

Convex Hull, TSP 같은 combinatorial task:
```
Input:  points [P_1, P_2, P_3, P_4, P_5]   (coordinates)
Output: tour [P_3, P_1, P_4, P_2, P_5]    (indices into input!)
```

Output vocabulary 가 *fixed* 가 아니라 *input-dependent*. Attention weights 가 자연스럽게 "which input element to pick".

```
Pointer:  p(y_t = i) = α_{ti}   ← attention weight as output prob
```

### Pointer-Generator (See 2017) 의 직관

Abstractive summarization 의 hybrid:
- 일반 단어: vocabulary 에서 generate
- Rare / proper noun: source 에서 copy
- $p_{\text{gen}}$: switch between modes

```
p_final = p_gen · p_vocab + (1 - p_gen) · p_attention
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Coverage Vector (Tu 2016)

Decoder step $j$ 까지 누적된 attention:

$$
c_j^{\text{cov}} = \sum_{j'=1}^{j} \alpha_{:,j'} \in \mathbb R^T
$$

(Element-wise sum across decoder steps)

### 정의 5.2 — Coverage-Augmented Score

Standard score $e_{ij}$ 에 coverage 정보 추가:

$$
e_{ij}^{\text{cov}} = e_{ij} + W_c c_{i, j-1}^{\text{cov}}
$$

또는 GRU-style update of coverage state.

### 정의 5.3 — Coverage Loss

Repetition penalty:

$$
\mathcal L_{\text{coverage}} = \sum_j \sum_i \min(\alpha_{ij}, c_{i, j-1}^{\text{cov}})
$$

이 loss 가 같은 position 의 반복 attention 을 discourage.

### 정의 5.4 — Pointer Network (Vinyals 2015)

Encoder-decoder with attention, **output = attention**:

$$
p(y_t = i | y_{<t}, x) = \alpha_{ti} = \mathrm{softmax}_i(e_{ti})
$$

Output dimension = $T$ (input length, *variable*).

### 정의 5.5 — Pointer-Generator Network (See 2017)

Generation probability:

$$
p_{\text{gen}} = \sigma(W_h h_t + W_s s_t + W_x x_t + b)
$$

Final distribution:

$$
p(y_t = w) = p_{\text{gen}} \cdot p_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \cdot \sum_{i: x_i = w} \alpha_{it}
$$

(Vocab 의 token 또는 source 의 token 으로부터 copy)

---

## 🔬 정리와 결과

### 정리 5.1 — Coverage 의 Repetition Reduction

Coverage loss 가 $\alpha_{ij}$ 를 이미 attended position 에서 reduce:

$$
\frac{\partial \mathcal L_{\text{coverage}}}{\partial \alpha_{ij}} > 0 \text{ for } c_{i, j-1}^{\text{cov}} \text{ large}
$$

→ 학습이 새 position 에 attend 하도록 push.

**Empirical** (Tu 2016): WMT'14 En→Fr BLEU +1.5 with coverage.

### 정리 5.2 — Pointer Network 의 Output Vocabulary

Pointer network 의 output vocab = input length $T$. Variable per example.

**Standard attention (cross-entropy)** 와 같은 mechanism, 그러나 *output* 으로 사용.

### 정리 5.3 — Combinatorial Tasks

Pointer network 가 다음 tasks 에서 효과적:
- **Convex hull**: input points 의 boundary 선택
- **TSP**: city sequence 의 permutation
- **Sorting**: input sequence 의 sorted permutation

**Empirical** (Vinyals 2015): TSP 20 cities 에서 ~3% optimality gap.

### 정리 5.4 — Pointer-Generator Hybrid

생성과 복사의 trade-off:

- 일반 vocabulary 단어: $p_{\text{gen}} \to 1$
- Rare / proper noun / number: $p_{\text{gen}} \to 0$

학습이 자동으로 결정.

**Empirical** (See 2017): CNN/Daily Mail summarization ROUGE-1 가 baseline +5.

### 정리 5.5 — Modern Connection (RAG)

Retrieval-Augmented Generation (RAG):
- External document store 에서 retrieve
- LLM 이 retrieved + original query 로 generate

이는 **pointer-generator 의 large-scale 일반화** — copy from external source.

---

## 💻 PyTorch 구현 검증

### 실험 1 — Coverage Mechanism

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoverageAttention(nn.Module):
    """Bahdanau attention + coverage tracking"""
    def __init__(self, H_enc, H_dec, d_attn):
        super().__init__()
        self.W1 = nn.Linear(H_enc, d_attn, bias=False)
        self.W2 = nn.Linear(H_dec, d_attn, bias=False)
        self.W_cov = nn.Linear(1, d_attn, bias=False)   # Coverage projection
        self.v = nn.Linear(d_attn, 1, bias=False)
    
    def forward(self, encoder_states, decoder_h, coverage):
        """
        encoder_states: (T, B, H_enc)
        decoder_h: (B, H_dec)
        coverage: (T, B) — accumulated attention
        """
        T_enc, B, _ = encoder_states.shape
        
        decoder_proj = self.W2(decoder_h).unsqueeze(0)
        encoder_proj = self.W1(encoder_states)
        cov_proj = self.W_cov(coverage.unsqueeze(-1))   # (T, B, d_attn)
        
        scores = self.v(torch.tanh(
            encoder_proj + decoder_proj + cov_proj
        )).squeeze(-1)
        
        alpha = F.softmax(scores, dim=0)
        context = (alpha.unsqueeze(-1) * encoder_states).sum(0)
        
        # Update coverage
        new_coverage = coverage + alpha
        return context, alpha, new_coverage

# Test
T_enc, B, H = 10, 4, 32
attn_cov = CoverageAttention(H, H, d_attn=16)
encoder_states = torch.randn(T_enc, B, H)
decoder_h = torch.randn(B, H)
coverage = torch.zeros(T_enc, B)

# Sequential decoding with coverage
S = 8
all_alphas = []
for s in range(S):
    decoder_h = torch.randn(B, H)
    ctx, alpha, coverage = attn_cov(encoder_states, decoder_h, coverage)
    all_alphas.append(alpha)

print(f'Final coverage shape: {coverage.shape}')
print(f'Coverage values (first sample): {coverage[:, 0]}')   # Sum across decoder steps
print(f'Sum of all alphas equals coverage? {(coverage - sum(all_alphas)).abs().max():.4e}')
```

### 실험 2 — Coverage Loss

```python
def coverage_loss(alphas, coverages):
    """
    alphas: list of (T, B) — attention at each step
    coverages: list of (T, B) — coverage at each step (before update)
    """
    losses = []
    for alpha, cov in zip(alphas, coverages):
        # min(α, cov) — discourage attending where coverage is high
        loss = torch.minimum(alpha, cov).sum(dim=0).mean()
        losses.append(loss)
    return sum(losses)

# 시뮬레이션: coverage 가 고려되지 않은 attention 의 repetition
torch.manual_seed(0)
T = 10
S = 5
alphas_no_cov = [F.softmax(torch.randn(T, 4), dim=0) for _ in range(S)]
coverages_progression = []
cov = torch.zeros(T, 4)
for a in alphas_no_cov:
    coverages_progression.append(cov.clone())
    cov = cov + a

cl_loss = coverage_loss(alphas_no_cov, coverages_progression)
print(f'Coverage loss (random attention): {cl_loss.item():.4f}')
# Random attention 도 일부 repetition 발생 — coverage loss 가 이를 penalize
```

### 실험 3 — Pointer Network 단순 구현

```python
class PointerNetwork(nn.Module):
    """Vinyals 2015 — output is index into input"""
    def __init__(self, D, H):
        super().__init__()
        self.encoder = nn.LSTM(D, H, batch_first=False)
        self.decoder_cell = nn.LSTMCell(D, H)
        self.W_q = nn.Linear(H, H, bias=False)
        self.W_k = nn.Linear(H, H, bias=False)
        self.v = nn.Linear(H, 1, bias=False)
        self.D, self.H = D, H
    
    def forward(self, x_seq, target_indices=None, max_len=None):
        """
        x_seq: (T, B, D)
        target_indices: (S, B) optional ground truth
        """
        T, B, _ = x_seq.shape
        if max_len is None:
            max_len = T
        
        # Encode
        encoder_states, (h, c) = self.encoder(x_seq)   # (T, B, H)
        
        # Decoder: pointer to input
        # Initial input: zero or learned
        dec_input = torch.zeros(B, self.D)
        h, c = h.squeeze(0), c.squeeze(0)
        
        outputs = []
        for s in range(max_len):
            h, c = self.decoder_cell(dec_input, (h, c))
            
            # Attention as pointer
            q = self.W_q(h).unsqueeze(0)   # (1, B, H)
            k = self.W_k(encoder_states)    # (T, B, H)
            scores = self.v(torch.tanh(q + k)).squeeze(-1)   # (T, B)
            log_probs = F.log_softmax(scores, dim=0)
            outputs.append(log_probs)
            
            # Next input: predicted or ground-truth
            if target_indices is not None and s < target_indices.size(0):
                idx = target_indices[s]
            else:
                idx = log_probs.argmax(dim=0)
            
            # Gather: x_seq at idx
            dec_input = x_seq[idx, torch.arange(B), :]   # (B, D)
        
        return torch.stack(outputs)   # (max_len, T, B)

# Toy: sort task — input numbers, output indices for sorted order
torch.manual_seed(0)
D, H, T_seq = 2, 64, 10
ptr_net = PointerNetwork(D, H)

# Generate data
B = 16
inputs = torch.rand(T_seq, B, D)
# Sort by first feature
sort_idx = inputs[:, :, 0].argsort(dim=0)   # (T, B) — indices for sorted

# Train
opt = torch.optim.Adam(ptr_net.parameters(), lr=1e-3)
for step in range(100):
    inputs = torch.rand(T_seq, B, D)
    target = inputs[:, :, 0].argsort(dim=0)
    log_probs = ptr_net(inputs, target_indices=target)
    
    # Cross-entropy
    loss = F.nll_loss(
        log_probs.reshape(-1, T_seq),
        target.reshape(-1)
    )
    opt.zero_grad(); loss.backward(); opt.step()

print(f'Pointer Network sort task final loss: {loss.item():.4f}')

# Test
inputs_test = torch.rand(T_seq, 1, D)
log_probs_test = ptr_net(inputs_test, max_len=T_seq)
predicted = log_probs_test.argmax(dim=1)   # (T_seq, 1)
true_sort = inputs_test[:, :, 0].argsort(dim=0)
print(f'Predicted sort:  {predicted.squeeze().tolist()[:5]}')
print(f'True sort:       {true_sort.squeeze().tolist()[:5]}')
```

### 실험 4 — Pointer-Generator Network

```python
class PointerGenerator(nn.Module):
    """See 2017 — hybrid generation + copying"""
    def __init__(self, V, D, H, d_attn=64):
        super().__init__()
        self.V = V
        self.emb = nn.Embedding(V, D)
        self.encoder = nn.LSTM(D, H, batch_first=False)
        self.decoder_cell = nn.LSTMCell(D, H)
        self.attn = CoverageAttention(H, H, d_attn)
        self.fc_vocab = nn.Linear(H + H, V)   # generation
        self.W_pgen = nn.Linear(H + H + D, 1)   # gate
    
    def forward(self, x_seq, y_seq):
        T, B = x_seq.shape
        
        x_emb = self.emb(x_seq)
        encoder_states, (h, c) = self.encoder(x_emb)
        h, c = h.squeeze(0), c.squeeze(0)
        coverage = torch.zeros(T, B)
        
        outputs_logp = []
        for s in range(y_seq.size(0) - 1):
            y_in = self.emb(y_seq[s])
            ctx, alpha, coverage = self.attn(encoder_states, h, coverage)
            
            # Decoder step
            h, c = self.decoder_cell(y_in, (h, c))
            
            # Generation distribution
            p_vocab = F.softmax(self.fc_vocab(torch.cat([h, ctx], dim=-1)), dim=-1)   # (B, V)
            
            # Generation gate
            p_gen = torch.sigmoid(self.W_pgen(torch.cat([h, ctx, y_in], dim=-1)))   # (B, 1)
            
            # Final distribution: p_gen * p_vocab + (1 - p_gen) * p_attention (over input vocab)
            # Map alpha (T, B) to vocab dim by indexing source tokens
            p_copy = torch.zeros(B, self.V)
            for b in range(B):
                for t in range(T):
                    p_copy[b, x_seq[t, b]] += alpha[t, b]
            
            p_final = p_gen * p_vocab + (1 - p_gen) * p_copy
            log_p = torch.log(p_final + 1e-9)
            outputs_logp.append(log_p)
        
        return torch.stack(outputs_logp), coverage

V, D, H = 100, 16, 32
torch.manual_seed(0)
pg = PointerGenerator(V, D, H)
x = torch.randint(0, V, (8, 2))
y = torch.randint(0, V, (5, 2))
log_p, cov = pg(x, y)
print(f'Pointer-Generator output: {log_p.shape}')   # (S-1, B, V)
print(f'Final coverage: {cov.shape}')
```

### 실험 5 — Coverage Effect on Repetition

```python
# Coverage 사용 vs 미사용 시 repetition 비교
def compute_repetition(predictions):
    """% of repeated tokens"""
    flat = predictions.flatten()
    unique = len(set(flat.tolist()))
    return 1.0 - unique / len(flat)

torch.manual_seed(0)
# Without coverage: simulate attention always picking same position
no_cov_preds = torch.tensor([5, 5, 5, 7, 5, 5])
print(f'Without coverage repetition: {compute_repetition(no_cov_preds)*100:.1f}%')

# With coverage: attention spreads
with_cov_preds = torch.tensor([5, 7, 3, 8, 1, 9])
print(f'With coverage repetition:    {compute_repetition(with_cov_preds)*100:.1f}%')
```

---

## 🔗 실전 활용

### 1. NMT with coverage

Phrase-level coverage for under/over-translation. WMT 표준 baseline.

### 2. Combinatorial optimization

Pointer Network 가 TSP, sorting, convex hull 의 ML approach. Recent: graph neural network + pointer.

### 3. Abstractive summarization

Pointer-Generator (See 2017) 가 CNN/DailyMail SOTA. 후속 BERT-based summarization 의 기반.

### 4. Question Answering

Reading comprehension 에서 answer span 을 input 에서 copy — pointer mechanism.

### 5. Retrieval-Augmented Generation (RAG)

Modern LLM + external knowledge — pointer-generator 의 large-scale 일반화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Coverage 가 모든 task 에 도움 | Some tasks (creative generation) 에서 coverage 가 한계 |
| Pointer 의 differentiable | Combinatorial constraint 처리 어려움 |
| Generation + copying 분리 | Continuous mixing 보다 discrete switch 가 limit |
| Attention 자체로 충분 | Multi-step reasoning 에는 inadequate |
| Single source | Multiple sources / multi-document 확장 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{Coverage: } c_j^{\text{cov}} = \sum_{j' < j} \alpha_{:,j'}}$$

$$\boxed{\text{Pointer: } p(y_t = i) = \alpha_{ti} \text{ (attention as output)}}$$

$$\boxed{\text{Pointer-Gen: } p = p_{\text{gen}} \cdot p_{\text{vocab}} + (1 - p_{\text{gen}}) \cdot p_{\text{copy}}}$$

| Mechanism | Purpose | Application |
|-----------|---------|-------------|
| **Coverage** | Repetition penalty | NMT, summarization |
| **Pointer Network** | Output = input index | TSP, sorting, parsing |
| **Pointer-Generator** | Generate + copy | Abstractive summarization |
| **RAG (modern)** | External knowledge | LLM + retrieval |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Coverage vector $c_j^{\text{cov}}$ 의 sum across all positions 는 무엇인가? 매 step 후 어떻게 변하는가?

<details>
<summary>해설</summary>

**Coverage definition**:
$$
c_j^{\text{cov}} = \sum_{j'=1}^{j} \alpha_{:,j'} \in \mathbb R^T
$$

**Sum across positions**:
$$
\sum_{i=1}^{T} c_{i,j}^{\text{cov}} = \sum_{j'=1}^{j} \sum_{i=1}^{T} \alpha_{ij'} = \sum_{j'=1}^{j} 1 = j
$$

(각 step 의 attention 이 합 1)

**Step 별 변화**:
- $j = 0$: sum = 0
- $j = 1$: sum = 1
- $j = 2$: sum = 2
- $j = T$: sum = $T$

**의미**:
- Coverage 의 total mass 가 decoder step 수만큼 증가
- 균등 분포 시 각 position 에 평균 $j/T$ attention
- Repetition 시 일부 position 에 집중 (variance 큼)

**Coverage loss interpretation**:
- $\min(\alpha_{ij}, c_{i,j-1}^{\text{cov}})$
- $c_{i,j-1}^{\text{cov}}$ 가 큰 position 에서 $\alpha_{ij}$ 가 작아지도록 학습
- Total mass conservation: 다른 position 으로 attention 분산

$\square$

</details>

**문제 2** (심화): Pointer Network 가 standard Seq2Seq 보다 combinatorial task 에 우월한 이유를 분석하라. Output vocabulary 의 *variable* size 가 핵심?

<details>
<summary>해설</summary>

**Standard Seq2Seq 의 한계**:
- Output vocab $|V|$ fixed
- TSP 의 city 수가 가변 → 어떻게 표현?
- Naive: city ID 를 token (1, 2, ..., 100) → vocab size 가 max possible cities

**Pointer Network 의 우월점**:

1. **Variable output vocab**:
   - Output dim = input length $T$
   - 매 example 의 $T$ 가 다름 — fixed vocab 불필요
   - Generalization: 학습된 model 이 다른 size 의 problem 도 풀 수 있음

2. **Permutation invariance**:
   - TSP: 같은 problem 의 city order 가 다양
   - Pointer attention 이 implicit invariance
   - Standard Seq2Seq 은 explicit augmentation 필요

3. **Attention 의 natural interpretation**:
   - "Pick element $i$" = $\alpha_i$ high
   - 이미 선택된 element 는 mask out — combinatorial constraint

4. **Generalization across sizes**:
   - 학습: 5-10 cities
   - Test: 20-50 cities
   - Pointer network 가 generalize (Vinyals 2015)
   - Standard Seq2Seq 은 fail (vocab mismatch)

**Empirical** (Vinyals 2015):
- TSP 50 cities: pointer network ~3% optimality gap
- Convex hull: 99% accuracy
- Sorting: 100% accuracy on small sequences

**한계**:
- Large combinatorial problems (1000+ items): exponential search space
- Hard constraints (e.g., 정확한 valid solution) — soft attention 의 한계
- Modern: RL + pointer (Bello 2017) 가 더 효과적

**Modern alternatives**:
- Graph Neural Networks + pointer (Khalil 2017)
- Transformer-based combinatorial models
- LLM 의 in-context combinatorial solver

**결론**: Pointer Network 의 *idea* (attention as selection) 가 NLP 외의 domain (combinatorial optimization) 에 attention paradigm 확장. **Output vocabulary 의 dynamic 정의** 가 핵심 — 이는 modern code generation, structured prediction 의 정신과 연결. $\square$

</details>

**문제 3** (논문 비평): Pointer-Generator 의 copy mechanism 이 OOV problem 을 해결한다. RAG (Retrieval-Augmented Generation) 가 이를 어떻게 large-scale 로 generalize 하는가?

<details>
<summary>해설</summary>

**Pointer-Generator (See 2017) 의 mechanism**:
- Input: source document
- Output: summary
- Copy: source 의 token 을 직접 사용 (proper noun, rare term)
- Generate: vocabulary 에서 새 token

**OOV (Out-Of-Vocabulary) 문제 해결**:
- Vocab 외 단어 (e.g., specific names, dates) 가 source 에 있으면 copy 가능
- Standard Seq2Seq 은 `<UNK>` 출력
- Pointer-Gen 은 정확한 단어 재현

**RAG (Retrieval-Augmented Generation)**:

1. **Architecture**:
   - Query: user question
   - Retriever: large document store 에서 relevant passage 검색 (e.g., DPR)
   - Generator: LLM (BART, T5) 가 retrieved + query 로 generate

2. **Pointer-Generator 와의 유사점**:
   - Copy mechanism: retrieved passage 에서 token 가져옴
   - Generation: LLM 의 vocabulary
   - Hybrid: explicit copy + generative

3. **Pointer-Generator 와의 차이**:
   - Source: 단일 document (Pointer-Gen) vs millions of documents (RAG)
   - Retrieval: implicit attention (Pointer-Gen) vs explicit retrieval step (RAG)
   - Scale: small (10s of papers) vs large (Wikipedia)

4. **RAG 의 component**:
   - **Retrieval**: dense vector search (FAISS) 또는 sparse (BM25)
   - **Reading**: retrieved passage 를 LLM 에 prepend
   - **Generation**: LLM 이 augmented context 로 답변

**Generalization steps**:

| | Pointer-Generator | RAG |
|--|-------------------|-----|
| **Source** | Single document | Document store |
| **Selection** | Attention (soft) | Retrieval (explicit) |
| **Generation** | RNN decoder | LLM |
| **Copy** | $\alpha$-weighted source tokens | Retrieved text in context |
| **Scale** | $10^3$ tokens | $10^9$ tokens |

**Modern RAG 의 evolution**:

1. **REALM** (Guu 2020): retrieval + masked LM pre-training
2. **RAG** (Lewis 2020): retrieval + seq2seq generation
3. **Atlas** (Izacard 2022): few-shot RAG
4. **Self-RAG** (Asai 2024): adaptive retrieval

**Lesson**:

1. **Attention 의 generalization**:
   - Pointer-Gen: attention over source positions
   - RAG: attention (explicit) over external documents
   - LLM in-context: attention over context tokens
   - **Same idea**, different scales

2. **Copy mechanism 의 universality**:
   - 명시적 copy (Pointer-Gen)
   - Implicit copy (LLM 의 in-context learning)
   - External retrieval (RAG)
   - 모두 *external information* 의 활용

3. **Knowledge access 의 paradigm shift**:
   - Pre-RAG: parametric knowledge (model weights)
   - RAG: non-parametric (retrieve + use)
   - Hybrid: both

4. **Architecture 의 progression**:
   - Pointer-Gen (2017): single document copy
   - RAG (2020): multi-document retrieval
   - LLM + tools (2024): execution + retrieval + reasoning

**현대 perspective**:

- LLM agent (GPT-4, Claude) 의 tool use = RAG 의 generalization
- Retrieval 이 standard component
- Pointer-Gen 의 *idea* 가 evolve, *implementation* 은 LLM-based

**결론**: Pointer-Generator 의 copy mechanism 이 *seed idea*. RAG 가 이를 large-scale knowledge retrieval 로 확장. Modern LLM 의 retrieval-augmented paradigm 의 정신적 조상. **Architecture 의 idea 가 scale 에 따라 진화 — 같은 정신, 다른 instantiation**. $\square$

</details>

---

<div align="center">

[◀ 이전](./04-luong-attention.md) | [📚 README](../README.md) | [다음 ▶](../ch7-modern-alternatives/01-parallelism-limit.md)

</div>
