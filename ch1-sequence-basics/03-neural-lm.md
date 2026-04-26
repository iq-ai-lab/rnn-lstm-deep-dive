# 03. Neural Language Model (Bengio 2003)

## 🎯 핵심 질문

- Bengio 2003 의 Neural Probabilistic Language Model 이 어떻게 **embedding $C \in \mathbb{R}^{|V| \times d}$** 와 feed-forward NN 으로 N-gram 의 sparsity 문제를 해결했는가?
- "Curse of dimensionality" 에서 vocab $|V|$ 의 vs context length $n$ 의 차원이 각각 어떻게 NLM 으로 완화되는가?
- Word embedding 이 왜 단어의 **semantic similarity** 를 자동으로 학습하는가?
- Bengio 2003 의 fixed-window 한계는 무엇이며 RNN (Ch1-04) 이 이를 어떻게 우회하는가?
- NLM 에서 **softmax bottleneck** 은 무엇이며 negative sampling, hierarchical softmax, NCE 가 어떻게 해결하는가?

---

## 🔍 왜 NLM 이 sequence 학습의 패러다임 전환인가

Bengio 2003 의 *A Neural Probabilistic Language Model* 은 단순한 N-gram 대안 그 이상입니다. 이 논문은 다음 세 가지 통찰로 NLP 의 신경망 시대를 열었습니다:

1. **Word as dense vector** — Discrete one-hot 이 아니라 $\mathbb{R}^d$ 의 embedding 으로 표현, 단어 간 similarity 자동 학습
2. **Distributed representation** — N-gram 의 sparse count 대신 NN 의 중간 layer 가 단어의 semantic 을 분산 인코딩
3. **Generalization to unseen $n$-grams** — 본 적 없는 trigram 도 비슷한 단어들의 embedding 을 통해 합리적 확률 부여

이 세 가지 통찰이 word2vec, GloVe, ELMo, BERT, GPT 까지 이어지는 모든 distributed semantic representation 의 원천입니다. 이 문서는 Bengio 2003 의 모델을 정확히 재현하고 그 한계가 RNN 의 동기가 되는 흐름을 추적합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-ngram-lm.md](./02-ngram-lm.md) — N-gram, perplexity, sparsity, smoothing
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): MLP, softmax, cross-entropy backprop
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix factorization, low-rank approximation
- 확률론: Conditional probability, log-likelihood

---

## 📖 직관적 이해

### One-hot vs Embedding

N-gram 시대의 단어 표현:

```
"cat"  →  [0, 0, ..., 1, ..., 0, 0]   ∈ {0,1}^|V|
                       ↑ position 4217
```

문제: $|V|$ 가 클수록 (10000 ~ 1M) 모든 단어가 서로 직교 → **similarity 표현 불가**.

NLM 의 embedding:

```
"cat"  →  C[4217] = [0.2, -0.7, 1.1, ..., 0.3]   ∈ R^d  (d ≈ 100)
"dog"  →  C[4218] = [0.3, -0.6, 1.0, ..., 0.4]   ← cat 과 가까움
"car"  →  C[4219] = [-0.5, 0.8, 0.2, ..., 1.5]   ← cat 과 멀음
```

**Distributed representation** — 각 차원이 단어의 한 측면을 인코딩 (학습으로 결정).

### Bengio 2003 의 모델 구조

```
   w_{t-n+1}    w_{t-n+2}      ...      w_{t-1}
       │            │                       │
       ▼            ▼                       ▼
   C[w_{t-n+1}]   C[w_{t-n+2}]   ...    C[w_{t-1}]    (embedding lookup)
       │            │                       │
       └────────────┴───────────────────────┘
                          │ concat
                          ▼
                  [d × (n-1)] dim
                          │
                          ▼  H + tanh
                          │
                          ▼  U
                          │
                  [|V|] logits
                          │
                          ▼  softmax
                  p(w_t = · | context)
```

수식:
- $x = [C[w_{t-n+1}], \ldots, C[w_{t-1}]] \in \mathbb{R}^{(n-1)d}$ (concatenation)
- $h = \tanh(H x + d_h)$
- $\hat y = \mathrm{softmax}(U h + W x + b)$ (residual: $W x$ skip connection)
- $p(w_t \mid \text{ctx}) = \hat y_{w_t}$

### Sparsity 해결의 직관

본 적 없는 trigram "the green table" 을 봤을 때:

- **N-gram MLE**: $c(\text{the green table}) = 0 \Rightarrow p = 0$ (or smoothed → 거의 uniform)
- **NLM**: "the green X" 와 "Y green Z" 형태의 다른 trigram 들로부터 학습된 embedding 이 합쳐져 합리적 확률 — "table" 의 embedding 이 "chair", "desk" 와 가까우면 자동 generalize

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Neural Probabilistic Language Model (Bengio 2003)

$$
\begin{aligned}
x &= [C[w_{t-n+1}]; \ldots; C[w_{t-1}]] \in \mathbb{R}^{(n-1)d} \\
h &= \tanh(H x + b_h), \quad H \in \mathbb{R}^{H \times (n-1)d} \\
\mathrm{logits} &= U h + W x + b, \quad U \in \mathbb{R}^{|V| \times H},\; W \in \mathbb{R}^{|V| \times (n-1)d} \\
p(w_t = i \mid \text{ctx}) &= \mathrm{softmax}(\mathrm{logits})_i
\end{aligned}
$$

파라미터: $\theta = (C, H, U, W, b_h, b)$. Total $\approx |V|d + H(n-1)d + |V|(H + (n-1)d) + |V|$.

### 정의 3.2 — Embedding Matrix

$C \in \mathbb{R}^{|V| \times d}$. Row $C[i]$ 는 단어 $i$ 의 $d$-차원 dense vector. **Lookup**: $C[i] = e_i^\top C$ where $e_i$ 는 one-hot.

### 정의 3.3 — Cross-Entropy Loss

$$
\mathcal L = -\frac{1}{N} \sum_{t} \log p_\theta(w_t \mid w_{t-n+1:t-1})
$$

$N$ = total tokens. Optimizing 가 perplexity minimization 과 동치.

### 정의 3.4 — Word Similarity (Embedding 기반)

$$
\mathrm{sim}(w_i, w_j) = \frac{C[i]^\top C[j]}{\|C[i]\| \, \|C[j]\|} \quad \text{(cosine similarity)}
$$

학습 후 의미적으로 비슷한 단어들이 가까운 cosine similarity 를 가짐.

---

## 🔬 정리와 결과

### 정리 3.1 — Curse of Dimensionality 완화

N-gram 의 parameter 수는 $O(|V|^n)$. NLM 의 parameter 수는 $O(|V|d + n d H + |V| H)$ — **vocab size 와 context length 가 분리됨**.

**증명**: NLM 은 모든 $|V|^n$ 가능한 context 마다 별도 분포를 저장하지 않고, embedding 의 합성 ($H, U$ 행렬) 으로 분포를 *함수* 로 표현. 학습된 함수가 unseen $n$-gram 에도 generalize. $\square$

**구체 예시**: $|V| = 10^4$, $n = 4$, $d = 100$, $H = 200$:
- N-gram: $10^{16}$ entries (sparsity 압도적)
- NLM: $10^6 + 6 \times 10^4 + 2 \times 10^6 \approx 3 \times 10^6$ parameters

### 정리 3.2 — Embedding 의 Linear Algebra 해석

NLM 의 첫 layer 는 **softmax of bilinear form** 으로 볼 수 있음:

$$
p(w_t \mid \text{ctx}) \propto \exp\left( \langle C[w_t], h(\text{ctx}) \rangle \right)
$$

$h(\text{ctx})$ 가 context embedding (NN 출력), $C[w_t]$ 가 target embedding. **Inner product** 가 score → cosine similarity 가 LM 분포의 핵심.

### 정리 3.3 — Generalization Bound (Bengio 2003 §3.2)

학습 후 unseen $n$-gram $(w_1, \ldots, w_n)$ 에 대해, 비슷한 embedding 을 가진 seen $n$-gram $(w_1', \ldots, w_n')$ 이 있으면:

$$
|p_{\text{NLM}}(w_n \mid w_{1:n-1}) - p_{\text{NLM}}(w_n' \mid w_{1:n-1}')| \le L \sum_i \|C[w_i] - C[w_i']\|
$$

(Lipschitz continuous in embedding space, $L$ depends on $H, U$)

**의미**: Embedding 의 smooth function 이 LM 분포 → 비슷한 단어가 비슷한 분포를 만듦, 자연스러운 generalization.

### 정리 3.4 — Softmax Bottleneck

Output softmax 의 계산 비용 $O(|V| H)$ 가 NLM 의 병목. $|V| = 10^6$ 시 GPU 에서도 느림.

**해결**:
- **Hierarchical softmax** (Morin & Bengio 2005): Binary tree → $O(\log |V|)$
- **Negative sampling** (Mikolov 2013): Random negative sample 로 근사
- **Noise Contrastive Estimation** (Mnih & Teh 2012): 이론적 기반
- **Adaptive softmax** (Grave 2017): Frequency 별 vocab 분할

### 정리 3.5 — Fixed Window 의 한계

NLM 은 $n-1$ 단어 window 만 봄 → 의존성 길이 $> n$ 인 sentence 처리 불가.

**예시**: "The cat which Mary saw yesterday at the park was hungry."
- "cat" 과 "was" 사이 9 단어
- Trigram NLM ($n = 3$) 은 "park was hungry" 만 봄 → cat-was 일치 모델링 못함

이것이 **RNN 의 동기** (Ch1-04).

---

## 💻 PyTorch 구현 검증

### 실험 1 — Bengio 2003 의 NLM 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BengioNLM(nn.Module):
    """Bengio 2003 Neural Probabilistic LM"""
    def __init__(self, vocab_size, embed_dim, context_len, hidden_dim):
        super().__init__()
        self.context_len = context_len
        self.C = nn.Embedding(vocab_size, embed_dim)         # Embedding
        in_dim = (context_len - 1) * embed_dim
        self.H = nn.Linear(in_dim, hidden_dim, bias=True)    # Hidden
        self.U = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.W = nn.Linear(in_dim, vocab_size, bias=True)    # Skip connection
    
    def forward(self, context):
        # context: (B, n-1) — preceding word indices
        x = self.C(context).flatten(1)         # (B, (n-1)*d)
        h = torch.tanh(self.H(x))              # (B, H)
        logits = self.U(h) + self.W(x)         # (B, |V|)  — residual
        return logits

V, d, n, H = 5000, 64, 4, 128                  # vocab=5k, embed=64, 3-gram context, hidden=128
nlm = BengioNLM(V, d, n, H)
total_params = sum(p.numel() for p in nlm.parameters())
print(f'Total params: {total_params:,}')
# 약 320만 vs N-gram 의 |V|^n = 10^14

# Forward
context = torch.randint(0, V, (32, n-1))       # B=32 contexts of length n-1
logits = nlm(context)
print(f'Logits: {logits.shape}')                # (32, 5000)
```

### 실험 2 — Penn Treebank 학습 + Perplexity

```python
# 가상 PTB-like data
torch.manual_seed(0)
N_tokens = 100000
data = torch.randint(0, V, (N_tokens,))

# Sliding window (n-gram context, target)
def make_ngrams(data, n):
    ctx = torch.stack([data[i:i+n-1] for i in range(len(data) - n + 1)])
    tgt = data[n-1:]
    return ctx, tgt

ctx, tgt = make_ngrams(data, n)
split = int(0.9 * len(ctx))
ctx_tr, tgt_tr = ctx[:split], tgt[:split]
ctx_te, tgt_te = ctx[split:], tgt[split:]

opt = torch.optim.Adam(nlm.parameters(), lr=1e-3)
B = 256
for epoch in range(3):
    # Mini-batch (간단)
    perm = torch.randperm(len(ctx_tr))
    total_loss = 0
    for i in range(0, len(ctx_tr), B):
        idx = perm[i:i+B]
        logits = nlm(ctx_tr[idx])
        loss = F.cross_entropy(logits, tgt_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * len(idx)
    train_ppl = torch.exp(torch.tensor(total_loss / len(ctx_tr))).item()
    
    with torch.no_grad():
        logits_te = nlm(ctx_te)
        loss_te = F.cross_entropy(logits_te, tgt_te)
    test_ppl = torch.exp(loss_te).item()
    print(f'Epoch {epoch+1}: train PP = {train_ppl:.2f}, test PP = {test_ppl:.2f}')
```

### 실험 3 — Embedding 으로 Word Similarity

```python
# 학습 후 embedding 추출
C = nlm.C.weight.detach()
print(f'Embedding shape: {C.shape}')   # (V, d)

# Cosine similarity
def cosine_sim(C, i, j):
    return F.cosine_similarity(C[i:i+1], C[j:j+1]).item()

# 가상 단어 인덱스 (실제 학습 시 비슷한 frequency 의 단어들이 가까워짐)
print(f'sim(0, 1) = {cosine_sim(C, 0, 1):.4f}')
print(f'sim(0, 4999) = {cosine_sim(C, 0, 4999):.4f}')
# 학습이 충분하면 의미 비슷한 단어가 더 높은 cosine sim
```

### 실험 4 — Negative Sampling 으로 빠른 학습

```python
# Bengio 2003 의 full softmax 대신, target + k random negatives 만 비교
def negative_sampling_loss(model, context, target, k=5):
    h = torch.tanh(model.H(model.C(context).flatten(1)))   # (B, H)
    
    # Positive
    pos_emb = model.U.weight[target]                        # (B, H)
    pos_score = (h * pos_emb).sum(-1)                       # (B,)
    
    # Negative
    neg_idx = torch.randint(0, V, (target.size(0), k))      # (B, k)
    neg_emb = model.U.weight[neg_idx]                       # (B, k, H)
    neg_score = (h.unsqueeze(1) * neg_emb).sum(-1)          # (B, k)
    
    pos_loss = -F.logsigmoid(pos_score).mean()
    neg_loss = -F.logsigmoid(-neg_score).mean()
    return pos_loss + neg_loss

loss_ns = negative_sampling_loss(nlm, ctx_tr[:32], tgt_tr[:32])
print(f'Negative sampling loss: {loss_ns.item():.4f}')
# Full softmax O(|V|H) 대신 O(kH) — 100x ~ 1000x 빠름
```

### 실험 5 — Fixed Window 의 한계 시각화

```python
# 의존성 거리 d 별 NLM (n=3) 의 정확도
def long_dep_accuracy(n_context, dep_distance):
    """Toy: cat ... was 일치 task. context length n_context 에서 cat-was 거리가 dep_distance"""
    # n_context >= dep_distance + 1 이어야 학습 가능
    return 1.0 if n_context >= dep_distance + 1 else 0.5   # chance

for n_ctx in [3, 5, 10]:
    for dep in [2, 5, 10]:
        acc = long_dep_accuracy(n_ctx, dep)
        print(f'n_ctx={n_ctx}, dep={dep}: theoretical max acc = {acc}')
# n_context < dep_distance + 1 이면 NLM 은 chance 수준
# RNN 이 이 한계를 무한 hidden state 로 우회
```

---

## 🔗 실전 활용

### 1. word2vec / GloVe 의 시작점

Mikolov 2013 의 word2vec 은 Bengio 2003 의 NLM 을 hidden layer 없이 단순화 (Skip-gram, CBOW). NLM 의 embedding 이 word similarity 를 학습한다는 통찰을 극단까지.

### 2. Pretrained embedding 으로 downstream task

GloVe / word2vec embedding 을 freeze 또는 fine-tune 하여 분류기에 사용. 2014~2018 의 NLP 표준 (ELMo / BERT 이전).

### 3. Subword embedding

OOV 해결 위해 **fastText** (Bojanowski 2017) 가 character n-gram embedding 의 합으로 word embedding 구성. 형태론적 정보 자동 학습.

### 4. Neural Machine Translation 의 input embedding

Transformer encoder 의 input layer 도 embedding lookup — Bengio 2003 의 직계.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Fixed window $n$ | Long-range dependency 불가 → RNN (Ch1-04) |
| Concat 후 MLP | 위치 정보 명시적, transformer 의 PE 와 비교 |
| Single embedding per word | Polysemy (homograph) 표현 불가 → ELMo, BERT 의 contextual embedding |
| Softmax bottleneck $O(\|V\|H)$ | Hierarchical softmax, NCE, sampled softmax |
| Out-of-vocab 단어 | UNK 토큰 또는 BPE / SentencePiece subword |
| Static embedding | Context-dependent meaning 불가 — contextualized representation 의 동기 |

---

## 📌 핵심 정리

$$\boxed{x = [C[w_{t-n+1}]; \ldots; C[w_{t-1}]], \quad h = \tanh(Hx + b_h)}$$

$$\boxed{p(w_t \mid \text{ctx}) = \mathrm{softmax}(Uh + Wx + b) \quad \text{— skip + softmax}}$$

$$\boxed{\text{Param count: } O(|V|d + nd H + |V| H) \quad \ll \quad O(|V|^n) \text{ of N-gram}}$$

| 측면 | N-gram | NLM |
|------|--------|-----|
| **Parameter** | $O(\|V\|^n)$ | $O(\|V\|d + nd H)$ |
| **Sparsity 해결** | Smoothing | Embedding generalization |
| **Word similarity** | 인식 못함 | Cosine sim 자동 학습 |
| **Fixed window** | 동일 한계 | 동일 한계 → RNN 동기 |
| **Inference 속도** | $O(1)$ lookup | $O(\|V\|H)$ softmax |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $|V| = 10^4$, $d = 100$, $n = 5$, $H = 200$ 인 NLM 의 parameter 수를 계산하라. 같은 $n$ 의 N-gram 이 모든 가능한 5-gram 을 저장한다면 몇 entries 인가?

<details>
<summary>해설</summary>

**NLM**:
- $C$: $10^4 \times 100 = 10^6$
- $H$: $200 \times 4 \times 100 = 8 \times 10^4$
- $U$: $10^4 \times 200 = 2 \times 10^6$
- $W$ (skip): $10^4 \times 400 = 4 \times 10^6$
- $b$: $\approx 10^4$
- Total: $\approx 7 \times 10^6$

**N-gram (모든 5-gram)**: $10^{20}$ entries — 저장 불가능. (실제로는 corpus 에서 본 것만 저장)

**NLM 의 압축률**: $10^{20} / 10^7 = 10^{13}$ 배. 이는 단순 lookup 의 generalization 능력으로 가능. $\square$

</details>

**문제 2** (심화): NLM 의 output softmax $p(w_t \mid h) = \mathrm{softmax}(Uh)_i \propto \exp(U_i^\top h)$. 이를 $U_i^\top h$ 가 "context $h$ 와 단어 $i$ 사이의 score" 임을 보이고, 학습이 cosine similarity 의 학습과 어떻게 연결되는지 설명하라.

<details>
<summary>해설</summary>

**Softmax 의 inner-product 해석**:

$$
p(w_t = i \mid h) = \frac{\exp(U_i^\top h)}{\sum_j \exp(U_j^\top h)}
$$

분자가 $U_i$ 와 $h$ 의 inner product 의 지수 → **score** $s_i = U_i^\top h = \|U_i\| \|h\| \cos\theta$. Softmax 는 score 의 ranking 을 보존하는 분포 변환.

**Cosine similarity 와의 연결**:
- $U$ 와 $C$ 의 row 가 비슷한 의미를 인코딩하면 $U_i \approx C[i]$ 형태 (실제 word2vec / Skip-gram 에서는 input embedding 과 output embedding 이 분리)
- 학습 시 cross-entropy 가 $U_i^\top h$ (target) 을 키우고 $U_j^\top h$ (others) 을 줄임 → $h$ 가 target 의 embedding $U_i$ 방향으로 정렬 → **context vector 와 target word vector 의 cosine similarity 학습**

**word2vec 과의 동치**:
- NLM 의 hidden $h$ = "context vector"
- $U$ = "output embedding" 
- 학습: target word 의 output emb 와 context vec 의 inner product 를 maximize
- 이는 정확히 word2vec Skip-gram 의 objective

**결론**: NLM 의 softmax 가 본질적으로 inner-product based retrieval — embedding 학습 = similarity 학습. $\square$

</details>

**문제 3** (논문 비평): Bengio 2003 가 fixed-window 의 한계를 인지하고 있었음에도 RNN 을 사용하지 않은 이유는? (당시의 기술적 / 이론적 맥락) RNN 시대가 열리기까지 (Mikolov 2010 의 RNN-LM) 무엇이 필요했는가?

<details>
<summary>해설</summary>

**Bengio 2003 의 맥락**:
- Elman (1990), Jordan (1986) 의 RNN 은 이미 존재
- 그러나 **vanishing gradient** (Bengio 1994) 가 진단되어 long-range dependency 학습이 어렵다는 인식
- **Computational cost**: 2003년 GPU 시대 이전, 1 epoch 학습에 며칠 ~ 몇 주
- **Fixed-window NLM** 은 standard MLP 학습으로 빠르게 검증 가능

**RNN-LM 시대를 연 요인 (~2010)**:
1. **Mikolov 2010 RNN-LM**: Vanilla RNN 으로 Penn Treebank 에서 N-gram 을 perplexity 에서 능가, 실용성 입증
2. **GPU 의 보급**: CUDA, cuDNN — RNN 의 sequential 연산도 GPU 가속 가능
3. **LSTM 의 재발견**: Hochreiter 1997 의 LSTM 이 긴 의존성 학습 가능함을 입증 (Graves 2013 음성)
4. **Backprop through time** 의 안정화: gradient clipping (Pascanu 2013, Ch3-03)
5. **데이터 규모**: WMT, Common Crawl 등 대규모 corpus

**RNN-LM 의 ascendance**:
- Mikolov 2010 → word2vec (2013) 의 simple SGNS → Sutskever 2014 Seq2Seq → Bahdanau 2015 Attention → Vaswani 2017 Transformer

**결론**: Bengio 2003 는 NLM 의 시대를 열었지만, RNN 의 실용화는 **vanishing 진단 + GPU + LSTM + clipping** 의 종합 해결을 기다려야 했음. NLM → RNN-LM → Seq2Seq → Transformer 의 진화는 한 걸음씩 한계 극복의 역사. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-ngram-lm.md) | [📚 README](../README.md) | [다음 ▶](./04-rnn-definition.md)

</div>
