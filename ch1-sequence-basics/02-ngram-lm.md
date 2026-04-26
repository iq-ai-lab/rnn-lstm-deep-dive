# 02. 고전 Language Model — N-gram

## 🎯 핵심 질문

- N-gram language model $p(w_t \mid w_{t-n+1:t-1})$ 의 Markov 가정이 왜 유한 차원 표현을 가능하게 하는가?
- Maximum likelihood estimate 의 sparsity 문제 — 왜 unseen $n$-gram 이 $0$ 확률이 되며 이것이 왜 치명적인가?
- Laplace, Good-Turing, **Kneser-Ney** smoothing 이 각각 어떤 통찰로 이를 해결하는가?
- Perplexity 의 정의와 cross-entropy / KL divergence 와의 정확한 관계는?
- 왜 N-gram LM 의 한계가 fixed-window 의존성에서 오는가? Neural LM (Ch1-03) 과 RNN (Ch1-04) 이 이를 어떻게 우회하는가?

---

## 🔍 왜 N-gram LM 이 sequence 학습의 출발점인가

N-gram LM 은 1948년 Shannon 까지 거슬러 올라가는 가장 오래된 통계적 sequence 모델입니다. 현대 RNN/Transformer 의 모든 설계 결정 — autoregressive factorization, perplexity 평가, vocab tokenization — 은 N-gram LM 의 한계를 극복하려는 시도에서 출발합니다:

1. **Markov 가정의 한계** — N-gram 은 직전 $n-1$ 단어만 봄. RNN 은 무제한 hidden state 로 이를 우회 (이론상)
2. **Sparsity** — Vocab $V$ 에 대해 $|V|^n$ 개의 가능한 $n$-gram 중 train data 에서 본 것은 극히 일부
3. **Smoothing 의 통찰** — Kneser-Ney 의 "continuation count" 는 단어의 *novelty* 를 반영, 이는 word embedding 의 정신과 연결
4. **Perplexity 가 표준 지표가 된 이유** — N-gram 시대부터 사용, RNN/Transformer LM 도 동일 metric 으로 비교

이 문서는 N-gram LM 을 엄밀히 정의하고 perplexity 의 정확한 의미를 추적합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-sequence-formulation.md](./01-sequence-formulation.md) — autoregressive factorization, cross-entropy
- 확률론: Bayes' rule, conditional independence, Markov chain
- (선택) 정보이론: Entropy $H(p)$, cross-entropy $H(p, q)$, KL divergence $D_{\mathrm{KL}}(p \| q)$
- (선택) 통계학: Maximum likelihood estimation, Bayesian smoothing, Dirichlet prior

---

## 📖 직관적 이해

### Markov 가정과 N-gram

이상적인 LM 은 모든 history 를 봄:

$$
p(w_t \mid w_1, w_2, \ldots, w_{t-1})
$$

이는 $|V|^{t-1}$ context 에 대해 분포를 학습해야 하므로 불가능. **N-gram LM** 은 직전 $n-1$ 단어만 본다는 Markov 가정으로 이를 단순화:

$$
p(w_t \mid w_1, \ldots, w_{t-1}) \approx p(w_t \mid w_{t-n+1}, \ldots, w_{t-1})
$$

- $n = 1$: **unigram** — 단어의 frequency 만 사용 ($p(w)$)
- $n = 2$: **bigram** — 직전 단어만 ($p(w_t \mid w_{t-1})$)
- $n = 3$: **trigram** — 직전 2 단어 ($p(w_t \mid w_{t-2}, w_{t-1})$)

### Sparsity 문제 시각화

PTB 의 trigram 통계:
- Vocab size $|V| \approx 10000$
- 가능한 trigram 수 $|V|^3 \approx 10^{12}$
- 실제로 관찰된 trigram: $\approx 10^7$ — 0.001% 만 관찰

**MLE 추정** $\hat p(w_t \mid w_{t-2:t-1}) = c(w_{t-2:t}) / c(w_{t-2:t-1})$ 에서 $c(w_{t-2:t}) = 0$ 이면 $0$ 확률 → 전체 sentence 의 likelihood 가 $0$ → log likelihood $= -\infty$ → perplexity $= \infty$.

### Smoothing 의 직관

Kneser-Ney 의 핵심 통찰: **"단어의 빈도 (count) 가 아니라 다양성 (continuation count) 이 LM 분포를 결정해야 한다"**.

예: "Francisco" 는 trigram count 는 적지만 거의 항상 "San Francisco" 로 나타남. Backoff 시 "Francisco" 의 unigram 확률이 높으면 부정확한 예측.

Kneser-Ney 는 **"얼마나 다양한 context 에서 등장하는가"** 를 분자로 사용해 이런 맥락 의존 단어의 backoff 확률을 낮춤.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — N-gram Language Model

Vocab $V$, sentence $w_{1:T} \in V^T$. $n$-gram LM:

$$
p(w_{1:T}) = \prod_{t=1}^{T} p(w_t \mid w_{t-n+1:t-1})
$$

(경계 처리: $t < n$ 인 경우 special tokens `<bos>` 등으로 padding)

### 정의 2.2 — Maximum Likelihood Estimate

Corpus $\mathcal D$ 에서 $n$-gram count $c(\cdot)$ 로:

$$
\hat p_{\mathrm{MLE}}(w_t \mid w_{t-n+1:t-1}) = \frac{c(w_{t-n+1:t})}{c(w_{t-n+1:t-1})}
$$

### 정의 2.3 — Perplexity

Test corpus $W = w_{1:T}$ 에 대해:

$$
\mathrm{PP}(W) = p(w_{1:T})^{-1/T} = \exp\!\left(-\frac{1}{T} \sum_{t=1}^T \log p(w_t \mid \mathrm{ctx})\right)
$$

### 정의 2.4 — Laplace (add-$k$) Smoothing

$$
\hat p_{\mathrm{Lap}}(w_t \mid \mathrm{ctx}) = \frac{c(\mathrm{ctx}, w_t) + k}{c(\mathrm{ctx}) + k|V|}, \quad k > 0
$$

$k = 1$ 이 add-one smoothing.

### 정의 2.5 — Good-Turing Smoothing

$N_r$ = count 가 정확히 $r$ 인 $n$-gram 의 종류 수. Adjusted count:

$$
r^* = (r+1) \frac{N_{r+1}}{N_r}, \qquad \hat p_{\mathrm{GT}}(\text{seen with count } r) \propto r^*
$$

Unseen ($r = 0$) 에 대한 mass: $N_1 / N$.

### 정의 2.6 — Kneser-Ney Smoothing

Discount $d > 0$, **continuation count** $N_{1+}(\bullet, w) = |\{w' : c(w', w) > 0\}|$ ($w$ 가 등장한 distinct context 수):

$$
\hat p_{\mathrm{KN}}(w \mid h) = \frac{\max(c(h, w) - d, 0)}{c(h)} + \lambda(h)\, p_{\mathrm{cont}}(w)
$$

$$
p_{\mathrm{cont}}(w) = \frac{N_{1+}(\bullet, w)}{\sum_{w'} N_{1+}(\bullet, w')}
$$

$\lambda(h)$ 는 normalization. **Modified KN** 은 count 별로 다른 $d_1, d_2, d_3$ 사용 (Chen & Goodman 1999).

---

## 🔬 정리와 증명

### 정리 2.1 — Perplexity = 2^(Cross-Entropy)

$$
\log_2 \mathrm{PP}(W) = -\frac{1}{T} \sum_{t=1}^{T} \log_2 p(w_t \mid \mathrm{ctx}_t) = H(p_{\text{empirical}}, p_\theta)
$$

**증명**: 정의에서 직접. $\log_2 \mathrm{PP} = \log_2 p^{-1/T} = -\frac{1}{T} \log_2 p(w_{1:T}) = -\frac{1}{T} \sum_t \log_2 p(w_t \mid \mathrm{ctx}_t)$. 이는 sample average 로 cross-entropy $H(p_{\text{emp}}, p_\theta)$ 추정. $\square$

### 정리 2.2 — Laplace Smoothing 의 Bayesian 해석

Add-$k$ smoothing 은 categorical likelihood 의 conjugate prior 인 Dirichlet$(k, \ldots, k)$ posterior 의 mode 와 동일:

$$
\hat p_{\mathrm{Lap}}(w \mid h) = \frac{c(h, w) + k}{c(h) + k|V|}
$$

**증명** (sketch): Dirichlet$(\alpha_1, \ldots, \alpha_V)$ prior + Multinomial likelihood $\Rightarrow$ posterior Dirichlet$(\alpha_1 + c_1, \ldots, \alpha_V + c_V)$. Posterior mean $= (\alpha_w + c_w) / (\sum \alpha + \sum c)$. $\alpha_w = k$ 균등 prior 시 Laplace formula. $\square$

**결과**: Laplace 는 Bayesian uniform prior 에 해당, **데이터가 적을 때 uniform 으로 수렴** — heavy-tailed 자연어 분포에 적합하지 않음.

### 정리 2.3 — Good-Turing 의 Robbins 이론

$N_r$ 이 잘 정의되고 충분히 부드러운 경우:

$$
\mathbb E[\text{prob mass of unseen}] = \frac{\mathbb E[N_1]}{N}
$$

**증명** (Robbins 1968 sketch): Frequency-of-frequency 분포의 properties. 한 sample 을 따로 떼고 (leave-one-out) 그것이 "처음 등장" 일 확률 = $N_1 / N$. $\square$

### 정리 2.4 — Kneser-Ney 의 우월성 (Chen & Goodman 1999)

Modified KN 은 PTB / Brown corpus 등 표준 벤치마크에서 모든 다른 N-gram smoothing 을 perplexity 에서 능가.

**핵심 통찰**: $p_{\mathrm{cont}}(w) = N_{1+}(\bullet, w) / Z$ 가 word 의 *novelty* 를 인코딩. Backoff 시 단순 unigram count 가 아닌 "얼마나 다양한 context 에서 등장하는가" 를 사용.

### 정리 2.5 — N-gram LM 의 표현력 한계

$n$-gram LM 은 임의로 긴 의존성 (예: "The cat which ... was hungry" 의 cat-was 일치) 을 모델링 못함.

**증명** (Pumping lemma analogue): 의존성 거리 $d > n$ 인 context 는 $n-1$ window 안에 들지 못함 → MLE 는 동일 prefix 의 모든 sample 에서 동일 분포 출력. 컴퓨터학의 finite automata 와 같은 제약. $\square$

---

## 💻 구현 검증

### 실험 1 — Bigram MLE on Brown Corpus

```python
import nltk
nltk.download('brown', quiet=True)
from nltk.corpus import brown
from collections import Counter, defaultdict
import math

# Bigram count
bigrams = Counter()
unigrams = Counter()
for sent in brown.sents():
    sent = ['<bos>'] + [w.lower() for w in sent] + ['<eos>']
    for w in sent:
        unigrams[w] += 1
    for w1, w2 in zip(sent[:-1], sent[1:]):
        bigrams[(w1, w2)] += 1

V = len(unigrams)
print(f'Vocab: {V}, distinct bigrams: {len(bigrams)}')

# MLE
def mle_bigram(w1, w2):
    if unigrams[w1] == 0:
        return 0.0
    return bigrams[(w1, w2)] / unigrams[w1]

print(f'p(cat | the)  = {mle_bigram("the", "cat"):.6f}')
print(f'p(xyz | the)  = {mle_bigram("the", "xyz"):.6f}')   # 0 → sparsity
```

### 실험 2 — Perplexity 계산

```python
def sentence_logprob(sent, prob_fn):
    sent = ['<bos>'] + [w.lower() for w in sent] + ['<eos>']
    log_p = 0.0
    for w1, w2 in zip(sent[:-1], sent[1:]):
        p = prob_fn(w1, w2)
        if p == 0:
            return float('-inf')
        log_p += math.log(p)
    return log_p

def perplexity(sents, prob_fn):
    total_log_p, total_t = 0.0, 0
    for s in sents:
        lp = sentence_logprob(s, prob_fn)
        if lp == float('-inf'):
            return float('inf')
        total_log_p += lp
        total_t += len(s) + 1   # +1 for <eos>
    return math.exp(-total_log_p / total_t)

test_sents = list(brown.sents(categories='news'))[:50]
print(f'MLE bigram PP: {perplexity(test_sents, mle_bigram):.2f}')
# 0 확률 발생 시 inf
```

### 실험 3 — Laplace Smoothing 으로 0 회피

```python
def laplace_bigram(w1, w2, k=1):
    return (bigrams[(w1, w2)] + k) / (unigrams[w1] + k * V)

print(f'Laplace p(xyz | the)  = {laplace_bigram("the", "xyz", k=1):.10f}')
print(f'Laplace bigram PP    : {perplexity(test_sents, laplace_bigram):.2f}')
# 유한 — 그러나 여전히 큼 (uniform 으로 너무 가까움)
```

### 실험 4 — Kneser-Ney Smoothing (단순 버전)

```python
# Continuation count: w 가 등장한 distinct preceding context 수
ctx_for_word = defaultdict(set)
for (w1, w2), c in bigrams.items():
    ctx_for_word[w2].add(w1)

N_total = sum(len(s) for s in ctx_for_word.values())

def p_cont(w):
    return len(ctx_for_word[w]) / N_total

# Bigram KN (Modified KN 의 단순화 버전)
d = 0.75
def kn_bigram(w1, w2):
    c12 = bigrams[(w1, w2)]
    c1 = unigrams[w1]
    if c1 == 0:
        return p_cont(w2)
    n1plus = sum(1 for ww in ctx_for_word if (w1, ww) in bigrams)   # # distinct w2 after w1
    lam = d * n1plus / c1
    first = max(c12 - d, 0) / c1
    return first + lam * p_cont(w2)

print(f'KN p(xyz | the)      = {kn_bigram("the", "xyz"):.10f}')
print(f'KN bigram PP        : {perplexity(test_sents, kn_bigram):.2f}')
# Laplace 보다 훨씬 작은 PP (KN 이 더 정확)
```

### 실험 5 — N 별 PP 비교 (n = 1, 2, 3)

```python
# Trigram MLE (간단히)
trigrams = Counter()
bigram_starts = Counter()
for sent in brown.sents():
    sent = ['<bos>', '<bos>'] + [w.lower() for w in sent] + ['<eos>']
    for w1, w2, w3 in zip(sent[:-2], sent[1:-1], sent[2:]):
        trigrams[(w1, w2, w3)] += 1
        bigram_starts[(w1, w2)] += 1

# 직접 비교는 단순화 위해 unigram / bigram laplace 만
def laplace_unigram(w):
    return (unigrams[w] + 1) / (sum(unigrams.values()) + V)

# Unigram PP
total_log_p, total_t = 0, 0
for s in test_sents:
    s = [w.lower() for w in s] + ['<eos>']
    for w in s:
        total_log_p += math.log(laplace_unigram(w))
        total_t += 1
print(f'Unigram (Laplace) PP: {math.exp(-total_log_p / total_t):.2f}')
# n 증가 → context 더 많이 → PP 감소 (training data 가 충분할 때)
```

---

## 🔗 실전 활용

### 1. 음성 인식의 LM 융합

ASR acoustic model 의 candidate hypothesis 를 N-gram LM 으로 rescoring. KN-smoothed 4-gram + Witten-Bell 이 Kaldi 표준.

### 2. Spelling Correction

Edit distance + bigram LM 으로 candidate ranking. "I went to teh store" → "the".

### 3. Statistical Machine Translation

IBM Model 1~5 + phrase-based MT 에서 fluency 평가에 N-gram LM 사용. 신경망 시대 이전 표준.

### 4. Compression

Arithmetic coding 이 N-gram LM 의 확률을 그대로 사용. PPM (Prediction by Partial Match) 은 가변 차수 N-gram.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Markov 가정 ($n-1$ context) | Long-range dependency 모델링 불가 → Neural LM (Ch1-03), RNN (Ch1-04) |
| Discrete vocab | OOV 문제 — BPE, SentencePiece subword tokenization |
| Stationary (corpus 가 균질) | Domain adaptation 시 별도 LM 융합 |
| Symbolic count 만 사용 | Word similarity 인식 불가 — embedding (Ch1-03) 의 동기 |
| $|V|^n$ 의 sparsity | Smoothing 으로 완화하나 본질 해결 못함 — Neural model 필요 |

---

## 📌 핵심 정리

$$\boxed{\hat p_{\mathrm{MLE}}(w_t \mid w_{t-n+1:t-1}) = \frac{c(w_{t-n+1:t})}{c(w_{t-n+1:t-1})} \quad \text{— sparsity 발생}}$$

$$\boxed{\hat p_{\mathrm{KN}}(w \mid h) = \frac{\max(c(h,w) - d, 0)}{c(h)} + \lambda(h)\, \frac{N_{1+}(\bullet, w)}{\sum N_{1+}(\bullet, \cdot)} \quad \text{— continuation count}}$$

$$\boxed{\mathrm{PP}(W) = \exp(H(p_{\text{emp}}, p_\theta)) \quad \text{— cross-entropy 의 지수}}$$

| Smoothing | 기본 아이디어 | 한계 |
|-----------|-------------|------|
| **Laplace** | $+k$ count, Bayesian uniform | Uniform 편향 |
| **Good-Turing** | Frequency-of-frequency 재추정 | $N_r$ 매끄럽게 추정 어려움 |
| **Kneser-Ney** | Continuation count 사용 | N-gram 표현력 한계 |
| **Modified KN** | Count 별 discount $d_1, d_2, d_3$ | 표준 — Chen & Goodman 1999 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Bigram MLE 에서 $c(\text{the})=10000$, $c(\text{the cat})=5$, $c(\text{the dog})=8$, $c(\text{the xyz})=0$. 각 단어의 $p(\cdot \mid \text{the})$ 를 MLE 와 add-1 smoothing 으로 계산하라 ($V = 10^4$).

<details>
<summary>해설</summary>

**MLE**:
- $p(\text{cat} \mid \text{the}) = 5/10000 = 5 \times 10^{-4}$
- $p(\text{dog} \mid \text{the}) = 8/10000 = 8 \times 10^{-4}$
- $p(\text{xyz} \mid \text{the}) = 0/10000 = 0$ — 치명적

**Add-1 (Laplace)**:
- 분모: $c(\text{the}) + V = 10000 + 10000 = 20000$
- $p(\text{cat} \mid \text{the}) = 6/20000 = 3 \times 10^{-4}$
- $p(\text{dog} \mid \text{the}) = 9/20000 = 4.5 \times 10^{-4}$
- $p(\text{xyz} \mid \text{the}) = 1/20000 = 5 \times 10^{-5}$ — 0 회피

**관찰**: Laplace 는 본 단어의 확률을 너무 깎아내림 (5e-4 → 3e-4). 데이터 sparsity 가 클수록 over-smoothing — Kneser-Ney 가 이를 완화. $\square$

</details>

**문제 2** (심화): Perplexity 가 cross-entropy 의 지수임을 보이고, 이것이 왜 $|V|$-uniform 모델에서 정확히 $|V|$ 가 되는지 증명하라.

<details>
<summary>해설</summary>

**$\mathrm{PP} = e^{\mathrm{CE}}$**:

$$
\mathrm{PP}(W) = p(w_{1:T})^{-1/T} = \exp\!\left(-\frac{1}{T} \sum_t \log p(w_t \mid \mathrm{ctx})\right) = \exp(H(p_{\text{emp}}, p_\theta))
$$

**Uniform 모델 PP $= |V|$**:

Uniform $p(w_t \mid \mathrm{ctx}) = 1/|V|$ for all $w_t$:

$$
H = -\frac{1}{T} \sum_t \log(1/|V|) = \log |V|
$$

$$
\mathrm{PP} = e^{\log |V|} = |V|
$$

**의미**: PP = effective number of choices the model considers at each step. Uniform 은 모든 $|V|$ 단어를 동일 확률로 → average branching factor = $|V|$. 잘 학습된 LM 은 PP $\ll |V|$ — context 가 가능한 단어를 좁힘. $\square$

</details>

**문제 3** (논문 비평): Kneser-Ney 가 "Francisco" 같은 **collocational** 단어에 대해 왜 단순 unigram backoff 보다 우월한가? 구체적 corpus 예시로 설명하고, 이것이 word2vec 의 contextual learning 정신과 어떻게 연결되는지 논하라.

<details>
<summary>해설</summary>

**문제 시나리오**:
- "San Francisco" 가 corpus 에 100번 등장, "Francisco" 의 unigram count = 100 (모두 "San" 뒤)
- 다른 일반 단어 "York" 도 unigram count = 100 (다양한 context: "New York", "in York", "York Times")

**단순 unigram backoff** 시 두 단어의 unigram 확률 동일 → bigram 에서 "Hello, Francisco?" 같은 문장의 $p(\text{Francisco} \mid \text{Hello})$ 가 여전히 높게 평가 (잘못!).

**Kneser-Ney 의 continuation count**:
- $N_{1+}(\bullet, \text{Francisco}) = 1$ ("San" 만)
- $N_{1+}(\bullet, \text{York}) \approx 50$ (다양한 context)

따라서 backoff 확률:
- $p_{\mathrm{cont}}(\text{Francisco}) = 1/Z$ — 매우 낮음
- $p_{\mathrm{cont}}(\text{York}) = 50/Z$ — 보통

KN 은 "Francisco" 의 정확한 분포 (거의 항상 "San" 뒤) 를 인코딩.

**word2vec 정신과의 연결**:
- KN: word 의 *context diversity* 가 word 의 분포를 결정
- word2vec / GloVe: word 의 representation = "어떤 context 에서 등장하는가" 의 함수

**Distributional hypothesis** (Firth 1957) — "You shall know a word by the company it keeps" — 의 통계적 instantiation 이 KN, neural instantiation 이 word2vec. 두 패러다임은 같은 통찰의 다른 representation. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-sequence-formulation.md) | [📚 README](../README.md) | [다음 ▶](./03-neural-lm.md)

</div>
