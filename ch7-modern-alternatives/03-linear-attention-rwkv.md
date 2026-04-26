# 03. Linear Attention 과 RNN 의 부활

## 🎯 핵심 질문

- **Katharopoulos 2020** 의 *Transformers are RNNs* 가 어떻게 softmax attention 을 kernel feature map 으로 근사하여 $O(T)$ inference 를 달성하는가?
- $\mathrm{Attn}(Q, K, V) = \phi(Q) (\phi(K)^\top V)$ 의 association 변경이 어떻게 RNN-like recurrence 를 만드는가?
- **RWKV** (Peng 2023) 의 attention-free RNN-like architecture — modern LLM 에서의 부활
- Linear attention 의 표현력 한계와 softmax attention 과의 trade-off
- Modern hybrid (Hyena, Performer) 와 *recurrent + parallel* 의 통합 trend

---

## 🔍 왜 Linear Attention 이 long context 의 핵심 idea 인가

Transformer 의 $O(T^2)$ memory 가 long context 의 fundamental 한계. Linear attention 의 가능성:

1. **$O(T)$ inference** — RNN 같은 streaming
2. **$O(T)$ memory** — Long context 가능
3. **Parallel training** — Transformer 의 parallelism 보존
4. **RNN-like recurrence** — Hidden state 표현

이는 modern LLM 의 핵심 도전 — Transformer 의 in-context learning 능력을 유지하면서 long context efficiency 달성.

이 문서는 linear attention 의 mathematical trick, RWKV 의 design, 그리고 RNN/Transformer 의 통합 paradigm 을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-cnn-sequence.md](./02-cnn-sequence.md) — CNN-based sequence
- [Ch6-04 Luong attention](../ch6-seq2seq-attention/04-luong-attention.md) — Attention 의 multiplicative form
- 정의: Kernel feature map, associativity of matmul

---

## 📖 직관적 이해

### Softmax Attention 의 결합 순서

Standard:
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

연산 순서:
1. $S = QK^\top / \sqrt{d}$ — $(B, T, T)$ matrix
2. $A = \mathrm{softmax}(S)$ — $(B, T, T)$
3. $\mathrm{output} = AV$ — $(B, T, d)$

**병목**: $T \times T$ attention matrix 가 $O(T^2)$ memory.

### Linear Attention 의 Trick

Softmax 가 *non-linear* — associativity 깨짐. Kernel approximation:
$$
\exp(Q K^\top) \approx \phi(Q) \phi(K)^\top
$$

(Some kernel feature map $\phi$)

그러면:
$$
\mathrm{Attn} = \phi(Q) (\phi(K)^\top V)
$$

연산 순서 변경:
1. $\phi(K)^\top V$ — $(B, d, d)$ matrix
2. $\phi(Q) \cdot [\phi(K)^\top V]$ — $(B, T, d)$

**메모리**: $T \times T$ 가 사라짐, $d \times d$ 만 남음.

### RNN-like Recurrence

Causal linear attention:
$$
\mathrm{output}_t = \phi(q_t) \sum_{s=1}^{t} \phi(k_s) v_s^\top \cdot (\text{normalization})
$$

여기서 $S_t = \sum_{s=1}^{t} \phi(k_s) v_s^\top$ 가 *recurrent state*:
$$
S_t = S_{t-1} + \phi(k_t) v_t^\top
$$

이는 정확히 **RNN의 update rule**! Output:
$$
\mathrm{output}_t = \phi(q_t) S_t
$$

### RWKV 의 정신

```
Standard Transformer:    O(T²) memory
Linear Attention:        O(T) memory + RNN-like inference
RWKV:                    Pure RNN-like + Transformer-quality training
```

RWKV: time-mixing + channel-mixing — gating 과 attention-free recurrence.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Kernel Feature Map

Function $\phi: \mathbb R^d \to \mathbb R^{d'}$ such that:
$$
\langle \phi(x), \phi(y) \rangle \approx K(x, y)
$$

For RBF kernel: $\phi$ = random Fourier features.
For Performer (Choromanski 2020): $\phi$ = positive random features.
For ELU + 1: $\phi(x) = \mathrm{ELU}(x) + 1$ (always positive).

### 정의 3.2 — Linear Attention

Kernel approximation of softmax:

$$
\mathrm{LinearAttn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf 1}
$$

Numerator: weighted sum, denominator: normalization.

### 정의 3.3 — Causal Linear Attention (Recurrent)

Output at position $t$:
$$
\mathrm{output}_t = \frac{\phi(q_t) \sum_{s \le t} \phi(k_s) v_s^\top}{\phi(q_t) \sum_{s \le t} \phi(k_s)^\top}
$$

State updates:
- Numerator state: $S_t = S_{t-1} + \phi(k_t) v_t^\top$, $S_t \in \mathbb R^{d \times d}$
- Denominator state: $z_t = z_{t-1} + \phi(k_t)$, $z_t \in \mathbb R^d$

### 정의 3.4 — RWKV Architecture (Peng 2023)

Time-mixing block:
$$
\begin{aligned}
r_t &= \sigma(W_r \mu_r x_t + U_r x_{t-1}) \\
k_t &= W_k \mu_k x_t + U_k x_{t-1} \\
v_t &= W_v \mu_v x_t + U_v x_{t-1} \\
\mathrm{wkv}_t &= \frac{\sum_{s \le t} e^{-(t-s) w} \cdot e^{k_s} v_s + e^{u + k_t} v_t}{\sum_{s \le t} e^{-(t-s) w} \cdot e^{k_s} + e^{u + k_t}}
\end{aligned}
$$

Time decay parameter $w$, bonus $u$.

Channel-mixing: standard FFN with Gating.

### 정의 3.5 — Performer (Choromanski 2020)

Random positive feature map:
$$
\phi(x) = \frac{1}{\sqrt{m}} \exp\left(\omega^\top x - \|x\|^2/2\right)
$$

$\omega \sim \mathcal N(0, I)$ random projections.

Unbiased estimator of softmax kernel: $\mathbb E[\phi(x)^\top \phi(y)] = \exp(x^\top y)$.

---

## 🔬 정리와 결과

### 정리 3.1 — Linear Attention Complexity

| | Time | Memory |
|--|------|--------|
| **Softmax attention** | $O(T^2 d)$ | $O(T^2 + Td)$ |
| **Linear attention** | $O(T d^2)$ | $O(d^2 + Td)$ |

For long $T$ ($T \gg d$): linear attention 우월.

### 정리 3.2 — RNN-like Recurrence

Causal linear attention 이 정확히 RNN update rule:
$$
S_t = S_{t-1} + \phi(k_t) v_t^\top
$$

**증명**: Causal sum $\sum_{s \le t} \phi(k_s) v_s^\top$ 의 recurrent computation. $\square$

**Implication**: Inference 시 $S$ 만 유지 (no KV-cache), $O(d^2)$ memory per layer.

### 정리 3.3 — Approximation Quality

Linear attention 의 표현력 < softmax attention:
- Softmax: arbitrary attention pattern (sharp, sparse 가능)
- Linear: low-rank approximation, smooth

**Empirical**: Linear attention 이 일부 task (long-range arena) 에서 softmax 와 동등, 일부 (NLP downstream) 에서 약함.

### 정리 3.4 — Performer 의 Unbiased Estimator

Random positive features:
$$
\mathbb E[\phi(x)^\top \phi(y)] = \exp(x^\top y)
$$

**증명** (sketch): Gaussian integral identity, $\int e^{\omega^\top (x-y)} d\omega \propto e^{(x-y)^\top (x-y)/2}$. $\square$

**Variance**: $O(1/\sqrt{m})$, $m$ = number of random features.

### 정리 3.5 — RWKV vs Transformer Comparison

RWKV 14B (2023):
- Comparable to GPT-Neo, OPT 6.7B in benchmarks
- Linear inference time
- Constant memory
- Open-source LLM

**Modern positioning**: Mamba 가 RWKV 의 후속 (더 효율적), 그러나 RWKV 의 정신은 same.

---

## 💻 PyTorch 구현 검증

### 실험 1 — Linear Attention 단순 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    """Katharopoulos 2020 — phi = ELU + 1"""
    def __init__(self, H, d_K=None):
        super().__init__()
        if d_K is None:
            d_K = H
        self.W_Q = nn.Linear(H, d_K, bias=False)
        self.W_K = nn.Linear(H, d_K, bias=False)
        self.W_V = nn.Linear(H, H, bias=False)
        self.d_K = d_K
    
    def feature_map(self, x):
        return F.elu(x) + 1   # Always positive
    
    def forward(self, x, causal=True):
        # x: (T, B, H)
        T, B, H = x.shape
        Q = self.feature_map(self.W_Q(x))   # (T, B, d_K)
        K = self.feature_map(self.W_K(x))   # (T, B, d_K)
        V = self.W_V(x)                      # (T, B, H)
        
        if causal:
            # RNN-like recurrence
            S = torch.zeros(B, self.d_K, H, device=x.device)
            z = torch.zeros(B, self.d_K, device=x.device)
            outputs = []
            for t in range(T):
                S = S + Q[t].unsqueeze(-1) * 0   # placeholder, S not Q-dependent
                S = S + K[t].unsqueeze(-1) * V[t].unsqueeze(1)   # outer product
                z = z + K[t]
                # Output = Q_t · S_t / (Q_t · z_t)
                num = (Q[t].unsqueeze(1) * S).sum(1)   # (B, H)
                den = (Q[t] * z).sum(-1, keepdim=True) + 1e-9
                outputs.append(num / den)
            return torch.stack(outputs)
        else:
            # Non-causal: full computation
            # phi(K)^T V: (B, d_K, H)
            KV = torch.einsum('tbd,tbh->bdh', K, V)
            # Q · KV: (T, B, H)
            num = torch.einsum('tbd,bdh->tbh', Q, KV)
            den = torch.einsum('tbd,bd->tb', Q, K.sum(0)).unsqueeze(-1) + 1e-9
            return num / den

# Test
torch.manual_seed(0)
T, B, H = 20, 4, 32
linear_attn = LinearAttention(H)
x = torch.randn(T, B, H)

out_causal = linear_attn(x, causal=True)
out_full = linear_attn(x, causal=False)
print(f'Causal output:    {out_causal.shape}')
print(f'Non-causal output: {out_full.shape}')
print(f'Causal 마지막 step ≈ non-causal? {(out_causal[-1] - out_full[-1]).abs().max():.4e}')
```

### 실험 2 — Linear vs Softmax Attention Speed

```python
import time

class StandardAttention(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.attn = nn.MultiheadAttention(H, num_heads=4, batch_first=False)
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def time_op(fn, n_iter=20):
    for _ in range(5):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start) / n_iter * 1000

print('Speed comparison:')
print(f'{"T":>6} {"Standard":>12} {"Linear":>10} {"Speedup":>10}')

for T in [50, 100, 500, 1000]:
    H = 64
    B = 8
    
    std_attn = StandardAttention(H).to(device)
    lin_attn = LinearAttention(H).to(device)
    x = torch.randn(T, B, H, device=device)
    
    t_std = time_op(lambda: std_attn(x))
    t_lin = time_op(lambda: lin_attn(x, causal=False))
    print(f'{T:>6} {t_std:>10.2f}ms {t_lin:>8.2f}ms {t_std/t_lin:>8.2f}x')
# T 클수록 linear attention 의 우위
```

### 실험 3 — Memory Comparison

```python
def measure_memory(fn):
    if device != 'cuda':
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    return torch.cuda.max_memory_allocated() / 1e6

if device == 'cuda':
    print('\nMemory comparison:')
    H = 64; B = 8
    for T in [100, 1000, 5000]:
        std_attn = StandardAttention(H).cuda()
        lin_attn = LinearAttention(H).cuda()
        x = torch.randn(T, B, H, device='cuda', requires_grad=True)
        
        m_std = measure_memory(lambda: std_attn(x).sum().backward())
        m_lin = measure_memory(lambda: lin_attn(x, causal=False).sum().backward())
        print(f'T={T:5d}: Standard={m_std:.1f}MB, Linear={m_lin:.1f}MB')
```

### 실험 4 — Recurrent Inference (RNN-like)

```python
class LinearAttentionRecurrent(nn.Module):
    """Inference 시 state 만 유지"""
    def __init__(self, H, d_K=None):
        super().__init__()
        if d_K is None:
            d_K = H
        self.W_Q = nn.Linear(H, d_K, bias=False)
        self.W_K = nn.Linear(H, d_K, bias=False)
        self.W_V = nn.Linear(H, H, bias=False)
        self.d_K = d_K
        self.H = H
    
    def feature_map(self, x):
        return F.elu(x) + 1
    
    def step(self, x_t, S, z):
        """Single step: O(d_K · H) computation, O(d_K · H) state"""
        Q_t = self.feature_map(self.W_Q(x_t))
        K_t = self.feature_map(self.W_K(x_t))
        V_t = self.W_V(x_t)
        
        S = S + K_t.unsqueeze(-1) * V_t.unsqueeze(-2)
        z = z + K_t
        
        out = (Q_t.unsqueeze(-2) * S).sum(-2) / ((Q_t * z).sum(-1, keepdim=True) + 1e-9)
        return out, S, z
    
    def forward_streaming(self, x_seq):
        """Streaming inference"""
        T, B, _ = x_seq.shape
        S = torch.zeros(B, self.d_K, self.H, device=x_seq.device)
        z = torch.zeros(B, self.d_K, device=x_seq.device)
        outputs = []
        for t in range(T):
            out, S, z = self.step(x_seq[t], S, z)
            outputs.append(out)
        return torch.stack(outputs), S

# Inference state size = d_K × H — *constant* per sample (T 와 무관)
H, d_K = 64, 32
recurrent_attn = LinearAttentionRecurrent(H, d_K)
x = torch.randn(100, 4, H)
out, final_state = recurrent_attn.forward_streaming(x)
print(f'Streaming output: {out.shape}')
print(f'Final state size: {final_state.numel()} floats')
print(f'For T=100, B=4: {final_state.numel()} (vs KV-cache {100 * H * 2 * 4})')
# Linear attention: O(d_K × H), KV-cache: O(T × H)
```

### 실험 5 — RWKV-style Time Mixing (단순화)

```python
class RWKV_TimeMixing(nn.Module):
    """RWKV 의 time-mixing block (simplified)"""
    def __init__(self, H):
        super().__init__()
        self.H = H
        # Mixing parameters
        self.mu_r = nn.Parameter(torch.zeros(H))
        self.mu_k = nn.Parameter(torch.zeros(H))
        self.mu_v = nn.Parameter(torch.zeros(H))
        # Linear projections
        self.W_r = nn.Linear(H, H, bias=False)
        self.W_k = nn.Linear(H, H, bias=False)
        self.W_v = nn.Linear(H, H, bias=False)
        self.W_o = nn.Linear(H, H, bias=False)
        # Time decay
        self.w = nn.Parameter(torch.zeros(H))   # log-decay
        self.u = nn.Parameter(torch.zeros(H))   # bonus
    
    def forward(self, x):
        """x: (T, B, H)"""
        T, B, H = x.shape
        
        # Token shift
        x_prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)
        
        # Mix current and previous
        xr = x * (1 + self.mu_r) - x_prev * self.mu_r
        xk = x * (1 + self.mu_k) - x_prev * self.mu_k
        xv = x * (1 + self.mu_v) - x_prev * self.mu_v
        
        r = torch.sigmoid(self.W_r(xr))
        k = self.W_k(xk)
        v = self.W_v(xv)
        
        # Recurrent WKV computation
        wkv_outputs = []
        a = torch.zeros(B, H, device=x.device)   # Numerator state
        b = torch.zeros(B, H, device=x.device)   # Denominator state
        for t in range(T):
            # Bonus term
            num = a + torch.exp(self.u + k[t]) * v[t]
            den = b + torch.exp(self.u + k[t])
            wkv_outputs.append(num / (den + 1e-9))
            # Update states with time decay
            decay = torch.exp(-torch.exp(self.w))
            a = decay * a + torch.exp(k[t]) * v[t]
            b = decay * b + torch.exp(k[t])
        
        wkv = torch.stack(wkv_outputs)
        return self.W_o(r * wkv)

torch.manual_seed(0)
rwkv_block = RWKV_TimeMixing(H=64)
x = torch.randn(20, 4, 64)
out = rwkv_block(x)
print(f'RWKV time-mixing output: {out.shape}')
print(f'Parameters: {sum(p.numel() for p in rwkv_block.parameters())}')
```

---

## 🔗 실전 활용

### 1. Long context LLM

GPT-NeoX-Mamba, RWKV-7B — 100K+ context 처리.

### 2. Real-time inference

Edge AI — KV-cache 없는 RNN-like inference.

### 3. Streaming generation

Audio, video real-time generation 의 fast inference.

### 4. Memory-constrained training

Long sequence training 시 linear attention 의 $O(T)$ memory.

### 5. Hybrid models

Jamba (AI21 2024): Mamba + Transformer layers — best of both.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Kernel approximation 충분 | Sharp attention 표현 어려움 |
| Linear feature map | Higher-order nonlinearity 한계 |
| RNN-like state | Specific state size 제한 |
| Causal masking 자연스러움 | Bidirectional 어려움 |
| Approximation accurate | Some tasks 에서 quality drop |

---

## 📌 핵심 정리

$$\boxed{\mathrm{Attn} = \phi(Q) (\phi(K)^\top V) \quad \text{— associativity 변경}}$$

$$\boxed{S_t = S_{t-1} + \phi(k_t) v_t^\top \quad \text{— RNN-like recurrence}}$$

$$\boxed{\text{Time/Memory: } O(T) \text{ (vs softmax } O(T^2))}$$

| Method | Approximation | Time | Memory | Quality |
|--------|--------------|------|--------|---------|
| **Softmax** | Exact | $O(T^2)$ | $O(T^2)$ | Best |
| **Linear (ELU+1)** | Crude | $O(T)$ | $O(T)$ | OK |
| **Performer** | Random features | $O(T)$ | $O(T)$ | Better |
| **RWKV** | Linear + gating | $O(T)$ | $O(T)$ | Competitive |
| **Mamba (next)** | Selective SSM | $O(T)$ | $O(T)$ | Best long context |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Linear attention 의 state size 가 standard attention 의 KV-cache 보다 작은 이유를 정량적으로 비교하라.

<details>
<summary>해설</summary>

**Standard attention KV-cache** (inference):
- Past tokens: $T$ steps
- Per token: 2 vectors (K, V) of dim $d_K, d_V$
- Per layer per head: $T \cdot (d_K + d_V)$
- Total: $L \cdot H \cdot T \cdot (d_K + d_V)$ for $L$ layers, $H$ heads

For GPT-3 (175B):
- $L = 96$, $H = 96$, $d_K = d_V = 128$
- Per token: $L \cdot H \cdot 2d = 96 \cdot 96 \cdot 256 = 2.36M$ floats
- For $T = 2048$: $4.8B$ floats = 19 GB (fp32)

**Linear attention state**:
- $S \in \mathbb R^{d_K \times d_V}$, $z \in \mathbb R^{d_K}$
- Per layer per head: $d_K \cdot d_V + d_K$
- *Independent of $T$*

For same model:
- Per layer per head: $128 \cdot 128 + 128 = 16{,}512$ floats
- Total: $L \cdot H \cdot 16K = 152M$ floats = 0.6 GB

**비교**:
- KV-cache: $19$ GB (for $T = 2048$)
- Linear state: $0.6$ GB (any $T$)
- **30x smaller**

For $T = 100K$: KV-cache 940 GB (infeasible), linear 0.6 GB.

**Implication**:
- Linear attention 이 long context inference 에 critical
- Modern LLM 의 long context 의 enable

$\square$

</details>

**문제 2** (심화): Linear attention 이 standard attention 의 *low-rank approximation* 인 이유를 설명하라. Rank 제한이 표현력에 어떤 영향을 주는가?

<details>
<summary>해설</summary>

**Standard attention** (full rank):
$$
A = \mathrm{softmax}(QK^\top / \sqrt{d}) \in \mathbb R^{T \times T}
$$

Rank 가 최대 $\min(T, d)$. For $T \gg d$, full rank $d$.

**Linear attention**:
$$
A = \phi(Q) \phi(K)^\top \in \mathbb R^{T \times T}
$$

Rank 가 최대 $d_\phi$ (feature map dimension).

**Low-rank 의 기하학적 의미**:
- Standard: each row $A_i$ 는 *arbitrary* $T$-dim distribution
- Linear: each row $A_i$ 는 $d_\phi$-dim subspace 안의 distribution

**표현력 영향**:

1. **Sharp attention**:
   - Standard: 한 position 에 sharp peak 가능 (one-hot)
   - Linear: $d_\phi$-dim subspace 에서 *smooth* distribution
   - 매우 sparse attention pattern 표현 어려움

2. **Long-range modeling**:
   - Standard: any pair (i, j) 의 strong attention 가능
   - Linear: pairs 가 $d_\phi$-dim 으로 compressed
   - **Information bottleneck** — long sequence 에서 일부 정보 lost

3. **In-context learning**:
   - Standard: arbitrary patterns from context
   - Linear: limited by $d_\phi$
   - **Empirical gap** in few-shot tasks

**Performer 의 random features**:
- $d_\phi = m$ (random projections)
- Variance $O(1/\sqrt{m})$
- $m \to \infty$: standard attention 으로 수렴
- 실제 $m$ 는 $T \cdot d^2 / \log T$ 정도 — 여전히 sub-quadratic

**RWKV / Mamba 의 보완**:
- Linear attention 의 한계를 *gating* 또는 *selective* 메커니즘으로 우회
- 표현력 향상

**Empirical**:
- LRA benchmark: linear attention 이 standard 와 동등
- NLP downstream (GLUE): standard 우월
- **Task-dependent** — long-range 에서 linear, NLP 에서 standard

**결론**: Linear attention 이 *low-rank* approximation — efficient 그러나 표현력 trade-off. Modern hybrid (Mamba) 가 selective gating 으로 표현력 회복. **No free lunch** — efficiency 와 expressiveness 의 trade-off. $\square$

</details>

**문제 3** (논문 비평): RWKV 와 Mamba 가 Transformer 의 in-context learning 능력을 부분적 잃지만 long context efficient. 이 trade-off 가 modern LLM 의 future 에 어떤 의미인가?

<details>
<summary>해설</summary>

**In-context learning (ICL)**:
- Transformer 의 emergent property
- Few-shot examples 로 task 학습 (no fine-tuning)
- GPT-3, GPT-4 의 capability foundation

**ICL 의 mechanism (가설)**:
- Attention 이 examples 와 query 사이 *relationship* 추출
- "Implicit gradient descent" 가 attention 에서 일어남 (Akyürek 2022)
- Sharp, position-specific attention 이 핵심

**RWKV / Mamba 의 ICL 한계**:

1. **Linear attention 의 low-rank**:
   - Sharp peak attention 어려움
   - Specific examples 에 strong attention 못함
   - **ICL 약함**

2. **Recurrent state 의 capacity**:
   - $O(d^2)$ state 가 *모든* past 압축
   - Distinct examples 의 정확한 retrieval 어려움
   - **Long-context ICL 특히 약함**

3. **Empirical**:
   - RWKV-14B: ICL benchmarks 에서 GPT-3 보다 약함
   - Mamba: 일부 task 에서 동등, 일부에서 약함
   - LongBench, MMLU 의 결과 mixed

**Long context efficiency**:

- 1M+ tokens 를 처리 (GPT-4 의 32K 와 비교)
- $O(T)$ time, $O(d^2)$ state
- Streaming 가능

**Trade-off 의 future implications**:

1. **Hybrid architectures**:
   - **Jamba** (AI21 2024): Mamba + Transformer layers
   - Mamba 가 long-range routing, attention 이 sharp lookup
   - Best of both worlds

2. **Task-specific architecture**:
   - Long-context summarization: Mamba
   - Few-shot reasoning: Transformer
   - General LLM: Hybrid

3. **Different scales**:
   - Small model (< 1B): Transformer 충분
   - Large + long context: Mamba/RWKV
   - LLM (10B+): Hybrid

4. **Pre-training paradigm**:
   - Transformer: standard pre-train + fine-tune
   - Mamba: pre-train challenges (특히 long context)
   - **Pre-training 의 algorithm 변경 필요**

**현대 (2024) trend**:

1. **Mamba 의 부상**:
   - Mamba-7B, Jamba-52B
   - Long context (1M tokens) 가능
   - 그러나 GPT-4 quality 에는 못 미침

2. **Hybrid 의 표준화**:
   - Linformer + standard attention
   - Conformer (audio)
   - DeepMind Hawk-Griffin (recurrent + attention)

3. **Architectural diversity**:
   - 2017-2022: Transformer dominant
   - 2023-2024: alternatives 부상
   - **Single architecture 로 수렴 안 함**

**Lessons**:

1. **No silver bullet**:
   - Each architecture 의 strength
   - Task / scale 의 함수

2. **Capacity vs computation trade-off**:
   - Transformer: high capacity, $O(T^2)$ cost
   - Mamba: efficient, lower capacity per token
   - 절충: hybrid

3. **Pre-training matters**:
   - Architecture 보다 *data* 와 *scale* 이 자주 결정
   - 같은 architecture 의 different training 이 different capability

**Future predictions**:

- **Short term (1-2 years)**: Hybrid architectures dominant
- **Medium**: New paradigm (e.g., explicit memory + computation 분리)
- **Long term**: Architecture-agnostic (data + scale 의 함수)

**결론**: RWKV/Mamba 가 long context efficiency 의 *enabler*, 그러나 ICL 의 *trade-off*. **Modern LLM 의 future 가 hybrid architecture 와 pre-training innovation**. Single architecture 의 dominance 시대는 *over* — architectural diversity 가 새 norm. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-cnn-sequence.md) | [📚 README](../README.md) | [다음 ▶](./04-s4-mamba.md)

</div>
