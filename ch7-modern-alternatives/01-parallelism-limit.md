# 01. 병렬성 부족 — Transformer 의 동기

## 🎯 핵심 질문

- **정리**: RNN 의 forward $h_t = f(h_{t-1}, x_t)$ 의 sequential 의존이 어떻게 sequence 내부 병렬화를 *fundamentally* 차단하는가?
- GPU utilization 이 sequence 길이 $T$ 에 반비례하는 정확한 이유?
- Transformer 의 self-attention 이 $O(T^2)$ 시간/공간 복잡도 trade-off 로 어떻게 *완전 병렬* 을 달성했는가?
- 동일 hardware 에서 LSTM vs Transformer 훈련 시간 측정 (Vaswani 2017 Table 1 재현)
- 왜 LSTM 이 Transformer 의 등장 (2017) 으로 대체되었는지의 *quantitative* 증거

---

## 🔍 왜 sequence parallelism 이 modern sequence model 의 결정적 요소인가

GPU 시대 이후 ML 의 모든 architecture 변화는 *parallelism* 의 함수:

1. **CNN > MLP**: Spatial parallelism (이미지의 공간 차원)
2. **Transformer > LSTM**: Sequence parallelism (sequence 차원)
3. **Mamba > Transformer**: Linear time + parallel scan (긴 sequence)

이 모든 변화의 root cause: **GPU 의 SIMT (Single Instruction Multiple Thread)**. 수천 개 cores 가 동시에 같은 연산을 다른 data 에 적용 — *sequential dependency* 가 utilization 의 적.

이 문서는 RNN 의 sequential 한계의 정확한 분석, Transformer 의 해법, 그리고 SSM (Mamba) 까지의 진화를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch6-05 Coverage and Pointer](../ch6-seq2seq-attention/05-coverage-pointer.md)
- [Ch2-04 BPTT 복잡도](../ch2-bptt/04-complexity.md) — Sequence parallelism limit
- 컴퓨터 구조: SIMT, GPU memory hierarchy

---

## 📖 직관적 이해

### Sequential Dependency 시각화

```
RNN:  h_0 → h_1 → h_2 → h_3 → h_4 → ...
       │     │     │     │     │
      각 step 이 이전 step 결과 필요
      
GPU 의 thousands of cores:
   t=1:  [ core_0, core_1, ..., core_4096 ]   ← 한 step 만 사용
   t=2:  [ core_0, core_1, ..., core_4096 ]   ← 다음 step
   ...
   
  Utilization per step: 1 / number_of_cores
```

Cores 의 1/N 만 사용 — 거대한 낭비.

### Transformer 의 병렬화

```
Self-Attention:
   모든 (i, j) pair 의 attention 동시 계산
   
   t=1:  [ all (i, j) pairs computed in parallel ]
         using all cores at once
         
   Utilization: 100% (in single matmul)
```

$T \times T$ attention matrix → matmul → 모든 cores 활용.

### Trade-off

```
                Time         Memory      Parallel
RNN (LSTM)     O(T·H²)      O(T·H)      X (sequential)
Transformer    O(T²·H)      O(T²)       O (parallel)
Mamba (SSM)    O(T·H)       O(T·H)      O (parallel scan)
```

- Transformer: 더 많은 work ($T^2$ vs $T$), 그러나 *parallel* — wall-clock 빠름
- Mamba: best of both — parallel scan 으로 linear

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Sequential Computation

함수 $f$ 의 sequential computation:
$$
y_t = f(y_{t-1}, x_t)
$$

$y_t$ 의 계산이 $y_{t-1}$ 의 결과 *필요* — depth $T$ 의 dependency chain.

### 정의 1.2 — Parallel Computation

함수 $f$ 의 parallel computation:
$$
y_i = f(x_i, x_{1:T}), \quad \text{independent across } i
$$

모든 $y_i$ 가 동시 계산 가능 — depth $1$.

### 정의 1.3 — Critical Path Length

함수의 sequential dependency depth:

- RNN: $T$
- Tree (e.g., reduction): $\log T$
- Embarrassingly parallel: $1$

### 정의 1.4 — GPU Utilization

$$
U = \frac{\text{actually used cores}}{\text{total cores}}
$$

Sequential RNN: $U \approx 1/T$ asymptotically.
Transformer: $U \approx 1$ (within matmul).

### 정의 1.5 — Wall-clock Time

$$
T_{\text{wall}} = \frac{T_{\text{compute}}}{U \cdot P}
$$

$P$: number of cores. Sequential 일수록 $T_{\text{wall}}$ 큼.

---

## 🔬 정리와 결과

### 정리 1.1 — RNN 의 Critical Path

RNN forward 의 critical path length = $T$ (sequence length).

**증명**: $h_T = f(h_{T-1}, x_T) = f(f(h_{T-2}, x_{T-1}), x_T) = \ldots$ — depth $T$ chain. $\square$

**Implication**: GPU 의 thousands of parallel cores 가 활용 안 됨 — utilization $= O(1/T)$.

### 정리 1.2 — Transformer 의 Critical Path

Self-attention 의 critical path = $O(\log T)$ (matmul reduction).

**증명**: Attention matrix $A = QK^\top$ 의 각 entry 가 *independently* 계산 가능. Matmul 의 reduction 이 tree-reduce → $\log T$ depth. $\square$

**Wall-clock**: $T_{\text{wall}}^{\text{Transformer}} = O(\log T) \ll T_{\text{wall}}^{\text{RNN}} = O(T)$.

### 정리 1.3 — Vaswani 2017 의 Empirical Comparison

WMT'14 En→De training time:
- Google NMT (LSTM): 16 days × 8 GPUs
- Transformer-Big: 3.5 days × 8 GPUs

**Speedup**: ~5x 빠름 (with better BLEU).

### 정리 1.4 — Memory Trade-off

Transformer: $O(T^2)$ memory for attention matrix.
- $T = 10000$: $10^8$ entries — 400 MB (fp32)
- $T = 100000$: $10^{10}$ — 40 GB (한 layer 만!)

이는 long sequence 의 한계 → Linear attention (Ch7-03), Mamba (Ch7-04).

### 정리 1.5 — Utilization at Scale

GPU H100 의 약 16K cores. RNN forward 의 utilization:
- $T = 100$: $\approx 16K / 100 \cdot \text{matmul size} \approx 1\%$
- $T = 10000$: $\approx 0.01\%$

Transformer 는 100% (within matmul).

---

## 💻 Empirical Measurement

### 실험 1 — RNN vs Transformer Forward Speed

```python
import torch
import torch.nn as nn
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TimingLSTM(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.lstm = nn.LSTM(D, H, batch_first=False)

class TimingTransformer(nn.Module):
    def __init__(self, D, H, n_head=4, n_layer=2):
        super().__init__()
        self.proj = nn.Linear(D, H)
        layer = nn.TransformerEncoderLayer(d_model=H, nhead=n_head, dim_feedforward=H*4)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)

def time_forward(model, x, n_iter=20):
    # Warmup
    for _ in range(5):
        if isinstance(model, TimingLSTM):
            model.lstm(x)
        else:
            h = model.proj(x)
            model.encoder(h)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_iter):
        if isinstance(model, TimingLSTM):
            model.lstm(x)
        else:
            h = model.proj(x)
            model.encoder(h)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start) / n_iter * 1000   # ms

D, H, B = 64, 256, 32
torch.manual_seed(0)

print(f'Forward time (B={B}, H={H}):')
print(f'{"T":>6} {"LSTM":>10} {"Transformer":>14} {"Ratio":>8}')

for T in [10, 50, 100, 500, 1000]:
    lstm_model = TimingLSTM(D, H).to(device)
    trf_model = TimingTransformer(D, H).to(device)
    x = torch.randn(T, B, D, device=device)
    
    t_lstm = time_forward(lstm_model, x)
    t_trf = time_forward(trf_model, x)
    ratio = t_lstm / t_trf
    print(f'{T:>6} {t_lstm:>10.2f} {t_trf:>14.2f} {ratio:>8.2f}x')

# T 클수록 Transformer 의 우위 명확
```

### 실험 2 — Critical Path 측정

```python
# 단일 sample (B=1) 의 sequential 영향 강조
T_vals = [10, 100, 1000]

print('\nForward time with B=1 (sequential dependency dominant):')
for T in T_vals:
    lstm = nn.LSTM(D, H).to(device)
    trf = nn.TransformerEncoderLayer(H, 4).to(device)
    
    x = torch.randn(T, 1, D, device=device)
    t_lstm = time_forward(TimingLSTM(D, H).to(device), x)
    
    h = nn.Linear(D, H).to(device)(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(20):
        trf(h)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_trf = (time.time() - start) / 20 * 1000
    
    print(f'T={T:5d}: LSTM={t_lstm:.2f}ms, Transformer={t_trf:.2f}ms, ratio={t_lstm/t_trf:.2f}x')
```

### 실험 3 — Memory Scaling

```python
def measure_memory(model_fn, T, B=4):
    if device != 'cuda':
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = model_fn().to(device)
    x = torch.randn(T, B, D, requires_grad=True, device=device)
    if isinstance(model, nn.LSTM):
        out, _ = model(x)
    else:
        out = model(x)
    out.sum().backward()
    return torch.cuda.max_memory_allocated() / 1e6

if device == 'cuda':
    print('\nMemory scaling:')
    print(f'{"T":>6} {"LSTM":>12} {"Transformer":>15}')
    for T in [100, 500, 1000, 2000]:
        m_lstm = measure_memory(lambda: nn.LSTM(D, H), T)
        m_trf = measure_memory(
            lambda: nn.TransformerEncoderLayer(H, 4),
            T
        )
        # Note: Transformer 도 D → H projection 필요, 단순화
        print(f'{T:>6} {m_lstm:>10.1f}MB {m_trf:>13.1f}MB')

# Transformer 의 O(T^2) memory 가 큰 T 에서 문제
```

### 실험 4 — GPU Utilization 시뮬레이션

```python
# 가상의 utilization 계산
def gpu_utilization_estimate(T, n_cores=16384, matmul_size=64*64):
    """Single sequence forward 의 utilization estimate"""
    rnn_per_step_work = matmul_size
    rnn_total_steps = T
    rnn_effective_parallel = rnn_per_step_work / n_cores
    rnn_utilization = min(1.0, rnn_effective_parallel)
    
    trf_total_work = T * T * matmul_size   # attention 의 work
    trf_parallel_work = trf_total_work / n_cores
    trf_utilization = min(1.0, trf_parallel_work)
    
    return rnn_utilization, trf_utilization

print('\nGPU Utilization estimate:')
for T in [10, 100, 1000, 10000]:
    u_rnn, u_trf = gpu_utilization_estimate(T)
    print(f'T={T:5d}: RNN={u_rnn*100:.2f}%, Transformer={u_trf*100:.2f}%')
# T 클수록 RNN 의 utilization gap 현저
```

### 실험 5 — Vaswani 2017 Style Comparison

```python
# Approximate WMT En→De training 비교 (toy scale)
class ToyMT(nn.Module):
    def __init__(self, V, D, H, model_type='lstm'):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.out = nn.Linear(H, V)
        if model_type == 'lstm':
            self.encoder = nn.LSTM(D, H, batch_first=False)
            self.decoder = nn.LSTM(D, H, batch_first=False)
        else:
            enc_layer = nn.TransformerEncoderLayer(H, 4, dim_feedforward=H*4)
            dec_layer = nn.TransformerDecoderLayer(H, 4, dim_feedforward=H*4)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
            self.proj = nn.Linear(D, H)

V, D, H = 1000, 32, 64
T_src, T_tgt, B = 50, 50, 32
torch.manual_seed(0)

print('\n5-step training time:')
for model_type in ['lstm', 'transformer']:
    model = ToyMT(V, D, H, model_type).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    src = torch.randint(0, V, (T_src, B), device=device)
    tgt = torch.randint(0, V, (T_tgt, B), device=device)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(5):
        emb_src = model.emb(src)
        emb_tgt = model.emb(tgt)
        if model_type == 'lstm':
            enc, _ = model.encoder(emb_src)
            dec, _ = model.decoder(emb_tgt)
            logits = model.out(dec)
        else:
            enc = model.encoder(model.proj(emb_src))
            dec = model.decoder(model.proj(emb_tgt), enc)
            logits = model.out(dec)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f'  {model_type:12s}: {elapsed:.2f} s for 5 training steps')
```

---

## 🔗 실전 활용

### 1. Modern NLP architecture choice

새 NLP 모델 설계 시 default 가 Transformer. RNN 은 specific edge cases (streaming, edge AI).

### 2. Sequence length 의 결정

Long context (>10K) 의 model 설계 시 sequence parallelism 의 trade-off 가 결정적:
- Linear attention (Ch7-03)
- Sparse attention (Longformer, Big Bird)
- State Space Model (Mamba, Ch7-04)

### 3. Hardware-aware design

Modern model 의 design 이 GPU/TPU 의 architectural feature 에 맞춰짐:
- Tensor cores
- Memory hierarchy
- Inter-GPU communication

### 4. FlashAttention

Transformer 의 attention 을 GPU memory hierarchy 에 최적화 — sequence parallelism 의 hardware-aware refinement.

### 5. Streaming inference

Inference 시 LSTM 의 sequential 이 *advantage* 가 됨 (state 만 유지). RNN-like inference 가 부활.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| GPU 의 SIMT 표준 | TPU, neuromorphic 의 다른 architecture |
| $O(T^2)$ memory acceptable | Long context 에서 한계 |
| Sequence parallelism 우월 | Streaming 에서는 sequential 이 유리 |
| Static $T$ | Dynamic $T$ 의 padding 이슈 |
| FP32 precision | Mixed precision 가 표준 |

---

## 📌 핵심 정리

$$\boxed{\text{RNN critical path: } O(T), \quad \text{Transformer: } O(\log T)}$$

$$\boxed{\text{Transformer time: } O(T^2 H) \text{ but parallel}}$$

$$\boxed{\text{RNN time: } O(T H^2) \text{ but sequential — wall-clock slower}}$$

| Architecture | Time | Memory | Critical Path | Wall-clock |
|--------------|------|--------|---------------|------------|
| **RNN/LSTM** | $O(TH^2)$ | $O(TH)$ | $T$ | $O(T)$ |
| **Transformer** | $O(T^2 H)$ | $O(T^2)$ | $O(\log T)$ | $O(\log T)$ |
| **Mamba (SSM)** | $O(TH)$ | $O(TH)$ | $O(\log T)$ | $O(\log T)$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GPU 16K cores, $T = 1000$, $H = 1024$ 의 RNN 과 Transformer 의 utilization 을 비교하라.

<details>
<summary>해설</summary>

**RNN per step**:
- Matmul: $H^2 = 10^6$ operations
- Cores used: $\min(16K, 10^6) = 16K$ (saturate)
- However, $T = 1000$ steps sequential → total time = $1000 \times t_{\text{step}}$

**RNN utilization**:
- Per step: 100% (16K cores all used)
- Effective: 100% per step, but sequential → 1 step at a time
- Wall-clock vs work: $T$ times slower than parallel computation

**Transformer per step**:
- $T^2 H = 10^9$ operations (attention)
- Parallel within matmul
- Cores used: 16K (saturate)
- Total time: 1 step (with parallel matmul)

**Transformer utilization**:
- 100% during matmul
- $T^2$ total work but in $O(\log T)$ depth → 1 wall-clock step

**Wall-clock comparison**:
- RNN: $T \cdot t_{\text{step}}$
- Transformer: $t_{\text{matmul}}$ ($T^2$ ops in parallel)
- For $T = 1000$, $H = 1024$:
  - RNN: $1000 \cdot 1$ μs = 1 ms
  - Transformer: $1$ μs (single matmul)
  - **1000x speedup** (이론적, real 은 30-100x)

**결론**: RNN 의 *intra-step* utilization 은 좋지만 *sequential dependency* 가 wall-clock 의 bottleneck. Transformer 가 utilization + parallelism 두 가지 win. $\square$

</details>

**문제 2** (심화): Transformer 의 $O(T^2)$ memory 가 long context 에서 한계를 만든다. 실용적 해법들을 정리하라.

<details>
<summary>해설</summary>

**$O(T^2)$ memory 의 영향**:
- $T = 10K$: $10^8$ entries × 4 bytes = 400 MB per layer per head
- $T = 100K$: $10^{10}$ → 40 GB
- 12-layer × 12-head Transformer: 멀티화 → infeasible

**해법들**:

1. **Sparse attention** (Reformer, Longformer):
   - $O(T \sqrt T)$ 또는 $O(T \log T)$
   - Local + sliding + global tokens
   - Trade-off: 표현력 일부 손실

2. **Linear attention** (Performer, Linformer):
   - Kernel approximation: $\phi(Q) \phi(K)^\top \approx QK^\top$
   - $O(T)$ memory and time
   - Approximate, but practical (Ch7-03)

3. **FlashAttention** (Dao 2022):
   - 같은 $O(T^2)$ work, but smarter memory access
   - $O(T)$ memory by tiling
   - 2-4x speedup with same accuracy

4. **State Space Model** (Mamba):
   - $O(T)$ time and memory
   - Recurrent inference
   - Different architecture, same long context (Ch7-04)

5. **Sliding window** (Mistral, Gemma):
   - Fixed window $W \ll T$
   - $O(T \cdot W)$ memory
   - 적정한 $W$ 가 long context 충분

6. **Compression**:
   - 압축된 KV-cache
   - Merging similar tokens
   - **Implicit memory** mechanism

**Hybrid approaches**:
- Mamba + attention layers
- Local attention + sparse global
- Linear attention + selective standard

**현대 trend** (2024):
- Long context (1M+ tokens) 가 standard
- Mamba, RWKV 같은 RNN-like 가 부활
- FlashAttention 의 hardware-aware 정신
- Hybrid architectures (Jamba, Hyena)

**결론**: $O(T^2)$ memory 가 *fundamental* 문제 — pure Transformer 는 long context 에 적합하지 않음. Modern solution 이 attention 의 *core idea* (parallel attention) 와 RNN 의 *core idea* (linear time) 의 best-of-both. $\square$

</details>

**문제 3** (논문 비평): RNN 이 Transformer 에 대체된 후, Mamba (2023) 가 다시 RNN-like architecture 를 부활시켰다. *왜 다시* RNN-like 인가? Sequential dependency 의 한계가 어떻게 극복되었는가?

<details>
<summary>해설</summary>

**Mamba 의 부활**:

1. **Long context 의 부담**:
   - Transformer: $O(T^2)$ — 100K context 에서 infeasible
   - LLM 의 long context window (1M tokens) 추세

2. **RNN 의 inherent advantage**:
   - $O(T)$ time
   - $O(H)$ inference state (KV-cache 불필요)
   - Streaming inference 자연스러움

3. **Sequential 한계의 극복**:
   - **Linear recurrence**: $h_t = A h_{t-1} + B x_t$ — *non-linear gates 없음*
   - **Parallel scan**: linear operations 가 binary tree 로 parallel 가능
   - $O(\log T)$ depth!

**Parallel Scan algorithm** (Blelloch 1990):

Linear recurrence $h_t = A h_{t-1} + B u_t$ 의 closed-form:
$$
h_T = A^T h_0 + \sum_{k=0}^{T-1} A^{T-1-k} B u_k
$$

이는 *associative* operation 의 prefix sum:
- Up-sweep: pair-wise reduction in tree
- Down-sweep: spread back to all positions
- $O(T)$ work, $O(\log T)$ depth

**Mamba 의 구체적 design**:

$$
h_t = \mathrm{diag}(A_t) h_{t-1} + B_t x_t
$$

- $A_t, B_t$ 가 input-dependent (selective)
- Diagonal $A_t$ → element-wise (scalar) recurrence per dim
- Hardware-aware kernel (FlashSSM)

**RNN 과의 차이**:

| | LSTM | Mamba |
|--|------|-------|
| **Recurrence** | Non-linear gates | Linear (selective) |
| **Time** | $O(TH^2)$ sequential | $O(TH)$ parallel scan |
| **Memory** | $O(H)$ state | $O(H)$ state |
| **Long context** | Difficult | Natural |
| **Parallelism** | None | Parallel scan |

**Why now (2023)**:

1. **Theoretical groundwork**:
   - HiPPO (Gu 2020): polynomial projection
   - S4 (Gu 2022): structured matrices
   - Mamba (Gu & Dao 2023): selective + hardware

2. **Empirical**:
   - LRA (Long Range Arena) benchmark 의 motivation
   - Transformer 의 long context 한계 명확
   - Linear attention 의 부분적 success

3. **Hardware-aware engineering**:
   - FlashAttention 의 success가 inspiration
   - GPU memory hierarchy 의 careful 활용
   - Custom CUDA kernels

**Lesson**:

1. **Non-linearity 가 root cause**:
   - LSTM 의 sequential 한계 = gates 의 non-linearity
   - Linear recurrence 는 parallelizable
   - Selective linearity 가 best of both

2. **Architecture 의 cyclical evolution**:
   - RNN (1990s) → LSTM (1997) → Transformer (2017) → SSM (2022+)
   - 같은 idea (long-range modeling) 의 다른 instantiation
   - Hardware advances 가 design choice 영향

3. **No silver bullet**:
   - Transformer 가 모든 task 우월 아님
   - Mamba 도 trade-off 있음 (in-context learning 약함)
   - Hybrid (Jamba: Mamba + attention) 가 현재 trend

**현대 perspective**:

- Mamba: long context 효율적
- Transformer: in-context learning, instruction following 강력
- LLM 의 future: hybrid architectures
- Each architecture 가 *complementary* strengths

**결론**: Mamba 의 부활이 RNN sequential 한계의 *workaround* — linear recurrence + parallel scan. **Sequential dependency 가 본질적 문제가 아니라 *non-linearity* 가 문제**였음. Linear sequential 은 parallelizable. 이 insight 가 modern long-context model 의 base. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch6-seq2seq-attention/05-coverage-pointer.md) | [📚 README](../README.md) | [다음 ▶](./02-cnn-sequence.md)

</div>
