# 04. BPTT 의 시간·메모리 복잡도

## 🎯 핵심 질문

- BPTT 의 forward / backward 시간 복잡도는 정확히 어떻게 분해되는가?
- 메모리 $O(TH)$ 가 어디에서 오며, gradient checkpointing (Chen 2016) 으로 어떻게 $O(\sqrt T)$ 까지 줄이는가?
- $T = 10000$, $H = 1000$ 의 LSTM 학습이 GPU 메모리 한계에 부딪히는 정확한 이유는?
- Sequence-internal **병렬성 부족** — 왜 RNN 이 GPU utilization 을 제한하며 이것이 Transformer 의 동기인가?
- **Batched RNN** 에서 batch dimension 의 병렬화는 어떻게 작동하는가? Sequence dimension 의 차이는?

---

## 🔍 왜 복잡도 분석이 RNN 의 한계를 정의하는가

RNN 의 모든 architectural 선택은 결국 시간/메모리 복잡도의 trade-off:

1. **Memory $O(TH)$** — Long sequence 학습의 hardware bottleneck
2. **Sequential dependency** — Sequence 내부 병렬화 불가, GPU utilization 의 fundamental limit
3. **Gradient checkpointing** — Memory 를 시간으로 trade
4. **Truncated BPTT** — Memory + accuracy trade
5. **Transformer 의 동기** — $O(T^2)$ 비용 감수 + 완전 병렬

이 문서는 BPTT 의 모든 비용을 명시적으로 분석하고, 현대 sequence 모델 (Linear Attention, Mamba) 이 이를 어떻게 극복하는지 정리합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [03-truncated-bptt.md](./03-truncated-bptt.md) — TBPTT memory 절약
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): 큰 O 표기, asymptotic analysis
- 컴퓨터 구조: GPU memory hierarchy (HBM, L2, register), parallel computation

---

## 📖 직관적 이해

### 시간 복잡도의 분해

**Per-step cost**:
- $W_{hh} h_{t-1}$: $O(H^2)$
- $W_{xh} x_t$: $O(HD)$
- $\tanh$: $O(H)$
- $W_{hy} h_t$: $O(HO)$

Total per step: $O(H^2 + HD + HO + H) = O(H^2)$ (보통 $H \gg D, O$).

**Sequence cost**: $T$ steps × $O(H^2)$ = $O(TH^2)$.

**Batch cost**: $B$ samples 동시 처리 — matmul 효율로 $O(BTH^2)$ 가 되지만 *batch 차원은 GPU 에서 병렬*.

### 메모리의 폭발

```
T = 10000 (long document)
H = 1000  (large model)
B = 32    (batch size)
float32 = 4 bytes

Forward activation: T × B × H × 4 = 10000 × 32 × 1000 × 4 = 1.28 GB
```

**그리고** $z_t$ 도 보존 (tanh' 계산용) → 2.56 GB. RNN cell 의 internal activation 까지 포함하면 5+ GB. A100 GPU 의 80GB 메모리에서 다른 모델 component 와 함께 쉽게 한계 도달.

### 병렬성의 직관

**Batch parallelism (GPU 친화적)**:
```
batch 0:  h_t^{(0)} ─→ h_{t+1}^{(0)}
batch 1:  h_t^{(1)} ─→ h_{t+1}^{(1)}     ← 동시 계산
batch 2:  h_t^{(2)} ─→ h_{t+1}^{(2)}
batch 3:  h_t^{(3)} ─→ h_{t+1}^{(3)}
```

이는 GPU 의 SIMT (Single Instruction Multiple Thread) 에 적합 — single matmul 로 처리.

**Sequence parallelism (RNN 에서 불가)**:
```
t=0 → t=1 → t=2 → t=3 → t=4 → ... → t=T
 │     │     │     │     │           │
 한 step 끝나야 다음 step 시작 가능 — sequential dependency
```

GPU 의 thousands of cores 가 한 step 에 사용되지 못하고 다음 step 대기 → **utilization** 이 sequence 길이에 반비례.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Forward Time Complexity

Single sample, $T$ steps, hidden $H$, input $D$, output $O$:

$$
T_{\text{forward}} = T \cdot (H^2 + HD + HO) + T \cdot O(H) = O(T H^2)
$$

(보통 $H \gg D, O$ 가정)

### 정의 4.2 — Backward Time Complexity

Forward 와 동일 복잡도 (각 forward op 마다 backward op):

$$
T_{\text{backward}} = O(T H^2)
$$

Total: $O(T H^2)$.

### 정의 4.3 — Memory Complexity

Forward activation 보존 (backward 위해):
- $h_t$: $T \times H$ floats
- $z_t$: $T \times H$ floats (또는 $h_t$ 만 보존하고 $z_t = \mathrm{atanh}(h_t)$ 재계산)

Total: $O(T H)$ per sample. Batched: $O(B T H)$.

### 정의 4.4 — Parallel vs Sequential Time

- **$T_{\text{seq}} = T \cdot t_{\text{step}}$**: Sequential time, $T$ 번의 sequential step
- **$T_{\text{par}}$**: Optimal parallel time on infinite cores

RNN 에서 $T_{\text{par}} = T \cdot t_{\text{step}}$ — sequential dependency 로 병렬화 불가.

비교 (Transformer): $T_{\text{seq}}^{\text{Trans}} = T^2 \cdot t_{\text{matmul}}$, $T_{\text{par}}^{\text{Trans}} = O(\log T)$.

### 정의 4.5 — Gradient Checkpointing (Chen 2016)

$\sqrt T$ checkpoint 만 보존, segment 마다 forward 재실행:
- Memory: $O(\sqrt T H)$
- Time: $O(T \cdot \sqrt T \cdot H^2)$ in worst case, 평균 $O(T H^2 \cdot 1.5)$ — 약 1.5x 추가 비용

---

## 🔬 정리와 결과

### 정리 4.1 — RNN BPTT 의 Total Complexity

Single sequence:
- Forward: $O(T H^2)$ time, $O(TH)$ memory
- Backward: $O(T H^2)$ time, $O(TH)$ memory (working memory + grad accumulators)

Batched ($B$ samples):
- Time: $O(B T H^2)$ — matmul 효율로 $B$ 차원 병렬
- Memory: $O(B T H)$ — 모든 sample 의 activation 별도

### 정리 4.2 — Memory-Optimal Gradient Checkpointing

Gradient checkpointing 의 optimal segment length $s$:

$$
s^* = \sqrt T \implies M = O(\sqrt T H), \; T_{\text{cost}} = O(T H^2 \log T)
$$

**증명** (sketch): Segment 가 $T/s$ 개. 각 segment 의 forward 가 $s$ steps. Backward 시 한 segment 재계산 + backward = $s + s = 2s$ ops. Total recompute = $T/s \cdot s = T$, 추가 forward total = $T$, 즉 1x 추가 비용. Memory = checkpoint $T/s$ + segment activation $s$ → minimize over $s$ gives $s = \sqrt T$. $\square$

(Refinement: Chen 2016 의 nested checkpointing 으로 $O(T \log T)$ time, $O(\log T)$ memory 가능)

### 정리 4.3 — Sequence-Internal Parallelism Bound

RNN 의 sequence 차원 GPU utilization:

$$
U_{\text{seq}}(T) \le \frac{1}{T} \cdot N_{\text{cores}}^{-1}
$$

(이상적으로 $T$ 번의 sequential step 이 모두 single core 에 의존, GPU 의 $N_{\text{cores}}$ 가 사용 안 됨)

**현실**: Batch 와 hidden 차원 matmul 로 일부 활용 ($O(BH)$ 차원 병렬), 그러나 sequence 차원 자체는 sequential.

**Transformer 와 비교**: $T \times T$ attention matrix 가 한 번에 계산 — sequence 차원도 병렬.

### 정리 4.4 — Memory vs Time vs Accuracy Trade-off

| Method | Memory | Time | Accuracy |
|--------|--------|------|----------|
| **Full BPTT** | $O(TH)$ | $O(TH^2)$ | Exact |
| **Gradient Checkpoint** $s = \sqrt T$ | $O(\sqrt T H)$ | $O(T H^2 \log T)$ | Exact |
| **TBPTT($k$)** | $O(kH)$ | $O(T H^2)$ | Biased ($O(\rho^k)$) |
| **RTRL** | $O(H^3)$ | $O(T H^4)$ | Exact (online) |
| **UORO** | $O(H^2)$ | $O(T H^2)$ | Stochastic (unbiased) |

### 정리 4.5 — Mixed Precision 의 Memory 절약

float32 → float16 으로:
- Memory $O(TH)$ → $O(TH/2)$ (2x 절약)
- Speed: 일부 GPU (V100, A100) 에서 Tensor Core 가속

**주의**: Precision 손실로 학습 불안정 — `torch.cuda.amp` 가 loss scaling 으로 보완.

---

## 💻 실험 측정

### 실험 1 — Forward Time Scaling

```python
import torch
import torch.nn as nn
import time

H, D = 256, 64
B = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

rnn = nn.LSTM(D, H, batch_first=False).to(device)

times_T = []
for T in [10, 50, 100, 500, 1000]:
    x = torch.randn(T, B, D, device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = rnn(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = rnn(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / 20
    times_T.append(elapsed)
    print(f'T={T:5d}: forward time = {elapsed*1000:.3f} ms')

# 선형 scaling 확인
ratios = [times_T[i+1] / times_T[i] for i in range(len(times_T)-1)]
print(f'Time ratios (should approach T_ratio): {ratios}')
```

### 실험 2 — Memory Scaling

```python
def measure_peak_memory(T, B, H, D):
    if device != 'cuda':
        return None
    rnn = nn.LSTM(D, H, batch_first=False).cuda()
    x = torch.randn(T, B, D, requires_grad=True, device='cuda')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    out, _ = rnn(x)
    loss = out.sum()
    loss.backward()
    
    return torch.cuda.max_memory_allocated() / 1e6  # MB

if device == 'cuda':
    for T in [100, 500, 1000, 5000]:
        m = measure_peak_memory(T, 32, 256, 64)
        print(f'T={T:5d}: peak memory = {m:.2f} MB')
```

### 실험 3 — Gradient Checkpointing Memory 절약

```python
import torch.utils.checkpoint as cp

class NormalLSTM(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.lstm = nn.LSTM(D, H, batch_first=False)
    def forward(self, x):
        return self.lstm(x)[0]

class CheckpointedLSTM(nn.Module):
    def __init__(self, D, H, segment_len=20):
        super().__init__()
        self.lstm = nn.LSTM(D, H, batch_first=False)
        self.seg = segment_len
    def forward_seg(self, x, h, c):
        out, (h, c) = self.lstm(x, (h, c))
        return out, h, c
    def forward(self, x):
        T, B, _ = x.shape
        H = self.lstm.hidden_size
        h = torch.zeros(1, B, H, device=x.device)
        c = torch.zeros(1, B, H, device=x.device)
        outs = []
        for s in range(0, T, self.seg):
            x_seg = x[s:s+self.seg]
            out, h, c = cp.checkpoint(self.forward_seg, x_seg, h, c, use_reentrant=False)
            outs.append(out)
        return torch.cat(outs, dim=0)

if device == 'cuda':
    T, B, H, D = 2000, 16, 256, 64
    
    # Normal
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m1 = NormalLSTM(D, H).cuda()
    x = torch.randn(T, B, D, device='cuda')
    out = m1(x); out.sum().backward()
    mem_normal = torch.cuda.max_memory_allocated() / 1e6
    
    # Checkpointed
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m2 = CheckpointedLSTM(D, H, segment_len=int(T**0.5)).cuda()
    x = torch.randn(T, B, D, device='cuda')
    out = m2(x); out.sum().backward()
    mem_ckpt = torch.cuda.max_memory_allocated() / 1e6
    
    print(f'Normal LSTM:        {mem_normal:.2f} MB')
    print(f'Checkpointed (s=√T): {mem_ckpt:.2f} MB')
    print(f'Saving: {mem_normal/mem_ckpt:.2f}x')
```

### 실험 4 — Sequence vs Batch Parallelism

```python
def time_op(op, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        op()
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        op()
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start) / n_iter

H = 256
T = 100

# Batch=1, T=100: sequential dominant
def case_seq():
    x = torch.randn(T, 1, D, device=device)
    return rnn(x)[0]

# Batch=32, T=100: batch parallelism 활용
def case_batch():
    x = torch.randn(T, 32, D, device=device)
    return rnn(x)[0]

t_seq = time_op(case_seq) * 1000
t_batch = time_op(case_batch) * 1000

print(f'B=1,  T=100: {t_seq:.3f} ms')
print(f'B=32, T=100: {t_batch:.3f} ms')
print(f'Throughput: B=1 → {1/t_seq*1000:.0f} samples/sec, B=32 → {32/t_batch*1000:.0f} samples/sec')
# Batch 가 효율적 — sequence 는 본질적으로 sequential
```

### 실험 5 — RNN vs Transformer 시간 비교

```python
class TransformerEncoder(nn.Module):
    def __init__(self, D, H, n_layers=2, n_head=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=H, nhead=n_head, batch_first=False)
        self.proj = nn.Linear(D, H)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
    def forward(self, x):
        return self.encoder(self.proj(x))

# 같은 hidden size 의 RNN (LSTM) vs Transformer
rnn_model = nn.LSTM(D, H, batch_first=False).to(device)
trf_model = TransformerEncoder(D, H).to(device)

print('Inference time comparison (T=100, B=32):')
for T in [50, 100, 500, 1000]:
    x = torch.randn(T, 32, D, device=device)
    t_rnn = time_op(lambda: rnn_model(x)[0]) * 1000
    t_trf = time_op(lambda: trf_model(x)) * 1000
    print(f'T={T:4d}: RNN {t_rnn:6.2f} ms  vs  Transformer {t_trf:6.2f} ms')
# 짧은 T: RNN 우위 (선형 vs T^2). 긴 T: Transformer 의 병렬성 우위 (선형 vs sequential)
```

---

## 🔗 실전 활용

### 1. Long-document modeling

$T = 10000$ token document — full BPTT 메모리 한계, gradient checkpointing 또는 TBPTT 필수.

### 2. Mixed precision training

A100 GPU 의 BF16 / FP16 으로 메모리 2x 절약 + 속도 향상. PyTorch `torch.cuda.amp` autocast.

### 3. Distributed training

- **Data parallelism**: Multiple GPU 가 다른 batch — sync gradient
- **Sequence parallelism**: One sequence 를 여러 GPU 로 split — Megatron-LM
- **Pipeline parallelism**: 다른 layer 를 다른 GPU 로

### 4. Inference optimization

- **KV-cache** (Transformer) — past token activation 보존, sequential generation 효율
- **Stateful RNN inference** — hidden state 만 유지, 무한 길이 generation 가능

### 5. Model compilation

- `torch.jit.script` — RNN forward 를 정적 graph 로 compile
- `torch.compile` (PyTorch 2.0) — graph optimization, kernel fusion

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Float precision 충분 | Long $T$ 에서 numerical error — float32 표준 |
| GPU memory 단일 | Multi-GPU sharding — ZeRO, FSDP |
| Synchronous update | Async training (Hogwild!) 가능 |
| Per-sample independent | Batched 처리 가능 |
| Activation 보존 | Checkpointing 으로 trade |

---

## 📌 핵심 정리

$$\boxed{\text{Forward + Backward time: } O(T H^2)}$$

$$\boxed{\text{Memory: } O(TH) \text{ — checkpointing 으로 } O(\sqrt T H)}$$

$$\boxed{\text{Sequence parallelism } \times \text{ — 본질적으로 sequential, } T_{\text{par}} = T \cdot t_{\text{step}}}$$

| 차원 | RNN | Transformer | Mamba |
|------|-----|-------------|-------|
| **Time** | $O(TH^2)$ | $O(T^2 H)$ | $O(TH)$ |
| **Memory (train)** | $O(TH)$ | $O(T^2)$ | $O(TH)$ |
| **Inference state** | $O(H)$ | $O(TH)$ KV-cache | $O(H)$ SSM state |
| **Sequence parallel** | × | ✓ | parallel scan ✓ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T = 1000$, $H = 1024$, $B = 64$, float32 인 LSTM 학습의 forward activation memory 를 계산하라. A100 GPU (80GB) 에서 가능한가?

<details>
<summary>해설</summary>

**LSTM activation per step**: $h_t, c_t, f_t, i_t, o_t, \tilde c_t$ — 6개 의 $H$-차원 vector.

**Total activation memory**:
$$
T \times B \times 6H \times 4\text{ bytes} = 1000 \times 64 \times 6 \times 1024 \times 4 = 1.57 \text{ GB}
$$

**Plus weight memory**:
- 4 gate weights $W_{ih}, W_{hh}$: $4 \times 2 \times H \times (H + D) \approx 4 \times 2 \times 1024 \times 1088 = 35 \text{ MB}$
- Adam optimizer state: weight × 3 ≈ 100 MB

**Plus input/output**:
- Input $x_t$: $T B D$ ≈ 26 MB (D=100)
- Output gradient: 같은 크기

**Total**: 약 **1.7 GB** — A100 80 GB 에서 충분. 그러나 실전에서는:
- Multiple LSTM layers
- Embedding layer (vocab 큼) 
- Other model components

이를 모두 합치면 80 GB 한계 도달 가능 ⇒ gradient checkpointing 또는 mixed precision 필요. $\square$

</details>

**문제 2** (심화): Gradient checkpointing 의 optimal segment $s^* = \sqrt T$ 를 라그랑지 multiplier 로 derive 하라.

<details>
<summary>해설</summary>

**Memory 비용**: 
- Checkpoint 수: $T/s$, 각 checkpoint: $H$ floats → checkpoint memory = $TH/s$
- Segment 내 activation: $s$ steps × $H$ → segment memory = $sH$
- Total: $M(s) = TH/s + sH$

**Time 비용** (대략):
- Forward: $T$ steps (한 번)
- Backward 시 재계산: 각 segment 한 번 → $T$ steps 추가
- Total time = $2T$ steps × $O(H^2)$ = $O(TH^2)$ — segment 길이와 무관

**Optimization**: Memory minimize:

$$
M(s) = TH/s + sH
$$

$$
\frac{dM}{ds} = -TH/s^2 + H = 0 \implies s^2 = T \implies s^* = \sqrt T
$$

**Optimal memory**:

$$
M(\sqrt T) = TH/\sqrt T + \sqrt T H = 2\sqrt T H = O(\sqrt T H)
$$

**원래 memory**: $O(TH)$ → 절약율 $\sqrt T$.

**라그랑지 일반화**: Time constraint $T_{\text{max}}$ 추가 시 $s$ 와 nested checkpointing 조합. Chen 2016 의 nested 으로 memory $O(\log T)$, time $O(T \log T)$ 가능. $\square$

</details>

**문제 3** (논문 비평): RNN 의 sequence-internal parallelism 한계 (정리 4.3) 가 "RNN 은 죽었다" 는 결론으로 이어졌지만, Mamba 같은 SSM 이 다시 부활시켰다. 어떤 수학적 trick 으로 Mamba 가 sequential 을 parallel 하게 만들었는가?

<details>
<summary>해설</summary>

**RNN 의 sequential 본질**:
- $h_t = f(h_{t-1}, x_t)$ — non-linear $f$ 시 단순 closed-form 없음
- 각 step 의 결과가 다음 step 의 input

**Linear RNN (SSM) 의 trick**:
- $h_t = A h_{t-1} + B x_t$ — **linear** in $h, x$
- 풀어쓰면:
$$
h_T = A^T h_0 + \sum_{k=1}^{T} A^{T-k} B x_k
$$
- 이는 **linear scan** 의 결과 — parallel scan 알고리즘으로 $O(\log T)$ depth 가능 (Blelloch 1990)

**Parallel Scan**:
- 각 step 의 $(A, Bx_t)$ pair 를 leaf 로 binary tree 구성
- Up-sweep: 두 노드 합쳐서 $(A_2 A_1, A_2 B_1 x_1 + B_2 x_2)$ — associative
- Down-sweep: parent state 를 children 에 전파
- Total: $O(T)$ work, $O(\log T)$ depth

**Mamba (Gu & Dao 2023) 의 추가 trick**:
- Selective SSM: $A, B, C$ 가 $x_t$ 에 의존 (input-dependent)
- 그러나 여전히 linear in $h$, parallel scan 가능
- **Hardware-aware**: GPU memory hierarchy 에 맞춘 fused scan kernel — Flash-Attention 의 SSM 버전

**Linear Attention 도 같은 원리**:
- $S_t = S_{t-1} + \phi(k_t) v_t^\top$ — linear in $S$
- Cumulative sum, parallel scan 가능

**일반화**: 
- 모든 *linear* recurrence 는 parallel scan 가능
- Non-linear gating (LSTM forget gate) 는 trick 으로 부분 parallel — RWKV, Mega
- **표현력 vs 병렬성** trade-off — 완전 일반 RNN 은 sequential 필수, 제한된 형태는 parallel

**결론**: RNN 의 sequential 한계는 *non-linear* 에서 옴. Linear SSM 은 본질적으로 병렬 가능 — Mamba 의 선택적 linear recurrence 가 이 통찰의 culmination. (Ch7-04 에서 자세히) $\square$

</details>

---

<div align="center">

[◀ 이전](./03-truncated-bptt.md) | [📚 README](../README.md) | [다음 ▶](./05-rtrl.md)

</div>
