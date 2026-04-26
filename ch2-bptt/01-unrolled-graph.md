# 01. Unrolled Computational Graph

## 🎯 핵심 질문

- RNN 의 시간축 unrolling 이란 무엇이며, 왜 이를 통해 BPTT 가 standard backpropagation 으로 환원되는가?
- Time step 별 hidden state $h_1, h_2, \ldots, h_T$ 가 어떻게 acyclic DAG 의 노드가 되는가?
- **Shared weight** $W_{hh}, W_{xh}$ 가 unrolled graph 에서 어떻게 표현되며, gradient 합산이 왜 자연스럽게 따라오는가?
- Static unrolling (Theano, TensorFlow 1.x) vs dynamic unrolling (PyTorch) 의 차이는?
- Unrolled graph 의 **memory-time trade-off** 와 gradient checkpointing (Chen 2016) 은 어떤 동기인가?

---

## 🔍 왜 unrolled graph 가 BPTT 의 출발점인가

RNN 은 본질적으로 *cyclic* 한 구조 — $h_t = f(h_{t-1}, x_t)$ 가 자기참조. 그러나 학습은 **acyclic** 미분 가능 그래프 (DAG) 위의 chain rule 을 요구합니다. **Unrolling** 은 이 cyclic 구조를 시간축으로 펼쳐 DAG 로 만드는 결정적 변환:

1. **Gradient 계산의 가능화** — Chain rule 이 DAG 에서 잘 정의됨, BPTT 가 standard backprop 으로 환원
2. **Shared weight 의 처리** — Weight 가 $T$ 번 반복되는 동일 변수, gradient 합산이 자동
3. **Computational complexity 의 명시화** — Forward $O(TH^2)$, Memory $O(TH)$ 가 unrolled graph 의 노드 수
4. **PyTorch dynamic graph 의 자연스러움** — RNN forward 가 매번 새 graph 를 build 하는 것이 unrolling 의 동적 구현

이 문서는 unrolled graph 의 정확한 구조와 BPTT 가 그 위에서 어떻게 실행되는지 시각화합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [Ch1-04 RNN 의 정의](../ch1-sequence-basics/04-rnn-definition.md)
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Backpropagation, computational graph, autograd
- 그래프 이론: DAG, topological order, dependency
- (선택) 자동미분: Forward-mode vs reverse-mode, chain rule

---

## 📖 직관적 이해

### Cyclic RNN vs Acyclic Unrolled Graph

```
Cyclic (RNN cell):                  Unrolled (DAG):

   ┌──────────┐                  x₁→[cell]→h₁→[cell]→h₂→[cell]→h₃ → ...
   │          │                          │       │       │
   x → [cell] →h                         y₁      y₂      y₃
       └──┘
       self-loop                  (모든 cell 이 같은 W_hh, W_xh 공유)
```

Cyclic graph 에서는 chain rule 적용 시 "어디서 시작할지" 가 모호. Unrolling 후에는 각 $h_t$ 가 distinct 노드, $h_t = f(h_{t-1}, x_t)$ 가 명시적 edge.

### Shared Weight 의 시각화

Unrolled graph 의 모든 cell 이 동일 $W_{hh}, W_{xh}$ 를 사용:

```
              W_hh, W_xh (단 하나의 변수)
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   [cell_1]   [cell_2]   [cell_3]   ...
       │           │           │
       ▼           ▼           ▼
       h₁          h₂          h₃
```

Backward 시:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial W_{hh}^{(t)}}
$$

여기서 $W_{hh}^{(t)}$ 는 step $t$ 에서의 사용 인스턴스. 합산은 PyTorch autograd 에서 자동 (variable 의 동일성 인식).

### Static vs Dynamic Unrolling

**Static (TensorFlow 1.x, Theano)**:
- 컴파일 시 $T$ 가 고정, graph 가 미리 build
- 같은 $T$ 의 batch 만 처리 가능 (또는 maximum $T$ + masking)
- 최적화 (XLA, fusion) 용이

**Dynamic (PyTorch, TF 2.x eager)**:
- 매 iteration 마다 forward 시 graph 가 build (`for t in range(T): ...`)
- 가변 $T$, conditional, while loop 자연스러움
- 디버깅 용이, 그러나 일부 최적화 불가

PyTorch 의 dynamic graph 가 RNN 에 자연스러운 이유: sequence length 가 batch 마다 다를 수 있고, control flow (early stopping, attention masking) 가 표현 쉽움.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Unrolled Computational Graph

RNN forward $h_t = f_\theta(h_{t-1}, x_t)$, $t = 1, \ldots, T$ 의 unrolled graph $G = (V, E)$:

- **Nodes** $V$: $\{x_1, \ldots, x_T, h_0, h_1, \ldots, h_T, y_1, \ldots, y_T, L\}$
- **Edges** $E$: $\{x_t \to h_t, h_{t-1} \to h_t, h_t \to y_t, y_t \to L\}$ for all $t$
- **Acyclic**: topological order $h_0, x_1, h_1, x_2, \ldots, h_T, L$ 존재

### 정의 1.2 — Shared Weight 와 Gradient 합산

Weight $W_{hh}$ 가 $T$ time step 에서 동일 변수로 사용. Backward 시:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \left.\frac{\partial L}{\partial W_{hh}}\right|_{\text{step } t}
$$

각 항은 step $t$ 에서의 partial contribution.

### 정의 1.3 — Topological Order

Forward 는 노드 의존성 순서대로:

$$
h_0 \to (x_1, h_0) \to z_1 \to h_1 \to (x_2, h_1) \to z_2 \to h_2 \to \ldots
$$

Backward 는 역순 — $L$ 부터 시작.

### 정의 1.4 — Memory and Time Complexity

- **Forward time**: $O(T \cdot (H^2 + HD))$
- **Forward memory**: $O(T \cdot H)$ — backward 를 위해 모든 $h_t, z_t$ 보존 필요
- **Backward time**: $O(T \cdot (H^2 + HD))$ — forward 와 같은 복잡도
- **Backward memory**: $O(T \cdot H)$ — gradient 누적

### 정의 1.5 — Gradient Checkpointing

$O(T)$ memory 를 $O(\sqrt T)$ 로 절약 (Chen 2016):
- $\sqrt T$ checkpoint 만 보존, backward 시 segment 마다 forward 재실행
- Time cost: $O(T \log T) \to O(T \sqrt T)$

---

## 🔬 정리와 결과

### 정리 1.1 — Unrolling 후 Standard Backprop 적용 가능

Unrolled graph $G$ 가 DAG 이므로 reverse-mode AD 가 잘 정의됨. 즉 BPTT $\equiv$ standard backprop on unrolled graph.

**증명**: Reverse-mode AD 의 정의 — 각 노드 $v$ 의 gradient $\bar v = \partial L / \partial v$ 를 topological order 의 역순으로 계산. Unrolled RNN 에서 각 $h_t$ 가 distinct 노드이므로 chain rule:

$$
\bar h_t = \bar h_{t+1} \frac{\partial h_{t+1}}{\partial h_t} + \bar y_t \frac{\partial y_t}{\partial h_t}
$$

이는 잘 정의된 chain rule, 즉 standard backprop. $\square$

### 정리 1.2 — Shared Weight Gradient 의 자동 합산

PyTorch autograd 는 동일 variable 에 대한 multiple usage 를 자동으로 합산:

```python
W = torch.randn(..., requires_grad=True)
y1 = W @ x1   # usage 1
y2 = W @ x2   # usage 2
loss = y1.sum() + y2.sum()
loss.backward()
# W.grad = (∂loss/∂W from y1) + (∂loss/∂W from y2)  — 자동 합산
```

**증명** (sketch): autograd 는 backward 시 각 노드의 gradient 를 leaf variable 에 accumulate. Multi-edge 가 있으면 다중 contribution 합산. $\square$

### 정리 1.3 — Memory Lower Bound

BPTT 가 정확한 gradient 를 계산하려면 모든 forward activation $\{h_t, z_t\}$ 을 보존해야 함.

**증명**: $\partial h_t / \partial h_{t-1} = W_{hh}^\top \mathrm{diag}(\sigma'(z_t))$ — $z_t$ (또는 $h_t$ 자체) 가 backward 에 필요. 보존 안 하면 재계산 (gradient checkpointing) 또는 gradient approximation. $\square$

### 정리 1.4 — Truncated BPTT 의 Memory 절약

$k$-truncation 시 마지막 $k$ time step 만 unroll, memory $O(kH)$.

(Ch2-03 에서 자세히)

### 정리 1.5 — Gradient Checkpointing 의 Pareto-Optimal Trade-off

Optimal segment 길이 $\sqrt T$ 시:
- Memory: $O(\sqrt T \cdot H)$
- Time: $O(T \cdot \sqrt T \cdot H^2)$ (forward 가 평균 $\sqrt T$ 번 재실행)

(Chen 2016 — Sublinear Memory Cost)

---

## 💻 PyTorch 구현 검증

### 실험 1 — Manual Unrolling vs nn.RNN

```python
import torch
import torch.nn as nn

D, H, T, B = 4, 8, 6, 2
torch.manual_seed(0)

# Manual unrolling
class ManualRNN(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.cell = nn.RNNCell(D, H, nonlinearity='tanh')
        self.H = H
    def forward(self, x_seq):
        T, B, _ = x_seq.shape
        h = torch.zeros(B, self.H)
        hs = []
        for t in range(T):
            h = self.cell(x_seq[t], h)   # Cyclic 호출 — unrolling
            hs.append(h)
        return torch.stack(hs)            # (T, B, H)

manual = ManualRNN(D, H)
x = torch.randn(T, B, D)
hs_manual = manual(x)
print(f'Manual unrolled hs: {hs_manual.shape}')   # (T, B, H)
```

### 실험 2 — Computational Graph 시각화

```python
# torchviz 로 unrolled graph 시각화 (선택)
try:
    from torchviz import make_dot
    loss = hs_manual.sum()
    dot = make_dot(loss, params=dict(manual.named_parameters()))
    dot.render('unrolled_rnn', format='png', cleanup=True)
    print('Unrolled graph saved as unrolled_rnn.png')
except ImportError:
    print('Install torchviz: pip install torchviz')
```

### 실험 3 — Shared Weight Gradient 자동 합산

```python
# Toy: 같은 W 가 두 번 사용
W = torch.randn(3, 3, requires_grad=True)
x1 = torch.randn(3)
x2 = torch.randn(3)

y1 = W @ x1
y2 = W @ x2
loss = (y1 + y2).sum()
loss.backward()

# Manual 검증
expected_grad = (
    torch.outer(torch.ones(3), x1) + torch.outer(torch.ones(3), x2)
)
print(f'Auto grad     : {W.grad}')
print(f'Manual sum    : {expected_grad}')
print(f'Match? {torch.allclose(W.grad, expected_grad)}')
```

### 실험 4 — Memory 측정 (Forward 보존)

```python
import torch

def measure_memory(T):
    rnn = nn.RNN(D, H, batch_first=False)
    x = torch.randn(T, 1, D, requires_grad=True)
    
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        rnn = rnn.cuda()
        x = x.cuda()
        torch.cuda.reset_peak_memory_stats()
        hs, _ = rnn(x)
        loss = hs.sum()
        loss.backward()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        return peak_mb
    return None

if torch.cuda.is_available():
    for T in [10, 100, 1000]:
        m = measure_memory(T)
        print(f'T={T}: peak memory = {m:.2f} MB  (grows linearly)')
else:
    print('CUDA not available — skip memory measurement')
```

### 실험 5 — Gradient Checkpointing

```python
import torch.utils.checkpoint as cp

class CheckpointRNN(nn.Module):
    def __init__(self, D, H, segment_len=10):
        super().__init__()
        self.cell = nn.RNNCell(D, H, nonlinearity='tanh')
        self.H = H
        self.segment_len = segment_len
    
    def forward_segment(self, h, x_seg):
        for t in range(x_seg.size(0)):
            h = self.cell(x_seg[t], h)
        return h, h.unsqueeze(0)   # h_T, hs_in_segment (단순화)
    
    def forward(self, x_seq):
        T, B, _ = x_seq.shape
        h = torch.zeros(B, self.H)
        all_hs = []
        for s in range(0, T, self.segment_len):
            x_seg = x_seq[s:s+self.segment_len]
            h, hs_seg = cp.checkpoint(self.forward_segment, h, x_seg, use_reentrant=False)
            all_hs.append(hs_seg)
        return torch.cat(all_hs, dim=0)

# Long sequence 학습 시 메모리 절약 (느려지지만)
ckpt_rnn = CheckpointRNN(D, H, segment_len=20)
x_long = torch.randn(100, 1, D)
hs_ckpt = ckpt_rnn(x_long)
print(f'Checkpoint RNN output: {hs_ckpt.shape}')
```

---

## 🔗 실전 활용

### 1. PyTorch 의 dynamic graph

PyTorch 의 `for t in range(T)` 패턴이 자연스러운 unrolling. 매 iteration 마다 graph 가 재구성되어 가변 $T$ 처리 자동.

### 2. JIT compilation (TorchScript)

`torch.jit.script` 으로 RNN forward 를 정적 graph 로 compile — 같은 $T$ pattern 의 batch 에서 속도 향상.

### 3. Truncated BPTT 의 implementation

`detach()` 로 gradient flow 차단 — 마지막 $k$ step 만 backward (Ch2-03).

### 4. Memory-bound 모델 (긴 sequence, 큰 hidden)

Gradient checkpointing (`torch.utils.checkpoint`) 또는 mixed-precision training (`amp`) 으로 메모리 절약.

### 5. Distributed training

Sequence dimension 으로 split (sequence parallel) 가능, layer dimension 으로 split (model parallel).

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Forward 가 backward 전에 모든 activation 보존 | $O(TH)$ memory — checkpointing, truncated BPTT |
| DAG (cyclic 아님) | RNN cell 자체는 cyclic, 매번 unrolling 필요 |
| Single thread per sequence | Sequence-internal parallelism 불가 → Transformer |
| Static $T$ (TF 1.x) | Dynamic $T$ → PyTorch, TF 2.x eager |
| Float precision 충분 | Long $T$ 에서 numerical instability — float32 vs float16 |

---

## 📌 핵심 정리

$$\boxed{\text{Unrolling: } h_t = f(h_{t-1}, x_t) \text{ becomes DAG with nodes } h_0, h_1, \ldots, h_T}$$

$$\boxed{\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial W_{hh}^{(t)}} \quad \text{— shared weight 합산}}$$

$$\boxed{\text{Memory } O(TH), \quad \text{Forward+Backward } O(TH^2)}$$

| 측면 | Static (TF 1.x) | Dynamic (PyTorch) |
|------|----------------|-------------------|
| **Graph construction** | Compile-time | Runtime |
| **가변 $T$** | Padding + masking | 자연스러움 |
| **Optimization** | XLA fusion 가능 | JIT 으로 부분 |
| **Debugging** | 어려움 | 쉬움 |
| **표준 사용** | Production | Research |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T = 4$ 인 RNN 의 unrolled graph 를 그리고, $\partial L / \partial W_{hh}$ 가 어떤 합으로 분해되는지 시각화하라.

<details>
<summary>해설</summary>

**Unrolled graph**:
```
x₁→[W]→h₁→[W]→h₂→[W]→h₃→[W]→h₄→y₄→L
        │       │       │       │
        y₁      y₂      y₃    (L_t = CE(y_t, target_t))
        │       │       │
        L₁      L₂      L₃
```
(여기서 $[W]$ 는 $W_{hh}$ 와 $W_{xh}$ 를 모두 포함)

**Gradient 분해**:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{4} \frac{\partial L_t}{\partial W_{hh}}
$$

각 항:

$$
\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^{t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \cdot \frac{\partial h_k}{\partial W_{hh}}
$$

**총합**: $4 \cdot \text{avg}(t) \approx 4 \cdot 2.5 = 10$ 개 contribution. $T$ 가 클수록 더 많은 path. $\square$

</details>

**문제 2** (심화): Gradient checkpointing 의 segment length $s$ 에 따른 time-memory trade-off 를 정량화하라. $s = 1$ (모든 step 재계산), $s = T$ (재계산 없음), $s = \sqrt T$ 의 비교.

<details>
<summary>해설</summary>

**메모리**:
- $s$-checkpointing: $T/s$ checkpoint + segment 내 $s$ activations = $O(T/s + s)$
- 최소: $s = \sqrt T$ → $O(\sqrt T)$
- $s = T$ (no checkpointing): $O(T)$
- $s = 1$ (every step): $O(T)$ — 매 step checkpoint 가 모든 activation

**Time**:
- Forward (학습): segment 마다 한 번 — $O(T)$
- Backward: segment 의 forward 를 재실행 후 backward — $O(T \cdot s)$ in worst case... 정확히는:
  - $s = T$: $O(T)$ (no recomputation)
  - $s = \sqrt T$: $O(T)$ (각 segment 한 번 재계산, 총 $T$ ops)
  - $s = 1$: $O(T)$ (단일 step 재계산만)

**Pareto-optimal $s = \sqrt T$**:
- Memory $O(\sqrt T)$, Time $O(T)$ (wallclock 약 1.3배)
- 일반 $T$: Memory $O(T)$, Time $O(T)$
- 비율: 메모리 $\sqrt T$ 배 절약, 시간 30% 증가

**실용**: $T = 10000$ 시 $s = 100$ → 메모리 100x 절약 ✓ $\square$

</details>

**문제 3** (논문 비평): PyTorch 의 dynamic graph 가 RNN 에 좋지만, Transformer 에는 (적어도 inference 에서는) static graph 가 더 적합한 이유는? 두 architecture 의 graph 특성 차이를 논하라.

<details>
<summary>해설</summary>

**RNN 의 dynamic graph 적합성**:
- 가변 $T$: 매 batch 마다 sequence length 다름
- Conditional control: early stopping, beam search
- Truncated BPTT: 동적 truncation length
- 자연스러운 cell-wise iteration

**Transformer 의 static graph 적합성**:
- **고정 attention pattern**: $T \times T$ matrix (또는 causal mask)
- **순수 matrix operations**: softmax, matmul — XLA / TensorRT 최적화 용이
- **Padding 으로 batching**: 같은 길이로 padding 하면 정적 shape
- **Inference 시 fixed length**: serving 에서 일정 max_len 으로 compile

**PyTorch 의 흐름**:
- Research: dynamic eager mode (PyTorch)
- Production: TorchScript, ONNX, TensorRT 로 static graph 변환
- 최근: `torch.compile` 으로 dynamic + static 의 best-of-both

**결론**: RNN 은 본질적으로 *control flow*-heavy — dynamic 이 자연스러움. Transformer 는 *batched matrix algebra*-heavy — static 이 자연스러움. Architecture 의 computational pattern 이 framework choice 를 결정. $\square$

</details>

---

<div align="center">

[◀ 이전](../ch1-sequence-basics/04-rnn-definition.md) | [📚 README](../README.md) | [다음 ▶](./02-bptt-derivation.md)

</div>
