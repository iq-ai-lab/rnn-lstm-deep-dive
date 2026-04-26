# 03. Neural Turing Machine 과 Memory Network

## 🎯 핵심 질문

- Graves 2014 의 Neural Turing Machine (NTM) 이 어떻게 **external memory** $M \in \mathbb R^{N \times d}$ 와 differentiable read/write head 로 RNN 의 memory capacity 한계를 우회하는가?
- **Content-based addressing** $w_t^{(c)} = \mathrm{softmax}(\beta \cos(k_t, M_t))$ 와 **location-based addressing** (shift kernel + sharpening) 의 결합 메커니즘은?
- NTM 이 어떻게 differentiable programming 의 출발점이 되었는가? Synthetic copy / repeat-copy / associative recall 에서 LSTM 을 능가
- **DNC** (Graves 2016) 의 Memory Network 일반화 — Differentiable Neural Computer
- Memory Network (Weston 2014) 와 NTM 의 차이 — read-only vs read/write

---

## 🔍 왜 External Memory 가 RNN 의 자연스러운 확장인가

LSTM 의 hidden $H$ 차원이 모든 history 를 압축. 한계:
1. **Capacity bottleneck** — Information loss in long sequence
2. **Random access 불가** — Sequential information retrieval
3. **Editable memory 부재** — 학습 중 specific position 수정 불가

NTM 의 해법:
1. **External memory** $N \times d$ — Capacity $N \cdot d$ floats (예: $128 \times 32 = 4096$)
2. **Differentiable addressing** — Content + location based, soft retrieval
3. **Read/write heads** — Memory 의 dynamic update

이는 von Neumann architecture (program + memory) 의 신경망 instantiation. **Differentiable programming** 의 시작점.

이 문서는 NTM 의 정확한 메커니즘과 synthetic task 에서의 우월성을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-stacked-rnn.md](./02-stacked-rnn.md) — Multi-layer architecture
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Cosine similarity, softmax
- (선택) Computer architecture: Memory hierarchy, addressing modes

---

## 📖 직관적 이해

### NTM 의 구조

```
   ┌──────────────────────────────────────────┐
   │              Memory M                     │
   │   row 1: [...]                            │
   │   row 2: [...]                            │
   │   ...                                     │
   │   row N: [...]                            │
   └──┬──────────────────────────┬─────────────┘
      │                          │
      │ Read head                │ Write head
      │                          │
      │  w_t^read              w_t^write
      │  (attention            (attention
      │   over rows)            over rows)
      │                          │
      ▼                          ▼
   ┌────┐                     ┌────┐
   │ r_t│                     │e_t,│
   └────┘                     │a_t │
      │                       └────┘
      ▼                          │
   ┌──────────┐                  │
   │ Controller│ ← (LSTM)        │
   │ (RNN)     │ ─────────────── ┘
   └──────────┘
      │
      ▼
   output y_t, key k_t, etc.
```

### Content-based Addressing

"이 query 와 비슷한 내용의 row 가 무엇?":

$$
w_i^{(c)} = \mathrm{softmax}_i(\beta \cdot \cos(k, M_i))
$$

- $k$: query vector (controller 출력)
- $M_i$: $i$-th row of memory
- $\beta$: sharpening parameter
- $\cos$: cosine similarity

### Location-based Addressing

"이전 attention 위치에서 shift":

$$
w_t = \text{shift}(w_{t-1}, s_t)
$$

여기서 $s_t \in \Delta^k$ shift distribution (예: -1, 0, +1 위치에 weight).

### Soft Read/Write

**Read**:
$$
r_t = \sum_i w_i M_i = w^\top M
$$

**Write**:
$$
M_i \leftarrow (1 - w_i e_i) M_i + w_i a_i
$$

$e_i$: erase vector, $a_i$: add vector.

### Differentiability

모든 operation (softmax, cosine, shift) 이 미분 가능 → end-to-end 학습.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Neural Turing Machine

Memory $M_t \in \mathbb R^{N \times d}$ ($N$ rows, $d$-dim), controller (LSTM):
$$
h_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1}, r_{t-1})
$$

(Previous read $r_{t-1}$ 도 input).

Read/write 가 differentiable address:

$$
w_t \in \Delta^{N-1} \quad \text{(probability simplex)}
$$

### 정의 3.2 — Content-based Addressing

$$
w_i^{(c)} = \frac{\exp(\beta \cdot K(k, M_i))}{\sum_j \exp(\beta \cdot K(k, M_j))}
$$

$K(\cdot, \cdot)$: cosine similarity. $\beta \ge 0$: sharpening.

### 정의 3.3 — Interpolation

Previous and content-based:

$$
w_t^{(g)} = g_t \cdot w_t^{(c)} + (1 - g_t) \cdot w_{t-1}
$$

$g_t \in [0, 1]$: interpolation gate.

### 정의 3.4 — Convolutional Shift

$$
\tilde w_i = \sum_j w_j^{(g)} \cdot s_t(i - j \mod N)
$$

$s_t$: shift distribution (e.g., over $\{-1, 0, +1\}$).

### 정의 3.5 — Sharpening

$$
w_i = \frac{\tilde w_i^{\gamma_t}}{\sum_j \tilde w_j^{\gamma_t}}
$$

$\gamma_t \ge 1$: sharpening parameter.

### 정의 3.6 — Read

$$
r_t = \sum_i w_i^{(\text{read})} \cdot M_{t, i}
$$

### 정의 3.7 — Write

$$
M_{t,i} = M_{t-1, i} \odot (1 - w_i^{(\text{write})} e_t) + w_i^{(\text{write})} a_t
$$

$e_t \in [0, 1]^d$: erase, $a_t \in \mathbb R^d$: add.

### 정의 3.8 — Differentiable Neural Computer (DNC)

NTM 의 일반화 (Graves 2016):
- Multiple read/write heads
- Memory linkage matrix (sequential ordering)
- Memory usage tracking (free cells)

---

## 🔬 정리와 결과

### 정리 3.1 — NTM 의 Memory Capacity

$N \times d$ memory: $N \cdot d$ floats.

**비교 LSTM**: hidden $H$ → $H$ floats (or $H + H$ for cell + hidden).

**NTM 전형 setting**: $N = 128, d = 20$ → 2560 floats. 비슷한 capacity LSTM: $H = 2560$ — but LSTM 은 single state, NTM 은 random access.

### 정리 3.2 — Random Access 의 표현력

NTM 은 임의 row 를 직접 access (content-based) — RNN 의 sequential access 보다 강력.

**Computational implication**: Algorithms requiring random memory access (sorting, copying, addressing) 가 자연스러움 — Turing-complete computation 의 closer instantiation.

### 정리 3.3 — Differentiable Programming Paradigm

NTM 이 **differentiable computation graph** with **memory operations** — programs that learn:

- Copy: read input, write to memory, read at later step
- Sort: comparison + write order
- Associative recall: key-value retrieval

Synthetic tasks 에서 LSTM 우월.

### 정리 3.4 — Memory Network (Weston 2014) 와의 차이

Memory Network: read-only memory (key-value pairs), single attention.
NTM: read + write, multiple heads, location-based addressing.

**Memory Network 의 강점**: Question Answering 에서 효과적 (Facebook bAbI).

### 정리 3.5 — Synthetic Task 결과

Graves 2014 에서:
- **Copy task**: LSTM 30 token 까지, NTM 100+
- **Repeat-copy**: NTM 가 LSTM 보다 generalize (학습한 length 이상)
- **Associative recall**: NTM 명확히 우월

---

## 💻 PyTorch 구현 검증

### 실험 1 — NTM Read/Write 단순 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTMReadHead(nn.Module):
    def __init__(self, M_dim, hidden_dim):
        super().__init__()
        self.k_proj = nn.Linear(hidden_dim, M_dim)   # key
        self.beta_proj = nn.Linear(hidden_dim, 1)     # sharpening
    
    def address(self, controller_out, M):
        """Content-based addressing"""
        k = self.k_proj(controller_out)               # (B, M_dim)
        beta = F.softplus(self.beta_proj(controller_out))   # (B, 1)
        
        # Cosine similarity: k vs each row of M
        k_norm = F.normalize(k, dim=-1)               # (B, M_dim)
        M_norm = F.normalize(M, dim=-1)               # (N, M_dim)
        sim = k_norm @ M_norm.T                        # (B, N)
        
        # Softmax with beta
        w = F.softmax(beta * sim, dim=-1)              # (B, N)
        return w
    
    def read(self, w, M):
        """r = w^T M"""
        return w @ M    # (B, M_dim)

# Test
B, N, M_dim, H = 2, 16, 8, 32
torch.manual_seed(0)

M = torch.randn(N, M_dim)
controller_out = torch.randn(B, H)

read_head = NTMReadHead(M_dim, H)
w = read_head.address(controller_out, M)
r = read_head.read(w, M)
print(f'Address weights: {w.shape}, sum: {w.sum(-1)}')   # Each row sums to 1
print(f'Read result:    {r.shape}')
```

### 실험 2 — Write Head + Memory Update

```python
class NTMWriteHead(nn.Module):
    def __init__(self, M_dim, hidden_dim):
        super().__init__()
        self.k_proj = nn.Linear(hidden_dim, M_dim)
        self.beta_proj = nn.Linear(hidden_dim, 1)
        self.e_proj = nn.Linear(hidden_dim, M_dim)   # Erase
        self.a_proj = nn.Linear(hidden_dim, M_dim)   # Add
    
    def address(self, c_out, M):
        k = self.k_proj(c_out)
        beta = F.softplus(self.beta_proj(c_out))
        sim = F.cosine_similarity(k.unsqueeze(1), M.unsqueeze(0), dim=-1)
        w = F.softmax(beta * sim, dim=-1)
        return w
    
    def write(self, c_out, M, w):
        e = torch.sigmoid(self.e_proj(c_out))   # erase
        a = self.a_proj(c_out)                   # add
        # M update: each row updated by w[i]
        # M_i = M_i ⊙ (1 - w_i e) + w_i a
        # Vectorized:
        # erase term: outer product w * e
        erase_term = w.unsqueeze(-1) * e.unsqueeze(1)   # (B, N, M_dim)
        # erase 평균을 1 batch 만 지원 (단순화)
        M_after_erase = M * (1 - erase_term[0])   # 첫 batch 만, vectorize 단순화
        add_term = w[0].unsqueeze(-1) * a[0].unsqueeze(0)
        M_new = M_after_erase + add_term
        return M_new

write_head = NTMWriteHead(M_dim, H)
w_w = write_head.address(controller_out, M)
M_after = write_head.write(controller_out, M, w_w)
print(f'Memory updated: shape {M_after.shape}, change norm: {(M_after - M).norm():.4f}')
```

### 실험 3 — 단순 NTM Cell

```python
class SimpleNTM(nn.Module):
    """Single read + single write head NTM"""
    def __init__(self, input_dim, output_dim, hidden_dim, M_size, M_dim):
        super().__init__()
        self.M_size, self.M_dim = M_size, M_dim
        self.controller = nn.LSTMCell(input_dim + M_dim, hidden_dim)
        self.read_head = NTMReadHead(M_dim, hidden_dim)
        self.write_head = NTMWriteHead(M_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        # Init memory
        self.register_buffer('M0', torch.randn(M_size, M_dim) * 0.1)
    
    def forward(self, x_seq):
        T, B, _ = x_seq.shape
        h = torch.zeros(B, self.controller.hidden_size)
        c = torch.zeros(B, self.controller.hidden_size)
        r = torch.zeros(B, self.M_dim)
        M = self.M0.clone()
        
        outputs = []
        for t in range(T):
            x_in = torch.cat([x_seq[t], r], dim=-1)
            h, c = self.controller(x_in, (h, c))
            
            # Read
            w_r = self.read_head.address(h, M)
            r = self.read_head.read(w_r, M)
            
            # Write
            w_w = self.write_head.address(h, M)
            M = self.write_head.write(h, M, w_w)
            
            outputs.append(self.output(h))
        return torch.stack(outputs)

D, K = 5, 3
ntm = SimpleNTM(input_dim=D, output_dim=K, hidden_dim=64, M_size=16, M_dim=20)
x = torch.randn(20, 4, D)
out = ntm(x)
print(f'NTM output: {out.shape}')
print(f'Total params: {sum(p.numel() for p in ntm.parameters()):,}')
```

### 실험 4 — Copy Task (NTM 의 표준 benchmark)

```python
def copy_task(T, vec_dim=8, B=32):
    """Copy task: input sequence + delimiter + reproduce"""
    seq = torch.randn(T, B, vec_dim)
    seq = (seq > 0).float()   # Binary
    
    # 추가 channel: 0 = input, 1 = delimiter, 2 = output (zeros)
    inputs = torch.zeros(T * 2 + 1, B, vec_dim + 1)
    inputs[:T, :, :vec_dim] = seq                   # Input phase
    inputs[T, :, vec_dim] = 1.0                      # Delimiter
    # Output phase: zeros (model fills in)
    
    targets = torch.zeros_like(inputs[:, :, :vec_dim])
    targets[T+1:, :, :] = seq    # Reproduce after delimiter
    
    return inputs, targets

# Train NTM and LSTM on copy task
class CopyLSTM(nn.Module):
    def __init__(self, vec_dim, H=128):
        super().__init__()
        self.lstm = nn.LSTM(vec_dim + 1, H)
        self.fc = nn.Linear(H, vec_dim)
    def forward(self, x):
        h, _ = self.lstm(x)
        return torch.sigmoid(self.fc(h))

torch.manual_seed(0)
T_copy = 5
vec_dim = 8

lstm_copy = CopyLSTM(vec_dim, H=128)
opt = torch.optim.Adam(lstm_copy.parameters(), lr=1e-3)

for step in range(100):
    x, y = copy_task(T_copy, vec_dim, B=32)
    pred = lstm_copy(x)
    # Loss only on output phase
    loss = F.binary_cross_entropy(pred[T_copy+1:], y[T_copy+1:])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 20 == 0:
        print(f'LSTM Copy T={T_copy}, step {step}: loss = {loss.item():.4f}')

# NTM 도 비슷하게 train (구현 완전성을 위해)
print('\nNTM 의 진정한 우위는 generalization (학습한 T 이상에서) — Graves 2014 참조')
```

### 실험 5 — Memory Operations 시각화

```python
import matplotlib.pyplot as plt

# Pre-trained NTM (이 demo 에서는 random) 의 attention pattern
torch.manual_seed(0)
ntm = SimpleNTM(input_dim=5, output_dim=3, hidden_dim=64, M_size=10, M_dim=20)
x = torch.randn(15, 1, 5)
T_seq = x.size(0)

# Track read attention
ntm.eval()
with torch.no_grad():
    h = torch.zeros(1, 64); c_lstm = torch.zeros(1, 64)
    r = torch.zeros(1, 20)
    M = ntm.M0.clone()
    
    read_attentions = []
    for t in range(T_seq):
        x_in = torch.cat([x[t], r], dim=-1)
        h, c_lstm = ntm.controller(x_in, (h, c_lstm))
        w_r = ntm.read_head.address(h, M)
        read_attentions.append(w_r.numpy())
        r = ntm.read_head.read(w_r, M)
        w_w = ntm.write_head.address(h, M)
        M = ntm.write_head.write(h, M, w_w)

attn = np.array(read_attentions).squeeze(1)   # (T, N)
plt.figure(figsize=(10, 4))
plt.imshow(attn, aspect='auto', cmap='Blues')
plt.xlabel('Memory row')
plt.ylabel('Time step')
plt.title('NTM Read Attention Pattern (random init)')
plt.colorbar()
plt.savefig('ntm_attention.png', dpi=120, bbox_inches='tight')
print('Saved NTM attention pattern')
```

---

## 🔗 실전 활용

### 1. Algorithmic learning

Synthetic tasks (sort, dynamic programming) — NTM 이 LSTM 보다 generalize 더 잘함.

### 2. Question Answering (Memory Network)

bAbI dataset — Memory Network (Weston 2014, Sukhbaatar 2015) 가 표준. Multi-hop reasoning.

### 3. Differentiable Neural Computer (DNC)

London Underground 경로 찾기, 가족 관계 추론 — Graves 2016 의 graph-based reasoning.

### 4. Meta-learning

External memory 가 task adaptation 의 mechanism — MANN (Santoro 2016).

### 5. Modern descendants

- **Differentiable Datalog** (DeepMind): logic programming
- **Neural Programmer** (Neelakantan 2016): symbolic + neural
- **Routing Transformer**: Memory-augmented attention
- **Retrieval-Augmented Generation (RAG)**: external knowledge retrieval

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Soft attention 충분 | Hard attention 의 sample efficiency 좋을 수도 |
| Differentiable everything | Discrete operations 시 RL 또는 Gumbel-softmax |
| Single memory | Hierarchical memory (multi-scale) 가능 |
| Simple addressing | Complex graph-based addressing (DNC) |
| Static memory size $N$ | Dynamic resizing 어려움 |

---

## 📌 핵심 정리

$$\boxed{\text{NTM} = \text{LSTM controller} + \text{external memory } M + \text{differentiable read/write}}$$

$$\boxed{w_i = \mathrm{softmax}(\beta \cos(k, M_i)) \quad \text{— content-based}}$$

$$\boxed{M_i \leftarrow M_i \odot (1 - w_i e) + w_i a \quad \text{— soft write}}$$

| Memory Type | Capacity | Read | Write |
|------------|----------|------|-------|
| **LSTM** | $H$ floats | Sequential | Implicit (gates) |
| **Memory Network** | $N \times d$ | Random (content) | None (read-only) |
| **NTM** | $N \times d$ | Random | Soft erase + add |
| **DNC** | $N \times d$ + linkage | Multiple heads | Multiple heads + tracking |

---

## 🤔 생각해볼 문제

**문제 1** (기초): NTM 의 read attention $w$ 가 sparse (one-hot like) 일 때 와 dense (uniform) 일 때 의 read 결과 차이를 설명하라.

<details>
<summary>해설</summary>

**Sparse $w$** ($w_i \approx 1$ for one $i$):
$$
r = \sum_j w_j M_j \approx M_i
$$

— 정확히 row $i$ 의 내용 retrieve. **Random access**.

**Dense $w$** ($w_i \approx 1/N$):
$$
r \approx \frac{1}{N} \sum_j M_j = \bar M
$$

— Memory 의 평균. **Distributed retrieval**.

**Sharpening parameter $\beta$ 의 효과**:
- $\beta \to \infty$: $w$ → one-hot (sparse)
- $\beta = 0$: $w$ uniform (dense)
- $\beta \approx 1$: smooth interpolation

**학습 dynamics**:
- 초기 ($\beta$ random): dense → 학습 쉬운 신호
- 후기 ($\beta$ 학습됨): sparse → precise random access

이 sharpening trick 이 *attention* 의 일반적 design pattern — Transformer 의 scaled dot-product 도 같은 정신. $\square$

</details>

**문제 2** (심화): NTM 의 write operation 이 erase + add 두 단계로 나뉜 이유를 explain 하라. 왜 단순 overwrite ($M_i \leftarrow a_i$) 가 충분하지 않은가?

<details>
<summary>해설</summary>

**Soft write 의 형태**:
$$
M_i \leftarrow M_i \odot (1 - w_i e_i) + w_i a_i
$$

**Erase + Add 의 의미**:
- Erase $e \in [0, 1]^d$: 어떤 dimensions 를 *지움*
- Add $a$: 어떤 새 정보를 *추가*
- 두 단계 분리로 **selective update** 가능

**단순 overwrite ($M_i \leftarrow a_i$) 의 문제**:

1. **Soft attention 시 information mixing**:
   - $w_i = 0.5$ 면 $M_i \leftarrow 0.5 \cdot a_i$ — 기존 정보 손실 + 부분 추가
   - "한 row 를 부분적으로 update" 의 의미 모호

2. **No retention control**:
   - 기존 $M_i$ 의 어떤 dimension 을 보존, 어떤 것을 변경할지 결정 불가
   - 모든 dimension 이 똑같이 변경

3. **Gradient flow**:
   - Soft replacement 가 gradient 의 source 약화
   - Erase + add 가 두 separate gradient path

**Erase + Add 의 우월**:

1. **Selective dimension-wise update**:
   - $e_j = 1, a_j = \text{new}$: dimension $j$ 새로 set
   - $e_j = 0$: dimension $j$ 보존
   - $e_j \in (0, 1)$: partial blend

2. **LSTM-like gating**:
   - Erase = forget gate analog (cell state 의 dimensional control)
   - Add = input gate analog
   - 같은 정신을 memory 에

3. **Reversibility (in soft sense)**:
   - $e = 0, a = \text{anything}$: $M$ 변화 없음 — write skip
   - $e = 1, a = $ same as $M_i$: identity write

**실제 NTM 학습**:
- 학습이 task 에 맞춰 erase/add pattern 학습
- Copy task: erase → 1, add → input value (full overwrite)
- Memory persistence: erase → 0 (preservation)

**결론**: Erase + add 의 분리가 *fine-grained control* 을 가능하게 함 — single overwrite 의 한계를 극복. LSTM 의 forget + input 분리와 같은 정신. $\square$

</details>

**문제 3** (논문 비평): NTM 이 "differentiable programming" 의 시작이지만 large-scale 응용에 사용되지 않는다. 그 이유와 modern descendant (Transformer's attention as memory) 의 관계는?

<details>
<summary>해설</summary>

**NTM 의 large-scale 사용 부재 이유**:

1. **Computational cost**:
   - Memory $N \times d$ 의 N 이 크면 cosine similarity $O(N)$ per query
   - Attention 의 $O(T^2)$ 와 비교: NTM 의 $O(NT)$ 가 비슷하지만 sequential
   - GPU parallelism 활용 어려움

2. **Training instability**:
   - Soft attention + sequential write 가 학습 어려움
   - Multiple heads 가 좋지만 parameter 폭발
   - Hyperparameter sensitive (sharpening, gates)

3. **Synthetic vs real**:
   - Synthetic copy / sort: NTM 우월
   - Real NLP/CV: data-driven attention 이 충분 (Transformer)
   - Algorithmic reasoning 은 niche

4. **Architectural complexity**:
   - 8+ specialized components (heads, addressing modes)
   - Debug 어려움
   - Vanilla Transformer 의 simplicity 가 우월

**Transformer's Attention as Memory**:

1. **Self-attention = memory query**:
   - Query $Q = W_Q h_i$
   - Key $K_t = W_K h_t$ for all $t$ — 모든 position 이 *memory cell*
   - Value $V_t = W_V h_t$
   - Read: $\sum_t \alpha_{it} V_t$, $\alpha_{it} = \mathrm{softmax}(QK^T)$

   이는 NTM 의 content-based read 와 *isomorphic*!

2. **Implicit memory**:
   - Sequence positions 자체가 memory cells
   - "Write" 는 layer 간 transformation
   - Query/key/value 분리가 NTM 의 read head 를 generalize

3. **Modern: KV-cache**:
   - Inference 시 past keys/values 를 cache
   - 명시적 external memory!
   - Each generated token 이 KV-cache 에 추가 (write)

4. **RAG (Retrieval-Augmented Generation)**:
   - External document store (Wikipedia 등)
   - Differentiable retrieval (or non-differentiable for efficiency)
   - NTM 의 정신을 large-scale knowledge 로 확장

**현대 differentiable programming**:
- **Neural ODE** (Chen 2018): differentiable simulation
- **Differentiable physics**: graphics, robotics
- **JAX**: 함수형 differentiable computing
- **Diffusion models**: differentiable generative process

**Lesson**:

1. **Idea 의 계승**:
   - NTM 의 *idea* (external memory, differentiable address) 가 Transformer 의 attention 으로 흡수
   - Specific implementation 이 less successful, idea 는 ubiquitous

2. **Simplicity wins**:
   - Vanilla Transformer 가 NTM 보다 단순
   - Single mechanism (self-attention) 이 specialized heads 보다 효과적

3. **Scale 의 force**:
   - NTM 의 algorithmic capability 가 small data 에서 명확
   - Large data 에서 gross statistics 가 algorithmic structure 압도

**결론**: NTM 의 *vision* (differentiable programming with memory) 이 modern Transformer / RAG / agent system 의 정신적 조상. *Implementation* 은 simpler architectures 에 자리. ML 의 진화 — *idea* 가 *implementation* 보다 오래 살아남음. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-stacked-rnn.md) | [📚 README](../README.md) | [다음 ▶](./04-esn.md)

</div>
