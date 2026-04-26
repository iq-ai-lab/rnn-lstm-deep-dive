# 02. Stacked / Deep RNN

## 🎯 핵심 질문

- Stacked RNN 의 multi-layer 구조 $h_t^{(l)} = \sigma(W^{(l)}_{hh} h_{t-1}^{(l)} + W^{(l)}_{xh} h_t^{(l-1)})$ 가 어떻게 표현력을 늘리는가?
- Depth (layer 수) 와 time (sequence length) 의 복잡도 trade-off — $O(L \cdot T \cdot d^2)$ 의 의미는?
- Stacked RNN 의 vanishing 이 horizontal (time) 과 vertical (depth) 두 축에서 발생하는 이유?
- Residual connection 이 vertical vanishing 을 어떻게 완화하는가?
- **Google NMT** (Wu 2016) 의 8-layer LSTM 이 어떻게 stable training 을 달성했는가? Layer-wise dropout?

---

## 🔍 왜 Stacking 이 RNN 의 표현력을 확장하는가

Single-layer RNN 의 한계:
1. **표현력 제한** — Single nonlinearity 후 직접 output, hierarchical feature 표현 어려움
2. **Long-range vs short-range 분리 안 됨** — 모든 task 가 한 hidden 에 압축
3. **Modularity 부족** — Layer 별 specialization 불가

Stacked RNN 이 이를 해결:
1. **Hierarchical representation** — 하위 layer 가 local, 상위가 global feature
2. **Increased capacity** — $L \times H$ effective hidden
3. **Modularity** — Layer 별 다른 abstraction level

그러나 새로운 도전:
- **Vertical vanishing** — Layer 간 gradient 감쇠
- **Computational cost** — $O(L \cdot T \cdot d^2)$
- **Optimization difficulty** — Deep network 의 saddle points

이 문서는 stacked RNN 의 정의, complexity, 그리고 deep training 의 challenges 와 해법을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-bidirectional.md](./01-bidirectional.md) — RNN 의 다양한 변형
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Deep network 의 vanishing gradient
- (선택) Residual connection: ResNet 의 정신

---

## 📖 직관적 이해

### Stacked RNN 의 구조

```
Layer 3:  h_1^{(3)} → h_2^{(3)} → h_3^{(3)} → ...
              ↑          ↑          ↑
Layer 2:  h_1^{(2)} → h_2^{(2)} → h_3^{(2)} → ...
              ↑          ↑          ↑
Layer 1:  h_1^{(1)} → h_2^{(1)} → h_3^{(1)} → ...
              ↑          ↑          ↑
Input:        x_1        x_2        x_3
```

- **Vertical**: $h^{(l)}$ 가 $h^{(l-1)}$ 의 input 으로 사용 (layer 간)
- **Horizontal**: 같은 layer 내 sequential dependency (time 간)

### Hierarchical Representation 비유

CNN 의 layer-wise feature:
- Layer 1: edge, corner
- Layer 2: texture, simple shape
- Layer 3: object part
- Layer 4: full object

Stacked RNN 도 비슷한 정신:
- Layer 1: word-level features
- Layer 2: phrase-level patterns
- Layer 3: sentence-level semantics
- Layer 4: document-level concepts

### Vertical Vanishing 의 직관

Single-layer LSTM: time 축의 vanishing 만 (CEC 로 일부 해결).

Stacked LSTM: time + depth 두 축의 vanishing.

```
                        ↑ vertical (deep)
                        │  vanishing 가능
                        │
  ←── horizontal (long) ──→
       vanishing (CEC 로 해결)
```

ResNet 정신의 residual connection 이 vertical vanishing 해결.

### Layer-wise Dropout

Variational dropout: 같은 mask 를 모든 time step 에서 사용 (Gal 2016).

```
Layer 2:  ●  ●  ●  ●  ●  ●  ●  ←  같은 dropout mask
          │  │  │  │  │  │  │
Layer 1:  ●  ○  ●  ●  ○  ●  ●  ←  같은 mask (다른 set, layer 별)
```

Standard dropout (random per step) 은 RNN 의 information flow 를 disrupt — variational 이 표준.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Stacked RNN

$L$-layer stacked RNN, layer $l \in \{1, \ldots, L\}$:

$$
h_t^{(l)} = f^{(l)}(h_{t-1}^{(l)}, h_t^{(l-1)})
$$

여기서 $h_t^{(0)} := x_t$ (input). 마지막 output $y_t$ 는 $h_t^{(L)}$ 의 함수.

### 정의 2.2 — Stacked LSTM

각 layer 가 LSTM:

$$
\begin{aligned}
h_t^{(l)}, c_t^{(l)} = \text{LSTM}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)}, c_{t-1}^{(l)})
\end{aligned}
$$

각 layer 별 distinct weights $\theta^{(l)} = (W_f^{(l)}, W_i^{(l)}, W_c^{(l)}, W_o^{(l)})$.

### 정의 2.3 — Parameter Count

$L$-layer LSTM, hidden $H$:

$$
|\theta_{\text{stacked}}| = 4H(D + H + 1) + (L-1) \times 4H(2H + 1)
$$

(Layer 1 의 input 차원 $D$, layer 2+ 의 input 차원 $H$ — 이전 layer 의 hidden)

### 정의 2.4 — Computational Complexity

- Forward time: $O(L \cdot T \cdot H^2)$
- Memory: $O(L \cdot T \cdot H)$
- Backward time: $O(L \cdot T \cdot H^2)$ (same as forward)

### 정의 2.5 — Residual Stacked RNN

Layer 간 residual:

$$
h_t^{(l)} = h_t^{(l-1)} + \text{LSTM}^{(l)}(\ldots)
$$

(IRNN 의 horizontal residual 의 vertical 버전)

---

## 🔬 정리와 결과

### 정리 2.1 — Hierarchical Capacity

$L$-layer stacked RNN 의 effective hidden capacity:

$$
\text{capacity} = L \cdot H
$$

(Each layer 가 $H$ floats, total $L H$)

비교 single-layer with $H' = LH$: 같은 floats 이지만 *flat* representation.

**의미**: Stacking 이 hierarchy 를 명시적으로 학습 — flat 보다 inductive bias 강함.

### 정리 2.2 — Vertical Gradient Flow

$$
\frac{\partial h_t^{(L)}}{\partial h_t^{(1)}} = \prod_{l=2}^{L} \frac{\partial h_t^{(l)}}{\partial h_t^{(l-1)}}
$$

Multi-layer matrix product → spectral radius 의 곱 → vanishing 가능.

**증명**: Chain rule across layers. $\square$

### 정리 2.3 — Total Gradient Vanishing

$$
\frac{\partial h_T^{(L)}}{\partial h_0^{(1)}} = \underbrace{\prod_l \prod_t (\text{horizontal Jacobians})}_{T \cdot L \text{ matrix products}}
$$

이중 vanishing 위험 — both depth and time.

### 정리 2.4 — Residual Connection 의 효과

Residual layer:
$$
h_t^{(l)} = h_t^{(l-1)} + f^{(l)}(...)
$$

$$
\frac{\partial h_t^{(l)}}{\partial h_t^{(l-1)}} = I + \frac{\partial f^{(l)}}{\partial h_t^{(l-1)}}
$$

$f^{(l)}$ 가 작으면 $\approx I$ — gradient 보존.

### 정리 2.5 — Layer-wise Variational Dropout

Same mask across time:
$$
h_t^{(l)} \leftarrow m^{(l)} \odot h_t^{(l)} \quad \text{(same } m^{(l)} \text{ for all } t)
$$

Standard dropout: independent $m_t^{(l)}$ — RNN information flow disrupt.
Variational: consistent dropout pattern — Bayesian interpretation (Gal 2016).

---

## 💻 PyTorch 구현 검증

### 실험 1 — PyTorch Stacked LSTM

```python
import torch
import torch.nn as nn

D, H, T, B, L = 10, 32, 20, 4, 3

# num_layers 로 stacked
stacked = nn.LSTM(D, H, num_layers=L, batch_first=False)
x = torch.randn(T, B, D)
out, (h_T, c_T) = stacked(x)

print(f'Output:    {out.shape}')        # (T, B, H) — 마지막 layer 의 hidden
print(f'h_T:       {h_T.shape}')        # (L, B, H) — 모든 layer 의 마지막
print(f'c_T:       {c_T.shape}')        # (L, B, H)

print(f'\nParameters per layer:')
for name, param in stacked.named_parameters():
    print(f'  {name}: {param.shape}')
# layer 1 의 input 차원 D, layer 2+ 의 input 차원 H

print(f'\nTotal params: {sum(p.numel() for p in stacked.parameters()):,}')
```

### 실험 2 — Manual Stacking (이해를 위한)

```python
class ManualStackedLSTM(nn.Module):
    def __init__(self, D, H, L):
        super().__init__()
        self.D, self.H, self.L = D, H, L
        # 각 layer 별 LSTMCell
        self.cells = nn.ModuleList([
            nn.LSTMCell(D if l == 0 else H, H) for l in range(L)
        ])
    
    def forward(self, x_seq, init_states=None):
        T, B, _ = x_seq.shape
        if init_states is None:
            h = [torch.zeros(B, self.H) for _ in range(self.L)]
            c = [torch.zeros(B, self.H) for _ in range(self.L)]
        else:
            h, c = init_states
        
        outputs = []
        for t in range(T):
            x_t = x_seq[t]
            # Layer-by-layer forward
            for l in range(self.L):
                h[l], c[l] = self.cells[l](x_t, (h[l], c[l]))
                x_t = h[l]   # 다음 layer 의 input
            outputs.append(h[-1])
        return torch.stack(outputs)

torch.manual_seed(0)
manual = ManualStackedLSTM(D, H, L)
out_manual = manual(x)
print(f'Manual stacked output: {out_manual.shape}')
```

### 실험 3 — Layer 별 Hidden 의 의미 분석

```python
class TrackingStackedLSTM(nn.Module):
    """각 layer 의 hidden 추적"""
    def __init__(self, D, H, L):
        super().__init__()
        self.cells = nn.ModuleList([
            nn.LSTMCell(D if l == 0 else H, H) for l in range(L)
        ])
        self.L = L
        self.H = H
    
    def forward(self, x_seq):
        T, B, _ = x_seq.shape
        h = [torch.zeros(B, self.H) for _ in range(self.L)]
        c = [torch.zeros(B, self.H) for _ in range(self.L)]
        all_layers = [[] for _ in range(self.L)]
        
        for t in range(T):
            x_t = x_seq[t]
            for l in range(self.L):
                h[l], c[l] = self.cells[l](x_t, (h[l], c[l]))
                x_t = h[l]
                all_layers[l].append(h[l].clone())
        return [torch.stack(layer) for layer in all_layers]

torch.manual_seed(0)
m = TrackingStackedLSTM(D, H, 3)
all_h = m(x)
for l, h_layer in enumerate(all_h):
    print(f'Layer {l+1}: shape {h_layer.shape}, norm avg {h_layer.norm(dim=-1).mean():.4f}')
# Lower layers 가 보통 더 작은 norm (less abstract), higher 가 더 큰 norm (more semantic)
```

### 실험 4 — Residual Stacked LSTM

```python
class ResidualStackedLSTM(nn.Module):
    """Vertical residual connection"""
    def __init__(self, D, H, L):
        super().__init__()
        self.input_proj = nn.Linear(D, H) if D != H else nn.Identity()
        self.cells = nn.ModuleList([
            nn.LSTMCell(H, H) for _ in range(L)
        ])
        self.L = L
        self.H = H
    
    def forward(self, x_seq):
        T, B, _ = x_seq.shape
        x_seq = self.input_proj(x_seq)   # Project to H if needed
        h = [torch.zeros(B, self.H) for _ in range(self.L)]
        c = [torch.zeros(B, self.H) for _ in range(self.L)]
        outputs = []
        for t in range(T):
            x_t = x_seq[t]
            for l in range(self.L):
                h_new, c[l] = self.cells[l](x_t, (h[l], c[l]))
                # Residual: x_t (이전 layer output) + h_new
                h[l] = x_t + h_new
                x_t = h[l]
            outputs.append(h[-1])
        return torch.stack(outputs)

torch.manual_seed(0)
res = ResidualStackedLSTM(D, H, L=8)   # 8 layer (deep)
out_res = res(x)
print(f'Residual deep LSTM output: {out_res.shape}')
print(f'Parameters: {sum(p.numel() for p in res.parameters()):,}')
```

### 실험 5 — Variational Dropout

```python
class VariationalDropoutLSTM(nn.Module):
    """Same mask across time steps (Gal 2016)"""
    def __init__(self, D, H, L, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(D, H, num_layers=L, dropout=dropout)
        # PyTorch 의 nn.LSTM 은 layer 간 dropout 만 (output side)
        # Variational dropout 는 hidden state 에 same mask 적용 (custom 구현 필요)
    
    def forward(self, x):
        return self.lstm(x)[0]

# Custom variational dropout 구현은 RNN cell 단위로 mask 보존 필요
# 단순화: PyTorch 의 dropout 옵션 사용
torch.manual_seed(0)
vdrop = VariationalDropoutLSTM(D, H, L=2, dropout=0.3)
vdrop.train()   # Dropout active
out_vdrop = vdrop(x)
print(f'With dropout: {out_vdrop.shape}')

# Note: PyTorch nn.LSTM 의 dropout 은 layer 사이 drop, time-step 별 다른 mask
# True variational dropout 는 fastai 등 high-level library 사용
```

---

## 🔗 실전 활용

### 1. Google NMT (Wu 2016)

8-layer LSTM encoder + 8-layer LSTM decoder. Residual connection 으로 deep training 안정화. WMT'14 SOTA.

### 2. Speech recognition

DeepSpeech 2 (Amodei 2016): 9-layer RNN (LSTM/GRU). Layer-wise dropout 표준.

### 3. Language modeling

AWD-LSTM (Merity 2018): 3-layer LSTM + variational dropout + ASGD. PTB SOTA at the time.

### 4. ELMo

2-layer BiLSTM + character-level CNN. Pre-trained on 1B Word Benchmark.

### 5. Modern usage

Transformer 가 sequence modeling 표준이 되면서 stacked RNN 사용 감소. 그러나 streaming, edge AI 에서 잔존.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Layer 별 distinct weights | Tied weights 가능, 그러나 표현력 손실 |
| Sequential per layer | 병렬화 어려움 — Transformer 의 우월 |
| Vertical vanishing | Residual, gradient checkpointing |
| 동일 hidden size | Layer 별 다른 size 가능 (trapezoidal) |
| Dense connection | Sparse, mixture-of-experts 가능 |

---

## 📌 핵심 정리

$$\boxed{h_t^{(l)} = f^{(l)}(h_{t-1}^{(l)}, h_t^{(l-1)}), \quad h_t^{(0)} = x_t}$$

$$\boxed{\text{Time complexity: } O(L \cdot T \cdot H^2)}$$

$$\boxed{\text{Vertical vanishing: residual} \;\; h^{(l)} = h^{(l-1)} + f^{(l)}}$$

| Layer count | Use case |
|-------------|----------|
| **1** | Simple tasks, fast |
| **2-3** | Standard NLP (PTB LM, NER) |
| **4-8** | Production NMT (Google, Baidu) |
| **8+** | Requires residual + careful training |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3-layer LSTM 의 parameter 수를 계산하라. $D = 100, H = 256$.

<details>
<summary>해설</summary>

**Layer 1**: $D = 100 \to H = 256$
$$
4 \times 256 \times (100 + 256 + 1) = 1024 \times 357 = 365{,}568
$$

**Layer 2, 3**: $H = 256 \to H = 256$
$$
4 \times 256 \times (256 + 256 + 1) = 1024 \times 513 = 525{,}312 \;\; \text{(each)}
$$

**Total**: $365{,}568 + 2 \times 525{,}312 = 1{,}416{,}192$ — 약 1.4M parameters.

**비교 single-layer with same total params** ($H' = ?$):
$4 H'(100 + H' + 1) = 1.4M \Rightarrow H' \approx 600$

— 1 layer 600 hidden vs 3 layer 256 hidden: 같은 parameter 이지만 stacked 가 hierarchical representation 학습 가능. $\square$

</details>

**문제 2** (심화): Vertical vanishing 의 spectral 분석을 시간 축의 그것과 비교하라. 두 축의 vanishing 이 *어떻게 결합* 되어 deep stacked RNN 의 학습 어려움을 만드는가?

<details>
<summary>해설</summary>

**Time-axis vanishing** (Ch3-01):
$$
\prod_{t=1}^{T} J_t \quad (J_t \text{ = horizontal Jacobian})
$$

Spectral radius $\rho(W_{hh}) \cdot \sigma'$ 의 $T$-거듭제곱.

**Depth-axis vanishing**:
$$
\prod_{l=1}^{L} K_l \quad (K_l \text{ = vertical Jacobian})
$$

여기서 $K_l = \partial h_t^{(l)} / \partial h_t^{(l-1)} = \mathrm{diag}(\sigma'(z_t^{(l)})) \cdot W^{(l)}$.

비슷한 spectral 분석 — $\rho(W^{(l)}) \cdot \sigma'$ 의 $L$-거듭제곱.

**결합 vanishing** (deep stacked RNN):
$$
\frac{\partial h_T^{(L)}}{\partial h_0^{(1)}} = \prod_{t} \prod_{l} (\text{Jacobian product})
$$

— $T \times L$ matrix products. 각 axis 가 spectral 의 거듭제곱 → 결합하면 $\rho^{T+L}$ 형태 (대략).

**Numerical example**:
- $\rho = 0.95$, $T = 100$, $L = 5$
- Time vanishing alone: $0.95^{100} = 0.006$
- Depth vanishing alone: $0.95^5 = 0.77$
- Combined: $0.95^{105} = 0.005$ — depth 가 marginal 추가

그러나 **strong vanishing** 또는 **deep network**:
- $\rho = 0.5$, $T = 50$, $L = 8$
- Time: $0.5^{50} \approx 10^{-15}$
- Depth: $0.5^8 \approx 0.004$
- Combined: $10^{-15} \cdot 0.004 = 4 \times 10^{-18}$ — catastrophic

**Solutions**:
1. **Time axis**: LSTM 의 CEC (cell path)
2. **Depth axis**: Residual connection (ResNet 의 RNN application)
3. **Both**: Highway Network — gated residual on both axes

**Empirical**:
- Plain stacked RNN: $L > 3$ 학습 매우 어려움
- LSTM + residual: $L = 8$ training stable (Google NMT)
- Transformer: $L = 12, 24, 96$ 까지 가능 — attention 이 둘 다 우회

**결론**: Vanishing 이 *additive* 가 아닌 *multiplicative* — 두 축의 곱셈. Deep RNN 에서는 *둘 다* 해결 필요. Modern Transformer 의 attention + residual + layer norm 가 두 축 모두 우회. $\square$

</details>

**문제 3** (논문 비평): Google NMT 의 8-layer LSTM 이 Transformer 이전의 NMT SOTA 였다. 왜 깊게 쌓는 것이 효과적이었으며, Transformer 가 이를 어떻게 *better* 하게 했는가?

<details>
<summary>해설</summary>

**Google NMT (Wu 2016)**:
- 8-layer LSTM encoder + 8-layer LSTM decoder
- Residual connection (vertical)
- Layer-wise dropout
- Attention layer (Bahdanau-style)
- 16 GPU days training

**왜 효과적**:

1. **Hierarchical representation**:
   - 하위 layer: word-level
   - 중간 layer: phrase, syntax
   - 상위 layer: semantic, pragmatic
   - Translation 의 각 abstraction level 학습

2. **Larger capacity**:
   - 8 × 1024 = 8192 effective hidden
   - Single-layer 8192 hidden 보다 hierarchical 학습

3. **Residual + dropout**:
   - Vertical vanishing 회피
   - Generalization 향상

4. **Attention 의 결합**:
   - Encoder-decoder bottleneck 회피 (Ch6-03)

**Transformer 의 우월** (Vaswani 2017):

1. **Sequence parallelism**:
   - LSTM: $T$ steps sequential — GPU underutilization
   - Transformer: 모든 position 동시 — full GPU usage
   - **결정적 advantage**: 같은 hardware 에서 더 큰 모델 가능

2. **Direct attention**:
   - LSTM: $h_8$ 가 input 의 압축
   - Transformer: 각 position 이 모든 input position direct access
   - Information bottleneck 우회

3. **Easier optimization**:
   - LSTM 의 두 축 vanishing
   - Transformer 의 layer norm + residual + attention
   - Stable training to 24+ layers

4. **Inductive bias**:
   - LSTM: Markov-like, position-aware (sequential)
   - Transformer: position-agnostic + PE — flexible
   - Long-range modeling 더 자연스러움

**Empirical**:
- Vaswani 2017 Table 2: Transformer-Big 이 Google NMT 대비 +0.7 BLEU (WMT'14 En→De)
- 학습 시간: Transformer 3.5 days, Google NMT 16 days
- **2배 짧은 시간에 더 높은 quality**

**Lesson**:

1. **Stacked LSTM 이 RNN 의 한계 내 maximum**:
   - 8-layer 가 capacity / training 의 sweet spot
   - 더 깊게는 vanishing 으로 어려움

2. **Architecture redesign 의 효과**:
   - LSTM 의 sequential 한계가 본질적 문제
   - Transformer 가 fundamental 변경 — sequential → parallel

3. **Hardware-architecture 공진화**:
   - GPU 의 SIMT 가 Transformer 에 적합
   - LSTM 은 historical (CPU 시대), Transformer 는 modern (GPU 시대)

**결론**: Stacked LSTM 이 RNN family 의 culmination, Google NMT 가 그 정수. 그러나 *paradigm shift* (Transformer) 가 incremental improvement (deeper LSTM) 를 능가. **Architecture design 의 본질적 변경** vs **incremental scaling** 의 차이가 ML 발전의 패턴. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-bidirectional.md) | [📚 README](../README.md) | [다음 ▶](./03-ntm-memory.md)

</div>
