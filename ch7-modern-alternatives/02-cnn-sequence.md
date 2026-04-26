# 02. CNN-based Sequence Model

## 🎯 핵심 질문

- **WaveNet** (van den Oord 2016) 의 causal dilated convolution 이 어떻게 $O(\log T)$ receptive field 를 만들고 완전 병렬을 달성하는가?
- Dilation rate $1, 2, 4, \ldots, 2^L$ 의 exponential growth 가 long-range dependency 를 어떻게 capture 하는가?
- **TCN** (Bai 2018) 의 *An Empirical Evaluation of Generic Convolutional and Recurrent Networks* 가 보여준 TCN > LSTM 의 다양한 task 결과
- Causal convolution 의 정의 — 미래 정보 누설 방지
- CNN-based sequence 의 한계: receptive field 의 *explicit* limit, position-aware 부족

---

## 🔍 왜 CNN 이 sequence 에 자연스러운 대안인가

CNN 은 image 에 특화 (translation invariance, spatial locality), 그러나 1D sequence 에도 적용 가능:

1. **Translation invariance** (1D): 같은 pattern 이 다른 position 에서도 detect
2. **Sparse local connection**: 인접 token 만 직접 connect — efficient
3. **Parallel computation**: 모든 position 동시 처리 — RNN 의 sequential 한계 회피
4. **Hierarchical receptive field**: layer 가 깊어질수록 더 long-range

WaveNet 과 TCN 이 이를 sequence model 의 alternative 로 입증.

이 문서는 dilated convolution 의 정확한 메커니즘, receptive field analysis, 그리고 sequence task 에서의 trade-off 를 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-parallelism-limit.md](./01-parallelism-limit.md) — RNN 의 sequence parallelism 한계
- [CNN Deep Dive](https://github.com/iq-ai-lab/cnn-deep-dive) — Convolution, padding, stride
- 정의: Dilation, causal masking

---

## 📖 직관적 이해

### Causal Convolution

```
Standard 1D conv (kernel=3):
  output[t] uses input[t-1], input[t], input[t+1]
  
Causal 1D conv:
  output[t] uses input[t-2], input[t-1], input[t]
  (no future leakage — autoregressive 가능)
```

Padding 과 shift 로 구현: kernel size $k$, left-padding $k-1$.

### Dilated Convolution

```
Dilation rate 1 (standard):   ◯ ◯ ◯
                              └─┴─┘ kernel uses adjacent

Dilation rate 2:               ◯ ● ◯ ● ◯
                              └───┴───┘ kernel uses every 2nd

Dilation rate 4:               ◯ ● ● ● ◯ ● ● ● ◯
                              └───────┴───────┘ kernel uses every 4th
```

같은 kernel size 로 더 큰 receptive field.

### WaveNet 의 Hierarchical Dilation

```
Layer 1: dilation 1   |  RF size 3
Layer 2: dilation 2   |  RF size 7
Layer 3: dilation 4   |  RF size 15
Layer 4: dilation 8   |  RF size 31
Layer L: dilation 2^L |  RF size 2^(L+1) - 1
```

$L$ layers → exponential receptive field $O(2^L)$ — **logarithmic depth** for $T$ context.

### Receptive Field 시각화

```
Layer 0 (input):  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
                                          │
Layer 1 (d=1):    ◯  ◯  ◯  ◯  ◯  ◯  ◯  ◯  ◯  ◯  ◯  ◯  ◯  ◯
                                          │
Layer 2 (d=2):                  ◯  ◯  ◯  ◯  ◯  ◯  ◯
                                          │
Layer 3 (d=4):                              ◯
                                          
RF for output position: 15 inputs (with dilation 1, 2, 4)
```

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Causal Convolution

Standard 1D conv:
$$
y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t + i - \lfloor k/2 \rfloor}
$$

Causal conv:
$$
y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t - i}
$$

(미래 사용 안 함)

### 정의 2.2 — Dilated Causal Convolution

Dilation rate $d$:
$$
y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t - d \cdot i}
$$

Receptive field per layer = $1 + (k-1) \cdot d$.

### 정의 2.3 — Stacked Dilated Conv (WaveNet)

Layer $\ell$ 의 dilation $d_\ell = 2^\ell$:

$$
\text{Total RF} = 1 + (k-1) \sum_{\ell=0}^{L-1} 2^\ell = 1 + (k-1)(2^L - 1)
$$

$L = 10, k = 2$: RF = 1024.

### 정의 2.4 — TCN (Temporal Convolutional Network)

Bai 2018:
- Causal dilated convolutions
- Residual connections (depth)
- Weight normalization
- Dropout

Standard architecture for sequence modeling.

### 정의 2.5 — Receptive Field Computation

Cumulative across layers:
$$
\text{RF}(L) = 1 + \sum_{\ell=1}^{L} (k_\ell - 1) \cdot d_\ell \cdot \prod_{m < \ell} s_m
$$

(stride $s_m$, dilation $d_\ell$, kernel $k_\ell$)

---

## 🔬 정리와 결과

### 정리 2.1 — Logarithmic Receptive Field

WaveNet/TCN with dilations $1, 2, 4, \ldots, 2^{L-1}$:
$$
\text{RF}(L) = 1 + (k-1)(2^L - 1) \approx (k-1) \cdot 2^L
$$

**$L \approx \log T$** 로 충분히 긴 receptive field 달성.

### 정리 2.2 — Parallel Computation

각 layer 의 모든 output position 이 *독립* 계산:
$$
y_t^{(\ell)} = f(x_{t - dk}, \ldots, x_t)
$$

Layer 내 critical path = 1 (single op).

**Total critical path**: $L = \log_2 T$ — RNN 의 $T$ 와 비교 dramatic improvement.

### 정리 2.3 — TCN > LSTM (Bai 2018)

다양한 sequence task 에서 TCN 이 LSTM 능가:
- Adding Problem (long-range): TCN converge faster
- Sequential MNIST: TCN 정확도 +1%
- Polyphonic Music: TCN BLEU 비슷 또는 우월
- Char-level language model: similar

**General observation**: TCN 이 LSTM 의 *전반적* 대안.

### 정리 2.4 — Receptive Field 의 Explicit Limit

CNN 의 RF 가 *fixed* (architecture 로 결정). 더 긴 dependency 학습 시:
- Layer 추가 → exponential RF 증가
- 그러나 *fixed* limit (no infinite memory like RNN)

### 정리 2.5 — Position Awareness 부족

Standard CNN 은 *translation invariant* — 같은 pattern 이 어디서든 같은 output. Sequence 에서는 *position* 도 중요할 수 있음 → positional encoding 또는 explicit position feature 추가 필요.

---

## 💻 PyTorch 구현 검증

### 실험 1 — Causal 1D Conv

```python
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    """Causal (autoregressive) 1D convolution"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)
        # Trim right padding (causal)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out

# Test
B, C, T = 2, 4, 10
x = torch.randn(B, C, T)
causal = CausalConv1d(C, C, kernel_size=3, dilation=1)
out = causal(x)
print(f'Input:  {x.shape}')
print(f'Output: {out.shape}')   # Same T
print(f'Causal verification: out[:, :, t] should not depend on x[:, :, > t]')

# 검증: x 의 future 를 변경해도 output 의 past 는 불변
x_modified = x.clone()
x_modified[:, :, T-1] = 100   # Last position 변경
out_modified = causal(x_modified)
print(f'out and out_modified differ in early positions? {(out[:, :, :T-1] - out_modified[:, :, :T-1]).abs().max():.4e}')
# 0 — past 는 future 변경에 영향 없음 (causal)
```

### 실험 2 — Dilated Conv 의 Receptive Field

```python
def receptive_field(kernel_size, dilations):
    """Stacked dilated conv 의 cumulative RF"""
    rf = 1
    for d in dilations:
        rf += (kernel_size - 1) * d
    return rf

dilations_wavenet = [2**i for i in range(10)]   # 1, 2, 4, ..., 512
rf = receptive_field(kernel_size=2, dilations=dilations_wavenet)
print(f'WaveNet 10 layers (k=2): RF = {rf}')
# 1024

# vs. standard CNN (no dilation)
rf_standard = receptive_field(kernel_size=3, dilations=[1]*10)
print(f'Standard 10 layers (k=3): RF = {rf_standard}')
# 21 — much smaller
```

### 실험 3 — TCN 구현 (Simplified)

```python
class TCNBlock(nn.Module):
    """Bai 2018 TCN block: dilated conv + residual"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + self.residual(x))

class TCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, n_layers, kernel_size=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_c = in_ch if i == 0 else hidden_ch
            dilation = 2 ** i
            layers.append(TCNBlock(in_c, hidden_ch, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.tcn(x)

# Test
torch.manual_seed(0)
T, B, D, H, L = 100, 4, 8, 32, 6
tcn = TCN(in_ch=D, hidden_ch=H, n_layers=L, kernel_size=2)
x = torch.randn(B, D, T)   # (B, C, T) for Conv1d
out = tcn(x)
print(f'TCN output: {out.shape}')
print(f'RF: {receptive_field(2, [2**i for i in range(L)])}')   # 64
```

### 실험 4 — TCN vs LSTM Adding Problem

```python
import numpy as np

def adding_problem_data(T, B):
    seqs = np.zeros((B, 2, T), dtype=np.float32)
    seqs[:, 0, :] = np.random.uniform(0, 1, (B, T))
    pos1 = np.random.randint(0, T // 2, B)
    pos2 = T // 2 + np.random.randint(0, T // 2, B)
    targets = np.zeros(B, dtype=np.float32)
    for b in range(B):
        seqs[b, 1, pos1[b]] = 1.0
        seqs[b, 1, pos2[b]] = 1.0
        targets[b] = seqs[b, 0, pos1[b]] + seqs[b, 0, pos2[b]]
    return torch.tensor(seqs), torch.tensor(targets)

class TCNAdder(nn.Module):
    def __init__(self, T, H=64, L=8):
        super().__init__()
        self.tcn = TCN(in_ch=2, hidden_ch=H, n_layers=L, kernel_size=2)
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        # x: (B, 2, T)
        h = self.tcn(x)
        # Take last position
        return self.fc(h[:, :, -1]).squeeze(-1)

class LSTMAdder(nn.Module):
    def __init__(self, H=128):
        super().__init__()
        self.lstm = nn.LSTM(2, H, batch_first=True)
        with torch.no_grad():
            self.lstm.bias_ih_l0[H:2*H].fill_(1.0)
        self.fc = nn.Linear(H, 1)
    def forward(self, x):
        # x: (B, 2, T) → (B, T, 2)
        h, _ = self.lstm(x.transpose(1, 2))
        return self.fc(h[:, -1, :]).squeeze(-1)

def train(model, T_seq, n_steps=100):
    torch.manual_seed(42); np.random.seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(n_steps):
        x, y = adding_problem_data(T_seq, 64)
        pred = model(x)
        loss = ((pred - y)**2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return np.mean(losses[-10:])

T_seq = 100
print(f'Adding Problem T={T_seq}:')
print(f'  TCN  final MSE: {train(TCNAdder(T_seq), T_seq):.4f}')
print(f'  LSTM final MSE: {train(LSTMAdder(), T_seq):.4f}')
# TCN 이 보통 LSTM 보다 빠른 수렴 (Bai 2018)
```

### 실험 5 — Speed Comparison

```python
import time

def time_forward(model, x, n_iter=20):
    for _ in range(5):
        model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - start) / n_iter * 1000

T, B = 1000, 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tcn = TCN(in_ch=8, hidden_ch=64, n_layers=10, kernel_size=2).to(device)
lstm = nn.LSTM(8, 128, batch_first=True).to(device)

x_tcn = torch.randn(B, 8, T, device=device)
x_lstm = torch.randn(B, T, 8, device=device)

t_tcn = time_forward(tcn, x_tcn)
def lstm_fwd():
    return lstm(x_lstm)
t_lstm = time_forward(lambda x: lstm(x), x_lstm)

print(f'\nForward time (T={T}, B={B}):')
print(f'  TCN:  {t_tcn:.2f} ms')
print(f'  LSTM: {t_lstm:.2f} ms')
print(f'  Speedup: {t_lstm/t_tcn:.1f}x')
# TCN 의 parallel computation 이 long sequence 에서 우월
```

---

## 🔗 실전 활용

### 1. Audio generation (WaveNet)

van den Oord 2016 의 raw audio generation. Speech synthesis (Tacotron + WaveNet).

### 2. Time series

Forecasting, anomaly detection — TCN 이 LSTM 의 강력한 alternative.

### 3. Language modeling

Bai 2018 의 char-level LM 에서 TCN 이 LSTM 능가. 그러나 Transformer 가 표준화.

### 4. Action recognition (video)

3D CNN + temporal dilation — video understanding 의 표준.

### 5. Modern hybrid

Conv-Transformer (Audio Transformer 의 conv encoder), HuBERT 등 — CNN feature + attention 결합.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Fixed receptive field | Long-range > RF 학습 불가 — depth 추가 |
| Translation invariance | Position-aware 한 task 에 부적합 — PE 추가 |
| Same kernel everywhere | Adaptive kernel (e.g., learned dilation) 가능 |
| Causal mask | Bi-directional 시 future leakage 위험 |
| 1D structure | 2D image-like sequence (e.g., spectrogram) 는 다른 처리 |

---

## 📌 핵심 정리

$$\boxed{\text{Causal conv: } y_t = \sum_i w_i x_{t-i}}$$

$$\boxed{\text{Dilated conv: } y_t = \sum_i w_i x_{t - d \cdot i}}$$

$$\boxed{\text{Stacked dilation } 2^\ell \implies \text{RF} = O(2^L)}$$

| Architecture | Time | Memory | RF | Parallel |
|--------------|------|--------|----|---------:|
| **RNN/LSTM** | $O(TH^2)$ | $O(TH)$ | Infinite (theory) | × |
| **WaveNet/TCN** | $O(TLH^2)$ | $O(TH)$ | $O(2^L)$ | ✓ |
| **Transformer** | $O(T^2H)$ | $O(T^2)$ | Global | ✓ |
| **Mamba** | $O(TH)$ | $O(TH)$ | Infinite | ✓ (scan) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Kernel size $k = 3$, 8 layers 의 dilated CNN (dilations $1, 2, 4, \ldots, 128$) 의 receptive field 를 계산하라.

<details>
<summary>해설</summary>

**RF formula**:
$$
\text{RF} = 1 + (k-1) \sum_{\ell=0}^{L-1} d_\ell
$$

**Calculation**:
- $\sum d = 1 + 2 + 4 + \ldots + 128 = 255$
- $(k-1) = 2$
- RF = $1 + 2 \times 255 = 511$

**비교**:
- Standard 8 layers (no dilation): $1 + 2 \times 8 = 17$
- Dilated 8 layers: $511$ — **30x larger**

**Implication**:
- 511 token context (e.g., paragraph)
- For longer (document-level), more layers 또는 더 큰 kernel
- Exponential RF growth efficient

$\square$

</details>

**문제 2** (심화): TCN 이 LSTM 보다 우월하지만 Transformer 이 TCN 보다 우월. 두 비교의 trade-off 를 분석하라.

<details>
<summary>해설</summary>

**TCN vs LSTM**:
- TCN advantages: parallel, larger gradient flow (residual + dilation), task-flexible
- LSTM advantages: infinite RF (theory), streaming inference, simpler

**TCN vs Transformer**:

**TCN advantages**:
1. **Smaller memory**: $O(TH)$ vs $O(T^2)$
2. **Position-aware**: convolution 의 spatial structure
3. **Local inductive bias**: 인접 token 간 강한 dependency
4. **Inference speed**: smaller compute per step

**Transformer advantages**:
1. **Global RF**: any position attend to any other
2. **Empirical superiority**: 대부분 NLP/CV task 에서
3. **Pre-training paradigm**: BERT, GPT 의 standard
4. **In-context learning**: emergent property

**구체적 비교** (sequence length $T = 1000$):

| | TCN | Transformer |
|--|-----|-------------|
| **Time** | $O(T \cdot L \cdot H^2)$ | $O(T^2 \cdot H)$ |
| **Memory** | $O(T \cdot H)$ | $O(T^2)$ |
| **RF** | $O(2^L) \approx 1024$ | Global $T$ |
| **Quality (BLEU)** | OK | Excellent |
| **Long context** | OK | Limited by memory |

**Use cases**:

- **TCN**: time series, audio (where local structure dominates)
- **Transformer**: NLP, vision (where global context matters)
- **Mamba**: long context (where both fail)

**Empirical**:
- WMT'14 En→Fr: Transformer +5 BLEU vs LSTM/TCN
- Image classification: ViT 이 ResNet 능가 (after pre-training)
- Speech recognition: hybrid (Conformer = Conv + Transformer)

**Lesson**:

1. **Architecture-task fit**:
   - Local structure: TCN
   - Global dependencies: Transformer
   - Long context: Mamba

2. **Inductive bias 의 가치**:
   - TCN 의 conv = locality bias
   - Transformer 의 attention = no bias
   - 적절한 bias 가 data 의 효율적 활용

3. **Hybrid 의 부상**:
   - Conformer (audio): conv + attention
   - ViT-based detection: conv backbone + attention head
   - Best of both worlds

**결론**: TCN 이 LSTM 의 자연스러운 alternative — convolution 의 parallelism. Transformer 이 attention 의 global RF 로 더 우월하지만 memory cost. *No single best* — task 와 scale 의 함수. $\square$

</details>

**문제 3** (논문 비평): WaveNet 이 raw audio generation 의 SOTA 였지만 이후 audio 분야에서도 Transformer-based (Whisper, Mu2) 가 dominate. 왜 conv-based 가 결국 자리를 잃었는가?

<details>
<summary>해설</summary>

**WaveNet 의 contributions**:
- Raw audio (16-22 kHz, 즉 16K-22K samples/sec) generation
- Causal dilated conv 로 receptive field 256ms
- TTS, music generation 의 처음 NN approach

**WaveNet 의 한계**:

1. **Receptive field 의 hard limit**:
   - Architecture 로 결정된 RF
   - Longer-range dependency (전체 문장의 context) 학습 어려움
   - Hierarchical conv (Tacotron + WaveNet) 가 partial solution

2. **Inference speed**:
   - Sample-by-sample autoregressive
   - 1 second audio: 22000 samples = 22000 forward passes
   - GPU 에서도 real-time 어려움
   - **Parallel WaveNet** (van den Oord 2017) 이 distillation 으로 완화

3. **Position awareness 부족**:
   - Conv 의 translation invariance 가 *항상* 좋지 않음
   - Speech 의 prosody, music 의 phrase structure 가 position-aware

**Modern audio architectures**:

1. **Whisper** (Radford 2022):
   - Audio Transformer
   - Encoder: log-mel spectrogram → Transformer
   - Decoder: transcript generation
   - Multilingual, multi-task
   - **Pre-training scale** (680K hours) 이 결정적

2. **AudioLM, MusicLM** (2023):
   - Discrete audio tokens (codec) + Transformer
   - In-context audio generation
   - Long-form coherent music

3. **Conformer**:
   - Conv + attention hybrid
   - Speech recognition SOTA (LibriSpeech)
   - Best of both: local conv + global attention

**왜 Transformer가 우월**:

1. **Pre-training paradigm**:
   - WaveNet: task-specific training
   - Transformer: large-scale pre-training + fine-tuning
   - Self-supervised audio (wav2vec 2.0, HuBERT)

2. **Multi-modal compatibility**:
   - Same architecture for audio, text, image
   - Cross-modal alignment (CLAP, Whisper)
   - WaveNet 은 audio-specific

3. **Scale**:
   - WaveNet: 4M parameters typical
   - AudioLM: 1B+ parameters
   - LLM scale 이 emergent capabilities

4. **Discrete tokenization**:
   - Continuous audio → discrete codec tokens
   - Standard Transformer applicable
   - Generation 이 token-level (faster)

**WaveNet 의 legacy**:

1. **Idea contribution**:
   - Causal dilated conv 의 *generalization* 가 modern audio model 에 흡수
   - HuBERT 의 conv encoder
   - Conformer 의 conv block

2. **Implementation 자체는 secondary**:
   - Raw audio sample-level 처리 less common
   - Discrete codec + Transformer 가 표준

**Lesson**:

1. **Architecture 의 부분적 흡수**:
   - WaveNet 의 conv 가 modern hybrid 에 흡수
   - Pure WaveNet 은 historical, 그러나 *idea* 가 alive

2. **Pre-training 의 importance**:
   - WaveNet 시대: task-specific
   - Modern: foundation model + fine-tune
   - Architecture 보다 training paradigm 이 더 결정적

3. **Multi-modal trend**:
   - Audio → text, image → text 가 standard
   - Single architecture (Transformer) 이 모든 modality
   - Audio-specific WaveNet 의 niche 줄어듦

**현대 (2024)**:
- WaveNet: historical, occasional research baseline
- AudioLM family: Transformer-based, in-context capable
- Conformer: speech recognition standard
- Diffusion (e.g., Diffwave): audio generation alternative

**결론**: WaveNet 이 *causal dilated conv* idea 로 audio generation revolutionize. 그러나 *Transformer + pre-training* paradigm shift 가 architecture 의 specific choice 를 less important. **Idea 는 alive, implementation 이 evolved**. ML 진화의 일반적 pattern. $\square$

</details>

---

<div align="center">

[◀ 이전](./01-parallelism-limit.md) | [📚 README](../README.md) | [다음 ▶](./03-linear-attention-rwkv.md)

</div>
