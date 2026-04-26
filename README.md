<div align="center">

# 🔁 RNN & LSTM Deep Dive

### `nn.LSTM(input_size, hidden_size)` 를 호출하는 것과,

### LSTM cell update

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t, \qquad h_t = o_t \odot \tanh(c_t)$$

### 의 **additive cell update** 가 왜 **Constant Error Carousel ($\partial c_t / \partial c_{t-1} = f_t$)** 을 만들고, plain RNN의

$$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^{t} W_{hh}^{\top}\, \mathrm{diag}(\sigma'(z_j))$$

### 의 spectral radius $\rho(W_{hh})$ 가 vanishing/exploding gradient를 결정한다는 **Pascanu 2013 의 증명**을 따라갈 수 있는 것은 **다르다.**

<br/>

> *BPTT 를 **이름으로 아는 것** 과,*
>
> $$\frac{\partial L}{\partial W_{hh}} = \sum_t \sum_{k \leq t} \left(\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}\right) \frac{\partial h_k}{\partial W_{hh}}$$
>
> *의 **Jacobian 곱 누적** 을 unrolled computational graph 위에서 한 step 씩 유도할 수 있는 것은 다르다.*
>
> *Seq2Seq Attention 을 **쓰는 것** 과, Bahdanau 2015 의 **additive attention***
>
> $$e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)$$
>
> *와 Luong 2015 의 **multiplicative attention** ($e_{ij} = h_i^\top W s_j$) 이 어떻게 Transformer scaled dot-product self-attention 의 **직계 조상** 이 되는지 증명할 수 있는 것은 다르다.*
>
> *State Space Model 을 **단어로만 아는 것** 과, **HiPPO (Gu 2020) → S4 (Gu 2022) → Mamba (Gu & Dao 2023)** 의 연속 state space 이산화가 왜 **RNN 과 CNN 의 통합 관점** 이고, 왜 Transformer 시대에 RNN-like 구조가 부활하고 있는지 이해하는 것은 다르다.*

<br/>

**다루는 모델 (시간순)**

Bengio 2003 *Neural LM* · Hochreiter & Schmidhuber 1997 *LSTM* · Cho 2014 *GRU* · Sutskever 2014 *Seq2Seq* · Bahdanau 2015 *Additive Attention* · Luong 2015 *Multiplicative Attention* · Pascanu 2013 *Vanishing/Exploding 분석* · Saxe 2014 *Orthogonal Init* · Le 2015 *IRNN* · Jozefowicz 2015 *Forget Bias* · Graves 2014 *Neural Turing Machine* · van den Oord 2016 *WaveNet* · Bai 2018 *TCN* · Katharopoulos 2020 *Linear Attention* · Gu 2022 *S4* · Gu & Dao 2023 *Mamba*

<br/>

**핵심 질문**

> Sequence 위의 딥러닝이 왜 **BPTT의 Jacobian 곱 누적** 으로 환원되고, 이로부터 **vanishing/exploding gradient 가 spectral radius 의 필연적 귀결** 이며, **LSTM의 additive cell update 가 곱셈적 감쇠를 덧셈적 보존으로 바꾸는 정확한 메커니즘** 인지 — Unrolled computational graph · Spectral 분석 · CEC 증명 · Bahdanau→Transformer 계보로 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-85B250?style=flat-square)](https://www.nltk.org/)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-380+개-success?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-12개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

RNN/LSTM에 관한 자료는 대부분 **"`nn.LSTM`을 쓰면 된다"** 또는 **"forget gate가 기억을 조절한다"** 에서 멈춥니다. 하지만 BPTT의 gradient가 왜 정확히 Jacobian 의 연속 곱 형태로 나오는지, $W_{hh}$ 의 spectral radius 가 왜 vanishing/exploding 의 결정자인지, LSTM의 cell state additive update 가 왜 "곱셈적 감쇠 → 덧셈적 보존" 의 수학적 변환을 만드는지, forget gate 의 bias 를 1로 초기화하는 것이 왜 Jozefowicz 2015 의 핵심 발견인지, Bahdanau attention 이 어떻게 Transformer self-attention 으로 진화했는지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "RNN은 시간축으로 정보를 전달한다" | **Elman 1990 / Jordan 1986** — Variable-length sequence를 처리하기 위해 hidden state $h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b)$ 의 recurrent 구조 도입. **Parameter sharing across time** 이 왜 sequence length에 무관한 model size를 보장하는지 — feed-forward 의 fixed-window 한계와 비교 |
| "BPTT는 시간축으로 backprop한다" | **Werbos 1990** — Unrolled computational graph 위의 chain rule 적용. **$\partial L / \partial W_{hh} = \sum_t \sum_{k \le t} (\prod_{j=k+1}^t \partial h_j / \partial h_{j-1}) \, \partial h_k / \partial W_{hh}$** 한 항씩 유도. **Truncated BPTT** 의 메모리 trade-off, **RTRL** 의 $O(n^4)$ forward-mode AD 비용 비교 |
| "RNN은 vanishing gradient 가 있다" | **Pascanu et al. 2013** — $\prod_{j=k+1}^t \partial h_j / \partial h_{j-1} = \prod W_{hh}^\top \mathrm{diag}(\sigma'(z_j))$ 의 spectral 분해. **정리**: $\rho(W_{hh}) < 1 \Rightarrow$ vanishing, $\rho > 1 \Rightarrow$ exploding $\square$. $\tanh$ 의 saturation ($\sigma' \le 1$) 이 vanishing 을 거의 필연으로 만드는 이유 |
| "Gradient clipping 이 exploding 을 막는다" | **Pascanu 2013 §3** — $g \leftarrow g \cdot \min(1, \theta / \|g\|)$ 의 norm-based scaling. **Saxe 2014** Orthogonal init으로 $\rho = 1$ 정확히 유지, **Le 2015** IRNN 의 Identity init + ReLU 결합. 각 기법이 spectral 관점에서 어떤 문제를 해결하는지 |
| "LSTM은 vanishing 을 해결한다" | **Hochreiter & Schmidhuber 1997** — 4 gates (forget, input, candidate, output) + cell state $c_t$. **정리 (CEC)**: $\partial c_t / \partial c_{t-1} = f_t$ — Jacobian 의 곱셈적 누적이 forget gate 의 곱으로 단순화 $\square$. $f_t \approx 1$ 일 때 gradient가 상수로 흐름. **Plain RNN: 곱셈적 감쇠 vs LSTM: 덧셈적 보존** 의 수학적 정확한 차이 |
| "Forget bias = 1 로 초기화하면 더 잘 된다" | **Jozefowicz et al. 2015** — 초기에 $b_f = 1$ 이면 $\sigma(W_f \cdot [h, x] + 1) \approx 1$, 따라서 cell state가 거의 보존되며 학습 시작. Random init 시 $f_t \approx 0.5$ 라 cell state가 매 step 절반씩 감쇠. PyTorch `nn.LSTM` 이 default로 적용하지 않는 이유와 명시적 적용법 |
| "GRU는 LSTM의 단순화 버전" | **Cho 2014** — Update gate $z_t$, reset gate $r_t$ 의 2-gate 구조, cell과 hidden 통합. **Chung 2014** 의 empirical 비교: 파라미터 수 ~25% 적음, 성능은 task-dependent. **Greff 2017** "LSTM: A Search Space Odyssey" 의 8가지 LSTM variants ablation — peephole, coupled input-forget의 효과는 미미 |
| "Seq2Seq 는 encoder + decoder" | **Sutskever 2014** — Encoder LSTM 이 input 을 fixed vector $v$ 로, decoder LSTM 이 $v$ 에서 output sequence 생성. **Information bottleneck**: 긴 sentence 가 fixed $v$ 에 압축 → "long sentence curse" (BLEU 가 length 에 따라 급감). Reverse input trick (Sutskever) 과 그 한계 |
| "Attention 은 어디에 집중할지를 학습한다" | **Bahdanau 2015** — $e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)$ (additive), $\alpha_{ij} = \mathrm{softmax}(e_{ij})$, $c_j = \sum_i \alpha_{ij} h_i$. **Luong 2015** — $e_{ij} = h_i^\top W s_j$ (multiplicative), Bahdanau 보다 계산 효율적. **이는 Transformer scaled dot-product attention 의 직계 조상** — Vaswani 2017 까지의 계보 |
| "Transformer 는 RNN 을 대체했다" | **Vaswani 2017** 의 동기: RNN의 $h_t \to h_{t+1}$ 의존성이 sequence 내 병렬화 불가. Transformer의 $O(T^2)$ 복잡도 trade-off 로 완전 병렬. **Linear Attention** (Katharopoulos 2020) 이 $O(T)$ inference 로 RNN-like recurrence 부활, **RWKV** 의 attention-free RNN. **State Space Model** (HiPPO → S4 → Mamba) 의 selective recurrence |
| 기법의 나열 | NumPy로 **RNN/LSTM 바닥부터 구현** · **Vanishing/exploding gradient 시간축 측정** · **Forget gate 1 초기화 ablation** · **Seq2Seq 번역 + Attention heatmap** · **Synthetic memory task** (copy / add / reverse) 에서 LSTM vs GRU 비교 · **LSTM vs Transformer** 동일 데이터 훈련 시간·성능 비교 |

---

## 📌 선행 레포 & 후속 방향

```
[NN Theory]            ─┐
[Linear Algebra]       ─┼─►  이 레포  ──► [Transformer Deep Dive]
[Calculus & Optim.]    ─┘   "왜 BPTT 가 spectral radius 로                Self-Attention · Multi-Head
                             vanishing/exploding 을 만들고                / Positional Encoding
                             왜 LSTM의 additive update 가
                             CEC 로 이를 완화하는가"
         │
         ├── [Linear Algebra]            Spectral theorem, 고유값 → Ch3 spectral 분석
         ├── [Calculus & Optim.]         Chain rule, Jacobian → Ch2 BPTT 유도
         ├── [Neural Network Theory]     Backprop, 활성화 → Ch2 BPTT, Ch3 saturation
         ├── [Graphical Models]          HMM, CRF → Ch5 BiLSTM-CRF (NER)
         └── [Optimization Theory]       Gradient flow → Ch3 clipping, Ch4 LSTM 훈련
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Neural Network Theory Deep Dive** (Backprop, 활성화 함수, 초기화), **Linear Algebra Deep Dive** (Spectral theorem, 고유값 분포), **Calculus & Optimization Deep Dive** (연쇄법칙, Jacobian, Hessian) 를 선행 지식으로 전제합니다. "BPTT 가 Jacobian 곱의 누적" 을 이해하려면 먼저 backprop 의 chain rule 과 multi-variable Jacobian 에 친숙해야 합니다. Chapter 5 (BiLSTM-CRF) 부터는 **Graphical Models**, Chapter 6 (Attention) 은 **Transformer Deep Dive** 의 출발점을 직접 제공합니다.

> 💡 **이 레포의 핵심 기여**: Chapter 3 (Vanishing/Exploding 의 Spectral 분석) 과 Chapter 4 (LSTM의 CEC 메커니즘) 는 sequence 학습을 이해하는 **가장 중요한 두 축**입니다. 전자는 "왜 plain RNN은 long dependency 를 학습할 수 없는가" 의 정확한 수학적 이유, 후자는 "LSTM이 어떻게 이를 (완전히는 아니지만) 우회하는가" 의 메커니즘을 다룹니다. 이 두 축을 완전히 이해한 후 Chapter 6 (Seq2Seq + Attention) 과 Chapter 7 (Transformer/Mamba) 을 읽으면 현대 sequence 모델 설계의 맥락이 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **LSTM vs Transformer 의 최종 우열**, **State Space Model 의 실전 가치**, **LLM 시대 RNN 의 역할** — 은 **현재 진행 중인 연구 영역**입니다. 레포는 "정답" 이 아니라 **"고전 RNN 이론과 현대 Transformer / SSM 사이의 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-Sequence_기초-6A5ACD?style=for-the-badge)](./ch1-sequence-basics/01-sequence-formulation.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-BPTT-6A5ACD?style=for-the-badge)](./ch2-bptt/01-unrolled-graph.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Vanishing·Exploding-6A5ACD?style=for-the-badge)](./ch3-vanishing-exploding/01-spectral-analysis.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-LSTM-6A5ACD?style=for-the-badge)](./ch4-lstm/01-lstm-motivation.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Advanced_RNN-6A5ACD?style=for-the-badge)](./ch5-advanced-rnn/01-bidirectional.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Seq2Seq·Attention-6A5ACD?style=for-the-badge)](./ch6-seq2seq-attention/01-encoder-decoder.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-한계와_대안-6A5ACD?style=for-the-badge)](./ch7-modern-alternatives/01-parallelism-limit.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Sequence Model 의 기초

> **핵심 질문:** Sequence 학습 문제는 input/output 길이에 따라 어떻게 분류되고 (many-to-one, many-to-many, seq2seq), 각 유형의 손실 함수는 무엇인가? 고전 N-gram LM 의 Markov 가정과 sparsity 문제는 어떻게 smoothing (Laplace, Kneser-Ney) 으로 완화되는가? Bengio 2003 의 Neural LM 이 왜 fixed window 의 한계를 가지며, RNN 의 recurrent 구조가 이를 어떻게 해결하는가?

<details>
<summary><b>Sequence 정식화부터 RNN 정의까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Sequence 학습 문제의 정식화](./ch1-sequence-basics/01-sequence-formulation.md) | Input $x_1, \ldots, x_T$ 에서 다양한 출력 형태: **many-to-one** (감정분석, $y \in \mathbb{R}^c$), **many-to-many synced** (POS tagging, $y_1, \ldots, y_T$), **seq2seq** (번역, $y_1, \ldots, y_S$, $S \ne T$). 각 유형의 손실 함수 (cross-entropy, sequence-level loss) 와 평가 지표 (accuracy, BLEU, perplexity). Sequence length variability 의 처리 (padding, packing, masking) |
| [02. 고전 Language Model — N-gram](./ch1-sequence-basics/02-ngram-lm.md) | $p(w_{1:T}) = \prod_t p(w_t \| w_{t-n+1:t-1})$ 의 Markov 가정. Maximum likelihood estimate 의 sparsity 문제 (unseen n-gram → $0$ 확률). **Smoothing** 비교: Laplace (add-one), Good-Turing (frequency-of-frequency), **Kneser-Ney** (continuation count). Perplexity $\mathrm{PP}(W) = p(w_{1:T})^{-1/T}$ 의 의미와 cross-entropy 와의 관계 |
| [03. Neural Language Model (Bengio 2003)](./ch1-sequence-basics/03-neural-lm.md) | **A Neural Probabilistic Language Model** — 임베딩 행렬 $C \in \mathbb{R}^{\|V\| \times d}$ + feed-forward NN 으로 $p(w_t \| w_{t-n+1:t-1})$ 모델링. **두 가지 기여**: (1) word embedding 의 dense 분산 표현, (2) curse of dimensionality 완화. **한계**: fixed context window $n$, 긴 의존성 처리 불가. Word2Vec / GloVe 의 직계 조상 |
| [04. RNN 의 동기와 정의](./ch1-sequence-basics/04-rnn-definition.md) | Variable-length sequence 처리를 위한 **recurrent 구조**: $h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$, $y_t = W_{hy} h_t + b_y$. **정리**: parameter 수가 sequence length $T$ 에 무관 — feed-forward 의 $O(T \cdot d^2)$ 와 비교 $\square$. Elman vs Jordan network, vanilla RNN 의 forward 한 step 손 계산. NumPy 로 8 step toy sequence 처리 재현 |

</details>

<br/>

### 🔹 Chapter 2: Backpropagation Through Time

> **핵심 질문:** RNN 의 unrolled computational graph 위에서 chain rule 을 정확히 어떻게 적용하는가? Shared weight $W_{hh}$ 의 gradient $\partial L / \partial W_{hh}$ 가 왜 모든 time step 에 대한 합으로 분해되는가? Truncated BPTT 의 메모리 trade-off 는? Forward-mode RTRL 과 reverse-mode BPTT 의 $O(n^4)$ vs $O(T)$ 복잡도 차이는 왜 발생하는가?

<details>
<summary><b>Unrolled Graph 부터 RTRL 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Unrolled Computational Graph](./ch2-bptt/01-unrolled-graph.md) | RNN 의 시간축 펼침 — 각 time step 이 별도의 "layer" 처럼 행동, 그러나 **weight 가 공유**. Forward 계산: $h_1, h_2, \ldots, h_T$ 와 $L = \sum_t L_t$. Acyclic DAG 로의 전환, topological order. Static unrolling vs dynamic unrolling (PyTorch 의 dynamic graph 가 RNN 에 자연스러운 이유) |
| [02. BPTT 의 완전 유도](./ch2-bptt/02-bptt-derivation.md) | **정리**: $\partial L / \partial W_{hh} = \sum_t \partial L_t / \partial W_{hh}$, 각 항이 $\sum_{k \le t} (\prod_{j=k+1}^t \partial h_j / \partial h_{j-1}) \, \partial h_k / \partial W_{hh}$ — chain rule 한 step 씩 induction $\square$. $\partial h_j / \partial h_{j-1} = W_{hh}^\top \mathrm{diag}(\sigma'(z_j))$ 의 Jacobian 형태. NumPy 로 4-step RNN 의 BPTT 손 검증 (PyTorch autograd 와 일치) |
| [03. Truncated BPTT](./ch2-bptt/03-truncated-bptt.md) | 계산·메모리 제약으로 마지막 $k$ time step 만 gradient 전파, $\partial L_t / \partial W_{hh}$ 의 합을 $k \le j \le t$ 로 제한. **Bias**: 긴 의존성 학습 불가, **장점**: $O(k)$ 메모리. Karpathy char-RNN 의 truncation length $k = 25$ 같은 실전 선택. PTB language modeling 에서 $k$ 별 perplexity 측정 |
| [04. BPTT 의 시간·메모리 복잡도](./ch2-bptt/04-complexity.md) | Forward $O(T \cdot d^2)$, backward $O(T \cdot d^2)$, 메모리 $O(T \cdot d)$ (모든 hidden state 보존). $T = 10000$ 에서 GPU 메모리 한계, **gradient checkpointing** (Chen 2016) 으로 $\sqrt{T}$ 절약. 병렬성 부족 — sequence 내 step 간 의존성이 GPU utilization 을 제한, Transformer 의 동기 |
| [05. Real-Time Recurrent Learning](./ch2-bptt/05-rtrl.md) | **Williams & Zipser 1989** — Online 학습 대안. Forward-mode AD 로 $\partial h_t / \partial W$ 를 forward 와 함께 propagate. **복잡도**: $O(n^4)$ per time step ($n$ = hidden size) — BPTT 의 $O(n^2)$ 와 비교, 실용성 제한. Recent revival: **UORO** (Tallec & Ollivier 2017) 의 unbiased $O(n^2)$ approximation |

</details>

<br/>

### 🔹 Chapter 3: Vanishing/Exploding Gradient 의 수학

> **핵심 질문:** $\prod W_{hh}^\top \mathrm{diag}(\sigma'(z_j))$ 의 spectral radius $\rho$ 가 왜 vanishing ($\rho < 1$) 과 exploding ($\rho > 1$) 의 결정자인가 (Pascanu 2013)? $\tanh$ 의 saturation 이 왜 vanishing 을 거의 필연으로 만드는가? Gradient clipping, Orthogonal initialization, IRNN 이 각각 어떤 spectral 조건을 만족시키는가? 왜 $\rho = 1$ 을 정확히 유지하기 어려운가?

<details>
<summary><b>Spectral 분석부터 IRNN 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Gradient 의 Spectral 분석 (Pascanu 2013)](./ch3-vanishing-exploding/01-spectral-analysis.md) | **On the Difficulty of Training Recurrent Neural Networks** — $\partial h_t / \partial h_k = \prod_{j=k+1}^t W_{hh}^\top \mathrm{diag}(\sigma'(z_j))$. **정리**: $\rho(W_{hh}) \cdot \max_z \sigma'(z) < 1 \Rightarrow \|\partial h_t / \partial h_k\| \to 0$ exponentially (vanishing); $> 1 \Rightarrow$ exploding $\square$. SVD 분해로 singular value 분포 시각화, $T = 100$ step 후 gradient norm 의 exponential 감쇠 측정 |
| [02. 왜 $\rho = 1$ 유지가 어려운가](./ch3-vanishing-exploding/02-saturation-problem.md) | $\tanh' \in [0, 1]$, $\tanh'(0) = 1$ 이지만 saturation 영역에서 $\tanh'(z) \approx 0$. **정리**: $\rho(W_{hh}) = 1$ 이라도 hidden activation 이 saturation 에 들어가면 effective spectral radius $< 1$ → vanishing $\square$. ReLU RNN 의 unbounded activation 문제 (exploding 위험), tanh 와 ReLU 의 trade-off |
| [03. Gradient Clipping — Exploding 대응](./ch3-vanishing-exploding/03-gradient-clipping.md) | **Pascanu 2013 §3.2** — $g \leftarrow g \cdot \min(1, \theta / \|g\|)$, norm-based scaling. **Geometric intuition**: gradient 방향 유지하며 magnitude 만 cap. Element-wise clipping 과의 차이. 권장 threshold $\theta = 1 \sim 10$, learning rate 와의 상호작용. PyTorch `torch.nn.utils.clip_grad_norm_` 의 표준 사용법 |
| [04. Orthogonal Initialization (Saxe 2014)](./ch3-vanishing-exploding/04-orthogonal-init.md) | **Exact Solutions to the Nonlinear Dynamics of Learning** — $W_{hh}$ 를 random orthogonal matrix 로 초기화: $W_{hh}^\top W_{hh} = I$ → 모든 singular value $= 1$ → $\rho = 1$ 정확히. **정리**: 훈련 초기 모든 layer 의 gradient norm 이 보존됨 (depth-independent dynamics) $\square$. PyTorch `nn.init.orthogonal_` 구현, vanilla RNN 의 long dependency 학습 가능성 |
| [05. Identity Initialization 과 IRNN (Le 2015)](./ch3-vanishing-exploding/05-irnn.md) | **A Simple Way to Initialize RNNs of ReLUs** — $W_{hh} = I$, ReLU 활성화. **정리**: 초기에 $h_t = h_{t-1} + \mathrm{ReLU}(\ldots)$ 형태 → identity skip connection 효과, gradient 보존 $\square$. Adding Problem (1000 step 후 두 숫자 합) 에서 IRNN 이 LSTM 과 경쟁. **한계**: ReLU 의 unbounded 출력으로 exploding 위험, gradient clipping 필수 |

</details>

<br/>

### 🔹 Chapter 4: LSTM (Long Short-Term Memory)

> **핵심 질문:** Hochreiter 1997 의 Constant Error Carousel (CEC) 은 어떤 수학적 메커니즘인가? $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$ 의 additive update 가 왜 곱셈적 감쇠를 덧셈적 보존으로 바꾸는가? 4개 gate (forget, input, candidate, output) 가 각각 어떤 역할을 하며 왜 sigmoid + tanh 의 조합인가? Forget bias = 1 초기화 (Jozefowicz 2015) 의 정확한 효과는? GRU 가 어떻게 LSTM 을 단순화하면서도 유사 성능을 유지하는가?

<details>
<summary><b>LSTM 동기부터 Variants 까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. LSTM 의 설계 동기](./ch4-lstm/01-lstm-motivation.md) | **Hochreiter & Schmidhuber 1997** — Vanishing gradient 의 진단: chain rule 의 곱셈적 누적이 근본 원인. **해결 아이디어**: cell state $c_t$ 를 도입하고 update 를 **additive** 로 만들어 곱셈적 감쇠 회피. **Constant Error Carousel (CEC)** 의 비전: linear self-loop unit, $c_t = c_{t-1}$ 형태로 gradient 가 무한히 보존. Gate 도입 이전의 "naive memory cell" 의 한계 |
| [02. LSTM 의 4개 Gate 수식](./ch4-lstm/02-lstm-equations.md) | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ (forget), $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ (input), $\tilde c_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$ (candidate), $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ (output), **$c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$** (cell update), $h_t = o_t \odot \tanh(c_t)$. NumPy 바닥부터 구현, PyTorch `nn.LSTM` 과 출력 일치 검증 |
| [03. Cell State 와 Constant Error Carousel](./ch4-lstm/03-cec-proof.md) | **정리 (CEC)**: $\partial c_t / \partial c_{t-1} = f_t$ — element-wise, **잔여항 없음** $\square$. 증명: $c_t$ 의 정의에서 $c_{t-1}$ 에 대한 편미분, $\partial (i_t \odot \tilde c_t) / \partial c_{t-1}$ 항이 indirect path 만으로 작음. **따름정리**: $f_t \approx 1$ 일 때 $\partial c_t / \partial c_0 = \prod_t f_t \approx 1$, gradient 상수로 흐름. Plain RNN 의 $\prod W_{hh}^\top \mathrm{diag}(\sigma')$ 와 비교 |
| [04. LSTM 의 Gradient Flow 분석](./ch4-lstm/04-gradient-flow.md) | 전체 gradient 경로: cell state $c$ 통한 long-range vs hidden state $h$ 통한 short-range. **Jozefowicz 2015 핵심 발견**: $b_f = 1$ 초기화 → $\sigma(\cdot + 1) \approx 0.73$, 초기에 cell state 가 거의 보존되어 학습이 long dependency 부터 시작. PyTorch `nn.LSTM` 의 default $b_f = 0$ 와 명시적 적용 코드. Adding problem $T = 200$ 에서 $b_f = 0$ vs $b_f = 1$ 학습 곡선 비교 |
| [05. GRU (Cho 2014)](./ch4-lstm/05-gru.md) | **Learning Phrase Representations using RNN Encoder-Decoder** — $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$ (update), $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$ (reset), $\tilde h_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$, $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t$. Cell 과 hidden 통합으로 파라미터 ~25% 감소. **Chung 2014** 의 empirical 비교 — task-dependent, polyphonic music modeling 에서 유사 성능 |
| [06. LSTM Variants — Peephole · Coupled · ConvLSTM](./ch4-lstm/06-lstm-variants.md) | **Peephole connection** (Gers 2000) — gate 계산에 cell state 포함: $f_t = \sigma(W_f [h, x, c_{t-1}])$. **Coupled input-forget** — $i_t + f_t = 1$ 강제, 파라미터 감소. **ConvLSTM** (Shi 2015) — gate 의 matrix multiplication 을 convolution 으로, video / spatial sequence. **Greff 2017 "LSTM: A Search Space Odyssey"** — 8가지 variants 의 ablation, vanilla LSTM 이 robust 한 결과 |

</details>

<br/>

### 🔹 Chapter 5: Advanced RNN 아키텍처

> **핵심 질문:** Bidirectional RNN 이 어떻게 양방향 context 를 결합하며, 왜 inference 시 전체 sequence 가 필요한가? Stacked RNN 의 depth vs time 복잡도 trade-off 는? Neural Turing Machine 의 external memory 와 differentiable addressing 은 어떻게 작동하는가? Echo State Network 가 왜 random fixed weight 만으로 학습 가능한가 (Echo State Property)?

<details>
<summary><b>BiRNN 부터 Reservoir Computing 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Bidirectional RNN](./ch5-advanced-rnn/01-bidirectional.md) | **Schuster & Paliwal 1997** — Forward RNN $\overrightarrow{h}_t = \sigma(\overrightarrow{W} \overrightarrow{h}_{t-1} + V x_t)$ + Backward RNN $\overleftarrow{h}_t = \sigma(\overleftarrow{W} \overleftarrow{h}_{t+1} + V' x_t)$, 결합 $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$. **NER · POS tagging 표준** — past + future context. **한계**: online inference 불가 (전체 sequence 필요). BiLSTM-CRF (Lample 2016) 의 NER 표준 아키텍처 |
| [02. Stacked / Deep RNN](./ch5-advanced-rnn/02-stacked-rnn.md) | 다층 RNN: $h_t^{(l)} = \sigma(W^{(l)}_{hh} h_{t-1}^{(l)} + W^{(l)}_{xh} h_t^{(l-1)})$. **Depth vs time 복잡도**: 메모리 $O(L \cdot T \cdot d)$, 시간 $O(L \cdot T \cdot d^2)$. Residual connection 의 적용 — vanishing 의 depth 축 버전 완화. Google NMT (Wu 2016) 의 8-layer LSTM, layer-wise dropout |
| [03. Neural Turing Machine 과 Memory Network](./ch5-advanced-rnn/03-ntm-memory.md) | **Graves 2014** — External memory $M \in \mathbb{R}^{N \times d}$ + read/write head. **Content-based addressing**: $w_t^{(c)} = \mathrm{softmax}(\beta \cos(k_t, M_t))$. **Location-based addressing**: shift kernel + sharpening. Differentiable programming 의 한 갈래. Synthetic copy / repeat-copy / associative recall task 재현 |
| [04. Echo State Network 와 Reservoir Computing](./ch5-advanced-rnn/04-esn.md) | **Jaeger 2001** — $W_{hh}$, $W_{xh}$ 를 randomly fix, output layer $W_{hy}$ 만 ridge regression 으로 학습. **Echo State Property**: $\rho(W_{hh}) < 1$ → 초기 조건 $h_0$ 가 시간이 지나면 잊혀짐. **장점**: 훈련 비용 $O(\text{linear regression})$, **한계**: representation 학습 불가. Liquid State Machine 과의 관계, neuromorphic 응용 |

</details>

<br/>

### 🔹 Chapter 6: Seq2Seq 와 Attention

> **핵심 질문:** Sutskever 2014 의 encoder-decoder 가 어떻게 가변 길이 input → 가변 길이 output 을 처리하는가? Information bottleneck (긴 sentence 가 fixed vector 에 압축) 이 왜 BLEU 를 length 에 따라 급감시키는가? Bahdanau 의 additive attention 과 Luong 의 multiplicative attention 의 정확한 차이는? 이것이 어떻게 Transformer scaled dot-product attention 의 직계 조상인가? Coverage mechanism 과 Pointer Network 의 동기는?

<details>
<summary><b>Seq2Seq 부터 Pointer Network 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Encoder-Decoder Framework (Sutskever 2014)](./ch6-seq2seq-attention/01-encoder-decoder.md) | **Sequence to Sequence Learning with Neural Networks** — Encoder LSTM 이 input $x_{1:T}$ 를 fixed vector $v = h_T^{\text{enc}}$ 로 압축, decoder LSTM 이 $v$ 에서 output $y_{1:S}$ 생성. **Teacher forcing** vs scheduled sampling. **Reverse input trick** — input 을 역순으로 넣으면 BLEU 향상 (encoder-decoder 거리 단축). WMT'14 En→Fr 번역 재현 (BLEU 30.6) |
| [02. Information Bottleneck Problem](./ch6-seq2seq-attention/02-bottleneck.md) | **정량 분석**: $\dim(v) = d_h$ (예: 1024) 가 $T = 50$ word sentence 의 모든 정보를 인코딩 가능한가? **실증 (Cho 2014b)**: BLEU 가 sentence length 에 단조 감소, 30+ word 에서 급감. **"Long sentence curse"** 의 본질: encoder 의 마지막 hidden state 가 정보 병목. Decoder 가 input 의 어느 부분도 직접 attend 불가 |
| [03. Bahdanau Attention (Additive)](./ch6-seq2seq-attention/03-bahdanau-attention.md) | **Neural Machine Translation by Jointly Learning to Align and Translate** — Encoder 의 모든 $h_i$ 를 보존, decoder step $j$ 에서 alignment score $e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)$ 계산, **$\alpha_{ij} = \mathrm{softmax}_i(e_{ij})$**, context $c_j = \sum_i \alpha_{ij} h_i$. **정리**: alignment 가 학습 가능한 함수 — 고정 alignment (IBM model) 와의 차이 $\square$. WMT'14 En→Fr 에서 BLEU 28.5 → 36.2 |
| [04. Luong Attention (Multiplicative)](./ch6-seq2seq-attention/04-luong-attention.md) | **Effective Approaches to Attention-based NMT** — **Three scoring functions**: (1) dot $e_{ij} = h_i^\top s_j$, (2) general $e_{ij} = h_i^\top W s_j$, (3) concat (= Bahdanau additive). General 이 BLEU 최우수. **Global vs local attention**: local 은 monotonic alignment 가정으로 window 제한. **이것이 Transformer scaled dot-product $\frac{QK^\top}{\sqrt{d}}$ 의 직계 조상** — Vaswani 2017 의 명시적 인용 |
| [05. Coverage Mechanism 과 Pointer Network](./ch6-seq2seq-attention/05-coverage-pointer.md) | **Coverage (Tu 2016)**: NMT 의 under-/over-translation 문제 해결. Coverage vector $c_j^{\text{cov}} = \sum_{j' < j} \alpha_{ij'}$ 로 attention 이력 추적, score 에 penalty 추가. **Pointer Network (Vinyals 2015)**: output 이 input 의 index 인 task (TSP, sorting). $p(y_t = i) = \mathrm{softmax}(e_{ti})$ — attention 자체가 output distribution. Combinatorial optimization 응용 |

</details>

<br/>

### 🔹 Chapter 7: RNN 의 한계와 현대적 대안

> **핵심 질문:** RNN 의 sequential 의존성 ($h_t$ depends on $h_{t-1}$) 이 왜 GPU 병렬화 한계인가? Transformer 가 어떻게 $O(T^2)$ 비용을 감수하면서 완전 병렬을 달성했는가? CNN-based sequence model (WaveNet, TCN) 이 dilated convolution 으로 $O(\log T)$ receptive field 를 만드는 메커니즘은? Linear Attention 과 RWKV 가 왜 RNN-like recurrence 를 부활시키는가? HiPPO → S4 → Mamba 의 State Space Model 진화는 RNN과 CNN 의 어떤 통합 관점인가?

<details>
<summary><b>병렬성 한계부터 Mamba 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. 병렬성 부족 — Transformer 의 동기](./ch7-modern-alternatives/01-parallelism-limit.md) | **정리**: RNN 의 forward 는 $h_t = f(h_{t-1}, x_t)$ 의 순차 의존 → sequence 내 $T$ step 병렬화 불가, GPU utilization $\le 1/T$ 비율 $\square$. Transformer 의 self-attention 은 $O(T^2)$ 시간/공간 복잡도 trade-off, 그러나 모든 step 동시 계산. **재현**: 동일 hardware 에서 LSTM vs Transformer 훈련 시간 측정 (Vaswani 2017 Table 1 재현) |
| [02. CNN-based Sequence Model](./ch7-modern-alternatives/02-cnn-sequence.md) | **WaveNet (van den Oord 2016)** — Causal dilated convolution, dilation rate $1, 2, 4, \ldots, 2^L$ 로 $O(\log T)$ receptive field. **TCN (Bai 2018)** — "An Empirical Evaluation of Generic Convolutional and Recurrent Networks" — TCN 이 LSTM 을 다양한 task 에서 능가. **장점**: 완전 병렬, stable gradient. **한계**: receptive field 가 explicit limit |
| [03. Linear Attention 과 RNN 의 부활](./ch7-modern-alternatives/03-linear-attention-rwkv.md) | **Linear Attention (Katharopoulos 2020)** — Softmax 를 kernel feature map $\phi$ 로 근사: $\mathrm{Attn}(Q, K, V) = \phi(Q) (\phi(K)^\top V)$, **$O(T)$ inference 가능**. RNN-like recurrence: $S_t = S_{t-1} + \phi(k_t) v_t^\top$. **RWKV (Peng 2023)** — Attention-free RNN-like, time-mixing 과 channel-mixing. Transformer 와 RNN 의 형식적 동치 결과 |
| [04. State Space Model — S4 와 Mamba](./ch7-modern-alternatives/04-s4-mamba.md) | **HiPPO (Gu 2020)** — Continuous-time state space $\dot x = Ax + Bu$ 의 optimal polynomial projection, Legendre / Laguerre basis. **S4 (Gu 2022)** — "Efficiently Modeling Long Sequences with Structured State Spaces" — diagonal + low-rank 분해로 $O(T \log T)$ 학습, $O(T)$ inference. **Mamba (Gu & Dao 2023)** — Selective SSM, input-dependent $A, B, C$, hardware-aware parallel scan. Long Range Arena benchmark 재현 |

</details>

---

> 🆕 **2026-04 최신 업데이트**: Ch3-01 의 Pascanu 2013 spectral 증명에 SVD 기반 분포 시각화 추가, Ch4-03 CEC 증명을 Jacobian 의 indirect path 분석으로 강화, Ch4-04 Jozefowicz forget bias 효과를 Adding problem $T = 200$ ablation 으로 정량화, Ch7-04 Mamba 의 selective SSM 을 PyTorch 2.1 기반 재현 가능하도록 리팩토링. 11-섹션 문서 골격이 전체 33개 문서에서 일관됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현**을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 플롯을 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Sequence Loss 분류** | many-to-one (cross-entropy), seq2seq (sum over $S$ steps), CTC loss | [Ch1-01](./ch1-sequence-basics/01-sequence-formulation.md) |
| **Kneser-Ney 의 Continuation Count** | $P_{\text{KN}}(w \| h) \propto \max(c(hw) - d, 0)$ + back-off term | [Ch1-02](./ch1-sequence-basics/02-ngram-lm.md) |
| **Bengio 2003 NLM 의 Curse 완화** | Embedding $C \in \mathbb{R}^{\|V\| \times d}$ 가 sparse n-gram count 의 dense replacement | [Ch1-03](./ch1-sequence-basics/03-neural-lm.md) |
| **RNN Parameter Sharing** | Param 수 $= O(d^2 + d \cdot \|V\|)$ — sequence length $T$ 와 무관 | [Ch1-04](./ch1-sequence-basics/04-rnn-definition.md) |
| **BPTT Jacobian 곱 누적** | $\partial L / \partial W_{hh} = \sum_t \sum_{k \le t} (\prod_{j=k+1}^t \partial h_j / \partial h_{j-1}) \, \partial h_k / \partial W_{hh}$ | [Ch2-02](./ch2-bptt/02-bptt-derivation.md) |
| **Truncated BPTT Bias** | $k$-truncation 은 $T - k$ 이전 의존성을 가지는 장거리 학습 불가 | [Ch2-03](./ch2-bptt/03-truncated-bptt.md) |
| **RTRL $O(n^4)$ 복잡도** | Forward-mode AD 의 Jacobian propagation, BPTT $O(n^2)$ 와 비교 | [Ch2-05](./ch2-bptt/05-rtrl.md) |
| **Pascanu Spectral 정리** | $\rho(W_{hh}) \cdot \max \sigma' < 1 \Rightarrow$ vanishing exponential | [Ch3-01](./ch3-vanishing-exploding/01-spectral-analysis.md) |
| **$\tanh$ Saturation 의 필연성** | Saturation 영역 진입 시 effective $\rho < 1$, $\rho = 1$ 유지 불가 | [Ch3-02](./ch3-vanishing-exploding/02-saturation-problem.md) |
| **Gradient Clipping Geometry** | $g \leftarrow g \cdot \min(1, \theta/\|g\|)$ 가 방향 보존 + magnitude cap | [Ch3-03](./ch3-vanishing-exploding/03-gradient-clipping.md) |
| **Orthogonal Init 의 Spectrum** | $W_{hh}^\top W_{hh} = I$ → 모든 singular value $= 1$, gradient norm 보존 | [Ch3-04](./ch3-vanishing-exploding/04-orthogonal-init.md) |
| **IRNN 의 Identity Skip 효과** | $W_{hh} = I$ + ReLU → $h_t = h_{t-1} + \mathrm{ReLU}(\ldots)$, residual-like | [Ch3-05](./ch3-vanishing-exploding/05-irnn.md) |
| **CEC 정리 (Hochreiter 1997)** | $\partial c_t / \partial c_{t-1} = f_t$ — Jacobian 의 곱셈적 누적이 $\prod f_t$ 로 단순화 | [Ch4-03](./ch4-lstm/03-cec-proof.md) |
| **LSTM Additive vs RNN Multiplicative** | $c_t = f_t c_{t-1} + i_t \tilde c_t$ vs $h_t = W \sigma(h_{t-1})$ — 기하급수 vs 선형 | [Ch4-04](./ch4-lstm/04-gradient-flow.md) |
| **Forget Bias = 1 효과 (Jozefowicz 2015)** | $\sigma(\cdot + 1) \approx 0.73$, 초기 cell state 보존 → Adding problem 학습 가속 | [Ch4-04](./ch4-lstm/04-gradient-flow.md) |
| **GRU 파라미터 절약** | 2 gates + 통합 hidden, LSTM 대비 ~25% 파라미터 감소, 유사 성능 (Chung 2014) | [Ch4-05](./ch4-lstm/05-gru.md) |
| **Greff 2017 Variants Ablation** | Peephole·coupled·forget-only 등 8 variants — vanilla LSTM 이 robust | [Ch4-06](./ch4-lstm/06-lstm-variants.md) |
| **BiLSTM Inference 한계** | $\overleftarrow h_t$ 가 미래 의존 → online inference 불가 | [Ch5-01](./ch5-advanced-rnn/01-bidirectional.md) |
| **NTM Differentiable Addressing** | Content-based + location-based head, sharpening parameter $\gamma$ | [Ch5-03](./ch5-advanced-rnn/03-ntm-memory.md) |
| **Echo State Property** | $\rho(W_{hh}) < 1 \Rightarrow$ 초기 조건 fading, ESN 학습 가능성 보장 | [Ch5-04](./ch5-advanced-rnn/04-esn.md) |
| **Seq2Seq Information Bottleneck** | BLEU 가 sentence length 에 단조 감소 (Cho 2014b 실증) | [Ch6-02](./ch6-seq2seq-attention/02-bottleneck.md) |
| **Bahdanau Additive Attention** | $e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)$, learned soft alignment | [Ch6-03](./ch6-seq2seq-attention/03-bahdanau-attention.md) |
| **Luong Multiplicative Attention** | $e_{ij} = h_i^\top W s_j$ — Transformer scaled dot-product 의 직계 조상 | [Ch6-04](./ch6-seq2seq-attention/04-luong-attention.md) |
| **Pointer Network** | $p(y_t = i) = \mathrm{softmax}(e_{ti})$ — attention 자체가 output distribution | [Ch6-05](./ch6-seq2seq-attention/05-coverage-pointer.md) |
| **RNN Parallelism Bound** | Sequence-internal GPU utilization $\le 1/T$ — Transformer 의 동기 | [Ch7-01](./ch7-modern-alternatives/01-parallelism-limit.md) |
| **WaveNet Dilated Conv** | Dilation $1, 2, 4, \ldots, 2^L$ 로 $O(\log T)$ receptive field | [Ch7-02](./ch7-modern-alternatives/02-cnn-sequence.md) |
| **Linear Attention $O(T)$** | $\phi(Q)(\phi(K)^\top V)$ 형태로 RNN-like recurrence 부활 | [Ch7-03](./ch7-modern-alternatives/03-linear-attention-rwkv.md) |
| **Mamba Selective SSM** | Input-dependent $A, B, C$, hardware-aware parallel scan, $O(T)$ inference | [Ch7-04](./ch7-modern-alternatives/04-s4-mamba.md) |

> 💡 **챕터별 문서·정리/정의 수**:
>
> | 챕터 | 문서 수 | 정리·정의 |
> |------|---------|------------|
> | Ch1 Sequence 기초 | 4 | ~46 |
> | Ch2 BPTT | 5 | ~58 |
> | Ch3 Vanishing/Exploding | 5 | ~57 |
> | Ch4 LSTM | 6 | ~71 |
> | Ch5 Advanced RNN | 4 | ~45 |
> | Ch6 Seq2Seq + Attention | 5 | ~58 |
> | Ch7 한계와 현대적 대안 | 4 | ~49 |
> | **합계** | **33** | **~384** |
>
> 추가로 **40+ 엄밀한 $\square$ 증명 + 99 연습문제 (모두 해설 포함) + 165+ NumPy/PyTorch 실험 코드**.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
torch==2.1.0
nltk==3.8.0                  # N-gram, Penn Treebank tokenizer
sentencepiece==0.1.99        # NMT subword
torchtext==0.16.0            # Seq2Seq 데이터로더
matplotlib==3.8.0
tqdm==4.66.0
jupyter==1.0.0
# 선택 사항
mamba-ssm==1.1.1             # Ch7-04 Mamba (CUDA 필요)
flash-attn==2.3.3            # Transformer 비교 (Ch7-01)
```

```bash
# 환경 설치 (CPU)
pip install numpy==1.26.0 scipy==1.11.0 torch==2.1.0 \
            nltk==3.8.0 sentencepiece==0.1.99 \
            torchtext==0.16.0 matplotlib==3.8.0 tqdm==4.66.0 jupyter==1.0.0

# NLTK 데이터
python -m nltk.downloader punkt brown gutenberg

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — RNN vs LSTM Gradient Flow 시간축 비교 (Ch3-01, Ch4-03)
import numpy as np
import matplotlib.pyplot as plt

def rnn_gradient_norm(T, spectral_radius=0.9, hidden=10):
    """Plain RNN: ||∂h_t / ∂h_0|| 가 ρ^t 로 exponential 감쇠"""
    W = np.random.randn(hidden, hidden)
    u, s, vt = np.linalg.svd(W)
    W = u @ np.diag(np.ones_like(s) * spectral_radius) @ vt   # ρ 정확히 설정
    grads, cur = [1.0], np.eye(hidden)
    for _ in range(T):
        cur = W.T @ cur * 0.5   # tanh' ≤ 1, 평균 ~0.5
        grads.append(np.linalg.norm(cur))
    return grads

def lstm_gradient_norm(T, f_gate=0.99):
    """LSTM CEC: ∂c_t / ∂c_0 = ∏ f_t — additive update 로 보존"""
    return [f_gate ** t for t in range(T + 1)]

T = 100
plt.figure(figsize=(10, 5))
plt.semilogy(rnn_gradient_norm(T, 0.9),  label='Plain RNN ρ=0.9 (vanish)')
plt.semilogy(rnn_gradient_norm(T, 1.1),  label='Plain RNN ρ=1.1 (explode)')
plt.semilogy(lstm_gradient_norm(T, 0.99), label='LSTM f≈1 (preserve)')
plt.xlabel('Time step'); plt.ylabel('||∂h_t / ∂h_0||  (log)')
plt.title('Vanishing/Exploding vs Constant Error Carousel')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# 대표 실험 ② — LSTM 바닥부터 구현 + PyTorch nn.LSTM 일치 검증 (Ch4-02)
import torch
import torch.nn as nn

class NumPyLSTM:
    """4 gates 명시적 구현 — Jozefowicz 2015 forget bias = 1"""
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.RandomState(seed)
        H, I = hidden_size, input_size
        scale = np.sqrt(1.0 / (I + H))
        self.Wf = rng.randn(H, I + H) * scale
        self.Wi = rng.randn(H, I + H) * scale
        self.Wc = rng.randn(H, I + H) * scale
        self.Wo = rng.randn(H, I + H) * scale
        self.bf = np.ones(H)              # ★ forget bias = 1
        self.bi = np.zeros(H)
        self.bc = np.zeros(H)
        self.bo = np.zeros(H)

    @staticmethod
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x_seq, h0, c0):
        h, c = h0, c0
        history = []
        for x in x_seq:
            xh = np.concatenate([x, h])
            f = self.sigmoid(self.Wf @ xh + self.bf)
            i = self.sigmoid(self.Wi @ xh + self.bi)
            g = np.tanh(  self.Wc @ xh + self.bc)
            o = self.sigmoid(self.Wo @ xh + self.bo)
            c = f * c + i * g            # ★ additive update
            h = o * np.tanh(c)
            history.append({'f': f, 'i': i, 'g': g, 'o': o, 'c': c, 'h': h})
        return h, c, history

# Synthetic Adding Problem (T=200) 에서 forget bias 효과 측정
# 결과: b_f = 1 → 5 epoch 만에 수렴, b_f = 0 → 50+ epoch 후에도 chance

# 대표 실험 ③ — Bahdanau Attention Heatmap (Ch6-03)
def bahdanau_score(h_enc, s_dec, W1, W2, v):
    """e_ij = v^T tanh(W1 h_i + W2 s_j)"""
    T_in, T_out = len(h_enc), len(s_dec)
    e = np.zeros((T_out, T_in))
    for j in range(T_out):
        for i in range(T_in):
            e[j, i] = v @ np.tanh(W1 @ h_enc[i] + W2 @ s_dec[j])
    alpha = np.exp(e) / np.exp(e).sum(axis=1, keepdims=True)
    return alpha

# WMT'14 En→Fr toy 예제로 attention heatmap 시각화
# "I love you" → "Je t'aime" 의 word-level alignment 확인

# 대표 실험 ④ — Pascanu Spectral 분포 (Ch3-01)
hidden = 100
for rho in [0.5, 0.9, 1.0, 1.1, 1.5]:
    W = np.random.randn(hidden, hidden)
    u, s, vt = np.linalg.svd(W)
    W = u @ np.diag(np.ones_like(s) * rho) @ vt
    eigvals = np.linalg.eigvals(W)
    print(f'ρ={rho}: |eig| max = {np.abs(eigvals).max():.4f}, mean = {np.abs(eigvals).mean():.4f}')
# 결과: SVD 로 설정한 ρ 가 정확히 spectral radius 와 일치
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 기법이 sequence 처리에 중요한가** | BPTT · vanishing · CEC · Attention 과의 연결 |
| 3 | 📐 **수학적 선행 조건** | NN Theory · LA · Calc · Optimization 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | 시간축 unrolled diagram · gate 역할 시각화 · attention heatmap |
| 5 | ✏️ **엄밀한 정의** | RNN · BPTT · LSTM · GRU · Seq2Seq · Attention · SSM |
| 6 | 🔬 **정리와 증명 / 결과** | Pascanu spectral · CEC · Bahdanau alignment · Mamba selective |
| 7 | 💻 **구현 (NumPy / PyTorch)** | RNN/LSTM 바닥부터, gradient norm 측정, attention 시각화 |
| 8 | 🔗 **실전 활용** | 언제 RNN, 언제 Transformer, 언제 SSM |
| 9 | ⚖️ **가정과 한계** | 병렬성 · 긴 sequence · 훈련 안정성 · sample efficiency |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 ($\boxed{}$ + 표) |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산 · 증명 재구성 · 구현 · 논문 비평 (`<details>` 펼침 해설) |

> 📚 **연습문제 총 99개** (33 문서 × 3 문제): 기초 / 심화 / 논문 비평 의 3-tier 구성, 모든 문제에 `<details>` 펼침 해설 포함. BPTT 손 유도부터 CEC 증명 재구성, Bahdanau→Luong→Transformer 의 unification, Mamba selective scan 의 hardware-aware 구현까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 480~520줄 (정의·증명·코드·연습문제 포함) 기준 **약 55분~1시간 15분**. 전체 33문서는 약 **30~40시간** 상당 (증명 재구성·실험 재현 포함 시 50시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "LSTM 은 쓰지만 왜 작동하는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 10~12시간)</b></summary>

<br/>

```
Day 1  Ch1-01  Sequence 학습 정식화
       Ch1-04  RNN 의 정의
Day 2  Ch2-01  Unrolled Computational Graph
       Ch2-02  BPTT 유도
Day 3  Ch3-01  Pascanu Spectral 분석
       Ch3-03  Gradient Clipping
Day 4  Ch4-01  LSTM 동기 (CEC 비전)
       Ch4-02  4 gate 수식
Day 5  Ch4-03  CEC 증명
       Ch4-04  Forget bias = 1
Day 6  Ch6-01  Encoder-Decoder
       Ch6-03  Bahdanau Attention
Day 7  Ch6-04  Luong Attention (Transformer 조상)
       Ch7-01  병렬성 한계
```

</details>

<details>
<summary><b>🟡 "BPTT 의 spectral 분석과 LSTM 의 CEC 메커니즘을 완전히 정복한다" — 이론 집중 (2주, 약 20~24시간)</b></summary>

<br/>

```
1주차 — BPTT 와 Vanishing/Exploding
  Day 1    Ch1-01~04   Sequence 학습 + RNN 정의
  Day 2    Ch2-01~02   Unrolled graph + BPTT 유도 손 재현
  Day 3    Ch2-03~05   Truncated BPTT + 복잡도 + RTRL
  Day 4    Ch3-01      Pascanu spectral 정리 증명 꼼꼼히
  Day 5    Ch3-02~03   Saturation + Gradient Clipping
  Day 6-7  Ch3-04~05   Orthogonal init + IRNN

2주차 — LSTM 과 Attention 의 계보
  Day 1    Ch4-01~02   LSTM 동기 + 4 gate 수식
  Day 2    Ch4-03      CEC 증명 — ∂c_t/∂c_{t-1} = f_t 손 유도
  Day 3    Ch4-04      Gradient flow + Jozefowicz forget bias
  Day 4    Ch4-05~06   GRU + Variants (Greff ablation)
  Day 5    Ch5 전체     BiRNN + Stacked + NTM + ESN
  Day 6    Ch6-01~04   Seq2Seq + Bahdanau + Luong 손 유도
  Day 7    Ch6-05      Coverage + Pointer + Transformer 직계 조상 확인
```

</details>

<details>
<summary><b>🔴 "RNN 부터 Mamba 까지 sequence 모델의 수학을 완전 정복한다" — 전체 정복 (10주, 약 35~45시간 + 실험 재현 10~15시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — Sequence 기초
         → N-gram smoothing 손 계산, Bengio 2003 NLM 재현
         → vanilla RNN NumPy 바닥부터 구현

2주차   Chapter 2 전체 — BPTT
         → 4-step RNN 의 BPTT 손 유도 + PyTorch autograd 일치 검증
         → Truncated BPTT $k$ 별 PTB perplexity 측정
         → RTRL 의 $O(n^4)$ vs BPTT $O(n^2)$ 비교 실험

3주차   Chapter 3 전체 — Vanishing/Exploding
         → Pascanu spectral 정리 증명 재구성
         → SVD 기반 spectral radius 분포 시각화
         → Orthogonal init vs IRNN vs LSTM 비교

4주차   Chapter 4 (1~3) — LSTM 동기와 CEC
         → Hochreiter 1997 의 CEC 비전 추적
         → 4 gate 수식 NumPy 바닥부터, PyTorch 출력 일치
         → CEC 정리 ∂c_t/∂c_{t-1} = f_t 손 증명

5주차   Chapter 4 (4~6) — Gradient flow + GRU + Variants
         → Adding Problem (T=200) 에서 forget bias = 1 ablation
         → GRU 파라미터 절약 측정 (LSTM 대비 ~25%)
         → Greff 2017 의 8 variants 재현

6주차   Chapter 5 전체 — Advanced RNN
         → BiLSTM-CRF 로 NER (CoNLL-2003) 재현
         → NTM 의 copy / repeat-copy task
         → ESN 의 Mackey-Glass time series 예측

7주차   Chapter 6 (1~3) — Seq2Seq + Bahdanau
         → Sutskever 2014 의 reverse input trick 재현
         → Cho 2014b 의 length 별 BLEU 감쇠 측정
         → Bahdanau additive attention WMT'14 En→Fr

8주차   Chapter 6 (4~5) — Luong + Pointer
         → Luong 의 dot/general/concat 비교
         → Pointer Network 의 TSP / sorting 응용
         → Transformer 까지의 계보 정리

9주차   Chapter 7 (1~2) — 병렬성 한계 + CNN
         → 동일 hardware 에서 LSTM vs Transformer 훈련 시간 측정
         → WaveNet dilated conv 의 receptive field 시각화
         → TCN vs LSTM 의 task-별 성능 비교 (Bai 2018 재현)

10주차  Chapter 7 (3~4) + 종합 — Linear Attention + Mamba
         → Katharopoulos 2020 Linear Attention 의 RNN-like recurrence
         → S4 의 HiPPO matrix 와 diagonal + low-rank 분해
         → Mamba 의 selective SSM 으로 LRA benchmark 재현
         → "RNN vs Transformer vs SSM" 설계 원리 정리
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | Spectral theorem · 고유값 분포 · matrix norm | **Ch3 전체** (spectral 분석), Ch4-03 (CEC 증명의 Jacobian) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 연쇄법칙 · Jacobian · gradient flow | **Ch2 전체** (BPTT), Ch3-03 (Clipping), Ch4-04 (Gradient flow) |
| [neural-network-theory-deep-dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive) | Backprop · 활성화 함수 · 초기화 | **전체 레포 전제**, Ch3-02 (saturation), Ch3-04 (orthogonal init) |
| [optimization-theory-deep-dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive) | GD · SGD · Adam · gradient flow 안정성 | **Ch3** (clipping 의 정당화), Ch4 (LSTM 훈련 안정성) |
| [graphical-models-deep-dive](https://github.com/iq-ai-lab/graphical-models-deep-dive) | HMM · CRF · Viterbi · Forward-Backward | **Ch5-01** (BiLSTM-CRF NER), Ch1-02 (HMM = linear RNN 원조) |
| [transformer-deep-dive](https://github.com/iq-ai-lab/transformer-deep-dive) *(다음)* | Self-Attention · Multi-Head · Positional Encoding | **Ch6-04** (Luong = scaled dot-product 조상), Ch7-01 (병렬성) |
| [gnn-deep-dive](https://github.com/iq-ai-lab/gnn-deep-dive) | Graph Laplacian · Message Passing · Graphormer | Ch6 (Attention 의 graph 일반화), Ch7 (구조 모델링 비교) |
| [cnn-deep-dive](https://github.com/iq-ai-lab/cnn-deep-dive) | Convolution · Translation Equivariance · Dilation | **Ch7-02** (WaveNet · TCN 의 dilated conv) |

> 💡 이 레포는 **"Sequence 위의 딥러닝이 왜 BPTT 의 Jacobian 곱 누적으로 환원되고, vanishing/exploding 이 왜 spectral radius 의 필연적 귀결이며, LSTM 이 어떻게 곱셈적 감쇠를 덧셈적 보존으로 바꾸는가"** 에 집중합니다. NN Theory 에서 backprop 을, Linear Algebra 에서 spectral decomposition 을, Calculus & Optimization 에서 chain rule 의 multi-variable 일반화를 익힌 후 오면 Chapter 2 (BPTT) 와 Chapter 4 (CEC) 의 증명이 훨씬 자연스럽습니다. **Transformer Deep Dive (다음 레포) 는 이 레포 Chapter 6 (Bahdanau→Luong attention) 을 직접 전제로 시작합니다.**

---

## 📖 Reference

### 🏛️ Sequence Model 기초 · Language Modeling
- **A Neural Probabilistic Language Model** (Bengio, Ducharme, Vincent, Janvin, 2003) — **Neural LM 의 원전**, word embedding 의 시작
- **Foundations of Statistical Natural Language Processing** (Manning & Schütze, 1999) — N-gram, smoothing 표준 교과서
- **An Empirical Study of Smoothing Techniques for Language Modeling** (Chen & Goodman, 1999) — Kneser-Ney 비교 분석
- **Speech and Language Processing** (Jurafsky & Martin, 3rd ed.) — sequence learning 종합 교과서

### 🔁 RNN 의 정의와 BPTT
- **Finding Structure in Time** (Elman, 1990) — **Elman network**
- **Serial Order: A Parallel Distributed Processing Approach** (Jordan, 1986) — Jordan network
- **Backpropagation Through Time: What it does and how to do it** (Werbos, 1990) — **BPTT 정식 유도**
- **A Learning Algorithm for Continually Running Fully Recurrent Neural Networks** (Williams & Zipser, 1989) — **RTRL**
- **Training Recurrent Networks Online Without Backtracking** (Tallec & Ollivier, 2017) — **UORO** (RTRL 의 unbiased $O(n^2)$)

### 🌊 Vanishing / Exploding Gradient
- **Long Short-Term Memory** (Hochreiter & Schmidhuber, 1997) — **LSTM 원전**, vanishing gradient 진단 + CEC
- **Learning Long-Term Dependencies with Gradient Descent is Difficult** (Bengio, Simard, Frasconi, 1994) — vanishing 문제 첫 형식화
- **On the Difficulty of Training Recurrent Neural Networks** (Pascanu, Mikolov, Bengio, 2013) — **Spectral 분석 + Gradient Clipping**
- **Exact Solutions to the Nonlinear Dynamics of Learning** (Saxe, McClelland, Ganguli, 2014) — **Orthogonal Initialization**
- **A Simple Way to Initialize Recurrent Networks of Rectified Linear Units** (Le, Jaitly, Hinton, 2015) — **IRNN**
- **Unitary Evolution Recurrent Neural Networks** (Arjovsky, Shah, Bengio, 2016) — Unitary RNN
- **Full-Capacity Unitary Recurrent Neural Networks** (Wisdom et al., 2016)

### 🧠 LSTM · GRU · Variants
- **Learning to Forget: Continual Prediction with LSTM** (Gers, Schmidhuber, Cummins, 2000) — **Forget gate 도입 + Peephole**
- **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation** (Cho et al., 2014) — **GRU 원전**
- **Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling** (Chung, Gulcehre, Cho, Bengio, 2014) — LSTM vs GRU
- **An Empirical Exploration of Recurrent Network Architectures** (Jozefowicz, Zaremba, Sutskever, 2015) — **Forget bias = 1 추천**
- **LSTM: A Search Space Odyssey** (Greff, Srivastava, Koutník, Steunebrink, Schmidhuber, 2017) — 8 variants ablation
- **Convolutional LSTM Network** (Shi et al., 2015) — **ConvLSTM**

### 🌐 Advanced RNN — BiRNN · NTM · ESN
- **Bidirectional Recurrent Neural Networks** (Schuster & Paliwal, 1997) — **BiRNN 원전**
- **Neural Architectures for Named Entity Recognition** (Lample, Ballesteros, Subramanian, Kawakami, Dyer, 2016) — **BiLSTM-CRF NER**
- **Neural Turing Machines** (Graves, Wayne, Danihelka, 2014) — **NTM**
- **Hybrid Computing using a Neural Network with Dynamic External Memory** (Graves et al., 2016) — **DNC**
- **Memory Networks** (Weston, Chopra, Bordes, 2014)
- **The "Echo State" Approach to Analysing and Training Recurrent Neural Networks** (Jaeger, 2001) — **ESN**

### 🎯 Seq2Seq · Attention
- **Sequence to Sequence Learning with Neural Networks** (Sutskever, Vinyals, Le, 2014) — **Seq2Seq 원전**
- **Learning Phrase Representations using RNN Encoder-Decoder** (Cho et al., 2014) — Encoder-Decoder
- **On the Properties of Neural Machine Translation: Encoder-Decoder Approaches** (Cho et al., 2014b) — **Long sentence curse 정량화**
- **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau, Cho, Bengio, 2015) — **Bahdanau Additive Attention**
- **Effective Approaches to Attention-based Neural Machine Translation** (Luong, Pham, Manning, 2015) — **Luong Multiplicative Attention**
- **Modeling Coverage for Neural Machine Translation** (Tu, Lu, Liu, Liu, Li, 2016) — **Coverage Mechanism**
- **Pointer Networks** (Vinyals, Fortunato, Jaitly, 2015) — **Pointer Network**
- **Get To The Point: Summarization with Pointer-Generator Networks** (See, Liu, Manning, 2017)
- **Listen, Attend and Spell** (Chan et al., 2016) — Speech recognition
- **Show, Attend and Tell** (Xu et al., 2015) — Image captioning

### 🚀 RNN 의 한계와 현대적 대안
- **Attention Is All You Need** (Vaswani et al., 2017) — **Transformer** (RNN 대체의 결정타)
- **WaveNet: A Generative Model for Raw Audio** (van den Oord et al., 2016) — **CNN-based sequence**
- **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling** (Bai, Kolter, Koltun, 2018) — **TCN**
- **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** (Katharopoulos, Vyas, Pappas, Fleuret, 2020) — **Linear Attention**
- **RWKV: Reinventing RNNs for the Transformer Era** (Peng et al., 2023) — **RWKV**
- **HiPPO: Recurrent Memory with Optimal Polynomial Projections** (Gu, Dao, Ermon, Rudra, Ré, 2020)
- **Efficiently Modeling Long Sequences with Structured State Spaces** (Gu, Goel, Ré, 2022) — **S4**
- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu & Dao, 2023) — **Mamba**
- **Long Range Arena: A Benchmark for Efficient Transformers** (Tay et al., 2021) — **LRA benchmark**

### 🔮 분석 · 이론
- **Visualizing and Understanding Recurrent Networks** (Karpathy, Johnson, Fei-Fei, 2015) — LSTM gate activation 분석
- **The Unreasonable Effectiveness of Recurrent Neural Networks** (Karpathy, 2015) — char-RNN tutorial
- **A Critical Review of Recurrent Neural Networks for Sequence Learning** (Lipton, Berkowitz, Elkan, 2015)
- **On the Practical Computational Power of Finite Precision RNNs for Language Recognition** (Weiss, Goldberg, Yahav, 2018)

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"`nn.LSTM(input_size, hidden_size)` 을 호출하는 것과 — Pascanu 2013 으로 $\rho(W_{hh})$ 가 vanishing/exploding 의 결정자임을 spectral 로 증명 · Hochreiter 1997 의 $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$ 의 additive update 가 $\partial c_t / \partial c_{t-1} = f_t$ 의 CEC 를 만들어 곱셈적 감쇠를 덧셈적 보존으로 바꾸는 메커니즘을 손 유도 · Bahdanau 2015 의 $e_{ij} = v^\top \tanh(W_1 h_i + W_2 s_j)$ 와 Luong 2015 의 $h_i^\top W s_j$ 가 Transformer scaled dot-product attention 의 직계 조상임을 추적 · Gu & Dao 2023 Mamba 의 selective SSM 이 RNN과 CNN의 통합 관점에서 Transformer 시대의 RNN 부활을 어떻게 설명하는지 재현 — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
