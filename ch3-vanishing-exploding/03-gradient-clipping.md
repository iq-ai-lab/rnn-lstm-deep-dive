# 03. Gradient Clipping — Exploding 대응

## 🎯 핵심 질문

- Pascanu 2013 의 norm-based gradient clipping $g \leftarrow g \cdot \min(1, \theta / \|g\|)$ 의 정확한 동작은?
- Element-wise clipping 과의 차이는 무엇이며, 왜 norm-based 가 표준이 되었는가?
- Gradient direction 이 보존되고 magnitude 만 cap 되는 이유 — geometric intuition
- Clipping threshold $\theta$ 의 선택 — Pascanu 의 권장값과 task 별 tuning
- Gradient clipping 과 learning rate 의 상호작용, Adam 같은 adaptive optimizer 와의 결합

---

## 🔍 왜 gradient clipping 이 RNN 학습의 필수 도구인가

Vanilla SGD + RNN 학습은 거의 항상 exploding gradient 문제를 만납니다. 한 번의 큰 gradient update 가:
- Weight 를 unstable region 으로 push
- NaN / Inf 발생
- 학습 발산

**Pascanu 2013 의 norm clipping** 이 이를 정확히 해결:
- **단순함**: 한 줄 코드 (`torch.nn.utils.clip_grad_norm_`)
- **이론적 정당성**: Direction 보존, magnitude 만 cap
- **표준화**: 모든 modern RNN/LSTM 학습에 사용

이 문서는 clipping 의 정확한 정의, geometric intuition, 그리고 다양한 변종을 다룹니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-saturation-problem.md](./02-saturation-problem.md) — Spectral analysis, exploding 진단
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Gradient descent, learning rate
- 선형대수: Vector norm, projection
- (선택) Trust region method: 비슷한 직관

---

## 📖 직관적 이해

### Exploding 의 catastrophe

```
loss
  ↑
  │     ╱ huge spike (exploding gradient → bad weight update)
  │    ╱
  │   /
  │  /  normal training
  │ /
  └────────────────────→ steps
```

한 번의 huge gradient → weight 가 stable region 밖으로 jump → NaN.

### Clipping 의 효과

```
gradient norm
  ↑
  │ ─ ─ ─ θ (threshold)─ ─ ─ ─ ─ ─ ─ ─ ─
  │              ╱─╲
  │            ╱     ╲
  │          ╱         ╲
  │ raw  ╱─╱             ╲
  │ ╱╱╱╱                   ╲    raw
  │
  └────────────────────────────→ steps
  
clipped:
  │ ─ ─ ─ θ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  │  cap  cap     cap
  │ ╱╱╱  ───────  ───  
  │ ╱╱╱
  └─────────────────────→
```

Clipping 후 gradient norm 이 $\theta$ 이상이면 정확히 $\theta$ 로 cap.

### Direction Preservation

```
g_raw  =  (5, 12)   ||g|| = 13
θ = 5

g_clipped = g_raw · 5/13 = (5·5/13, 12·5/13) = (1.92, 4.62)
||g_clipped|| = 5  ✓
direction same: (5,12)/13 = (1.92, 4.62)/5  ✓
```

방향은 보존, 크기만 줄임. 이는 SGD 의 descent direction 을 따라가되 step 크기를 제한.

### Element-wise vs Norm-based

```
g_raw = (-5, 20)
θ = 5

Element-wise: clip(g_i, -5, 5) → (-5, 5)
  Direction:  (-5, 20)/√425 ≠ (-5, 5)/√50  ← 변경됨!

Norm-based: g · 5/||g|| → (-5/√17, 20/√17)·5 = (-1.21, 4.85)
  Direction:  same as raw  ✓
```

Element-wise 는 direction 변경 — descent 가 아닌 다른 방향으로 update 가능.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Norm-Based Gradient Clipping (Pascanu 2013)

Threshold $\theta > 0$:

$$
g \leftarrow \begin{cases} g \cdot \theta / \|g\| & \text{if } \|g\| > \theta \\ g & \text{otherwise} \end{cases}
$$

또는 한 줄:

$$
g \leftarrow g \cdot \min(1, \theta / \|g\|)
$$

여기서 $\|g\|$ 는 보통 L2 norm.

### 정의 3.2 — Element-wise Gradient Clipping

각 component:

$$
g_i \leftarrow \mathrm{clip}(g_i, -\theta, +\theta) = \min(\max(g_i, -\theta), +\theta)
$$

### 정의 3.3 — Per-Layer Norm Clipping

Layer $\ell$ 별로 별도 norm 계산:

$$
g^{(\ell)} \leftarrow g^{(\ell)} \cdot \min(1, \theta_\ell / \|g^{(\ell)}\|)
$$

PyTorch `clip_grad_norm_` 의 default 는 model 전체 — **global norm**.

### 정의 3.4 — Adaptive Clipping (Pascanu 2013 §3.2 Variant)

매 step 의 average gradient norm 의 multiple:

$$
\theta_t = c \cdot \mathrm{avg}(\|g_{t'}\|)_{t' < t}
$$

학습 진행에 따라 자동 조정.

### 정의 3.5 — Gradient Clipping 의 Trust Region 해석

Constraint:

$$
\theta^{\text{new}} = \arg\min_{\theta'} \nabla L(\theta)^\top (\theta' - \theta) \quad \text{s.t.} \quad \|\theta' - \theta\| \le \eta \theta_{\text{clip}}
$$

→ Steepest descent direction (gradient) 의 normalized step.

---

## 🔬 정리와 결과

### 정리 3.1 — Direction Preservation of Norm Clipping

Clipped gradient $g'$ 에 대해:

$$
g' = \alpha g \quad \text{where} \quad \alpha = \min(1, \theta / \|g\|) \ge 0
$$

따라서 $g'$ 와 $g$ 가 같은 direction (positive scalar multiple).

**증명**: 자명. $\alpha \ge 0$ 이므로 같은 ray. $\square$

**의미**: Descent direction 보존 — SGD 의 convergence 보장 유지.

### 정리 3.2 — Loss Decrease Guarantee

Lipschitz gradient $L$-smooth function $f$ 에 대해, clipped SGD step:

$$
f(\theta - \eta g') \le f(\theta) - \eta \alpha \|g\|^2 + \frac{L \eta^2 \alpha^2 \|g\|^2}{2}
$$

$\alpha \le 1$ 이므로 $\eta < 2/L$ 시 monotone decrease.

**증명**: $f(\theta - \eta g') = f(\theta) - \eta \alpha \nabla f^\top g + O(\eta^2)$. $g = \nabla f$ 이면 $\nabla f^\top g = \|\nabla f\|^2 = \|g\|^2$. $\square$

### 정리 3.3 — Global Norm vs Per-Layer Norm

Global norm: $\|g\| = \sqrt{\sum_\ell \|g^{(\ell)}\|^2}$. 한 layer 의 explosion 이 다른 layer 의 update 도 cap.

Per-layer norm: 각 layer 독립 — 한 layer 만 cap.

**Trade-off**:
- Global: simpler, 모든 layer 가 동일 비율 cap
- Per-layer: layer 별 dynamics 다를 때 더 정확

**PyTorch standard**: global norm.

### 정리 3.4 — Pascanu 의 Threshold 선택

권장값 $\theta = 1.0$ — Pascanu 2013 의 PTB 실험.

**이유**:
- Average $\|g\|$ 가 1 정도 (Adam optimizer 가 normalize)
- Spike 가 average 의 5x ~ 100x 이므로 1 로 cap 시 spike 만 영향
- Smaller $\theta$: 학습 느려짐, larger: spike 회피 못함

### 정리 3.5 — Adam + Clipping 의 상호작용

Adam 의 update:

$$
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

여기서 $m_t / \sqrt{v_t}$ 가 이미 normalized. Clipping 의 effect 는 spike 시에만.

**관찰**: Adam + clipping 이 SGD + clipping 보다 더 안정 — Adam 의 second moment 가 effective learning rate 자동 조정.

---

## 💻 구현 검증

### 실험 1 — Manual Clipping 구현

```python
import torch
import torch.nn as nn

def manual_clip_grad_norm(parameters, max_norm):
    """Norm-based clipping 직접 구현"""
    grads = [p.grad for p in parameters if p.grad is not None]
    
    # Total norm
    total_norm = torch.norm(torch.stack([g.norm() for g in grads]))
    
    # Clip if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)
    
    return total_norm

# Test on RNN
H = 50
torch.manual_seed(0)
rnn = nn.LSTM(10, H)
x = torch.randn(20, 4, 10)
out, _ = rnn(x)
loss = out.sum()
loss.backward()

print(f'Before clipping:')
total = sum(p.grad.norm()**2 for p in rnn.parameters() if p.grad is not None) ** 0.5
print(f'  ||g|| = {total:.4f}')

# Clip
total_pre = manual_clip_grad_norm(rnn.parameters(), max_norm=1.0)
total_post = sum(p.grad.norm()**2 for p in rnn.parameters() if p.grad is not None) ** 0.5
print(f'After clipping (θ=1.0):')
print(f'  ||g|| = {total_post:.4f}')

# PyTorch built-in 과 일치 확인
torch.manual_seed(0)
rnn2 = nn.LSTM(10, H)
out, _ = rnn2(x)
out.sum().backward()
total_native = nn.utils.clip_grad_norm_(rnn2.parameters(), max_norm=1.0)
print(f'PyTorch built-in pre-clip norm: {total_native:.4f}')
```

### 실험 2 — Norm vs Element-wise Clipping 의 Direction 비교

```python
import numpy as np

g = np.array([5.0, 12.0])
theta = 5.0

# Norm-based
g_norm = g * min(1, theta / np.linalg.norm(g))
print(f'Original:    g = {g}, ||g|| = {np.linalg.norm(g):.4f}')
print(f'Norm clip:   g = {g_norm}, ||g|| = {np.linalg.norm(g_norm):.4f}')
print(f'  Direction match: {np.allclose(g/np.linalg.norm(g), g_norm/np.linalg.norm(g_norm))}')

# Element-wise
g_elem = np.clip(g, -theta, +theta)
print(f'Elem clip:   g = {g_elem}, ||g|| = {np.linalg.norm(g_elem):.4f}')
print(f'  Direction match: {np.allclose(g/np.linalg.norm(g), g_elem/np.linalg.norm(g_elem))}')
# Element-wise 가 direction 변경
```

### 실험 3 — 학습 안정성 비교

```python
import matplotlib.pyplot as plt

def train_with_clipping(use_clip, n_steps=200, theta=1.0):
    torch.manual_seed(42)
    rnn = nn.RNN(10, 50, batch_first=False, nonlinearity='tanh')
    opt = torch.optim.SGD(rnn.parameters(), lr=0.1)
    
    losses, norms = [], []
    for step in range(n_steps):
        x = torch.randn(50, 8, 10)   # T=50, B=8
        target = torch.randn(50, 8, 50)
        out, _ = rnn(x)
        loss = ((out - target)**2).mean()
        
        opt.zero_grad()
        loss.backward()
        
        # Pre-clip norm
        norm = sum(p.grad.norm()**2 for p in rnn.parameters() if p.grad is not None) ** 0.5
        norms.append(norm.item())
        
        if use_clip:
            nn.utils.clip_grad_norm_(rnn.parameters(), theta)
        
        opt.step()
        losses.append(loss.item())
    return losses, norms

losses_no, norms_no = train_with_clipping(use_clip=False)
losses_yes, norms_yes = train_with_clipping(use_clip=True, theta=1.0)

print(f'Without clipping: max grad norm = {max(norms_no):.2f}')
print(f'With clipping:    max grad norm = {max(norms_yes):.2f}')
print(f'Without clipping: final loss = {losses_no[-1]:.4f}')
print(f'With clipping:    final loss = {losses_yes[-1]:.4f}')

# Without clipping 시 norm spike → loss 발산 가능
```

### 실험 4 — Threshold $\theta$ 선택의 Sensitivity

```python
for theta in [0.1, 1.0, 10.0, 100.0]:
    losses, _ = train_with_clipping(use_clip=True, theta=theta, n_steps=100)
    print(f'θ={theta:6.1f}: final loss = {losses[-1]:.4f}, mean loss = {np.mean(losses):.4f}')
# 너무 작은 θ: 학습 느림. 너무 큰 θ: clipping 효과 없음
```

### 실험 5 — Adam + Clipping vs SGD + Clipping

```python
def train_optimizer(opt_name, use_clip, n_steps=100):
    torch.manual_seed(42)
    rnn = nn.LSTM(10, 50, batch_first=False)
    if opt_name == 'sgd':
        opt = torch.optim.SGD(rnn.parameters(), lr=0.1)
    else:
        opt = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    
    losses = []
    for step in range(n_steps):
        x = torch.randn(50, 8, 10)
        target = torch.randn(50, 8, 50)
        out, _ = rnn(x)
        loss = ((out - target)**2).mean()
        opt.zero_grad(); loss.backward()
        if use_clip:
            nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses

print('SGD without clip:', train_optimizer('sgd', False)[-1])
print('SGD with clip:   ', train_optimizer('sgd', True)[-1])
print('Adam without clip:', train_optimizer('adam', False)[-1])
print('Adam with clip:   ', train_optimizer('adam', True)[-1])
# Adam + clip 이 가장 안정
```

---

## 🔗 실전 활용

### 1. PyTorch 표준 학습 루프

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
이 한 줄이 modern RNN/Transformer 학습의 표준.

### 2. LLM training (GPT, Llama)

대규모 Transformer 학습에서도 clipping 필수 — 보통 $\theta = 1.0$. Adam 과 결합.

### 3. RL (PPO, A2C)

Policy gradient 의 spike 빈번 — 더 작은 $\theta$ (0.5 ~ 1.0) 으로 안정화.

### 4. GAN training

Discriminator/Generator 의 다른 dynamics — 둘 다 clipping 으로 stable training.

### 5. Mixed precision

Float16 gradient 의 underflow/overflow — `GradScaler` + clipping 결합.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Direction 보존 우월 | Element-wise 가 specific cases 에서 빠를 수도 |
| Global norm 적합 | Layer 별 dynamics 다를 때 per-layer 더 정확 |
| Static $\theta$ | Adaptive clipping 가능 |
| Magnitude 만 문제 | Direction 자체가 wrong 한 경우 (saddle point) — clipping 무용 |
| Convergence 보장 | $\theta$ 너무 작으면 학습 느림 |

---

## 📌 핵심 정리

$$\boxed{g \leftarrow g \cdot \min(1, \theta / \|g\|) \quad \text{— direction 보존, magnitude cap}}$$

$$\boxed{\text{Pascanu 2013: } \theta \approx 1.0 \text{ 권장 (PTB 표준)}}$$

$$\boxed{\text{Convergence guarantee: } \alpha = \min(1, \theta/\|g\|) \ge 0}$$

| Method | Direction | Magnitude | Use case |
|--------|-----------|-----------|----------|
| **Norm clipping** | Preserved | $\le \theta$ | 표준 (RNN, Transformer) |
| **Element-wise** | Changed | each $\le \theta$ | 거의 사용 안 함 |
| **Per-layer norm** | Per-layer preserved | Per-layer $\le \theta_\ell$ | Layer dynamics 다른 경우 |
| **Adaptive** | Preserved | Auto $\theta$ | Streaming, online |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $g = (3, 4, 12)$, $\theta = 5$ 인 경우의 norm clipping 결과를 계산하고, direction 이 보존됨을 확인하라.

<details>
<summary>해설</summary>

$\|g\| = \sqrt{9 + 16 + 144} = \sqrt{169} = 13$

$\|g\| > \theta$ 이므로 clip:

$$
\alpha = \theta / \|g\| = 5/13
$$

$$
g' = g \cdot 5/13 = (15/13, 20/13, 60/13) \approx (1.15, 1.54, 4.62)
$$

**검증**:
- $\|g'\| = \sqrt{1.15^2 + 1.54^2 + 4.62^2} = \sqrt{1.32 + 2.37 + 21.34} = \sqrt{25.04} \approx 5$ ✓
- Direction: $g/\|g\| = (3,4,12)/13$, $g'/\|g'\| = (1.15, 1.54, 4.62)/5 = (3,4,12)/13$ ✓ same

$\square$

</details>

**문제 2** (심화): Element-wise clipping 이 direction 을 변경하는 이유를 기하학적으로 설명하고, 어떤 case 에서 element-wise 가 더 나을 수 있는지 논하라.

<details>
<summary>해설</summary>

**Geometric**:
- Norm clip: gradient vector 를 unit sphere 까지 projection — radial scaling
- Element-wise: gradient vector 를 hypercube $[-\theta, \theta]^n$ 안으로 projection — clipping per axis

**왜 direction 변경**:

$g = (5, 20)$, $\theta = 5$:
- $g_{\text{element}} = (5, 5)$ — direction (1, 4) → (1, 1) 변경
- $g_{\text{norm}} = (5, 20) \cdot 5/\sqrt{425} = (1.21, 4.85)$ — same direction

Element-wise 는 큰 component 만 cap, 작은 component 는 그대로 — relative ratio 변경.

**Element-wise 가 나은 case**:

1. **Sparse gradient**: 한 dimension 이 dominate (단어 embedding 의 active token gradient). 그 dimension 만 cap, 다른 dim 영향 없음.

2. **Different scale per parameter**: Layer norm 없는 deep network 에서 layer 별 gradient scale 다름. Element-wise 가 layer 별 cap.

3. **Numerical stability**: Float16 에서 단일 dim overflow 방지. 다른 dim 은 정상 학습.

**실전**:
- 표준 NLP/CV: norm clipping
- Embedding-heavy task: 일부 element-wise 시도 가능
- Hybrid: per-tensor norm clipping (PyTorch `clip_grad_norm_` per tensor)

**결론**: Direction 보존이 *일반적으로* 우월, 그러나 task 별 sparse/scale 특성에 따라 element-wise 도 valid. $\square$

</details>

**문제 3** (논문 비평): Gradient clipping 이 RNN 의 *vanishing* 에는 도움이 되지 않는다. 왜 그런가? Vanishing 의 기본 해법 (LSTM, Transformer) 과 비교하라.

<details>
<summary>해설</summary>

**Clipping 은 magnitude 의 cap**:
- Exploding: $\|g\| > \theta$ 시 cap → 효과적
- Vanishing: $\|g\| \to 0$ → cap 할 것 없음, 그대로 0

**Vanishing 은 magnitude 자체가 0 으로 가는 것** — clipping 으로 막을 수 없음. 오히려 작은 gradient 에 *upscale* 가 필요한데 이는 noise 도 증폭.

**Vanishing 해법들**:

1. **Architectural** (root cause 해결):
   - **LSTM** (Ch4): Cell state 의 additive update — Jacobian 의 곱셈적 누적 우회
   - **Residual connection**: $h^{l+1} = h^l + f(h^l)$ — identity skip
   - **Transformer**: Attention 으로 direct connection — distance-independent gradient

2. **Initialization**:
   - Orthogonal init (Ch3-04): $\rho = 1$ 정확히
   - Identity init (Ch3-05): $W = I$ 시작
   - LSTM forget bias = 1 (Jozefowicz 2015): cell state 보존

3. **Optimization**:
   - Adam: adaptive learning rate, vanishing 영역에서 큰 step
   - LR warmup: 초기 작은 LR 으로 stable start

4. **Regularization**:
   - Layer norm: $z$ 의 scale 정규화 → saturation 회피
   - Dropout: gradient flow 다양화

**Clipping 의 역할**:
- *Symptom* (exploding) 만 치료
- Vanishing 은 *architectural* 해결 필요
- 둘 다: clipping + LSTM + Adam + clip = robust training

**비유**:
- Clipping: 댐 (dam) — exploding flood 막음
- LSTM: 새 물길 — vanishing drought 우회
- 둘 다 필요

**결론**: Clipping 은 단순하고 효과적이지만 *exploding only*. Vanishing 은 fundamental architectural 변경 필요. 이것이 LSTM 등이 deep learning 의 진화의 핵심인 이유. $\square$

</details>

---

<div align="center">

[◀ 이전](./02-saturation-problem.md) | [📚 README](../README.md) | [다음 ▶](./04-orthogonal-init.md)

</div>
