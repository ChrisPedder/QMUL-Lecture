# Part B: Modern AI Architectures

**From Theory to Practice**

---

## Agenda: Part B

- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers and Attention
- Bayesian Neural Networks *(linking to your notebook!)*
- Practical considerations and challenges

Note: Part B builds on the foundations from Part A and connects to the practical work in the course notebooks.

---

## Convolutional Neural Networks

**Problem:** Traditional neural networks don't handle images well

**Why?**
- Images have spatial structure
- Too many parameters for fully connected layers
- Need translation invariance

**Solution:** <span class="key-term">Convolution</span> operations

Note: CNNs revolutionized computer vision starting with AlexNet in 2012.

----

### Convolution Operation

A filter (kernel) slides across the input:

```
Input Image:        Filter:          Output:
┌─────────┐        ┌─────┐         ┌─────┐
│ 1 2 3 4 │        │ 1 0 │         │  •  │
│ 5 6 7 8 │    ×   │ 0 1 │    =    │  •  │
│ 9 0 1 2 │        └─────┘         └─────┘
└─────────┘
```

**Key idea:** Share weights across spatial locations

<span class="small">Filter learns to detect features (edges, textures, patterns)</span>

----

### CNN Architecture Layers

1. **Convolutional Layers:** Feature extraction
2. **Pooling Layers:** Dimensionality reduction
3. **Fully Connected Layers:** Classification

**Example: Image Classification**

```python
Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
     ↓
Conv2D(64, 3×3) → ReLU → MaxPool(2×2)
     ↓
Flatten → Dense(128) → Dense(10)
```

---

## Recurrent Neural Networks

**Problem:** Need to process sequential data
- Text, time series, speech
- Fixed-size neural networks don't have memory

**Solution:** <span class="key-term">Recurrence</span> - feed outputs back as inputs

----

### RNN Structure

At each time step $t$:

$$h_t = f(W_h h_{t-1} + W_x x_t + b)$$

Where:
- $h_t$ is the hidden state at time $t$
- $x_t$ is the input at time $t$
- $W_h, W_x$ are learned weight matrices

**Key insight:** Hidden state $h_t$ carries information from previous steps

Note: This allows the network to have "memory" of what came before.

----

### RNN Variants

**Problems with vanilla RNNs:**
- Vanishing gradients
- Difficulty learning long-term dependencies

**Solutions:**

<div class="columns">
<div class="column">

**LSTM**
(Long Short-Term Memory)
- Forget gate
- Input gate
- Output gate

</div>
<div class="column">

**GRU**
(Gated Recurrent Unit)
- Simpler than LSTM
- Update gate
- Reset gate

</div>
</div>

<span class="emphasis">Both solve long-term dependency problems</span>

---

## The Transformer Revolution

**2017:** "Attention Is All You Need" paper

**Key innovation:** Replace recurrence with <span class="key-term">self-attention</span>

**Advantages:**
- Parallel processing (much faster training)
- Better at capturing long-range dependencies
- State-of-the-art in NLP and beyond

Note: Transformers are the foundation of modern LLMs like GPT and BERT.

----

### Attention Mechanism

**Core idea:** Different inputs should get different amounts of "attention"

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = Queries (what we're looking for)
- $K$ = Keys (what's available)
- $V$ = Values (what we retrieve)

**Intuition:** Similar to a lookup in a dictionary, but soft and differentiable

----

### Transformer Applications

**Natural Language Processing:**
- GPT series (generative)
- BERT (understanding)
- T5 (text-to-text)

**Computer Vision:**
- Vision Transformers (ViT)
- Surpassing CNNs on some tasks

**Multi-modal:**
- CLIP (vision + language)
- Stable Diffusion (text → image)

<span class="emphasis">Transformers are now the default architecture for many tasks</span>

---

## Bayesian Neural Networks

**Connection to your notebook:** `bayesian_nn_mnist_demo.ipynb`

**Standard Neural Networks:**
- Point estimates for weights
- No uncertainty quantification

**Bayesian Neural Networks:**
- Probability distributions over weights
- Output includes confidence estimates

Note: This connects directly to the practical work you'll be doing.

----

### Why Bayesian Approaches?

<div class="columns">
<div class="column">

**Standard NN:**
```python
prediction = model(x)
# Just a single number
```

</div>
<div class="column">

**Bayesian NN:**
```python
mean, variance = model(x)
# Prediction + uncertainty
```

</div>
</div>

**Use cases:**
- Medical diagnosis (critical decisions)
- Autonomous vehicles (safety)
- Scientific discovery (exploratory)

<span class="key-term">Knowing when you don't know is crucial!</span>

----

### Bayesian Inference Challenges

**The problem:** Computing posterior distribution is intractable

$$p(w|D) = \frac{p(D|w)p(w)}{p(D)}$$

**Solutions:**
- Variational Inference (used in your notebook)
- Monte Carlo Dropout
- Sampling methods (MCMC)

**Trade-off:** Computational cost vs. uncertainty quality

---

## Practical Considerations

### Training Deep Networks

**Challenges:**
1. Vanishing/exploding gradients
2. Overfitting
3. Computational resources
4. Hyperparameter tuning

**Solutions:**
- Careful initialization
- Batch normalization
- Dropout and regularization
- Learning rate scheduling
- Data augmentation

Note: These are critical for making deep learning work in practice.

----

### The Overfitting Problem

<div class="columns">
<div class="column">

**Training error decreases**
**Test error increases**

*Model memorizes training data*

</div>
<div class="column">

**Regularization techniques:**
- L1/L2 regularization
- Dropout
- Early stopping
- Data augmentation
- More training data

</div>
</div>

**Goal:** Models that <span class="emphasis">generalize</span> to new data

----

### Data Considerations

**Quality over Quantity:**
- Clean, representative data
- Balanced classes
- Proper train/validation/test splits

**Data Augmentation Examples:**
- Images: rotation, flipping, cropping
- Text: back-translation, synonym replacement
- Audio: time stretching, pitch shifting

**Remember:** <span class="emphasis">Garbage in, garbage out</span>

---

## Modern AI Pipeline

```python
# 1. Data Collection & Preparation
data = load_and_clean_data()
train, val, test = split_data(data)

# 2. Model Definition
model = create_model(architecture="transformer")

# 3. Training
model.fit(train, validation_data=val,
          epochs=100, callbacks=[early_stopping])

# 4. Evaluation
metrics = model.evaluate(test)

# 5. Deployment
serve_model(model, endpoint="/predict")
```

Note: This is a simplified view, but captures the essential workflow.

---

## Ethical Considerations

**AI systems can:**
- Amplify biases present in training data
- Lack transparency ("black boxes")
- Have environmental costs (large models)
- Raise privacy concerns

**Our responsibility:**
- Be aware of limitations
- Test for bias and fairness
- Document model behavior
- Consider societal impact

<span class="emphasis">Technical excellence is not enough</span>

---

## Current Frontiers

**Active Research Areas:**

1. **Few-shot learning:** Learn from limited examples
2. **Transfer learning:** Leverage pre-trained models
3. **Federated learning:** Privacy-preserving training
4. **Neural architecture search:** Automated model design
5. **Explainable AI:** Understanding model decisions
6. **Multimodal learning:** Combining vision, language, etc.

Note: Many opportunities for research and innovation!

---

## Connecting Theory to Practice

**Your Course Work:**

1. **Jupyter Notebooks:** Hands-on implementations
   - `bayesian_nn_mnist_demo.ipynb` - Uncertainty quantification
   - More to come!

2. **GitHub Repository:** Code examples and resources

3. **Assignments:** Apply these concepts to real problems

**Key Skill:** Moving from papers to working code

Note: Encourage students to experiment with the notebooks and modify them.

---

## Resources for Deep Dive

**Books:**
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Pattern Recognition and Machine Learning" by Bishop

**Online Courses:**
- Fast.ai Practical Deep Learning
- Stanford CS231n (Computer Vision)
- Stanford CS224n (NLP)

**Papers:**
- Attention Is All You Need (Transformers)
- AlexNet, ResNet, BERT, GPT papers

**Tools:**
- PyTorch, TensorFlow
- Hugging Face Transformers
- Weights & Biases (experiment tracking)

---

## Summary: Parts A & B

**Part A - Foundations:**
- AI, ML, and Deep Learning hierarchy
- Learning paradigms and mathematical basis
- Historical context and current enablers

**Part B - Modern Approaches:**
- CNNs for spatial data (images)
- RNNs and LSTMs for sequences
- Transformers as current state-of-the-art
- Bayesian approaches for uncertainty
- Practical and ethical considerations

<span class="key-term">You now have the foundation to dive deep into AI!</span>

---

## Next Steps

**Before Next Lecture:**
1. Review these slides (available on GitHub)
2. Run `bayesian_nn_mnist_demo.ipynb`
3. Read: Chapter 1-3 of course textbook
4. Think about project ideas

**Coming Next:**
- Advanced optimization techniques
- Hands-on: Building your first transformer
- Guest lecture: AI in industry

---

## Thank You!

### Questions & Discussion

<div style="margin-top: 3em;">

**Stay Connected:**
- 📧 Email: [your.email@qmul.ac.uk]
- 💻 GitHub: [github.com/[username]/QMUL-Lecture]
- 🕐 Office Hours: [Day/Time]
- 💬 Course Forum: [link]

</div>

<div style="margin-top: 2em; font-size: 0.8em; color: #7f8c8d;">
Slides created with Reveal.js • Press 'S' for speaker notes • Press '?' for keyboard shortcuts
</div>
