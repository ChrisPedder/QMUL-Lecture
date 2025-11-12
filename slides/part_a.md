# Part A: Introduction to Artificial Intelligence

**QMUL Lecture Series**

*Press 'S' for speaker notes • Arrow keys to navigate*

---

## Learning Objectives

By the end of Part A, you will be able to:

- Define key concepts in artificial intelligence
- Understand the historical context of AI development
- Identify different paradigms in AI
- Recognize the relationship between AI, ML, and Deep Learning

Note: Welcome to Part A. Today we'll cover foundational concepts that will be essential for the rest of the course.

---

## What is Artificial Intelligence?

<span class="key-term">Artificial Intelligence (AI)</span> is the simulation of human intelligence processes by machines, especially computer systems.

**Key processes include:**

- Learning (acquiring information and rules)
- Reasoning (using rules to reach conclusions)
- Self-correction

Note: Start with a broad definition. Ask students what they think AI means before revealing this slide.

----

### The AI Hierarchy

```
┌─────────────────────────────────┐
│   Artificial Intelligence       │  ← Broad field
│  ┌───────────────────────────┐  │
│  │   Machine Learning        │  │  ← Learning from data
│  │  ┌─────────────────────┐  │  │
│  │  │  Deep Learning      │  │  │  ← Neural networks
│  │  └─────────────────────┘  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

<span class="small">Each inner layer is a subset of the outer layer</span>

---

## Brief History of AI

**1950s:** The Birth
- Alan Turing's "Computing Machinery and Intelligence"
- Dartmouth Conference (1956) - AI formally founded

**1960s-70s:** Early Enthusiasm
- Expert systems and symbolic AI

**1980s-90s:** First AI Winter → Revival
- Machine learning emerges as dominant paradigm

**2000s-Present:** Deep Learning Revolution
- Big data + computational power = breakthrough results

Note: Emphasize that AI has gone through cycles of hype and disappointment, known as "AI winters"

---

## AI Paradigms

<div class="columns">
<div class="column">

### Symbolic AI
("Good Old-Fashioned AI")

- Rule-based systems
- Logic and reasoning
- Knowledge representation
- Expert systems

</div>
<div class="column">

### Connectionist AI
(Machine Learning)

- Learning from data
- Statistical methods
- Neural networks
- Pattern recognition

</div>
</div>

**Modern AI:** Combines both approaches!

---

## Machine Learning: Core Concept

<span class="emphasis">Instead of programming rules explicitly, we learn patterns from data</span>

**Traditional Programming:**
```python
def is_spam(email):
    if "winner" in email or "free money" in email:
        return True
    return False
```

**Machine Learning:**
```python
model = train(spam_data, not_spam_data)
prediction = model.predict(new_email)
```

Note: This is a critical distinction. In ML, we don't hand-code the rules.

----

### Types of Machine Learning

**Supervised Learning**
- Learn from labeled data
- Classification and regression
- Example: Image classification, price prediction

**Unsupervised Learning**
- Find patterns in unlabeled data
- Clustering and dimensionality reduction
- Example: Customer segmentation

**Reinforcement Learning**
- Learn through trial and error
- Agent, environment, rewards
- Example: Game playing, robotics

---

## Mathematical Foundations

A simple linear model:

$$y = wx + b$$

Where:
- $y$ is the prediction
- $x$ is the input feature
- $w$ is the weight (learned parameter)
- $b$ is the bias (learned parameter)

**Learning = Finding optimal $w$ and $b$**

Note: Don't worry if students aren't comfortable with math yet - we'll build this up gradually.

----

### The Learning Process

1. **Initialize:** Start with random weights
2. **Forward Pass:** Make predictions
3. **Calculate Loss:** How wrong are we?
4. **Backward Pass:** Compute gradients
5. **Update:** Adjust weights to reduce loss
6. **Repeat:** Until convergence

This is <span class="key-term">gradient descent</span>!

---

## Neural Networks: Basics

<div class="columns">
<div class="column">

**Biological Inspiration:**
- Neurons receive signals
- Process information
- Send output to other neurons

</div>
<div class="column">

**Artificial Neurons:**
- Receive inputs $x_i$
- Apply weights $w_i$
- Sum: $z = \sum w_i x_i + b$
- Activate: $a = f(z)$

</div>
</div>

$$\text{output} = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

Note: The activation function f introduces non-linearity, allowing networks to learn complex patterns.

---

## Why Deep Learning Now?

Three key enablers:

1. **Big Data**
   - Internet-scale datasets
   - Millions of labeled examples

2. **Computational Power**
   - GPUs and specialized hardware
   - Cloud computing

3. **Algorithmic Innovations**
   - Better architectures (CNNs, Transformers)
   - Improved training techniques

<span class="emphasis">All three are necessary!</span>

---

## Key Takeaways: Part A

- AI encompasses multiple approaches and paradigms
- Machine learning learns patterns from data rather than explicit rules
- Deep learning is a subset of ML using neural networks
- Modern success requires: data, computation, and algorithms
- Understanding the fundamentals is crucial for advanced topics

**Coming up in Part B:** Practical applications and modern architectures

Note: Take questions before moving to Part B. This is a good break point.

---

## Questions?

<div style="margin-top: 2em; font-size: 0.9em;">

**Resources:**
- Course repository: github.com/[your-username]/QMUL-Lecture
- Office hours: [Your time]
- Email: [Your email]

</div>
