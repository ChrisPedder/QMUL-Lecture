# How Machines Learn, Where They Struggle, and What Animals Do Differently

## Overview

This repository contains educational materials for a lecture exploring machine learning fundamentals, the limitations of standard deep learning approaches (particularly around uncertainty quantification), and how biological systems learn using principles from the Free Energy Principle.

The materials are designed for teaching and include interactive notebooks, visualizations, and presentation slides.

## Key Learning Objectives

- Understand how neural networks learn through backpropagation and gradient descent
- Recognize the limitations of standard DNNs, particularly overconfidence on out-of-distribution data
- Learn how Bayesian Neural Networks quantify uncertainty
- Explore the Free Energy Principle as a model for biological learning and behavior
- Compare machine learning approaches with biological intelligence

## Repository Structure

```
.
├── LICENSE
├── README.md
├── pyproject.toml                      # Project dependencies
├── requirements.txt                    # Alternative dependency list
├── uv.lock                             # Locked dependencies
├── setup.sh                            # Environment setup script
├── launch_bayesian_notebook.sh         # Quick launcher for BNN demo
│
├── notebooks/
│   ├── visualizations.ipynb            # Neural network visualizations & animations
│   ├── bayesian_nn_mnist_demo.ipynb    # Bayesian vs Standard DNN comparison
│   ├── bacterium_visualization.ipynb   # FEP bacterium agent visualization
│   └── outputs/                        # Generated figures and animations
│       ├── backpropagation.gif
│       ├── curve_fitting_mse.gif
│       ├── neural_network_forward_pass.gif
│       ├── single_layer_neural_network.png
│       ├── single_layer_nn_simple.png
│       ├── vgg16_architecture.png
│       └── vgg16_architecture_detailed.png
│
├── free_energy_simulations/
│   ├── fep_agent_simulation.py         # FEP agent in 3D environment
│   └── fep_simulation_results.png      # Simulation output visualization
│
└── slides/
    ├── index.html                      # Main presentation (Reveal.js)
    ├── part_a.md                       # Part A: ML Fundamentals
    ├── part_b.md                       # Part B: Uncertainty & FEP
    ├── README.md                       # Slides documentation
    ├── github_pages_setup.md           # Deployment instructions
    └── images/                         # Slide assets
        ├── ai_hierarchy.png
        ├── backpropagation.gif
        ├── curve_fitting_mse.gif
        ├── gaussian_weights_pdf.png
        ├── neural_network_forward_pass.gif
        ├── softmax_visualization.png
        ├── step0_base_bacterium.png
        ├── step1_with_sensors.png
        ├── step2_with_actuators.png
        ├── step3_complete.png
        └── vgg16_architecture.png
```

## Content Overview

### Part A: Machine Learning Fundamentals

**Notebooks**: `visualizations.ipynb`, `bayesian_nn_mnist_demo.ipynb`

- Neural network architecture and forward propagation
- Loss functions and optimization
- Backpropagation and gradient descent
- Animated visualizations of learning dynamics

### Part B: Uncertainty and the Free Energy Principle

**Notebooks**: `bacterium_visualization.ipynb`  
**Simulations**: `fep_agent_simulation.py`

#### Bayesian Neural Networks
- Comparison of standard DNNs vs BNNs on MNIST
- Out-of-distribution detection (trained on digits 0-4, tested on 5-9)
- Uncertainty quantification through weight distributions
- Robustness to noisy inputs

#### Free Energy Principle
- FEP agent navigating a 3D environment
- Perception as inference (belief updating)
- Active inference (action selection to minimize surprisal)
- Biological plausibility of predictive processing

## Setup Instructions

### Prerequisites

- Python 3.12 or higher
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd QMUL-Lecture
   ```

2. **Run the setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   Or manually with UV:
   ```bash
   # Install UV if needed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment
   uv venv --python 3.12
   source .venv/bin/activate

   # Install dependencies
   uv pip install -e .

   # Install Jupyter kernel
   python -m ipykernel install --user --name bayesian-nn --display-name "Bayesian NN (Python 3.12)"
   ```

3. **Launch notebooks**
   ```bash
   # For the Bayesian NN demo specifically
   ./launch_bayesian_notebook.sh

   # Or open any notebook
   jupyter notebook notebooks/
   ```

4. **Run the FEP simulation**
   ```bash
   python free_energy_simulations/fep_agent_simulation.py
   ```

### Viewing the Slides

The slides are built with Reveal.js and can be viewed locally or deployed to GitHub Pages:

```bash
cd slides
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

See `slides/github_pages_setup.md` for deployment instructions.

## Key Concepts Demonstrated

### Standard Deep Neural Networks
- Point estimates for weights
- Deterministic predictions
- Often overconfident on unfamiliar data
- No inherent uncertainty quantification

### Bayesian Neural Networks
- Probability distributions over weights
- Epistemic uncertainty through posterior sampling
- Better calibrated confidence on OOD data
- Natural framework for "knowing what you don't know"

### Free Energy Principle
- Generative models of the environment
- Variational free energy as a bound on surprise
- Perception as minimizing prediction error
- Action as making predictions come true
- Unifying framework for perception, learning, and action

## Expected Learning Outcomes

After working through these materials, students will be able to:

1. **Explain** how gradient descent and backpropagation enable neural network learning
2. **Identify** situations where standard DNNs may fail (OOD data, adversarial examples)
3. **Compare** uncertainty estimates from BNNs vs confidence from standard DNNs
4. **Describe** the Free Energy Principle and its components (generative model, free energy, active inference)
5. **Discuss** how biological systems might implement predictive processing

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GPU memory errors | Reduce batch size in training functions |
| Slow BNN training | BNNs require multiple forward passes; reduce epochs for demos |
| Import errors | Verify installation with `uv pip list` or `pip list` |
| Slides not loading | Ensure you're serving via HTTP, not opening file directly |
| FEP simulation diverging | Check that numerical stability fixes are applied |

## Further Reading

### Bayesian Deep Learning
- [Weight Uncertainty in Neural Networks (Blundell et al., 2015)](https://arxiv.org/abs/1505.05424)
- [Practical Variational Inference for Neural Networks (Graves, 2011)](https://papers.nips.cc/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html)
- [TensorFlow Probability Documentation](https://www.tensorflow.org/probability)

### Free Energy Principle
- [The Free Energy Principle for Action and Perception (Friston et al., 2017)](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20unified%20brain%20theory.pdf)
- [Active Inference: A Process Theory (Friston et al., 2017)](https://direct.mit.edu/neco/article/29/1/1/8207/Active-Inference-A-Process-Theory)
- [A Tutorial on Active Inference (Da Costa et al., 2020)](https://arxiv.org/abs/2001.07203)

### General
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/)
- [Bayesian Reasoning and Machine Learning (Barber)](http://www.cs.ucl.ac.uk/staff/d.barber/brml/)

## License

This educational material is provided for teaching purposes. See LICENSE file for details.

## Acknowledgments

Developed for QMUL lecture on machine learning, uncertainty, and biological intelligence.
