# Bayesian Neural Networks vs Standard DNNs: MNIST Classification Demo

## Overview
This project provides an educational demonstration of the key differences between standard Deep Neural Networks (DNNs) and Bayesian Neural Networks (BNNs) for classification tasks, with a focus on uncertainty quantification and out-of-distribution detection.

## Key Learning Objectives
- Understand the difference in architecture between DNNs and BNNs
- Observe how uncertainty is quantified in Bayesian approaches
- Compare model behavior on in-distribution vs out-of-distribution data
- Analyze robustness to noisy inputs
- Understand the computational trade-offs

## Setup Instructions

### Prerequisites
- Python 3.12 or higher
- UV package manager (will be installed if not present)

### Installation Steps

1. **Clone or download this project**

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   Or manually:
   ```bash
   # Install UV if needed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment with Python 3.12
   uv venv --python 3.12
   
   # Activate environment
   source .venv/bin/activate
   
   # Install dependencies
   uv pip install -e .
   
   # Install Jupyter kernel
   python -m ipykernel install --user --name bayesian-nn --display-name "Bayesian NN (Python 3.12)"
   ```

3. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook bayesian_nn_mnist_demo.ipynb
   ```

4. **Select the correct kernel:**
   - In Jupyter, go to Kernel → Change kernel → Bayesian NN (Python 3.12)

## Project Structure

```
.
├── pyproject.toml                    # Project dependencies
├── setup.sh                          # Setup script
├── bayesian_nn_mnist_demo.ipynb     # Main notebook
└── README.md                         # This file
```

## Notebook Contents

The notebook is organized into the following sections:

1. **Data Loading & Visualization**: Download and explore MNIST dataset
2. **Data Preparation**: Filter digits 0-4 for training, 5-9 for OOD testing
3. **Standard DNN Training**: Train a traditional neural network classifier
4. **DNN Testing**: Evaluate on in-distribution and out-of-distribution data
5. **Noisy Data Creation**: Generate noisy versions of test data
6. **Bayesian NN Training**: Train a Bayesian neural network using TensorFlow Probability
7. **BNN Testing**: Evaluate with uncertainty quantification
8. **Comparative Analysis**: Direct comparison of DNN vs BNN behavior
9. **Summary & Insights**: Key takeaways and practical implications

## Key Concepts Demonstrated

### Standard Deep Neural Networks
- Single point estimates for weights
- Deterministic predictions
- May be overconfident on unfamiliar data

### Bayesian Neural Networks
- Probability distributions over weights
- Uncertainty quantification through multiple forward passes
- Better calibrated confidence estimates
- Natural out-of-distribution detection

## Expected Outcomes

Students will observe:
- **Higher confidence gap** in BNNs between in-distribution and OOD data
- **Better calibrated uncertainty** in BNNs when facing noisy inputs
- **Overconfidence issues** in standard DNNs on OOD samples
- **Computational cost** differences between the two approaches

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**: Reduce batch size in training functions
2. **Installation Problems**: Ensure Python 3.12 is installed
3. **Slow Training**: BNNs naturally take longer; consider reducing epochs
4. **Import Errors**: Verify all dependencies installed with `uv pip list`

## Further Reading

- [Weight Uncertainty in Neural Networks (Blundell et al., 2015)](https://arxiv.org/abs/1505.05424)
- [TensorFlow Probability Documentation](https://www.tensorflow.org/probability)
- [Practical Variational Inference for Neural Networks (Graves, 2011)](https://papers.nips.cc/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html)

## License
This educational material is provided for teaching purposes.
