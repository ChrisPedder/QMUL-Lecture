#!/bin/bash

# Setup script for Bayesian Neural Network Demo
echo "Setting up Bayesian NN Demo environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Detected macOS"
else
    print_warning "This script is optimized for macOS but will attempt to run anyway"
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    print_status "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    # Verify installation
    if ! command -v uv &> /dev/null; then
        print_error "Failed to install uv. Please install manually from https://github.com/astral-sh/uv"
        exit 1
    fi
else
    print_status "uv is already installed"
fi

# Remove old virtual environment if it exists
if [ -d ".venv" ]; then
    print_warning "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.12
print_status "Creating virtual environment with Python 3.12..."
uv venv --python 3.12

# Check if venv was created successfully
if [ ! -d ".venv" ]; then
    print_error "Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies using requirements.txt (simpler approach)
print_status "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
else
    print_warning "requirements.txt not found, installing dependencies individually..."

    # Install core dependencies
    uv pip install tensorflow>=2.15.0
    uv pip install tf-keras>=2.15.0
    uv pip install "tensorflow-probability[tf]>=0.23.0"
    uv pip install numpy>=1.24.0
    uv pip install matplotlib>=3.7.0
    uv pip install seaborn>=0.12.0
    uv pip install jupyter notebook ipykernel>=6.25.0
    uv pip install scikit-learn>=1.3.0
    uv pip install pandas>=2.0.0
    uv pip install tqdm>=4.65.0
    uv pip install ipywidgets>=8.0.0
fi

# Verify key packages are installed
print_status "Verifying installation..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "TensorFlow installation failed"
    print_warning "Trying alternative installation method..."
    pip install tensorflow tf-keras "tensorflow-probability[tf]"
fi

# Install Jupyter kernel
print_status "Installing Jupyter kernel..."
python -m ipykernel install --user --name bayesian-nn --display-name "Bayesian NN (Python 3.12)"

# Create a launcher script
print_status "Creating launcher script..."
cat > launch_notebook.sh << 'EOF'
#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Check if jupyter is available
if ! command -v jupyter &> /dev/null; then
    echo "Jupyter not found in virtual environment. Installing..."
    pip install jupyter notebook
fi

# Launch Jupyter notebook
echo "Launching Jupyter Notebook..."
jupyter notebook bayesian_nn_mnist_demo.ipynb
EOF

chmod +x launch_notebook.sh

# Print instructions
echo ""
print_status "Setup complete!"
echo ""
echo "================================================================"
echo "To run the notebook, use ONE of these methods:"
echo ""
echo "METHOD 1 (Recommended): Use the launcher script"
echo "  ./launch_notebook.sh"
echo ""
echo "METHOD 2: Manual activation"
echo "  source .venv/bin/activate"
echo "  jupyter notebook bayesian_nn_mnist_demo.ipynb"
echo ""
echo "METHOD 3: Direct Python module call"
echo "  source .venv/bin/activate"
echo "  python -m notebook bayesian_nn_mnist_demo.ipynb"
echo "================================================================"
echo ""
print_warning "Note: Always use the virtual environment's jupyter, not the system one!"
