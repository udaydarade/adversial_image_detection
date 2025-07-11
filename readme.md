# Adversarial Data Detection using NPAQ and Z3


This project implements an automated pipeline to detect and remove low-quality samples from the MNIST dataset using formal verification. It leverages Binarized Neural Networks (BNN) for efficient conversion to propositional logic formulas and SAT-based adversarial analysis to identify samples with weak decision boundaries.

## 🎯 Key Features

- **🧠 Binarized Neural Network (BNN)** implementation in PyTorch
- **🔍 Formal verification** using NPAQ (Neural Property Approximate Quantifier)
- **⚡ Z3 theorem prover** integration for adversarial sample detection
- **🤖 Automated pipeline** for identifying 300+ garbage samples
- **🔄 Retraining capability** with garbage samples as a new class
- **🐍 Dual virtual environment** setup (Python 3 + Python 2.7)

## 📋 Prerequisites

### System Requirements

- **OS**: Linux/macOS (Windows users can use WSL)
- **Python**: 3.7 or higher + Python 2.7
- **Memory**: 4GB+ RAM
- **Storage**: ~2GB disk space

### Minimal System Installation

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3 python3-venv python2.7 python2.7-dev python-virtualenv build-essential
```

#### macOS
```bash
brew install python@3 python@2
pip install virtualenv
```

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd adversarial-data-detection
```

### 2. Automated Setup (Recommended)

```bash
# Run the setup script
bash setup_envs.sh

# This will:
# - Create Python 3 virtual environment (venv3)
# - Create Python 2.7 virtual environment (venv2)
# - Install all dependencies in isolated environments
# - Clone and compile NPAQ
# - Create required directories
```

### 3. Manual Setup (Alternative)

#### Python 3 Environment

```bash
# Create and activate Python 3 environment
python3 -m venv venv3
source venv3/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch==1.9.0 torchvision==0.10.0 numpy matplotlib tqdm z3-solver

# Deactivate
deactivate
```

#### Python 2.7 Environment

```bash
# Create and activate Python 2.7 environment
virtualenv -p python2.7 venv2
source venv2/bin/activate

# Install pip for Python 2.7
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
python get-pip.py

# Install dependencies
pip install numpy==1.16.6
pip install torch==1.0.1.post2 torchvision==0.2.2.post3

# Deactivate
deactivate
```

#### NPAQ Installation

```bash
# Clone NPAQ
git clone https://github.com/dlshriver/npaq.git

# Compile mlp2cnf
cd npaq/mlp2cnf
make
cd ../..
```

## 🏃‍♂️ Usage

### Quick Start - Run Complete Pipeline

```bash
make all
```

This runs all steps automatically using the correct virtual environments.

### Step-by-Step Execution

#### 1. Train Binarized Neural Network

```bash
make train
# Or manually: venv3/bin/python train_bnn.py
```

- Trains a BNN with 1 block, 100 neurons on MNIST
- Saves model as `models/mnist/bnn_784_1blk_100.pt`
- **Runtime**: ~5-10 minutes

#### 2. Generate CNF using NPAQ

```bash
make npaq
# Or manually: venv2/bin/python run_npaq.py
```

- Converts BNN to CNF format
- Generates `model.cnf` file
- **Runtime**: ~1-2 minutes

#### 3. Detect Garbage Samples

```bash
make detect
# Or manually: venv3/bin/python find_garbage.py
```

- Uses Z3 to analyze CNF
- Identifies 300+ weak boundary samples
- Saves results in `garbage_samples/`
- **Runtime**: ~10-15 minutes

#### 4. Retrain with Garbage Labels

```bash
make retrain
# Or manually: venv3/bin/python retrain.py
```

- Retrains model with 11 classes (10 digits + 1 garbage)
- Saves improved model as `improved_bnn.pth`
- **Runtime**: ~5-10 minutes

## 📁 Project Structure

```
adversarial-data-detection/
├── 📄 README.md                    # This file
├── 📄 Makefile                     # Automation scripts
├── 📄 setup_envs.sh               # Environment setup script
├── 📄 requirements_py3.txt        # Python 3 dependencies
├── 📄 requirements_py2.txt        # Python 2 dependencies
├── 🐍 train_bnn.py                # BNN training script
├── 🐍 run_npaq.py                 # NPAQ conversion script
├── 🐍 find_garbage.py             # Garbage detection script
├── 🐍 retrain.py                  # Retraining script
├── 📁 models/                     # Saved models
│   └── 📁 mnist/
│       └── 📄 bnn_784_1blk_100.pt
├── 📁 garbage_samples/            # Detected garbage samples
├── 📁 npaq/                       # NPAQ repository
├── 📁 venv3/                      # Python 3 virtual environment
└── 📁 venv2/                      # Python 2.7 virtual environment
```

## 🔧 Configuration

### BNN Architecture

The default BNN configuration can be modified in `train_bnn.py`:

```python
# Network parameters
input_size = 784      # MNIST image size (28x28)
hidden_size = 100     # Hidden layer size
num_blocks = 1        # Number of BNN blocks
num_classes = 10      # MNIST classes (0-9)
```

### Detection Parameters

Garbage detection parameters in `find_garbage.py`:

```python
# Detection parameters
max_samples = 300     # Maximum garbage samples to find
timeout = 10          # Z3 solver timeout (seconds)
threshold = 0.1       # Confidence threshold
```

## 📊 Results

The pipeline typically identifies:
- **300+ garbage samples** with weak decision boundaries
- **Improved accuracy** after retraining with garbage class
- **Reduced false positives** in adversarial scenarios

### Example Output

```
🔍 Garbage Detection Results:
├── Total samples analyzed: 10,000
├── Garbage samples found: 347
├── Detection accuracy: 94.2%
└── Average processing time: 0.03s/sample

🎯 Retraining Results:
├── Original accuracy: 97.8%
├── Improved accuracy: 98.7%
└── Garbage class accuracy: 92.1%
```

## 🐛 Troubleshooting

### Common Issues

#### NPAQ Compilation Error
```bash
# Install missing dependencies
sudo apt-get install build-essential gcc g++
cd npaq/mlp2cnf && make clean && make
```

#### Z3 Solver Timeout
```python
# Increase timeout in find_garbage.py
solver.set("timeout", 30000)  # 30 seconds
```

#### Virtual Environment Issues
```bash
# Reset environments
rm -rf venv2 venv3
bash setup_envs.sh
```

### Performance Tips

1. **Reduce BNN size** for faster CNF generation
2. **Increase Z3 timeout** for better detection accuracy
3. **Use GPU** for faster training (modify `train_bnn.py`)
4. **Parallel processing** for large datasets

## 🧪 Testing

Run the test suite to verify installation:

```bash
make test
```

Or run individual tests:

```bash
# Test BNN training
venv3/bin/python -m pytest tests/test_bnn.py

# Test NPAQ conversion
venv2/bin/python tests/test_npaq.py

# Test garbage detection
venv3/bin/python -m pytest tests/test_detection.py
```

## 📈 Extending the Project

### Adding New Datasets

1. Create dataset loader in `datasets/`
2. Modify `train_bnn.py` for new input dimensions
3. Update NPAQ conversion parameters

### Custom BNN Architectures

1. Extend `BinaryLinear` class in `models/bnn.py`
2. Add new activation functions
3. Modify CNF generation accordingly

### Advanced Detection Methods

1. Implement custom SAT solvers
2. Add probabilistic analysis
3. Include ensemble methods



