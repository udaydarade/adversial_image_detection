# Makefile for Adversarial Data Detection with NPAQ

PYTHON3 := python3
PYTHON2 := python2.7
VENV3 := venv
PIP3 := $(VENV3)/bin/pip
PYTHON3_VENV := $(VENV3)/bin/python

.PHONY: all setup clone-npaq train npaq detect retrain clean help

all: setup train npaq detect retrain

help:
	@echo "Adversarial Data Detection Pipeline"
	@echo "==================================="
	@echo "  make setup     - Set up environment and dependencies"
	@echo "  make train     - Train BNN model with PyTorch"
	@echo "  make npaq      - Run NPAQ to generate CNF"
	@echo "  make detect    - Detect garbage samples using Z3"
	@echo "  make retrain   - Retrain with garbage labels"
	@echo "  make all       - Run complete pipeline"
	@echo "  make clean     - Clean generated files"

setup:
	@echo "Setting up Python 3 environment..."
	@$(PYTHON3) -m venv $(VENV3)
	@$(PIP3) install --upgrade pip
	@$(PIP3) install -r requirements.txt
	@echo "Python 3 environment ready!"
	@echo ""
	@echo "For NPAQ (Python 2.7), please ensure you have:"
	@echo "1. Python 2.7 installed"
	@echo "2. Run: pip2 install -r requirements-npaq.txt"
	@echo "3. Clone NPAQ: git clone https://github.com/Shivvrat/npaq.git"

clone-npaq:
	@if [ ! -d "npaq" ]; then \
		echo "Cloning NPAQ repository..."; \
		git clone https://github.com/Shivvrat/npaq.git; \
		cd npaq/mlp2cnf && make; \
	else \
		echo "NPAQ already exists"; \
	fi

train:
	@echo "Training BNN model..."
	@$(PYTHON3_VENV) train_bnn.py

npaq:
	@echo "Running NPAQ to generate CNF..."
	@$(PYTHON2) run_npaq.py

detect:
	@echo "Detecting garbage samples..."
	@$(PYTHON3_VENV) find_garbage.py

retrain:
	@echo "Retraining with garbage labels..."
	@$(PYTHON3_VENV) retrain.py

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV3)
	@rm -rf __pycache__ *.pyc
	@rm -f *.pth *.pt *.cnf *.npz
	@rm -rf data/MNIST garbage_samples/
	@rm -rf models/
	@rm -f cnf_path.txt
	@echo "Cleanup complete!"

# Additional helpful targets
test-npaq:
	@echo "Testing NPAQ installation..."
	@cd npaq && python2.7 npaq --help

download-data:
	@echo "Downloading MNIST dataset..."
	@$(PYTHON3_VENV) -c "from torchvision import datasets; datasets.MNIST('./data', download=True)"