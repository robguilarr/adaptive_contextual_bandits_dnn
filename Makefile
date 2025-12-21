PROJECT_ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_DIR := $(PROJECT_ROOT)/venv
PIP := $(VENV_DIR)/bin/pip

include .env
export $(shell sed 's/=.*//' .env)

.PHONY: venv install clean

venv:
	@echo "Creating virtual environment at $(VENV_DIR)..."
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created."

install: venv
	@echo "Activating virtual environment and installing dependencies..."
	@source $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip setuptools wheel && \
		$(PIP) install -e $(PROJECT_ROOT)
	@echo "Dependencies installed."

clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Cleanup complete."
