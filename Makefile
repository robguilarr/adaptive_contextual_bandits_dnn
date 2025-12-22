PROJECT_ROOT := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_DIR := $(PROJECT_ROOT)/venv
PIP := $(VENV_DIR)/bin/pip

# Include .env if it exists
-include .env
ifneq ($(wildcard .env),)
	export $(shell sed 's/=.*//' .env)
endif

.PHONY: venv install clean

venv:
	@echo "Creating virtual environment at $(VENV_DIR)..."
	@python3.10 --version || (echo "Error: python3.10 not found. Please install Python 3.10." && exit 1)
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Removing existing virtual environment..."; \
		rm -rf $(VENV_DIR); \
	fi
	@python3.10 -m venv $(VENV_DIR)
	@echo "Virtual environment created with Python 3.10."
	@$(VENV_DIR)/bin/python --version

install: venv
	@echo "Activating virtual environment and installing dependencies..."
	@$(VENV_DIR)/bin/python --version
	@source $(VENV_DIR)/bin/activate && \
		$(PIP) install --upgrade pip setuptools wheel && \
		$(PIP) install -e $(PROJECT_ROOT)
	@echo "Dependencies installed."

clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Cleanup complete."
