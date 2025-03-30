#! /bin/bash

# Compile the requirements.txt file
pip-compile --output-file=../requirements/requirements.txt ../pyproject.toml
pip-compile --extra=dev --output-file=../requirements/requirements-dev.txt ../pyproject.toml

# Install the requirements
pip-sync ../requirements/requirements-dev.txt

# Install the project
pip install -e ..
