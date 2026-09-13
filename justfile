# Default recipe
default: help

# Show available commands
help:
    @just --list

# Install git hooks (pre-commit) for the whole repo
install-hooks:
    uvx pre-commit install

# Run all pre-commit hooks against every file in the repo
lint-all:
    uvx pre-commit run --all-files
