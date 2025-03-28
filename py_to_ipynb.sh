#!/bin/bash

# Convert all Python scripts to py:percent format jupyter notebooks
echo "Converting Python files to py:percent formatted notebooks..."

# Find all Python files in the current directory only (no recursion) and convert them to notebook files
find . -maxdepth 1 -name "*.py" -exec jupytext --from py:percent --to notebook {} \;

echo "Conversion complete!"
