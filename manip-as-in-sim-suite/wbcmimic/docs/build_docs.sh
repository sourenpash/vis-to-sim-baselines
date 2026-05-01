#!/bin/bash
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


# Build documentation with MkDocs
# This script builds the HTML documentation from the Markdown files

echo "Building UniMimic documentation..."

# Change to docs directory
cd "$(dirname "$0")"

# Install MkDocs and dependencies if not already installed
pip install mkdocs mkdocs-material "mkdocstrings[python]" pymdown-extensions

# Build the documentation
mkdocs build

echo "Documentation built successfully!"
echo "HTML files are in the 'site' directory"
echo ""
echo "To serve the documentation locally, run:"
echo "  cd docs && mkdocs serve"
echo ""
echo "The documentation will be available at http://localhost:8000"