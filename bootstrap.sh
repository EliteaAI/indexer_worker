#!/bin/bash
set -x

(
  echo "Sandbox base: ${SANDBOX_BASE}"
  cd "${SANDBOX_BASE}"

  if [ ! -f "bin/deno" ]; then
    if [ "$(uname -m)" = "aarch64" ]; then
      curl -LOJ https://github.com/denoland/deno/releases/download/v2.6.8/deno-aarch64-unknown-linux-gnu.zip
      unzip deno-aarch64-unknown-linux-gnu.zip
      rm deno-aarch64-unknown-linux-gnu.zip
    else
      curl -LOJ https://github.com/denoland/deno/releases/download/v2.6.8/deno-x86_64-unknown-linux-gnu.zip
      unzip deno-x86_64-unknown-linux-gnu.zip
      rm deno-x86_64-unknown-linux-gnu.zip
    fi
    mkdir bin
    mv deno bin/deno
    chmod a+x bin/deno
  fi

  if [ ! -d "pyodide" ]; then
    curl -LOJ https://github.com/pyodide/pyodide/releases/download/0.29.3/pyodide-0.29.3.tar.bz2
    tar xf pyodide-0.29.3.tar.bz2
    rm pyodide-0.29.3.tar.bz2
    pip3 wheel -w pyodide dill chardet
  fi

  if [ ! -d ".nvm" ]; then
    echo "Node.js not found, installing via nvm..."

    # Create .nvm directory
    mkdir -p .nvm

    # Set NVM_DIR to install in sandbox
    export NVM_DIR="${SANDBOX_BASE}/.nvm"

    # Download and install nvm
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

    # Load nvm
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # Download and install Node.js
    nvm install 24

    # Verify the Node.js version
    node -v # Should print "v24.12.0"

    # Verify npm version
    npm -v # Should print "11.6.2"
  else
    echo "Node.js is already installed in sandbox"
    # Load nvm
    export NVM_DIR="${SANDBOX_BASE}/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
  fi
) || true


MCP_CONFIG="/data/plugins/indexer_worker/config.yml"
echo "Installing MCP server packages from: $MCP_CONFIG"
if command -v yq &> /dev/null; then
  # Use yq for YAML parsing
  # Install npm packages
  yq -r '.mcp_servers | to_entries[] | select(.value.runtime == "npm") | .value.package' "$MCP_CONFIG" 2>/dev/null | while read -r pkg; do
    if [ -n "$pkg" ] && [ "$pkg" != "null" ]; then
      echo "Pre-caching npm package: $pkg"
      npx -y "$pkg" --help >/dev/null 2>&1 || true
    fi
  done

  # Install python packages
  yq -r '.mcp_servers | to_entries[] | select(.value.runtime == "python") | .value.package' "$MCP_CONFIG" 2>/dev/null | while read -r pkg; do
    if [ -n "$pkg" ] && [ "$pkg" != "null" ]; then
      echo "Installing python package: $pkg"
      pip3 install "$pkg" || true
    fi
  done
fi

echo "Bootstrap script completed"
