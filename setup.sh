#!/usr/bin/env bash
# Set up the repository after cloning.
# Creates symlinks so that `import breaching` and `from core.models import ...`
# resolve to the vendored Geminio submodule, then patches in our custom attack
# configs alongside the upstream ones.
set -euo pipefail

cd "$(dirname "$0")"

# Initialize submodule if needed
git submodule update --init --recursive

# Symlink breaching into the repo root so `import breaching` works
if [ ! -e breaching ]; then
    ln -s vendor/Geminio/breaching breaching
    echo "Created symlink: breaching -> vendor/Geminio/breaching"
fi

# Copy our custom attack configs into the breaching config directory
for cfg in configs/attack/*.yaml; do
    dest="vendor/Geminio/breaching/config/attack/$(basename "$cfg")"
    cp "$cfg" "$dest"
    echo "Installed attack config: $(basename "$cfg")"
done

echo "Setup complete."
