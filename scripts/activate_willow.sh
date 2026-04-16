#!/usr/bin/env bash
# =============================================================================
# Willow WBT — ecosystem activation
#
# Source this file (do NOT run it) to activate the willow ecosystem:
#
#   source scripts/activate_willow.sh
#
# What it does:
#   - Initializes ~/.willow_deps/miniconda3 as the active conda
#   - Registers all three ecosystems in envs_dirs so conda env list shows everything:
#       ~/.willow_deps/miniconda3/envs/         (willow_wbt, gmr)
#       ~/.holosoma_deps/miniconda3/envs/       (upstream: hsretargeting, hsmujoco, hsgym, hssim, hsinference)
#       ~/.holosoma_custom_deps/miniconda3/envs/ (custom fork: same env names)
#   - Activates the willow_wbt conda env
#
# After sourcing, use conda normally:
#   conda activate gmr
#   conda activate hsinference      # upstream
#   conda activate hsinference      # (same name — activate by path if ambiguous)
# =============================================================================

# Guard: must be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: activate_willow.sh must be sourced, not executed."
  echo "  Use:  source scripts/activate_willow.sh"
  exit 1
fi

export WORKSPACE_DIR="$HOME/.willow_deps"
export CONDA_ROOT="$WORKSPACE_DIR/miniconda3"

# Add willow miniconda to PATH (prepend so it takes priority)
if [[ -d "$CONDA_ROOT/bin" ]]; then
  export PATH="$CONDA_ROOT/bin:$PATH"
fi

# Initialize conda for this shell session
if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  echo "WARNING: willow miniconda not found at $CONDA_ROOT"
  echo "  Run ./install.sh first."
  return 1
fi

# Register all three ecosystems in ~/.condarc so `conda env list` shows everything.
# conda config --add is idempotent (deduplicates automatically).
# This is a one-time setup — subsequent sources are no-ops.
for envs_dir in \
    "$HOME/.holosoma_custom_deps/miniconda3/envs" \
    "$HOME/.holosoma_deps/miniconda3/envs" \
    "$CONDA_ROOT/envs"; do
  if [[ -d "$envs_dir" ]]; then
    conda config --add envs_dirs "$envs_dir" 2>/dev/null || true
  fi
done

conda activate willow_wbt

echo "Willow WBT ecosystem active"
echo "  ~/.willow_deps/              willow_wbt, gmr"
echo "  ~/.holosoma_deps/            hsretargeting, hsmujoco, hsgym, hssim, hsinference (upstream)"
echo "  ~/.holosoma_custom_deps/     hsretargeting, hsmujoco, hsgym, hssim, hsinference (custom)"
echo ""
echo "  conda env list   → shows all envs across all three ecosystems"
