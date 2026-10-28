# Source from bash: source scripts/activate.sh
CS593_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONDA_ENVS_PATH="$CS593_ROOT/.conda/envs${CONDA_ENVS_PATH:+:$CONDA_ENVS_PATH}"
if command -v conda >/dev/null 2>&1; then
    CS593_CONDA_BASE="$(conda info --base)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CS593_CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CS593_CONDA_BASE="$HOME/anaconda3"
else
    echo "Conda was not found. Use the Python venv instructions in README.md." >&2
    return 1
fi
source "$CS593_CONDA_BASE/etc/profile.d/conda.sh"
if [ -d "$CS593_ROOT/.conda/envs/cs593/conda-meta" ]; then
    conda activate "$CS593_ROOT/.conda/envs/cs593" || return 1
else
    conda activate cs593 || return 1
fi
export PYTHONPATH="$CS593_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="$CS593_ROOT/.cache/numba"
export MUJOCO_GL=egl
