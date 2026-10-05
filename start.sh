#!/bin/sh
# Keep the data directory override, if supplied, while locating this checkout.
script_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
cd "$script_dir" || exit 1

missing=0
for tool in uv ffmpeg ffprobe; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing prerequisite: %s\n' "$tool" >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    printf '%s\n' 'Install the missing tools, then run: sh start.sh' \
        'See docs/QUICKSTART.md or:' \
        'https://github.com/AsaChiri/video2tenhou/blob/main/docs/QUICKSTART.md' >&2
    exit 1
fi

# Install only when the runtime is missing, incomplete or older than pyproject.toml,
# or when asked to: a new PyTorch build invalidates every recognition cache.
update=${VIDEO2TENHOU_UPDATE:-0}
if [ "${1:-}" = --update ]; then
    update=1
    shift
fi
UV_PROJECT_ENVIRONMENT=.venv-runtime
export UV_PROJECT_ENVIRONMENT
installed=.venv-runtime/video2tenhou-pyproject.toml
if [ "$update" = 1 ] || ! cmp -s pyproject.toml "$installed" 2>/dev/null; then
    printf '%s\n' 'Preparing video2tenhou for this computer. Leave this terminal open.'
    UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-auto}
    if [ "${VIDEO2TENHOU_DEVICE:-auto}" = cpu ]; then
        UV_TORCH_BACKEND=cpu
    fi
    export UV_TORCH_BACKEND
    upgrade=
    if [ "$update" = 1 ]; then
        upgrade='--upgrade-package torch --upgrade-package torchvision'
    fi
    rm -f "$installed"
    uv venv --allow-existing --python 3.12 .venv-runtime || exit "$?"
    uv pip install --python .venv-runtime --torch-backend "$UV_TORCH_BACKEND" \
        $upgrade --editable .
    status=$?
    if [ "$status" -eq 0 ]; then
        cp pyproject.toml "$installed"
        status=$?
    fi
    if [ "$status" -ne 0 ]; then
        printf '%s\n' 'Setup failed. Check your connection and disk space, and update uv if needed. See docs/QUICKSTART.md.' >&2
        exit "$status"
    fi
fi
uv run --no-sync video2tenhou web "$@"
status=$?
if [ "$status" -ne 0 ]; then
    printf '%s\n' 'The application stopped with an error; see the message above.' >&2
fi
exit "$status"
