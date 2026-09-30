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

printf '%s\n' 'Checking this computer and preparing video2tenhou. Leave this terminal open.'
UV_PROJECT_ENVIRONMENT=.venv-runtime
UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-auto}
if [ "${VIDEO2TENHOU_DEVICE:-auto}" = cpu ]; then
    UV_TORCH_BACKEND=cpu
fi
export UV_PROJECT_ENVIRONMENT UV_TORCH_BACKEND
uv venv --allow-existing --python 3.12 .venv-runtime || exit "$?"
uv pip install --python .venv-runtime --torch-backend "$UV_TORCH_BACKEND" --upgrade-package torch --upgrade-package torchvision --editable .
status=$?
if [ "$status" -ne 0 ]; then
    printf '%s\n' 'Setup failed. Check your connection and disk space, and update uv if needed. See docs/QUICKSTART.md.' >&2
    exit "$status"
fi
uv run --no-sync video2tenhou web "$@"
status=$?
if [ "$status" -ne 0 ]; then
    printf '%s\n' 'The application stopped with an error; see the message above.' >&2
fi
exit "$status"
