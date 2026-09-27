#!/bin/sh
# Keep the data directory override, if supplied, while locating this checkout.
script_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
cd "$script_dir" || exit 1

missing=0
for tool in uv ffmpeg ffprobe tesseract; do
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
if ! tesseract --list-langs 2>/dev/null | grep -qx 'eng'; then
    printf '%s\n' 'Missing Tesseract English language data: eng' \
        'Install the English language data; see docs/QUICKSTART.md.' >&2
    exit 1
fi

printf '%s\n' 'Preparing video2tenhou. The first launch downloads Python dependencies.'
uv sync --frozen --no-dev
status=$?
if [ "$status" -ne 0 ]; then
    printf '%s\n' 'Setup failed. See the message above and docs/QUICKSTART.md.' >&2
    exit "$status"
fi
printf '%s\n' 'Opening the studio. Leave this terminal open while using the application.'
uv run --no-sync video2tenhou web "$@"
status=$?
if [ "$status" -ne 0 ]; then
    printf '%s\n' 'The application stopped with an error; see the message above.' >&2
fi
exit "$status"
