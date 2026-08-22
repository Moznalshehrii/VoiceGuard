#!/bin/bash
# Keep reconnecting and resuming a download that gets reset by the server
# every ~30-90s. Each fresh curl invocation picks up exactly where the last
# one was cut off (-C -), so repeated short-lived connections add up to a
# complete file even though no single connection survives to the end.
set -u

URL="$1"
OUT="$2"
EXPECTED_SIZE="$3"
MAX_ATTEMPTS="${4:-500}"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    current_size=0
    if [ -f "$OUT" ]; then
        current_size=$(stat -f%z "$OUT")
    fi

    if [ "$current_size" -ge "$EXPECTED_SIZE" ]; then
        echo "Done: $OUT reached $current_size bytes (expected $EXPECTED_SIZE)"
        exit 0
    fi

    echo "[attempt $attempt] size=$current_size / $EXPECTED_SIZE ($(( current_size * 100 / EXPECTED_SIZE ))%)"
    curl -sS -L -C - --max-time 120 -o "$OUT" "$URL" 2>&1 | tail -1
    sleep 1
done

echo "Gave up after $MAX_ATTEMPTS attempts"
exit 1
