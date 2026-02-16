#!/bin/bash
set -e

# Transcribe audio files on remote inference machine via Wake-on-LAN
# Usage: ./transcribe-remote.sh [options] <audio_path_on_nas>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/transcribe-remote.conf"

# Default values
CONTEXT=""
HOTWORDS_FILE=""
OUTPUT=""
AUDIO_DIR=""
POST_ACTION=""

# Trap to warn if interrupted
trap 'echo "⚠️  Script interrupted! Inference machine may still be running." >&2' EXIT

# Usage function
usage() {
    cat <<EOF
Usage: $(basename "$0") [options] <audio_path_on_nas>

Options:
  -c, --context <string>     Hotwords/context string
  -f, --hotwords-file <path> Hotwords file (NAS path as seen in Docker)
  -o, --output <path>        Output JSON path (NAS path as seen in Docker)
  -d, --audio-dir <path>     Process directory instead of single file
  --shutdown                  Shutdown after transcription (override config)
  --suspend                   Suspend after transcription (override config)
  --no-sleep                  Don't suspend/shutdown after transcription
  --config <path>            Path to config file (default: scripts/transcribe-remote.conf)
  -h, --help                 Show this help message

Examples:
  $(basename "$0") /NAS_1/audio/recording.mp3
  $(basename "$0") -c "term1, term2" -o /NAS_1/results/out.json /NAS_1/audio/file.mp3
  $(basename "$0") -d /NAS_1/audio/batch --shutdown

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--context)
            CONTEXT="$2"
            shift 2
            ;;
        -f|--hotwords-file)
            HOTWORDS_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -d|--audio-dir)
            AUDIO_DIR="$2"
            shift 2
            ;;
        --shutdown)
            POST_ACTION="shutdown"
            shift
            ;;
        --suspend)
            POST_ACTION="suspend"
            shift
            ;;
        --no-sleep)
            POST_ACTION="none"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "❌ Unknown option: $1" >&2
            usage
            ;;
        *)
            if [[ -z "$AUDIO_PATH" ]]; then
                AUDIO_PATH="$1"
            else
                echo "❌ Multiple audio paths specified" >&2
                usage
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "$AUDIO_PATH" && -z "$AUDIO_DIR" ]]; then
    echo "❌ Error: Audio path or directory required" >&2
    usage
fi

if [[ -n "$AUDIO_PATH" && -n "$AUDIO_DIR" ]]; then
    echo "❌ Error: Cannot specify both audio file and directory" >&2
    usage
fi

# Source config file
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Config file not found: $CONFIG_FILE" >&2
    echo "Copy scripts/transcribe-remote.conf.example to scripts/transcribe-remote.conf and fill in your settings" >&2
    exit 1
fi

source "$CONFIG_FILE"

# Validate required config variables
for var in INFERENCE_MAC INFERENCE_HOST SSH_USER DOCKER_CONTAINER DOCKER_WORKDIR NAS_MOUNT_DOCKER DEFAULT_MODEL DEFAULT_ATTN; do
    if [[ -z "${!var}" ]]; then
        echo "❌ Required config variable not set: $var" >&2
        exit 1
    fi
done

# Use config default if post-action not specified on command line
if [[ -z "$POST_ACTION" ]]; then
    POST_ACTION="$DEFAULT_POST_ACTION"
fi

# Check wakeonlan is installed
if ! command -v wakeonlan &> /dev/null; then
    echo "❌ wakeonlan not found. Install with: brew install wakeonlan" >&2
    exit 1
fi

# Step 1: Wake the inference machine
echo "🔌 Waking inference machine ($INFERENCE_MAC)..."
wakeonlan "$INFERENCE_MAC"

# Step 2: Wait for SSH to become available
echo "⏳ Waiting for SSH connection to $INFERENCE_HOST..."
elapsed=0
while ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$SSH_USER@$INFERENCE_HOST" true 2>/dev/null; do
    if [[ $elapsed -ge $SSH_WAIT_TIMEOUT ]]; then
        echo "❌ SSH connection timeout after ${SSH_WAIT_TIMEOUT}s" >&2
        exit 1
    fi
    sleep "$SSH_POLL_INTERVAL"
    elapsed=$((elapsed + SSH_POLL_INTERVAL))
    echo "  ... ${elapsed}s elapsed"
done

echo "✅ SSH connection established"

# Step 3: Start Docker container
echo "🐳 Starting Docker container..."
if ! ssh "$SSH_USER@$INFERENCE_HOST" "docker start $DOCKER_CONTAINER 2>/dev/null || docker ps -q -f name=$DOCKER_CONTAINER" &>/dev/null; then
    echo "❌ Failed to start Docker container: $DOCKER_CONTAINER" >&2
    exit 1
fi

echo "✅ Docker container running"

# Step 4: Build transcription command
TRANSCRIBE_CMD="python ${DOCKER_WORKDIR}/demo/vibevoice_asr_inference_with_context.py"
TRANSCRIBE_CMD+=" --model_path $DEFAULT_MODEL"
TRANSCRIBE_CMD+=" --load_in_8bit"
TRANSCRIBE_CMD+=" --attn_implementation $DEFAULT_ATTN"

if [[ -n "$AUDIO_DIR" ]]; then
    TRANSCRIBE_CMD+=" --audio_dir $AUDIO_DIR"
else
    TRANSCRIBE_CMD+=" --audio_files $AUDIO_PATH"
fi

if [[ -n "$CONTEXT" ]]; then
    TRANSCRIBE_CMD+=" --context \"$CONTEXT\""
fi

if [[ -n "$HOTWORDS_FILE" ]]; then
    TRANSCRIBE_CMD+=" --hotwords_file $HOTWORDS_FILE"
fi

if [[ -n "$OUTPUT" ]]; then
    TRANSCRIBE_CMD+=" --output $OUTPUT"
fi

# Step 5: Run transcription
echo "🎙️  Starting transcription..."
set +e  # Don't exit on error, we want to handle it
if ssh "$SSH_USER@$INFERENCE_HOST" "docker exec $DOCKER_CONTAINER bash -c '$TRANSCRIBE_CMD'"; then
    TRANSCRIBE_EXIT=0
    echo "✅ Transcription completed successfully"
else
    TRANSCRIBE_EXIT=$?
    echo "❌ Transcription failed with exit code $TRANSCRIBE_EXIT" >&2
fi
set -e

# Step 6: Post-action
case "$POST_ACTION" in
    suspend)
        echo "💤 Suspending inference machine..."
        ssh "$SSH_USER@$INFERENCE_HOST" "sudo systemctl suspend" || echo "⚠️  Suspend command sent, but may have failed"
        ;;
    shutdown)
        echo "🔌 Shutting down inference machine..."
        ssh "$SSH_USER@$INFERENCE_HOST" "sudo shutdown -h now" || echo "⚠️  Shutdown command sent, but may have failed"
        ;;
    none)
        echo "✅ Machine left running"
        ;;
    *)
        echo "⚠️  Unknown post-action: $POST_ACTION" >&2
        ;;
esac

# Step 7: Report summary
echo ""
echo "════════════════════════════════════════"
echo "Summary:"
if [[ -n "$OUTPUT" ]]; then
    echo "  Output: $OUTPUT"
elif [[ -n "$AUDIO_DIR" ]]; then
    echo "  Processed directory: $AUDIO_DIR"
else
    echo "  Processed file: $AUDIO_PATH"
fi
echo "  Exit code: $TRANSCRIBE_EXIT"
echo "  Post-action: $POST_ACTION"
echo "════════════════════════════════════════"

# Clear trap and exit with transcription's exit code
trap - EXIT
exit $TRANSCRIBE_EXIT
