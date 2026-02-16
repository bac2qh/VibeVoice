# Wake-on-LAN Automated Transcription Guide

This guide explains how to set up and use the Wake-on-LAN automated transcription system for VibeVoice ASR. The system allows you to wake a remote Ubuntu GPU machine, run transcription on audio files stored on a shared NAS, and automatically suspend/shutdown the machine when done.

## Architecture

```
Mac (Controller)  →  Wake-on-LAN  →  Ubuntu GPU Machine  →  Docker Container
                                           ↓
                                      NAS Storage
                                    (/mnt/NAS_1)
```

- **Mac**: Orchestration controller (runs `transcribe-remote.sh`)
- **Ubuntu GPU Machine**: Inference server (8-bit quantized VibeVoice-ASR)
- **Docker Container**: Pre-configured environment with VibeVoice installed
- **NAS**: Shared storage for audio files and transcription results

## Prerequisites

### Mac Requirements

- Same LAN as inference machine
- `wakeonlan` utility: `brew install wakeonlan`
- SSH access to inference machine (passwordless key-based auth recommended)
- Access to NAS mount point

### Ubuntu GPU Machine Requirements

- NVIDIA GPU with CUDA support
- Docker installed and configured
- Network card supporting Wake-on-LAN
- NAS mounted (e.g., at `/mnt/NAS_1`)
- Docker container named `vibevoice` with VibeVoice installed

## One-Time Setup

### Step 1: Configure Inference Machine

On the **Ubuntu GPU machine**, run the setup script:

```bash
cd /path/to/VibeVoice
bash scripts/setup-inference-server.sh
```

This interactive script will:
1. Detect your network interface
2. Enable Wake-on-LAN (requires `ethtool`)
3. Create systemd service to persist WoL across reboots
4. Configure Docker container restart policy
5. Set up passwordless sudo for suspend/shutdown
6. Display your MAC address and IP for Mac configuration

**Important**: Copy the MAC address and IP address displayed at the end.

### Step 2: Configure Mac Controller

On your **Mac**, create the configuration file:

```bash
cd /path/to/VibeVoice
cp scripts/transcribe-remote.conf.example scripts/transcribe-remote.conf
```

Edit `scripts/transcribe-remote.conf` with values from Step 1:

```bash
# Inference machine
INFERENCE_MAC="aa:bb:cc:dd:ee:ff"     # MAC address from setup script
INFERENCE_HOST="192.168.1.100"         # IP address from setup script
SSH_USER="your-username"               # SSH user on inference machine

# Docker
DOCKER_CONTAINER="vibevoice"
DOCKER_WORKDIR="/root/VibeVoice"

# NAS paths (as seen inside Docker container)
NAS_MOUNT_DOCKER="/NAS_1"

# Defaults
DEFAULT_MODEL="microsoft/VibeVoice-ASR"
DEFAULT_ATTN="flash_attention_2"
DEFAULT_POST_ACTION="suspend"          # suspend | shutdown | none

# Timeouts (seconds)
SSH_WAIT_TIMEOUT=300
SSH_POLL_INTERVAL=5
```

### Step 3: Set Up SSH Keys (Recommended)

For passwordless operation, set up SSH key authentication:

```bash
# On Mac: Generate SSH key if you don't have one
ssh-keygen

# Copy public key to inference machine
ssh-copy-id your-username@192.168.1.100

# Test connection
ssh your-username@192.168.1.100 true
```

### Step 4: Test Wake-on-LAN

1. Suspend the inference machine:
   ```bash
   ssh your-username@192.168.1.100 "sudo systemctl suspend"
   ```

2. Wake it from your Mac:
   ```bash
   wakeonlan aa:bb:cc:dd:ee:ff
   ```

3. Wait 10-30 seconds and verify:
   ```bash
   ssh your-username@192.168.1.100 "echo 'Machine is awake!'"
   ```

If this doesn't work, check the Troubleshooting section below.

## Usage

### Basic Transcription

Transcribe a single audio file:

```bash
./scripts/transcribe-remote.sh /NAS_1/audio/recording.mp3
```

The script will:
1. Wake the inference machine
2. Wait for SSH connection
3. Start Docker container
4. Run transcription with 8-bit quantization
5. Suspend the machine (default behavior)

### Specify Output Path

```bash
./scripts/transcribe-remote.sh \
  -o /NAS_1/results/output.json \
  /NAS_1/audio/recording.mp3
```

### Add Hotwords/Context

Use hotwords to improve accuracy for domain-specific terms:

```bash
# Inline context string
./scripts/transcribe-remote.sh \
  -c "VibeVoice, PyTorch, CUDA, transformer" \
  /NAS_1/audio/technical-talk.mp3

# From file
./scripts/transcribe-remote.sh \
  -f /NAS_1/context/medical-terms.txt \
  /NAS_1/audio/medical-dictation.mp3
```

### Batch Processing

Process all audio files in a directory:

```bash
./scripts/transcribe-remote.sh \
  -d /NAS_1/audio/batch-2024-02 \
  --shutdown
```

### Control Post-Action

```bash
# Shutdown instead of suspend
./scripts/transcribe-remote.sh --shutdown /NAS_1/audio/file.mp3

# Leave machine running
./scripts/transcribe-remote.sh --no-sleep /NAS_1/audio/file.mp3

# Suspend (default, can be changed in config)
./scripts/transcribe-remote.sh --suspend /NAS_1/audio/file.mp3
```

### Full Example

```bash
./scripts/transcribe-remote.sh \
  -c "VibeVoice, Microsoft, neural codec" \
  -o /NAS_1/results/research-meeting-2024-02-15.json \
  --suspend \
  /NAS_1/audio/research-meeting-2024-02-15.mp3
```

## Troubleshooting

### Wake-on-LAN Not Working

**Symptoms**: Machine doesn't wake when `wakeonlan` is sent.

**Solutions**:

1. **Verify WoL is enabled**:
   ```bash
   ssh user@host "sudo ethtool eth0 | grep Wake-on"
   ```
   Should show `Wake-on: g` (not `d`)

2. **Check BIOS/UEFI settings**:
   - Enable "Wake on LAN" or "Power on by PCI-E device"
   - Disable "Deep Sleep" or "ErP Mode" (they prevent WoL)

3. **Verify network card supports WoL**:
   ```bash
   sudo ethtool eth0 | grep "Supports Wake-on"
   ```
   Should show `Supports Wake-on: pg` or similar (must include `g`)

4. **Check if machine is truly suspended** (not shutdown):
   - WoL typically works with suspend, not full shutdown
   - Some BIOS settings required for shutdown WoL

5. **Verify MAC address is correct**:
   ```bash
   ssh user@host "ip link show eth0"
   ```
   Copy the exact MAC address to your config file

6. **Test from same subnet**:
   - WoL packets don't cross routers by default
   - Mac and inference machine must be on same LAN

### SSH Connection Timeout

**Symptoms**: Script waits for SSH but times out.

**Solutions**:

1. **Increase timeout in config**:
   ```bash
   SSH_WAIT_TIMEOUT=600  # 10 minutes
   ```

2. **Check if machine booted properly**:
   - Monitor via physical display or remote management (iDRAC, iLO, etc.)
   - Check network cable is connected

3. **Verify SSH service starts at boot**:
   ```bash
   sudo systemctl enable ssh
   ```

4. **Check firewall isn't blocking SSH**:
   ```bash
   sudo ufw status
   sudo ufw allow 22/tcp
   ```

### Docker Container Issues

**Symptoms**: "Failed to start Docker container" error.

**Solutions**:

1. **Verify container exists**:
   ```bash
   ssh user@host "docker ps -a"
   ```

2. **Check container name matches config**:
   ```bash
   DOCKER_CONTAINER="vibevoice"  # in transcribe-remote.conf
   ```

3. **Manually start container**:
   ```bash
   ssh user@host "docker start vibevoice"
   ```

4. **Check Docker service is running**:
   ```bash
   ssh user@host "sudo systemctl status docker"
   ```

5. **Verify NAS is mounted inside container**:
   ```bash
   ssh user@host "docker exec vibevoice ls /NAS_1"
   ```

### Transcription Failures

**Symptoms**: Transcription starts but fails partway through.

**Solutions**:

1. **Check GPU memory**:
   - 8-bit quantization requires ~16GB VRAM for VibeVoice-ASR
   - Monitor with: `ssh user@host "docker exec vibevoice nvidia-smi"`

2. **Verify audio file is accessible**:
   ```bash
   ssh user@host "docker exec vibevoice ls -lh /NAS_1/audio/file.mp3"
   ```

3. **Check audio format**:
   - VibeVoice supports MP3, WAV, FLAC, etc.
   - Verify with: `file audio.mp3`

4. **Review logs**:
   ```bash
   ssh user@host "docker logs vibevoice"
   ```

5. **Test inference locally**:
   ```bash
   ssh user@host
   docker exec -it vibevoice bash
   cd /root/VibeVoice
   python demo/vibevoice_asr_inference_with_context.py \
     --model_path microsoft/VibeVoice-ASR \
     --audio_files /NAS_1/audio/test.mp3 \
     --load_in_8bit
   ```

### Permission Issues

**Symptoms**: "sudo: password required" or "permission denied" errors.

**Solutions**:

1. **Verify sudoers configuration**:
   ```bash
   ssh user@host "sudo cat /etc/sudoers.d/vibevoice-transcribe"
   ```
   Should contain:
   ```
   your-username ALL=(ALL) NOPASSWD: /bin/systemctl suspend, /sbin/shutdown
   ```

2. **Test manually**:
   ```bash
   ssh user@host "sudo systemctl suspend"
   ```
   Should not ask for password

3. **Check file permissions**:
   ```bash
   ls -l /etc/sudoers.d/vibevoice-transcribe
   ```
   Should be `-r--r-----` (0440)

## Advanced Configuration

### Custom Model Path

To use a different model or locally fine-tuned model, edit `transcribe-remote.conf`:

```bash
DEFAULT_MODEL="/path/to/your/model"
```

### Multiple Inference Machines

Create separate config files for different machines:

```bash
cp scripts/transcribe-remote.conf scripts/transcribe-machine-2.conf
# Edit machine-2 config with different MAC/IP

# Use with --config flag
./scripts/transcribe-remote.sh --config scripts/transcribe-machine-2.conf /NAS_1/audio/file.mp3
```

### Integration with Cron/Automation

For scheduled batch processing:

```bash
# Add to crontab (runs daily at 2 AM)
0 2 * * * /path/to/VibeVoice/scripts/transcribe-remote.sh -d /NAS_1/audio/daily --shutdown
```

### Custom Python Script

Instead of using `vibevoice_asr_inference_with_context.py`, you can modify `transcribe-remote.sh` to call a custom script with your own processing pipeline.

## Performance Notes

- **8-bit quantization**: ~16GB VRAM required for VibeVoice-ASR-7B
- **Long-form audio**: Can process up to 60 minutes in a single pass
- **Batch processing**: Sequential (one file at a time)
- **Typical transcription time**: ~0.1-0.2x real-time (10 min audio → 1-2 min processing)

## Security Considerations

- **SSH keys**: Use strong SSH keys and keep private keys secure
- **Sudoers**: Only grants suspend/shutdown, no other root privileges
- **NAS access**: Ensure NAS permissions are properly configured
- **Network**: Use on trusted LAN only (WoL packets are unauthenticated)
- **Config file**: Never commit `transcribe-remote.conf` (contains IP/MAC addresses)

## See Also

- [VibeVoice ASR Documentation](../README.md)
- [8-bit Quantization Guide](./8bit-guide.md)
- [Fine-tuning Guide](../finetuning-asr/README.md)
