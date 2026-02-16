#!/bin/bash
# One-time setup script for the Ubuntu inference machine
# Run this script ON THE INFERENCE MACHINE to configure Wake-on-LAN and other settings

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  VibeVoice Inference Server Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This script will configure your Ubuntu machine for remote transcription"
echo "with Wake-on-LAN support. It requires sudo access."
echo ""

# Utility functions
confirm() {
    local prompt="$1"
    local response
    while true; do
        read -p "$prompt [y/N]: " response
        case "$response" in
            [Yy]* ) return 0;;
            [Nn]* | "" ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Step 1: Detect network interface
echo "Step 1: Detecting network interface..."
DEFAULT_IFACE=$(ip route | grep default | awk '{print $5}' | head -n 1)

if [[ -z "$DEFAULT_IFACE" ]]; then
    echo "❌ Could not detect default network interface" >&2
    echo "Please run 'ip route' to find your interface manually" >&2
    exit 1
fi

echo "  Detected interface: $DEFAULT_IFACE"
echo ""

if ! confirm "Use $DEFAULT_IFACE for Wake-on-LAN?"; then
    read -p "Enter interface name: " DEFAULT_IFACE
    if [[ -z "$DEFAULT_IFACE" ]]; then
        echo "❌ Interface name required" >&2
        exit 1
    fi
fi

# Step 2: Enable WoL via ethtool
echo ""
echo "Step 2: Enabling Wake-on-LAN..."

if ! command -v ethtool &> /dev/null; then
    echo "  ethtool not found. Installing..."
    if confirm "Install ethtool?"; then
        sudo apt-get update && sudo apt-get install -y ethtool
    else
        echo "❌ ethtool is required for Wake-on-LAN" >&2
        exit 1
    fi
fi

if confirm "Enable Wake-on-LAN on $DEFAULT_IFACE?"; then
    sudo ethtool -s "$DEFAULT_IFACE" wol g
    echo "  ✅ Wake-on-LAN enabled"

    # Verify
    WOL_STATUS=$(sudo ethtool "$DEFAULT_IFACE" | grep "Wake-on:" | awk '{print $2}')
    echo "  Current Wake-on-LAN status: $WOL_STATUS"

    if [[ "$WOL_STATUS" != *"g"* ]]; then
        echo "  ⚠️  Warning: Wake-on-LAN may not be properly enabled"
        echo "  Some network cards don't support WoL or require BIOS configuration"
    fi
else
    echo "  ⏭️  Skipped Wake-on-LAN enabling"
fi

# Step 3: Persist WoL across reboots
echo ""
echo "Step 3: Persisting Wake-on-LAN across reboots..."

if confirm "Create systemd service to enable WoL at boot?"; then
    sudo tee /etc/systemd/system/wol.service > /dev/null <<EOF
[Unit]
Description=Enable Wake-on-LAN
After=network.target

[Service]
Type=oneshot
ExecStart=/sbin/ethtool -s $DEFAULT_IFACE wol g

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable wol.service
    echo "  ✅ Systemd service created and enabled"
else
    echo "  ⏭️  Skipped systemd service creation"
fi

# Step 4: Docker container restart policy
echo ""
echo "Step 4: Configuring Docker container restart policy..."

read -p "Enter Docker container name [vibevoice]: " CONTAINER_NAME
CONTAINER_NAME=${CONTAINER_NAME:-vibevoice}

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if confirm "Set $CONTAINER_NAME to restart automatically (unless-stopped)?"; then
        docker update --restart unless-stopped "$CONTAINER_NAME"
        echo "  ✅ Docker restart policy updated"
    else
        echo "  ⏭️  Skipped Docker restart policy"
    fi
else
    echo "  ⚠️  Container '$CONTAINER_NAME' not found"
    echo "  Run 'docker ps -a' to see available containers"
fi

# Step 5: Sudoers for passwordless suspend/shutdown
echo ""
echo "Step 5: Configuring passwordless suspend/shutdown..."
echo "  This allows the Mac controller to suspend/shutdown this machine via SSH"
echo ""

CURRENT_USER=$(whoami)

if confirm "Allow $CURRENT_USER to run suspend/shutdown without password?"; then
    SUDOERS_FILE="/etc/sudoers.d/vibevoice-transcribe"

    echo "$CURRENT_USER ALL=(ALL) NOPASSWD: /bin/systemctl suspend, /sbin/shutdown" | sudo tee "$SUDOERS_FILE" > /dev/null
    sudo chmod 0440 "$SUDOERS_FILE"

    # Verify syntax
    if sudo visudo -c -f "$SUDOERS_FILE" &> /dev/null; then
        echo "  ✅ Sudoers configuration created"
    else
        echo "  ❌ Sudoers syntax error! Removing file..." >&2
        sudo rm -f "$SUDOERS_FILE"
        exit 1
    fi
else
    echo "  ⏭️  Skipped sudoers configuration"
    echo "  Note: You'll need to enter password for suspend/shutdown commands"
fi

# Step 6: Display MAC address
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Setup Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""

MAC_ADDRESS=$(ip link show "$DEFAULT_IFACE" | grep -o -E '([[:xdigit:]]{2}:){5}[[:xdigit:]]{2}' | head -n 1)
IP_ADDRESS=$(ip addr show "$DEFAULT_IFACE" | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)

echo "Network Interface: $DEFAULT_IFACE"
echo "MAC Address: $MAC_ADDRESS"
echo "IP Address: $IP_ADDRESS"
echo ""
echo "Copy these values to your Mac's transcribe-remote.conf:"
echo "  INFERENCE_MAC=\"$MAC_ADDRESS\""
echo "  INFERENCE_HOST=\"$IP_ADDRESS\""
echo "  SSH_USER=\"$CURRENT_USER\""
echo ""

# Step 7: SSH key reminder
echo "════════════════════════════════════════════════════════════════"
echo "  SSH Setup Reminder"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "For passwordless SSH access from your Mac:"
echo "  1. On Mac: Run 'ssh-keygen' if you don't have a key"
echo "  2. On Mac: Run 'ssh-copy-id $CURRENT_USER@$IP_ADDRESS'"
echo "  3. Test: Run 'ssh $CURRENT_USER@$IP_ADDRESS true'"
echo ""

# Test WoL capability
echo "════════════════════════════════════════════════════════════════"
echo "  Testing Wake-on-LAN"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To test Wake-on-LAN from your Mac:"
echo "  1. Install wakeonlan: brew install wakeonlan"
echo "  2. Suspend this machine: sudo systemctl suspend"
echo "  3. From Mac: wakeonlan $MAC_ADDRESS"
echo "  4. This machine should wake up"
echo ""

echo "✅ Setup complete!"
