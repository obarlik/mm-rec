#!/bin/bash
# Setup SSH key authentication for Phoenix (one-time)

REMOTE_USER="onurbarlik@hotmail.com"
REMOTE_HOST="phoenix"

echo "================================================"
echo "SSH Key Setup for Phoenix"
echo "================================================"

# 1. Check if SSH key exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Creating SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
else
    echo "✅ SSH key already exists"
fi

# 2. Copy key to Phoenix
echo ""
echo "Copying SSH key to Phoenix..."
echo "(You'll need to enter password ONE TIME)"
ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}

echo ""
echo "================================================"
echo "✅ SSH Key Setup Complete!"
echo "================================================"
echo ""
echo "Test (should not ask for password):"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'echo Hello from Phoenix'"
echo ""
