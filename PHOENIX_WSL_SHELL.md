# Phoenix WSL Default Shell Setup

## Problem
SSH to Phoenix defaults to cmd.exe instead of WSL bash.

## Solution: Set WSL as Default Shell

### Option 1: Windows OpenSSH Config (Recommended)

On Phoenix (Windows), edit SSH server config:

```powershell
# Run as Administrator on Phoenix
notepad C:\ProgramData\ssh\sshd_config
```

Add this line:
```
ForceCommand wsl bash
```

Restart SSH service:
```powershell
Restart-Service sshd
```

### Option 2: User Shell Config

On Phoenix, create/edit `~/.ssh/rc`:
```bash
#!/bin/bash
exec wsl bash
```

### Option 3: Per-User Default (Best)

On Phoenix (PowerShell as Admin):
```powershell
# Set WSL bash as default shell for your user
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\bash.exe" -PropertyType String -Force
```

Restart SSH:
```powershell
Restart-Service sshd
```

### Verify

From local machine:
```bash
ssh onurbarlik@hotmail.com@phoenix 'echo $SHELL'
# Should output: /bin/bash
```

---

## Current Workaround

Our deployment script wraps commands with `wsl bash -c`:
```bash
ssh user@phoenix "wsl bash -c 'command'"
```

This works but is verbose. Setting default shell is cleaner.

---

## After Setting Default Shell

Update `deploy_to_phoenix.sh` to remove `wsl bash -c` wrapper:

```bash
# Before (current)
ssh ${REMOTE_USER}@${REMOTE_HOST} "wsl bash -c 'mkdir -p ${REMOTE_DIR}'"

# After (with default shell)
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_DIR}"
```
