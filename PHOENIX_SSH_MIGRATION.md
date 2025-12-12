# Phoenix SSH Server Migration (Bitvise → OpenSSH)

## Why Switch?

**Bitvise SSH Server** has limited WSL integration.  
**Windows OpenSSH Server** has native WSL support and better compatibility.

---

## Migration Steps

### 1. Install OpenSSH Server (Phoenix - PowerShell Admin)

```powershell
# Check if already installed
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'

# Install if not present
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start and enable service
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# Allow firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

### 2. Configure WSL as Default Shell

```powershell
# Set WSL bash as default shell
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\bash.exe" -PropertyType String -Force

# Restart SSH service
Restart-Service sshd
```

### 3. Stop Bitvise (Phoenix)

```powershell
# Stop Bitvise service
Stop-Service BvSshServer

# Disable auto-start
Set-Service -Name BvSshServer -StartupType 'Disabled'
```

### 4. Test from Local Machine

```bash
# Should connect to WSL directly
ssh onurbarlik@hotmail.com@phoenix 'echo $SHELL'
# Expected: /bin/bash

# Should not ask for password (if SSH key set up)
ssh onurbarlik@hotmail.com@phoenix 'pwd'
```

### 5. Copy SSH Keys (if needed)

If Bitvise had your keys, copy to OpenSSH:

```powershell
# On Phoenix
# Bitvise keys location: C:\Program Files\Bitvise SSH Server\BvSshServer-Settings\
# OpenSSH keys location: C:\ProgramData\ssh\

# Copy authorized_keys
Copy-Item "$env:USERPROFILE\.ssh\authorized_keys" "C:\ProgramData\ssh\administrators_authorized_keys"
```

---

## Quick Migration Script (Phoenix - PowerShell Admin)

```powershell
# Install OpenSSH
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Configure
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Value "C:\Windows\System32\bash.exe" -PropertyType String -Force

# Stop Bitvise
Stop-Service BvSshServer
Set-Service -Name BvSshServer -StartupType 'Disabled'

# Restart OpenSSH
Restart-Service sshd

Write-Host "✅ Migration complete! Test with: ssh user@phoenix 'echo \$SHELL'"
```

---

## Troubleshooting

### Port Conflict
If both services try to use port 22:

```powershell
# Check what's using port 22
netstat -ano | findstr :22

# Stop Bitvise first
Stop-Service BvSshServer
```

### Permission Issues

```powershell
# Fix authorized_keys permissions
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /inheritance:r
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /grant "SYSTEM:(F)"
icacls "C:\ProgramData\ssh\administrators_authorized_keys" /grant "BUILTIN\Administrators:(F)"
```

### SSH Key Not Working

```bash
# Re-copy SSH key from local machine
ssh-copy-id onurbarlik@hotmail.com@phoenix
```

---

## After Migration

### Update Deployment Script

Remove `wsl bash -c` wrappers:

```bash
# Old (with Bitvise)
ssh user@phoenix "wsl bash -c 'mkdir -p ~/dir'"

# New (with OpenSSH + default shell)
ssh user@phoenix "mkdir -p ~/dir"
```

### Verify

```bash
# Should work without password
ssh onurbarlik@hotmail.com@phoenix 'uname -a'

# Should show WSL
ssh onurbarlik@hotmail.com@phoenix 'cat /etc/os-release'
```

---

## Benefits

✅ Native WSL support  
✅ No `wsl bash -c` wrapper needed  
✅ Better SSH key authentication  
✅ Standard OpenSSH config  
✅ Faster deployment

---

**Recommendation:** Migrate to OpenSSH for better WSL integration! 🚀
