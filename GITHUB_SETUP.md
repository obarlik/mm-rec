# GitHub Setup Guide

## Current Status

✅ Code committed locally  
❌ No GitHub remote configured

---

## Steps to Push to GitHub

### 1. Create GitHub Repository

Go to: https://github.com/new

- **Repository name:** `mm-rec`
- **Description:** MM-Rec: Memory-driven Recurrent Model
- **Visibility:** Private (recommended) or Public
- **Initialize:** Do NOT add README, .gitignore, or license (we have them)

Click "Create repository"

### 2. Add Remote and Push

After creating the repo, GitHub will show you the URL. Use it:

```bash
cd /home/onur/workspace/mm-rec

# Add remote (replace with your actual URL)
git remote add origin https://github.com/YOUR-USERNAME/mm-rec.git

# Push code
git push -u origin main
```

### 3. Deploy on Phoenix

Once pushed to GitHub:

```bash
# On Phoenix
ssh onurbarlik@hotmail.com@phoenix
git clone https://github.com/YOUR-USERNAME/mm-rec.git
cd mm-rec
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -r server/requirements.txt
python server/train_server.py
```

**Very fast!** Git clone takes seconds, not minutes.

---

## Alternative: Use Existing Repo

If you already have a GitHub account with repos, you can:

1. Use existing repo
2. Create new branch for mm-rec
3. Push there

---

**Next:** Create GitHub repo and provide the URL
