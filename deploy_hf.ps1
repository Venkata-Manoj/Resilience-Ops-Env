# PowerShell script to deploy to Hugging Face Spaces using Git

$ErrorActionPreference = "Stop"

# Get HF username
$HF_USER = python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to get Hugging Face username. Make sure you're logged in."
    exit 1
}

Write-Host "Deploying to Hugging Face Spaces as: $HF_USER"
Write-Host ""

# Space details
$SPACE_NAME = "resilience-ops-env"
$SPACE_REPO = "$HF_USER/$SPACE_NAME"
$REMOTE_URL = "https://huggingface.co/spaces/$SPACE_REPO"

Write-Host "Step 1: Creating Space (if not exists)..."
python -c @"
from huggingface_hub import HfApi, SpaceInfo
api = HfApi()
try:
    api.repo_info(repo_id='$SPACE_REPO', repo_type='space')
    print('Space already exists')
except Exception as e:
    print('Creating new space...')
    api.create_repo(repo_id='$SPACE_NAME', repo_type='space', space_sdk='docker')
    print('Space created successfully')
"@

Write-Host ""
Write-Host "Step 2: Checking for existing Git remote..."

if (Test-Path .git) {
    Write-Host "Git repository exists"
    $remotes = git remote -v 2>$null
    if ($remotes -match "huggingface") {
        Write-Host "Hugging Face remote already configured"
    } else {
        Write-Host "Adding Hugging Face remote..."
        git remote add huggingface $REMOTE_URL 2>$null
        if ($LASTEXITCODE -ne 0) {
            git remote set-url huggingface $REMOTE_URL 2>$null
        }
    }
} else {
    Write-Host "Initializing Git repository..."
    git init
    git remote add origin $REMOTE_URL 2>$null
}

Write-Host ""
Write-Host "Step 3: Preparing files for deployment..."

# Create a clean branch for deployment
git checkout -b deploy-temp 2>$null
if ($LASTEXITCODE -ne 0) {
    git checkout deploy-temp 2>$null
}

# Add all necessary files
git add -f Dockerfile
git add -f inference.py
git add -f models.py
git add -f client.py
git add -f __init__.py
git add -f requirements.txt
git add -f openenv.yaml
git add -f README.md
git add -f server/
git add -f .dockerignore

Write-Host ""
Write-Host "Step 4: Committing and pushing..."

git commit -m "Deploy to Hugging Face Spaces" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Nothing new to commit"
}

try {
    git push -u origin deploy-temp:main --force
    Write-Host ""
    Write-Host "✅ Deployment successful!"
    Write-Host "Space URL: https://huggingface.co/spaces/$SPACE_REPO"
} catch {
    Write-Error "Git push failed: $_"
    exit 1
}

Write-Host ""
Write-Host "Step 5: Checking deployment status..."
Write-Host "Visit https://huggingface.co/spaces/$SPACE_REPO to see build status"
