# Quick Start Guide

## New Developer Onboarding (5 minutes)

### 1. Prerequisites

Install these once:
- [Docker Desktop](https://www.docker.com/products/docker-desktop) (includes Docker Compose)
- [Git](https://git-scm.com/download/win)
- (Optional) [VS Code](https://code.visualstudio.com/) with Python extension

### 2. Clone & Start

```powershell
# Clone repository
git clone <repo-url>
cd footballtracking

# Switch to feature branch
git checkout feature/refactor-ui

# Build and run (Windows)
.\start.ps1 build
.\start.ps1 run

# Or using Docker directly:
docker-compose build
docker-compose up -d
```

**App will open at:** http://localhost:8501

### 3. Verify It Works

1. Upload a video (go to Upload page)
2. Check preprocessing settings
3. Run analysis
4. View results with tracking data

---

## Daily Development Workflow

| Task | Command |
|------|---------|
| Start working | `.\start.ps1 run` |
| View logs | `.\start.ps1 logs` |
| Stop for the day | `.\start.ps1 stop` |
| Rebuild after code changes | `.\start.ps1 build` then `.\start.ps1 run` |
| Full reset | `.\start.ps1 clean` |

---

## Cross-Platform Commands

| Platform | Build | Run | Stop |
|----------|-------|-----|------|
| **Windows (PowerShell)** | `.\start.ps1 build` | `.\start.ps1 run` | `.\start.ps1 stop` |
| **Mac/Linux** | `make build` | `make run` | `make stop` |
| **Any (Docker)** | `docker-compose build` | `docker-compose up -d` | `docker-compose down` |

---

## Troubleshooting

### "Docker not running"
Start Docker Desktop first.

### "Port 8501 already in use"
Edit `docker-compose.yml`, change `8501:8501` to `8502:8501`, then access http://localhost:8502

### "GPU not available"
- Windows: Ensure WSL2 backend is enabled in Docker Desktop settings
- Linux: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- To run on CPU only: Remove the `deploy:` section from `docker-compose.yml`

### "Permission denied on data folder" (Linux/Mac)
```bash
sudo chown -R $USER:$USER ./data
```

---

## Project Structure (What You Need to Know)

```
footballtracking/
├── app/                    # Streamlit UI code
│   ├── Home.py            # Entry point
│   ├── pages/             # Page modules (upload, analysis, results)
│   ├── config.py          # Paths & constants
│   └── utils.py           # Shared UI utilities
├── src/                   # Computer vision pipeline
│   └── pipeline/          # Detection, tracking, etc.
├── data/                  # Runtime data (gitignored, in Docker volume)
│   ├── raw/               # Uploaded videos
│   ├── processed/         # Preprocessed videos
│   ├── insights/          # Analysis outputs (CSV, JSON)
│   └── annotations/       # Generated annotations
├── docker-compose.yml     # Docker orchestration
├── Dockerfile             # Container definition
└── requirements.txt       # Python dependencies
```

---

## First Code Change?

1. Edit files in `app/` or `src/`
2. Run `.\start.ps1 build` to rebuild
3. Run `.\start.ps1 run` to test
4. Commit changes: `git add . && git commit -m "your message"`

---

## IDE Setup (VS Code Recommended)

Extensions:
- Python
- Docker
- Pylance

Settings for container development:
```json
{
    "python.analysis.extraPaths": ["app", "src"],
    "python.defaultInterpreterPath": "/usr/bin/python3"
}
```

---

## Questions?

Check `DOCKER.md` for detailed Docker docs.
