# Docker Setup

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (Windows/Mac) or [Docker Engine](https://docs.docker.com/engine/install/) (Linux)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Quick Start

### Windows (PowerShell)

```powershell
# Using the helper script
.\start.ps1 build
.\start.ps1 run

# Or directly with Docker
docker-compose build
docker-compose up -d
```

### Mac/Linux

```bash
# Using Make
make build
make run

# Or directly with Docker
docker-compose build
docker-compose up -d
```

Access the app at: http://localhost:8501

## Commands

### Windows (PowerShell)

| Command | Description |
|---------|-------------|
| `.\start.ps1 build` | Build Docker image |
| `.\start.ps1 run` | Run in development mode |
| `.\start.ps1 stop` | Stop containers |
| `.\start.ps1 logs` | View logs |
| `.\start.ps1 shell` | Open container shell |
| `.\start.ps1 clean` | Remove all containers/volumes |
| `.\start.ps1 status` | Check container status |

### Mac/Linux (Make)

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make run` | Run in development mode |
| `make run-prod` | Run in production mode |
| `make stop` | Stop containers |
| `make logs` | View logs |
| `make shell` | Open container shell |
| `make clean` | Remove all containers/volumes |

### Any Platform (Docker directly)

| Command | Description |
|---------|-------------|
| `docker-compose build` | Build image |
| `docker-compose up -d` | Run container |
| `docker-compose down` | Stop container |
| `docker-compose logs -f` | View logs |
| `docker-compose ps` | Check status |

## Production Deployment

```bash
# Linux/Mac
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Windows PowerShell
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Data Persistence

Data is stored in Docker volumes:
- `./data` → Container: `/app/data`
- `model-cache` → Ultralytics model cache

## GPU Support

GPU is enabled by default in `docker-compose.yml`. To disable:

```yaml
# Remove or comment out the deploy section
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## Troubleshooting

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use port 8502 on host
```

### Model download fails
Models are cached in the `model-cache` volume. If download fails:
```bash
make clean
make build
make run
```

### Permission denied on data folder
```bash
# Fix permissions
sudo chown -R 1000:1000 ./data
```
