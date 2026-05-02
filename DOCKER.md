# Docker Setup

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Quick Start

```bash
# Build and run
make build
make run

# Or manually:
docker-compose up --build -d
```

Access the app at: http://localhost:8501

## Commands

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make run` | Run in development mode |
| `make run-prod` | Run in production mode |
| `make stop` | Stop containers |
| `make logs` | View logs |
| `make shell` | Open container shell |
| `make clean` | Remove all containers/volumes |

## Production Deployment

```bash
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
