# Football Tracker - Quick Start Script for Windows
# Usage: .\start.ps1 [build|run|stop|logs|clean]

param(
    [Parameter()]
    [ValidateSet("build", "run", "stop", "logs", "shell", "clean", "status", "help")]
    [string]$Command = "help"
)

$appPort = 8501
$containerName = "football-tracker"

function Show-Help {
    Write-Host "Football Tracker - Docker Commands" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\start.ps1 [command]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Green
    Write-Host "  build    - Build Docker image"
    Write-Host "  run      - Start container"
    Write-Host "  stop     - Stop container"
    Write-Host "  logs     - View container logs"
    Write-Host "  shell    - Open container shell"
    Write-Host "  clean    - Remove all containers and volumes"
    Write-Host "  status   - Check container status"
    Write-Host "  help     - Show this help"
    Write-Host ""
}

function Build-Image {
    Write-Host "Building Docker image..." -ForegroundColor Cyan
    docker-compose build
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful!" -ForegroundColor Green
    } else {
        Write-Host "Build failed. Check Docker is running." -ForegroundColor Red
    }
}

function Start-Container {
    Write-Host "Starting Football Tracker..." -ForegroundColor Cyan
    docker-compose up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Container started!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Access the app at: http://localhost:$appPort" -ForegroundColor Yellow
        Write-Host ""
        Start-Sleep -Seconds 3
        # Try to open browser
        try {
            Start-Process "http://localhost:$appPort"
        } catch {
            Write-Host "Open your browser manually to: http://localhost:$appPort" -ForegroundColor Gray
        }
    }
}

function Stop-Container {
    Write-Host "Stopping container..." -ForegroundColor Cyan
    docker-compose down
    Write-Host "Stopped." -ForegroundColor Green
}

function Show-Logs {
    Write-Host "Showing logs (Ctrl+C to exit)..." -ForegroundColor Cyan
    docker-compose logs -f
}

function Open-Shell {
    Write-Host "Opening container shell..." -ForegroundColor Cyan
    docker-compose exec $containerName bash
}

function Clean-All {
    Write-Host "WARNING: This will delete all containers and data!" -ForegroundColor Red
    $confirm = Read-Host "Type 'yes' to continue"
    if ($confirm -eq "yes") {
        docker-compose down -v
        docker rmi football-tracker:latest 2>$null
        docker system prune -f
        Write-Host "Cleaned up." -ForegroundColor Green
    } else {
        Write-Host "Cancelled." -ForegroundColor Yellow
    }
}

function Show-Status {
    docker-compose ps
}

# Main switch
switch ($Command) {
    "build" { Build-Image }
    "run" { Start-Container }
    "stop" { Stop-Container }
    "logs" { Show-Logs }
    "shell" { Open-Shell }
    "clean" { Clean-All }
    "status" { Show-Status }
    default { Show-Help }
}
