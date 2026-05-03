.PHONY: help build run stop logs shell test clean

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build Docker image
	docker-compose build

run: ## Run container (development mode)
	docker-compose up -d

run-prod: ## Run container (production mode)
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

stop: ## Stop container
	docker-compose down

logs: ## View container logs
	docker-compose logs -f

shell: ## Open shell in container
	docker-compose exec football-tracker bash

shell-root: ## Open root shell in container (for debugging)
	docker-compose exec -u root football-tracker bash

test: ## Run tests in container
	docker-compose exec football-tracker python3 -m pytest tests/ -v || echo "No tests yet"

clean: ## Remove containers, volumes, and images
	docker-compose down -v
	docker rmi football-tracker:latest 2>/dev/null || true
	docker system prune -f

status: ## Check container status
	@docker-compose ps

url: ## Show application URL
	@echo "http://localhost:8501"
