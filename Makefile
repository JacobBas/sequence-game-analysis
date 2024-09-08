help:
# stupid trick that looks for targets that have ## after the target, then prints it
	@echo "Make goals:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

run: ## running the code
	uv run main.py

format: ## formatting the code using black and isort
	uv run ruff format
	uv run isort .
