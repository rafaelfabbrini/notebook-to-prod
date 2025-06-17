TRAIN_IMAGE_NAME = trainer
API_IMAGE_NAME = api-server
MLFLOW_IMAGE_NAME = mlflow-ui
CONTAINER_DATA_FILE_PATH = /app/data.csv

define local_run_config
	-v $(abspath $(DATA_PATH)):$(CONTAINER_DATA_FILE_PATH):ro \
	-v $(abspath mlruns):/app/mlruns:rw
endef

# ----------------------------------------
# Training
# ----------------------------------------

build-training:
	docker build -f Dockerfile.train -t $(TRAIN_IMAGE_NAME) .

init-mlflow:
	@if [ ! -f "mlruns/0/meta.yaml" ]; then \
		echo "Initializing MLflow directory..."; \
		python scripts/init_mlflow.py; \
	else \
		echo "MLflow directory already initialized."; \
	fi

train: init-mlflow
	docker run --rm \
		$(if $(API_KEY),-e API_KEY=$(API_KEY)) \
		$(if $(DATA_PATH),$(local_run_config)) \
		$(TRAIN_IMAGE_NAME) \
		$(if $(DATA_PATH),--data-path=$(CONTAINER_DATA_FILE_PATH))

# --entrypoint /bin/bash \

# ----------------------------------------
# API
# ----------------------------------------

build-api:
	docker build -f Dockerfile.api -t $(API_IMAGE_NAME) .

run-api:
	docker run --rm \
		-p 8000:8000 \
		$(if $(API_KEY),-e API_KEY=$(API_KEY)) \
		-v $(abspath mlruns):/app/mlruns \
		$(API_IMAGE_NAME)

api-health:
	@curl -s -X GET http://localhost:8000/health

api-info:
	@curl -s -X GET http://localhost:8000/info

predict:
	@if [ -z "$(file)" ]; then \
		echo "Error: Please provide a JSON file path." \
		exit 1; \
	fi
	@curl -s -X POST \
		-H "Content-Type: application/json" \
		-H "X-API-Key: $(API_KEY)" \
		-d @$(file) \
		http://localhost:8000/predict

# ----------------------------------------
# MLflow
# ----------------------------------------

build-mlflow:
	docker build -f Dockerfile.mlflow -t $(MLFLOW_IMAGE_NAME) .

mlflow:
	docker run --rm \
		-p 5000:5000 \
		-v $(abspath mlruns):/app/mlruns:rw \
		$(MLFLOW_IMAGE_NAME) \
		--backend-store-uri file:/app/mlruns \
		--host 0.0.0.0
