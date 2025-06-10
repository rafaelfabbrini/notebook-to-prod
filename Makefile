TRAIN_IMAGE_NAME = trainer
API_IMAGE_NAME = api-server
CONTAINER_DATA_PATH = /app/data

# ----------------------------------------
# Training
# ----------------------------------------

build-training:
	docker build -f Dockerfile.train -t $(TRAIN_IMAGE_NAME) .

train:
	docker run --rm \
		-v $(shell pwd)/data:$(CONTAINER_DATA_PATH) \
		$(TRAIN_IMAGE_NAME) \
		$(if $(DATA_PATH),--data-path=$(CONTAINER_DATA_PATH)/$(notdir $(DATA_PATH)))

# ----------------------------------------
# API
# ----------------------------------------

build-api:
	docker build -f Dockerfile.api -t $(API_IMAGE_NAME) .

run-api:
	docker run --rm -p 8000:8000 $(API_IMAGE_NAME)

predict:
	@curl -s -X POST \
		-H "Content-Type: application/json" \
		-d @$(PREDICT_FILE) \
		http://localhost:8000/predict
		

# ----------------------------------------
# MLflow
# ----------------------------------------
mlflow:
	mlflow ui --backend-store-uri file:./mlruns
