SERVICE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(SERVICE_DIR)/..)
DOCKERFILE := $(SERVICE_DIR)/Dockerfile
DOCKER_CONTEXT := $(PROJECT_ROOT)

CONTAINER_SERVICE_CONFIG := /app/servicer/config/api_service_hanwam_docker.yml
CONTAINER_OUTPUT_DIR := /app/outputs/servicer
OUTPUT_DIR ?= $(PROJECT_ROOT)/outputs/servicer

PROJECT_VERSION := $(shell git -C $(PROJECT_ROOT) describe --abbrev=8 --always --dirty)
ifneq ($(CI_COMMIT_TAG),)
PROJECT_VERSION := $(CI_COMMIT_TAG)
endif

IMAGE_REPO ?= hanwam-service
IMAGE_TAG ?= $(PROJECT_VERSION)
DOCKER_IMAGE := $(IMAGE_REPO):$(IMAGE_TAG)
DOCKER_IMAGE_LATEST=$(IMAGE_REPO):latest
PLATFORM ?= linux/amd64
REGISTRY_HOST ?= registry.example.com
REGISTRY_USERNAME ?=

.PHONY: docker-login
docker-login:
	@test -n "$(REGISTRY_USERNAME)" || (echo "REGISTRY_USERNAME is required" && exit 1)
	@test -n "$$REGISTRY_PASSWORD" || (echo "REGISTRY_PASSWORD is required" && exit 1)
	@printf '%s\n' "$$REGISTRY_PASSWORD" | docker login --username="$(REGISTRY_USERNAME)" --password-stdin $(REGISTRY_HOST)

.PHONY: docker-build
docker-build:
	@docker build \
		--network host \
		--platform $(PLATFORM) \
		--build-arg HTTP_PROXY \
		--build-arg HTTPS_PROXY \
		--build-arg NO_PROXY \
		--build-arg http_proxy \
		--build-arg https_proxy \
		--build-arg no_proxy \
		--build-arg SERVICE_CONFIG=$(CONTAINER_SERVICE_CONFIG) \
		-f $(DOCKERFILE) \
		-t $(DOCKER_IMAGE) \
		$(DOCKER_CONTEXT)

.PHONY: docker-push
docker-push:
	@docker push $(DOCKER_IMAGE)
	@docker tag $(DOCKER_IMAGE) $(DOCKER_IMAGE_LATEST)
	@docker push $(DOCKER_IMAGE_LATEST)

.PHONY: docker-run
docker-run:
	@mkdir -p "$(OUTPUT_DIR)"
	@docker run --rm \
		--platform $(PLATFORM) \
		-p 24243:24243 \
		--mount "type=bind,source=$(abspath $(OUTPUT_DIR)),target=$(CONTAINER_OUTPUT_DIR)" \
		$(DOCKER_IMAGE)

.PHONY: docker-run-last
docker-run-last:
	@docker run --rm \
		--platform $(PLATFORM) \
		-p 24243:24243 \
		--mount "type=bind,source=$(abspath $(OUTPUT_DIR)),target=$(CONTAINER_OUTPUT_DIR)" \
		$(DOCKER_IMAGE_LATEST)
