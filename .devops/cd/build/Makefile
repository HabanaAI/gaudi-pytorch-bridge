# get VEBOSE from env var or default to false
ifeq ($(VERBOSE),)
	VERBOSE = FALSE
endif
# get SW_META_REPO from env var or default to sw-build-meta-manifest
ifeq ($(SW_META_REPO),)
	SW_META_REPO = sw-build-meta-manifest
endif
# get CCACHE_DIR from env var or default
ifeq ($(CCACHE_DIR),)
	CCACHE_DIR = $(HOME)/.cd_cache/pytorch/ccache
endif
BUILD_OS ?= ubuntu20.04
WORK_DIR ?= $(CURDIR)
OUTPUT_DIR ?= $(WORK_DIR)/output
BUILDS_DIR ?= $(WORK_DIR)/builds
LOGS_DIR ?= $(WORK_DIR)/habana_logs
VENV_DIR ?= $(WORK_DIR)/venv
PACK_DIR ?= $(WORK_DIR)/packages
CSV_FILE_SUFFIX = current
# we want to save these places outside the repo/workspace scope
CACHE_DIR ?= $(HOME)/.cd_cache/pytorch/cache
CCACHE_CONF ?= $(CURDIR)/ccache.conf

DOCKER_IMAGE ?= artifactory-kfs.habana-labs.com/devops-docker-local/habana-builder-pt:$(BUILD_OS)-$(RELEASE_VERSION)
DOCKER_IMAGE_NDA ?= artifactory-kfs.habana-labs.com/devops-docker-local/habana-tools-cd:nda

SW_ARTIFACT_ROOT_URL ?= https://artifactory-kfs.habana-labs.com/artifactory/$(SW_META_REPO)/$(RELEASE_VERSION)/$(RELEASE_BUILD_ID)/$(subst .,,$(BUILD_OS))/binary

PT_VERSION ?= $(shell jq -rj '.pt_versions.current.version' $(SRC_DIR)/pytorch-integration/.devops/build_profiles/profiles.json)

# Hide or not the calls depending of VERBOSE
ifeq ($(VERBOSE),TRUE)
	HIDE =
else
	HIDE = @
endif

.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

clean-all: clean-build clean-cache ## clean all possible folders

clean-build: clean-venv ## clean the build env
	@echo removing folder
	$(HIDE)rm -rf $(BUILDS_DIR) $(LOGS_DIR) $(OUTPUT_DIR) $(PACK_DIR)

clean-cache: ## clean the shared cache and conda folders
	$(HIDE) rm -rf $(CACHE_DIR) $(CCACHE_DIR)

clean-venv: ## clean python virtual environment
	$(HIDE)rm -rf $(VENV_DIR)

validate: ## make sure we have the right variables
ifndef RELEASE_VERSION
	$(error RELEASE_VERSION is not set)
endif
ifndef RELEASE_BUILD_ID
	$(error RELEASE_BUILD_ID is not set)
endif
ifndef SRC_DIR
	$(error SRC_DIR is not set)
endif

init: ## init the environemnt - pull image and create folders
	@if [ ! -d "$(BUILDS_DIR)" ]; then \
		echo "builds folder '$(BUILDS_DIR)' does not exists, creating it ..."; \
		mkdir -p $(BUILDS_DIR); \
	fi
	@if [ ! -d "$(OUTPUT_DIR)" ]; then \
		echo "output folder '$(OUTPUT_DIR)' does not exists, creating it ..."; \
		mkdir -p $(OUTPUT_DIR); \
	fi
	@if [ ! -d "$(PACK_DIR)" ]; then \
		echo "packages folder '$(PACK_DIR)' does not exists, creating it ..."; \
		mkdir -p $(PACK_DIR); \
	fi
	@if [ ! -d "$(LOGS_DIR)" ]; then \
		echo "builds folder '$(LOGS_DIR)' does not exists, creating it ..."; \
		mkdir -p $(LOGS_DIR); \
	fi
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "venv folder '$(VENV_DIR)' does not exists, creating it ..."; \
		mkdir -p $(VENV_DIR); \
	fi
	@if [ ! -d "$(CACHE_DIR)" ]; then \
		echo "cache folder '$(CACHE_DIR)' does not exists, creating it ..."; \
		mkdir -p $(CACHE_DIR); \
	fi
	@if [ ! -d "$(CCACHE_DIR)/.ccache" ]; then \
		echo "ccache folder '$(CCACHE_DIR)' does not exists, creating it ..."; \
		mkdir -p $(CCACHE_DIR)/.ccache; \
	fi
	@echo copy ccache.conf to ccache dir $(CCACHE_DIR)/.ccache
	cp -f $(CCACHE_CONF) $(CCACHE_DIR)/.ccache/
	@echo get builder image - $(DOCKER_IMAGE)
	$(HIDE)docker pull -q $(DOCKER_IMAGE)

init-sw-deps: validate init ## download and install sw build dependencies
	@echo download and install sw depenedencies sw_release_build for $(BUILD_OS)
	$(HIDE)docker run -t --rm --network=host --user $(shell id -u):$(shell id -g) --ulimit memlock=-1:-1 \
	--tmpfs /tmp:rw,exec,nosuid,mode=777,size=40G \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/group:/etc/group:ro \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-v $(SRC_DIR):/npu-stack \
	-v $(BUILDS_DIR):/builds \
	-w /tmp \
	-e HOME=/tmp -e _STACK=/npu-stack -e _BUILD=/builds -e HABANA_NO_VENV=1 -e DOCKER_CI_ENV=true \
	-e SOURCE_DIR=/tmp/sw_release_build -e TARGET_DIR=/builds -e OS_NAME=$(BUILD_OS) \
	$(DOCKER_IMAGE) \
	"wget -nv -P /tmp $(SW_ARTIFACT_ROOT_URL)/sw_release_build-$(RELEASE_VERSION)_$(RELEASE_BUILD_ID).tar.gz && \
	mkdir -p /tmp/sw_release_build && \
	tar -xzf /tmp/sw_release_build-$(RELEASE_VERSION)_$(RELEASE_BUILD_ID).tar.gz -C /tmp/sw_release_build && \
	/npu-stack/cd_release/scripts/hl_tree_builder/tree_builder.sh /npu-stack/cd_release/config/sw_release_build.ext.csv"

init-python: validate init ## install python build dependencies
	@echo Install python build dependencies
	$(HIDE)docker run -t --rm \
	--network=host --user $(shell id -u):$(shell id -g) --ulimit memlock=-1:-1 \
	--tmpfs /tmp:rw,exec,nosuid,mode=777,size=40G \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/group:/etc/group:ro \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-v $(SRC_DIR):/npu-stack \
	-v $(BUILDS_DIR):/builds \
	-v $(CACHE_DIR):/tmp/.cache \
	-v $(LOGS_DIR):/tmp/.habana_logs \
	-v $(VENV_DIR):/tmp/.venv \
	-e HOME=/tmp -e DOCKER_CI_ENV=true -e _STACK=/npu-stack -e _BUILD=/builds \
	$(DOCKER_IMAGE) \
	'install_pytorch_build_requirements'

install-fork: validate init ## install the PT fork whl
	@echo Install python build dependencies
	$(HIDE)docker run -t --rm \
	--network=host --user $(shell id -u):$(shell id -g) --ulimit memlock=-1:-1 \
	--tmpfs /tmp:rw,exec,nosuid,mode=777,size=40G \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/group:/etc/group:ro \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-v $(SRC_DIR):/npu-stack \
	-v $(BUILDS_DIR):/builds \
	-v $(LOGS_DIR):/tmp/.habana_logs \
	-v $(VENV_DIR):/tmp/.venv \
	-v $(CACHE_DIR):/tmp/.cache \
	-w /tmp \
	-e HOME=/tmp -e DOCKER_CI_ENV=true -e _STACK=/npu-stack -e _BUILD=/builds \
	$(DOCKER_IMAGE) \
	'pip3 install -U $${PYTORCH_FORK_RELEASE_BUILD}/pkgs/*torch*.whl'

collect-pt-versions: validate init ## extract versions from build_profiles.json and save them in .env file
	@echo Collect versions from build_profiles.json into .env file
	echo "TORCH_VERSION=$(shell jq -rj '.pt_versions.current.version' $(SRC_DIR)/pytorch-integration/.devops/build_profiles/profiles.json)" >> $(SRC_DIR)/pytorch-integration/.devops/cd/scripts/.env
	echo "TORCHVISION_VERSION=$(shell jq -rj '.pt_versions.current.extras.torchvision' $(SRC_DIR)/pytorch-integration/.devops/build_profiles/profiles.json)" >> $(SRC_DIR)/pytorch-integration/.devops/cd/scripts/.env
	echo "TORCHAUDIO_VERSION=$(shell jq -rj '.pt_versions.current.extras.torchaudio' $(SRC_DIR)/pytorch-integration/.devops/build_profiles/profiles.json)" >> $(SRC_DIR)/pytorch-integration/.devops/cd/scripts/.env
	echo "TORCHTEXT_VERSION=$(shell jq -rj '.pt_versions.current.extras.torchtext' $(SRC_DIR)/pytorch-integration/.devops/build_profiles/profiles.json)" >> $(SRC_DIR)/pytorch-integration/.devops/cd/scripts/.env
	echo "TORCHDATA_VERSION=$(shell jq -rj '.pt_versions.current.extras.torchdata' $(SRC_DIR)/pytorch-integration/.devops/build_profiles/profiles.json)" >> $(SRC_DIR)/pytorch-integration/.devops/cd/scripts/.env

build-%: validate init
	@echo building $* for $(BUILD_OS)
	$(HIDE)docker run -t --rm \
	--network=host --user $(shell id -u):$(shell id -g) --ulimit memlock=-1:-1 \
	--tmpfs /tmp:rw,exec,nosuid,mode=777,size=40G \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/group:/etc/group:ro \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-v $(SRC_DIR):/npu-stack \
	-v $(BUILDS_DIR):/builds \
	-v $(CCACHE_DIR):/build_cache \
	-v $(OUTPUT_DIR):/output \
	-v $(LOGS_DIR):/tmp/.habana_logs \
	-v $(VENV_DIR):/tmp/.venv \
	-v $(CACHE_DIR):/tmp/.cache \
	-v $$HOME:$$HOME \
	-w /tmp \
	-e HOME=/tmp -e DOCKER_CI_ENV=true -e _STACK=/npu-stack -e _BUILD=/builds \
	-e RELEASE_VERSION=$(RELEASE_VERSION) -e RELEASE_BUILD_NUMBER=$(RELEASE_BUILD_ID) \
	-e HL_BUILD_INFO=$(SRC_DIR)/pytorch-integration/.devops/cd/build/build.json -e LOGS_FOLDER=/output \
	$(DOCKER_IMAGE) \
	'source /npu-stack/cd_release/scripts/build/compile.sh $*'

archive-next: CSV_FILE_SUFFIX = next
archive-next: PT_VERSION = $(shell jq -rj '.pt_versions.next.version' $(SRC_DIR)/pytorch-integration/.devops/build_profiles/profiles.json)
archive-next: archive

archive: validate init ## archive all whls to tgz package
ifndef PT_VERSION
	$(error PT_VERSION is not set)
endif
	@echo create package for pytorch version - $(PT_VERSION)
	$(HIDE)docker run -t --rm \
	--network=host --user $(shell id -u):$(shell id -g) --ulimit memlock=-1:-1 \
	--tmpfs /tmp:rw,exec,nosuid,mode=777 \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/group:/etc/group:ro \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-v $(SRC_DIR):/npu-stack \
	-v $(BUILDS_DIR):/builds \
	-v $(PACK_DIR):/output \
	-v $(LOGS_DIR):/tmp/.habana_logs \
	-w /tmp \
	-e HOME=/tmp -e _STACK=/npu-stack -e _BUILD=/builds -e HABANA_NO_VENV=1 \
	-e TARGET_DIR=/output -e OS_NAME=$(BUILD_OS) \
	$(DOCKER_IMAGE) \
	/npu-stack/cd_release/scripts/hl_tree_builder/tree_builder.sh /npu-stack/pytorch-integration/.devops/cd/build/package_$(CSV_FILE_SUFFIX).csv
	@echo archiving pytorch to tgz package
	tar -czvf $(OUTPUT_DIR)/pytorch_modules$(PT_BUILD_TYPE)-v$(PT_VERSION)_$(RELEASE_VERSION)_$(RELEASE_BUILD_ID).tgz --directory=$(PACK_DIR) .

test-nda: validate ## run NDA tests on selected package name
ifndef PACKAGE_NAME
	$(error PACKAGE_NAME is not set)
endif
	@echo get builder image - $(DOCKER_IMAGE_NDA)
	$(HIDE)docker pull -q $(DOCKER_IMAGE_NDA)
	@echo test nda - $(PACKAGE_NAME)
	$(HIDE)docker run -t --rm \
	--network=host --user $(shell id -u):$(shell id -g) \
	-v /etc/localtime:/etc/localtime:ro \
	-v /etc/group:/etc/group:ro \
	-v /etc/passwd:/etc/passwd:ro \
	-v /etc/shadow:/etc/shadow:ro \
	-v $(SRC_DIR):/npu-stack \
	-v $(PACK_DIR):/output \
	$(DOCKER_IMAGE_NDA) \
	bash -c "/npu-stack/cd_release/tests/packages/nda/run.sh /output/$(PACKAGE_NAME)"
