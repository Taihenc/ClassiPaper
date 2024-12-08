# Load .env file
ifneq ("$(wildcard .env)","")
    include .env
    export
endif

# Check OS and define EXTRACT_CMD for both archives
ifeq ($(OS),Windows_NT)
	EXTRACT_RAW_CMD = 7z x $(RAW_ARCHIVE) -o$(RAW_PROVIDED_DIR)
	EXTRACT_SCRAPED_CMD = 7z x $(RAW_SCRAPED) -o$(RAW_SCRAPED_DIR)
else
	EXTRACT_RAW_CMD = unrar x $(RAW_ARCHIVE) $(RAW_PROVIDED_DIR)/
	EXTRACT_SCRAPED_CMD = unrar x $(RAW_SCRAPED) $(RAW_SCRAPED_DIR)/
endif

# Targets
all: extract

extract:
	@if [ -d "$(RAW_PROVIDED_DIR)" ] && [ "$$(ls -A $(RAW_PROVIDED_DIR))" ]; then \
		echo "Warning: Destination folder $(RAW_PROVIDED_DIR) is not empty. Aborting."; \
		exit 1; \
	fi
	@if [ -d "$(RAW_SCRAPED_DIR)" ] && [ "$$(ls -A $(RAW_SCRAPED_DIR))" ]; then \
		echo "Warning: Destination folder $(RAW_SCRAPED_DIR) is not empty. Aborting."; \
		exit 1; \
	fi
	@mkdir -p $(RAW_PROVIDED_DIR)
	@mkdir -p $(RAW_SCRAPED_DIR)
	@echo "Extracting $(RAW_ARCHIVE) into $(RAW_PROVIDED_DIR)..."
	$(EXTRACT_RAW_CMD)
	@echo "Extracting $(RAW_SCRAPED) into $(RAW_SCRAPED_DIR)..."
	$(EXTRACT_SCRAPED_CMD)

clean_raw_data:
	@rm -rf $(RAW_PROVIDED_DIR)/*
	@rm -rf $(RAW_SCRAPED_DIR)/*

help:
	@echo "Targets: all, extract, clean_raw_data, help"
