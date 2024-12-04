RAW_ARCHIVE = './data/Data 2018-2023.rar'
RAW_PROVIDED_DIR = './data/raw-provided'

ifeq ($(OS),Windows_NT)
	EXTRACT_CMD = 7z x $(RAW_ARCHIVE) -o$(RAW_PROVIDED_DIR)
else
	EXTRACT_CMD = unrar x $(RAW_ARCHIVE) $(RAW_PROVIDED_DIR)/
endif

all: extract

extract:
	@mkdir -p $(RAW_PROVIDED_DIR)
	@echo "Extracting $(RAW_ARCHIVE) into $(RAW_PROVIDED_DIR)..."
	$(EXTRACT_CMD)

clean:
	@rm -rf $(RAW_PROVIDED_DIR)/*

help:
	@echo "Targets: all, extract, clean, help"
