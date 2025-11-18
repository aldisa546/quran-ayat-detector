# Makefile for scalable YOLO pipeline

.PHONY: install prepare-data convert merge augment train evaluate visualize download-images sync-labels quality-control

VISUALIZE_MODEL ?= models/yolo-ayat-detector_best.pt
VISUALIZE_TARGET ?= data/processed/images
VISUALIZE_OUTPUT ?= experiments/visualizations/latest
VISUALIZE_CONF ?= 0.25
VISUALIZE_DEVICE ?=
VISUALIZE_RECURSIVE ?= 0

install:
	pip install -r requirements.txt

prepare-data: convert augment

convert:
	python3 src/data_processing/xml_to_yolo.py \
		--xml-dir experiments/model-v1/visualizations \
		--output-dir data/processed/labels \
		--classes configs/classes.txt \
		--overwrite

merge:
	python3 src/data_processing/dataset_merger.py \
		--sources data/raw/dataset_v1 data/raw/dataset_v2 \
		--images-subdir images \
		--labels-subdir annotations \
		--dest data/raw/merged \
		--overwrite

augment:
	python3 src/data_processing/augmentation.py \
		--images-dir data/processed/images \
		--labels-dir data/processed/labels \
		--output-dir data/processed/augmented \
		--samples-per-image 1

train:
	python3 src/training/train.py \
		--data-config configs/data.yaml \
		--model-config configs/model.yaml

evaluate:
	python3 src/evaluation/evaluate.py \
		--weights models/yolo-ayat-detector_best.pt \
		--data-config configs/data.yaml

visualize:
	python3 src/data_processing/visualize_bounding_box.py \
		--model $(MODEL) \
		--target-folder $(TARGET) \
		--output-dir $(OUTPUT) \
		--conf $(CONF) \
		$(if $(DEVICE),--device $(DEVICE),) \
		$(if $(filter 1 true yes on,$(RECURSIVE)),--recursive,)

download-images:
	python3 src/data_processing/download_images.py \
		--config configs/variants.yaml \
		--output-dir data/processed/images

sync-labels:
	rsync -avz -e "ssh -p 46540" data/processed/labels/ root@108.39.26.2:/workspace/quran-ayat-detector/data/processed/labels/

sync-images:
	rsync -avz -e "ssh -p 46540" data/processed/images/ root@108.39.26.2:/workspace/quran-ayat-detector/data/processed/images/

quality-control:
	@if [ -z "$(QC_XML_DIR)" ]; then \
		echo "Error: QC_XML_DIR must point to a directory of XML files."; \
		echo "Usage: make quality-control QC_XML_DIR=path/to/xmls"; \
		exit 1; \
	fi
	python3 src/data_processing/quality_control.py \
		--xml-dir $(QC_XML_DIR) \
		$(if $(CSV),--csv $(CSV),)