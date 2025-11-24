# Makefile for scalable YOLO pipeline

.PHONY: install prepare-data convert merge augment train evaluate visualize download-images sync-labels quality-control train-ayah-classifier validate-dataset fix-dataset

VISUALIZE_MODEL ?= models/yolo-ayat-detector_best.pt
VISUALIZE_TARGET ?= data/processed/images
VISUALIZE_OUTPUT ?= experiments/visualizations/latest
VISUALIZE_CONF ?= 0.25
VISUALIZE_DEVICE ?=
VISUALIZE_RECURSIVE ?= 0

# Ayah classifier training variables
AYAH_TRAIN_DIR ?= datasets/data/processed/ayah_classifier_train
AYAH_VAL_DIR ?= datasets/data/processed/ayah_classifier_test
AYAH_TEST_DIR ?= datasets/data/processed/ayah_classifier_test
AYAH_OUTPUT_DIR ?= datasets/data/processed/cropped_ayah_markers_cls
AYAH_MODEL ?= yolov8n-cls.pt
AYAH_EPOCHS ?= 25
AYAH_BATCH ?= 32
AYAH_IMGSZ ?= 224
AYAH_DEVICE ?=
AYAH_PROJECT ?= experiments
AYAH_RUN_NAME ?= ayah-classifier
AYAH_FORCE_REBUILD ?=

install:
	pip install -r requirements.txt

prepare-data: convert augment

convert:
	python3 src/data_processing/xml_to_yolo.py \
		--xml-dir data/raw/dataset_v3 \
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

sync-labels-detection:
	rsync -avz -e "ssh -p 41361" data/processed/labels/ root@96.241.192.5:/workspace/quran-ayat-detector/data/processed/labels/

sync-images-detection:
	rsync -avz -e "ssh -p 41361" data/processed/images/ root@96.241.192.5:/workspace/quran-ayat-detector/data/processed/images/

sync-labels-cls-train:
	rsync -avz -e "ssh -p 41361" data/processed/ayah_classifier_train/ root@96.241.192.5:/workspace/quran-ayat-detector/data/processed/ayah_classifier_train/

sync-labels-cls-test:
	rsync -avz -e "ssh -p 41361" data/processed/ayah_classifier_test/ root@96.241.192.5:/workspace/quran-ayat-detector/data/processed/ayah_classifier_test/

quality-control:
	@if [ -z "$(QC_XML_DIR)" ]; then \
		echo "Error: QC_XML_DIR must point to a directory of XML files."; \
		echo "Usage: make quality-control QC_XML_DIR=path/to/xmls"; \
		exit 1; \
	fi
	python3 src/data_processing/quality_control.py \
		--xml-dir $(QC_XML_DIR) \
		$(if $(CSV),--csv $(CSV),)

run-ayah-classifier-inference:
	python3 run_ayah_classifier_inference.py \
		--model $(AYAH_MODEL) \
		--test-dir $(AYAH_TEST_DIR) \
		--output $(AYAH_OUTPUT_DIR)

train-ayah-classifier:
	python3 src/training/train_ayah_classifier.py \
		--train-dir $(AYAH_TRAIN_DIR) \
		--val-dir $(AYAH_VAL_DIR) \
		--test-dir $(AYAH_TEST_DIR) \
		--output-dir $(AYAH_OUTPUT_DIR) \
		--model $(AYAH_MODEL) \
		--epochs $(AYAH_EPOCHS) \
		--batch $(AYAH_BATCH) \
		--imgsz $(AYAH_IMGSZ) \
		--project $(AYAH_PROJECT) \
		--run-name $(AYAH_RUN_NAME) \
		$(if $(AYAH_DEVICE),--device $(AYAH_DEVICE),) \
		$(if $(filter 1 true yes on,$(AYAH_FORCE_REBUILD)),--force-rebuild,)

validate-dataset:
	python3 src/data_processing/validate_yolo_dataset.py \
		--data-config configs/data.yaml \
		$(if $(FIX),--fix,) \
		$(if $(VERBOSE),--verbose,)

fix-dataset:
	python3 src/data_processing/fix_yolo_dataset.py \
		--data-config configs/data.yaml \
		--clear-cache \
		$(if $(VERBOSE),--verbose,)