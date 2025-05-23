# Skin Disease Predictor

This project implements a deep learning model to classify skin lesion images into five categories using transfer learning with MobileNetV2. The dataset is based on the HAM10000 skin lesion dataset.

---

## Features

- Preprocessing and data augmentation for robust training
- Handling of class imbalance with computed class weights
- Transfer learning with MobileNetV2 pretrained on ImageNet
- Fine-tuning of the top layers of the base model
- Early stopping and model checkpointing to prevent overfitting
- Model evaluation with detailed classification report

---

## Dataset

The HAM10000 dataset is used, and only a subset of 5 skin disease classes is considered:

- **akiec** – Actinic keratoses and intraepithelial carcinoma
- **bcc** – Basal cell carcinoma
- **bkl** – Benign keratosis-like lesions
- **mel** – Melanoma
- **df** – Dermatofibroma

- Images stored in two folders:  
  `skin_disease_data/HAM10000_images_part_1`  
  `skin_disease_data/HAM10000_images_part_2`

---

## Ignored Files (via `.gitignore`)

To keep the repository clean and avoid pushing unnecessary or large files, the following are excluded from version control:

- Python cache and compiled files:  
  `__pycache__/`, `*.pyc`

- Model files:  
  `*.h5`, `*.keras`

- Dataset images and data files:  
  `*.jpg`, `*.jpeg`, `*.png`, `*.webp`, `*.csv`  
  Also the folder `skin_disease_data/` itself is ignored to avoid uploading large datasets

Make sure to keep a local copy of your dataset and models, as they are not pushed to GitHub.

---

## Requirements

- Python 3.7+  
- TensorFlow 2.x  
- scikit-learn  
- pandas  
- numpy

Install dependencies via:

```bash
pip install tensorflow scikit-learn pandas numpy
