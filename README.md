# Skin Disease Classifier - SkinSure AI

This project is an AI-based Skin Disease Classifier that uses deep learning techniques to analyze and classify skin lesion images.

---

##  Features

- Preprocessing and data augmentation for robust training  
- Class imbalance handled with computed class weights  
- Transfer learning with MobileNetV2 pretrained on ImageNet  
- Fine-tuning of top layers for improved performance  
- Early stopping and model checkpointing to prevent overfitting  
- Evaluation using accuracy and classification report  

---

##  Classes Used

This version uses the **4 most frequent classes** from the HAM10000 dataset:

- **nv** – Melanocytic nevi  
- **mel** – Melanoma  
- **bcc** – Basal cell carcinoma  
- **bkl** – Benign keratosis-like lesions  

---

##  Dataset

The project uses the [**HAM10000 Dataset**](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (Kaggle).

**Note**: Only a subset of 4 classes is used for training to improve performance and balance.

- Download and extract the images and metadata.
- Images are stored in:  
  - `skin_disease_data/HAM10000_images_part_1`  
  - `skin_disease_data/HAM10000_images_part_2`

The `.csv` file `HAM10000_metadata.csv` contains the labels and metadata.

---

##  Model File

The trained Keras model is **not pushed** to GitHub (to keep the repo light and clean).

-  [Download skin_model.keras](https://drive.google.com/file/d/1iCenVIKIzp6iZLkUc-cKam2KCGJMAuNz/view?usp=drive_link)


---
 ## Disclaimer
 
This project is for educational and internship demonstration purposes only.

It is not a medical tool and should not be used for diagnosis or treatment. Always consult a licensed dermatologist for real medical concerns.

---
##  Ignored Files (`.gitignore`)

The following files and folders are excluded from version control to keep the repo clean:

- Python cache files:  
  `__pycache__/`, `*.pyc`

- Model files:  
  `*.h5`, `*.keras`

- Dataset images and CSVs:  
  `*.jpg`, `*.jpeg`, `*.png`, `*.webp`, `*.csv`  
  `skin_disease_data/` folder is ignored

⚠️ Ensure you keep local copies of both the dataset and model.

---

##  Requirements

- Python 3.7+  
- TensorFlow 2.x  
- scikit-learn  
- pandas  
- numpy

Install dependencies with:

```bash
pip install tensorflow scikit-learn pandas numpy

