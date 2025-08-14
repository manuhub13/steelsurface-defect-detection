STEEL SURFACE DEFECT DETECTION

> A Computer Vision project for detecting and classifying steel surface defects using Convolutional Neural Networks (CNN), TensorFlow/Keras, and OpenCV.
> This model is trained on the NEU Surface Defect Dataset and achieves high accuracy in detecting six types of defects.

FEATURES
> Preprocessing of input images (resizing, normalization, augmentation).
> CNN-based classification for six steel defect categories:
    .Crazing
    .Inclusion
    .Patches
    .Pitted Surface
    .Rolled-In Scale
    .Scratches

> Batch prediction with CSV output (filename, true label, predicted label, confidence).
> Visual overlay of predictions on images with color-coded bounding labels using OpenCV.
> Evaluation script to generate precision, recall, and F1-score from prediction CSV.

PROJECT STRUCTURE
steel_defect_detection/
│── src/
│   ├── preprocessing.py        # Image preprocessing functions
│   ├── train.py                # CNN model training
│   ├── predict.py              # Single image prediction
│   ├── batch_predict.py        # Predict all images in a folder
│   ├── eval_from_csv.py        # Evaluate predictions from CSV
│   ├── test_preprocessing.py   # Test preprocessing on sample image
│── outputs/
│   ├── predictions.csv         # Prediction results
│   ├── labeled_images/         # Images with prediction overlays
│── requirements.txt            # Python dependencies
│── README.md                   # Project documentation
│── .gitignore

INSTALLATION & SETUP

1️. Clone the Repository
git clone git@github.com:BinduKammara33/steelsurface-defect-detection.git
cd steelsurface-defect-detection

2️. Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3️. Install Dependencies
pip install -r requirements.txt

USAGE

>Train the Model
python3 src/train.py

>Predict a Single Image
python3 src/predict.py data/NEU-CLS/Crazing/crazing_10.jpg

>Batch Prediction
python3 src/batch_predict.py

>Evaluate Predictions
python3 src/eval_from_csv.py

CLASSIFICATION REPORT EXAMPLE:

                  precision    recall  f1-score   support
Crazing             0.99      1.00      0.99       283
Inclusion           0.90      0.90      0.90       283
Patches             0.99      1.00      0.99       284
Pitted_Surface      0.88      0.93      0.91       283
Rolled_In_Scale     0.98      1.00      0.99       283
Scratches           0.99      0.92      0.95       284
Accuracy            0.96      1700
