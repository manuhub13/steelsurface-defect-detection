#  Steel Surface Defect Detection

A deep learning–based Computer Vision project to detect **six types of steel surface defects** using **Convolutional Neural Networks (CNNs)** and **OpenCV**.  
Achieved **96% accuracy** on the NEU Surface Defect Dataset.



## Project Structure

steel_defect_detection/
│
├── src/ # All source code
│ ├── train.py # Model training
│ ├── predict.py # Single image prediction
│ ├── batch_predict.py # Batch prediction for folders
│ ├── eval_from_csv.py # Evaluate predictions from CSV
│ ├── preprocessing.py # Image preprocessing logic
│ └── init.py
│
├── outputs/ # Outputs from model runs
│ ├── labeled_images/ # All labeled prediction images
│ └── sample_images/ # Small set of images for README
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

##  Dataset
**NEU Surface Defect Dataset**  
Contains 6 defect classes:
- Crazing  
- Inclusion  
- Patches  
- Pitted Surface  
- Rolled-In Scale  
- Scratches  

Each image is **200×200 pixels** grayscale, representing real steel surface textures.

Dataset link: [NEU Surface Defect Dataset](https://www.kaggle.com/datasets)

---

##  How to Run

### 1️. Install dependencies

pip install -r requirements.txt

2️. Train the model
python src/train.py

3. Predict for a single image
python src/predict.py data/NEU-CLS/Crazing/crazing_10.jpg

4. Run batch predictions
python src/batch_predict.py

5. Evaluate predictions
python src/eval_from_csv.py

## Results

Classification Report:

                  precision    recall  f1-score   support

        Crazing       0.99      1.00      0.99       283
      Inclusion       0.90      0.90      0.90       283
        Patches       0.99      1.00      0.99       284
 Pitted_Surface       0.88      0.93      0.91       283
Rolled_In_Scale       0.98      1.00      0.99       283
      Scratches       0.99      0.92      0.95       284

       accuracy                           0.96      1700
      macro avg       0.96      0.96      0.96      1700
   weighted avg       0.96      0.96      0.96      1700

##Sample Output

| Defect Type      | Predicted Output |
|------------------|------------------|
| **Crazing**      | ![Crazing](outputs/sample_images/crazing_after_output.jpg) |
| **Inclusion**    | ![Inclusion](outputs/sample_images/inclusion_after_output.jpg) |
| **Patches**      | ![Patches](outputs/sample_images/patches_after_output.jpg) |
| **Pitted Surface** | ![Pitted Surface](outputs/sample_images/pitted_surface_after_output.jpg) |
| **Rolled-In Scale** | ![Rolled-In Scale](outputs/sample_images/rolled_in_scale_after_output.jpg) |
| **Scratches**    | ![Scratches](outputs/sample_images/scratches_after_output.jpg) |

## Model Evaluation
**Confusion Matrix:**
![Confusion Matrix](outputs/confusion_matrix.png)
**Predictions CSV:**
[Download predictions.csv](outputs/predictions.csv)


##Tech Stack
Programming Language: Python 3.12
Libraries: TensorFlow/keras(Model Training), OpenCV, NumPy, Pandas, scikit-learn, Matplotlib
Techniques: CNN-based Image Classification, OpenCV Preprocessing, Batch Prediction, Evaluation Metrics
