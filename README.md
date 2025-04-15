# Multiclass_Vegetable_Image_Classification_Using_Deep_Learning
Project Jupyter notebook[https://github.com/Sainadh-Bahadursha/Multiclass_Vegetable_Image_Classification_Using_Deep_Learning/blob/main/notebook/Vegetable_Image_Classification.ipynb]
Project PDF [https://github.com/Sainadh-Bahadursha/Multiclass_Vegetable_Image_Classification_Using_Deep_Learning/blob/main/notebook/Vegetable_Image_Classification.ipynb] 
## **Overview**  
This project aims to classify vegetable images into different categories using deep learning models. The objective is to develop a model that accurately identifies the type of vegetable present in an image by utilizing convolutional neural networks (CNNs) and pretrained architectures. Several models, including Simple CNN, VGG16, ResNet50, InceptionV3, and MobileNetV2, are trained and compared to assess their performance.  

---  

## **Dataset Description**  
- **Source:** A collection of labeled images representing different types of vegetables.  
- **Classes:** Multiple vegetable categories such as tomato, potato, onion, carrot, etc.  
- **Structure:**  
  - `train/` - Training images categorized by folders.  
  - `test/` - Test images for performance evaluation.  
  - `validation/` - Validation images for model tuning.  

---  

## **Key Steps**  

### 1. **Data Preprocessing and Augmentation**  
- Resizing images to a fixed size suitable for CNN models.  
- Augmentation techniques such as rotation, flipping, zooming, and brightness adjustment to enhance generalization.  
- Splitting the dataset into training, validation, and test sets for model evaluation.  

### 2. **Feature Extraction and Model Building**  
- **Simple CNN Model:** A custom CNN architecture with multiple convolutional layers followed by fully connected layers.  
- **Pretrained Models:** Transfer learning using pretrained architectures such as VGG16, ResNet50, InceptionV3, and MobileNetV2 with fine-tuning.  
- Implemented batch normalization and dropout to control overfitting.  

### 3. **Model Training and Evaluation**  
- Training models using augmented datasets with batch size, learning rate, and early stopping configurations.  
- Evaluation metrics include accuracy, precision, recall, F1-score, and ROC-AUC to compare performance.  
- Logging results and visualizing confusion matrices for error analysis.  

### 4. **MLflow Integration for Experiment Tracking**  
- Automated logging of model parameters, evaluation metrics, and training artifacts.  
- Version control for model comparison and reproducibility.  

---  

## **Model Performance Comparison**  

| **Model**               | **Test Accuracy** | **Train Accuracy** | **Validation Accuracy** | **Overfitting** | **Inference Speed** | **Epochs** | **Training Time (min)** |  
|--------------------------|-------------------|--------------------|------------------------|------------------|---------------------|------------|--------------------|  
| Simple CNN               | 76.07%            | 98.13%             | 82.46%                 | Severe           | Fast                | 10         | 25                 |  
| Complex CNN              | 94.30%            | 99.64%             | 95.53%                 | Minimal          | Moderate            | 51         | 170                |  
| Complex CNN + Aug        | 90.31%            | 96.85%             | 92.03%                 | Low              | Moderate            | 58         | 160                |  
| VGG16                    | 89.70%            | 100%               | 95.69%                 | Moderate         | Slow                | 43         | 320                |  
| VGG16 + Aug              | 90.03%            | 98.05%             | 94.58%                 | Low              | Slow                | 20         | 320                |  
| ResNet50                 | 72.08%            | 90.87%             | 79.59%                 | High             | Moderate            | 48         | 135                |  
| ResNet50 + Aug           | 71.79%            | 71.61%             | 65.71%                 | Underfitting     | Moderate            | 20         | 60                 |  
| InceptionV3              | 91.74%            | 100%               | 97.29%                 | Minimal          | Slow                | 50         | 97                 |  
| InceptionV3 + Aug        | 93.16%            | 99.24%             | 97.13%                 | Low              | Slow                | 20         | 45                 |  
| MobileNetV2              | 94.02%            | 100%               | 98.88%                 | Minimal          | Fast                | 50         | 42                 |  
| **MobileNetV2 + Aug**    | **96.01%**        | 97.73%             | 98.09%                 | Low              | **Fastest**          | 20         | 19                 |  

---  

## **Best Model According to Each Metric**  

| **Metric**                | **Best Model**                  |
|---------------------------|---------------------------------|
| Accuracy                   | MobileNetV2 + Augmentation     |

---  

## **Model Insights**  
- **MobileNetV2 + Augmentation** achieved the highest test accuracy (96.01%) with minimal overfitting and the fastest inference time.  
- **InceptionV3 and Complex CNN models** also showed competitive accuracy but with higher resource requirements.  
- **VGG16 and ResNet50 models** exhibited slower inference speeds and moderate overfitting despite high accuracy.  
- **Data augmentation** improved model robustness and reduced overfitting, especially in complex models.  

---  

## **Recommendations**  
1. **Model Selection:** Use MobileNetV2 with augmentation for deployment due to its high accuracy and fastest inference speed.  
2. **Data Augmentation:** Refine augmentation strategies to improve class diversity and reduce overfitting.  
3. **Noise Filtering:** Apply attention-based models or fine-tune CNN architectures to handle background noise.  
4. **Optimization:** Implement ReduceLROnPlateau and EarlyStopping to reduce training time and improve convergence.  
5. **Deployment:** Use TensorFlow Lite or ONNX for lightweight deployment in edge devices.
