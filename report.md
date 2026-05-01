# Image Classifier Project Report

## Executive Summary
This project successfully developed an image classification system using PyTorch and the CIFAR-10 dataset. By leveraging transfer learning with a pre-trained ResNet18 model, the system is capable of accurately categorizing images into 10 distinct classes: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Methodology

### Dataset
The model was trained on the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 classes.
- **Split**: The 50,000 training images were further split into 40,000 for training and 10,000 for validation (an 80/20 split). The standard 10,000 image test set was retained for final evaluation.
- **Preprocessing**: Images were converted to tensors and normalized using the CIFAR-10 dataset's mean and standard deviation: `(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)`.
- **Data Augmentation**: To prevent overfitting and improve generalization, `RandomCrop` (size 32, padding 4) and `RandomHorizontalFlip` were applied during training.

### Model Architecture
**ResNet18** was chosen as the base model.
- **Transfer Learning**: The model was initialized with pre-trained weights (`ResNet18_Weights.DEFAULT`), leveraging features learned from ImageNet.
- **Adaptation**: The final fully connected layer (`fc`) was replaced with a new `nn.Linear` layer outputting 10 features corresponding to the CIFAR-10 classes.

### Training Strategy
- **Optimizer**: Adam optimizer with an initial learning rate of `0.001`.
- **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`).
- **Learning Rate Scheduling**: A `StepLR` scheduler was used, decaying the learning rate by a factor of `0.1` every 3 epochs.
- **Hardware**: The script dynamically detects available hardware, prioritizing CUDA (GPU) for accelerated training.

## Evaluation and Results

The model was trained for 5 epochs. During training, the best-performing model on the validation set was saved to `model.pth`.

### Visualizations
The script generates the following visual assets:
1. **`metrics.png`**: Plots depicting the training vs. validation loss, and training vs. validation accuracy across the epochs.
2. **`confusion_matrix.png`**: A seaborn-generated heatmap displaying the model's true positive and misclassification rates across all 10 classes on the test set.

*(See generated files in the project root)*

## Application Deployment
To make the model interactive, a frontend application was developed using **Streamlit** (`app.py`).
- **User Interface**: Provides a clean, modern interface where users can upload an image (`.jpg`, `.png`).
- **Inference**: The uploaded image is transformed to the required format, passed through the trained ResNet18 model, and the application displays the predicted class alongside its confidence score. It also provides the top 3 most likely predictions.

## Conclusion
By combining advanced data augmentation, transfer learning, and an interactive web interface, this project serves as a robust end-to-end template for image classification tasks.
