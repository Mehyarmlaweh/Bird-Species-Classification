# **Introduction**

This project focuses on bird species classification using the **Caltech-UCSD Birds-200-2011** dataset. The dataset originally contains 200 bird categories, but this study is limited to 30 categories for simplicity and improved focus. 

---

# **Preprocessing & Transformations**

- **Resizing**: (229, 229), Horizontal Flipping, Random Rotation, Color Jitter.
- **Random Erasing**: Helps generalize the model by randomly removing parts of the image.
- **CutMix**: Replaces removed regions of an image with patches from another image, improving object localization.
- **YOLO for Object Detection**: Used to crop the bird region from the image, focusing on the most relevant parts.

---

# **Before and After Preprocessing**

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F10313295%2F52cff5e5b52b7b06bdcc2f161a71a531%2Fbirds.jpg?generation=1735819987512381&alt=media)
---

# **Model Selection**

- **VGG19**: Deep CNN with 19 layers (16 convolutional + 3 fully connected). Simplicity and uniform structure.
  - *Accuracy*: 0.72 on test (Public Score).

- **Inception_v3**: 48 layers including inception modules for multi-scale feature extraction.
  - *Accuracy*: 0.82 on test.

- **EfficientNet_b7**: Lightweight model with 813 layers, optimized for efficiency by scaling depth, width, and resolution.
  - *Accuracy*: 0.87 on test.

- **ConvNeXt**: Modernized CNN (~350 layers) inspired by transformers, providing larger receptive fields and advanced optimizations.
  - *Best performing model*: Accuracy > 90%.

---

# **Training Analysis**

- **Optimizer**: AdamW
  - *Learning Rate*: 1e-4 (optimized for smooth convergence).
  - *Weight Decay*: 1e-4 (reduces overfitting).

- **Scheduler**: CosineAnnealingWarmRestarts
  - *T_0*: 10 (Restart after 10 epochs).
  - *T_mult*: 2 (Gradual increase in restart intervals).

- **Loss Function**: CrossEntropyLoss
  - Effective for multi-class classification.

---

# **Epochs and Accuracy Results**

- **30 Epochs**: 96.23%  
- **50 Epochs**: 93.71%  
- **20 Epochs**: 95.81%  
- **Validation Accuracy**: 95.56%

---

# **Key Observations**

1. ConvNeXt architecture outperforms others.
2. Effective use of augmentation techniques (CutMix, Random Erasing).
3. YOLO preprocessing enhances image focus.
4. Training time: 2 hours using Kaggle T4 GPUs x2.

---

## **Reference**
This work is inspired by the paper titled:  
*"Bird Detection and Species Classification: Using YOLOv5 and Deep Transfer Learning Models"*  
Published in *IJACSA-International Journal of Advanced Computer Science and Applications*, Vol. 14, No. 7, 2023

---

## **Notebook**
Check the complete notebook for this project in my Kaggle profile: [Mehyar Mlaweh's Kaggle Profile](https://www.kaggle.com/mehyarmlaweh).

## **THANKS**
**Mehyar MLAWEH **  
[mehyar.mlaweh@dauphine.eu](mailto:mehyar.mlaweh@dauphine.eu)
