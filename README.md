# content-moderating-system

A powerful and scalable Content Moderation System leveraging **Logistic Regression (LR)** for text moderation and **YOLOv8 / CNN** for image moderation, deployed on **AWS** for seamless integration and performance. 🚀

---

## 📌 Features

- **Text Moderation:**  
  - Uses **Logistic Regression (LR)** to detect offensive, harmful, or inappropriate text content.  
  - Trained with curated datasets to identify abusive language, spam, and other violations.  

- **Image Moderation:**  
  - Supports two models: **ResNet18** (for image classification).  
  - Detects violence, and other inappropriate visual content.  

- **AWS Deployment:**  
  - Scalable deployment using **AWS Lambda, EC2 , S3  **.  
  - Integration with **AWS S3** for image storage and retrieval.
  - Use of AWS SageMaker for training and testing image moderation model.

---


## 🧰 Tech Stack

| Area         | Tools Used                        |
|--------------|-----------------------------------|
| Language     | Python 3                          |
| Backend      | Flask                             |
| ML Models    | Logistic Regression, ResNet18     |
| ML Libraries | Scikit-learn, Torch, OpenCV       |
| Frontend     | HTML/CSS + JavaScript (basic)     |
| Cloud        | AWS EC2, S3, SageMaker            |

---


##  Machine Learning Models

###  Text Moderation Model
- **Dataset**: Offensive/non-offensive tweets
- **Preprocessing**: Tokenization, stopword removal, punctuation removal
- **Vectorization**: TF-IDF
- **Classifier**: Logistic Regression

###  Image Moderation Model
- **Dataset**: Roboflow dataset – Violence vs Non-Violence
- **Architecture**: Custom **ResNet18**
- **Training**: Performed in **AWS SageMaker**, optimized for inference
- **Inference**: Returns label + optionally blurred image

---
