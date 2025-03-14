## **Areca Nut Stem Disease Detection**  

### **Project Overview**  
This project focuses on detecting stem diseases in areca nut trees using deep learning techniques. It employs **ResNet20v1**, a Convolutional Neural Network (CNN), to classify areca nut stems as healthy or diseased. The system aims to provide farmers with an automated and accurate disease detection solution, reducing the reliance on traditional manual inspections.

---

### **Project Structure**  

```
Areca_Nut_Stem_Disease_Detection/
│── Arecanut_dataset/                    # Dataset of areca nut stem images
│── final_testing-20230830T042904Z-001/   # Images for final model testing
│── saved_models/                         # Trained model files
│── app.ipynb                             # Jupyter Notebook for training and evaluation
│── app.py                                # Python script for model inference (prediction)
│── README.md                             # Project documentation
```

---

### **Dataset**  
- The **Arecanut_dataset/** folder contains images categorized into different classes, such as:
  - **Healthy**
  - **Stem Bleeding**
  - **Stem Cracking**  
- These images are used to train the model for disease classification.  

---

### **Technologies Used**  
- **Programming Language:** Python  
- **Frameworks & Libraries:**
  - TensorFlow/Keras – for model training and prediction  
  - OpenCV – for image preprocessing  
  - NumPy & Pandas – for data manipulation  
  - Matplotlib – for visualization  

---

### **Installation Guide**  
To set up the project on your local machine, follow these steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-repo/ArecaNutStemDiseaseDetection.git
   cd ArecaNutStemDiseaseDetection
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```



4. **Run the model for disease prediction**  
   ```bash
   python app.py
   ```

---

### **Model Training Process**  
1. **Data Preprocessing**  
   - Resize images to a fixed size  
   - Convert images to RGB format  
   - Normalize pixel values  

2. **Model Architecture (ResNet20v1)**  
   - Uses **20-layer deep ResNet model**  
   - Optimized for disease classification  
   - Trained on labeled areca nut stem images  

3. **Evaluation Metrics**  
   - Accuracy  
   - Confidence  

---

### **Usage**  
- Upload an image of an areca nut stem.  
- The model predicts whether the stem is **healthy or diseased**.  
- If diseased, the system provides a **confidence score**.

---

### **Future Enhancements**  
- Improve model accuracy with a larger dataset.  
- Develop a mobile/web app for real-time disease detection.  
- Implement **transfer learning** with advanced CNN models.  
- Integrate with IoT sensors for real-time monitoring.  

---

