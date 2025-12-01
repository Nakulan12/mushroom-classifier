# Mushroom Classification – Machine Learning Project
A supervised machine learning project to classify mushrooms as **edible** or **poisonous** based on their morphological characteristics.

---

## 1. Introduction
Accurately identifying mushrooms is essential for ecological research and human safety. Many species appear visually similar, yet some contain dangerous toxins.  
This project builds a machine learning classification model that predicts mushroom edibility using structured feature data.

---

## 2. Dataset
The project uses a publicly available mushroom dataset containing:

- **61,069 rows**
- **21 predictive features**
- **Binary target variable**
  - `e` → Edible  
  - `p` → Poisonous

### Feature Categories
- Cap characteristics (shape, diameter, surface, color)  
- Gill characteristics (color, attachment, spacing)  
- Stem characteristics (height, width, root, color)  
- Veil color & type  
- Bruising / bleeding behavior  
- Habitat & season  
- Spore print color  

Dataset file: `mushroom.csv`

---

## 3. Project Workflow

### 3.1 Data Cleaning
- Imputed missing values  
- Normalized categorical symbols  
- Removed or corrected inconsistent entries  

### 3.2 Preprocessing
- Label Encoding for categorical features  
- Train/Test split (80/20)  

### 3.3 Model Training
Models evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

**Selected Model: Random Forest Classifier**  
Reason: Best accuracy, robustness, and reliability with mixed data.

### 3.4 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

**Final Model Accuracy: 99.9%**

---

## 4. Visualizations
Visual analysis includes:

- Class distribution  
- Correlation heatmap  
- Cap diameter distribution  
- Habitat distribution  
- Stem height/width histograms  

Generated using: `visualize_data.py`


---

## 6. Installation & Usage
```
6.1 Create Virtual Environment
python -m venv venv
6.2 Activate Environment
Windows:
venv\Scripts\activate
6.3 Install Dependencies
pip install -r requirements.txt
6.4 Train Model
python train_final_model.py
6.5 Evaluate Model
python evaluate_model.py
6.6 Generate Visualizations
python visualize_data.py
6.7 Launch Streamlit App (Optional)
streamlit run app.py
```
