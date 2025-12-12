# Real Estate Value Predictor

A machine learning–powered web application designed to estimate housing prices using Linear Regression and engineered real estate features. The system is deployed with Streamlit and provides a modern interface, authentication, and real-time visual output.

## Overview
This project demonstrates a complete workflow for building, training, and deploying a regression model using the Ames Housing dataset. Users can input property characteristics such as quality, living area, garage size, and year built, and receive an estimated market value in USD.  
The pipeline includes preprocessing, scaling, encoding, and correlation-based feature selection to improve prediction accuracy.

## Features
- Predicts real estate prices using a trained Linear Regression model  
- StandardScaler applied to numerical features  
- One-hot encoding for categorical inputs  
- Correlation-based feature selection  
- Secure login system  
- Custom-designed Streamlit user interface  
- Real-time prediction results with structured summary  
- Pie chart visualization for feature contribution  
- Model artifacts stored as `.pkl` files  

## Technology Stack

### Machine Learning
- Linear Regression  
- Ridge Regression  
- StandardScaler  
- One-hot Encoding  
- Train/Validation Split  
- Cross-Validation  

### Web Application
- Streamlit  
- Python 3.x  
- Matplotlib  

## Repository Structure
RealEstateValuePredictor/
│
├── app.py # Streamlit web application
├── model.pkl # Trained ML model
├── scaler.pkl # Scaler used during training
├── columns.pkl # Ordered feature list
├── requirements.txt # Dependencies
└── README.md # Documentation

## Model Artifacts
| File          | Description |
|---------------|-------------|
| `model.pkl`   | Trained Linear Regression model |
| `scaler.pkl`  | StandardScaler fitted on training data |
| `columns.pkl` | Feature names used during training |

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt

### Launch Application
streamlit run app.py
### Default Credentials
[ Username: Ayush
Password: lolipop123 ]

--Example Prediction
Input:
Overall Quality: 7
Living Area: 2200 sq ft
Garage Cars: 2
Full Bathrooms: 2
Year Built: 2003

Output:

Estimated Price: $245,800

--Future Improvements

Additional models (Random Forest, Gradient Boosting, XGBoost)
Deployment on Streamlit Cloud or HuggingFace Spaces
Historical prediction logging
Confidence interval estimation
More detailed feature support

--License

This project is open-source and free to modify.

--Acknowledgments

Scikit-Learn
Pandas & NumPy
Streamlit
Matplotlib
Ames Housing Dataset
