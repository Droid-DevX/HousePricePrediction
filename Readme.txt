# Real Estate Value Predictor

A machine learning–powered web application for predicting real estate property values using regression models and engineered housing features.

## Description

This project implements an end-to-end workflow for training and deploying a house price prediction system using the Ames Housing dataset. It applies *Linear Regression* with scaling, one-hot encoding, and correlation-based feature selection. The system includes user *authentication*, data visualization (feature summary table and pie chart), and stores reusable model artifacts as .pkl files. A *Streamlit* web application provides an interactive interface where users can input property attributes and receive price predictions in USD.

## Getting Started

### Dependencies

* Python 3.x
* Required libraries listed in requirements.txt (including Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib)
* Compatible with Windows, macOS, and Linux
* Streamlit for running the web interface

### Installing

1.  Download or clone the repository:

    bash
    git clone [https://github.com/your-username/RealEstateValuePredictor.git](https://github.com/your-username/RealEstateValuePredictor.git)
    cd RealEstateValuePredictor
    

2.  Install required dependencies:

    bash
    pip install -r requirements.txt
    

3.  Ensure the following serialized model/data files, generated during the training phase, exist in the project directory:
    * model.pkl
    * scaler.pkl
    * columns.pkl

### Executing program

1.  Launch the Streamlit application:

    bash
    streamlit run app.py  # Assuming your main application file is named app.py
    

2.  Use the default login credentials:
    * *Username:* Ayush
    * *Password:* lolipop123

3.  After login:
    * Enter property attributes using the input fields.
    * View the predicted price in USD.
    * Review the feature summary table and pie chart visualization.

## Help

Common issues:

* *Missing model/scaler/column files:* Ensure .pkl files are placed in the project directory.
* *Module import errors:* Reinstall dependencies:
    bash
    pip install -r requirements.txt
    

## Authors

Contributors names and contact info

* Ayush Tandon (Droid-DevX)
    * GitHub: https://github.com/Droid-DevX/Droid-DevX

## Version History

* 0.2
    * UI improvements
    * Authentication added
    * Visualization features added
    * Model optimization using Ridge Regression
* 0.1
    * Initial regression model and basic Streamlit interface

## License

This project is licensed under the *MIT License*. See the LICENSE file for full details.

## Acknowledgments

Inspiration, code snippets, and data sources:
* Ames Housing Dataset
* Scikit-learn
* Streamlit
* Pandas & NumPy
* Matplotlib

