# Risk Evaluation Service for Retail Brands
![risk](https://eu-images.contentstack.com/v3/assets/blt69509c9116440be8/blt33c03cb25339d5ab/65d618d442e6eb040afe24cd/RISK_2HEYDPC.jpg?width=1280&auto=webp&quality=95&format=jpg&disable=upscale)

### Proof of Concept: Home Credit Default Risk Prediction

(_Inspired by the 2018 Home Credit [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk/overview)_)
### Project Overview
This project demonstrates a proof-of-concept for risk evaluation using machine learning, focusing on predicting loan default risk. As part of a start-up product team simulation, I built a complete pipeline for risk assessment, from data exploration and preprocessing to model development and deployment. The model is aimed at helping financial institutions assess the risk of lending, especially in cases where customers have limited financial history.

### Business Problem
Financial institutions often struggle to assess the credit risk of individuals with little or no credit history, such as first-time homebuyers and small business owners. This project explores whether machine learning can improve the accuracy of these assessments by analyzing a range of customer data, including loan applications, credit bureau reports, and payment histories.

### Data Sources
The project uses a comprehensive set of financial data from Home Credit, including:
- Current loan applications
- Previous loan applications
- Historical loan balances
- Credit Bureau data
- Payment history records<br><br>
All data can be found [here](https://www.kaggle.com/c/home-credit-default-risk/data).
### Approach
1. **Data Exploration**: Analyzed and visualized the relationships within the data to understand key features related to loan default risk.
2. **Data Preprocessing**: Handled missing data, performed feature engineering, and created scalable preprocessing steps.
3. **Predictive Modeling**: Developed and tuned machine learning models to predict the likelihood of loan default, with an emphasis on performance metrics.
4. **App Deployment**: Deployed the final model as a containerized application to demonstrate its real-world use case.

### Project Structure
```bash
.
├── README.md
├── data 
(contains some ready-made folders for data created in notebooks)
├── deployment
│   ├── Dockerfile
│   ├── app
│   │   └── main.py
│   ├── data (folder for X.parquet which is generated in notebook 5_ML_models)
│   ├── model
│   │   └── model.pkl
│   └── requirements.txt 
|             (requirements for containerization)
├── notebooks 
(RUN THESE IN ORDER TO SEE MY DEVELOPMENT PROCESS FOR THIS PROJECT)
│   ├── 1_feature_investigation_main_dataset.ipynb
│   ├── 2_feature_engineering_supplementary_data.ipynb
│   ├── 3_feature_preprocess_preliminary_models.ipynb
│   ├── 4_EDA.ipynb
│   └── 5_ML_models.ipynb
├── requirements.txt 
|   (an exact replica of my development environment)
└── utils
    ├── __init__.py
    ├── feature_tools.py
    ├── machine_learning.py
    ├── plot.py
    └── utils.py
```

### Usage
I recommend creating a virtual environment, in this case I call it "home-credit".

In terminal:
```terminal
python -m venv home-credit 
```
Activate venv in terminal
```
source home-credit/bin/activate
```
side note: can deactivate venv with 
```terminal
deactivate
```
Install all requirements by first going to the directory where requirements.txt is (e.g. project root directory) 
```terminal
cd name/of/root/directory
```
and then typing in terminal:
```terminal
pip install -r requirements.txt
```

Now you are ready to run the Jupyter notebooks found in the __notebooks__ directory using your favorite IDE or 
```terminal
jupyter lab
```
Step through the notebooks sequentially to gain an understanding of my workflow and the predictive algorithm that I generated.

Moreover, in the __deployment__ directory there are all the necessary files in order run a containerized version of the loan prediction app.


### Requirements
See full list of requirements with exact versions to recreate my development environment in requirements.txt<br><br>
__Key Requirements__:
- Boruta
- jupyterlab
- lightgbm
- matplotlib
- numpy
- optuna
- pandas
- phik
- scikit-learn
- scipy
- seaborn
- shap
- tqdm

### License
[MIT](https://opensource.org/license/mit)  

#### Contact
Miguel A. Diaz-Acevedo at migueldiazacevedo@gmail.com