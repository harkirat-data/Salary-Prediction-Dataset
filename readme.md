SALARY PREDICTOR

This dataset contains 250,000 job records designed for salary prediction, data analysis, and machine learning projects.

It includes information about job titles, experience level, education background, skills, industry type, company size, and work location.

I analyzed this dataset and build models to predict salaries based on the above features.
The trained model is serialized using Pickle and deployed on Streamlit Cloud.

Features :- 

- Live Salary Prediction — Input your profile information and get instant salary estimate
- EDA Visualizations — Box plots, violin plots, scatter plots
- XGBoost Model — R² score of 0.976 on 250K records 
- Pickle Deployment — Model pre-trained locally, loaded on cloud (because Streamlit was using too much memory)
- UI — Built with Streamlit

Project Structure :- 
  
Job/
├── app.py                            # Streamlit web app
├── save_model.py                     # Train & save model locally
├── model.pkl                         # Saved XGBoost model + metadata
├── job_salary_prediction_dataset.csv # Dataset 
├── JS.ipynb                          # EDA + Model training & comparison notebook
├── requirements.txt                  # Dependencies
└── readme.md

How It Works :-

Dataset (250K rows)
      ↓
One-Hot Encoding (Converts categorical columns into numerical format)(10 → 43 columns)
      ↓
Train-Test Split (training and testing sets- (80/20) ratio)
      ↓
XGBoost Training (locally)
      ↓
Model Saved as model.pkl
      ↓
Streamlit loads pkl → Predicts instantly without re-training

Model Result :- 

XGBoost performs well than LinearRegression and Random Forest across all the metrics

Run Locally :-

# 1. Clone the repo
git clone https://github.com/harkirat-data/Salary-Prediction-Dataset.git
cd Salary-Prediction-Dataset

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and save model (only once)
python save_model.py

# 4. Run the app
streamlit run app.py

Live Demo: https://salary-prediction-dataset-cfnu7txed6nbolzxrjwpk4.streamlit.app/