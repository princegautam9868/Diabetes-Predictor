Diabetes Predictor:- 

Project Overview
The Diabetes Predictor is a machine learning project designed to predict the likelihood of diabetes in individuals based on certain health metrics. This tool includes an interactive web application built with Streamlit, which enables users to enter health data and receive real-time diabetes risk predictions. This project promotes awareness of potential diabetes risks through a user-friendly interface.



Directory Structure :-
AI-ML-Project-Submissions/
└── Akriti_Verma_DiabetesPredictor/
    ├── 01_README.md           # Project overview and instructions
    ├── 02_src/                # Source code files
    ├── 03_data/               # Placeholder for additional data files
    ├── 04_requirements.txt    # Dependencies list
    ├── 05_main_script.py      # Main script to run predictions
    ├── 06_results/            # Output files, plots, or analysis



Installation:-

1. Set Up Environment
Ensure you have Python installed, then create and activate a virtual environment.

code -
python -m venv env
.\env\Scripts\activate

2. Install Dependencies
Use the 04_requirements.txt file to install the necessary packages:

code - 
pip install -r 04_requirements.txt



Usage:-
1. Running the Web Application
To launch the interactive web app, use the following command in your terminal:

code - 
streamlit run 05_main_script.py
This will open the app in your browser, where you can enter relevant health metrics to receive a 
diabetes risk prediction.



Running the Main Script:-

1. Prepare the Dataset
Ensure diabetes.csv is located in the main Diabetes_prediction/ directory:

code - 
E:\Diabetes_prediction\diabetes.csv

2. Run the Web Application
To start the app and interact with the model via the web interface, run the following command:

code - 
streamlit run 05_main_script.py
This will open the app in your browser, where you can adjust sliders for various health metrics to receive a real-time diabetes risk prediction.



Requirements:-

Dependencies are specified in 04_requirements.txt and include:
Pandas
NumPy
Scikit-Learn
Streamlit
Matplotlib
Plotly
Seaborn




















bcode
pip install -r 04_requirements.txt
Usage
Running the Web Application
To launch the interactive web app, use the following command in your terminal:

bash
Copy code
streamlit run app.py
This will open the app in your browser, where you can enter relevant health metrics to receive a diabetes risk prediction.