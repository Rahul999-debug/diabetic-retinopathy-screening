AI DIABETIC RETINOPATHY SCREENING
====================================


MODEL
-----
Architecture: EfficientNet-B0
Dataset: APTOS 2019
Classes: 5
Input: 224 x 224 retinal image

CLASSES
-------
0 = No DR
1 = Mild
2 = Moderate
3 = Severe
4 = Proliferative


FILES
-----
app.py
    Streamlit application.

best_model.pth
    Trained EfficientNet-B0 model.

gradcam_example.png
    Grad-CAM explainability example.

training_macro_f1.png
    Training performance graph.

requirements.txt
    Python dependencies.

run_app.bat
    Windows one-click launcher.


INSTALLATION ON WINDOWS
=======================

1. Install Python 3.10 or 3.11.

2. Open Command Prompt in this folder.

3. Create a virtual environment:

   python -m venv venv

4. Activate it:

   venv\Scripts\activate

5. Install dependencies:

   pip install -r requirements.txt

6. Start the application:

   python -m streamlit run app.py

OR simply double-click:

   run_app.bat


APPLICATION
===========

The application allows a user to:

1. Upload a retinal fundus image.
2. Preprocess the image.
3. Run EfficientNet-B0.
4. Predict one of five DR severity classes.
5. Display model confidence.
6. Display all class probabilities.
7. Display a screening recommendation.


IMPORTANT
=========

This is an AI screening prototype.

It is NOT a replacement for examination
by a qualified ophthalmologist.


EXPECTED FOLDER
===============

DR_Screening_Local/
    app.py
    best_model.pth
    requirements.txt
    run_app.bat
    gradcam_example.png
    training_macro_f1.png
