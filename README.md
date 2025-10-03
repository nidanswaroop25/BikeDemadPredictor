# 🚲 London Bike Sharing Prediction

This is a small project where I try to predict how many bikes will be rented in London at a given time.
I used Python, some data analysis, and a machine learning model (Random Forest) to do the predictions.

---

## What I did

* Loaded the dataset (`london_merged.csv`)
* Converted the timestamp into useful info like **hour, day of week, month, year**
* Looked at the data with some graphs (bike rentals by hour, weekday, etc.)
* Trained a Random Forest Regressor model
* Checked how good the model is using RMSE and R² score
* Plotted which features are important and compared actual vs predicted rentals

---

## How to run it

1. Clone this repo
2. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```
3. Put the dataset (`london_merged.csv`) in the same folder
4. Run the script:

   ```bash
   python bike_sharing_prediction.py
   ```

---

## Tools I used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* scikit-learn

---

That’s it 🙂 This was a fun beginner project to practice data analysis and machine learning.
