from flask import Flask, render_template, request
from model import predict_traffic, train_and_predict, predict_feature_for_datetime
from datetime import datetime
import pandas as pd
import pickle

app = Flask(__name__)

# -----------------------------------------------------------------------------
# 1) Load Pre-Trained Model
# -----------------------------------------------------------------------------
with open("trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------------------------------------------------------
# 2) Load or Prepare Weather Data (e.g., "all_weather_data.csv")
# -----------------------------------------------------------------------------
weather_data = pd.read_csv("all_weather_data.csv", sep=",", engine="python")


# -----------------------------------------------------------------------------
# 3) Flask Route
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None

    if request.method == "POST":
        try:
            # -----------------------------------------------------------------
            # Parse form inputs
            # -----------------------------------------------------------------
            date_str = request.form.get("date")
            time_str = request.form.get("time")  # "HH:MM"
            if not date_str or not time_str:
                raise ValueError("Date and/or time is missing.")

            # Convert time string to a Python time object
            time_obj = datetime.strptime(time_str, '%H:%M').time()

            # -----------------------------------------------------------------
            # Handle optional inputs
            # -----------------------------------------------------------------
            lufttemperatur = request.form.get('lufttemperatur')
            if not lufttemperatur or lufttemperatur.strip() == "":
                # If user didn't provide "lufttemperatur",
                # predict it using your custom train_and_predict() approach:
                model_temp, df_weather_processed, _, _ = train_and_predict(
                    weather_data, 
                    'Lufttemperatur'
                )
                # Build a pandas Timestamp combining date + hour
                prediction_datetime = pd.Timestamp(date_str) + pd.Timedelta(hours=time_obj.hour)
                lufttemperatur = predict_feature_for_datetime(
                    df_weather_processed, 
                    prediction_datetime, 
                    'Lufttemperatur'
                )
            lufttemperatur = float(lufttemperatur)

            vindretning = request.form.get("vindretning")
            vindretning = float(vindretning) if vindretning and vindretning.strip() != "" else 0.0

            vindstyrke = request.form.get("vindstyrke")
            vindstyrke = float(vindstyrke) if vindstyrke and vindstyrke.strip() != "" else 4.0

            lufttrykk = request.form.get("lufttrykk")
            lufttrykk = float(lufttrykk) if lufttrykk and lufttrykk.strip() != "" else 1013.0

            vindkast = request.form.get("vindkast")
            vindkast = float(vindkast) if vindkast and vindkast.strip() != "" else 0.0

            solskinstid = request.form.get("solskinstid")
            solskinstid = float(solskinstid) if solskinstid and solskinstid.strip() != "" else 0.0

            # -----------------------------------------------------------------
            # Call your custom predict_traffic() function
            # -----------------------------------------------------------------
            prediction = predict_traffic(
                date_str, 
                time_obj,
                vindretning,
                lufttemperatur,
                lufttrykk,
                solskinstid,
                vindkast,
                vindstyrke,
                model
            )

        except Exception as e:
            # Catch any exception and store it for display
            error_message = f"Error occurred: {str(e)}"

    # Render the page, passing any results or errors to the template
    return render_template("website.html", 
                           prediction=prediction,
                           error_message=error_message)

# -----------------------------------------------------------------------------
# 4) Run the Flask App
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(port=8080, debug=True)
