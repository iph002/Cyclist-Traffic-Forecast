import numpy as np
import pandas as pd
import os
import pickle

# Extra libraries
import matplotlib.pyplot as plt

# Scikit-Learn imports
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

########################################
# Step 1: Prepare the Traffic Data
########################################

df_trafikk = pd.read_csv("raw_data/trafikkdata.csv", sep=";", engine="python")

# Ensure 'Trafikkmengde' is numeric and drop invalid
df_trafikk["Trafikkmengde"] = pd.to_numeric(df_trafikk["Trafikkmengde"], errors="coerce")
df_trafikk.dropna(subset=["Trafikkmengde"], inplace=True)

# If 'Antall timer ugyldig' exists, apply filters
if "Antall timer ugyldig" in df_trafikk.columns:
    df_trafikk["Antall timer ugyldig"] = pd.to_numeric(df_trafikk["Antall timer ugyldig"], errors="coerce")
    df_trafikk.dropna(subset=["Antall timer ugyldig"], inplace=True)
    df_trafikk = df_trafikk[(df_trafikk["Felt"] == "Totalt") & (df_trafikk["Antall timer ugyldig"] < 1)]

# Combine date + time into a single datetime
df_trafikk["Dato_Fra_tidspunkt"] = pd.to_datetime(
    df_trafikk["Dato"] + " " + df_trafikk["Fra tidspunkt"],
    errors="coerce"
)
df_trafikk.dropna(subset=["Dato_Fra_tidspunkt"], inplace=True)

# Keep only datetime and traffic columns
df_trafikk = df_trafikk[["Dato_Fra_tidspunkt", "Trafikkmengde"]]
df_trafikk.rename(columns={"Dato_Fra_tidspunkt": "Dato"}, inplace=True)

# Set as the index
df_trafikk.set_index("Dato", inplace=True)


########################################
# Step 2: Prepare the Weather Data
########################################

path_to_files = "raw_data"
all_files = [f for f in os.listdir(path_to_files) if f.endswith(".csv") and f.startswith("Florida")]

df_weather_list = []
for file in all_files:
    full_path = os.path.join(path_to_files, file)
    info = pd.read_csv(full_path, sep=",", engine="python")
    
    # Combine date + time into a single column
    info["Dato_Tid"] = pd.to_datetime(info["Dato"] + " " + info["Tid"], errors="coerce")
    
    # Drop old columns and rename
    info.drop(columns=["Dato", "Tid"], inplace=True)
    info.rename(columns={"Dato_Tid": "Dato"}, inplace=True)
    
    # Append cleaned data
    df_weather_list.append(info)

# Concatenate all weather frames
df_weather = pd.concat(df_weather_list, ignore_index=True)

# Drop irrelevant columns if present
df_weather.drop(["Relativ luftfuktighet", "Globalstraling"], axis=1, inplace=True, errors="ignore")

# Drop rows without a valid 'Dato'
df_weather.dropna(subset=["Dato"], inplace=True)
df_weather["Dato"] = pd.to_datetime(df_weather["Dato"], errors="coerce")
df_weather.dropna(subset=["Dato"], inplace=True)
df_weather.set_index("Dato", inplace=True)

# Resample to hourly data
df_weather = df_weather.resample("60min").agg({
    "Solskinstid": "sum",
    "Lufttemperatur": "mean",
    "Vindretning": "median",
    "Vindstyrke": "mean",
    "Lufttrykk": "mean",
    "Vindkast": "mean",
})


########################################
# Step 3: Merge Traffic and Weather
########################################

df = df_trafikk.merge(df_weather, left_on=["Dato"], right_on=["Dato"])
df = df.reset_index()
df.rename(columns={"Dato": "Date"}, inplace=True)


########################################
# Step 4: Feature Engineering
########################################

def categorize_weekend_weekday(row):
    day_name = row["Date"].day_name()
    return 0 if day_name in ["Saturday", "Sunday"] else 1

def categorize_day(row):
    day_name = row["Date"].day_name()
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_name) + 1

df["DayType"] = df.apply(categorize_weekend_weekday, axis=1)
df["DayAsInteger"] = df.apply(categorize_day, axis=1)

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Hour"] = df["Date"].dt.hour


########################################
# Step 5: Train/Validation/Test Split
########################################

df_train, df_val = train_test_split(df, test_size=0.3, shuffle=False)
df_val, df_test = train_test_split(df_val, test_size=0.5, shuffle=False)

feature_cols = [
    "Solskinstid", "Lufttemperatur", "Vindretning", "Vindstyrke",
    "Lufttrykk", "Vindkast", "DayType", "DayAsInteger",
    "Year", "Month", "Day", "Hour"
]

X_train = df_train[feature_cols]
y_train = df_train["Trafikkmengde"].astype(float)


########################################
# Step 6: Handle Missing Data (Imputation)
########################################

def fill_with_mean(df_input):
    df_output = df_input.copy()
    numeric_cols = df_output.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df_output[col] = df_output[col].fillna(df_output[col].mean())
        
    return df_output

def fill_with_knn(df_input):
    df_output = df_input.copy()
    numeric_cols = df_output.select_dtypes(include=[np.number]).columns
    
    imputer = KNNImputer(n_neighbors=2)
    imputed_numeric = imputer.fit_transform(df_output[numeric_cols])
    
    imputed_numeric_df = pd.DataFrame(
        imputed_numeric, 
        columns=numeric_cols, 
        index=df_output.index
    )
    df_output[numeric_cols] = imputed_numeric_df
    
    return df_output

imputers = {
    "mean": fill_with_mean,
    "knn": fill_with_knn,
}

imputer_model = LinearRegression()
imputers_results = {}

best_imputation_method = "mean"


df_train_imputed = imputers[best_imputation_method](df_train)
df_val_imputed   = imputers[best_imputation_method](df_val)
df_test_imputed  = imputers[best_imputation_method](df_test)

X_train = df_train_imputed[feature_cols]
y_train = df_train_imputed["Trafikkmengde"].astype(float)

X_val = df_val_imputed[feature_cols]
y_val = df_val_imputed["Trafikkmengde"].astype(float)

X_test = df_test_imputed[feature_cols]
y_test = df_test_imputed["Trafikkmengde"].astype(float)


########################################
# Step 7: Fit Final Model
########################################

# Reuse imputer_model or create a new LinearRegression; here we reuse
imputer_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = imputer_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("Validation RMSE:", val_rmse)


########################################
# Step 8: Save Model
########################################

with open("trained_model.pkl", "wb") as file:
    pickle.dump(imputer_model, file)

print("Code cleaned and executed successfully.\n")


########################################
# Additional Functions
########################################

def plot_feature_importances(importances, feature_names, model_name):
    """
    Plot feature importances for RandomForest, DecisionTree, etc.
    """
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for {model_name}")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.ylabel("Relative Importance")
    plt.tight_layout()
    plt.show()


def predict_traffic(Date, Time, Vindretning, Lufttemperatur, Lufttrykk, Solskinstid, Vindkast, Vindstyrke, model):
    """
    Predict traffic volume at a given date/time and weather conditions
    using a chosen scikit-learn model (e.g., best_dt or best_rf).
    """
    def categorize_weekend_weekday_input(date_obj):
        return 0 if date_obj.day_name() in ["Saturday", "Sunday"] else 1

    def categorize_day_input(date_obj):
        return ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(
            date_obj.day_name()
        ) + 1

    date_obj = pd.to_datetime(Date)
    day_type = categorize_weekend_weekday_input(date_obj)
    day_as_integer = categorize_day_input(date_obj)

    # Break out date/time fields
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    hour = Time.hour if hasattr(Time, "hour") else pd.to_datetime(Time).hour

    # Build input row
    input_df = pd.DataFrame({
        "Solskinstid":    [Solskinstid],
        "Lufttemperatur": [Lufttemperatur],
        "Vindretning":    [Vindretning],
        "Vindstyrke":     [Vindstyrke],
        "Lufttrykk":      [Lufttrykk],
        "Vindkast":       [Vindkast],
        "DayType":        [day_type],
        "DayAsInteger":   [day_as_integer],
        "Year":           [year],
        "Month":          [month],
        "Day":            [day],
        "Hour":           [hour]
    })

    # Predict and round
    pred = model.predict(input_df)
    return round(pred[0])


def train_and_predict(df_weather_2, feature_name):
    """
    Simple rolling-based decomposition to model 'trend' + 'seasonal'
    for a single weather feature (like 'Lufttemperatur').
    """
    window_size = 365 * 24  # 365 days * 24 hours
    df_weather_copy = df_weather_2.copy()

    df_weather_copy["trend"] = df_weather_copy[feature_name].rolling(window_size).mean()
    df_weather_copy["seasonal"] = df_weather_copy[feature_name] - df_weather_copy["trend"]

    # Drop NaN values from rolling
    df_weather_copy.dropna(inplace=True)

    # Split (train/val)
    train_size = int(0.8 * len(df_weather_copy))
    train = df_weather_copy.iloc[:train_size]
    val   = df_weather_copy.iloc[train_size:]

    # Train RandomForest for trend
    X_trend = np.array(range(len(train))).reshape(-1, 1)
    y_trend = train["trend"].values

    model_trend = RandomForestRegressor()
    model_trend.fit(X_trend, y_trend)

    return model_trend, train, val, df_weather_copy


def predict_feature_for_datetime(df_weather_2, target_datetime, feature_name):
    """
    Time-series approach to predict a future (or missing) feature
    at a specific datetime, combining trend + seasonal.
    """
    assert isinstance(df_weather_2.index, pd.DatetimeIndex), "DataFrame must have a DateTime index"

    model_trend, _, _, _ = train_and_predict(df_weather_2, feature_name)

    # Predict trend
    elapsed_hours = int((target_datetime - df_weather_2.index[0]).total_seconds() / 3600.0)
    trend_pred = model_trend.predict(np.array([[elapsed_hours]]))

    # Seasonal from 1 year ago
    seasonal_time_last_year = target_datetime - pd.DateOffset(years=1)
    if seasonal_time_last_year in df_weather_2.index:
        seasonal_component = df_weather_2.loc[seasonal_time_last_year, "seasonal"]
    else:
        seasonal_component = 0

    return trend_pred + seasonal_component

