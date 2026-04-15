import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.title("Car Price Prediction Dashboard 🚗")

# Upload file
file = st.file_uploader("Upload your Car Dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape of dataset:", df.shape)

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # ================= EDA =================
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Feature Distribution")
        selected_col = st.selectbox("Select column", df.columns)
        fig2, ax2 = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax2)
        st.pyplot(fig2)

    # ================= Target =================
    target = st.selectbox("🎯 Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # ================= Split =================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ================= Model =================
    st.subheader("🤖 Model Selection")

    model_name = st.selectbox(
        "Choose Model", ["Linear Regression", "Random Forest", "SVR"]
    )

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100)
    else:
        model = SVR()

    # ================= Train =================
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ================= Metrics =================
    st.subheader("📊 Performance Metrics")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col3, col4 = st.columns(2)

    with col3:
        st.metric("MAE (Error)", round(mae, 2))
        st.metric("MSE", round(mse, 2))

    with col4:
        st.metric("RMSE (Accuracy)", round(rmse, 2))
        st.metric("R2 Score (Performance)", round(r2, 2))

    # ================= K-Fold =================
    st.subheader("🔁 K-Fold Cross Validation")

    k = st.slider("Select K value", 3, 10, 5)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    st.write("Average R2 Score:", round(np.mean(scores), 3))

    # ================= Feature Importance =================
    if model_name == "Random Forest":
        st.subheader("📌 Feature Importance")
        importance = model.feature_importances_
        feat_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_df.set_index("Feature"))

    # ================= Hyperparameter Tuning =================
    st.subheader("⚙️ Hyperparameter Tuning")

    if st.button("Run Tuning (Random Forest)"):
        params = {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10],
        }

        grid = GridSearchCV(RandomForestRegressor(), params, cv=3)
        grid.fit(X_train, y_train)

        st.success(f"Best Parameters: {grid.best_params_}")

    # ================= Prediction =================
    st.subheader("🔮 Enter Car Details")

    input_data = {}

    for col in X.columns:
        if col.lower() == "year":
            input_data[col] = st.number_input(
                f"Enter {col}", min_value=2000, max_value=2025, step=1
            )

        elif col.lower() == "kms_driven":
            input_data[col] = st.number_input(
                f"Enter {col}", min_value=0, step=1000
            )

        elif col.lower() == "owner":
            input_data[col] = st.selectbox(f"Select {col}", [0, 1, 2, 3])

        elif col.lower() == "fuel_type":
            fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            input_data[col] = {"Petrol": 2, "Diesel": 1, "CNG": 0}[fuel]

        elif col.lower() == "seller_type":
            seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
            input_data[col] = {"Dealer": 0, "Individual": 1}[seller]

        elif col.lower() == "transmission":
            trans = st.selectbox("Transmission", ["Manual", "Automatic"])
            input_data[col] = {"Manual": 1, "Automatic": 0}[trans]

        else:
            input_data[col] = st.number_input(f"Enter {col}", value=0.0)

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    st.success(f"💰 Estimated Car Price: {round(prediction[0], 2)} Lakhs")