from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from database import (
    get_statistics,
    get_transactions,
    initialize_database,
    save_transaction,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def get_database_status():
    try:
        initialize_database()
        return True, None
    except Exception as exc:
        return False, str(exc)


def money(value):
    return f"${float(value):,.2f}"


st.title("🛡️ Banking Transaction Fraud Detection")
st.caption(
    "Academic demonstration using a public credit-card dataset, a Random Forest model, "
    "Streamlit, and MySQL."
)

bundle = load_model_bundle()
dataset = load_dataset()

if bundle is None:
    st.error("Trained model not found. Run `train_model.py` first.")
    st.code(r".\venv\Scripts\python.exe train_model.py", language="powershell")
    st.stop()

if dataset is None:
    st.error("Dataset not found. Place `creditcard.csv` in the `data` folder.")
    st.stop()

model = bundle["model"]
features = bundle["features"]
metrics = bundle.get("metrics", {})
db_ok, db_error = get_database_status()

if not db_ok:
    st.error(f"MySQL connection failed: {db_error}")
    st.info(
        "Start MySQL from the XAMPP Control Panel. If your MySQL root account has a "
        "password, update DB_CONFIG in database.py."
    )

with st.sidebar:
    st.header("Project details")
    st.write("**Model:** Random Forest")
    st.write("**Database:** MySQL / XAMPP")
    st.write("**Dataset rows:**", f"{len(dataset):,}")
    st.write("**Input features:**", len(features))
    st.divider()
    st.warning(
        "Demo only: this project is not a certified banking security system and should "
        "not be used to approve or reject real transactions."
    )

# Summary metrics from the training test split
metric_cols = st.columns(4)
metric_cols[0].metric("Precision", f"{metrics.get('precision', 0):.3f}")
metric_cols[1].metric("Recall", f"{metrics.get('recall', 0):.3f}")
metric_cols[2].metric("F1 score", f"{metrics.get('f1', 0):.3f}")
metric_cols[3].metric("Dataset fraud cases", f"{int((dataset['Class'] == 1).sum()):,}")

if db_ok:
    try:
        stats = get_statistics()
        st.subheader("Prediction overview")
        cols = st.columns(4)
        cols[0].metric("Saved predictions", int(stats["saved_predictions"]))
        cols[1].metric("Predicted fraud", int(stats["predicted_fraud"]))
        cols[2].metric("Predicted legitimate", int(stats["predicted_legitimate"]))
        cols[3].metric(
            "Average fraud probability",
            f"{float(stats['average_fraud_probability']) * 100:.2f}%",
        )
    except Exception as exc:
        st.warning(f"Could not load database statistics: {exc}")

tab_predict, tab_history, tab_data, tab_metrics = st.tabs(
    ["🔍 Predict transaction", "🗂️ Prediction history", "📊 Dataset explorer", "📈 Model metrics"]
)

with tab_predict:
    st.subheader("Choose a testing method")
    mode = st.radio(
        "Prediction mode",
        [
            "Historical dataset transaction (unchanged features)",
            "Manual demo (amount/time with a selected feature profile)",
        ],
        help=(
            "Historical mode uses all original feature values from one dataset row. "
            "Manual demo changes only Time and Amount and is not a realistic newly captured transaction."
        ),
    )

    actual_class = None
    input_type = "historical_sample"

    with st.form("prediction_form"):
        if mode == "Historical dataset transaction (unchanged features)":
            sample_index = st.number_input(
                "Dataset row index",
                min_value=0,
                max_value=len(dataset) - 1,
                value=0,
                step=1,
                help="Zero-based row index from creditcard.csv.",
            )
            row = dataset.iloc[int(sample_index)]
            st.write(
                f"Selected row: **{int(sample_index)}** | "
                f"Amount: **{money(row['Amount'])}** | "
                f"Actual dataset label: **{'Fraud' if int(row['Class']) == 1 else 'Legitimate'}**"
            )
            st.caption("The model receives the original feature values from this row.")
            submitted = st.form_submit_button("Predict selected transaction", type="primary")
            if submitted:
                input_df = pd.DataFrame(
                    [[row[column] for column in features]],
                    columns=features,
                )
                actual_class = int(row["Class"])
                input_type = "historical_sample"
        else:
            st.warning(
                "Demo mode: V1–V28 are taken from a selected dataset row. Only Time and "
                "Amount are changed, so this is a demonstration input—not a real transaction."
            )
            sample_options = {
                "Legitimate example (Class 0)": dataset.index[dataset["Class"] == 0],
                "Fraud example (Class 1)": dataset.index[dataset["Class"] == 1],
            }
            sample_kind = st.selectbox("Choose a feature profile", list(sample_options.keys()))
            selected_indices = sample_options[sample_kind]
            profile_index = st.selectbox(
                "Choose dataset profile row",
                options=[int(i) for i in selected_indices[:1000]],
                index=0,
            )
            amount_value = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=float(dataset.loc[profile_index, "Amount"]),
                step=1.0,
                format="%.2f",
            )
            time_value = st.number_input(
                "Transaction time (seconds from dataset start)",
                min_value=0.0,
                value=float(dataset.loc[profile_index, "Time"]),
                step=60.0,
                format="%.2f",
            )
            submitted = st.form_submit_button("Predict demo transaction", type="primary")
            if submitted:
                row = dataset.loc[profile_index]
                input_df = pd.DataFrame(
                    [[row[column] for column in features]],
                    columns=features,
                )
                if "Amount" in input_df.columns:
                    input_df.loc[0, "Amount"] = amount_value
                if "Time" in input_df.columns:
                    input_df.loc[0, "Time"] = time_value
                actual_class = None
                input_type = "manual_demo"

    if submitted:
        try:
            prediction = int(model.predict(input_df[features])[0])
            probability = float(model.predict_proba(input_df[features])[0][1])

            st.divider()
            result_col, probability_col = st.columns(2)
            if prediction == 1:
                result_col.error("🚨 Model prediction: POTENTIAL FRAUD")
            else:
                result_col.success("✅ Model prediction: LIKELY LEGITIMATE")
            probability_col.metric("Estimated fraud probability", f"{probability * 100:.2f}%")
            st.progress(max(0.0, min(1.0, probability)))

            if actual_class is not None:
                actual_name = "Fraud" if actual_class == 1 else "Legitimate"
                predicted_name = "Fraud" if prediction == 1 else "Legitimate"
                if actual_class == prediction:
                    st.success(f"Dataset label: {actual_name}. Prediction matches this label.")
                else:
                    st.warning(
                        f"Dataset label: {actual_name}, but the model predicted {predicted_name}. "
                        "This is a model error on this example."
                    )

            if db_ok:
                try:
                    record_id = save_transaction(
                        transaction_time=float(input_df.iloc[0]["Time"]),
                        amount=float(input_df.iloc[0]["Amount"]),
                        actual_class=actual_class,
                        predicted_class=prediction,
                        fraud_probability=probability,
                        input_type=input_type,
                    )
                    st.success(f"Prediction saved to MySQL successfully (record ID: {record_id}).")
                except Exception as exc:
                    st.error(f"Prediction was calculated, but could not be saved to MySQL: {exc}")
            else:
                st.info("Prediction calculated, but MySQL is unavailable so it was not saved.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab_history:
    st.subheader("Saved prediction history")
    if not db_ok:
        st.info("Start MySQL in XAMPP to view saved predictions.")
    else:
        try:
            history = get_transactions(limit=500)
            if not history:
                st.info("No predictions have been saved yet.")
            else:
                history_df = pd.DataFrame(history)
                history_df["predicted_label"] = history_df["predicted_class"].map(
                    {0: "Legitimate", 1: "Fraud"}
                )
                history_df["actual_label"] = history_df["actual_class"].map(
                    {0: "Legitimate", 1: "Fraud"}
                ).fillna("Not available")
                history_df["fraud_probability"] = (
                    history_df["fraud_probability"].astype(float) * 100
                ).round(2)
                history_df = history_df.rename(
                    columns={
                        "id": "Record ID",
                        "transaction_time": "Time (seconds)",
                        "amount": "Amount",
                        "actual_label": "Dataset label",
                        "predicted_label": "Prediction",
                        "fraud_probability": "Fraud probability (%)",
                        "input_type": "Input type",
                        "created_at": "Saved at",
                    }
                )
                display_columns = [
                    "Record ID", "Time (seconds)", "Amount", "Dataset label",
                    "Prediction", "Fraud probability (%)", "Input type", "Saved at"
                ]
                st.dataframe(history_df[display_columns], use_container_width=True, hide_index=True)
                csv_data = history_df[display_columns].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download history as CSV",
                    data=csv_data,
                    file_name="banking_fraud_prediction_history.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(str(exc))

with tab_data:
    st.subheader("Dataset explorer")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total transactions", f"{len(dataset):,}")
    c2.metric("Legitimate", f"{int((dataset['Class'] == 0).sum()):,}")
    c3.metric("Fraud", f"{int((dataset['Class'] == 1).sum()):,}")

    class_counts = dataset["Class"].value_counts().sort_index().rename(
        index={0: "Legitimate", 1: "Fraud"}
    )
    st.bar_chart(class_counts)
    st.write("Preview")
    st.dataframe(dataset.head(20), use_container_width=True, hide_index=True)

with tab_metrics:
    st.subheader("Model evaluation")
    if metrics:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Metric": "Precision", "Value": metrics.get("precision", 0)},
                    {"Metric": "Recall", "Value": metrics.get("recall", 0)},
                    {"Metric": "F1 score", "Value": metrics.get("f1", 0)},
                    {"Metric": "ROC-AUC", "Value": metrics.get("roc_auc", 0)},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.info(
        "Fraud datasets are highly imbalanced. Accuracy alone can be misleading; examine "
        "precision, recall, F1, and the confusion matrix printed when training the model."
    )
