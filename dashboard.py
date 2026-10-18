from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #111827 55%, #172554 100%);
        color: #f8fafc;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        padding: 18px;
        border-radius: 14px;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, "Model file not found."

    try:
        bundle = joblib.load(MODEL_PATH)

        # Supports the dictionary format saved by train_model.py.
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            features = bundle.get("features")
            metrics = bundle.get("metrics", {})
        else:
            # Also supports a directly saved sklearn model.
            model = bundle
            features = getattr(model, "feature_names_in_", None)
            metrics = {}

        if features is None:
            features = [
                "Time",
                *[f"V{i}" for i in range(1, 29)],
                "Amount",
            ]

        return {
            "model": model,
            "features": list(features),
            "metrics": metrics,
        }, None

    except Exception as exc:
        return None, str(exc)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None

    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None


def save_to_mysql(
    transaction_time,
    amount,
    predicted_class,
    probability,
):
    """Attempt to save a prediction; app still works if MySQL is unavailable."""
    try:
        from database import initialize_database, save_transaction

        initialize_database()

        return save_transaction(
            transaction_time=float(transaction_time),
            amount=float(amount),
            actual_class=None,
            predicted_class=int(predicted_class),
            fraud_probability=float(probability),
            input_type="manual_demo",
        ), None

    except Exception as exc:
        return None, str(exc)


def format_money(value):
    return f"${float(value):,.2f}"


# Load model
bundle, model_error = load_model()

st.title("🛡️ Banking Transaction Fraud Detection")
st.write(
    "Analyze transaction features with a trained Random Forest model."
)

if bundle is None:
    st.error(f"Unable to load the trained model: {model_error}")
    st.info(
        "Confirm that `models/random_forest.joblib` is committed to your "
        "GitHub repository and that the model was saved successfully."
    )
    st.stop()

model = bundle["model"]
features = bundle["features"]
metrics = bundle["metrics"]

# The full dataset is optional for online deployment.
dataset = load_dataset()

with st.sidebar:
    st.header("Project dashboard")
    st.success("Model loaded successfully")
    st.write("**Algorithm:** Random Forest")
    st.write(f"**Input features:** {len(features)}")
    st.write(
        "**Dataset:** Available"
        if dataset is not None
        else "**Dataset:** Optional / not bundled"
    )

    st.divider()
    st.caption(
        "Academic demonstration only. Do not use this application "
        "as the sole basis for real banking decisions."
    )

# Overview
m1, m2, m3, m4 = st.columns(4)

m1.metric("Algorithm", "Random Forest")
m2.metric("Model inputs", len(features))
m3.metric(
    "F1 Score",
    f"{float(metrics.get('f1', 0)):.3f}"
    if isinstance(metrics, dict)
    else "N/A",
)
m4.metric(
    "Dataset",
    f"{len(dataset):,} rows" if dataset is not None else "Not loaded",
)

if dataset is None:
    st.info(
        "The full credit-card dataset is not included in the cloud deployment. "
        "Manual predictions still work. Dataset exploration is available "
        "when `data/creditcard.csv` is present."
    )

st.divider()

tab_predict, tab_data, tab_metrics = st.tabs(
    ["🔍 Predict Transaction", "📊 Dataset Explorer", "📈 Model Metrics"]
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
with tab_predict:
    st.subheader("Transaction analysis")
    st.write(
        "Enter transaction details and the feature values required by the model."
    )

    if "Time" in features:
        time_default = 0.0
    else:
        time_default = 0.0

    if "Amount" in features:
        amount_default = 100.0
    else:
        amount_default = 0.0

    with st.form("transaction_form"):
        col1, col2 = st.columns(2)

        with col1:
            transaction_time = st.number_input(
                "Transaction time (seconds)",
                min_value=0.0,
                value=time_default,
                step=60.0,
            )

        with col2:
            amount = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=amount_default,
                step=10.0,
                format="%.2f",
            )

        other_features = [
            feature
            for feature in features
            if feature not in ("Time", "Amount")
        ]

        feature_values = {}

        with st.expander(
            "Advanced model features",
            expanded=False,
        ):
            st.caption(
                "Enter the actual feature values expected by your trained model. "
                "For the common Kaggle dataset, V1–V28 are anonymized features. "
                "Zero values are only a software demonstration, not a reliable "
                "representation of a real transaction."
            )

            columns = st.columns(3)

            for index, feature in enumerate(other_features):
                with columns[index % 3]:
                    feature_values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.6f",
                        key=f"input_{feature}",
                    )

        submitted = st.form_submit_button(
            "🔎 Analyze Transaction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            values = {}

            for feature in features:
                if feature == "Time":
                    values[feature] = transaction_time
                elif feature == "Amount":
                    values[feature] = amount
                else:
                    values[feature] = feature_values.get(feature, 0.0)

            input_df = pd.DataFrame(
                [[values[feature] for feature in features]],
                columns=features,
            )

            prediction = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

                probability = (
                    float(probabilities[classes.index(1)])
                    if 1 in classes
                    else 0.0
                )
            else:
                probability = float(prediction)

            st.divider()
            st.subheader("Analysis result")

            result_col, probability_col = st.columns(2)

            with result_col:
                if prediction == 1:
                    st.error("🚨 Potential Fraud Detected")
                else:
                    st.success("✅ Likely Legitimate Transaction")

            with probability_col:
                st.metric(
                    "Estimated fraud probability",
                    f"{probability * 100:.2f}%",
                )

            st.progress(
                max(0.0, min(1.0, probability)),
                text="Model-estimated fraud probability",
            )

            st.caption(
                "This is a model estimate, not a guarantee. Prediction quality "
                "depends on representative, correctly supplied input features."
            )

            with st.expander("View input sent to the model"):
                st.dataframe(
                    input_df,
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("Save prediction to MySQL"):
                if st.button("Save prediction", key="save_prediction"):
                    record_id, db_error = save_to_mysql(
                        transaction_time=transaction_time,
                        amount=amount,
                        predicted_class=prediction,
                        probability=probability,
                    )

                    if record_id is not None:
                        st.success(
                            f"Prediction saved. Record ID: {record_id}"
                        )
                    else:
                        st.warning(
                            "Prediction succeeded, but MySQL storage is "
                            "unavailable. Streamlit Cloud cannot automatically "
                            "access MySQL running on your personal XAMPP setup."
                        )
                        with st.expander("Database error details"):
                            st.code(str(db_error))

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------
with tab_data:
    st.subheader("Dataset overview")

    if dataset is None:
        st.info(
            "Dataset explorer is unavailable because the CSV is not present. "
            "To enable it locally, place the original dataset at "
            "`data/creditcard.csv`."
        )
    elif "Class" not in dataset.columns:
        st.warning("The CSV does not contain the expected 'Class' column.")
    else:
        total = len(dataset)
        fraud_count = int((dataset["Class"] == 1).sum())
        legitimate_count = int((dataset["Class"] == 0).sum())

        a, b, c = st.columns(3)
        a.metric("Total transactions", f"{total:,}")
        b.metric("Legitimate transactions", f"{legitimate_count:,}")
        c.metric("Fraud transactions", f"{fraud_count:,}")

        st.write("Transaction class distribution")

        chart_data = pd.DataFrame(
            {
                "Transaction type": ["Legitimate", "Fraud"],
                "Count": [legitimate_count, fraud_count],
            }
        ).set_index("Transaction type")

        st.bar_chart(chart_data)

        st.write("Dataset preview")
        st.dataframe(
            dataset.head(25),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------------
with tab_metrics:
    st.subheader("Model evaluation metrics")

    if isinstance(metrics, dict) and metrics:
        metric_rows = []

        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ):
            if name in metrics:
                metric_rows.append(
                    {
                        "Metric": name.replace("_", " ").title(),
                        "Score": float(metrics[name]),
                    }
                )

        if metric_rows:
            metric_df = pd.DataFrame(metric_rows)
            st.dataframe(
                metric_df,
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                metric_df.set_index("Metric")
            )
        else:
            st.info(
                "The model file does not contain recognized evaluation metrics."
            )
    else:
        st.info(
            "No saved evaluation metrics were found in the model file. "
            "Run train_model.py locally to generate them."
        )

    st.caption(
        "Fraud detection datasets are highly imbalanced. Review precision, "
        "recall, F1 score, and ROC-AUC rather than relying on accuracy alone."
    )

st.divider()
st.caption("Banking Transaction Fraud Detection | Academic project")