# Banking Transaction Fraud Detection

A student project using Python, scikit-learn Random Forest, Streamlit, and MySQL/XAMPP.

## 1. Expected dataset

Place `creditcard.csv` at `data/creditcard.csv`. It should include `Time`, `V1` through `V28`, `Amount`, and `Class` (0 = legitimate, 1 = fraud).

## 2. Start MySQL

Open XAMPP Control Panel and start **MySQL**. The default configuration in `database.py` uses MySQL at `127.0.0.1:3306`, user `root`, and a blank password. Update `DB_CONFIG` if your local setup differs.

## 3. Install dependencies

From the project folder in Command Prompt or PowerShell:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you do not already have a virtual environment, create one:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Train the model

```powershell
.\venv\Scripts\python.exe train_model.py
```

This prints precision, recall, F1, ROC-AUC, and the confusion matrix, then writes `models/random_forest.joblib`.

## 5. Launch the dashboard

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser. Or use:

```powershell
.\venv\Scripts\python.exe main.py
```

## Notes

- The historical mode predicts a dataset row without changing its feature values and compares the prediction with its known dataset label.
- Manual demo mode reuses V1–V28 from a dataset profile and changes only Time and Amount. It is for demonstration only and does not represent a real, newly captured transaction.
- The project is for academic demonstration, not real-world banking decisions.
