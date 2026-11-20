🧠 DataNarrate

Upload a CSV. Pick what to predict. See exactly what the pipeline decides, and the proof that it did it.

DataNarrate is a Streamlit app that takes any tabular dataset and a target column of your choice, then runs a fully transparent machine-learning pipeline: it profiles the data, decides how to preprocess every column using explicit rules, verifies each step on your real data, compares 8 classifiers against a fixed baseline, and explains the result in plain language.

Nothing is a black box. Every preprocessing decision carries a written reason, every step is checked, and the improvement over a naive baseline is broken down so you can see where it actually comes from.

✨ Features
Any dataset, any target. Pick any column as the target. Continuous targets are binned, rare classes are merged, ID-like targets are rejected, and features that duplicate the target (leakage) are excluded, with each decision explained.
Accurate missing-data detection. Hidden placeholders such as ?, N/A, null or blank text are converted to real missing values, so the reported missing percentage is correct. A Missing data report is shown right after upload.
Rule-based preprocessing with reasons. Every column gets a decision (drop, impute, encode, scale, clip outliers) and a plain-language reason.
Keeps heavily-missing columns. Columns are dropped only when they are almost empty (95% or more missing). Anything less is kept with a "was missing" flag or an explicit Missing category.
Preprocessing proof. Each step is re-run on your data with before/after statistics, pass/fail checks and example rows.
8 models compared. Logistic regression, random forest, gradient boosting, SVM, k-nearest neighbours, naive Bayes, a neural network and a soft-voting ensemble, all cross-validated with the same folds.
Transparent baseline and improvement ladder. The winner is compared with a fixed logistic-regression baseline, and the gain is split into class balancing, preprocessing and model choice.
Significance check. A paired t-test on per-fold scores tells you whether the winner's lead over the runner-up is real or could be noise.
Full metric suite. Accuracy, precision, recall, F1 and ROC-AUC, plus a confusion matrix for every model.
Plain-language narration of the whole run, generated with an LLM (optional).
Run history saved to a local SQLite database.
🚀 Getting started
1. Clone the repo
bash
git clone https://github.com/tanishkadhaka/datanarrate.git
cd datanarrate
2. Create an environment and install dependencies
bash
conda create -n datanarrate python=3.11
conda activate datanarrate
pip install pandas numpy scikit-learn scipy streamlit plotly python-dotenv groq pytest

(venv works just as well as conda.)

3. Add your API key (optional)

The plain-language narration uses the Groq API. Create a file named .env in the project root:

GROQ_API_KEY=your_key_here

Without a key, everything else still works. The app shows a warning and skips only the narration tab. Never commit your .env file.

4. Run the app
bash
streamlit run app.py

Open the local URL Streamlit prints (usually http://localhost:8501), upload a CSV, choose the target column and click Run DataNarrate.

Run from the command line instead
bash
python run_pipeline.py path/to/data.csv target_column
🔬 How it works
Stage	What happens	Where
0. Missing-data detection	Placeholders like ? become real missing values; a report is built	src/missing.py
1. Target preparation	Cleans the target, bins continuous targets, merges rare classes, removes leakage	src/target_preparer.py
2. Profiling	Measures missingness, cardinality, outliers and class balance for every column	src/profiler.py
3. Preprocessing decisions	Rules R1 to R8 decide what to do with each column and record why	src/preprocessor.py
4. Proof	Re-runs each decision on your data and checks it did what it claimed	src/preprocessing_proof.py
5. Model selection	Cross-validates 8 models and a baseline, builds the improvement ladder	src/model_selector.py
6. Significance	Paired t-test between the top two models	src/significance.py
7. Narration	Explains the run in plain language	src/narrator.py
8. Storage	Saves every run to SQLite	src/storage.py
🕳️ How missing values are treated

Rule R0 in src/missing.py runs first, right after upload: placeholder values such as ?, N/A, null, - and blank text are converted to real missing values, so the app reports the true missing percentage. For example, the Kaggle adult.csv shows about 0.88% missing instead of 0%. A text column that was only text because of placeholders (such as Telco TotalCharges) is converted back to numeric.

Values that might mean missing but could be real categories (such as Unknown, or 9999 in a numeric column) are only reported as warnings, never changed.

Rules R1 to R4b in src/preprocessor.py then decide per column, and each decision records a reason:

Rule	Condition	Action
R1	Identifier-like column (95% or more unique values)	Dropped before anything else
R2	95% or more missing	Dropped: almost no real values are left to learn from
R2a	Numeric, 40% to 95% missing	Kept: median fill plus a 0/1 "was missing" flag column
R2b	Categorical, 40% to 95% missing	Kept: filled with an explicit Missing category
R3	Numeric, under 40% missing	Filled with the median
R3b	Numeric, 5% to 40% missing	Also gets a "was missing" flag column
R4	Categorical, under 15% missing	Filled with the most frequent category
R4b	Categorical, 15% to 40% missing	Filled with an explicit Missing category

The thresholds are constants at the top of src/preprocessor.py and can be changed. For example, set MISSING_DROP_THRESHOLD = 100.0 to drop only completely empty columns.

Imputation runs inside the cross-validation pipeline, so test rows never influence the fill values. Rows are never deleted for missing features, only for a missing target.

In the app you will see a Missing data report after upload and a How missing values were handled table in the Preprocessing tab after a run.

📏 Other preprocessing rules
Rule	Action
R5	High-cardinality categorical column: frequency encoding
R6	Low-cardinality categorical column: one-hot encoding
R7	Numeric column: standard scaling
R8	Outliers: clipped to IQR fences (or to the 1st and 99th percentiles when the IQR is 0), learned inside each training fold; rows are never deleted
🗂️ Project structure
datanarrate/
├── app.py                    # Streamlit UI
├── run_pipeline.py           # Command-line runner
├── main.py                   # Example end-to-end run on the Telco churn data
├── fetch_adult.py            # Downloads the Adult income dataset
├── run_outlier_*.py          # Outlier (R8) experiments
├── src/
│   ├── missing.py            # R0: hidden-missing-value detection and report
│   ├── target_preparer.py    # Target cleaning, binning, leakage checks
│   ├── profiler.py           # Dataset and column profiling
│   ├── preprocessor.py       # Rules R1 to R8
│   ├── preprocessing_proof.py# Evidence that each step worked
│   ├── model_selector.py     # 8 models, baseline, improvement ladder
│   ├── ablation.py           # Per-step ablation study
│   ├── outlier.py            # Outlier clipping helpers
│   ├── significance.py       # Paired t-test
│   ├── narrator.py           # LLM narration
│   └── storage.py            # SQLite run history
├── tests/                    # pytest test suite
├── results/                  # Saved experiment outputs
├── adult_income.csv          # Sample dataset
└── mushroom.csv              # Sample dataset
🧪 Running the tests
bash
python -m pytest -q

The full suite takes a few minutes because it trains real models. To run only the missing-data tests:

bash
python -m pytest tests/test_missing.py tests/test_preprocessor.py tests/test_preprocessing_proof.py -q
📊 Sample datasets
adult_income.csv: Adult census income (target: class)
mushroom.csv: Mushroom classification
Telco customer churn (loaded by main.py)

You can also try the Kaggle version of adult.csv, where missing values are written as ?. This is exactly the case the missing-data detector was built for.

🛠️ Tech stack

Python, Streamlit, scikit-learn, pandas, NumPy, SciPy, Plotly, SQLite, Groq API, pytest.

👩‍💻 Author

Built by Tanishka Dhaka.
