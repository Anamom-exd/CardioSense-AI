# CardioSense AI – Viva Q&A (beginner-friendly)

All numbers below come from the project's own files (`outputs/model_comparison.csv`, `outputs/metrics.json`, `outputs/threshold_table.csv`, `outputs/cleaning_report.csv`). If a number here ever disagrees with those files, trust the files.

---

## 1. Objective and dataset

**Q. What does your project do?**
It predicts whether a person is likely to have cardiovascular (heart/blood-vessel) disease from routine measurements: age, gender, height, weight, blood pressure, cholesterol, glucose, smoking, alcohol and physical activity. It compares seven machine-learning models, picks one, and serves it in a Streamlit web app.

**Q. Which dataset did you use?**
The Kaggle "Cardiovascular Disease dataset" by Svetlana Ulianova: 70,000 patient records, 11 input columns plus an `id` and the target `cardio` (1 = disease, 0 = no disease). The classes are almost perfectly balanced (35,021 vs 34,979). The README records the file's SHA-256 so anyone can check they have the same file. The dataset's licence is listed as "Unknown" and we say so.

**Q. Is this a classification or regression problem?**
Binary classification: the answer is disease / no disease. The model also gives a probability.

---

## 2. Data cleaning, features and splitting

**Q. What cleaning did you do?**
70,000 rows → removed 24 exact duplicates (ignoring `id`) → 69,976 → removed 1,329 rows with impossible blood pressure (we kept systolic 70–250, diastolic 40–200, and systolic must be greater than diastolic) → removed 124 rows with impossible height/weight/BMI → **68,523 rows**. The raw data had values like systolic = 16,020 and −150, which are obviously typing errors.

**Q. What features did you create?**
`age_years` (age is stored in days; we divide by 365.25 and round) and `bmi` (= weight / height²). We dropped `id` (meaningless) and the age-in-days column (replaced by years). The model uses 12 features.

**Q. How did you split the data?**
70 / 15 / 15 into train / validation / test (47,966 / 10,278 / 10,279 rows), **stratified** (each part keeps the same share of disease cases) with a fixed random seed 42 so results are reproducible. Train = learn; validation = choose between models; test = final, untouched report card.

**Q. Why do you need a validation set as well as a test set?**
If we chose the model by looking at test results, the test score would be optimistic because we "peeked". So the model is chosen on validation, and the test set is only used to report.

**Q. Do you use scaling?** Only where needed. Logistic Regression, KNN and SVM are sensitive to feature scale, so they sit in a pipeline with `StandardScaler` that is fitted on training data only. Tree-based models do not need scaling.

---

## 3. Data leakage and duplicates across splits

**Q. What is data leakage?**
When information that wouldn't be available at prediction time (or information from the test set) sneaks into training, making scores look better than they really are.

**Q. How did you avoid it?**
The target is not among the features; scalers are fitted inside pipelines on training data only; model selection uses validation only. `outputs/leakage_check.csv` records these checks.

**Q. Are there duplicate records across train and test?**
Not exact duplicates (those were removed first). But different people can have the *same feature values*: 1,050 validation rows and 1,095 test rows have a feature vector that also appears in training (about 10%). This is natural for a dataset with few, coarse columns (cholesterol is 1/2/3, etc.), not copying of the test set. It can make the score slightly generous for those rows, and we report it openly. The same feature values do not always have the same label, which is part of why accuracy is capped (see section 6).

---

## 4. The seven models and why Gradient Boosting

**Q. Which models did you compare?**
1. Logistic Regression: a straight-line (linear) boundary; fast, easy to explain.
2. KNN (k = 25): looks at the 25 most similar patients and takes a vote.
3. Decision Tree (entropy, depth 6): a flow-chart of yes/no questions. It uses entropy (information gain) as in ID3, but scikit-learn builds CART-style trees, so it is *not* exact ID3.
4. SVM (RBF kernel): finds a curved boundary with a large margin. It has no built-in probability here, so it can't be the deployed model.
5. Random Forest: many trees voting together.
6. Gradient Boosting: trees added one after another, each fixing the previous errors (200 trees, depth 3, learning rate 0.1).
7. HistGradientBoosting: a faster boosting variant.

(An MLP/neural network was tried earlier and deliberately removed from the project.)

**Q. Why did you pick Gradient Boosting?**
It had the highest **validation ROC-AUC** (0.8002), and it gives real probabilities that the app needs. The selection rule was fixed in `train_models.py` before looking at the test set.

**Q. Random Forest has a slightly higher test accuracy (73.74% vs 73.66%). Why not pick it?**
Selection was on validation, not test, and ROC-AUC rather than accuracy. The differences between the top models are tiny (the top three are within about 0.3 percentage points), well inside normal random variation, so the choice is not statistically meaningful. Picking by test score would be the leakage mistake described above. Honest answer: several models are essentially tied; Gradient Boosting was chosen by a rule decided in advance.

**Q. Did you tune the models or try ensembles?**
We ran cross-validated experiments (tuning, extra features, voting/stacking, calibration and thresholds). None gave a meaningful gain over the baseline under a pre-declared corrected significance criterion, so the simple model was kept. These are in a separate experiments folder, not in the main pipeline.

---

## 5. Metrics

For the final model on the test set: accuracy 0.7366, precision 0.7559, recall 0.6909, F1 0.7219, ROC-AUC 0.8022.

- **Accuracy**: share of all predictions that are right (73.7% here).
- **Precision**: of the people the model flagged as diseased, how many really were (75.6%).
- **Recall (sensitivity)**: of the people who really have disease, how many the model caught (69.1%). In screening, missing a sick person is usually the costlier mistake, so recall matters.
- **F1**: the balance (harmonic mean) of precision and recall.
- **ROC-AUC**: the chance that a randomly chosen diseased person gets a higher score than a randomly chosen healthy person. 0.5 = guessing, 1.0 = perfect. It does not depend on one cut-off.
- **Confusion matrix** (test set): true negatives 4,058; false positives 1,135; false negatives 1,572; true positives 3,514. The 1,572 false negatives are sick people the model called healthy.

**Q. Why is the train accuracy close to the test accuracy?** Because the models are not overfitting: for Gradient Boosting, train accuracy is 0.7397 versus test 0.7366. The ceiling is the data, not memorisation.

---

## 6. Why accuracy is about 74%

**Q. Why not 90%?**
The data limits it. People with *identical* feature values agree on the disease label only about 58.6% of the time, so even a perfect model could not be right on all of them. The columns are coarse and self-reported (e.g. cholesterol is 1/2/3), and important clinical information (family history, ECG, lipid values, diet) is missing. Our extra experiments (tuning, ensembles, new features) did not move the score meaningfully. Any claim of 85–90% on this dataset should be questioned (look for leakage or a different dataset).

**Q. Is 74% good?**
It is typical for this dataset and well above the 50% of guessing, with ROC-AUC 0.80. It is good enough to demonstrate a machine-learning pipeline, but not good enough for real medical decisions.

---

## 7. The 0.4 / 0.6 risk bands and the decision threshold

**Q. What are the risk bands in the app?**
Probability below 0.4 → "Low estimated risk"; 0.4 to under 0.6 → "Moderate"; 0.6 or more → "High". They are fixed, **illustrative** labels, not clinical cut-offs and not tuned on data. The model's probability is also not clinically calibrated, so 0.7 does not literally mean "70% of such patients are ill".

**Q. What is the decision threshold on the Threshold tab?**
A separate idea: the cut-off at which the model's output becomes the yes/no prediction "disease". At 0.5 (default): recall 69.1%, 1,572 missed, 1,135 false alarms. Lowering it to 0.4 raises recall to 77.6% (1,140 missed) but creates more false alarms (1,664). At 0.3: recall 87.6%, 2,620 false alarms, accuracy falls to 68.4%. So there is a trade-off: catch more sick people vs. alarm more healthy ones.

**Q. Limitations of the bands?** They are arbitrary round numbers; the "Moderate" band is not validated; the same band could mean different things for different groups of patients; they must not be read as a diagnosis.

---

## 8. Limitations and why it is not a clinical tool

- Public Kaggle data of unknown collection method, partly self-reported, with data-entry errors we had to remove.
- Only 11 simple inputs; no ECG, lipid panel, family history, medication, etc.
- About 74% accuracy: roughly 3 in 10 predictions are wrong, and about 31% of true patients are missed at the default cut-off.
- No external validation on another hospital's data; no fairness analysis across groups; probabilities not calibrated.
- Not approved or evaluated as a medical device.

That is why the app shows an "educational use only" notice and does not give medical advice. It should never replace a doctor.

---

## 9. Streamlit, running locally and deployment

**Q. What is Streamlit?** A Python library that turns a script into a web app without writing HTML/JavaScript. `app/app.py` loads the saved model, shows four tabs (Predict, Threshold analysis, Model performance, Data insights) and validates inputs (e.g. systolic must exceed diastolic; BMI must be in the trained 12–60 range).

**Q. How do you run it?** Install the packages from `requirements.txt` (scikit-learn is pinned to 1.9.1 because the saved model was pickled with that version), then `python -m streamlit run app/app.py`. It opens at `http://localhost:8501`.

**Q. Is it deployed online?** **Yes.** It runs on Streamlit Community Cloud at https://cardiosense-risk-predictor.streamlit.app/ (free tier, built from the GitHub repo's `main` branch, main file `app/app.py`). Free apps can go to sleep when unused, so the first visit may take a moment. It also runs locally.

**Q. Does it work on phones?** The layout is responsive and was checked in **browser device emulation** (375×812 phone, 768×1024 tablet, desktop). The live site was checked the same way (no horizontal overflow on any tab, and test inputs gave the same probabilities as the local model). It has **not** been tested on a physical phone or tablet, or in other browsers. Streamlit Cloud adds its own toolbar and badges, which can cover part of the page on phone-width screens. Same-Wi-Fi access is documented in the README but was not verified end to end.

---

## 10. Likely difficult questions

**Q. Your best model isn't the one with the best test accuracy. Isn't that cherry-picking in reverse?**
No. It is the opposite of cherry-picking: we fixed the rule (highest validation ROC-AUC, must output probabilities) before seeing test results. The test differences are far smaller than noise.

**Q. Why ROC-AUC and not accuracy for selection?**
The dataset is balanced, but ROC-AUC measures ranking quality over all cut-offs, so it is less sensitive to one arbitrary threshold and works with the probability the app shows.

**Q. Is the 10% feature overlap between train and test a problem?**
It is a mild caveat, not leakage: the rows are different people who happen to share the same coarse values. We disclose it. Since the labels for identical features often disagree, it does not give the model a free answer.

**Q. Why is your Decision Tree not exact ID3?**
ID3 is a specific older algorithm. scikit-learn implements CART (binary splits). We set `criterion="entropy"` so splits use information gain, as ID3 does, but the tree structure is CART-style. We say this in the report.

**Q. Why can't the SVM be deployed?**
We trained it without probability estimates (to save time), so it only gives a decision score, and the app needs a probability. Its ROC-AUC is computed from the decision scores.

**Q. If recall is more important, why not lower the threshold?**
It is possible (Threshold tab), but every drop creates more false alarms. The right trade-off depends on the cost of a missed case versus an unnecessary follow-up, which is a clinical decision we cannot make from this data. We show the trade-off and keep 0.5 as default.

**Q. Which feature matters most?**
Systolic blood pressure by a clear margin (permutation importance on the validation set, `outputs/feature_importance.csv`), then age and cholesterol. This matches common medical knowledge, which is a useful sanity check. Importance shows association in this data, not cause.

**Q. How do you know your results are reproducible?**
Fixed seed (42), pinned scikit-learn version, a hashed raw file, and scripts (`preprocess.py`, `train_models.py`) that regenerate the splits and results. Re-running preprocessing in a clean copy gave byte-identical split files.

**Q. What would you do next?**
Test on physical devices, deploy to a public host, validate on an external dataset, add calibration and explanations (e.g. SHAP), and involve clinicians for sensible risk bands.
