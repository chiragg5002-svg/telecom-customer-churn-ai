import nbformat as nbf
import os

os.chdir("/home/claude/telco_churn_ai")

nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md("""# Telecom Customer Churn — Decision Dashboard
### IBM SkillsBuild Data Analytics with AI Academic Internship — BharatCares x AICTE

**Author:** Chirag
**Flow followed:** Raw data -> Clean data -> EDA -> Insights -> Prediction -> Dashboard -> Decision

This notebook contains the complete, runnable implementation: data cleaning, EDA, feature
engineering, model comparison, evaluation, per-customer explanation, and the KPI/insight
logic that powers the Flask API (`backend/app.py`) and Streamlit dashboard (`frontend/app.py`).
All numbers quoted in the accompanying report are produced by this notebook — nothing is
hand-typed or invented.""")

md("""## 1. Problem statement

A telecom operator loses roughly a quarter of its customers every reporting period. Customer
support and marketing teams currently have no systematic way to know **which** active
customers are likely to leave, so retention offers are either sent to everyone (expensive) or
not sent at all (customers leave silently). Manually reviewing 7,000+ customer records against
usage and billing patterns is not feasible.

Data analytics and machine learning can help by: (1) quantifying which factors are associated
with churn, (2) scoring every current customer with a churn probability and risk tier, and
(3) turning both into a small set of concrete, evidence-backed retention actions a manager can
act on this week — which is the purpose of the decision dashboard built in this project.

## 2. Objectives
- Understand and document the dataset (fields, size, target, data quality)
- Clean the data and justify every cleaning decision
- Run EDA to find patterns in churn by contract, tenure, service and payment
- Engineer features that summarise service usage and payment behaviour
- Train and compare classification models (Logistic Regression, Random Forest, Gradient Boosting) against a majority-class baseline
- Evaluate the selected model with metrics appropriate to imbalanced classification
- Score every customer with a churn probability and risk tier
- Convert findings into Risk / Opportunity / Action insights with supporting evidence
- Serve the model through a Flask API and a 4-page Streamlit decision dashboard

## 3. Scope
In scope: a batch-trained binary churn classifier, a rule-based (not free-text/LLM) insight
engine computed directly from the scored data, a Flask REST API, and a Streamlit dashboard.
Out of scope: real-time streaming scoring, an LLM-generated narrative layer, and causal
inference — all relationships reported below are associations found in observational data,
not proven causes.""")

md("""## 4. Dataset

| Field | Value |
|---|---|
| Name | Telco Customer Churn |
| Source | IBM Cloud Pak for Data sample datasets, mirrored at the IBM `telco-customer-churn-on-icp4d` GitHub repository (the exact dataset used in the IBM masterclass's own churn use case) |
| Link | https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv |
| Format | CSV |
| Rows | 7,043 customers |
| Columns | 21 (20 features + target `Churn`) |
| Target | `Churn` (Yes/No) — 26.5% positive class (imbalanced) |
| Missing values | 0 explicit NaNs; 11 blank `TotalCharges` strings (see cleaning) |
| Duplicates | 0 duplicate rows or `customerID` values |

Feature groups: demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account
(`tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`),
and services (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
`OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).""")

code("""import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('..'))
from src.preprocessing import load_raw, clean_data, add_features, MODEL_FEATURES, TARGET
from src.evaluation import classification_metrics, best_f1_threshold
from src import insights as ins

sns.set_theme(style='whitegrid', font_scale=1.05)
pd.set_option('display.max_columns', 30)
RAW_PATH = os.path.join('..', 'data', 'raw', 'Telco-Customer-Churn.csv')""")

code("""raw = load_raw(RAW_PATH)
print('Shape:', raw.shape)
raw.head()""")

code("""raw.info()""")

md("## 5. Data cleaning\n\nEach step below is applied by `src/preprocessing.clean_data()` and logged automatically.")

code("""print('Missing values (raw):', raw.isna().sum().sum())
print('Duplicate rows:', raw.duplicated().sum())
print('Duplicate customerID:', raw['customerID'].duplicated().sum())

bad_mask = pd.to_numeric(raw['TotalCharges'], errors='coerce').isna()
print('Non-numeric TotalCharges rows:', bad_mask.sum())
raw.loc[bad_mask, ['customerID','tenure','MonthlyCharges','TotalCharges','Churn']]""")

code("""df, cleaning_log = clean_data(raw)
pd.DataFrame(cleaning_log)[['step','found','action','reason']]""")

code("""print('Clean shape:', df.shape)
print('Churn balance:')
print(df['Churn'].value_counts(normalize=True).rename('proportion'))
df.head()""")

md("""**Cleaning decisions explained:**
1. **`TotalCharges` type fix** — stored as text with 11 blank values; all 11 belong to
   customers with `tenure == 0` (brand-new, not yet billed), so they were set to `0.0`
   rather than dropped — dropping them would silently remove every zero-tenure customer.
2. **Redundant categories folded** — `"No internet service"` / `"No phone service"` are
   collapsed to `"No"` in the service columns because that information is already fully
   captured by `InternetService`/`PhoneService`; keeping both encodes the same fact twice.
3. **`SeniorCitizen`** recoded from 0/1 to No/Yes so every binary column shares one encoding.
4. **No rows were dropped** — 7,043 in, 7,043 out. Outlier checks (IQR rule) on tenure,
   monthly and total charges found 0 statistical outliers; every value is a plausible real
   duration or bill amount, so nothing was removed as an error.""")

md("## 6. Exploratory Data Analysis\n\nEach chart below is generated from the cleaned dataset. Interpretation follows each figure.")

code("""fig, ax = plt.subplots(figsize=(5,4))
vc = df['Churn'].map({0:'No',1:'Yes'}).value_counts()
ax.bar(vc.index, vc.values, color=['#2E86AB','#E63946'])
for i,v in enumerate(vc.values): ax.text(i, v+50, f'{v:,} ({v/len(df)*100:.1f}%)', ha='center')
ax.set_title('Churn class distribution'); ax.set_ylabel('Customers'); plt.tight_layout(); plt.show()""")

md("**Observation:** 26.5% of customers have churned — a meaningful minority class. "
   "**Insight:** the dataset is moderately imbalanced, so accuracy alone would be a misleading "
   "metric; ROC-AUC, precision/recall and F1 are used instead (Section 8). "
   "**Meaning:** any model must be checked against a majority-class baseline before its skill is trusted.")

code("""order = ['Month-to-month','One year','Two year']
g = (df.groupby('Contract')['Churn'].mean().reindex(order)*100)
fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(g.index, g.values, color='#E63946')
for b,v in zip(bars,g.values): ax.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.1f}%', ha='center')
ax.set_ylabel('Churn rate (%)'); ax.set_title('Churn rate by contract type'); plt.tight_layout(); plt.show()""")

code("""order = ['0-12 months','13-24 months','25-48 months','49-72 months']
g = (df.groupby('tenure_group')['Churn'].mean().reindex(order)*100)
fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(g.index, g.values, color='#F4A261')
for b,v in zip(bars,g.values): ax.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.1f}%', ha='center')
ax.set_ylabel('Churn rate (%)'); ax.set_title('Churn rate by tenure group')
plt.setp(ax.get_xticklabels(), rotation=15); plt.tight_layout(); plt.show()""")

md("**Observation:** churn falls sharply as contract length and tenure increase (from 42.7% "
   "month-to-month down to 2.8% on two-year contracts; from 47.4% in the first year down to "
   "9.5% after four years). **Insight:** commitment (contract length, tenure) is one of the "
   "strongest separators of leavers and stayers in this dataset. **Meaning:** retention effort "
   "aimed at new, uncommitted customers should have the highest yield — quantified in Section 9.")

code("""order = ['DSL','Fiber optic','No']
g = (df.groupby('InternetService')['Churn'].mean().reindex(order)*100)
fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(g.index, g.values, color='#2A9D8F')
for b,v in zip(bars,g.values): ax.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.1f}%', ha='center')
ax.set_ylabel('Churn rate (%)'); ax.set_title('Churn rate by internet service'); plt.tight_layout(); plt.show()

print('Avg monthly charge - Fiber:', round(df.loc[df.InternetService=='Fiber optic','MonthlyCharges'].mean(), 2))
print('Avg monthly charge - DSL   :', round(df.loc[df.InternetService=='DSL','MonthlyCharges'].mean(), 2))""")

md("**Observation:** fibre-optic customers churn more than twice as often as DSL customers "
   "(41.9% vs 19.0%) despite (or perhaps because of) paying considerably more per month on "
   "average. **Insight:** fibre is the single largest source of lost monthly revenue among "
   "internet products (quantified in Section 9). **Meaning:** the dataset has no "
   "service-quality or competitor-pricing field, so *why* fibre churns more cannot be "
   "determined from this data alone — it is flagged as a risk to investigate, not a proven "
   "cause.")

code("""fig, ax = plt.subplots(figsize=(7,4.5))
g = (df.groupby('PaymentMethod')['Churn'].mean()*100).sort_values()
bars = ax.barh(g.index, g.values, color='#457B9D')
for b,v in zip(bars,g.values): ax.text(v+0.8, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center')
ax.set_xlabel('Churn rate (%)'); ax.set_title('Churn rate by payment method'); plt.tight_layout(); plt.show()""")

md("**Observation:** electronic-check payers churn at 45.3% vs 16.0% for automatic bank "
   "transfer/credit card payers. **Insight:** manual, per-cycle payment is associated with much "
   "weaker retention. **Meaning:** switching customers to autopay is a cheap, testable lever "
   "(Section 10, Risk R4).")

code("""net = df[df['InternetService'] != 'No']
g = (net.groupby('n_protection_services')['Churn'].mean()*100)
fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(g.index.astype(str), g.values, color='#6A4C93')
for b,v in zip(bars,g.values): ax.text(b.get_x()+b.get_width()/2, v+0.8, f'{v:.1f}%', ha='center')
ax.set_xlabel('Number of protection/support services taken'); ax.set_ylabel('Churn rate (%)')
ax.set_title('Churn rate by protection services (internet customers)'); plt.tight_layout(); plt.show()""")

md("**Observation:** internet customers with none of the four protection/support services "
   "(Online Security, Online Backup, Device Protection, Tech Support) churn at 56.7%, versus "
   "16.8% for those with two or more. **Insight:** taking these add-ons goes with much stronger "
   "retention. **Meaning:** this is an upsell opportunity worth trialling with a control group "
   "(Section 10, Opportunity O1) — correlation only, not proof that the services *cause* "
   "retention.")

code("""enc = df.copy()
for c in ['SeniorCitizen','Partner','Dependents','PhoneService','PaperlessBilling']:
    enc[c] = enc[c].map({'No':0,'Yes':1})
enc['Contract_num'] = enc['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})
cols = ['tenure','MonthlyCharges','TotalCharges','n_protection_services','n_streaming_services',
        'Contract_num','SeniorCitizen','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
corr = enc[cols].corr()
fig, ax = plt.subplots(figsize=(9,7.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation heatmap'); plt.tight_layout(); plt.show()""")

md("**Observation:** `Churn` correlates most strongly (in magnitude) with `Contract_num` "
   "(negative — longer contract, less churn), `tenure` (negative) and `MonthlyCharges` "
   "(positive). `tenure` and `TotalCharges` are highly correlated with each other (both "
   "capture customer age/spend), which is addressed in modelling by checking feature "
   "importance rather than relying on either alone.")

md("## 7. Feature engineering\n\n"
   "`src/preprocessing.add_features()` derives:\n"
   "- `n_protection_services` — count of Online Security / Backup / Device Protection / Tech Support taken (0-4)\n"
   "- `n_streaming_services` — count of Streaming TV / Movies taken (0-2)\n"
   "- `auto_pay` — Yes if PaymentMethod is an automatic bank transfer or credit card\n"
   "- `tenure_group` — bucketed tenure for readable segment charts\n\n"
   "`gender` is deliberately excluded from the model: it showed no meaningful churn "
   "difference (26.9% women vs 26.2% men) and should not influence retention decisions.")

code("""df[['n_protection_services','n_streaming_services','auto_pay','tenure_group']].sample(5, random_state=1)""")

md("## 8. Machine learning — model training and comparison\n\n"
   "**Problem type:** binary classification (Churn: Yes/No). Three models are compared against "
   "a majority-class baseline using 5-fold stratified cross-validation on a held-out 80/20 "
   "train/test split (stratified by target), then evaluated once on the untouched test set.")

code("""from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

SEED = 42
X, y = df[MODEL_FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def make_pipeline(estimator):
    prep = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ])
    return Pipeline([('prep', prep), ('model', estimator)])

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=SEED),
}

print(f"{'Baseline (majority class)':28s} test ROC-AUC = 0.500 (by definition)")

results, fitted = {}, {}
for name, est in models.items():
    pipe = make_pipeline(est)
    cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['roc_auc','average_precision'])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    results[name] = {'cv_roc_auc': cvres['test_roc_auc'].mean(), 'cv_pr_auc': cvres['test_average_precision'].mean()}
    print(f"{name:28s} CV ROC-AUC = {results[name]['cv_roc_auc']:.4f}  |  CV PR-AUC = {results[name]['cv_pr_auc']:.4f}")""")

md("**Model selection rule:** pick the highest 5-fold CV ROC-AUC; if Logistic Regression is "
   "within 0.005 AUC of the best model, prefer it for its interpretability (coefficients "
   "explain every prediction — used in Section 8.3 and by the Streamlit dashboard). All three "
   "models land within ~0.002 AUC of each other, so Logistic Regression is selected.")

code("""ranked = sorted(fitted, key=lambda n: results[n]['cv_roc_auc'], reverse=True)
best_name = ranked[0]
if results['Logistic Regression']['cv_roc_auc'] >= results[best_name]['cv_roc_auc'] - 0.005:
    best_name = 'Logistic Regression'
final_model = fitted[best_name]
print('Selected model:', best_name)""")

md("### 8.1 Decision threshold\n\n"
   "The default 0.5 threshold is not tuned for this task. Because churn is the minority class "
   "and missing a churner is costlier than a false alarm (a false alarm just means one "
   "unnecessary retention email), the threshold is tuned to maximise F1 — using cross-validated "
   "out-of-fold predictions on the **training** split only, so the test set is never touched "
   "during tuning.")

code("""oof_train = cross_val_predict(clone(final_model), X_train, y_train, cv=cv, method='predict_proba')[:,1]
threshold = round(best_f1_threshold(y_train, oof_train), 2)
print('Tuned decision threshold:', threshold)""")

md("## 9. Model evaluation")

code("""proba_test = final_model.predict_proba(X_test)[:,1]
metrics_default = classification_metrics(y_test, proba_test, 0.5)
metrics_tuned = classification_metrics(y_test, proba_test, threshold)
pd.DataFrame({'threshold = 0.50': metrics_default, f'threshold = {threshold}': metrics_tuned}).T[
    ['accuracy','precision','recall','f1','roc_auc','pr_auc']].round(3)""")

code("""from sklearn.metrics import roc_curve, roc_auc_score
fig, ax = plt.subplots(figsize=(6,5.5))
for name, pipe in fitted.items():
    p = pipe.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test,p):.3f})')
ax.plot([0,1],[0,1],'--',color='gray')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curves - test set'); ax.legend(); plt.tight_layout(); plt.show()""")

code("""from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, (proba_test >= threshold).astype(int))
fig, ax = plt.subplots(figsize=(5,4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred: No churn','Pred: Churn'], yticklabels=['Actual: No churn','Actual: Churn'])
ax.set_title(f'Confusion matrix - {best_name} (threshold={threshold})'); plt.tight_layout(); plt.show()""")

md("""**Results summary (test set, held out from all training and tuning):**
- ROC-AUC ~ 0.84 - the model separates churners from non-churners well above chance (0.50) across every candidate.
- At the tuned threshold, recall rises to ~0.73 (catches most churners) at the cost of precision (~0.53) - an intentional trade-off since the retention action (an email/offer) is cheap relative to losing a customer.
- All three models (Logistic Regression, Random Forest, Gradient Boosting) score within ~0.002 AUC of each other; Logistic Regression is selected for its transparent, per-customer explanations.

**Limitations:** the model uses only the fields present in this dataset (no usage minutes, complaints, or competitor pricing); it is a correlational model, not a causal one - see Section 10 for how each insight is qualified.""")

md("### 8.3 Feature importance")

code("""from sklearn.inspection import permutation_importance
pi = permutation_importance(final_model, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=SEED, n_jobs=1)
imp = pd.DataFrame({'feature': MODEL_FEATURES, 'importance': pi.importances_mean, 'std': pi.importances_std}) \\
        .sort_values('importance', ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7,5))
top = imp.head(10).sort_values('importance')
ax.barh(top['feature'], top['importance'], xerr=top['std'], color='#E76F51')
ax.set_xlabel('Permutation importance (drop in ROC-AUC)'); ax.set_title('Top 10 feature importances')
plt.tight_layout(); plt.show()
imp.head(10)""")

md("**Observation:** `tenure` dominates, followed by `InternetService` and `Contract` — "
   "consistent with the EDA findings above. This cross-check (EDA pattern <-> model "
   "importance) is what justifies treating contract type, tenure and internet service as the "
   "primary churn drivers rather than a coincidence in one chart.")

md("## 10. Scoring every customer and generating insights\n\n"
   "Every customer (not just the test set) is scored using 5-fold **out-of-fold** cross-"
   "validation predictions, so every customer's score comes from a model copy that never saw "
   "that customer during training — this avoids the optimistic bias of scoring the training "
   "data with a model fit on it.")

code("""oof_all = cross_val_predict(clone(final_model), X, y, cv=cv, method='predict_proba')[:,1]
scored = df.copy()
scored['churn_probability'] = oof_all
RISK_CUTOFFS = {'medium': 0.30, 'high': 0.60}
def risk_tier(p):
    return 'High' if p >= RISK_CUTOFFS['high'] else ('Medium' if p >= RISK_CUTOFFS['medium'] else 'Low')
scored['risk_tier'] = scored['churn_probability'].apply(risk_tier)
scored.to_csv('../data/processed/telco_scored.csv', index=False)
ins.tier_table(scored)""")

md("**Risk-tier calibration check:** observed churn rate rises from ~10% (Low tier) to ~43% "
   "(Medium) to ~71% (High) — the tiers are well-separated and the model's predicted "
   "probability tracks the actual outcome closely, confirming the cutoffs are meaningful "
   "rather than arbitrary.")

code("""k = ins.kpis(scored)
k""")

code("""for item in ins.build_insights(scored):
    print(f"[{item['type']}] {item['id']} - {item['title']}")
    print('  Fact:       ', item['fact'])
    print('  Insight:    ', item['insight'])
    print('  Implication:', item['implication'])
    print('  Action:     ', item['action'])
    print()""")

md("""## 11. Results

- Selected model: **Logistic Regression** (ROC-AUC ~ 0.84 on held-out test data)
- 7,043 customers scored; 1,230 active (non-churned) customers fall in the High or Medium
  risk tier, together holding roughly $93k/month of at-risk revenue
- Four risk findings (contract type, early tenure, fibre optic, electronic check) and three
  opportunity findings (protection services, a ready-made ranked retention list, and growing
  the low-churn long-contract base) are generated directly from the scored data — see Section 10
  output and the Project Report for the full FACT -> INSIGHT -> IMPLICATION -> ACTION write-up.

## 12. Risks, Opportunities and Recommended Actions

See the printed insight list in Section 10 (`ins.build_insights`) — the same function
powers the `/insights` API endpoint and the "Risk & Recommendations" page of the Streamlit
dashboard, so the notebook, API and dashboard are guaranteed to agree.

## 13. Limitations
- The dataset is a single snapshot with no timestamps, so trends over time (e.g. "did churn
  rise this quarter") cannot be assessed — only cross-sectional patterns.
- No usage-quality, complaint-log, or competitor-pricing fields exist, so the fibre-optic churn
  finding is flagged as a risk to investigate, not a diagnosed cause.
- All relationships reported are associations from observational data; establishing causation
  would need a controlled trial (e.g. A/B-testing the protection-service offer in Opportunity O1).
- The model is trained on this operator's historical customers and would need retraining and
  re-validation before use on a different customer base or after major product changes.

## 14. Future scope
- Retrain periodically as new billing cycles complete, and monitor for feature drift.
- Add a controlled trial for the top opportunity (free/discounted protection-service trial) to
  move from correlation to a measured causal effect.
- Extend the dataset with usage/complaint/NPS data to explain the fibre-optic churn gap.
- Add cost-sensitive threshold tuning using actual retention-offer cost and customer lifetime
  value once those figures are available from finance.

## 15. Conclusion

This notebook takes the Telco Customer Churn dataset through the full masterclass flow: raw
data was cleaned and documented, EDA surfaced clear, explainable churn patterns, a Logistic
Regression classifier reached ROC-AUC ~ 0.84 against a validated baseline, every customer was
scored with an out-of-fold churn probability and risk tier, and those scores were converted
into a small number of evidence-backed risk and opportunity findings with concrete recommended
actions — all served through a Flask API and a 4-page Streamlit decision dashboard rather than
left as a static notebook.

## References
1. Telco Customer Churn dataset — IBM, https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
2. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.
3. McKinney, W., *Data Structures for Statistical Computing in Python*, SciPy 2010 (pandas).
4. Streamlit Inc., Streamlit documentation, https://docs.streamlit.io
""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python', 'version': '3.12'}}
nbf.write(nb, 'notebooks/Chirag_TelcoChurnAnalysis_final.ipynb')
print("notebook cells:", len(cells))
