const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  Table,
  TableRow,
  TableCell,
  WidthType,
  ShadingType,
  ImageRun,
  PageBreak,
  AlignmentType,
  BorderStyle,
  Header,
  Footer,
  PageNumber,
  TableOfContents,
  ExternalHyperlink,
  VerticalAlign,
} = require("docx");

const ROOT = __dirname;
const IMG = path.join(ROOT, "report_images");
const model = JSON.parse(
  fs.readFileSync(path.join(ROOT, "model", "model_metrics.json")),
);

function imgDims(file, maxWidth) {
  const sizeOf = require("image-size");
  const buf = fs.readFileSync(path.join(IMG, file));
  const dim = sizeOf.imageSize(new Uint8Array(buf));
  const w = Math.min(maxWidth, dim.width);
  const h = (dim.height / dim.width) * w;
  return { width: w, height: h };
}

function figure(file, caption, maxWidth = 560) {
  const { width, height } = imgDims(file, maxWidth);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      children: [
        new ImageRun({
          data: fs.readFileSync(path.join(IMG, file)),
          transformation: { width, height },
          type: "png",
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [
        new TextRun({
          text: caption,
          italics: true,
          size: 20,
          color: "555555",
        }),
      ],
    }),
  ];
}

function h1(text) {
  return new Paragraph({
    text,
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
  });
}
function h2(text) {
  return new Paragraph({
    text,
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
  });
}
function h3(text) {
  return new Paragraph({
    text,
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text, ...opts })],
  });
}
function pRuns(runs) {
  return new Paragraph({ spacing: { after: 160 }, children: runs });
}
function bullet(text, level = 0) {
  return new Paragraph({ text, bullet: { level }, spacing: { after: 80 } });
}
function bold(text) {
  return new TextRun({ text, bold: true });
}

function cell(text, opts = {}) {
  const { header = false, width, shade } = opts;
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade
      ? { type: ShadingType.CLEAR, color: "auto", fill: shade }
      : header
        ? { type: ShadingType.CLEAR, color: "auto", fill: "1F3864" }
        : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [
      new Paragraph({
        children: [
          new TextRun({
            text: String(text),
            bold: header,
            color: header ? "FFFFFF" : "000000",
            size: header ? 20 : 20,
          }),
        ],
      }),
    ],
  });
}

function table(headerRow, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headerRow.map((t, i) =>
          cell(t, { header: true, width: colWidths[i] }),
        ),
      }),
      ...rows.map(
        (r) =>
          new TableRow({
            children: r.map((t, i) => cell(t, { width: colWidths[i] })),
          }),
      ),
    ],
  });
}

const M = model.test_metrics_default_threshold;
const T = model.test_metrics_tuned_threshold;
const pct = (x) => `${(x * 100).toFixed(1)}%`;

// ---------------------------------------------------------------- TITLE ----
const titlePage = [
  new Paragraph({
    spacing: { before: 2400 },
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({ text: "Telecom Customer Churn", bold: true, size: 56 }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [
      new TextRun({
        text: "A Decision Support Dashboard",
        size: 32,
        color: "444444",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 800 },
    children: [
      new TextRun({
        text: "IBM SkillsBuild Data Analytics with AI Academic Internship",
        size: 24,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 800 },
    children: [new TextRun({ text: "BharatCares x AICTE", size: 24 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1200 },
    children: [
      new TextRun({ text: "Prepared by: Chirag", size: 22, bold: true }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({
        text: "Final-year B.E., Artificial Intelligence and Data Science",
        size: 20,
        color: "555555",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [
      new TextRun({
        text: "Srinivas Institute of Technology (VTU), Mangaluru",
        size: 20,
        color: "555555",
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1200 },
    children: [
      new TextRun({
        text: new Date().toISOString().slice(0, 10),
        size: 20,
        color: "777777",
      }),
    ],
  }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ------------------------------------------------------------ ABSTRACT -----
const abstract = [
  h1("Abstract"),
  p(
    "Telecom operators typically lose roughly a quarter of their customer base every reporting " +
      "period, yet most support and marketing teams have no systematic way to identify which " +
      "currently active customers are likely to leave next. This project builds a complete, " +
      "working decision-support system on the IBM Telco Customer Churn dataset (7,043 customers, " +
      "21 fields), following the raw data -> clean data -> EDA -> insights -> prediction -> " +
      "dashboard -> decision flow. The data was cleaned and documented, exploratory analysis " +
      `surfaced clear churn patterns around contract type, tenure and payment method, and a ` +
      `Logistic Regression classifier was selected after comparison against Random Forest, ` +
      `Gradient Boosting and a majority-class baseline (test ROC-AUC ${M.roc_auc.toFixed(3)}). ` +
      `Every customer was then scored with an out-of-fold churn probability and risk tier, and ` +
      `those scores were converted into seven evidence-backed Risk and Opportunity findings, each ` +
      `carrying a concrete recommended action and an addressable customer count and revenue ` +
      `figure. The system is served through a Flask REST API and a four-page Streamlit decision ` +
      `dashboard, so the output is an actionable, ranked retention list rather than a static ` +
      `report.`,
  ),
];

// ---------------------------------------------------------- INTRODUCTION ---
const introduction = [
  h1("1. Introduction"),
  p(
    "Customer churn — a customer discontinuing service — is one of the most direct threats to " +
      "recurring revenue in subscription-based businesses such as telecom operators. Acquiring a " +
      "new customer is well documented to cost substantially more than retaining an existing one, " +
      "which makes early, targeted retention action valuable if the operator can identify who is " +
      "at risk before they leave. This project addresses that need using the standard analytics-" +
      "to-decision pipeline taught in the IBM SkillsBuild Data Analytics with AI masterclass: " +
      "clean data feeds exploratory analysis, exploratory analysis feeds a predictive model, the " +
      "model's output feeds a small set of business insights, and the insights are delivered " +
      "through a dashboard designed for a decision-maker rather than a data scientist.",
  ),
];

// ------------------------------------------------------- PROBLEM STATEMENT -
const problemStatement = [
  h1("2. Problem Statement"),
  h2("What problem exists?"),
  p(
    "A telecom operator has no systematic way to know which of its currently active customers " +
      "are likely to churn. Retention budget is therefore either spread across the entire " +
      "customer base (expensive and inefficient) or not deployed at all (customers leave without " +
      "any intervention).",
  ),
  h2("Why is it important?"),
  p(
    "In this dataset, 26.5% of customers have already churned, representing an estimated " +
      "$139,131 of monthly recurring revenue already lost. A further 1,230 currently active " +
      "customers are scored High or Medium risk, holding roughly $92,929 of monthly revenue " +
      "that remains exposed.",
  ),
  h2("What difficulties exist with traditional/manual approaches?"),
  p(
    "Manually reviewing thousands of customer billing and service records to judge who might " +
      "leave next is not feasible at scale, and ad-hoc judgement does not produce a ranked, " +
      "reproducible list that a retention team can work through, nor does it quantify how much " +
      "revenue is at stake.",
  ),
  h2("What data is available?"),
  p(
    "The Telco Customer Churn dataset provides 7,043 customer records with demographic, " +
      "account and service-usage fields, plus a historical Churn outcome — sufficient to train a " +
      "supervised classification model and to segment customers by known churn drivers.",
  ),
  h2("How can data analytics help?"),
  p(
    "Exploratory data analysis quantifies which customer segments churn most (by contract, " +
      "tenure, internet service and payment method), turning anecdote into a ranked, evidenced " +
      "set of risk factors.",
  ),
  h2("How can AI/ML help?"),
  p(
    "A supervised classification model combines all available fields simultaneously to produce " +
      "a single, ranked churn-probability score per customer — something manual segment analysis " +
      "cannot do — enabling a prioritised retention list rather than broad, unfocused segments.",
  ),
  h2("What does the proposed system provide?"),
  p(
    "A trained and evaluated churn model, a scored dataset covering all 7,043 customers, a " +
      "REST API for on-demand scoring of new customers, and a four-page decision dashboard that " +
      "presents KPIs, trends, drivers, risks, opportunities and a downloadable prioritised " +
      "retention list.",
  ),
];

// -------------------------------------------------------------- OBJECTIVES -
const objectives = [
  h1("3. Objectives"),
  bullet(
    "Understand and document the dataset: fields, size, target balance and data quality",
  ),
  bullet("Clean the data and justify every cleaning decision taken"),
  bullet(
    "Perform exploratory data analysis to identify churn patterns and drivers",
  ),
  bullet(
    "Engineer features that summarise service bundling and payment behaviour",
  ),
  bullet(
    "Train and compare multiple classification models against a validated baseline",
  ),
  bullet(
    "Evaluate the selected model with metrics appropriate for imbalanced classification",
  ),
  bullet(
    "Create meaningful, purposeful visualisations for each stage of the analysis",
  ),
  bullet(
    "Build an interactive dashboard organised around decisions, not just data",
  ),
  bullet("Generate insights and classify each as a Risk or an Opportunity"),
  bullet("Provide evidence-based, quantified recommendations for each insight"),
];

// ------------------------------------------------------------------- SCOPE -
const scope = [
  h1("4. Scope"),
  p(
    "In scope: a batch-trained binary churn classifier; a rule-based insight engine computed " +
      "directly from the scored dataset (not free-text or LLM-generated); a Flask REST API; and " +
      "a four-page Streamlit decision dashboard.",
  ),
  p(
    "Out of scope: real-time streaming inference, an LLM-generated narrative layer, and causal " +
      "inference. Every relationship reported in this project is an association observed in " +
      "historical data, not a proven cause — this distinction is maintained throughout the " +
      "insights in Section 10.",
  ),
];

// --------------------------------------------------------- DATASET DESC ----
const datasetDescRows = [
  ["Dataset name", "Telco Customer Churn"],
  [
    "Source",
    "IBM Cloud Pak for Data sample datasets (IBM telco-customer-churn-on-icp4d GitHub repository)",
  ],
  [
    "Dataset link",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
  ],
  ["Format", "CSV"],
  ["Rows", "7,043 customers"],
  ["Columns", "21 (20 features + target Churn)"],
  ["Target variable", "Churn (Yes/No) - 26.5% positive class"],
  [
    "Missing values",
    "0 explicit nulls; 11 blank TotalCharges strings, resolved during cleaning",
  ],
  ["Duplicate values", "0 duplicate rows or customerID values"],
];
const datasetDescription = [
  h1("5. Dataset Description"),
  table(["Field", "Value"], datasetDescRows, [2600, 6800]),
  h3("Feature groups"),
  bullet("Demographics: gender, SeniorCitizen, Partner, Dependents"),
  bullet(
    "Account: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges",
  ),
  bullet(
    "Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, " +
      "DeviceProtection, TechSupport, StreamingTV, StreamingMovies",
  ),
];

// ------------------------------------------------------------- CLEANING ----
const cleaningRows = model.cleaning_log.map((l) => [
  l.step,
  String(l.found),
  l.action,
  l.reason,
]);
const dataCleaning = [
  h1("6. Data Cleaning"),
  p(
    "Each step below is applied programmatically by src/preprocessing.clean_data() and its " +
      "log is captured automatically - the table is not hand-typed.",
  ),
  table(
    ["Step", "Found", "Action taken", "Reason"],
    cleaningRows,
    [2000, 900, 2600, 3900],
  ),
  h3("Key decisions explained"),
  bullet(
    "TotalCharges is stored as text with 11 blank values; every one of the 11 belongs to a " +
      "customer with tenure = 0 (not yet billed), so they were set to 0.0 rather than dropped.",
  ),
  bullet(
    "\u201CNo internet service\u201D / \u201CNo phone service\u201D categories were folded into " +
      "\u201CNo\u201D because that information is already fully captured by InternetService / " +
      "PhoneService - keeping both would double-encode the same fact.",
  ),
  bullet(
    "SeniorCitizen was recoded from 0/1 to No/Yes so every binary field shares one encoding.",
  ),
  bullet(
    "No rows were removed: 7,043 in, 7,043 out. The IQR outlier check on tenure, monthly " +
      "and total charges found zero statistical outliers - every value is a plausible duration or " +
      "bill amount.",
  ),
];

// ------------------------------------------------------------------- EDA ---
const eda = [
  h1("7. Exploratory Data Analysis"),
  p(
    "Nine charts were produced to understand distributions, trends and relationships in the " +
      "cleaned data. Each is followed by an Observation -> Insight -> Meaning interpretation.",
  ),
  ...figure("01_churn_balance.png", "Figure 1. Churn class distribution.", 380),
  p(
    "Observation: 26.5% of customers have churned - a meaningful minority class. Insight: the " +
      "dataset is moderately imbalanced, so plain accuracy would be a misleading metric. Meaning: " +
      "ROC-AUC, precision/recall and F1 are used for evaluation instead (Section 12).",
  ),
  ...figure(
    "02_churn_by_contract.png",
    "Figure 2. Churn rate by contract type.",
  ),
  ...figure("03_churn_by_tenure.png", "Figure 3. Churn rate by tenure group."),
  p(
    "Observation: churn falls sharply as contract length and tenure increase - from 42.7% " +
      "month-to-month down to 2.8% on two-year contracts, and from 47.4% in the first year down " +
      "to 9.5% after four years. Insight: commitment (contract length and tenure) is one of the " +
      "strongest observable separators of leavers and stayers. Meaning: retention effort aimed at " +
      "new, uncommitted customers should have the highest yield (quantified in Section 10).",
  ),
  ...figure(
    "04_tenure_distribution.png",
    "Figure 4. Tenure distribution by churn outcome.",
    420,
  ),
  ...figure(
    "05_monthly_charges_distribution.png",
    "Figure 5. Monthly charges distribution by churn outcome.",
    420,
  ),
  ...figure(
    "06_churn_by_internet.png",
    "Figure 6. Churn rate by internet service type.",
  ),
  p(
    "Observation: fibre-optic customers churn more than twice as often as DSL customers " +
      "(41.9% vs 19.0%) despite paying considerably more per month on average ($91.50 vs " +
      "$58.10). Insight: fibre is the single largest source of lost monthly revenue among " +
      "internet products. Meaning: the dataset has no service-quality or competitor-pricing " +
      "field, so the cause cannot be determined from this data alone - it is flagged as a risk to " +
      "investigate, not a proven cause.",
  ),
  ...figure(
    "07_churn_by_payment.png",
    "Figure 7. Churn rate by payment method.",
    500,
  ),
  p(
    "Observation: electronic-check payers churn at 45.3% versus 16.0% for customers paying by " +
      "automatic bank transfer or credit card. Insight: manual, per-cycle payment is associated " +
      "with much weaker retention. Meaning: switching customers to autopay is a cheap, testable " +
      "lever (Section 10, Risk R4).",
  ),
  ...figure(
    "08_churn_by_protection.png",
    "Figure 8. Churn rate by number of protection/support services (internet customers).",
  ),
  p(
    "Observation: internet customers with none of the four protection/support services churn " +
      "at 56.7%, versus 16.8% for those with two or more. Insight: taking these add-ons goes with " +
      "much stronger retention. Meaning: this is an upsell opportunity worth trialling with a " +
      "control group (Section 10, Opportunity O1).",
  ),
  ...figure(
    "09_correlation_heatmap.png",
    "Figure 9. Correlation heatmap of numeric and encoded features.",
    480,
  ),
  p(
    "Observation: Churn correlates most strongly with Contract (negative - longer contract, " +
      "less churn), tenure (negative) and MonthlyCharges (positive). tenure and TotalCharges are " +
      "highly correlated with each other, which is addressed in modelling by examining feature " +
      "importance rather than relying on either field alone.",
  ),
];

// ---------------------------------------------------------------- INSIGHTS -
const insightsSection = [
  h1("8. Insight Generation"),
  p(
    "Beyond the EDA charts above, the cleaned and scored dataset was queried directly for " +
      "seven Risk and Opportunity findings (full detail with FACT -> INSIGHT -> IMPLICATION -> " +
      "ACTION in Section 15). Every figure quoted is computed by src/insights.py from real data - " +
      "none are invented or estimated by an AI model.",
  ),
  h3("Example: Data Evidence -> Observation -> Insight -> Implication"),
  bullet(
    "Data evidence: month-to-month customers churn at 42.7%, one-year at 11.3%, two-year at 2.8%",
  ),
  bullet(
    "Observation: churn falls monotonically and steeply as contract length increases",
  ),
  bullet(
    "Insight: contract length is one of the strongest observable separators of leavers and stayers",
  ),
  bullet(
    "Implication: this may represent a retention lever worth testing - moving suitable " +
      "month-to-month customers onto longer terms",
  ),
];

// -------------------------------------------------------- ML METHODOLOGY ---
const mlMethodology = [
  h1("9. Machine Learning Methodology"),
  p("Problem type: binary classification (Churn: Yes/No)."),
  h3("Pipeline"),
  bullet(
    "Train/test split: 80% / 20%, stratified by target, random_state = 42 " +
      `(${model.train_rows.toLocaleString()} train / ${model.test_rows.toLocaleString()} test rows)`,
  ),
  bullet(
    "Preprocessing: StandardScaler on numeric features, OneHotEncoder on categorical features, inside a scikit-learn Pipeline",
  ),
  bullet(
    "Candidates compared: Logistic Regression, Random Forest, Gradient Boosting, each against a DummyClassifier (majority-class) baseline",
  ),
  bullet(
    "Validation: 5-fold stratified cross-validation on the training split, scored on ROC-AUC and PR-AUC",
  ),
  bullet(`Model selection rule: ${model.selection_rule}`),
  bullet(
    `Decision threshold rule: ${model.threshold_rule} (selected threshold: ${model.decision_threshold})`,
  ),
  bullet(
    "Every customer is finally scored using out-of-fold cross-validated predictions, so no customer is ever scored by a model that was trained on that same customer",
  ),
];

// -------------------------------------------------------------- TRAINING ---
const comparisonRows = Object.entries(model.comparison).map(([name, r]) => [
  name,
  r.cv_roc_auc === 0.5 && name.startsWith("Baseline")
    ? "0.500 (by definition)"
    : r.cv_roc_auc.toFixed(4),
  r.test.roc_auc.toFixed(4),
  r.test.f1.toFixed(4),
]);
const modelTraining = [
  h1("10. Model Training"),
  p(
    "Four models were trained and compared: a majority-class baseline and three classifiers, " +
      "each wrapped in the same preprocessing pipeline for a fair comparison.",
  ),
  table(
    ["Model", "5-fold CV ROC-AUC", "Test ROC-AUC", "Test F1 (threshold 0.5)"],
    comparisonRows,
    [3000, 2300, 2000, 2100],
  ),
  p(
    `Logistic Regression, Random Forest and Gradient Boosting all land within about 0.002 AUC ` +
      `of each other. Logistic Regression was selected as the simplest model within 0.005 AUC of ` +
      `the best performer, because its coefficients allow every individual prediction to be ` +
      `explained in plain terms - used both in Section 11 and by the "Predict a Customer" page ` +
      `of the dashboard.`,
  ),
];

// ------------------------------------------------------------- EVALUATION --
const evalRows = [
  ["Accuracy", M.accuracy.toFixed(3), T.accuracy.toFixed(3)],
  ["Precision", M.precision.toFixed(3), T.precision.toFixed(3)],
  ["Recall", M.recall.toFixed(3), T.recall.toFixed(3)],
  ["F1-score", M.f1.toFixed(3), T.f1.toFixed(3)],
  ["ROC-AUC", M.roc_auc.toFixed(3), T.roc_auc.toFixed(3)],
  ["PR-AUC", M.pr_auc.toFixed(3), T.pr_auc.toFixed(3)],
];
const modelEvaluation = [
  h1("11. Model Evaluation"),
  p(
    "Metrics appropriate for imbalanced binary classification are reported at both the default " +
      "0.5 threshold and the tuned threshold (0.32), on the held-out test set only.",
  ),
  table(
    ["Metric", "Threshold = 0.50", `Threshold = ${model.decision_threshold}`],
    evalRows,
    [3200, 3200, 3200],
  ),
  ...figure(
    "11_roc_curves.png",
    "Figure 10. ROC curves - model comparison on the test set.",
    380,
  ),
  ...figure(
    "12_confusion_matrix.png",
    "Figure 11. Confusion matrix at the tuned threshold.",
    340,
  ),
  p(
    `At the tuned threshold, recall rises to ${pct(T.recall)} (catching most churners) at the ` +
      `cost of precision (${pct(T.precision)}) - an intentional trade-off, since the retention ` +
      `action (an email or offer) is cheap relative to losing a customer outright.`,
  ),
  h3("Feature importance"),
  ...figure(
    "10_feature_importance.png",
    "Figure 12. Top 10 permutation feature importances.",
    420,
  ),
  p(
    "tenure dominates, followed by InternetService and Contract - consistent with the EDA " +
      "findings in Section 7. This agreement between the EDA patterns and the model's own " +
      "feature importance is what justifies treating contract type, tenure and internet service " +
      "as the primary churn drivers.",
  ),
  h3("Model limitations"),
  bullet(
    "The model uses only the fields present in this dataset - no usage minutes, complaint history, or competitor pricing",
  ),
  bullet(
    "It is a correlational model: features found \u201Cimportant\u201D are associated with churn, not proven causes of it",
  ),
  bullet(
    "Trained on this operator's historical customers; would need retraining before use on a different customer base",
  ),
];

// ---------------------------------------------------------- DASHBOARD DESIGN
const dashboardDesign = [
  h1("12. Dashboard Design"),
  p(
    "The dashboard follows the masterclass's five-level structure (KPIs -> Trends -> Drivers -> " +
      "Risk -> Action) across four pages, so it never becomes an undirected data dump.",
  ),
  h3("Top 3 KPIs (above the fold)"),
  bullet(
    `Churn rate - ${pct(0.2653698707936959)} of customers have churned - answers \u201Cis the business healthy?\u201D in one glance`,
  ),
  bullet(
    "Monthly revenue lost to churn - quantifies the cost of the problem in dollars, not just customer count",
  ),
  bullet(
    "Active customers at risk (High + Medium tier) - the actionable number: who to act on now, and how much revenue is exposed",
  ),
  h3("Page structure"),
  bullet(
    "Page 1 - Executive Overview: the three KPIs above, supporting KPIs, churn split and risk-tier distribution",
  ),
  bullet(
    "Page 2 - Trend & Driver Analysis: churn trend across the customer lifecycle, a selectable driver breakdown, and model feature importances",
  ),
  bullet(
    "Page 3 - Predict a Customer: score any customer profile on demand, with a plain-language explanation of what pushed the score up or down",
  ),
  bullet(
    "Page 4 - Risk & Recommendations: the seven Risk/Opportunity findings and a downloadable, ranked retention list",
  ),
  p(
    "Every chart on every page answers one of the six framing questions from the masterclass " +
      "(what is happening / how is it changing / why / what could go wrong / where is the " +
      "opportunity / what should we do) - no chart was included without a specific question it answers.",
  ),
];

// ------------------------------------------------------- SYSTEM ARCHITECTURE
const systemArchitecture = [
  h1("13. System Architecture"),
  ...figure(
    "00_architecture.png",
    "Figure 13. End-to-end system architecture.",
    560,
  ),
  p(
    "Data flows one direction only: the raw CSV is cleaned and explored, features are " +
      "engineered, three candidate models are trained and evaluated, the winning model and its " +
      "metrics are saved to disk, every customer is scored, and the Flask API and Streamlit " +
      "dashboard both read from those same saved, versioned artifacts - so the notebook, the API " +
      "and the dashboard can never disagree with each other.",
  ),
];

// -------------------------------------------------------------- IMPLEMENTATION
const implementation = [
  h1("14. Implementation"),
  h3("Backend - Flask API"),
  p("Six endpoints, each with a single clear purpose:"),
  bullet("GET /health - service and model-load status"),
  bullet(
    "GET /model_info - selected model, decision threshold, evaluation metrics, input schema",
  ),
  bullet(
    "GET /dataset - row/column counts, dtypes, missing-value count, a data sample",
  ),
  bullet("GET /kpis - the executive-overview KPI values"),
  bullet(
    "GET /insights - the seven Risk/Opportunity findings, computed live from the scored data",
  ),
  bullet(
    "POST /predict - scores one customer; validates every field, rejects impossible combinations (e.g. streaming services set to Yes with no internet service), and returns a plain-language explanation",
  ),
  h3("Frontend - Streamlit dashboard"),
  p(
    "Four pages as described in Section 12, reading directly from the saved model and scored " +
      "dataset (so it works even if the Flask API is not running). Built and tested with Plotly " +
      "charts, a live prediction form, and a downloadable prioritised retention CSV.",
  ),
  h3("Testing performed"),
  bullet(
    "Automated smoke tests on every Flask endpoint (valid input, missing fields, invalid categories, impossible combinations, malformed JSON, unknown routes) - all return the expected status codes",
  ),
  bullet(
    "A full clean-environment re-run of src/train_model.py reproduces identical metrics (random_state = 42 throughout)",
  ),
  bullet(
    "The Streamlit dashboard was launched and screenshotted headlessly across all four pages, including submitting the prediction form, to confirm it renders without exceptions",
  ),
];

// ---------------------------------------------------------------- RESULTS --
const results = [
  h1("15. Results"),
  p(
    `The selected Logistic Regression model reached a test ROC-AUC of ${M.roc_auc.toFixed(3)} ` +
      `(F1 ${T.f1.toFixed(3)} at the tuned threshold), well above the 0.5 AUC of the majority-` +
      `class baseline. All 7,043 customers were scored and tiered; 1,230 currently active ` +
      `customers fall in the High or Medium risk tier, together holding approximately $92,929 of ` +
      `monthly recurring revenue.`,
  ),
  p("Seven findings were generated directly from the scored data:"),
];

function insightBlock(
  id,
  type,
  title,
  fact,
  insight,
  implication,
  action,
  addressable,
) {
  const out = [
    pRuns([bold(`${type} ${id}. `), new TextRun({ text: title, bold: true })]),
    pRuns([bold("Fact. "), new TextRun(fact)]),
    pRuns([bold("Insight. "), new TextRun(insight)]),
    pRuns([bold("Implication. "), new TextRun(implication)]),
    pRuns([bold("Recommended action. "), new TextRun(action)]),
  ];
  if (addressable)
    out.push(
      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({
            text: addressable,
            italics: true,
            size: 20,
            color: "555555",
          }),
        ],
      }),
    );
  return out;
}

const insightData = [
  [
    "R1",
    "Risk",
    "Month-to-month contracts hold most of the churn",
    "Month-to-month customers churn at 42.7% versus 11.3% on one-year and 2.8% on two-year contracts.",
    "They are 55.0% of customers but account for 88.6% of all churners.",
    "The gap between the shortest and longest contract is 39.9 percentage points, and Contract is among the top three drivers in the model (association, not proof of cause).",
    "Offer contract-upgrade incentives to the 1,170 active month-to-month customers scored High/Medium risk ($87,309 monthly revenue).",
    "Addressable now: 1,170 active customers, $87,309/month revenue.",
  ],
  [
    "R2",
    "Risk",
    "The first year is the danger zone",
    "Customers in their first 12 months churn at 47.4%, versus 9.5% for customers with 49-72 months of tenure.",
    "55.5% of all churners left within their first 12 months; the average churned customer had 18 months of tenure vs 38 for active ones.",
    "Most churn happens early, so early-lifecycle retention addresses the largest share of churners.",
    "Launch a 12-month onboarding and check-in programme; start with the 589 active customers under 12 months tenure scored High/Medium risk ($37,237 monthly revenue).",
    "Addressable now: 589 active customers, $37,237/month revenue.",
  ],
  [
    "R3",
    "Risk",
    "Fibre optic customers churn 2.2x as often as DSL customers",
    "Fibre optic churn is 41.9% versus 19.0% for DSL, and fibre customers pay $92 per month on average versus $58.",
    "Higher-priced fibre customers leave far more often. The data has no service-quality or competitor field, so the reason cannot be determined from it.",
    "Fibre customers account for 82.2% of all monthly revenue lost to churn ($114,300), the largest share of any internet product.",
    "Run a short exit/at-risk survey on fibre customers to find whether price, reliability or competition is the cause; 844 active fibre customers are High/Medium risk ($74,377 monthly revenue).",
    "Addressable now: 844 active customers, $74,377/month revenue.",
  ],
  [
    "R4",
    "Risk",
    "Electronic-check payers churn far more than automatic payers",
    "Electronic-check customers churn at 45.3% versus 16.0% for customers paying by automatic bank transfer or credit card.",
    "Manual, per-cycle payment is associated with much weaker retention than automatic payment.",
    "The data cannot explain why, but the gap is large enough to justify testing a payment incentive.",
    "Offer a small incentive to switch to automatic payment, starting with the 685 active electronic-check customers scored High/Medium risk ($53,692 monthly revenue).",
    "Addressable now: 685 active customers, $53,692/month revenue.",
  ],
  [
    "O1",
    "Opportunity",
    "Protection and support services go with retention",
    "Internet customers with none of the four protection/support services (Online Security, Online Backup, Device Protection, Tech Support) churn at 56.7%, versus 16.8% for those with two or more.",
    "1,267 internet customers have no protection service. Customers who take these services stay much more often.",
    "Selling these add-ons is an opportunity to deepen the relationship. The data shows association; an A/B trial is needed to prove the add-ons cause retention.",
    "Trial a free/discounted Tech Support or Online Security period for the 433 active no-protection internet customers scored High/Medium risk ($28,467 monthly revenue), and compare against a control group.",
    "Addressable now: 433 active customers, $28,467/month revenue.",
  ],
  [
    "O2",
    "Opportunity",
    "A ranked retention list is available today",
    "The model separates customers cleanly: observed churn is 10.0% in the Low tier, 42.8% in Medium and 71.2% in High (out-of-fold scores).",
    "1,230 customers who have NOT yet left are scored High or Medium risk. They hold $92,929 of monthly revenue (29.3% of active revenue).",
    "Retention budget can be focused on a small, high-yield group instead of all customers.",
    "Contact the 294 High-risk active customers first ($24,180 monthly revenue), then the Medium tier; re-score monthly and track the save rate.",
    "Addressable now: 1,230 active customers, $92,929/month revenue.",
  ],
  [
    "O3",
    "Opportunity",
    "Long-contract customers are a stable revenue base",
    "Two-year customers churn at only 2.8% and represent 24.1% of customers and 22.6% of monthly revenue.",
    "Long contracts are associated with very low churn; growing this group would stabilise revenue.",
    "Moving suitable month-to-month customers onto longer terms is a lever worth testing.",
    "Make one-year/two-year plans the default recommendation at sign-up and at renewal touch-points.",
    null,
  ],
];

let insightParas = [];
for (const row of insightData)
  insightParas = insightParas.concat(insightBlock(...row));

// -------------------------------------------------------- RISKS/OPPS/ACTIONS
const risksOppsActions = [
  h1("16. Risks, Opportunities and Recommended Actions"),
  p(
    "Full detail for all seven findings (also served live by the /insights API endpoint and " +
      "the dashboard's Risk & Recommendations page, so all three are always in agreement):",
  ),
  ...insightParas,
];

// ------------------------------------------------------------- LIMITATIONS -
const limitations = [
  h1("17. Limitations"),
  bullet(
    "Single-snapshot dataset with no timestamps - trends over time (e.g. is churn rising this quarter) cannot be assessed, only cross-sectional patterns",
  ),
  bullet(
    "No usage-quality, complaint-log, or competitor-pricing fields exist, so the fibre-optic churn finding (R3) is flagged as a risk to investigate, not a diagnosed cause",
  ),
  bullet(
    "All relationships reported are associations from observational data; establishing causation would require a controlled trial (e.g. A/B-testing the protection-service offer in O1)",
  ),
  bullet(
    "The model is trained on this operator's historical customers and would need retraining and re-validation before use on a different customer base or after major product changes",
  ),
];

// ------------------------------------------------------------- FUTURE SCOPE
const futureScope = [
  h1("18. Future Scope"),
  bullet(
    "Retrain periodically as new billing cycles complete, and monitor for feature drift",
  ),
  bullet(
    "Run a controlled A/B trial on the top opportunity (protection-service trial) to move from correlation to a measured causal effect",
  ),
  bullet(
    "Extend the dataset with usage, complaint or NPS data to explain the fibre-optic churn gap",
  ),
  bullet(
    "Add cost-sensitive threshold tuning using actual retention-offer cost and customer lifetime value once available from finance",
  ),
];

// -------------------------------------------------------------- CONCLUSION -
const conclusion = [
  h1("19. Conclusion"),
  p(
    "This project takes the Telco Customer Churn dataset through the complete masterclass flow: " +
      "raw data was cleaned and documented, exploratory analysis surfaced clear and explainable " +
      "churn patterns, a Logistic Regression classifier was trained and evaluated against a " +
      "validated baseline and two stronger candidate models, every customer was scored with an " +
      "out-of-fold churn probability and risk tier, and those scores were converted into seven " +
      "evidence-backed risk and opportunity findings with concrete, quantified recommended " +
      "actions. The result is served through a working Flask API and a four-page Streamlit " +
      "decision dashboard, so the final deliverable is an actionable, ranked retention list - not " +
      "a static report or an unused model file.",
  ),
];

// -------------------------------------------------------------- REFERENCES -
const references = [
  h1("20. References"),
  bullet(
    "Telco Customer Churn dataset - IBM, https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
  ),
  bullet(
    "Pedregosa et al., Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research 12, 2011",
  ),
  bullet(
    "McKinney, W., Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 2010 (pandas)",
  ),
  bullet("Streamlit Inc., Streamlit documentation, https://docs.streamlit.io"),
  bullet(
    "Flask (Pallets Projects), Flask documentation, https://flask.palletsprojects.com",
  ),
];

// ------------------------------------------------------------- APPENDIX ----
const appendix = [
  h1("21. Appendix - Dashboard Screenshots"),
  ...figure(
    "dash_1_overview.png",
    "Screenshot 1. Executive Overview page.",
    560,
  ),
  ...figure(
    "dash_2_trends.png",
    "Screenshot 2. Trend & Driver Analysis page.",
    560,
  ),
  ...figure(
    "dash_3b_predict_result.png",
    "Screenshot 3. Predict a Customer page, with a scored result.",
    560,
  ),
  ...figure(
    "dash_4_risk.png",
    "Screenshot 4. Risk & Recommendations page.",
    560,
  ),
];

// ------------------------------------------------------------------ BUILD --
const sections = [
  {
    properties: {
      page: { margin: { top: 1000, bottom: 1000, left: 1100, right: 1100 } },
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [
              new TextRun({
                text: "Telecom Customer Churn - Project Report",
                size: 16,
                color: "888888",
              }),
            ],
          }),
        ],
      }),
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", size: 18, color: "888888" }),
              new TextRun({
                children: [PageNumber.CURRENT],
                size: 18,
                color: "888888",
              }),
              new TextRun({ text: " of ", size: 18, color: "888888" }),
              new TextRun({
                children: [PageNumber.TOTAL_PAGES],
                size: 18,
                color: "888888",
              }),
            ],
          }),
        ],
      }),
    },
    children: [
      ...titlePage,
      ...abstract,
      ...introduction,
      ...problemStatement,
      ...objectives,
      ...scope,
      ...datasetDescription,
      ...dataCleaning,
      ...eda,
      ...insightsSection,
      ...mlMethodology,
      ...modelTraining,
      ...modelEvaluation,
      ...dashboardDesign,
      ...systemArchitecture,
      ...implementation,
      ...results,
      ...risksOppsActions,
      ...limitations,
      ...futureScope,
      ...conclusion,
      ...references,
      ...appendix,
    ],
  },
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { bold: true, size: 30, color: "1F3864" },
        paragraph: { spacing: { before: 360, after: 160 } },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { bold: true, size: 26, color: "2E5395" },
        paragraph: { spacing: { before: 280, after: 120 } },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { bold: true, size: 23, color: "444444" },
        paragraph: { spacing: { before: 200, after: 100 } },
      },
    ],
  },
  sections,
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(ROOT, "Chirag_ProjectReport.docx"), buf);
  console.log("wrote Chirag_ProjectReport.docx", buf.length, "bytes");
});
