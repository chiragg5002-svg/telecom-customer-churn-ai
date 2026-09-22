"""Streamlit decision dashboard for the Telco churn project.

Run from the project root:  streamlit run frontend/app.py
Reads directly from the processed data / saved model (no backend server
required), so it also works when the Flask API is not running.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import insights as ins  # noqa: E402
from src.inference import Predictor, ValidationError  # noqa: E402
from src.preprocessing import CATEGORY_VALUES, INPUT_CATEGORICAL, SCORED_PATH  # noqa: E402

st.set_page_config(page_title="Telco Churn - Decision Dashboard", layout="wide")

COLORS = {"Low": "#2A9D8F", "Medium": "#F4A261", "High": "#E63946",
         "No": "#2E86AB", "Yes": "#E63946"}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(SCORED_PATH)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


df = load_data()
k = ins.kpis(df)
insight_list = ins.build_insights(df)

st.title("Telecom customer churn - decision dashboard")
st.caption("IBM SkillsBuild Data Analytics with AI Academic Internship - BharatCares x AICTE")

page = st.sidebar.radio("Section", [
    "1. Executive Overview", "2. Trend & Driver Analysis",
    "3. Predict a Customer", "4. Risk & Recommendations",
])
st.sidebar.divider()
st.sidebar.caption(f"{k['customers']:,} customers | churn rate {k['churn_rate']*100:.1f}%")

# ============================================================ PAGE 1 ======
if page == "1. Executive Overview":
    st.header("Is the business healthy? — top 3 KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn rate", f"{k['churn_rate']*100:.1f}%",
             help=f"{k['churned_customers']:,} of {k['customers']:,} customers have churned.")
    c2.metric("Monthly revenue lost to churn", f"${k['mrr_lost']:,.0f}",
             f"{k['mrr_lost_pct']*100:.1f}% of total monthly revenue", delta_color="inverse")
    c3.metric("Active customers at risk (High+Medium)",
             f"{k['high_risk_active'] + k['medium_risk_active']:,}",
             f"${k['high_risk_active_mrr'] + k['medium_risk_active_mrr']:,.0f}/mo revenue exposed",
             delta_color="inverse")

    st.divider()
    st.subheader("Supporting KPIs")
    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Total customers", f"{k['customers']:,}")
    c5.metric("Avg monthly charge", f"${k['avg_monthly_charge']:.2f}")
    c6.metric("Avg tenure - churned", f"{k['avg_tenure_churned']:.0f} mo")
    c7.metric("Avg tenure - active", f"{k['avg_tenure_active']:.0f} mo")

    st.divider()
    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Churn split")
        vc = df["Churn"].map({0: "Retained", 1: "Churned"}).value_counts()
        fig = px.pie(values=vc.values, names=vc.index, hole=0.55,
                    color=vc.index, color_discrete_map={"Retained": COLORS["No"], "Churned": COLORS["Yes"]})
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        st.subheader("Customers by predicted risk tier")
        tt = ins.tier_table(df)
        fig = px.bar(tt, x="risk_tier", y="customers", color="risk_tier",
                    color_discrete_map=COLORS, category_orders={"risk_tier": ["Low", "Medium", "High"]},
                    text="customers")
        fig.update_layout(showlegend=False, yaxis_title="Customers", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================ PAGE 2 ======
elif page == "2. Trend & Driver Analysis":
    st.header("How is it changing, and why?")

    st.subheader("Trend: churn by customer lifecycle stage")
    order = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
    tg = df.groupby("tenure_group")["Churn"].mean().reindex(order).reset_index()
    tg["Churn"] *= 100
    fig = px.line(tg, x="tenure_group", y="Churn", markers=True, text=tg["Churn"].round(1))
    fig.update_traces(textposition="top center", line_color="#E63946")
    fig.update_layout(yaxis_title="Churn rate (%)", xaxis_title="Tenure group")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{k['churned_customers']:,} churners had {k['avg_tenure_churned']:.0f} months of tenure on "
              f"average, versus {k['avg_tenure_active']:.0f} for customers who stayed.")

    st.subheader("Key drivers of churn")
    driver = st.selectbox("Segment by", ["Contract", "InternetService", "PaymentMethod", "auto_pay",
                                         "n_protection_services", "SeniorCitizen", "Partner", "Dependents"])
    seg = ins.segment_table(df, driver).sort_values("churn_rate", ascending=False)
    seg_disp = seg.copy()
    seg_disp["churn_rate"] = (seg_disp["churn_rate"] * 100).round(1)
    fig = px.bar(seg_disp, x=driver, y="churn_rate", text="churn_rate", color="churn_rate",
                color_continuous_scale="Reds")
    fig.update_layout(yaxis_title="Churn rate (%)", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg.rename(columns={"customers": "Customers", "churned": "Churned",
                                     "churn_rate": "Churn rate", "avg_monthly": "Avg monthly $",
                                     "mrr_lost": "Monthly $ lost"}),
                use_container_width=True, hide_index=True)

    st.subheader("What the model actually leans on")
    try:
        imp = pd.read_csv("model/feature_importance.csv").head(10)
    except FileNotFoundError:
        imp = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "feature_importance.csv")).head(10)
    fig = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                error_x="std", color_discrete_sequence=["#E76F51"])
    fig.update_layout(xaxis_title="Permutation importance (drop in ROC-AUC)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================ PAGE 3 ======
elif page == "3. Predict a Customer":
    st.header("Score a customer")
    predictor = load_predictor()
    meta = predictor.meta
    st.caption(f"Model: {meta['selected_model']} | decision threshold: {meta['decision_threshold']}")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.number_input("Tenure (months)", 0, 120, 12)
            monthly = st.number_input("Monthly charges ($)", 0.0, 500.0, 70.0, step=0.5)
            senior = st.selectbox("Senior citizen", CATEGORY_VALUES["SeniorCitizen"])
            partner = st.selectbox("Partner", CATEGORY_VALUES["Partner"])
            dependents = st.selectbox("Dependents", CATEGORY_VALUES["Dependents"])
            contract = st.selectbox("Contract", CATEGORY_VALUES["Contract"])
        with c2:
            internet = st.selectbox("Internet service", CATEGORY_VALUES["InternetService"])
            phone = st.selectbox("Phone service", CATEGORY_VALUES["PhoneService"])
            multi = st.selectbox("Multiple lines", CATEGORY_VALUES["MultipleLines"])
            paperless = st.selectbox("Paperless billing", CATEGORY_VALUES["PaperlessBilling"])
            payment = st.selectbox("Payment method", CATEGORY_VALUES["PaymentMethod"])
        with c3:
            sec = st.selectbox("Online security", CATEGORY_VALUES["OnlineSecurity"])
            backup = st.selectbox("Online backup", CATEGORY_VALUES["OnlineBackup"])
            devprot = st.selectbox("Device protection", CATEGORY_VALUES["DeviceProtection"])
            tech = st.selectbox("Tech support", CATEGORY_VALUES["TechSupport"])
            tv = st.selectbox("Streaming TV", CATEGORY_VALUES["StreamingTV"])
            movies = st.selectbox("Streaming movies", CATEGORY_VALUES["StreamingMovies"])
        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        payload = {"tenure": tenure, "MonthlyCharges": monthly, "SeniorCitizen": senior,
                  "Partner": partner, "Dependents": dependents, "PhoneService": phone,
                  "MultipleLines": multi, "InternetService": internet, "OnlineSecurity": sec,
                  "OnlineBackup": backup, "DeviceProtection": devprot, "TechSupport": tech,
                  "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
                  "PaperlessBilling": paperless, "PaymentMethod": payment}
        try:
            result = predictor.predict(payload)
        except ValidationError as e:
            st.error("Invalid input: " + "; ".join(e.errors))
        else:
            for w in result["warnings"]:
                st.warning(w)
            c1, c2, c3 = st.columns(3)
            c1.metric("Churn probability", f"{result['churn_probability']*100:.1f}%")
            c2.metric("Risk tier", result["risk_tier"])
            c3.metric("Predicted to churn?", "Yes" if result["predicted_churn"] else "No")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=result["churn_probability"] * 100,
                            gauge={"axis": {"range": [0, 100]},
                                  "bar": {"color": COLORS[result["risk_tier"]]},
                                  "steps": [{"range": [0, 30], "color": "#E1F5EE"},
                                           {"range": [30, 60], "color": "#FAEEDA"},
                                           {"range": [60, 100], "color": "#FAECE7"}]},
                            title={"text": "Churn probability (%)"}))
            st.plotly_chart(fig, use_container_width=True)
            if result["explanation"]:
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown("**Pushes risk up**")
                    for it in result["explanation"]["increases_risk"]:
                        st.write(f"- {it['feature']} = {it['value']}")
                with e2:
                    st.markdown("**Pulls risk down**")
                    for it in result["explanation"]["decreases_risk"]:
                        st.write(f"- {it['feature']} = {it['value']}")
                st.caption(result["explanation"]["note"])

# ============================================================ PAGE 4 ======
else:
    st.header("Risk, opportunity and recommended actions")
    st.caption("Every item below is computed from the scored dataset - none are invented.")
    def esc(text: str) -> str:
        return text.replace("$", "\\$")  # Streamlit renders bare $..$ as LaTeX

    for item in insight_list:
        color = "#E63946" if item["type"] == "Risk" else "#2A9D8F"
        with st.container(border=True):
            st.markdown(f"<span style='color:{color};font-weight:600'>{item['type']}</span> &nbsp; "
                       f"**{item['title']}**", unsafe_allow_html=True)
            st.write(f"**Fact.** {esc(item['fact'])}")
            st.write(f"**Insight.** {esc(item['insight'])}")
            st.write(f"**Implication.** {esc(item['implication'])}")
            st.write(f"**Recommended action.** {esc(item['action'])}")
            if item["addressable_customers"]:
                st.caption(f"Addressable now: {item['addressable_customers']:,} active customers, "
                          f"\\${item['addressable_mrr']:,.0f}/month revenue.")

    st.divider()
    st.subheader("Prioritised retention list (highest risk first)")
    active = df[df["Churn"] == 0].sort_values("churn_probability", ascending=False)
    cols = ["customerID", "risk_tier", "churn_probability", "tenure", "Contract",
           "InternetService", "MonthlyCharges", "PaymentMethod"]
    st.dataframe(active[cols].head(50).rename(columns={
        "customerID": "Customer ID", "risk_tier": "Risk", "churn_probability": "Churn prob.",
        "tenure": "Tenure (mo)", "MonthlyCharges": "Monthly $"}), use_container_width=True, hide_index=True)
    st.download_button("Download full prioritised list (CSV)",
                       active[cols].to_csv(index=False).encode(), "retention_priority_list.csv", "text/csv")
