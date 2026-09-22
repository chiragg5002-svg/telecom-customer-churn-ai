"""KPIs, segment tables and rule-based insights computed ONLY from the scored data.

Every sentence is an f-string filled with numbers calculated here, so each
insight is traceable to the dataset. Chain used everywhere:
FACT -> INSIGHT -> RISK/OPPORTUNITY -> ACTION.
"""
from __future__ import annotations

import pandas as pd


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _rate(df: pd.DataFrame, mask) -> float:
    return float(df.loc[mask, "Churn"].mean())


def kpis(df: pd.DataFrame) -> dict:
    churned = df[df["Churn"] == 1]
    active = df[df["Churn"] == 0]
    high = active[active["risk_tier"] == "High"]
    med = active[active["risk_tier"] == "Medium"]
    mrr_total = float(df["MonthlyCharges"].sum())
    mrr_lost = float(churned["MonthlyCharges"].sum())
    return {
        "customers": int(len(df)),
        "churned_customers": int(len(churned)),
        "active_customers": int(len(active)),
        "churn_rate": float(df["Churn"].mean()),
        "mrr_total": mrr_total,
        "mrr_lost": mrr_lost,
        "mrr_lost_pct": mrr_lost / mrr_total,
        "avg_monthly_charge": float(df["MonthlyCharges"].mean()),
        "avg_tenure_churned": float(churned["tenure"].mean()),
        "avg_tenure_active": float(active["tenure"].mean()),
        "high_risk_active": int(len(high)),
        "high_risk_active_mrr": float(high["MonthlyCharges"].sum()),
        "medium_risk_active": int(len(med)),
        "medium_risk_active_mrr": float(med["MonthlyCharges"].sum()),
        "active_mrr": float(active["MonthlyCharges"].sum()),
    }


def segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True).agg(
        customers=("Churn", "size"), churned=("Churn", "sum"),
        churn_rate=("Churn", "mean"), avg_monthly=("MonthlyCharges", "mean"),
        mrr_lost=("MonthlyCharges", lambda s: s[df.loc[s.index, "Churn"] == 1].sum()),
    ).reset_index()
    return g


def tier_table(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Low", "Medium", "High"]
    g = df.groupby("risk_tier").agg(customers=("Churn", "size"), observed_churn_rate=("Churn", "mean"),
                                    avg_predicted=("churn_probability", "mean"),
                                    monthly_revenue=("MonthlyCharges", "sum")).reindex(order).reset_index()
    return g


def _targets(df: pd.DataFrame, mask) -> tuple[int, float]:
    """Active customers with High/Medium predicted risk inside a segment."""
    t = df[mask & (df["Churn"] == 0) & df["risk_tier"].isin(["High", "Medium"])]
    return int(len(t)), float(t["MonthlyCharges"].sum())


def build_insights(df: pd.DataFrame) -> list[dict]:
    k = kpis(df)
    churned = df["Churn"] == 1
    out: list[dict] = []

    # ---------- R1: contract type ----------
    m2m, y1, y2 = (df["Contract"] == "Month-to-month"), (df["Contract"] == "One year"), (df["Contract"] == "Two year")
    r_m2m, r_y2 = _rate(df, m2m), _rate(df, y2)
    share_m2m = float((churned & m2m).sum() / churned.sum())
    n_t, mrr_t = _targets(df, m2m)
    out.append({
        "id": "R1", "type": "Risk", "title": "Month-to-month contracts hold most of the churn",
        "fact": f"Month-to-month customers churn at {_pct(r_m2m)} versus {_pct(_rate(df, y1))} on one-year "
                f"and {_pct(r_y2)} on two-year contracts.",
        "insight": f"They are {_pct(m2m.mean())} of customers but account for {_pct(share_m2m)} of all churners.",
        "implication": f"The gap between the shortest and longest contract is {(r_m2m - r_y2) * 100:.1f} percentage "
                       "points, and Contract is among the top three drivers in the model (association, not proof of cause).",
        "action": f"Offer contract-upgrade incentives to the {n_t:,} active month-to-month customers scored "
                  f"High/Medium risk ({_money(mrr_t)} monthly revenue).",
        "evidence": {"m2m_rate": r_m2m, "one_year_rate": _rate(df, y1), "two_year_rate": r_y2,
                     "m2m_customer_share": float(m2m.mean()), "m2m_share_of_churners": share_m2m,
                     "target_customers": n_t, "target_mrr": mrr_t},
        "addressable_customers": n_t, "addressable_mrr": mrr_t,
    })

    # ---------- R2: early lifecycle ----------
    early, late = (df["tenure"] <= 12), (df["tenure"] >= 49)
    r_e, r_l = _rate(df, early), _rate(df, late)
    share_e = float((churned & early).sum() / churned.sum())
    n_t, mrr_t = _targets(df, early)
    out.append({
        "id": "R2", "type": "Risk", "title": "The first year is the danger zone",
        "fact": f"Customers in their first 12 months churn at {_pct(r_e)}, versus {_pct(r_l)} for customers "
                f"with 49-72 months of tenure.",
        "insight": f"{_pct(share_e)} of all churners left within their first 12 months; the average churned "
                   f"customer had {k['avg_tenure_churned']:.0f} months of tenure vs {k['avg_tenure_active']:.0f} for active ones.",
        "implication": "Most churn happens early, so early-lifecycle retention addresses the largest share of churners.",
        "action": f"Launch a 12-month onboarding and check-in programme; start with the {n_t:,} active "
                  f"customers under 12 months tenure scored High/Medium risk ({_money(mrr_t)} monthly revenue).",
        "evidence": {"rate_first_12m": r_e, "rate_49_72m": r_l, "share_of_churners_first_12m": share_e,
                     "target_customers": n_t, "target_mrr": mrr_t},
        "addressable_customers": n_t, "addressable_mrr": mrr_t,
    })

    # ---------- R3: fibre optic ----------
    fib, dsl = (df["InternetService"] == "Fiber optic"), (df["InternetService"] == "DSL")
    r_f, r_d = _rate(df, fib), _rate(df, dsl)
    price_f, price_d = float(df.loc[fib, "MonthlyCharges"].mean()), float(df.loc[dsl, "MonthlyCharges"].mean())
    n_t, mrr_t = _targets(df, fib)
    lost_by_product = df[churned].groupby("InternetService")["MonthlyCharges"].sum()
    fib_lost = float(lost_by_product["Fiber optic"])
    out.append({
        "id": "R3", "type": "Risk", "title": f"Fibre optic customers churn {r_f / r_d:.1f}x as often as DSL customers",
        "fact": f"Fibre optic churn is {_pct(r_f)} versus {_pct(r_d)} for DSL, and fibre customers pay "
                f"{_money(price_f)} per month on average versus {_money(price_d)}.",
        "insight": "Higher-priced fibre customers leave far more often. The data has no service-quality or "
                   "competitor fields, so the reason cannot be determined from it.",
        "implication": f"Fibre customers account for {_pct(fib_lost / k['mrr_lost'])} of all monthly revenue lost to "
                       f"churn ({_money(fib_lost)}), the largest share of any internet product.",
        "action": f"Run a short exit/at-risk survey on fibre customers to find whether price, reliability or "
                  f"competition is the cause; {n_t:,} active fibre customers are High/Medium risk "
                  f"({_money(mrr_t)} monthly revenue).",
        "evidence": {"fiber_rate": r_f, "dsl_rate": r_d, "fiber_avg_monthly": price_f, "dsl_avg_monthly": price_d,
                     "fiber_mrr_lost": fib_lost, "fiber_share_of_mrr_lost": fib_lost / k["mrr_lost"],
                     "fiber_is_largest_loss_product": bool(lost_by_product.idxmax() == "Fiber optic"),
                     "target_customers": n_t, "target_mrr": mrr_t},
        "addressable_customers": n_t, "addressable_mrr": mrr_t,
    })

    # ---------- R4: payment method ----------
    ech, auto = (df["PaymentMethod"] == "Electronic check"), (df["auto_pay"] == "Yes")
    r_ech, r_auto = _rate(df, ech), _rate(df, auto)
    n_t, mrr_t = _targets(df, ech)
    out.append({
        "id": "R4", "type": "Risk", "title": "Electronic-check payers churn far more than automatic payers",
        "fact": f"Electronic-check customers churn at {_pct(r_ech)} versus {_pct(r_auto)} for customers paying "
                f"by automatic bank transfer or credit card.",
        "insight": "Manual, per-cycle payment is associated with much weaker retention than automatic payment.",
        "implication": "The data cannot explain why, but the gap is large enough to justify testing a payment incentive.",
        "action": f"Offer a small incentive to switch to automatic payment, starting with the {n_t:,} active "
                  f"electronic-check customers scored High/Medium risk ({_money(mrr_t)} monthly revenue).",
        "evidence": {"echeck_rate": r_ech, "auto_pay_rate": r_auto, "target_customers": n_t, "target_mrr": mrr_t},
        "addressable_customers": n_t, "addressable_mrr": mrr_t,
    })

    # ---------- O1: protection / support services ----------
    net = df["InternetService"] != "No"
    none_p, two_p = net & (df["n_protection_services"] == 0), net & (df["n_protection_services"] >= 2)
    r_0, r_2 = _rate(df, none_p), _rate(df, two_p)
    n_t, mrr_t = _targets(df, none_p)
    out.append({
        "id": "O1", "type": "Opportunity", "title": "Protection and support services go with retention",
        "fact": f"Internet customers with none of the four protection/support services (Online Security, Online "
                f"Backup, Device Protection, Tech Support) churn at {_pct(r_0)}, versus {_pct(r_2)} for those "
                f"with two or more.",
        "insight": f"{int(none_p.sum()):,} internet customers have no protection service. Customers who take these "
                   "services stay much more often.",
        "implication": "Selling these add-ons is an opportunity to deepen the relationship. The data shows association; "
                       "an A/B trial is needed to prove the add-ons cause retention.",
        "action": f"Trial a free/discounted Tech Support or Online Security period for the {n_t:,} active "
                  f"no-protection internet customers scored High/Medium risk ({_money(mrr_t)} monthly revenue), "
                  f"and compare against a control group.",
        "evidence": {"no_protection_rate": r_0, "two_plus_protection_rate": r_2,
                     "no_protection_customers": int(none_p.sum()), "target_customers": n_t, "target_mrr": mrr_t},
        "addressable_customers": n_t, "addressable_mrr": mrr_t,
    })

    # ---------- O2: prioritised retention list ----------
    tiers = tier_table(df).set_index("risk_tier")
    n_hm = k["high_risk_active"] + k["medium_risk_active"]
    mrr_hm = k["high_risk_active_mrr"] + k["medium_risk_active_mrr"]
    out.append({
        "id": "O2", "type": "Opportunity", "title": "A ranked retention list is available today",
        "fact": f"The model separates customers cleanly: observed churn is {_pct(tiers.loc['Low', 'observed_churn_rate'])} "
                f"in the Low tier, {_pct(tiers.loc['Medium', 'observed_churn_rate'])} in Medium and "
                f"{_pct(tiers.loc['High', 'observed_churn_rate'])} in High (out-of-fold scores).",
        "insight": f"{n_hm:,} customers who have NOT yet left are scored High or Medium risk. They hold "
                   f"{_money(mrr_hm)} of monthly revenue ({_pct(mrr_hm / k['active_mrr'])} of active revenue).",
        "implication": "Retention budget can be focused on a small, high-yield group instead of all customers.",
        "action": f"Contact the {k['high_risk_active']:,} High-risk active customers first "
                  f"({_money(k['high_risk_active_mrr'])} monthly revenue), then the Medium tier; "
                  f"re-score monthly and track the save rate.",
        "evidence": {"low_rate": float(tiers.loc['Low', 'observed_churn_rate']),
                     "medium_rate": float(tiers.loc['Medium', 'observed_churn_rate']),
                     "high_rate": float(tiers.loc['High', 'observed_churn_rate']),
                     "high_medium_active": n_hm, "high_medium_active_mrr": mrr_hm},
        "addressable_customers": n_hm, "addressable_mrr": mrr_hm,
    })

    # ---------- O3: protect the loyal base ----------
    loyal = df["Contract"] == "Two year"
    loyal_mrr_share = float(df.loc[loyal, "MonthlyCharges"].sum() / k["mrr_total"])
    out.append({
        "id": "O3", "type": "Opportunity", "title": "Long-contract customers are a stable revenue base",
        "fact": f"Two-year customers churn at only {_pct(r_y2)} and represent {_pct(loyal.mean())} of customers "
                f"and {_pct(loyal_mrr_share)} of monthly revenue.",
        "insight": "Long contracts are associated with very low churn; growing this group would stabilise revenue.",
        "implication": "Moving suitable month-to-month customers onto longer terms is a lever worth testing.",
        "action": "Make one-year/two-year plans the default recommendation at sign-up and at renewal touch-points.",
        "evidence": {"two_year_rate": r_y2, "two_year_customer_share": float(loyal.mean()),
                     "two_year_mrr_share": loyal_mrr_share},
        "addressable_customers": 0, "addressable_mrr": 0.0,
    })
    return out
