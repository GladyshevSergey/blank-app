import os
import yaml
import duckdb
import pandas as pd
import streamlit as st
import altair as alt
from textwrap import dedent

# ---------- CONFIG ----------
st.set_page_config(page_title="Elasticity Prototype (SEB schema)", layout="wide")
DB_PATH = "results.duckdb"
CSV_PATH = "results.csv"
POLICY_PATH = "policies.yaml"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# ---------- HELPERS ----------
@st.cache_data(show_spinner=False)
def load_policies(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None

@st.cache_data(show_spinner=False)
def load_data_to_duckdb(csv_path: str, db_path: str) -> pd.DataFrame:
    # Read with pandas to coerce types first (CSV may have strings for numbers)
    df = pd.read_csv(csv_path)

    # Coerce numerics (safe cast)
    num_cols = [
        "n_obs","n_price_points","price_cv","orders_cv","price_min","price_max","price_avg",
        "orders_avg","orders_std","revenue_avg","price_elasticity","elasticity_pval",
        "elasticity_r2","expected_change_+5pct_sales_pct","expected_change_-5pct_sales_pct",
        "expected_change_+1pct_sales_pct","expected_change_-1pct_sales_pct"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Coerce booleans & dates
    if "is_price_driver" in df.columns:
        df["is_price_driver"] = df["is_price_driver"].apply(_coerce_bool)
    if "last_update" in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce").dt.date

    # Create convenient aliases expected by the UI
    # elasticity = price_elasticity, r2 = elasticity_r2, base_* approximations
    df["elasticity"] = df.get("price_elasticity", pd.Series([None]*len(df)))
    df["r2"] = df.get("elasticity_r2", pd.Series([None]*len(df)))
    df["base_price"] = df.get("price_avg", pd.Series([None]*len(df)))
    # Use orders_avg as base volume; if unavailable, fall back to revenue/price
    if "orders_avg" in df.columns:
        df["base_volume"] = df["orders_avg"]
    else:
        df["base_volume"] = pd.NA
        if "revenue_avg" in df.columns and "price_avg" in df.columns:
            with pd.option_context('mode.use_inf_as_na', True):
                df["base_volume"] = (df["revenue_avg"] / df["price_avg"]).replace([pd.NA], None)

    # Persist into DuckDB
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS elasticities;")
    con.register("df_view", df)
    con.execute("""
        CREATE TABLE elasticities AS
        SELECT * FROM df_view
    """)
    out = con.execute("SELECT * FROM elasticities;").df()
    con.close()
    return out

@st.cache_data(show_spinner=False)
def query_duckdb(sql: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    df = con.execute(sql).df()
    con.close()
    return df

def ci_crosses_zero(row: pd.Series) -> bool:
    a, b = row.get("ci_low"), row.get("ci_high")
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    try:
        return float(a) * float(b) <= 0
    except Exception:
        return False

def price_move_guardrail(policies: dict, price_change_pct: float) -> dict:
    max_abs_move = float(policies["constraints"]["max_abs_price_move_pct"])
    ok = abs(price_change_pct) <= max_abs_move
    return {
        "ok": ok,
        "max_allowed": max_abs_move,
        "message": "" if ok else f"Requested move ±{abs(price_change_pct):.1f}% exceeds policy limit of ±{max_abs_move:.1f}%."
    }

def r2_flag(policies: dict, r2: float) -> str:
    try:
        weak_thr = float(policies["r2_thresholds"]["weak"]["lt"])
    except Exception:
        weak_thr = 0.10
    if pd.isna(r2):
        return ""
    return policies["r2_thresholds"]["weak"]["caveat"] if r2 < weak_thr else ""

def sensitivity_band(policies: dict, elasticity: float) -> str:
    if pd.isna(elasticity):
        return "No elasticity value available."
    e = abs(elasticity)
    low = policies["confidence_bands"]["low"]
    mid = policies["confidence_bands"]["mid"]
    high = policies["confidence_bands"]["high"]
    if e < float(low["abs_e_lt"]):
        return f"Low sensitivity. {low['guidance']}"
    if e >= float(high["abs_e_ge"]):
        return f"High sensitivity. {high['guidance']}"
    return f"Moderate sensitivity. {mid['guidance']}"

def what_if_calc(base_price: float, base_volume: float, elasticity: float, price_change_pct: float):
    dp = price_change_pct / 100.0
    dq = (elasticity or 0.0) * dp
    new_price = (base_price or 0.0) * (1 + dp)
    new_volume = (base_volume or 0.0) * (1 + dq)
    base_revenue = (base_price or 0.0) * (base_volume or 0.0)
    new_revenue = new_price * new_volume
    delta_rev_pct = (new_revenue / base_revenue - 1) * 100.0 if base_revenue else float("nan")
    return {
        "new_price": new_price,
        "new_volume": new_volume,
        "new_revenue": new_revenue,
        "delta_volume_pct": dq * 100.0,
        "delta_revenue_pct": delta_rev_pct
    }

def explain_with_rules(policies: dict, row: pd.Series) -> list[str]:
    bands = sensitivity_band(policies, row.elasticity)
    r2_note = r2_flag(policies, row.r2)
    # Your file doesn’t include CI; keep the check but it will usually be off
    ci_note = "Elasticity CI spans 0 → effect uncertain; prefer small moves & test." if policies["constraints"].get("require_ci_not_cross_zero") and ci_crosses_zero(row) else ""
    # Enrich with your flags
    driver = f"Price driver: {'Yes' if bool(row.get('is_price_driver')) else 'No'}."
    reason = row.get("non_price_elastic_reason")
    reason_note = f"Non-price driver hint: {reason}." if isinstance(reason, str) and reason else ""

    bullets = [
        f"Verdict: {bands} (ε={row.elasticity:.2f} if available, R²={row.r2:.2f} if available).",
        driver + (" " + reason_note if reason_note else ""),
        f"Evidence: level={row.level_type} → {row.level_value}, channel={row.channel}, last_update={row.last_update}.",
        "Next actions: sanity-check availability/promo/content; consider A/B for key SKUs."
    ]
    if r2_note:
        bullets.insert(1, f"Caveat: {r2_note}")
    if ci_note:
        bullets.insert(1, ci_note)
    return bullets[: int(policies.get("output_format", {}).get("bullets", 4))]

# ---------- LOAD ----------
if not os.path.exists(CSV_PATH):
    st.error(f"Missing {CSV_PATH}. Add your model export CSV and reload.")
    st.stop()

policies = load_policies(POLICY_PATH)
df = load_data_to_duckdb(CSV_PATH, DB_PATH)

# ---------- SIDEBAR ----------
st.sidebar.header("Filters")
include_overall = st.sidebar.checkbox("Include 'overall' rows", value=False)

channels = ["(All)"] + sorted(df["channel"].dropna().unique().tolist())
level_types = ["(All)"] + sorted(df["level_type"].dropna().unique().tolist())

sel_channel = st.sidebar.selectbox("Channel", channels)
sel_level_type = st.sidebar.selectbox("Level type", level_types)
search_value = st.sidebar.text_input("Filter level_value (contains)", "")

query = "SELECT * FROM elasticities WHERE 1=1"
if not include_overall:
    query += " AND (level_type IS NULL OR level_type <> 'overall')"
if sel_channel != "(All)":
    query += f" AND channel = '{sel_channel.replace(\"'\",\"''\")}'"
if sel_level_type != "(All)":
    query += f" AND level_type = '{sel_level_type.replace(\"'\",\"''\")}'"
if search_value:
    query += f" AND CAST(level_value AS VARCHAR) ILIKE '%{search_value.replace(\"'\",\"''\")}%'"

view_df = query_duckdb(query)

st.title("Elasticity Prototype — SEB schema")
st.caption("Dashboard • Explain • What-If — backed by a local DuckDB, using your CSV schema")

# ---------- TABS ----------
tab_dash, tab_explain, tab_whatif = st.tabs(["📊 Dashboard", "💬 Explain (rules + optional AI)", "🧮 What-If"])

with tab_dash:
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.subheader("Slice")
        st.dataframe(view_df, use_container_width=True, hide_index=True)
        st.caption(f"{len(view_df)} rows")
    with c2:
        st.subheader("Top |price_elasticity| in selection")
        if len(view_df) > 0 and "price_elasticity" in view_df.columns:
            chart_df = (
                view_df.assign(abs_e=view_df["price_elasticity"].abs())
                .sort_values("abs_e", ascending=False)
                .head(min(25, len(view_df)))
            )
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("abs_e:Q", title="|price_elasticity|"),
                    y=alt.Y("level_value:N", sort="-x", title="level_value"),
                    color=alt.Color("level_type:N", title="level_type"),
                    tooltip=["channel","level_type","level_value","price_elasticity","elasticity_r2","revenue_avg","last_update"]
                )
                .properties(height=520)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No elasticity column found in this slice.")

with tab_explain:
    st.subheader("Pick a row to explain")
    if len(view_df) == 0:
        st.info("No data in this slice.")
    else:
        opts = list(view_df.index)
        fmt = lambda i: f"{view_df.loc[i,'level_value']} • {view_df.loc[i,'channel']} • ε={view_df.loc[i,'price_elasticity']:.2f}" if pd.notna(view_df.loc[i,"price_elasticity"]) else f"{view_df.loc[i,'level_value']} • {view_df.loc[i,'channel']}"
        idx = st.selectbox("Row", options=opts, format_func=fmt)
        row = view_df.loc[idx]

        st.markdown(
            f"**Level:** {row['level_type']} → **{row['level_value']}**  \n"
            f"**Channel:** {row['channel']}  \n"
            f"**ε:** {row.get('price_elasticity', float('nan')):.2f}  •  **R²:** {row.get('elasticity_r2', float('nan')):.2f}  •  **p-val:** {row.get('elasticity_pval', float('nan')):.3f}  \n"
            f"**Base price:** {row.get('price_avg', float('nan')):.2f}  •  **Base volume (orders_avg):** {row.get('orders_avg', float('nan')):.2f}  \n"
            f"**Last update:** {row.get('last_update')}"
        )

        # Deterministic rules output
        st.write("**Rules-based interpretation**")
        rules_row = pd.Series({
            "elasticity": row.get("price_elasticity"),
            "r2": row.get("elasticity_r2"),
            "level_type": row.get("level_type"),
            "level_value": row.get("level_value"),
            "channel": row.get("channel"),
            "last_update": row.get("last_update"),
            "is_price_driver": row.get("is_price_driver"),
            "non_price_elastic_reason": row.get("non_price_elastic_reason"),
        })
        for b in explain_with_rules(policies, rules_row):
            st.write(f"- {b}")

        # Optional GPT narrative grounded by YAML
        st.write("**AI narrative (optional)**")
        if not OPENAI_API_KEY:
            st.info("Set OPENAI_API_KEY env var to enable AI narrative.")
        else:
            try:
                from openai import OpenAI
                client = OpenAI()
                policy_snippet = yaml.dump(policies, sort_keys=False)
                prompt = dedent(f"""
                You are a pricing analyst. Convert these inputs into a short business interpretation.
                Follow the YAML rules; include cautions if evidence is weak.

                YAML_RULES:
                {policy_snippet}

                INPUT_ROW:
                channel={row.get('channel')}
                level_type={row.get('level_type')}
                level_value={row.get('level_value')}
                price_elasticity={row.get('price_elasticity')}
                elasticity_r2={row.get('elasticity_r2')}
                elasticity_pval={row.get('elasticity_pval')}
                price_avg={row.get('price_avg')}
                orders_avg={row.get('orders_avg')}
                is_price_driver={row.get('is_price_driver')}
                non_price_elastic_reason={row.get('non_price_elastic_reason')}
                last_update={row.get('last_update')}

                Output exactly 4 bullet points: verdict, expected_volume_change (qualitative), revenue_implication (qualitative), next_actions.
                """)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.2,
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.warning(f"AI call failed: {e}")

with tab_whatif:
    st.subheader("What-If (policy-guarded)")
    if len(view_df) == 0:
        st.info("No data in this slice.")
    else:
        idx = st.selectbox(
            "Row for what-if",
            options=list(view_df.index),
            key="whatif_idx",
            format_func=lambda i: f"{view_df.loc[i,'level_value']} • {view_df.loc[i,'channel']} • ε={view_df.loc[i,'price_elasticity']:.2f}" if pd.notna(view_df.loc[i,"price_elasticity"]) else f"{view_df.loc[i,'level_value']} • {view_df.loc[i,'channel']}"
        )
        row = view_df.loc[idx]
        price_move = st.slider("Planned price change (%)", min_value=-20.0, max_value=20.0, value=5.0, step=0.5)

        # Guardrails
        g = price_move_guardrail(policies, price_move)
        if not g["ok"]:
            st.error(g["message"])

        # Prefer your precomputed expected_change_* for exact ±1% / ±5%;
        # otherwise compute via elasticity.
        pre = None
        if abs(price_move - 5.0) < 1e-9 and "expected_change_+5pct_sales_pct" in row:
            pre = row["expected_change_+5pct_sales_pct"]
        elif abs(price_move + 5.0) < 1e-9 and "expected_change_-5pct_sales_pct" in row:
            pre = row["expected_change_-5pct_sales_pct"]
        elif abs(price_move - 1.0) < 1e-9 and "expected_change_+1pct_sales_pct" in row:
            pre = row["expected_change_+1pct_sales_pct"]
        elif abs(price_move + 1.0) < 1e-9 and "expected_change_-1pct_sales_pct" in row:
            pre = row["expected_change_-1pct_sales_pct"]

        if pd.notna(pre) if pre is not None else False:
            delta_vol_pct = float(pre)
            # compute new volume and revenue based on pre delta
            base_price = float(row.get("price_avg", 0.0) or 0.0)
            base_vol = float(row.get("orders_avg", 0.0) or 0.0)
            new_price = base_price * (1 + price_move/100.0)
            new_volume = base_vol * (1 + delta_vol_pct/100.0)
            base_revenue = base_price * base_vol
            new_revenue = new_price * new_volume
            delta_rev_pct = (new_revenue / base_revenue - 1) * 100.0 if base_revenue else float("nan")
            out = dict(new_price=new_price, new_volume=new_volume, new_revenue=new_revenue,
                       delta_volume_pct=delta_vol_pct, delta_revenue_pct=delta_rev_pct)
            used = "Precomputed expected_change_*"
        else:
            out = what_if_calc(
                base_price=float(row.get("price_avg", 0.0) or 0.0),
                base_volume=float(row.get("orders_avg", 0.0) or 0.0),
                elasticity=float(row.get("price_elasticity", 0.0) or 0.0),
                price_change_pct=float(price_move),
            )
            used = "Elasticity approximation (ΔQ% ≈ ε·ΔP%)"

        c1, c2, c3 = st.columns(3)
        c1.metric("New price", f"{out['new_price']:.2f}", f"{price_move:+.1f}%")
        c2.metric("New volume", f"{out['new_volume']:.1f}", f"{out['delta_volume_pct']:+.2f}%")
        c3.metric("New revenue", f"{out['new_revenue']:.2f}", f"{out['delta_revenue_pct']:+.2f}%")
        st.caption(f"Method: {used}.  Last update: {row.get('last_update')}.")

        # Flags
        flags = []
        caveat = r2_flag(policies, row.get("elasticity_r2"))
        if caveat:
            flags.append(caveat)
        nper = int(row.get("n_price_points")) if pd.notna(row.get("n_price_points")) else None
        if nper is not None and nper < 5:
            flags.append("Very few price points; results may be unstable.")
        if not bool(row.get("is_price_driver")) and isinstance(row.get("non_price_elastic_reason"), str) and row.get("non_price_elastic_reason"):
            flags.append(f"Non-price driver likely: {row.get('non_price_elastic_reason')}.")
        if flags:
            st.warning(" ".join(flags))
