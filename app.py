"""
Polished web front-end for the Kochiyo agent.

Run locally:   streamlit run app.py
Free hosting:  push this repo to GitHub, then deploy at
               https://share.streamlit.io (Streamlit Community Cloud, free)
               or https://huggingface.co/spaces (free CPU Space, "Streamlit"
               SDK). Set GROQ_API_KEY as a secret on whichever platform you
               use -- never commit it.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.clean import load_and_clean
from src.kpis import compute_kpis
from src.agent import Agent

HERE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Kochiyo Order Insights",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px;}

    .kochiyo-hero {
        background: linear-gradient(135deg, #b3261e 0%, #7a1a14 100%);
        padding: 2rem 2.25rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .kochiyo-hero h1 {margin: 0; font-size: 2rem; font-weight: 700;}
    .kochiyo-hero p {margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 0.95rem;}

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #eee;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricLabel"] {font-size: 0.8rem; color: #666;}

    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    .kochiyo-caption {color: #888; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_resource
def load_everything():
    df, report = load_and_clean(HERE / "data" / "kochiyo_orders_export.csv")
    kpis = compute_kpis(df)
    return df, report, kpis


@st.cache_resource
def get_agent(_df):
    return Agent(_df)


df, clean_report, kpis = load_everything()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="kochiyo-hero">
    <h1>🍜 Kochiyo Order Insights</h1>
    <p>Live KPIs and natural-language Q&amp;A over {kpis['sales_summary']['delivered_line_items']:,}
    delivered orders across {len(kpis['revenue_by_store'])} stores &middot;
    {kpis['date_range']['start'][:10]} to {kpis['date_range']['end'][:10]}</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊  Dashboard", "💬  Ask a question"])

# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------
with tab1:
    s = kpis["sales_summary"]
    ob = kpis["order_status_breakdown"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total revenue", f"¥{s['total_revenue_yen']:,.0f}")
    c2.metric("Delivered orders", f"{s['delivered_line_items']:,}")
    c3.metric("Units sold", f"{s['total_units_sold']:,}")
    c4.metric("Avg order value", f"¥{s['avg_line_value_yen']:,.0f}")
    c5.metric("Avg rating", f"{kpis['avg_customer_rating']:.1f} ⭐" if kpis["avg_customer_rating"] else "—")

    c1, c2 = st.columns(2)
    c1.metric("Cancellation rate", f"{ob['cancellation_rate_pct']}%")
    c2.metric("Refund rate", f"{ob['refund_rate_pct']}%")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top items by revenue")
        top_rev = pd.DataFrame(kpis["top_items_by_revenue"][:8]).set_index("item")
        st.bar_chart(top_rev, horizontal=True, color="#b3261e")
    with col2:
        st.subheader("Peak order hours")
        hrs = pd.DataFrame(kpis["peak_order_hours"])
        hrs["hour"] = hrs["hour"].apply(lambda h: f"{h:02d}:00")
        hrs = hrs.set_index("hour")[["order_count"]]
        st.bar_chart(hrs, color="#b3261e")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Revenue by store")
        store_df = pd.DataFrame(kpis["revenue_by_store"]).set_index("store")
        st.bar_chart(store_df, horizontal=True, color="#7a1a14")
    with col4:
        st.subheader("Revenue by payment method")
        pay_df = pd.DataFrame(kpis["revenue_by_payment_method"]).set_index("method")
        st.bar_chart(pay_df, horizontal=True, color="#7a1a14")

    with st.expander("🧹 Data cleaning report — what was fixed and why"):
        st.json(clean_report)
        st.caption("Full write-up of every issue found (date-format ambiguity, "
                   "duplicate rows, item-name variants, etc.) is in README.md.")

# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------
with tab2:
    st.caption("Ask about sales, top/least-selling items, peak hours, cancellations, ratings, "
               "stores, payment methods — in plain English.")

    EXAMPLE_QUESTIONS = [
        "What was our best-selling item in June?",
        "What time of day do we get the most orders?",
        "What's the least selling item?",
        "What's our cancellation rate?",
        "Which store makes the most money?",
    ]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.write("**Try:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked_question = None
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            clicked_question = q

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                with st.expander("🔧 Tool calls made"):
                    st.json(msg["trace"])

    typed_question = st.chat_input("Ask about the order data...")
    question = clicked_question or typed_question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                agent = get_agent(df)
                with st.spinner("Checking the data..."):
                    answer, trace = agent.run(question)
                st.markdown(answer)
                if trace:
                    with st.expander("🔧 Tool calls made"):
                        st.json(trace)
                st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})
            except RuntimeError as e:
                st.error(str(e))
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}"})

    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

st.markdown('<p class="kochiyo-caption">Built with pandas + Groq/Ollama tool-calling. '
            'No paid services used.</p>', unsafe_allow_html=True)
