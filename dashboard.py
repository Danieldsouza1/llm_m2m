"""
dashboard.py
------------
Streamlit dashboard for the Smart Energy Grid AI Fault Management project.

Run with:
    streamlit run dashboard.py

Sections:
  1. Project Overview   — what the system is, architecture, how it works
  2. Benchmark Results  — accuracy/latency comparison of all 4 approaches
  3. Live Run History   — results.csv viewer with per-run breakdown
  4. Fault Browser      — all 10 library faults with correct actions
"""

import os
import sys
import time
import subprocess

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
BASELINE_CSV    = os.path.join(BASE_DIR, "evaluation", "baseline_results.csv")
RESULTS_CSV     = os.path.join(BASE_DIR, "evaluation", "results.csv")
FAULT_GEN_PATH  = os.path.join(BASE_DIR, "fault_generator.py")

sys.path.insert(0, BASE_DIR)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Grid AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }

    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .label {
        color: #8892b0;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #ccd6f6;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-card .sub {
        color: #64ffda;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    .section-header {
        color: #64ffda;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.2rem;
    }
    .big-title {
        color: #ccd6f6;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #8892b0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .approach-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.1rem;
    }
    .badge-llmrag   { background: #1a3a2a; color: #64ffda; border: 1px solid #64ffda44; }
    .badge-llmonly  { background: #2a1a3a; color: #bd93f9; border: 1px solid #bd93f944; }
    .badge-ragonly  { background: #1a2a3a; color: #79b8ff; border: 1px solid #79b8ff44; }
    .badge-rulebased{ background: #3a2a1a; color: #ffb86c; border: 1px solid #ffb86c44; }

    .fault-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.6rem;
    }
    .fault-card .fault-id   { color: #64ffda; font-size: 0.75rem; font-weight: 700; }
    .fault-card .fault-type { color: #8892b0; font-size: 0.75rem; margin-bottom: 0.3rem; }
    .fault-card .fault-err  { color: #ccd6f6; font-size: 0.88rem; font-weight: 500; }
    .action-retry    { color: #79b8ff; font-weight: 700; }
    .action-redirect { color: #64ffda; font-weight: 700; }
    .action-shutdown { color: #ff5555; font-weight: 700; }

    .status-success { color: #50fa7b; font-weight: 600; }
    .status-failure { color: #ff5555; font-weight: 600; }

    .info-box {
        background: #1e2130;
        border-left: 3px solid #64ffda;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0;
        color: #a8b2d8;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_baseline() -> pd.DataFrame | None:
    if not os.path.exists(BASELINE_CSV):
        return None
    try:
        df = pd.read_csv(BASELINE_CSV)
        # Handle both old format (model,accuracy,latency) and new format
        if "model" in df.columns and "approach" not in df.columns:
            df = df.rename(columns={"model": "approach"})
        return df
    except Exception:
        return None


def load_results() -> pd.DataFrame | None:
    if not os.path.exists(RESULTS_CSV):
        return None
    try:
        df = pd.read_csv(RESULTS_CSV)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def action_color(action: str) -> str:
    colors = {"RETRY": "#79b8ff", "REDIRECT": "#64ffda", "SHUTDOWN": "#ff5555"}
    return colors.get(str(action).upper(), "#8892b0")


def approach_badge(name: str) -> str:
    cls = {
        "LLM+RAG":    "badge-llmrag",
        "LLM-Only":   "badge-llmonly",
        "RAG-Only":   "badge-ragonly",
        "Rule-Based": "badge-rulebased",
    }.get(name, "badge-rulebased")
    return f'<span class="approach-badge {cls}">{name}</span>'


FAULT_LIBRARY = [
    {"id": "F001", "type": "line_overload",        "agent": "Transmission1",
     "error": "ERROR: Line overload in sector B",                        "correct": "REDIRECT"},
    {"id": "F002", "type": "line_disconnect",       "agent": "Transmission1",
     "error": "ERROR: Transmission line disconnected at node 7",         "correct": "REDIRECT"},
    {"id": "F003", "type": "phase_imbalance",       "agent": "Transmission1",
     "error": "ERROR: Phase imbalance detected on line 3",               "correct": "RETRY"},
    {"id": "F004", "type": "transformer_fault",     "agent": "Substation1",
     "error": "ERROR: Transformer overheating at substation 2",          "correct": "SHUTDOWN"},
    {"id": "F005", "type": "breaker_failure",       "agent": "Substation1",
     "error": "ERROR: Circuit breaker failed to trip at bus 4",          "correct": "SHUTDOWN"},
    {"id": "F006", "type": "voltage_sag",           "agent": "Substation1",
     "error": "ERROR: Voltage sag detected at distribution bus 6",       "correct": "RETRY"},
    {"id": "F007", "type": "frequency_deviation",   "agent": "Plant1",
     "error": "ERROR: Generator output frequency deviation",             "correct": "RETRY"},
    {"id": "F008", "type": "turbine_shutdown",      "agent": "Plant1",
     "error": "ERROR: Turbine emergency shutdown initiated",             "correct": "SHUTDOWN"},
    {"id": "F009", "type": "partial_load_loss",     "agent": "Plant1",
     "error": "ERROR: Partial load loss on generator unit 3",            "correct": "REDIRECT"},
    {"id": "F010", "type": "ground_fault",          "agent": "Transmission1",
     "error": "ERROR: Ground fault on feeder line F12",                  "correct": "RETRY"},
]


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚡ Energy Grid AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Overview", "📊  Benchmark Results",
         "🔴  Live Run History", "🗂️  Fault Browser"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='color:#8892b0; font-size:0.78rem; line-height:1.7'>
    <b style='color:#64ffda'>Model</b><br>mistral:7b-instruct-q4_0<br><br>
    <b style='color:#64ffda'>Embeddings</b><br>nomic-embed-text<br><br>
    <b style='color:#64ffda'>Vector DB</b><br>ChromaDB (in-memory)<br><br>
    <b style='color:#64ffda'>Broker</b><br>MQTT localhost:1883<br><br>
    <b style='color:#64ffda'>Manual chunks</b><br>53 (section-aware)
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

if "Overview" in page:
    st.markdown('<p class="section-header">Smart Energy Grid</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-title">AI Fault Management System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Comparing Rule-Based, LLM-Only, RAG-Only, and LLM+RAG approaches for autonomous grid fault recovery</p>', unsafe_allow_html=True)

    # Key metrics row
    baseline_df = load_baseline()
    results_df  = load_results()

    col1, col2, col3, col4 = st.columns(4)

    best_acc = "—"
    best_name = "—"
    if baseline_df is not None and "approach" in baseline_df.columns and "is_correct" in baseline_df.columns:
        summary = baseline_df.groupby("approach")["is_correct"].mean()
        best_name = summary.idxmax()
        best_acc  = f"{summary.max():.0%}"
    elif baseline_df is not None and "accuracy" in baseline_df.columns:
        idx = baseline_df["accuracy"].idxmax()
        best_name = baseline_df.loc[idx, "approach"] if "approach" in baseline_df.columns else "LLM+RAG"
        best_acc  = f"{baseline_df.loc[idx, 'accuracy']:.0%}"

    total_runs = len(results_df) if results_df is not None else 0
    failures   = len(results_df[results_df["status"] == "FAILURE"]) if results_df is not None else 0
    recoveries = 0
    if results_df is not None and "decision" in results_df.columns:
        recoveries = len(results_df[(results_df["status"] == "FAILURE") &
                                    (results_df["decision"].notna()) &
                                    (results_df["decision"] != "")])

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Best Approach</div>
            <div class="value" style="font-size:1.3rem">{best_name}</div>
            <div class="sub">{best_acc} accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Library Faults</div>
            <div class="value">10</div>
            <div class="sub">predefined scenarios</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Live Events Logged</div>
            <div class="value">{total_runs}</div>
            <div class="sub">{failures} failures detected</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Recovery Rate</div>
            <div class="value">{f'{recoveries/max(failures,1):.0%}' if failures else '—'}</div>
            <div class="sub">of failures handled</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### How the system works")
        st.markdown("""<div class="info-box">
        The grid controller manages three agents — <b>PowerPlant</b>, <b>Transmission</b>,
        and <b>Substation</b> — over an MQTT message broker. Tasks are assigned via LLM planning.
        When a fault occurs, the controller queries a RAG pipeline built from a 7-section
        power engineering manual, retrieves the 6 most relevant chunks, and uses Mistral 7B
        to reason over them and decide: <b style="color:#79b8ff">RETRY</b>,
        <b style="color:#64ffda">REDIRECT</b>, or <b style="color:#ff5555">SHUTDOWN</b>.
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Four approaches compared")
        approaches = [
            ("Rule-Based",  "badge-rulebased", "Keyword matching on error message. Instant but brittle — fails on novel fault wording."),
            ("LLM-Only",    "badge-llmonly",   "Mistral reasons from error alone, no manual context. Inconsistent on small quantized model."),
            ("RAG-Only",    "badge-ragonly",    "Retrieves manual chunks, scores by keyword frequency. No LLM reasoning involved."),
            ("LLM+RAG",     "badge-llmrag",     "Retrieves relevant chunks then LLM reasons over them. Best accuracy, especially on novel faults."),
        ]
        for name, cls, desc in approaches:
            st.markdown(f"""
            <div style="margin-bottom:0.7rem">
                <span class="approach-badge {cls}">{name}</span>
                <span style="color:#8892b0; font-size:0.85rem; margin-left:0.5rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("#### System architecture")
        # Architecture diagram using plotly
        fig = go.Figure()

        nodes = [
            ("Fault Generator",    0.5,  0.92, "#1a3a2a", "#64ffda"),
            ("Grid Controller",    0.5,  0.72, "#1a1a3a", "#bd93f9"),
            ("Mistral 7B",         0.18, 0.52, "#2a1a1a", "#ffb86c"),
            ("RAG Pipeline",       0.82, 0.52, "#1a2a2a", "#79b8ff"),
            ("MQTT Broker",        0.5,  0.38, "#1e2130", "#8892b0"),
            ("PowerPlant",         0.18, 0.15, "#1a2a3a", "#79b8ff"),
            ("Transmission",       0.5,  0.15, "#1a2a3a", "#79b8ff"),
            ("Substation",         0.82, 0.15, "#1a2a3a", "#79b8ff"),
            ("Logger / CSV",       0.18, 0.38, "#1a2a1a", "#50fa7b"),
        ]

        edges = [
            (0.5,0.88, 0.5,0.76),
            (0.5,0.68, 0.5,0.42),
            (0.5,0.72, 0.18,0.56),
            (0.5,0.72, 0.82,0.56),
            (0.5,0.34, 0.18,0.21),
            (0.5,0.34, 0.5,0.21),
            (0.5,0.34, 0.82,0.21),
            (0.5,0.38, 0.18,0.42),
        ]

        for x0,y0,x1,y1 in edges:
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="#2d3250", width=1.5))

        for label, x, y, bg, border in nodes:
            fig.add_shape(type="rect",
                          x0=x-0.13, y0=y-0.055, x1=x+0.13, y1=y+0.055,
                          line=dict(color=border, width=1.5),
                          fillcolor=bg)
            fig.add_annotation(x=x, y=y, text=label,
                               font=dict(color=border, size=11, family="monospace"),
                               showarrow=False)

        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            xaxis=dict(visible=False, range=[0,1]),
            yaxis=dict(visible=False, range=[0,1]),
            height=340, margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BENCHMARK RESULTS
# ════════════════════════════════════════════════════════════════════════════

elif "Benchmark" in page:
    st.markdown('<p class="section-header">Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-title">Benchmark Results</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">All 4 approaches tested against 10 library faults + 3 LLM-generated novel faults</p>', unsafe_allow_html=True)

    baseline_df = load_baseline()

    # Run evaluation button
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_eval = st.button("▶  Run Evaluation", type="primary", use_container_width=True)
    with col_status:
        if run_eval:
            with st.spinner("Running baseline_eval.py — this takes 15–25 minutes..."):
                result = subprocess.run(
                    [sys.executable, os.path.join(BASE_DIR, "evaluation", "baseline_eval.py")],
                    capture_output=True, cwd=BASE_DIR,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    encoding="utf-8", errors="replace"
)
            if result.returncode == 0:
                st.success("Evaluation complete! Refresh the page to see updated results.")
            else:
                st.error(f"Evaluation failed: {result.stderr[-500:] if result.stderr else 'unknown error'}")
            baseline_df = load_baseline()

    st.markdown("---")

    if baseline_df is None:
        st.info("No benchmark results found. Run the evaluation above, or run `python evaluation/baseline_eval.py` in your terminal first.")
        st.stop()

    # ── Compute summary ──────────────────────────────────────────────────────
    if "is_correct" in baseline_df.columns and "approach" in baseline_df.columns:
        # New format — full results per fault
        summary = (
            baseline_df.groupby("approach")
            .agg(accuracy=("is_correct","mean"), avg_latency=("latency","mean"), n=("is_correct","count"))
            .reset_index()
            .sort_values("accuracy", ascending=False)
        )
        has_full_data = True
    else:
        # Old format — just accuracy and latency per approach
        summary = baseline_df.rename(columns={"latency":"avg_latency"}).copy()
        summary["n"] = "—"
        has_full_data = False

    APPROACH_COLORS = {
        "LLM+RAG":    "#64ffda",
        "RAG-Only":   "#79b8ff",
        "Rule-Based": "#ffb86c",
        "LLM-Only":   "#bd93f9",
    }

    # ── Summary metric cards ─────────────────────────────────────────────────
    cols = st.columns(len(summary))
    for i, (_, row) in enumerate(summary.iterrows()):
        name    = row["approach"]
        acc     = row["accuracy"]
        lat     = row["avg_latency"]
        color   = APPROACH_COLORS.get(name, "#ccd6f6")
        with cols[i]:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{name}</div>
                <div class="value" style="color:{color}">{acc:.0%}</div>
                <div class="sub">avg {lat:.2f}s latency</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    col_chart1, col_chart2 = st.columns(2, gap="medium")

    with col_chart1:
        st.markdown("##### Accuracy by approach")
        fig_acc = go.Figure(go.Bar(
            x=summary["approach"],
            y=summary["accuracy"],
            marker_color=[APPROACH_COLORS.get(n, "#8892b0") for n in summary["approach"]],
            marker_line_width=0,
            text=[f"{v:.0%}" for v in summary["accuracy"]],
            textposition="outside",
            textfont=dict(color="#ccd6f6", size=13),
        ))
        fig_acc.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            yaxis=dict(tickformat=".0%", gridcolor="#2d3250",
                       range=[0, 1.15], color="#8892b0"),
            xaxis=dict(color="#8892b0"),
            margin=dict(l=10,r=10,t=10,b=10),
            height=280, showlegend=False,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_chart2:
        st.markdown("##### Average latency by approach")
        fig_lat = go.Figure(go.Bar(
            x=summary["approach"],
            y=summary["avg_latency"],
            marker_color=[APPROACH_COLORS.get(n, "#8892b0") for n in summary["approach"]],
            marker_line_width=0,
            text=[f"{v:.2f}s" for v in summary["avg_latency"]],
            textposition="outside",
            textfont=dict(color="#ccd6f6", size=13),
        ))
        fig_lat.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            yaxis=dict(gridcolor="#2d3250", color="#8892b0"),
            xaxis=dict(color="#8892b0"),
            margin=dict(l=10,r=10,t=10,b=10),
            height=280, showlegend=False,
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── Full results table ───────────────────────────────────────────────────
    if has_full_data:
        st.markdown("##### Per-fault breakdown")

        # Pivot table: fault vs approach
        pivot = baseline_df.pivot_table(
            index=["fault_id","fault_type","correct_action","source"],
            columns="approach",
            values="decision",
            aggfunc="first"
        ).reset_index()

        # Color decisions
        def style_decision(val, correct):
            if pd.isna(val):
                return "—"
            mark = "✅" if val == correct else "❌"
            color = action_color(val)
            return f'<span style="color:{color};font-weight:600">{val}</span> {mark}'

        html_rows = ""
        for _, row in pivot.iterrows():
            correct = row.get("correct_action", "")
            source_badge = (
                '<span style="color:#64ffda;font-size:0.7rem">★ novel</span>'
                if "llm" in str(row.get("source","")).lower()
                else '<span style="color:#8892b0;font-size:0.7rem">library</span>'
            )
            html_rows += f"""<tr>
                <td style="color:#64ffda;font-weight:700">{row.get('fault_id','')}</td>
                <td style="color:#8892b0">{row.get('fault_type','')}</td>
                <td style="color:{action_color(correct)};font-weight:700">{correct}</td>
                <td>{source_badge}</td>
                <td>{style_decision(row.get('Rule-Based'), correct)}</td>
                <td>{style_decision(row.get('LLM-Only'),  correct)}</td>
                <td>{style_decision(row.get('RAG-Only'),  correct)}</td>
                <td>{style_decision(row.get('LLM+RAG'),   correct)}</td>
            </tr>"""

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.83rem">
            <thead>
                <tr style="border-bottom:2px solid #2d3250">
                    <th style="text-align:left;padding:0.5rem;color:#64ffda">ID</th>
                    <th style="text-align:left;padding:0.5rem;color:#64ffda">Fault type</th>
                    <th style="text-align:left;padding:0.5rem;color:#64ffda">Correct</th>
                    <th style="text-align:left;padding:0.5rem;color:#64ffda">Source</th>
                    <th style="text-align:left;padding:0.5rem;color:#ffb86c">Rule-Based</th>
                    <th style="text-align:left;padding:0.5rem;color:#bd93f9">LLM-Only</th>
                    <th style="text-align:left;padding:0.5rem;color:#79b8ff">RAG-Only</th>
                    <th style="text-align:left;padding:0.5rem;color:#64ffda">LLM+RAG</th>
                </tr>
            </thead>
            <tbody>{html_rows}</tbody>
        </table>"""
        st.markdown(table_html, unsafe_allow_html=True)

        # Novel vs library comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Novel faults vs library faults  —  LLM+RAG")
        llm_rag_df = baseline_df[baseline_df["approach"] == "LLM+RAG"]
        if len(llm_rag_df) > 0:
            groups = llm_rag_df.groupby(
                llm_rag_df["source"].apply(
                    lambda s: "Novel" if "llm" in str(s).lower() or "fallback" in str(s).lower()
                    else "Library"
                )
            )["is_correct"].agg(["mean","count"]).reset_index()
            groups.columns = ["Source", "Accuracy", "N"]

            col_a, col_b, col_spacer = st.columns([1,1,2])
            for i, (_, row) in enumerate(groups.iterrows()):
                col = col_a if i == 0 else col_b
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <div class="label">{row['Source']} faults</div>
                        <div class="value" style="color:#64ffda">{row['Accuracy']:.0%}</div>
                        <div class="sub">n = {int(row['N'])}</div>
                    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE RUN HISTORY
# ════════════════════════════════════════════════════════════════════════════

elif "Live" in page:
    st.markdown('<p class="section-header">Live System</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-title">Run History</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Events logged by grid_controller.py — one row per task per run</p>', unsafe_allow_html=True)

    results_df = load_results()

    if results_df is None:
        st.info("No live run data found. Run `python grid_controller.py` (with agents running) to generate data.")
        st.stop()

    # ── Summary metrics ──────────────────────────────────────────────────────
    total    = len(results_df)
    success  = len(results_df[results_df["status"] == "SUCCESS"])
    failure  = len(results_df[results_df["status"] == "FAILURE"])
    avg_lat  = results_df["latency"].mean()

    correct_count = 0
    scored_count  = 0
    if "is_correct" in results_df.columns:
        scored = results_df[results_df["is_correct"].notna() & (results_df["is_correct"] != "")]
        scored_count  = len(scored)
        correct_count = int(scored["is_correct"].astype(float).sum())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total events</div>
            <div class="value">{total}</div>
            <div class="sub">across all runs</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Success rate</div>
            <div class="value" style="color:#50fa7b">{success/total:.0%}</div>
            <div class="sub">{success} of {total} tasks</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Failure rate</div>
            <div class="value" style="color:#ff5555">{failure/total:.0%}</div>
            <div class="sub">{failure} faults triggered</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        if scored_count > 0:
            rec_acc = correct_count / scored_count
            st.markdown(f"""<div class="metric-card">
                <div class="label">Recovery accuracy</div>
                <div class="value" style="color:#64ffda">{rec_acc:.0%}</div>
                <div class="sub">{correct_count}/{scored_count} correct</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="metric-card">
                <div class="label">Avg latency</div>
                <div class="value">{avg_lat:.2f}s</div>
                <div class="sub">per task</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Event log table ──────────────────────────────────────────────────────
    st.markdown("##### Event log")

    display_cols = ["timestamp","task","agent","status","fault_type",
                    "error","decision","correct_action","is_correct","latency"]
    show_cols = [c for c in display_cols if c in results_df.columns]

    # Build styled HTML table
    header_cells = "".join(
        f'<th style="text-align:left;padding:0.5rem 0.7rem;color:#64ffda;'
        f'border-bottom:2px solid #2d3250;white-space:nowrap">{c.replace("_"," ").title()}</th>'
        for c in show_cols
    )

    body_rows = ""
    for _, row in results_df.iterrows():
        cells = ""
        for c in show_cols:
            val = row.get(c, "")
            if c == "status":
                cls = "status-success" if val == "SUCCESS" else "status-failure"
                cells += f'<td style="padding:0.45rem 0.7rem"><span class="{cls}">{val}</span></td>'
            elif c == "decision" and val:
                color = action_color(str(val))
                cells += f'<td style="padding:0.45rem 0.7rem;color:{color};font-weight:600">{val}</td>'
            elif c == "is_correct":
                try:
                    is_empty = str(val) == "" or str(val) == "nan" or pd.isna(val)
                except Exception:
                    is_empty = False
                if is_empty:
                    cells += f'<td style="padding:0.45rem 0.7rem;color:#8892b0">—</td>'
                else:
                    mark = "✅" if str(val) == "1" or val == 1 else "❌"
                    cells += f'<td style="padding:0.45rem 0.7rem">{mark}</td>'
            elif c == "latency" and val:
                cells += f'<td style="padding:0.45rem 0.7rem;color:#8892b0">{float(val):.3f}s</td>'
            elif c == "error" and val:
                short = str(val)[:50] + ("…" if len(str(val)) > 50 else "")
                cells += f'<td style="padding:0.45rem 0.7rem;color:#ff5555;font-size:0.82rem">{short}</td>'
            else:
                cells += f'<td style="padding:0.45rem 0.7rem;color:#a8b2d8">{val}</td>'
        bg = "#1e2130" if _ % 2 == 0 else "#161823"
        body_rows += f'<tr style="background:{bg}">{cells}</tr>'

    table_html = f"""
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{body_rows}</tbody>
    </table>
    </div>"""
    st.markdown(table_html, unsafe_allow_html=True)

    # ── Latency chart ────────────────────────────────────────────────────────
    if "task" in results_df.columns and "latency" in results_df.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Latency per task")
        colors_per_status = results_df["status"].map(
            {"SUCCESS": "#50fa7b", "FAILURE": "#ff5555"}
        ).fillna("#8892b0")

        fig_lat = go.Figure(go.Bar(
            x=list(range(len(results_df))),
            y=results_df["latency"].astype(float),
            marker_color=colors_per_status,
            marker_line_width=0,
            hovertext=results_df["task"] + " — " + results_df["status"],
        ))
        fig_lat.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            yaxis=dict(title="seconds", gridcolor="#2d3250", color="#8892b0"),
            xaxis=dict(title="event index", color="#8892b0"),
            margin=dict(l=10,r=10,t=10,b=10),
            height=240,
        )
        st.plotly_chart(fig_lat, use_container_width=True)
        st.markdown('<span style="color:#50fa7b">█</span> Success &nbsp;&nbsp; <span style="color:#ff5555">█</span> Failure',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FAULT BROWSER
# ════════════════════════════════════════════════════════════════════════════

elif "Fault" in page:
    st.markdown('<p class="section-header">Reference</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-title">Fault Library</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">10 predefined fault scenarios used for benchmarking — each with a ground-truth correct action</p>', unsafe_allow_html=True)

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns([1,1,2])
    with col_f1:
        filter_action = st.selectbox("Filter by correct action",
                                     ["All", "RETRY", "REDIRECT", "SHUTDOWN"])
    with col_f2:
        filter_agent = st.selectbox("Filter by agent",
                                    ["All", "Plant1", "Transmission1", "Substation1"])

    filtered = [
        f for f in FAULT_LIBRARY
        if (filter_action == "All" or f["correct"] == filter_action)
        and (filter_agent == "All" or f["agent"] == filter_agent)
    ]

    st.markdown(f"<p style='color:#8892b0;font-size:0.82rem'>Showing {len(filtered)} of {len(FAULT_LIBRARY)} faults</p>",
                unsafe_allow_html=True)

    # Action distribution
    action_counts = {"RETRY": 0, "REDIRECT": 0, "SHUTDOWN": 0}
    for f in filtered:
        action_counts[f["correct"]] += 1

    col_r, col_rd, col_s, col_spacer = st.columns([1,1,1,3])
    for col, (action, count) in zip([col_r, col_rd, col_s], action_counts.items()):
        color = action_color(action)
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{action}</div>
                <div class="value" style="color:{color}">{count}</div>
                <div class="sub">faults</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Fault cards — two columns
    col_left, col_right = st.columns(2, gap="medium")
    for i, fault in enumerate(filtered):
        col = col_left if i % 2 == 0 else col_right
        action_cls = f"action-{fault['correct'].lower()}"
        with col:
            st.markdown(f"""
            <div class="fault-card">
                <div>
                    <span class="fault-id">{fault['id']}</span>
                    &nbsp;·&nbsp;
                    <span class="fault-type">{fault['type']}</span>
                    &nbsp;·&nbsp;
                    <span style="color:#8892b0;font-size:0.75rem">{fault['agent']}</span>
                </div>
                <div class="fault-err" style="margin:0.4rem 0">{fault['error']}</div>
                <div style="font-size:0.8rem">
                    Correct action:
                    <span class="{action_cls}">{fault['correct']}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # Decision guide
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Decision guide")
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("""<div class="fault-card" style="border-color:#79b8ff44">
            <div style="color:#79b8ff;font-weight:700;font-size:1rem;margin-bottom:0.5rem">🔄 RETRY</div>
            <div style="color:#a8b2d8;font-size:0.83rem;line-height:1.6">
            Fault is likely <b>transient</b> — it self-clears once the section is de-energized.
            Re-send the same task to the same agent after a short wait.<br><br>
            <i>Examples: voltage sag, frequency deviation, phase imbalance, ground fault</i>
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="fault-card" style="border-color:#64ffda44">
            <div style="color:#64ffda;font-weight:700;font-size:1rem;margin-bottom:0.5rem">↪️ REDIRECT</div>
            <div style="color:#a8b2d8;font-size:0.83rem;line-height:1.6">
            Primary path is <b>blocked</b> but function can be performed via an alternate route.
            Send the task to a different agent or reroute power flows.<br><br>
            <i>Examples: line overload, line disconnect, partial load loss</i>
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="fault-card" style="border-color:#ff555544">
            <div style="color:#ff5555;font-weight:700;font-size:1rem;margin-bottom:0.5rem">🛑 SHUTDOWN</div>
            <div style="color:#a8b2d8;font-size:0.83rem;line-height:1.6">
            Fault is <b>severe or unsafe</b> — continuing risks equipment damage or cascading failure.
            Halt the pipeline and wait for physical inspection.<br><br>
            <i>Examples: transformer overheating, breaker failure, turbine emergency trip</i>
            </div>
        </div>""", unsafe_allow_html=True)
