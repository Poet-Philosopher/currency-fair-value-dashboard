"""
app.py: Currency Fair Value Dashboard (Streamlit + Plotly)

Run with:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data_engine as de

st.set_page_config(page_title="Currency fair value", page_icon="💱", layout="wide")

SPOT_COLOR = "#2563eb"
FAIR_COLOR = "#d97706"
OVER_COLOR = "#dc2626"
UNDER_COLOR = "#16a34a"
GRID_GREY = "rgba(100,116,139,0.55)"

FILL_LABELS = {
    "Step (forward-fill each CPI print)": "ffill",
    "Smooth (linear interpolation between prints)": "interpolate",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fmt_rate(x: float) -> str:
    """JPY-style rates get 2 decimals, majors like EUR/USD get 4."""
    return f"{x:,.2f}" if x >= 50 else f"{x:,.4f}"


def show_chart(fig: go.Figure) -> None:
    """Render a Plotly figure full width across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit without the `width` argument
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Fetching FX and CPI data...")
def load_data(pair_label, start, end, base_year, fill_method, base_cpi_id, quote_cpi_id, extend_stale):
    return de.build_dataset(
        pair_label, start, end, base_year, fill_method,
        base_cpi_id or None, quote_cpi_id or None, extend_stale_cpi=extend_stale,
    )


def build_figure(result: de.PPPResult, show_bands: bool) -> go.Figure:
    df = result.data
    base, quote = result.pair.base, result.pair.quote
    dec = 2 if df["spot"].median() >= 50 else 4
    strong = de.STRONG_PCT

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.68, 0.32],
    )

    # --- Row 1: spot vs PPP fair value ---------------------------------------
    if show_bands:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["fair_value"] * (1 + strong / 100), mode="lines",
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["fair_value"] * (1 - strong / 100), mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor="rgba(100,116,139,0.14)",
            hoverinfo="skip", name=f"PPP ±{strong:.0f}% band",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["spot"], mode="lines", name="Spot rate",
        line=dict(color=SPOT_COLOR, width=1.6),
        hovertemplate=f"Spot: %{{y:,.{dec}f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["fair_value"], mode="lines", name="PPP fair value",
        line=dict(color=FAIR_COLOR, width=2.2, dash="dash"),
        hovertemplate=f"PPP fair value: %{{y:,.{dec}f}}<extra></extra>",
    ), row=1, col=1)

    # --- Row 2: % deviation oscillator ---------------------------------------
    dev = df["deviation_pct"]
    fig.add_trace(go.Scatter(
        x=df.index, y=dev.clip(lower=0), mode="lines", name=f"{base} overvalued",
        line=dict(color=OVER_COLOR, width=0.8), fill="tozeroy", fillcolor="rgba(220,38,38,0.30)",
        hovertemplate="Overvalued: %{y:+.1f}%<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=dev.clip(upper=0), mode="lines", name=f"{base} undervalued",
        line=dict(color=UNDER_COLOR, width=0.8), fill="tozeroy", fillcolor="rgba(22,163,74,0.30)",
        hovertemplate="Undervalued: %{y:+.1f}%<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color=GRID_GREY, width=1), row=2, col=1)
    fig.add_hline(
        y=float(dev.mean()), line=dict(color=GRID_GREY, width=1, dash="dash"),
        annotation_text=f"avg {dev.mean():+.1f}%", annotation_position="top left",
        row=2, col=1,
    )
    if show_bands:
        for level in (strong, -strong):
            fig.add_hline(y=level, line=dict(color=GRID_GREY, width=1, dash="dot"), row=2, col=1)

    fig.update_xaxes(
        showspikes=True, spikemode="across", spikethickness=1, spikecolor=GRID_GREY,
        rangeselector=dict(buttons=[
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        row=1, col=1,
    )
    fig.update_yaxes(title_text=f"{quote} per 1 {base}", row=1, col=1)
    fig.update_yaxes(title_text="Spot vs fair value", ticksuffix="%", row=2, col=1)
    fig.update_layout(
        height=760, hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"),
    )
    return fig


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
today = dt.date.today()

with st.sidebar:
    st.header("Settings")

    preset = st.selectbox("Currency pair", de.PRESET_PAIRS + ["Custom pair"])
    if preset == "Custom pair":
        ccys = sorted(de.CPI_SERIES)
        base_ccy = st.selectbox("Base currency", ccys, index=ccys.index("EUR"))
        quote_ccy = st.selectbox("Quote currency", ccys, index=ccys.index("USD"))
        if base_ccy == quote_ccy:
            st.error("Pick two different currencies.")
            st.stop()
        pair_label = f"{base_ccy}/{quote_ccy}"
    else:
        pair_label = preset
    st.caption("Quoted as units of the second currency per 1 unit of the first.")

    base_year = st.slider(
        "PPP base year", min_value=de.MIN_BASE_YEAR, max_value=today.year - 1, value=2015,
        help="Fair value is anchored to the average spot rate in this year, i.e. "
             "the year you assume the pair was at equilibrium.",
    )

    date_range = st.date_input(
        "Date range",
        value=(dt.date(today.year - 10, 1, 1), today),
        min_value=dt.date(de.MIN_BASE_YEAR, 1, 1),
        max_value=today,
    )

    fill_choice = st.radio(
        "Monthly to daily alignment", list(FILL_LABELS),
        help="CPI is published monthly (some countries quarterly); FX trades daily.",
    )
    extend_stale = st.checkbox(
        "Estimate CPI not yet published", value=True,
        help="Some CPI series stop months or years before today. When on, the missing months "
             "are estimated from that country's trailing 12-month inflation instead of "
             "holding the last CPI level flat.",
    )
    show_bands = st.checkbox(f"Show ±{de.STRONG_PCT:.0f}% bands", value=True)

    with st.expander("Advanced"):
        key_source = de.fred_key_source()
        st.caption(
            f"FRED API key: found in {key_source}." if key_source
            else "FRED API key: not set. Using FRED's public endpoints instead."
        )
        st.caption("Override the FRED series used for CPI (index levels only).")
        base_cpi_override = st.text_input("Base currency CPI series ID", value="")
        quote_cpi_override = st.text_input("Quote currency CPI series ID", value="")
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Select both a start and an end date in the sidebar.")
    st.stop()
start_date, end_date = date_range
if start_date >= end_date:
    st.error("The start date must be before the end date.")
    st.stop()
if base_year > end_date.year:
    st.error("The base year cannot be later than the end of the date range.")
    st.stop()

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
try:
    result = load_data(
        pair_label, start_date, end_date, base_year, FILL_LABELS[fill_choice],
        base_cpi_override.strip(), quote_cpi_override.strip(), extend_stale,
    )
except de.DataUnavailableError as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # network failures, unexpected upstream changes
    st.error("Unexpected error while loading data.")
    st.exception(exc)
    st.stop()

sig = de.compute_signal(result)
df = result.data
base, quote = result.pair.base, result.pair.quote
last = df.iloc[-1]

# ----------------------------------------------------------------------------
# Header, signal, KPI cards
# ----------------------------------------------------------------------------
st.title(f"{result.pair.label} fair value")
st.caption(
    f"Relative purchasing power parity, anchored to the {result.base_year} average rate. "
    f"Data as of {sig.as_of:%d %b %Y}."
)

st.markdown(
    f"""
    <div style="padding:0.9rem 1.1rem;border-radius:0.5rem;border-left:6px solid {sig.color};
                background:{sig.color}1a;margin-bottom:1rem;">
      <div style="font-size:1.4rem;font-weight:700;line-height:1.3;">{sig.headline}</div>
      <div style="opacity:0.75;">Spot {fmt_rate(last.spot)} against a PPP fair value of
      {fmt_rate(last.fair_value)} ({quote} per 1 {base}).</div>
    </div>
    """,
    unsafe_allow_html=True,
)

one_day = df["spot"].pct_change().iloc[-1] * 100 if len(df) > 1 else 0.0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current spot", fmt_rate(last.spot), f"{one_day:+.2f}% 1d", delta_color="off")
c2.metric(
    "PPP fair value", fmt_rate(last.fair_value),
    help=f"Base-year ({result.base_year}) average spot was {fmt_rate(result.meta['spot_base_year_avg'])}, "
         "adjusted by the inflation differential since then.",
)
c3.metric(
    "Deviation from fair value", f"{sig.deviation_pct:+.1f}%", sig.label, delta_color="off",
    help="(Spot / fair value - 1). Positive means the base currency is overvalued.",
)
c4.metric(
    "Percentile in window", f"{sig.percentile:.0f}th",
    help="Share of days in the selected range with a lower deviation than today.",
)

# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
show_chart(build_figure(result, show_bands))

# ----------------------------------------------------------------------------
# Data quality notes, methodology, raw data
# ----------------------------------------------------------------------------
meta = result.meta
cpi_last = meta["cpi_last_obs"]
extended = meta.get("cpi_extended", {})
notes, very_stale = [], False
for side, ccy in (("base", base), ("quote", quote)):
    gap = (df.index[-1] - cpi_last[side]).days
    if gap <= de.LAG_TOLERANCE_DAYS:
        continue  # ordinary publication lag
    very_stale = very_stale or gap > de.STALE_CPI_DAYS
    if side in extended:
        info = extended[side]
        tail = f", for the first {de.MAX_EXTRAPOLATION_DAYS // 365} years and flat after that" if info["capped"] else ""
        notes.append(
            f"**{ccy}** CPI ends in {cpi_last[side]:%b %Y}. The months since are estimated from its "
            f"trailing 12-month inflation ({info['annual_rate_pct']:+.1f}% a year{tail}), not published data."
        )
    else:
        notes.append(f"**{ccy}** CPI ends in {cpi_last[side]:%b %Y}; fair value after that holds the last level flat.")
if notes:
    (st.warning if very_stale else st.info)("  \n".join(notes) + "  \nRecent fair value is approximate.")
if min(meta["cpi_obs_in_base_year"].values()) < 4:
    st.warning(
        f"One of the CPI series has fewer than four observations in {result.base_year}, "
        "so the base-year average may be unrepresentative."
    )

with st.expander("Methodology"):
    st.latex(
        r"S^{*}_t = S_0 \times "
        r"\frac{CPI^{quote}_t / CPI^{quote}_0}{CPI^{base}_t / CPI^{base}_0}"
    )
    st.markdown(
        f"""
- **S₀** is the average spot rate during {result.base_year}; **CPI₀** is each country's average CPI in that year.
- If the quote country's prices rise faster than the base country's, fair value (quote per base) rises.
- **Deviation** = Spot / S* − 1. Positive means the base currency ({base}) is overvalued against {quote}; negative means undervalued.
- CPI series used: {base}: `{meta['cpi_series']['base']}` (latest {cpi_last['base']:%b %Y}), {quote}: `{meta['cpi_series']['quote']}` (latest {cpi_last['quote']:%b %Y}), from FRED.
- PPP is a long-run anchor. Misalignments can last for years and are not a trading signal on their own. This is not investment advice.
        """
    )

with st.expander("Data"):
    table = df.rename(columns={
        "spot": "Spot", "fair_value": "PPP fair value", "deviation_pct": "Deviation %",
        "cpi_base": f"{base} CPI (base year = 1)", "cpi_quote": f"{quote} CPI (base year = 1)",
    })
    st.dataframe(table.round(4).sort_index(ascending=False))
    st.download_button(
        "Download CSV", table.to_csv().encode("utf-8"),
        file_name=f"{result.pair.base}{result.pair.quote}_ppp_{result.base_year}.csv",
        mime="text/csv",
    )
