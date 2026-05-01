"""
HTS Classifier — Streamlit Web UI
---------------------------------
A trade compliance demo tool: enter a product description, get suggested
HTS codes with confidence scoring, official chapter links, and an audit trail.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import io
from pathlib import Path

from classifier import classify, chapter_notes_url, get_chapter
from database import init_db, save_classification, get_all_classifications

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HTS Classifier",
    page_icon="📦",
    layout="wide",
)

# Initialize the audit trail database on first run
init_db()

# ---------------------------------------------------------------------------
# Data loading (cached so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_hts_data():
    """Load the processed HTSUS data with hierarchical descriptions."""
    csv_path = Path(__file__).parent / "hts_processed.csv"
    return pd.read_csv(csv_path)

df = load_hts_data()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📦 HTS Classification Suggester")
st.markdown(
    "Enter a product description in plain English to get suggested "
    "**Harmonized Tariff Schedule** codes with confidence scoring. "
    "Designed for use as a first-pass classification aid in trade compliance workflows."
)

st.warning(
    "⚠️  **For demonstration only.** Final HTS classification requires "
    "application of the General Rules of Interpretation (GRI) and review "
    "of Section/Chapter Notes by a qualified classifier. Always verify "
    "against the official HTSUS at hts.usitc.gov."
)

# ---------------------------------------------------------------------------
# Sidebar — stats + audit trail
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Database")
    st.metric("HTS codes loaded", f"{len(df):,}")

    saved = get_all_classifications()
    st.metric("Saved classifications", len(saved))

    st.divider()
    st.header("Audit Trail")
    if saved:
        if st.button("📥 Export all as CSV"):
            df_export = pd.DataFrame(saved)
            csv_buf = io.StringIO()
            df_export.to_csv(csv_buf, index=False)
            st.download_button(
                "Download classifications.csv",
                data=csv_buf.getvalue(),
                file_name="classifications_audit_trail.csv",
                mime="text/csv",
            )

        st.caption("Recent classifications:")
        for entry in saved[:5]:
            with st.container():
                st.markdown(
                    f"**{entry['chosen_hts_code']}** — "
                    f"_{entry['product_description'][:40]}_"
                )
                st.caption(
                    f"{entry['timestamp'][:10]} · "
                    f"{entry['confidence']} · "
                    f"{'top pick' if entry['was_top_suggestion'] else 'overridden'}"
                )
    else:
        st.caption("No classifications saved yet. Run a search and save one.")

# ---------------------------------------------------------------------------
# Main search area
# ---------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Product description",
        placeholder="e.g., men's cotton t-shirt, leather women's handbag, lithium-ion battery for laptop",
        key="query_input",
    )
with col2:
    top_n = st.selectbox("Suggestions", [5, 10, 20], index=1)

if query:
    with st.spinner("Searching HTSUS..."):
        results = classify(query, df, top_n=top_n)

    if not results:
        st.error("No matches found. Try a different description or include more specific terms (material, gender, function).")
    else:
        st.success(f"Found {len(results)} candidate codes for: **{query}**")

        # Top suggestion gets a featured display
        top = results[0]
        st.subheader("🎯 Top suggestion")
        with st.container(border=True):
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.markdown(f"### `{top['hts_code']}`")
                st.markdown(f"**{top['description']}**")
            with cols[1]:
                conf_color = {
                    "HIGH": "🟢",
                    "MEDIUM": "🟡",
                    "LOW": "🟠",
                    "VERY LOW": "🔴",
                }.get(top['confidence'], "⚪")
                st.metric("Confidence", f"{conf_color} {top['confidence']}")
                st.metric("Score", f"{top['score']}/100")
            with cols[2]:
                duty = top['duty_general'] if top['duty_general'] != 'nan' else "(see parent code)"
                st.metric("General duty", duty)
                if top['duty_column2'] and top['duty_column2'] != 'nan':
                    st.caption(f"Column 2 (non-NTR): {top['duty_column2']}")

            with st.expander("Why this code? (matching reasoning)"):
                if top['reasons']:
                    for reason in top['reasons']:
                        st.markdown(f"- {reason}")
                else:
                    st.caption("No specific attribute matches; ranked on text similarity only.")

            chapter = get_chapter(top['hts_code'])
            st.markdown(
                f"📖 [View Chapter {chapter} on official HTSUS →]({chapter_notes_url(top['hts_code'])})"
            )

            # Save form
            with st.form("save_top"):
                notes = st.text_area("Classification notes (optional)", placeholder="e.g., Verified material composition with supplier; GRI 1 applies.")
                submitted = st.form_submit_button("💾 Save top suggestion to audit trail")
                if submitted:
                    save_classification(
                        product_description=query,
                        chosen_hts_code=top['hts_code'],
                        chosen_description=top['description'],
                        confidence=top['confidence'],
                        score=top['score'],
                        top_suggestion_code=top['hts_code'],
                        notes=notes,
                    )
                    st.success(f"Saved {top['hts_code']} to audit trail.")
                    st.rerun()

        # Other candidates
        if len(results) > 1:
            st.subheader("Other candidates")
            for i, r in enumerate(results[1:], start=2):
                with st.expander(f"#{i}  `{r['hts_code']}`  —  {r['confidence']} ({r['score']}/100)"):
                    st.markdown(f"**{r['description']}**")
                    cols = st.columns(3)
                    with cols[0]:
                        duty = r['duty_general'] if r['duty_general'] != 'nan' else "(see parent code)"
                        st.markdown(f"**General duty:** {duty}")
                    with cols[1]:
                        if r['duty_column2'] and r['duty_column2'] != 'nan':
                            st.markdown(f"**Column 2:** {r['duty_column2']}")
                    with cols[2]:
                        if r['unit'] and r['unit'] != 'nan':
                            st.markdown(f"**Unit:** {r['unit']}")

                    if r['reasons']:
                        st.caption("Reasons: " + " · ".join(r['reasons']))

                    chapter = get_chapter(r['hts_code'])
                    st.markdown(f"📖 [Chapter {chapter} reference →]({chapter_notes_url(r['hts_code'])})")

                    # Override save — track when users pick something other than the top
                    if st.button(f"Save as override (override top suggestion)", key=f"save_{i}"):
                        save_classification(
                            product_description=query,
                            chosen_hts_code=r['hts_code'],
                            chosen_description=r['description'],
                            confidence=r['confidence'],
                            score=r['score'],
                            top_suggestion_code=top['hts_code'],
                            notes=f"User overrode top suggestion ({top['hts_code']}).",
                        )
                        st.success(f"Saved {r['hts_code']} as override.")
                        st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built as a trade compliance portfolio project. Data source: "
    "U.S. International Trade Commission HTSUS. "
    "[hts.usitc.gov](https://hts.usitc.gov) · "
    "Classifications are suggestions only and do not constitute legal advice."
)
