"""
Digital Pavement Condition Evaluation and Maintenance Decision Tool
TCG633 - Bridge and Road Maintenance | Individual Project

Run with:  streamlit run app.py
"""

import io
import base64
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pavement_logic import (
    DEFAULT_DEFECT_WEIGHTS, DEFAULT_SEVERITY_FACTORS,
    DEFAULT_PCI_BANDS, DEFAULT_IRI_BANDS, DEFAULT_COST_PER_100M,
    compute_pci, compute_iri, compute_hybrid,
    bands_to_dataframe, dataframe_to_bands,
    add_hybrid_reasoning, estimate_costs, cost_scenario, simulate_maintenance,
)

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pavement Condition Evaluation Tool",
    page_icon="🛣️",
    layout="wide",
)

CONDITION_COLORS = {
    "Very Good": "#2E7D32",
    "Good": "#9E9D24",
    "Fair": "#F57C00",
    "Poor": "#C62828",
}


def natural_sort_key(s):
    """Sort strings naturally: 1,2,3...10 instead of 1,10,2,3."""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    .condition-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-weight: 600; font-size: 0.85rem; color: white;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    h1, h2, h3 { font-family: 'Source Sans Pro', sans-serif; }
</style>
""", unsafe_allow_html=True)


def colored_condition_table(df: pd.DataFrame, cond_cols: list) -> "pd.io.formats.style.Styler":
    """Apply background color to condition columns for visual scanning."""
    def style_cond(val):
        base = str(val).split(" ")[0] if val else ""
        for k in CONDITION_COLORS:
            if str(val).startswith(k):
                color = CONDITION_COLORS[k]
                return f"background-color: {color}22; color: {color}; font-weight: 600;"
        return ""
    sty = df.style
    for c in cond_cols:
        if c in df.columns:
            sty = sty.map(style_cond, subset=[c])
    return sty


# ---------------------------------------------------------------------------
# Sidebar: data source, mode, lookup editing
# ---------------------------------------------------------------------------
st.sidebar.title("🛣️ Tool Controls")

st.sidebar.subheader("1. Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Use built-in sample dataset", "Upload my own file"],
    index=0,
)

uploaded_file = None
if data_source == "Upload my own file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel (.xlsx) with 'PCI_Input' and 'IRI_Input' sheets, or two CSVs",
        type=["xlsx", "csv"],
    )

st.sidebar.subheader("2. Evaluation Mode")
mode = st.sidebar.radio("Select mode", ["PCI", "IRI", "Hybrid (PCI + IRI)"], index=2)

with st.sidebar.expander("3. Edit Lookup Tables (advanced)"):
    st.caption("Adjust weighting/severity factors and condition bands to match your standard.")
    edit_lookup = st.checkbox("Enable lookup editing", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("TCG633 Bridge & Road Maintenance — Individual Project")
st.sidebar.caption("Digital Pavement Condition Evaluation and Maintenance Decision Tool")


# ---------------------------------------------------------------------------
# Load default sample data
# ---------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    pci = pd.read_csv("sample_data/pci_input.csv")
    iri = pd.read_csv("sample_data/iri_input.csv")
    return pci, iri


def _find_header_row(xls, sheet_name, key_col="Section ID", scan_rows=10):
    """Find the row index containing the real column headers, since our
    dataset sheets have title/subtitle rows above the header."""
    preview = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=scan_rows)
    for i, row in preview.iterrows():
        if key_col in row.values:
            return i
    return 0


def load_uploaded_data(file):
    if file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(file)
        pci_header = _find_header_row(xls, "PCI_Input")
        iri_header = _find_header_row(xls, "IRI_Input")
        pci = pd.read_excel(xls, sheet_name="PCI_Input", header=pci_header)
        iri = pd.read_excel(xls, sheet_name="IRI_Input", header=iri_header)
        # Drop fully-blank rows; keep Section ID as string
        pci = pci[pci["Section ID"].notna()]
        iri = iri[iri["Section ID"].notna()]
        pci["Section ID"] = pci["Section ID"].astype(str)
        iri["Section ID"] = iri["Section ID"].astype(str)
        return pci, iri
    else:
        st.sidebar.warning("CSV upload: please upload PCI_Input first, then IRI_Input separately below.")
        return None, None


if data_source == "Upload my own file" and uploaded_file is not None:
    pci_input_raw, iri_input_raw = load_uploaded_data(uploaded_file)
    if pci_input_raw is None:
        st.stop()
else:
    pci_input_raw, iri_input_raw = load_default_data()

if "pci_input" not in st.session_state:
    st.session_state.pci_input = pci_input_raw.copy()
if "iri_input" not in st.session_state:
    st.session_state.iri_input = iri_input_raw.copy()

# Ensure Notes/Photo columns always exist (old datasets may only have "Notes / Photo Ref")
if "Notes / Photo Ref" in st.session_state.pci_input.columns and "Notes" not in st.session_state.pci_input.columns:
    st.session_state.pci_input = st.session_state.pci_input.rename(columns={"Notes / Photo Ref": "Notes"})
if "Notes" not in st.session_state.pci_input.columns:
    st.session_state.pci_input["Notes"] = ""
if "Photo" not in st.session_state.pci_input.columns:
    st.session_state.pci_input["Photo"] = ""

# Reset session data if a new file is uploaded
if data_source == "Upload my own file" and uploaded_file is not None:
    if st.sidebar.button("Reload uploaded data"):
        st.session_state.pci_input = pci_input_raw.copy()
        st.session_state.iri_input = iri_input_raw.copy()

st.sidebar.subheader("Data Management")
dm1, dm2 = st.sidebar.columns(2)
with dm1:
    if st.button("🗑️ Clear All Data", use_container_width=True,
                  help="Empty the input tables — useful for demoing data entry from scratch."):
        st.session_state.pci_input = pd.DataFrame(
            columns=["Section ID", "Defect Type", "Severity", "Area Affected (%)", "Notes", "Photo"]
        )
        st.session_state.iri_input = pd.DataFrame(
            columns=["Section ID", "IRI (m/km)", "Notes"]
        )
        st.rerun()
with dm2:
    if st.button("↩️ Restore Sample Data", use_container_width=True,
                  help="Bring back the original 10-section dataset."):
        st.session_state.pci_input = pci_input_raw.copy()
        st.session_state.iri_input = iri_input_raw.copy()
        st.rerun()


# ---------------------------------------------------------------------------
# Lookup table state
# ---------------------------------------------------------------------------
if "defect_weights" not in st.session_state:
    st.session_state.defect_weights = DEFAULT_DEFECT_WEIGHTS.copy()
if "severity_factors" not in st.session_state:
    st.session_state.severity_factors = DEFAULT_SEVERITY_FACTORS.copy()
if "pci_bands" not in st.session_state:
    st.session_state.pci_bands = DEFAULT_PCI_BANDS.copy()
if "iri_bands" not in st.session_state:
    st.session_state.iri_bands = DEFAULT_IRI_BANDS.copy()

if edit_lookup:
    with st.sidebar.expander("Defect Weighting Factors", expanded=False):
        for k in list(st.session_state.defect_weights.keys()):
            st.session_state.defect_weights[k] = st.number_input(
                k, value=float(st.session_state.defect_weights[k]), step=0.1, key=f"w_{k}"
            )
    with st.sidebar.expander("Severity Factors", expanded=False):
        for k in list(st.session_state.severity_factors.keys()):
            st.session_state.severity_factors[k] = st.number_input(
                k, value=float(st.session_state.severity_factors[k]), step=0.1, key=f"s_{k}"
            )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛣️ Digital Pavement Condition Evaluation and Maintenance Decision Tool")
st.caption(
    "District JKR Maintenance Division — Secondary Road Network Assessment "
    "| PCI (ASTM D6433-based) & IRI (Roughness) Evaluation"
)

tab_input, tab_results, tab_dashboard, tab_planning, tab_about = st.tabs(
    ["📥 Data Input", "📊 Computation & Results", "📈 Dashboard", "🛠️ Planning Tools", "ℹ️ Methodology"]
)

# ---------------------------------------------------------------------------
# TAB 1: Data Input
# ---------------------------------------------------------------------------
with tab_input:
    st.subheader("Pavement Condition Input Data")
    st.caption(
        "Each entry submits **both** a PCI defect observation and an IRI roughness reading "
        "for the same section together. Add multiple defects or IRI readings per section by submitting again."
    )

    # =======================================================================
    # SECTION A: Combined PCI + IRI form — PRIMARY WORKFLOW
    # =======================================================================
    st.markdown("### ➕ Add a Section Entry")

    existing_pci_sections = list(set(
        st.session_state.pci_input["Section ID"].dropna().astype(str).tolist()
    ))
    existing_iri_sections = list(set(
        st.session_state.iri_input["Section ID"].dropna().astype(str).tolist()
    ))
    all_existing_sections = sorted(
        set(existing_pci_sections + existing_iri_sections),
        key=natural_sort_key
    )

    # --- Section selection OUTSIDE the form so it persists correctly ---
    st.markdown("#### 📍 Section Identification")
    NEW_SECTION_SENTINEL = "＋ Type a new section name..."
    section_options = all_existing_sections + [NEW_SECTION_SENTINEL]

    selected_option = st.selectbox(
        "Section Name / ID",
        options=section_options,
        index=len(section_options) - 1 if not all_existing_sections else 0,
        key="section_selector",
        help="Pick an existing section, or choose '＋ Type a new section name...' to add a new one",
    )

    if selected_option == NEW_SECTION_SENTINEL:
        f_section_outer = st.text_input(
            "New section name",
            placeholder="e.g. Jalan Pinang Jawa, Section B, KM5",
            key="new_section_name_input",
        )
    else:
        f_section_outer = selected_option
        st.caption(f"Adding data to existing section: **{f_section_outer}**")

    st.markdown("---")

    with st.form("combined_entry_form", clear_on_submit=True):
        st.markdown("#### 🔍 PCI — Defect Observation")
        p1, p2, p3 = st.columns(3)
        with p1:
            f_defect = st.selectbox("Defect Type", list(st.session_state.defect_weights.keys()))
        with p2:
            f_severity = st.selectbox(
                "Severity", list(st.session_state.severity_factors.keys()), index=1
            )
        with p3:
            f_area = st.slider(
                "Area Affected (%)",
                min_value=0.0, max_value=100.0, value=5.0, step=0.5,
                help="Drag to set value. You can also click on the number shown above the slider to type a precise value directly.",
            )
        f_notes = st.text_input("PCI Notes (optional)", "")
        f_photo = st.file_uploader(
            "📷 Attach a defect photo (optional)", type=["png", "jpg", "jpeg"],
            key="pci_photo_upload"
        )

        st.markdown("---")
        st.markdown("#### 📏 IRI — Roughness Reading")
        st.caption(
            "Enter one IRI reading per submission. Submit again with the same section to add more — "
            "the app will average all readings for that section automatically."
        )
        i1, i2 = st.columns(2)
        with i1:
            g_iri = st.slider(
                "IRI (m/km)",
                min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                help="Drag to set value. Click the number shown above the slider to type precisely. For values above 10, type directly. Typical range: 1–6 m/km",
            )
        with i2:
            g_iri_notes = st.text_input("IRI Notes (optional)", "", key="iri_notes_field")

        submitted = st.form_submit_button(
            "✅ Submit Entry", use_container_width=True, type="primary"
        )

        if submitted:
            section_val = f_section_outer.strip() if isinstance(f_section_outer, str) else str(f_section_outer)
            if not section_val or section_val in ("", "None"):
                st.error("Please enter a Section Name / ID before submitting.")
            else:
                # Store photo separately to avoid bloating the dataframe
                # (large base64 strings in a dataframe get copied on every rerender → memory crash)
                photo_key = ""
                if f_photo is not None:
                    if "photo_store" not in st.session_state:
                        st.session_state.photo_store = {}
                    photo_key = f"{section_val}_{f_defect}_{len(st.session_state.photo_store)}"
                    img_bytes = f_photo.getvalue()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    mime = "image/png" if f_photo.type == "image/png" else "image/jpeg"
                    st.session_state.photo_store[photo_key] = f"data:{mime};base64,{img_b64}"

                pci_row = pd.DataFrame([{
                    "Section ID": section_val,
                    "Defect Type": f_defect,
                    "Severity": f_severity,
                    "Area Affected (%)": f_area,
                    "Notes": f_notes,
                    "Photo": photo_key,  # just a small key string, not the image itself
                }])
                st.session_state.pci_input = pd.concat(
                    [st.session_state.pci_input, pci_row], ignore_index=True
                )
                # Add IRI record
                iri_row = pd.DataFrame([{
                    "Section ID": section_val,
                    "IRI (m/km)": g_iri,
                    "Notes": g_iri_notes,
                }])
                st.session_state.iri_input = pd.concat(
                    [st.session_state.iri_input, iri_row], ignore_index=True
                )
                st.success(
                    f"✅ Added to **{section_val}**: "
                    f"{f_defect} ({f_severity}, {f_area}%) + IRI {g_iri} m/km"
                    + (" 📷" if photo_key else "")
                )

    st.divider()

    # =======================================================================
    # SECTION B: Review & edit (table view)
    # =======================================================================
    st.markdown("##### 📋 Current Data")
    st.caption("Edit values or delete rows directly in the tables below.")

    st.markdown("**PCI — Defect Records**")
    if st.session_state.pci_input.empty:
        st.info("No PCI records yet. Add your first entry using the form above.")
    else:
        # Resolve photo keys to data URIs just for display — don't store URIs in dataframe
        display_pci = st.session_state.pci_input.copy()
        photo_store = st.session_state.get("photo_store", {})
        display_pci["Photo"] = display_pci["Photo"].map(
            lambda k: photo_store.get(k, "") if k else ""
        )
        edited = st.data_editor(
            display_pci,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Defect Type": st.column_config.SelectboxColumn(
                    options=list(st.session_state.defect_weights.keys())
                ),
                "Severity": st.column_config.SelectboxColumn(
                    options=list(st.session_state.severity_factors.keys())
                ),
                "Area Affected (%)": st.column_config.NumberColumn(
                    min_value=0, max_value=100, step=0.1,
                    help="Click to select, then type directly"
                ),
                "Photo": st.column_config.ImageColumn("Photo", help="Defect photo, if attached"),
            },
            key="pci_editor",
        )
        # Write edits back but restore photo keys (not URIs) to keep session state lean
        if edited is not None:
            edited["Photo"] = st.session_state.pci_input["Photo"].values[:len(edited)] if len(edited) <= len(st.session_state.pci_input) else edited["Photo"].map(lambda v: "" if v and v.startswith("data:") else v)
            st.session_state.pci_input = edited

    st.markdown("**IRI — Roughness Readings**")
    if st.session_state.iri_input.empty:
        st.info("No IRI readings yet. Add your first entry using the form above.")
    else:
        st.session_state.iri_input = st.data_editor(
            st.session_state.iri_input,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "IRI (m/km)": st.column_config.NumberColumn(
                    min_value=0, max_value=20, step=0.1,
                    help="Click to select, then type directly"
                ),
            },
            key="iri_editor",
        )

    # =======================================================================
    # SECTION C: Upload data from a file (secondary)
    # =======================================================================
    with st.expander("📤 Upload data from a file instead", expanded=False):
        st.caption(
            "Upload a CSV/Excel to replace the current data. "
            "Need the format? Grab a blank template first."
        )
        up_col1, up_col2 = st.columns(2)

        with up_col1:
            st.markdown("**PCI defect data**")
            pci_template = pd.DataFrame({
                "Section ID": ["Jalan Bako KM3", "Jalan Bako KM3"],
                "Defect Type": ["Potholes", "Longitudinal Crack"],
                "Severity": ["Medium", "Low"],
                "Area Affected (%)": [5.0, 3.0],
                "Notes": ["", ""],
            })
            tmpl_buf = io.StringIO()
            pci_template.to_csv(tmpl_buf, index=False)
            st.download_button(
                "⬇️ Download PCI template", tmpl_buf.getvalue(),
                file_name="pci_template.csv", mime="text/csv", key="pci_tmpl_dl",
            )
            new_pci_file = st.file_uploader(
                "Upload PCI defect data", type=["csv", "xlsx"], key="pci_upload_widget"
            )
            if new_pci_file is not None:
                try:
                    new_pci_df = pd.read_csv(new_pci_file) if new_pci_file.name.endswith(".csv") \
                        else pd.read_excel(new_pci_file)
                    required = {"Section ID", "Defect Type", "Severity", "Area Affected (%)"}
                    if not required.issubset(set(new_pci_df.columns)):
                        st.error(f"File must contain columns: {', '.join(required)}")
                    else:
                        if "Notes" not in new_pci_df.columns:
                            new_pci_df["Notes"] = ""
                        if "Photo" not in new_pci_df.columns:
                            new_pci_df["Photo"] = ""
                        new_pci_df["Section ID"] = new_pci_df["Section ID"].astype(str)
                        if st.button("Replace PCI data with uploaded file", key="confirm_pci_replace"):
                            st.session_state.pci_input = new_pci_df
                            st.success(f"Loaded {len(new_pci_df)} PCI records.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        with up_col2:
            st.markdown("**IRI roughness data**")
            iri_template = pd.DataFrame({
                "Section ID": ["Jalan Bako KM3", "Jalan Bako KM3"],
                "IRI (m/km)": [2.1, 2.3],
                "Notes": ["", ""],
            })
            tmpl_buf2 = io.StringIO()
            iri_template.to_csv(tmpl_buf2, index=False)
            st.download_button(
                "⬇️ Download IRI template", tmpl_buf2.getvalue(),
                file_name="iri_template.csv", mime="text/csv", key="iri_tmpl_dl",
            )
            new_iri_file = st.file_uploader(
                "Upload IRI roughness data", type=["csv", "xlsx"], key="iri_upload_widget"
            )
            if new_iri_file is not None:
                try:
                    new_iri_df = pd.read_csv(new_iri_file) if new_iri_file.name.endswith(".csv") \
                        else pd.read_excel(new_iri_file)
                    required = {"Section ID", "IRI (m/km)"}
                    if not required.issubset(set(new_iri_df.columns)):
                        st.error(f"File must contain columns: {', '.join(required)}")
                    else:
                        new_iri_df["Section ID"] = new_iri_df["Section ID"].astype(str)
                        if st.button("Replace IRI data with uploaded file", key="confirm_iri_replace"):
                            st.session_state.iri_input = new_iri_df
                            st.success(f"Loaded {len(new_iri_df)} IRI readings.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Could not read file: {e}")

        st.caption(
            "Or upload a single Excel file with both `PCI_Input` and `IRI_Input` sheets "
            "using the **Data Source** option in the sidebar instead."
        )



# ---------------------------------------------------------------------------
# Compute results (used by Results + Dashboard tabs)
# ---------------------------------------------------------------------------
pci_summary = compute_pci(
    st.session_state.pci_input,
    defect_weights=st.session_state.defect_weights,
    severity_factors=st.session_state.severity_factors,
    pci_bands=st.session_state.pci_bands,
)
iri_summary = compute_iri(st.session_state.iri_input, iri_bands=st.session_state.iri_bands)
hybrid_summary = compute_hybrid(pci_summary, iri_summary,
                                 pci_bands=st.session_state.pci_bands,
                                 iri_bands=st.session_state.iri_bands)

# ---------------------------------------------------------------------------
# TAB 2: Computation & Results
# ---------------------------------------------------------------------------
with tab_results:
    st.subheader("Results — Combined PCI, IRI & Hybrid")

    # Always build one unified table with all columns
    hybrid_reasoned = add_hybrid_reasoning(hybrid_summary)
    display_df = hybrid_reasoned[[
        "Section ID", "PCI", "PCI Condition", "Avg IRI (m/km)", "IRI Condition",
        "Hybrid Condition", "Hybrid Recommendation"
    ]].rename(columns={"Hybrid Recommendation": "Maintenance Recommendation"})
    cond_col = ["PCI Condition", "IRI Condition", "Hybrid Condition"]
    final_cond_col = "Hybrid Condition"

    # -----------------------------------------------------------------
    # Filters (fully opt-in — table shows everything until user turns this on)
    # -----------------------------------------------------------------
    use_filters = st.toggle("🔎 Filter results", value=False,
                             help="Turn on to narrow down the table by section, condition, or defect type.")

    all_sections = sorted(display_df["Section ID"].dropna().astype(str).unique().tolist(), key=natural_sort_key)
    all_defects = sorted(st.session_state.pci_input["Defect Type"].dropna().unique().tolist()) if mode != "IRI" else []

    if use_filters:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_sections = st.multiselect(
                "Section ID", options=all_sections, default=[],
                placeholder="All sections",
                help="Leave empty to include all sections, or pick specific ones",
            )
        with fc2:
            sel_conditions = st.multiselect(
                "Condition", options=["Very Good", "Good", "Fair", "Poor"], default=[],
                placeholder="All conditions",
                help="Leave empty to include all conditions, or pick specific ones",
            )
        with fc3:
            if mode != "IRI":
                sel_defects = st.multiselect(
                    "Defect Type present", options=all_defects, default=[],
                    placeholder="All defect types",
                    help="Leave empty to include all defect types, or pick specific ones",
                )
            else:
                sel_defects = []
                st.caption("Defect type filter is only available in PCI / Hybrid mode.")
    else:
        sel_sections, sel_conditions, sel_defects = [], [], []

    # Apply filters — an empty selection means "no constraint" on that field
    filtered_df = display_df.copy()
    if sel_sections:
        filtered_df = filtered_df[filtered_df["Section ID"].isin(sel_sections)]
    if sel_conditions:
        filtered_df = filtered_df[filtered_df[final_cond_col].isin(sel_conditions)]
    if sel_defects:
        sections_with_defect = st.session_state.pci_input[
            st.session_state.pci_input["Defect Type"].isin(sel_defects)
        ]["Section ID"].dropna().astype(str).unique().tolist()
        filtered_df = filtered_df[filtered_df["Section ID"].isin(sections_with_defect)]

    if use_filters and (sel_sections or sel_conditions or sel_defects):
        st.caption(f"Showing {len(filtered_df)} of {len(display_df)} sections")

    st.dataframe(
        colored_condition_table(filtered_df, cond_col),
        use_container_width=True,
        height=min(45 * (len(filtered_df) + 1), 450),
    )

    # Quick stats row (reflects filtered results)
    counts = filtered_df[final_cond_col].value_counts()
    cols = st.columns(4)
    for i, band in enumerate(["Very Good", "Good", "Fair", "Poor"]):
        n = int(counts.get(band, 0))
        cols[i].metric(band, n, help=f"{n} of {len(filtered_df)} shown sections")

    csv_buf = io.StringIO()
    filtered_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download Filtered Results as CSV",
        csv_buf.getvalue(),
        file_name=f"pavement_results_{mode.split()[0].lower()}.csv",
        mime="text/csv",
    )

    if mode.startswith("Hybrid"):
        with st.expander("🧠 Why this Hybrid condition? (per-section reasoning)", expanded=False):
            shown_ids = filtered_df["Section ID"].tolist()
            reasoning_view = hybrid_reasoned[hybrid_reasoned["Section ID"].isin(shown_ids)]
            for _, row in reasoning_view.iterrows():
                badge_color = CONDITION_COLORS.get(row["Hybrid Condition"], "#888")
                st.markdown(
                    f"**Section {row['Section ID']}** "
                    f"<span style='color:{badge_color}; font-weight:600;'>[{row['Hybrid Condition']}]</span>"
                    f"<br>{row['Why']}",
                    unsafe_allow_html=True,
                )
                st.markdown("---")


# ---------------------------------------------------------------------------
# TAB 3: Dashboard
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("Network Condition Dashboard")

    if mode == "PCI":
        value_col, cond_col, label = "PCI", "PCI Condition", "PCI"
        chart_df = pci_summary
        rec_col = "PCI Recommendation"
    elif mode == "IRI":
        value_col, cond_col, label = "Avg IRI (m/km)", "IRI Condition", "IRI (m/km)"
        chart_df = iri_summary
        rec_col = "IRI Recommendation"
    else:
        value_col, cond_col, label = "PCI", "Hybrid Condition", "Hybrid (worse of PCI/IRI)"
        chart_df = hybrid_summary
        rec_col = "Hybrid Recommendation"

    # -----------------------------------------------------------------
    # Dashboard section filter (opt-in, same pattern as Results tab)
    # -----------------------------------------------------------------
    all_dash_sections = sorted(chart_df["Section ID"].dropna().astype(str).unique().tolist(), key=natural_sort_key)
    use_dash_filter = st.toggle(
        "🔎 Filter sections", value=False,
        help="Focus the dashboard on specific sections only — leave off to show the full network."
    )
    if use_dash_filter and all_dash_sections:
        sel_dash_sections = st.multiselect(
            "Show sections", options=all_dash_sections, default=[],
            placeholder="Leave empty for all, or pick specific sections"
        )
        if sel_dash_sections:
            chart_df = chart_df[chart_df["Section ID"].astype(str).isin(sel_dash_sections)].copy()
            st.caption(f"Showing {len(chart_df)} of {len(all_dash_sections)} sections")

    # -----------------------------------------------------------------
    # Network Health Score badge
    # -----------------------------------------------------------------
    if mode == "IRI":
        # Convert IRI (lower=better, roughly 0-6 range) to a 0-100 health scale
        avg_iri_raw = chart_df["Avg IRI (m/km)"].mean() if len(chart_df) else 0
        health_score = max(0, round(100 - (avg_iri_raw / 6 * 100), 1))
    else:
        health_score = round(chart_df["PCI"].mean(), 1) if len(chart_df) else 0

    if health_score >= 85:
        health_label, health_color = "Very Good", CONDITION_COLORS["Very Good"]
    elif health_score >= 70:
        health_label, health_color = "Good", CONDITION_COLORS["Good"]
    elif health_score >= 55:
        health_label, health_color = "Fair", CONDITION_COLORS["Fair"]
    else:
        health_label, health_color = "Poor", CONDITION_COLORS["Poor"]

    st.markdown(
        f"""
        <div style='padding:14px 20px; border-radius:10px; background-color:{health_color}18;
                    border-left:5px solid {health_color}; margin-bottom:14px;'>
            <span style='font-size:0.85rem; color:#aaa;'>NETWORK HEALTH SCORE</span><br>
            <span style='font-size:2rem; font-weight:700; color:{health_color};'>{health_score}/100</span>
            <span style='font-size:1.1rem; font-weight:600; color:{health_color}; margin-left:10px;'>{health_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------
    # Network-level KPI summary
    # -----------------------------------------------------------------
    n_total = len(chart_df)
    n_poor = int((chart_df[cond_col] == "Poor").sum())
    n_good_or_better = int(chart_df[cond_col].isin(["Very Good", "Good"]).sum())
    avg_value = chart_df[value_col].mean() if n_total else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sections", n_total)
    k2.metric(f"Network Avg {label.split(' ')[0]}", f"{avg_value:.1f}")
    k3.metric("Sections in Good+ Condition", f"{n_good_or_better}/{n_total}",
              help="Very Good or Good condition")
    k4.metric("Sections Needing Major Action", n_poor,
              delta=None if n_poor == 0 else f"{n_poor} Poor", delta_color="inverse")

    st.divider()

    c1, c2 = st.columns([3, 2])

    with c1:
        bar = px.bar(
            chart_df, x="Section ID", y=value_col, color=cond_col,
            color_discrete_map=CONDITION_COLORS,
            title=f"{label} by Section",
            text=value_col,
        )
        bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        bar.update_layout(xaxis=dict(dtick=1), showlegend=True)
        st.plotly_chart(bar, use_container_width=True)

    with c2:
        pie_counts = chart_df[cond_col].value_counts().reset_index()
        pie_counts.columns = ["Condition", "Count"]
        pie = px.pie(
            pie_counts, names="Condition", values="Count",
            color="Condition", color_discrete_map=CONDITION_COLORS,
            title="Condition Distribution", hole=0.45,
        )
        st.plotly_chart(pie, use_container_width=True)

    if mode.startswith("Hybrid"):
        st.markdown("**PCI vs IRI Comparison per Section**")
        comp = go.Figure()
        comp.add_trace(go.Scatter(
            x=hybrid_summary["Section ID"], y=hybrid_summary["PCI"],
            mode="lines+markers", name="PCI (0-100 scale)"
        ))
        comp.add_trace(go.Scatter(
            x=hybrid_summary["Section ID"], y=hybrid_summary["Avg IRI (m/km)"] * 20,
            mode="lines+markers", name="IRI x20 (scaled for comparison)"
        ))
        comp.update_layout(xaxis_title="Section ID", yaxis_title="Score (scaled)",
                            title="Do PCI and IRI agree on condition across sections?")
        st.plotly_chart(comp, use_container_width=True)
        st.caption(
            "When PCI and IRI disagree on a section's condition, the Hybrid Index takes "
            "the more conservative (worse) classification — prioritizing road user safety."
        )

    st.divider()

    # -----------------------------------------------------------------
    # Defect frequency across the network (PCI / Hybrid only)
    # -----------------------------------------------------------------
    if mode != "IRI":
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Most Common Defect Types Across the Network**")
            defect_counts = st.session_state.pci_input["Defect Type"].value_counts().reset_index()
            defect_counts.columns = ["Defect Type", "Occurrences"]
            freq_chart = px.bar(
                defect_counts.sort_values("Occurrences"), x="Occurrences", y="Defect Type",
                orientation="h", title="Defect Frequency (all sections)",
                color="Occurrences", color_continuous_scale="OrRd",
            )
            freq_chart.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(freq_chart, use_container_width=True)

        with d2:
            st.markdown("**Severity Breakdown of Recorded Defects**")
            sev_counts = st.session_state.pci_input["Severity"].value_counts().reindex(
                ["Low", "Medium", "High"]
            ).fillna(0).reset_index()
            sev_counts.columns = ["Severity", "Count"]
            sev_colors = {"Low": "#9E9D24", "Medium": "#F57C00", "High": "#C62828"}
            sev_chart = px.bar(
                sev_counts, x="Severity", y="Count", color="Severity",
                color_discrete_map=sev_colors, title="Defect Severity Distribution",
            )
            sev_chart.update_layout(showlegend=False)
            st.plotly_chart(sev_chart, use_container_width=True)

        st.divider()

    # -----------------------------------------------------------------
    # Maintenance action plan summary
    # -----------------------------------------------------------------
    st.markdown("**Maintenance Action Plan — Sections per Recommended Action**")
    action_counts = chart_df[rec_col].value_counts().reset_index()
    action_counts.columns = ["Recommended Action", "Number of Sections"]
    action_chart = px.bar(
        action_counts, x="Number of Sections", y="Recommended Action",
        orientation="h", title="Sections by Maintenance Action Needed",
        text="Number of Sections",
    )
    action_chart.update_layout(showlegend=False, yaxis=dict(automargin=True))
    st.plotly_chart(action_chart, use_container_width=True)
    st.caption(
        "Use this to estimate workload and prioritize budget allocation across "
        "maintenance categories for the upcoming cycle."
    )

    st.divider()

    # -----------------------------------------------------------------
    # Priority leaderboard — ranked worst sections
    # -----------------------------------------------------------------
    st.markdown("**🚧 Priority Leaderboard — Sections Ranked by Urgency**")
    leaderboard = chart_df.copy()
    cond_rank = {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3}
    leaderboard["_rank"] = leaderboard[cond_col].map(cond_rank)
    leaderboard = leaderboard.sort_values(["_rank", value_col]).drop(columns="_rank")

    if (leaderboard[cond_col] == "Poor").any() or (leaderboard[cond_col] == "Fair").any():
        top_n = min(5, len(leaderboard))
        st.dataframe(
            colored_condition_table(leaderboard.head(top_n), [cond_col]),
            use_container_width=True,
            height=45 * (top_n + 1),
        )
        st.caption(f"Top {top_n} sections most in need of intervention, worst first.")
    else:
        st.success("No sections currently in Fair or Poor condition — network is in good shape overall.")


# ---------------------------------------------------------------------------
# TAB 4: Planning Tools (cost estimator + before/after simulator)
# ---------------------------------------------------------------------------
with tab_planning:
    st.subheader("🛠️ Maintenance Planning Tools")
    st.caption("Estimate costs and simulate the impact of maintenance actions before committing budget.")

    plan_tab1, plan_tab2 = st.tabs(["💰 Cost Estimator", "🔄 Before / After Simulator"])

    # =======================================================================
    # Cost Estimator
    # =======================================================================
    with plan_tab1:
        st.markdown("##### Cost per Maintenance Action (RM, per 100m section)")
        st.caption("Adjust these to match local JKR rates — defaults are illustrative estimates.")

        if "cost_map" not in st.session_state:
            st.session_state.cost_map = DEFAULT_COST_PER_100M.copy()

        unique_actions = sorted(set(st.session_state.cost_map.keys()))
        cost_cols = st.columns(2)
        for i, action in enumerate(unique_actions):
            with cost_cols[i % 2]:
                st.session_state.cost_map[action] = st.number_input(
                    action, min_value=0, value=int(st.session_state.cost_map[action]),
                    step=500, key=f"cost_{action}",
                )

        st.divider()

        if mode == "PCI":
            cost_rec_col = "PCI Recommendation"
            cost_chart_df = pci_summary
            cost_cond_col = "PCI Condition"
        elif mode == "IRI":
            cost_rec_col = "IRI Recommendation"
            cost_chart_df = iri_summary
            cost_cond_col = "IRI Condition"
        else:
            cost_rec_col = "Hybrid Recommendation"
            cost_chart_df = hybrid_summary
            cost_cond_col = "Hybrid Condition"

        cost_df = estimate_costs(cost_chart_df, cost_rec_col, st.session_state.cost_map)
        total_cost = cost_df["Estimated Cost (RM)"].sum()

        st.markdown("##### Estimated Total Network Maintenance Cost")
        st.metric("Total (all sections)", f"RM {total_cost:,.0f}")

        st.dataframe(
            colored_condition_table(
                cost_df[["Section ID", cost_cond_col, cost_rec_col, "Estimated Cost (RM)"]],
                [cost_cond_col],
            ),
            use_container_width=True,
            height=min(40 * (len(cost_df) + 1), 400),
        )

        st.divider()
        st.markdown("##### 🎯 Budget Scenario Planner")
        st.caption("Set a budget — see which sections get prioritized first (worst condition first).")
        budget = st.number_input(
            "Available budget (RM)", min_value=0, value=int(min(50000, total_cost)), step=1000
        )
        scenario_df = cost_scenario(cost_df, cost_cond_col, budget)
        n_fundable = int(scenario_df["Within Budget"].sum())
        st.info(f"With RM {budget:,.0f}, you can fully address **{n_fundable} of {len(scenario_df)}** sections, prioritizing worst condition first.")

        def highlight_budget(row):
            color = "#2E7D3222" if row["Within Budget"] else "#C6282822"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            scenario_df[["Section ID", cost_cond_col, "Estimated Cost (RM)", "Cumulative Cost (RM)", "Within Budget"]].style.apply(highlight_budget, axis=1),
            use_container_width=True,
            height=min(40 * (len(scenario_df) + 1), 400),
        )

    # =======================================================================
    # Before / After Simulator
    # =======================================================================
    with plan_tab2:
        st.markdown("##### Simulate the Impact of a Maintenance Action")
        st.caption("Pick a section and a maintenance action to see the projected before/after condition.")

        sim_sections = sorted(hybrid_summary["Section ID"].dropna().astype(str).unique().tolist(), key=natural_sort_key)
        if not sim_sections:
            st.warning("No section data available to simulate.")
        else:
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_section = st.selectbox("Section ID", sim_sections, key="sim_section")
            sim_row = hybrid_summary[
                hybrid_summary["Section ID"].astype(str) == str(sim_section)
            ].iloc[0]
            current_pci = float(sim_row["PCI"]) if pd.notna(sim_row["PCI"]) else 0.0
            current_iri = float(sim_row["Avg IRI (m/km)"]) if pd.notna(sim_row["Avg IRI (m/km)"]) else 0.0

            with sim_col2:
                sim_action = st.selectbox(
                    "Maintenance Action to Apply",
                    list(DEFAULT_COST_PER_100M.keys()),
                    index=list(DEFAULT_COST_PER_100M.keys()).index(sim_row["Hybrid Recommendation"])
                    if sim_row["Hybrid Recommendation"] in DEFAULT_COST_PER_100M else 0,
                    key="sim_action",
                )

            result = simulate_maintenance(current_pci, current_iri, sim_action)

            def metric_with_condition_badge(label, value, condition):
                color = CONDITION_COLORS.get(condition, "#888")
                st.markdown(
                    f"""
                    <div style='margin-bottom:8px;'>
                        <span style='font-size:0.85rem; color:#aaa;'>{label}</span><br>
                        <span style='font-size:1.6rem; font-weight:600;'>{value}</span><br>
                        <span style='display:inline-block; margin-top:2px; padding:2px 10px;
                                      border-radius:10px; font-size:0.8rem; font-weight:600;
                                      background-color:{color}22; color:{color};'>{condition}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(f"##### Projected Impact on Section {sim_section}")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**PCI**")
                bc1, bc2 = st.columns(2)
                with bc1:
                    metric_with_condition_badge("Before", f"{result['before_pci']:.1f}", result['before_pci_class'])
                with bc2:
                    metric_with_condition_badge("After", f"{result['after_pci']:.1f}", result['after_pci_class'])
            with r2:
                st.markdown("**IRI (m/km)**")
                ic1, ic2 = st.columns(2)
                with ic1:
                    metric_with_condition_badge("Before", f"{result['before_iri']:.2f}", result['before_iri_class'])
                with ic2:
                    metric_with_condition_badge("After", f"{result['after_iri']:.2f}", result['after_iri_class'])

            sim_chart = go.Figure()
            sim_chart.add_trace(go.Bar(
                x=["Before", "After"], y=[result["before_pci"], result["after_pci"]],
                name="PCI", marker_color="#1f77b4",
            ))
            sim_chart.update_layout(title="PCI Before vs After", yaxis_range=[0, 100], showlegend=False)
            st.plotly_chart(sim_chart, use_container_width=True)

            est_action_cost = st.session_state.get("cost_map", DEFAULT_COST_PER_100M).get(sim_action, 0)
            st.success(
                f"Applying **{sim_action}** to Section {sim_section} is projected to improve its "
                f"condition from **{result['before_pci_class']}** to **{result['after_pci_class']}**, "
                f"at an estimated cost of **RM {est_action_cost:,.0f}**."
            )
            st.caption(
                "Note: this is a simplified projection based on typical recovery assumptions for course "
                "demonstration purposes, not a substitute for an engineering structural assessment."
            )


# ---------------------------------------------------------------------------
# TAB 5: Methodology / About
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Methodology")

    st.markdown("""
**Pavement Condition Index (PCI)**

For each defect observed in a section:

```
Deduct Value = Weighting Factor × Severity Factor × Area Affected (%)
```

All deduct values in a section are summed, then:

```
PCI = 100 − (Sum of Deduct Values), floored at 0
```

This is a linear simplification of ASTM D6433's curve-based deduct value
method, adapted for course use. The full standard uses non-linear deduct
curves per defect type and a Corrected Deduct Value (CDV) process to avoid
double-penalizing sections with many simultaneous defects.

**International Roughness Index (IRI)**

```
Section IRI = Average of segment-level IRI readings (m/km)
```

**Hybrid Index**

Combines PCI and IRI by selecting the more conservative (worse) of the two
classifications for each section — reflecting the engineering judgment that
either indicator showing poor condition is sufficient grounds for concern.
    """)

    st.markdown("**Condition Bands**")
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("*PCI Bands*")
        st.table(bands_to_dataframe(st.session_state.pci_bands, "PCI"))
    with bc2:
        st.markdown("*IRI Bands*")
        st.table(bands_to_dataframe(st.session_state.iri_bands, "IRI"))

    st.markdown("**Defect Weighting Factors**")
    st.table(pd.DataFrame(
        list(st.session_state.defect_weights.items()),
        columns=["Defect Type", "Weighting Factor"]
    ))

    st.caption(
        "References: ASTM D6433 (PCI Survey Standard); JKR Pavement Maintenance Manual; "
        "JKR IRI Classification Guidance."
    )
