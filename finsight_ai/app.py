import streamlit as st
from streamlit_option_menu import option_menu
import importlib.util
from pathlib import Path
import sys
import base64

# Page configuration
st.set_page_config(
    page_title="Finsight AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .css-1d391kg {display: none;}
    [data-testid="stSidebar"] {display: none;}
    .css-1lcbmhc {display: none;}
    .css-1outpf7 {display: none;}
    section[data-testid="stSidebar"] {display: none !important;}
    .css-1y4p8pa {padding-left: 0 !important;}
    .css-1d391kg {width: 0 !important;}
</style>
""", unsafe_allow_html=True)

logo_path = Path(__file__).parent / "assets" / "img" / "logo.png"
partner_logo_path = Path(__file__).parent / "assets" / "partner_logo.png"

try:
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")
except Exception:
    logo_b64 = ""
partner_b64 = ""
if partner_logo_path.exists():
    try:
        with open(partner_logo_path, "rb") as f:
            partner_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        partner_b64 = ""

partner_html = ""
if partner_b64:
    partner_html = f'<img alt="partner" src="data:image/png;base64,{partner_b64}" style="height:26px;display:block;"/>'

header_html = f"""
<a href="?home=1" style="text-decoration:none;">
<div style="display:flex;justify-content:center;align-items:center;margin:8px 0 10px 0;gap:12px;cursor:pointer;">
  <img alt="logo" src="data:image/png;base64,{logo_b64}" style="height:38px;display:block;"/>
  <div style="font-weight:800;font-size:20px;letter-spacing:1.2px;color:#EDE9FE;">FINSIGHT AI</div>
  {partner_html}
</div>
</a>
"""

params = st.query_params
if "home" in params:
    st.session_state["show_home"] = True
if "show_home" not in st.session_state:
    st.session_state["show_home"] = False

# Home (landing) page
if st.session_state["show_home"]:
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:14px;margin:24px 0 10px 0;">
            <img alt="logo" src="data:image/png;base64,{logo_b64}" style="height:72px;display:block;"/>
            <div style="font-weight:800;font-size:26px;letter-spacing:1.4px;color:#EDE9FE;">FINSIGHT AI</div>
            <div style="color:#9CA3AF;max-width:820px;text-align:center;line-height:1.6;">
                AI-first AML and finance analytics platform. Explore your data, monitor intelligent dashboards,
                and leverage machine learning for fraud detection, anomaly discovery, forecasting, and risk.
                Designed to be clean, fast, and delightful to use.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Start button
    st.markdown(
        """
        <style>
        /* Home page only: style native Streamlit button */
        div[data-testid="stButton"] > button {
            background: linear-gradient(90deg,#8B5CF6,#3B82F6) !important;
            border: 1px solid rgba(139,92,246,0.55) !important;
            padding: 10px 22px !important;
            color: #fff !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            letter-spacing: .4px !important;
            cursor: pointer !important;
        }
        div[data-testid="stButton"] > button:hover { filter: brightness(1.05); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([3, 2, 3])
    with c2:
        if st.button("Get started", key="start_btn", use_container_width=True):
            st.session_state["show_home"] = False
            if "home" in st.query_params:
                del st.query_params["home"]
            st.rerun()

    st.stop()
else:
    st.markdown(header_html, unsafe_allow_html=True)


_theme_base = (st.get_option("theme.base") or "dark").lower()
_is_dark = _theme_base == "dark"

CONTAINER_BG = "#0F172A" if _is_dark else "#F8FAFC"
TEXT_INACTIVE = "#9CA3AF" if _is_dark else "#6B7280"
ICON_INACTIVE = "#A78BFA" if _is_dark else "#7C3AED"
HOVER_BG = "#1E1B2E" if _is_dark else "#EEF2FF"
BORDER_COLOR = "rgba(139,92,246,0.45)" if _is_dark else "rgba(139,92,246,0.35)"
ACTIVE_GRADIENT = "linear-gradient(90deg, #8B5CF6 0%, #3B82F6 100%)"

page = option_menu(
    menu_title=None,  # required
            options=["Global", "Analysis", "Forecasts", "Risks", "Client"],  # required
          icons=["globe2", "search", "graph-up", "shield-check", "person"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": CONTAINER_BG, "border": "0"},
        "icon": {"color": ICON_INACTIVE, "font-size": "22px"},
        "nav-link": {
            "font-size": "16px",
            "font-weight": "600",
            "color": TEXT_INACTIVE,
            "text-align": "center",
            "margin": "6px 8px",
            "padding": "10px 20px",
            "border-radius": "12px",
            "height": "50px",
            "display": "flex",
            "align-items": "center",
            "gap": "8px",
            "background": "transparent",
            "border": "1px solid transparent",
            "--hover-color": HOVER_BG,
        },
        "nav-link-selected": {
            "background": ACTIVE_GRADIENT,
            "color": "#FFFFFF",
            "border-radius": "12px",
            "box-shadow": "0 4px 12px rgba(139,92,246,0.35)",
        },
    }
)

# Extra CSS for hover effects and active icons
st.markdown(
    f"""
    <style>
    /* Hover inactive tab */
    div[data-testid=\"stHorizontalBlock\"] ul li a:hover {{
        background: {HOVER_BG} !important;
        border: 1px solid #8B5CF6 !important;
        color: #111827 !important;
    }}

    /* Active icon */
    .nav-link-selected svg {{ fill: #FFFFFF !important; color: #FFFFFF !important; }}

    /* Inactive icons */
    ul li a svg {{ color: {ICON_INACTIVE} !important; }}

    /* Neon underline of the active tab */
    .nav-link-selected {{ position: relative; }}
    .nav-link-selected:after {{
        content: "";
        position: absolute;
        left: 8px; right: 8px; bottom: -6px;
        height: 3px; border-radius: 3px;
        background: {ACTIVE_GRADIENT};
        box-shadow: 0 0 12px rgba(139,92,246,0.35);
    }}

    /* Menu container */
    div[data-testid=\"stHorizontalBlock\"] ul {{ 
        position: relative; 
        padding: 10px 12px 14px 12px; 
        margin: 6px 0 0 0;
        background: {CONTAINER_BG}; 
        border: 1px solid {BORDER_COLOR}; 
        border-radius: 16px; 
        box-shadow: 0 4px 16px rgba(0,0,0,0.35), 0 0 0 1px rgba(139,92,246,0.18) inset;
    }}
    div[data-testid=\"stHorizontalBlock\"] ul:after {{
        content: ""; position: absolute; left: 10px; right: 10px; bottom: 8px; height: 2px; border-radius: 2px;
        background: linear-gradient(90deg, rgba(139,92,246,0.0) 0%, #8B5CF6 25%, #3B82F6 75%, rgba(59,130,246,0.0) 100%);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='neon-separator'></div>", unsafe_allow_html=True)


def run_streamlit_module(py_file: Path):
    page_dir = str(py_file.parent)
    if page_dir not in sys.path:
        sys.path.insert(0, page_dir)
    spec = importlib.util.spec_from_file_location("_finsight_page", str(py_file))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    if hasattr(module, "main") and callable(module.main):
        module.main()

if page == "Global":
    dashboard_path = Path(__file__).parent / "pages" / "1_Dashboard_Global" / "Dashboard.py"
    if dashboard_path.exists():
        run_streamlit_module(dashboard_path)
    else:
        st.error(f"Fichier Dashboard introuvable: {dashboard_path}")
    
elif page == "Analysis":
    analysis_path = Path(__file__).parent / "pages" / "2_Analyse" / "Analysis.py"
    if analysis_path.exists():
        run_streamlit_module(analysis_path)
    else:
        st.error(f"Fichier Analysis introuvable: {analysis_path}")
    
elif page == "Forecasts":
    
    prevision_option = st.radio(
        "Choose forecast type:",
        ["📈 Forecasting", "💸 Fraud Loss Forecast"],
        horizontal=True,
        label_visibility="visible"
    )
    st.markdown("---")
    
    if prevision_option == "📈 Forecasting":
        forecasting_path = Path(__file__).parent / "pages" / "3_Previsions" / "Forecasting.py"
        if forecasting_path.exists():
            run_streamlit_module(forecasting_path)
        else:
            st.subheader("Forecasting")
            st.info("Prévisions de volumes et montants avec SARIMA/Prophet.")
        
    elif prevision_option == "💸 Fraud Loss Forecast":
        fraud_loss_path = Path(__file__).parent / "pages" / "3_Previsions" / "FraudLoss.py"
        if fraud_loss_path.exists():
            run_streamlit_module(fraud_loss_path)
        else:
            st.subheader("Fraud Loss Forecast")
            st.info("Prévisions des pertes dues à la fraude.")
    
elif page == "Risks":
    
    risk_path = Path(__file__).parent / "pages" / "4_Risques" / "RiskCFaR.py"
    if risk_path.exists():
        run_streamlit_module(risk_path)
    else:
        st.subheader("Risk & CFaR")
        st.info("Analyse des risques et Cash-Flow-at-Risk.")
    
elif page == "Client":
    
    client_option = st.radio(
        "Choose client analysis:",
        ["👤 Client Profile", "🕸️ Transaction Graph"],
        horizontal=True,
        label_visibility="visible"
    )
    st.markdown("---")
    
    if client_option == "👤 Client Profile":
        profil_path = Path(__file__).parent / "pages" / "5_Client" / "ProfilClient.py"
        if profil_path.exists():
            run_streamlit_module(profil_path)
        else:
            st.subheader("Profil Client")
            st.info("Vue 360° des clients et analyse comportementale.")
        
    elif client_option == "🕸️ Transaction Graph":
        graphe_path = Path(__file__).parent / "pages" / "5_Client" / "GrapheTransactions.py"
        if graphe_path.exists():
            run_streamlit_module(graphe_path)
        else:
            st.subheader("Graphe des Transactions")
            st.info("Analyse de réseau AML et détection de communautés.")
    
else:
    st.title("🏦 Finsight AI")
    st.write("Select a page from the menu")