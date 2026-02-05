import streamlit as st
import pandas as pd
import pdfplumber
import re
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np
import warnings
from typing import Any, Dict, Optional, Tuple, List, Union
import io # For downloading data
import copy # For deep copying data for export

# --- Configuration ---
warnings.filterwarnings("ignore")
DEFAULT_CURRENCY = "USD" # Default used if extraction fails
DEFAULT_SCALE = 1 # Default to 1 (no scaling) if extraction fails

# =============================================
# 🎨 DASHBOARD CONFIGURATION & STYLING
# =============================================
st.set_page_config(
    page_title="World-Class FP&A AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header with attribution
st.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <h1>📊 Financial Analysis AI Dashboard</h1>
    <p><em>Created by Matthew Blackman with assistance from DeepSeek AI</em></p>
</div>
""", unsafe_allow_html=True)

# (Styling CSS remains the same)
LIGHT_MODE_CSS = """
<style>
    /* Main background */
    .stApp { background-color: #f8f9fa; }
    /* Cards */
    .metric-card { background-color: white; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border: 1px solid #e9ecef; }
    /* Positive values (used in insights) */
    .positive { color: #198754; font-weight: bold; }
    /* Negative values (used in insights AND KPIs) */
    .negative { color: #dc3545; font-weight: bold; }
    h4 { margin-bottom: 0.75rem; color: #495057; }
    p { font-size: 0.95rem; line-height: 1.5; color: #6c757d; }
    table { width: 100%; }
    /* Style for KPI value */
    .kpi-value { font-size: 1.8em; font-weight: 600; line-height: 1; color: #1a1a1a; } /* Default color for light mode */
    /* Style for Segment KPI labels */
    .segment-kpi-label { font-weight: bold; color: #495057; margin-bottom: 0.2rem; font-size: 0.9em;}
    /* Style for Segment KPI values */
    .segment-kpi-value { font-size: 1.1em; font-weight: 500; color: #1a1a1a; margin-bottom: 0.5rem;}
</style>
"""
DARK_MODE_CSS = """
<style>
    /* Main background */
    .stApp { background-color: #1a1a1a; color: #e9ecef; }
    /* Cards */
    .metric-card { background-color: #2d2d2d; color: #e9ecef; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.3); border: 1px solid #495057; }
    /* Text */
    .stMarkdown, .stText, .stMetric,
    .stTextInput > label, .stSelectbox > label, .stCheckbox > label,
    .stRadio > label, .stButton > button, .stFileUploader > label {
        color: #e9ecef !important;
    }
    h1, h2, h3 { color: #ffffff !important; }
    h4 { margin-bottom: 0.75rem; color: #f8f9fa; }
    p { font-size: 0.95rem; line-height: 1.5; color: #adb5bd; }
    /* Positive values (used in insights AND KPIs) */
    .positive { color: #4ade80; font-weight: bold; }
    /* Negative values (used in insights AND KPIs) */
    .negative { color: #f87171; font-weight: bold; }
    /* Input background */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stFileUploader div[data-baseweb="file-uploader"] { background-color: #3f3f3f; color: #e9ecef; border: 1px solid #5f5f5f; }
    /* Buttons */
    .stButton button { background-color: #4f46e5; color: white; border: none; }
    .stButton button:hover { background-color: #4338ca; }
    /* Plotly chart background */
     .js-plotly-plot .plotly, .js-plotly-plot .plotly svg { background-color: rgba(0,0,0,0) !important; }
    /* Style for KPI value (color overridden by inline style based on value) */
    .kpi-value { font-size: 1.8em; font-weight: 600; line-height: 1; color: #e9ecef;} /* Default color for dark mode */
    /* Metric Delta (keep styling for default delta) */
    .stApp[data-theme="dark"] .stMetric > div:nth-child(3) { color: #adb5bd !important; }
    /* Markdown table appearance in dark mode */
    table { width: 100%; color: #e9ecef; }
    th { background-color: #3f3f3f; border: 1px solid #5f5f5f; }
    td { border: 1px solid #495057; }
     /* Style for Segment KPI labels */
    .segment-kpi-label { font-weight: bold; color: #f8f9fa; margin-bottom: 0.2rem; font-size: 0.9em; }
    /* Style for Segment KPI values */
    .segment-kpi-value { font-size: 1.1em; font-weight: 500; color: #e9ecef; margin-bottom: 0.5rem;}
</style>
"""

def apply_theme(theme_mode: str) -> None:
    if theme_mode == "dark": st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    else: st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# =============================================
# 🛠️ UTILITY FUNCTIONS
# =============================================
def safe_float_conversion(value: Any) -> Optional[float]:
    if value is None: return None
    try:
        if isinstance(value, (int, float)): return float(value)
        if isinstance(value, str):
            value = value.replace('$', '').replace('€', '').replace('£', '').replace(',', '').strip()
            if value.startswith('(') and value.endswith(')'): value = '-' + value[1:-1]
            if value.endswith('%'): return float(value[:-1]) / 100.0 if not value.endswith('¢%') else float(value[:-2]) / 100.0
            elif value.endswith('¢'): return float(value[:-1]) / 100.0
            if not value: return None
            return float(value)
        return float(value)
    except (ValueError, TypeError): return None

def format_currency(value: Optional[Union[float, int]], currency: str = DEFAULT_CURRENCY, scale: Optional[Union[float, int]] = None, precision: int = 1) -> str:
    if value is None or not isinstance(value, (int, float)) or pd.isna(value): return "N/A"
    eff_scale = float(scale) if scale and isinstance(scale, (int, float)) and scale != 0 else 1.0
    abs_val_original = abs(value * eff_scale)
    display_divisor = 1.0; scale_suffix = ""
    if abs_val_original >= 1_000_000_000: display_divisor = 1_000_000_000.0 / eff_scale; scale_suffix = "B"
    elif abs_val_original >= 1_000_000: display_divisor = 1_000_000.0 / eff_scale; scale_suffix = "M"
    elif abs_val_original >= 1_000: display_divisor = 1_000.0 / eff_scale; scale_suffix = "k"
    try:
        display_num = value / display_divisor
        formatted_num = f"{display_num:,.{precision}f}"
        if scale_suffix and precision > 0:
            formatted_num = formatted_num.replace(f'.{"0"*precision}', '')
        elif scale_suffix and precision == 0:
             formatted_num = f"{int(display_num):,}"
        return f"{currency} {formatted_num}{scale_suffix}"
    except (TypeError, ValueError, ZeroDivisionError): return "N/A"

def format_percentage(value: Optional[float], precision: int = 1) -> str:
    if value is None or not isinstance(value, (int, float)) or pd.isna(value): return "N/A"
    try:
        return f"{value * 100:.{precision}f}%"
    except (TypeError, ValueError): return "N/A"

# =============================================
# 📄 DATA EXTRACTION & HARMONIZATION (Foundation Layer)
# =============================================
@st.cache_data(show_spinner="📄 Reading PDF structure...", ttl=3600)
def basic_pdf_text_extraction(pdf_file) -> str:
    full_text = ""; pdf_stream = None
    try:
        pdf_stream = io.BytesIO(pdf_file) if isinstance(pdf_file, bytes) else pdf_file
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages: page_text = page.extract_text(x_tolerance=1, y_tolerance=3); full_text += page_text + "\n" if page_text else ""
        return full_text
    except Exception as e: st.error(f"🚨 Error reading PDF: {e}"); return ""
    finally:
        if isinstance(pdf_stream, io.BytesIO): pdf_stream.close()

def extract_financial_data(pdf_file, company_type: str = "insurance") -> Dict[str, Any]:
    financials = {"metadata": {"company_type": company_type, "company_name": "Unknown Company", "report_date": None, "currency": DEFAULT_CURRENCY, "scale": DEFAULT_SCALE, "extraction_method": "Basic Regex (Placeholder)", "ifrs17_model_detected": "Unknown", "warnings": []}, "Income Statement": {}, "Balance Sheet": {}, "Cash Flow": {}, "IFRS17 Metrics": {}, "IFRS17 Movements": {"CSM Movement": {}, "Risk Adjustment Movement": {}, "Income Breakdown": {}, "Loss Component Movement": {}}, "Ratios": {}, "Segments": {}, "ESG": {}, "Market": {}, "Historical Data": {}, "Notes Data": {}}
    full_text = basic_pdf_text_extraction(pdf_file)
    if not full_text: financials["metadata"]["warnings"].append("Failed to extract text."); return financials
    first_lines = [line.strip() for line in full_text.split('\n')[:20] if line.strip()]
    if first_lines: name_match = re.search(r"^(.*?)(?:ANNUAL REPORT|FINANCIAL STATEMENTS|$)", first_lines[0], re.IGNORECASE); financials["metadata"]["company_name"] = name_match.group(1).strip() if name_match and len(name_match.group(1).strip()) > 3 else first_lines[0]
    date_match = re.search(r"(?:year ended|as at|as of|ended)\s+(?:December|Dec)\s+31,\s*(\d{4})", full_text, re.IGNORECASE | re.MULTILINE) or re.search(r"31\s+(?:December|Dec)\s+(\d{4})", full_text, re.IGNORECASE | re.MULTILINE) or re.search(r"(?:year ended|as at|as of|ended)\s+(?:March|Mar)\s+31,\s*(\d{4})", full_text, re.IGNORECASE | re.MULTILINE) # Added March
    if date_match:
        year = date_match.group(1)
        month_day = "03-31" if "March" in date_match.group(0) or "Mar" in date_match.group(0) else "12-31"
        financials["metadata"]["report_date"] = f"{year}-{month_day}"
    else: year_match = re.search(r"\b(20\d{2})\b", "\n".join(first_lines)); financials["metadata"]["report_date"] = f"{year_match.group(1)}-12-31" if year_match else None; financials["metadata"]["warnings"].append("Could not determine report date.")

    scale_match_expressed = re.search(r"expressed in (thousands|millions) of (?:Barbados|BBD|US|U\.S\.)\s*(dollars|\$)", full_text, re.IGNORECASE | re.MULTILINE)
    scale_match_parentheses = re.search(r"\((?:in|figures in)\s+(thousands|millions)\s+of\s+([A-Z]{3}|US|U\.S\.|Barbados)\s*dollars?\)", full_text, re.IGNORECASE | re.MULTILINE)
    currency_match_symbol = re.search(r"expressed in\s+(?:US|U\.S\.|Barbados)\s*\$|US\$|BBD\$", full_text, re.IGNORECASE | re.MULTILINE)
    currency = DEFAULT_CURRENCY; scale = DEFAULT_SCALE

    if scale_match_expressed:
        scale = 1000 if scale_match_expressed.group(1).lower() == "thousands" else 1000000
        currency = "USD" if "US" in scale_match_expressed.group(2) else "BBD" if "Barbados" in scale_match_expressed.group(2) else DEFAULT_CURRENCY
    elif scale_match_parentheses:
        scale = 1000 if scale_match_parentheses.group(1).lower() == "thousands" else 1000000
        curr_str = scale_match_parentheses.group(2).upper()
        currency = "USD" if "US" in curr_str else "BBD" if "BARBADOS" in curr_str else curr_str if len(curr_str)==3 else DEFAULT_CURRENCY
    elif currency_match_symbol:
        currency = "USD" if "US" in currency_match_symbol.group(0) else "BBD" if "BBD" in currency_match_symbol.group(0) else DEFAULT_CURRENCY
        scale_match_only = re.search(r"\(in\s+(thousands|millions)\)", full_text, re.IGNORECASE | re.MULTILINE)
        scale = (1000 if scale_match_only.group(1).lower() == "thousands" else 1000000) if scale_match_only else DEFAULT_SCALE
        if not scale_match_only: financials["metadata"]["warnings"].append(f"Determined currency ({currency}) but not scale.")
    else:
        currency_code_match = re.search(r"\(([A-Z]{3})\)", full_text)
        currency = currency_code_match.group(1) if currency_code_match else DEFAULT_CURRENCY
        financials["metadata"]["warnings"].append("Could not determine currency/scale.")

    financials["metadata"]["scale"] = int(scale) if scale > 0 else 1
    financials["metadata"]["currency"] = currency

    patterns = {
            "Income Statement": {
                "Insurance Revenue": r"Insurance\s+revenue\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Insurance
                "Insurance Service Expenses": r"Insurance\s+service\s+expenses\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Insurance
                "Net Investment Income": r"Net\s+investment\s+income(?:/\(loss\))?\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Common
                "Net insurance finance expenses": r"Net\s+insurance\s+finance\s+expenses\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Insurance
                "Net Income": r"Net\s+(?:income|earnings|loss)\s+(?:for\s+the\s+year)?\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Common (Added loss)
                "Net Income Attributable to Shareholders": r"Net\s+(?:income|earnings|loss)\s+attributable\s+to\s+common\s+shareholders\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Common (Added loss)
                "Total Revenue": r"Total\s+Revenue\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Airport - Using a broader pattern as line item may vary
                "Revenue from contracts with customers": r"Revenue\s+from\s+contracts\s+with\s+customers\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Airport Revenue source 1
                "Revenue for lease contracts": r"Revenue\s+for\s+lease\s+contracts\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Airport Revenue source 2
                "Aeronautical Cost": r"Aeronautical\s+Cost\s*.*?\\n?.*?(-?\\(?[\\d,.]+\\)?)\\s*$", # Airport (Placeholder pattern - likely won't match)
                "Non-aeronautical Cost": r"Non-aeronautical\s+Cost\s*.*?\\n?.*?(-?\\(?[\\d,.]+\\)?)\\s*$", # Airport (Placeholder pattern - likely won't match)
                "Total Expenses": r"Total\s+Expenses\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$", # Airport - Using a broader pattern
            },
            "Balance Sheet": {
                "Total Assets": r"Total\s+assets\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$",
                "Total Liabilities": r"Total\s+liabilities(?!\s+and\s+equity)\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$",
                "Total Equity": r"Total\s+equity\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$",
                "Shareholders' Equity": r"Total liabilities and shareholder's equity.*?(-?\(?[\d,.]+\)?)\s*$", # Alt for Equity
                "CSM": r"Contractual\s+service\s+margin(?!\\s*\\(?net)\\s*.*?\\n?.*?(-?\\(?[\\d,.]+\\)?)\\s*$", # Insurance
                "Risk Adjustment": r"Risk\\s+adjustment(?:.*non.financial.*)?\\s*.*?\\n?.*?(-?\\(?[\\d,.]+\\)?)\\s*$" # Insurance
            },
            "IFRS17 Metrics": { # Insurance Specific
                "Insurance Service Result": r"INSURANCE\s+SERVICE\s+RESULT\s*.*?\n?.*?(-?\(?[\d,.]+\)?)\s*$",
                 "CSM": r"Net\s+CSM(?:.*to\s+shareholders)?,\\s+end\\s+of\\s+period\\s*.*?\\n?.*?(-?\\(?[\\d,.]+\\)?)\\s*$",
                 "New Business CSM": r"New\\s+business\\s+CSM\\s*.*?\\n?.*?(-?\\(?[\\d,.]+\\)?)\\s*$"
            },
            "Ratios": {
                 "Core EPS": r"Core\s+(?:basic|fully\\s+diluted)\\s+EPS(?:\\s+\\(US[\u00A2$]\\s*per\\s+share\\))?\\s*.*?(-?[\\d.]+\u00A2?)\\s*$", # Insurance
                 "Book Value per share": r"Book\\s+value\\s+\\(per\\s+share\\)\\s*.*?\\n?.*?US\\$([\\d.]+)\\s*$", # Insurance
                 "LICAT Ratio": r"(?:LICAT|MCCSR)\\s+(?:Ratio|capital\\s+ratio)\\s*.*?(-?[\\d.]+)%\\s*$", # Insurance
                 "Reported ROE": r"Reported\\s+ROE\\s*\\(.*?(-?[\\d.]+)%\\s*$", # Insurance
                 "Core ROE": r"Core\\s+ROE\\s*.*?(-?[\\d.]+)%\\s*$", # Insurance
                 # Airport / General
                 "Debt to Equity Ratio": r"Debt-equity\\s+ratio\\s*.*?([\\d.]+):1",
                 "Current Ratio": r"Current\\s+ratio\\s*.*?([\\d.]+)\\s*$",
                 "Debt Service Coverage Ratio": r"Debt\\s+Service\\s+Coverage\\s+Ratio\\s*.*?(-?[\\d.]+)\\s*$",
            },
             "Other KPIs": { # Airport Specific
                 "Total Passengers": r"Total\\s+Passengers\\s*\\n*.*?([\\d,]+)\\s*\\n*Total\\s+Aircraft",
                 "Aircraft Movements": r"Aircraft\\s+Movements\\s*\\n*.*?([\\d,]+)\\s*\\n*Total\\s+Passengers",
                 "Cargo Handled (kg)": r"Cargo\\s+of\\s+([\\d,]+)\\s+kgs",
                 "Mail Handled (kg)": r"Mail\\s+handled\\s+was\\s+([\\d,]+)\\s+kgs",
            }
    }
    extracted_values = {}
    for section, section_patterns in patterns.items():
        extracted_values[section] = {}
        for key, pattern in section_patterns.items():
            try: matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE); value = safe_float_conversion(matches[-1][0] if isinstance(matches[-1], tuple) else matches[-1]) if matches else None; extracted_values[section][key] = value if value is not None else None
            except Exception as e: financials["metadata"]["warnings"].append(f"Regex error {key}: {e}")
    current_scale = financials["metadata"]["scale"]
    for section, values in extracted_values.items():
        for key, value in values.items():
            if value is None: continue
            target_section = section; target_key = key.replace(" Attributable to Shareholders", "").replace(" per share", "")
            if section == "Ratios": final_value = value / 100.0 if "%" in patterns[section][key] else value
            elif section == "Other KPIs": final_value = value # KPIs are usually raw numbers
            elif section != "Ratios": final_value = value * current_scale
            else: final_value = value
            if section == "Balance Sheet" and target_key == "CSM": target_section = "IFRS17 Metrics"
            elif section == "Balance Sheet" and target_key == "Risk Adjustment": target_section = "IFRS17 Metrics"
            elif section == "Balance Sheet" and target_key == "Shareholders' Equity" and financials["Balance Sheet"].get("Total Equity") is None : target_key="Total Equity" # Use alternative equity source if primary missing

            if target_section not in financials: financials[target_section] = {}
            if target_key not in financials[target_section] or financials[target_section].get(target_key) is None: financials[target_section][target_key] = final_value

    ni = financials.get("Income Statement", {}).get("Net Income");
    assets = financials.get("Balance Sheet", {}).get("Total Assets");
    equity = financials.get("Balance Sheet", {}).get("Total Equity")

    if ni is not None and equity is not None and equity != 0:
        historical_equity = financials.get("Historical Data", {}).get("Total Equity", {})
        prev_year = str(int(financials["metadata"]["report_date"].split("-")[0]) - 1) if financials["metadata"]["report_date"] else None
        avg_equity = (equity + historical_equity.get(prev_year, equity)) / 2 if prev_year and historical_equity.get(prev_year) is not None else equity
        if "ROE" not in financials.get("Ratios", {}) and "Reported ROE" not in financials.get("Ratios", {}) and "Core ROE" not in financials.get("Ratios", {}):
             financials.setdefault("Ratios", {})["ROE"] = ni / avg_equity if avg_equity != 0 else None

    if ni is not None and assets is not None and assets != 0:
        historical_assets = financials.get("Historical Data", {}).get("Total Assets", {})
        prev_year = str(int(financials["metadata"]["report_date"].split("-")[0]) - 1) if financials["metadata"]["report_date"] else None
        avg_assets = (assets + historical_assets.get(prev_year, assets)) / 2 if prev_year and historical_assets.get(prev_year) is not None else assets
        if "ROA" not in financials.get("Ratios", {}):
            financials.setdefault("Ratios", {})["ROA"] = ni / avg_assets if avg_assets != 0 else None

    debt = financials.get("Balance Sheet", {}).get("Total Liabilities")
    if debt is not None and equity is not None and equity != 0 and "Debt to Equity Ratio" not in financials.get("Ratios", {}):
        financials.setdefault("Ratios", {})["Debt to Equity Ratio"] = debt / equity

    if "Net Income" not in financials.get("Income Statement", {}) and "Net Income Attributable" in financials.get("Income Statement", {}): financials["Income Statement"]["Net Income"] = financials["Income Statement"].pop("Net Income Attributable")
    bs = financials["Balance Sheet"]; income_statement = financials["Income Statement"]; ratios = financials.setdefault("Ratios", {})
    if "Total Assets" in bs and "Total Liabilities" in bs and "Total Equity" not in bs: bs["Total Equity"] = bs["Total Assets"] - bs["Total Liabilities"]
    elif "Total Assets" in bs and "Total Equity" in bs and "Total Liabilities" not in bs: bs["Total Liabilities"] = bs["Total Assets"] - bs["Total Equity"]

    if not any(financials.get(sec) for sec in ["Income Statement", "Balance Sheet", "IFRS17 Metrics", "Other KPIs", "Ratios"]): financials["metadata"]["warnings"].append("Extraction failed."); st.error("⚠️ Basic Regex Extraction Failed."); financials["metadata"]["extraction_method"] = "Basic Regex (Failed)"
    else: financials["metadata"]["extraction_method"] = "Basic Regex (Success: Minimal)"
    return financials

# =============================================
# 🧠 AI ANALYSIS & FORECASTING (Core AI Layer)
# =============================================
@st.cache_data(show_spinner="🔮 Running AI forecasts...", ttl=600)
def create_prophet_forecast(historical_data: Dict[str, float], periods: int = 3, freq: str = 'Y') -> Tuple[Optional[pd.DataFrame], Optional[Prophet]]:
    numeric_data = {k: v for k, v in historical_data.items() if isinstance(v, (int, float))}
    if not numeric_data or len(numeric_data) < 2: st.warning("Insufficient historical data for Prophet forecasting."); return None, None
    try:
        df = pd.DataFrame(list(numeric_data.items()), columns=['ds', 'y'])
        if all(re.match(r"^\d{4}$", str(d)) for d in df['ds']):
             df['ds'] = pd.to_datetime(df['ds'].astype(str) + '-12-31')
        else:
             df['ds'] = pd.to_datetime(df['ds'])

        df = df.sort_values('ds')
        model = Prophet(yearly_seasonality='auto', changepoint_prior_scale=0.1)
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        return forecast, model
    except Exception as e:
        st.error(f"Prophet forecasting error: {e}")
        return None, None

def generate_ai_insights(data: Dict[str, Any], historical_data: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """
    Generates template-based AI insights including ratio explanations and benchmarking.
    """
    insights = []
    metadata = data.get("metadata", {})
    company_name = metadata.get("company_name", "The Company")
    report_date = metadata.get("report_date", "N/A")
    currency = metadata.get("currency", DEFAULT_CURRENCY)
    scale = metadata.get("scale")
    company_type = metadata.get("company_type", "general")

    report_year_str = report_date.split("-")[0] if report_date and '-' in report_date else "Current"
    try:
        report_year = int(report_year_str)
        prev_year = report_year - 1
        prev_year_str = str(prev_year)
    except:
        report_year = None
        prev_year = None
        prev_year_str = "Previous"

    scale_desc = "Millions" if scale == 1_000_000 else "Thousands" if scale == 1000 else "" if scale == 1 else f"Scale: {scale}"

    # --- Executive Summary ---
    insights.append(f"""<div class='metric-card'>
        <h4>Executive Summary (AI Generated)</h4>
        <p>AI analysis of <strong>{company_name}</strong>'s financial report for {report_date}. Key findings cover profitability, {f"{company_type} specific" if company_type not in ['general', 'other'] else ""} metrics, financial position, and segment performance.</p>
        <p><em>(Note: Template-based commentary with added ratio explanations and general benchmarking.)</em></p>
        {f"<p>⚠️ Extraction warnings: {', '.join(metadata.get('warnings', []))}</p>" if metadata.get('warnings') else ""}
        {f"<p>Data Scale: Values in {currency} {scale_desc}</p>" if scale !=1 else "<p>Data Scale: Values as reported.</p>"}
    </div>""")

    income = data.get("Income Statement", {})
    ratios = data.get("Ratios", {})
    hist_data = data.get("Historical Data", {})
    other_kpis = data.get("Other KPIs", {})

    ni_current_key = "Net Income Attributable to Shareholders" if "Net Income Attributable to Shareholders" in income else "Net Income"
    ni_current = income.get(ni_current_key)
    ni_hist = hist_data.get("Net Income", {})
    ni_prev = ni_hist.get(prev_year_str) if prev_year_str in ni_hist else None

    core_eps_current = ratios.get("Core EPS")
    core_eps_hist = hist_data.get("Core EPS", {})
    core_eps_prev = core_eps_hist.get(prev_year_str) if prev_year_str in core_eps_hist else None

    roe_key = "Reported ROE" if "Reported ROE" in ratios else "Core ROE" if "Core ROE" in ratios else "ROE"
    roe_current = ratios.get(roe_key)
    roe_hist = hist_data.get("ROE", {}) # Assuming historical ROE is stored under "ROE"
    roe_prev = roe_hist.get(prev_year_str) if prev_year_str in roe_hist else None


    # --- Profitability Insights ---
    profit_insight = "<div class='metric-card'><h4>Profitability Insights</h4>"
    if company_type == 'airport':
        rev_contracts = income.get("Revenue from contracts with customers")
        rev_lease = income.get("Revenue for lease contracts")
        total_rev = income.get("Total Revenue")

        if total_rev is not None:
             profit_insight += f"<p>Total Revenue for {report_year_str}: <strong>{format_currency(total_rev, currency, scale, 1)}</strong>.</p>"

        ni = income.get("Net Income")
        if ni is not None:
            ni_formatted = format_currency(ni, currency, scale, 1)
            ni_color_class = 'negative' if ni < 0 else 'positive'
            profit_insight += f"<p>Net Income/(Loss) for {report_year_str}: <strong class='{ni_color_class}'>{ni_formatted}</strong>.</p>"

            # Add historical comparison for Airport NI if available
            ni_hist_airport = hist_data.get("Net Income", {})
            ni_prev_airport = ni_hist_airport.get(prev_year_str) if prev_year_str in ni_hist_airport else None
            if ni_prev_airport is not None and isinstance(ni_prev_airport, (int, float)) and ni_prev_airport != 0:
                ni_change_pct_airport = (ni - ni_prev_airport) / abs(ni_prev_airport)
                trend_desc_airport = "increase" if ni_change_pct_airport > 0 else "decrease"
                profit_insight += f"<p>This represents a <span class='{'positive' if ni_change_pct_airport > 0 else 'negative'}'>{format_percentage(ni_change_pct_airport)} {trend_desc_airport}</span> from {prev_year_str}'s {format_currency(ni_prev_airport, currency, scale, 1)}.</p>"
            elif ni_prev_airport is not None:
                 profit_insight += f"<p>Comparison to previous year ({format_currency(ni_prev_airport, currency, scale, 1)}) available.</p>"


        if rev_contracts is not None or rev_lease is not None:
            profit_insight += "<p>Revenue Breakdown:</p><ul>"
            if rev_contracts is not None:
                rev_contracts_share = (rev_contracts / total_rev) if total_rev is not None and total_rev != 0 else None
                profit_insight += f"<li>Revenue from contracts with customers: {format_currency(rev_contracts, currency, scale, 1)} ({format_percentage(rev_contracts_share) if rev_contracts_share is not None else 'N/A'} share)</li>"
            if rev_lease is not None:
                rev_lease_share = (rev_lease / total_rev) if total_rev is not None and total_rev != 0 else None
                profit_insight += f"<li>Revenue from lease contracts: {format_currency(rev_lease, currency, scale, 1)} ({format_percentage(rev_lease_share) if rev_lease_share is not None else 'N/A'} share)</li>"
            profit_insight += "</ul>"
            profit_insight += "<p><em>Note: Detailed cost breakdown by Aeronautical/Non-Aeronautical segments not available in extracted data.</em></p>"

    # --- Profitability for Non-Airport types ---
    if company_type != 'airport' and ni_current is not None:
        ni_formatted = format_currency(ni_current, currency, scale, 1)
        ni_color_class = 'negative' if ni_current < 0 else 'positive'
        profit_insight += f"<p>Net Income{' attrib.' if ni_current_key.startswith('Net Income Attrib') else ''} for {report_year_str}: <strong class='{ni_color_class}'>{ni_formatted}</strong>.</p>"
        if ni_prev is not None and isinstance(ni_prev, (int, float)) and ni_prev != 0:
            ni_change_pct = (ni_current - ni_prev) / abs(ni_prev)
            trend_desc = "increase" if ni_change_pct > 0 else "decrease"
            profit_insight += f"<p>This represents a <span class='{'positive' if ni_change_pct > 0 else 'negative'}'>{format_percentage(ni_change_pct)} {trend_desc}</span> from {prev_year_str}'s {format_currency(ni_prev, currency, scale, 1)}.</p>"
            # Specific note for Sagicor 2024 data vs 2023 (with Ivari gain)
            ni_2023_specific = hist_data.get("Detailed Historical Data", {}).get("Net Income 2023 Reported")
            ivari_gain_2023 = hist_data.get("Detailed Historical Data", {}).get("Ivari Acquisition Gain 2023")
            ni_2023_norm = hist_data.get("Detailed Historical Data", {}).get("Net Income 2023 Normalized")
            if company_type == 'insurance' and report_year == 2024 and ni_2023_specific == ni_prev and ivari_gain_2023 and ni_2023_norm:
                 profit_insight += f"<p><i>Note: High {prev_year_str} result included one-time gain of {format_currency(ivari_gain_2023, currency, scale, 1)}. Normalized {prev_year_str} NI was {format_currency(ni_2023_norm, currency, scale, 1)}.</i></p>"
        elif ni_prev is not None:
             profit_insight += f"<p>Comparison to previous year ({format_currency(ni_prev, currency, scale, 1)}) available.</p>"

    elif company_type != 'airport':
         profit_insight += "<p>Net Income data not found.</p>"

    # EPS (Earnings Per Share)
    if core_eps_current is not None:
        profit_insight += f"<p>Core EPS: <strong>{core_eps_current * 100:.1f}¢</strong>"
        if core_eps_prev is not None and core_eps_prev != 0:
            core_eps_change_pct = (core_eps_current - core_eps_prev) / abs(core_eps_prev)
            core_eps_trend = "increased" if core_eps_change_pct > 0 else "decreased"
            profit_insight += f", a <span class='{'positive' if core_eps_change_pct > 0 else 'negative'}'>{format_percentage(core_eps_change_pct)} {core_eps_trend}</span> from {core_eps_prev * 100:.1f}¢.</p>"
        else:
            profit_insight += ". This represents the core profit generated per outstanding share.</p>"
    elif 'Diluted EPS' in ratios and ratios.get('Diluted EPS') is not None:
        eps_current = ratios.get('Diluted EPS')
        eps_hist = hist_data.get("Diluted EPS", {})
        eps_prev = eps_hist.get(prev_year_str) if prev_year_str in eps_hist else None
        profit_insight += f"<p>Diluted EPS: <strong>{format_currency(eps_current, currency, 1, 2)}</strong>" # Assuming EPS is in base currency units, not scaled cents
        if eps_prev is not None and eps_prev != 0:
            eps_change_pct = (eps_current - eps_prev) / abs(eps_prev)
            eps_trend = "increased" if eps_change_pct > 0 else "decreased"
            profit_insight += f", a <span class='{'positive' if eps_change_pct > 0 else 'negative'}'>{format_percentage(eps_change_pct)} {eps_trend}</span> from {format_currency(eps_prev, currency, 1, 2)}.</p>"
        else:
             profit_insight += ". This is the net income available to common shareholders per share, considering potential dilution.</p>"

    # ROE (Return on Equity) - Added Explanation
    if roe_current is not None:
        roe_label = roe_key.replace("Reported ", "").replace("Group ","") # Clean up label
        profit_insight += f"<p>{roe_label}: <strong>{format_percentage(roe_current)}</strong>. This measures the return generated on shareholders' investment (Net Income / Average Equity). Higher is often preferred, indicating efficient use of equity."
        if roe_prev is not None:
            roe_delta = roe_current - roe_prev
            roe_trend = "improved" if roe_delta > 0 else "declined"
            profit_insight += f" This <span class='{'positive' if roe_delta > 0 else 'negative'}'>{roe_trend}</span> from {format_percentage(roe_prev)} in {prev_year_str}.</p>"
        else:
            profit_insight += "</p>" # Close paragraph if no trend

    profit_insight += "</div>"
    insights.append(profit_insight)

    # --- Operational KPIs (Airport) ---
    if company_type == "airport":
        airport_kpi_insight = "<div class='metric-card'><h4>Operational KPIs</h4>"
        passengers = other_kpis.get("Total Passengers")
        movements = other_kpis.get("Aircraft Movements")
        cargo = other_kpis.get("Cargo Handled (kg)")
        mail = other_kpis.get("Mail Handled (kg)")

        passengers_hist = hist_data.get("Total Passengers", {})
        passengers_prev = passengers_hist.get(prev_year_str) if prev_year_str in passengers_hist else None
        movements_hist = hist_data.get("Aircraft Movements", {})
        movements_prev = movements_hist.get(prev_year_str) if prev_year_str in movements_hist else None

        if passengers is not None:
            airport_kpi_insight += f"<p>Total Passengers: <strong>{passengers:,.0f}</strong>."
            if passengers_prev is not None and passengers_prev != 0:
                pax_change = (passengers - passengers_prev) / abs(passengers_prev)
                pax_trend = "increased" if pax_change > 0 else "decreased"
                airport_kpi_insight += f" This <span class='{'positive' if pax_change > 0 else 'negative'}'>{pax_trend} by {format_percentage(pax_change)}</span> from {passengers_prev:,.0f} in {prev_year_str}.</p>"
            else:
                 airport_kpi_insight += "</p>"
        if movements is not None:
             airport_kpi_insight += f"<p>Aircraft Movements: <strong>{movements:,.0f}</strong>."
             if movements_prev is not None and movements_prev != 0:
                 mov_change = (movements - movements_prev) / abs(movements_prev)
                 mov_trend = "increased" if mov_change > 0 else "decreased"
                 airport_kpi_insight += f" This <span class='{'positive' if mov_change > 0 else 'negative'}'>{mov_trend} by {format_percentage(mov_change)}</span> from {movements_prev:,.0f} in {prev_year_str}.</p>"
             else:
                 airport_kpi_insight += "</p>"

        if cargo is not None: airport_kpi_insight += f"<p>Cargo Handled: <strong>{cargo:,.0f} kg</strong>.</p>"
        if mail is not None: airport_kpi_insight += f"<p>Mail Handled: <strong>{mail:,.0f} kg</strong>.</p>"
        if not any([passengers, movements, cargo, mail]):
            airport_kpi_insight += "<p>Key operational KPIs not found.</p>"
        airport_kpi_insight += "</div>"
        insights.append(airport_kpi_insight)


    # --- IFRS 17 Insights (Insurance) ---
    if company_type == "insurance":
        ifrs_metrics = data.get("IFRS17 Metrics", {})
        csm = ifrs_metrics.get("CSM") # Contractual Service Margin
        ra = ifrs_metrics.get("Risk Adjustment")
        isr = ifrs_metrics.get("Insurance Service Result")
        ifin = ifrs_metrics.get("Insurance Finance Income/Expenses") # Often negative (expense)

        csm_movements = data.get("IFRS17 Movements", {}).get("CSM Movement", {})
        ifrs_insight = "<div class='metric-card'><h4>IFRS 17 Key Metrics</h4>"
        has_ifrs_data = False
        csm_start = None
        csm_end = None

        if csm is not None:
            has_ifrs_data = True
            ifrs_insight += f"<p>Contractual Service Margin (CSM): <strong>{format_currency(csm, currency, scale, 1)}</strong>. This represents the expected unearned profit from existing insurance contracts."
            # Try to find start/end CSM from movements for trend analysis
            csm_start_key = next((k for k in csm_movements if 'beginning' in k.lower()), None)
            csm_end_key = next((k for k in csm_movements if 'end' in k.lower() or 'closing' in k.lower()), None)
            csm_start = csm_movements.get(csm_start_key)
            # Use BS CSM if end movement not found or if movement is zero but BS isn't
            csm_end = csm_movements.get(csm_end_key) if csm_movements.get(csm_end_key) is not None else csm

            if csm_start is not None and csm_end is not None and csm_start != 0:
                csm_change_pct = (csm_end - csm_start) / abs(csm_start)
                trend_desc = "increased" if csm_change_pct > 0 else "decreased"
                ifrs_insight += f" The CSM balance <span class='{'positive' if csm_change_pct > 0 else 'negative'}'>{trend_desc} by {format_percentage(csm_change_pct)}</span> during the period."

                # Key Drivers from movements
                new_biz_key = next((k for k in csm_movements if 'new business' in k.lower()), None)
                csm_amort_key = next((k for k in csm_movements if 'recognised' in k.lower() or 'amortization' in k.lower() or 'services provided' in k.lower()), None)
                new_biz = csm_movements.get(new_biz_key)
                csm_amort = csm_movements.get(csm_amort_key) # Usually negative in statements, positive when described as amortization
                drivers = []
                if new_biz is not None: drivers.append(f"New Business (+{format_currency(new_biz, currency, scale, 1)})")
                if csm_amort is not None: drivers.append(f"Amortization/Release ({format_currency(csm_amort, currency, scale, 1)})") # Show value as extracted

                if drivers: ifrs_insight += f" Key drivers include: {'; '.join(drivers)}."
                ifrs_insight += "</p>" # Close CSM paragraph
            elif csm_start is not None and csm_end is not None:
                 ifrs_insight += f" CSM movement details available but starting balance was zero or near zero.</p>" # Close CSM paragraph
            else:
                 ifrs_insight += f" CSM movement details (start/end values) not found for trend analysis.</p>" # Close CSM paragraph

        if ra is not None:
            has_ifrs_data = True
            ifrs_insight += f"<p>Risk Adjustment (RA): <strong>{format_currency(ra, currency, scale, 1)}</strong>. This buffer accounts for the uncertainty in the amount and timing of future cash flows.</p>"

        if isr is not None:
            has_ifrs_data = True
            ins_rev = income.get("Insurance Revenue")
            profit_margin = (isr / ins_rev) if ins_rev and ins_rev != 0 else None
            isr_formatted = format_currency(isr, currency, scale, 1)
            isr_color_class = 'negative' if isr < 0 else 'positive'
            ifrs_insight += f"<p>Insurance Service Result: <strong class='{isr_color_class}'>{isr_formatted}</strong>. This reflects the profit from insurance activities (premiums, claims, expenses) excluding investment components."
            if profit_margin is not None:
                ifrs_insight += f" Represents a <span class='{'positive' if profit_margin > 0 else 'negative'}'>{format_percentage(profit_margin)} margin</span> on insurance revenue.</p>"
            else:
                ifrs_insight += ".</p>" # Close paragraph

        if ifin is not None:
            has_ifrs_data = True
            ifin_desc = "expense" if ifin < 0 else "income"
            ifin_formatted = format_currency(ifin, currency, scale, 1)
            ifin_color_class = 'negative' if ifin < 0 else 'positive' # Expense is negative impact
            ifrs_insight += f"<p>Net Insurance Finance {ifin_desc}: <strong class='{ifin_color_class}'>{ifin_formatted}</strong>. Reflects the impact of time value of money and financial risks on insurance contract liabilities.</p>"

        if not has_ifrs_data:
            ifrs_insight += "<p>Core IFRS 17 metrics (CSM, RA, ISR, Net Finance Exp) not found in extracted data.</p>"

        ifrs_insight += "</div>"
        insights.append(ifrs_insight)

    # --- Financial Position & Ratios Insights ---
    balance_sheet = data.get("Balance Sheet", {})
    total_assets = balance_sheet.get("Total Assets")
    total_equity = balance_sheet.get("Total Equity") # Use harmonized Total Equity

    ratio_insight = "<div class='metric-card'><h4>Financial Position & Ratios</h4>"
    has_ratio_data = False # Flag to track if any ratios were found

    # Insurance Specific Ratios
    if company_type == "insurance":
        licat_key = "Group LICAT Ratio" if "Group LICAT Ratio" in ratios else "LICAT Ratio"
        licat = ratios.get(licat_key)
        comb_ratio = ratios.get("Combined Ratio") # P&C specific

        if licat is not None:
            has_ratio_data = True
            licat_pct = licat * 100
            # Status based on typical Canadian LICAT levels (adjust as needed for other regions)
            status = "Strong" if licat_pct > 150 else "Adequate" if licat_pct >= 120 else "Monitor"
            ratio_insight += f"<p>Capital ({licat_key}): <strong>{format_percentage(licat)}</strong> ({status}). This measures the insurer's available capital versus required capital, indicating its ability to absorb losses. Higher is generally better; regulatory minimums exist (e.g., ~100-120%), targets often higher.</p>"

        if comb_ratio is not None: # More relevant for P&C
            has_ratio_data = True
            status = "Profitable Underwriting" if comb_ratio < 1.0 else "Unprofitable Underwriting"
            ratio_insight += f"<p>Underwriting (Combined Ratio): <strong>{format_percentage(comb_ratio)}</strong> ({status}). This is a key measure of P&C underwriting profitability (Expenses + Losses) / Earned Premiums. Below 100% indicates profit from core underwriting; lower is better.</p>"

    # Banking Specific Ratios
    efficiency_ratio = ratios.get("Efficiency Ratio")
    net_interest_margin = ratios.get("Net Interest Margin")
    cet1_ratio = ratios.get("CET1 Ratio")
    if company_type == "banking":
        if net_interest_margin is not None:
            has_ratio_data = True
            nim_status = "Healthy" if net_interest_margin > 0.025 else "Moderate" if net_interest_margin > 0.015 else "Low" # Example thresholds
            ratio_insight += f"<p>Net Interest Margin (NIM): <strong>{format_percentage(net_interest_margin)}</strong> ({nim_status}). Measures profitability of lending activities (Interest Earned vs. Interest Paid relative to earning assets). Higher is generally better, but depends on business model and environment.</p>"
        if efficiency_ratio is not None:
            has_ratio_data = True
            # Status based on typical banking levels (Lower is better)
            status = "Efficient" if efficiency_ratio < 0.60 else "Moderate" if efficiency_ratio < 0.70 else "Needs Improvement"
            ratio_insight += f"<p>Efficiency Ratio: <strong>{format_percentage(efficiency_ratio)}</strong> ({status}). Measures operational efficiency (Non-Interest Expenses / Revenue). Lower indicates better cost control relative to income generation.</p>"
        if cet1_ratio is not None:
            has_ratio_data = True
            # Status based on typical regulatory requirements + buffers
            status = "Strong" if cet1_ratio > 0.12 else "Adequate" if cet1_ratio > 0.09 else "Monitor"
            ratio_insight += f"<p>CET1 Ratio: <strong>{format_percentage(cet1_ratio)}</strong> ({status}). Core measure of bank solvency (Highest Quality Capital / Risk-Weighted Assets). Higher indicates a stronger buffer against potential losses.</p>"

    # Technology Specific Ratios
    gross_margin_pct = ratios.get("Gross Margin %")
    op_margin_pct = ratios.get("Operating Margin %")
    if company_type == "technology":
        if gross_margin_pct is not None:
            has_ratio_data = True
            status = "High" if gross_margin_pct > 0.5 else "Moderate" if gross_margin_pct > 0.3 else "Low" # Example thresholds
            ratio_insight += f"<p>Gross Margin: <strong>{format_percentage(gross_margin_pct)}</strong> ({status}). Indicates profitability after direct costs of producing goods/services ((Revenue - COGS) / Revenue). Higher is generally better, reflecting pricing power or cost efficiency.</p>"
        if op_margin_pct is not None:
            has_ratio_data = True
            status = "Strong" if op_margin_pct > 0.20 else "Moderate" if op_margin_pct > 0.10 else "Low" # Example thresholds
            ratio_insight += f"<p>Operating Margin: <strong>{format_percentage(op_margin_pct)}</strong> ({status}). Shows profitability from core business operations before interest and taxes (Operating Income / Revenue). Higher indicates better operational efficiency and pricing power.</p>"

    # General / Common Ratios (Leverage, Liquidity, ROA)
    debt_equity_key = "Financial leverage ratio" if "Financial leverage ratio" in ratios else "Debt to Equity Ratio"
    debt_equity = ratios.get(debt_equity_key)
    if debt_equity is not None:
        has_ratio_data = True
        # General status based on common interpretations (Lower is generally less risky)
        status = "Low Leverage" if debt_equity < 0.5 else "Moderate Leverage" if debt_equity < 1.5 else "High Leverage"
        ratio_display_val = format_percentage(debt_equity) if debt_equity_key == "Financial leverage ratio" else f'{debt_equity:.2f}x'
        leverage_label = "Financial Leverage" if debt_equity_key == "Financial leverage ratio" else "Debt to Equity"
        ratio_insight += f"<p>Leverage ({leverage_label}): <strong>{ratio_display_val}</strong> ({status}). Measures financial leverage (Debt / Equity). A higher ratio indicates more reliance on debt financing relative to equity, which can amplify returns but also increases risk. Acceptable levels vary significantly by industry.</p>"

    current_ratio = ratios.get("Current Ratio")
    if current_ratio is not None:
        has_ratio_data = True
        # Status based on general liquidity health (Higher often better, but not too high)
        status = "Healthy Liquidity" if current_ratio > 1.5 else "Adequate Liquidity" if current_ratio >= 1.0 else "Low Liquidity"
        ratio_insight += f"<p>Current Ratio: <strong>{current_ratio:.2f}x</strong> ({status}). Measures short-term liquidity (Current Assets / Current Liabilities). Indicates the ability to cover near-term obligations with near-term assets. A ratio > 1 is typically desired; 1.5-2.0 often seen as healthy, though industry norms vary.</p>"

    debt_service_coverage = ratios.get("Debt Service Coverage Ratio")
    if debt_service_coverage is not None:
        has_ratio_data = True
        # Status based on ability to cover debt payments (Higher is better)
        status = "Strong Coverage" if debt_service_coverage > 2.0 else "Adequate Coverage" if debt_service_coverage >= 1.0 else "Weak Coverage"
        ratio_insight += f"<p>Debt Service Coverage Ratio: <strong>{debt_service_coverage:.2f}x</strong> ({status}). Measures the ability to service debt payments with operating income (e.g., Cash Flow Available for Debt Service / Total Debt Service payments). Crucial for lenders; > 1 indicates sufficient funds generated, higher provides more cushion.</p>"

    roa = ratios.get("ROA")
    if roa is not None:
        has_ratio_data = True
        # ROA Explanation Added
        roa_status = "Positive" if roa > 0 else "Negative" # Simple status for now
        ratio_insight += f"<p>Return on Assets (ROA): <strong>{format_percentage(roa)}</strong> ({roa_status}). Measures how efficiently assets are used to generate profit (Net Income / Average Total Assets). Higher is generally better, but benchmarks vary widely by industry (e.g., asset-heavy industries typically have lower ROA than asset-light ones).</p>"
    # ROE is handled in Profitability Section

    if not has_ratio_data:
        ratio_insight += "<p>Key financial ratios used for detailed assessment were not found or calculated from the extracted data.</p>"

    # Basic Balance Sheet Summary
    if total_assets is not None and total_equity is not None and total_assets !=0 :
        equity_pct = (total_equity / total_assets)
        ratio_insight += f"<p>Balance Sheet Summary: Total Assets: {format_currency(total_assets, currency, scale, 1)}, Total Equity: {format_currency(total_equity, currency, scale, 1)}. Equity represents <span class='{'positive' if equity_pct > 0.2 else 'negative'}'>{format_percentage(equity_pct)}</span> of total assets.</p>" # Example threshold for color

    ratio_insight += "</div>"
    insights.append(ratio_insight)


    # --- Segment Performance Insights ---
    segments = data.get("Segments", {})
    if segments:
        segment_insight = "<div class='metric-card'><h4>Segment Performance</h4>"
        # Determine the primary performance metric based on company type or data availability
        perf_metric = 'Core Earnings' # Default for Insurance (Sagicor Sample)
        if company_type == 'banking': perf_metric = 'Net Income'
        elif company_type == 'technology': perf_metric = 'Net Sales'
        elif company_type == 'airport': perf_metric = 'Revenue' # Airports often focus on Revenue per segment

        # Check if the default metric exists in the first segment's data
        first_seg_data = next(iter(segments.values()), {})
        if not isinstance(first_seg_data, dict) or perf_metric not in first_seg_data:
             # Fallback logic if default metric not found
            if isinstance(first_seg_data, dict):
                if 'Net Sales' in first_seg_data: perf_metric = 'Net Sales'
                elif 'Revenue' in first_seg_data: perf_metric = 'Revenue'
                elif 'Operating Income' in first_seg_data: perf_metric = 'Operating Income'
                elif 'Net Income' in first_seg_data: perf_metric = 'Net Income'
                else: perf_metric = next(iter(first_seg_data.keys()), 'Value') # Last resort: take the first key
            else: perf_metric = 'Value' # If segment data is just numbers, not dicts

        segment_insight += f"<p><em>Analysis based on segment '{perf_metric}'.</em></p>"

        try:
            valid_segments = {}
            # Handle cases where segment data might be simple values or dicts
            first_value_type = type(next(iter(segments.values()), None))

            if first_value_type == dict:
                valid_segments = {k: v.get(perf_metric) for k, v in segments.items()
                                  if isinstance(v, dict) and v.get(perf_metric) is not None and isinstance(v.get(perf_metric), (int, float))}
                 # If primary metric yielded no valid data, try a common fallback like Net Income
                if not valid_segments:
                    fallback_metric = 'Net Income'
                    valid_segments = {k: v.get(fallback_metric) for k, v in segments.items()
                                     if isinstance(v, dict) and v.get(fallback_metric) is not None and isinstance(v.get(fallback_metric), (int, float))}
                    if valid_segments: perf_metric = fallback_metric # Update metric label if fallback used

            elif issubclass(first_value_type, (int, float)):
                valid_segments = {k: v for k, v in segments.items() if isinstance(v, (int, float))}
                perf_metric = "Value" # Label appropriately

            if valid_segments:
                # Exclude Head Office / Other / Corporate segments from commentary comparison if they exist
                commentary_segments = {k: v for k, v in valid_segments.items()
                                       if not re.search("Head Office|Other|Corporate|Consolidated|Adjustments", k, re.IGNORECASE)}

                if commentary_segments:
                    # Find top and worst performing operating segments
                    top_segment = max(commentary_segments, key=commentary_segments.get)
                    worst_segment = min(commentary_segments, key=commentary_segments.get)

                    top_val = commentary_segments[top_segment]
                    worst_val = commentary_segments[worst_segment]

                    top_val_fmt = format_currency(top_val, currency, scale, 1)
                    worst_val_fmt = format_currency(worst_val, currency, scale, 1)

                    top_color = 'positive' if top_val >= 0 else 'negative'
                    worst_color = 'positive' if worst_val >= 0 else 'negative'

                    segment_insight += f"<p><strong>{top_segment}</strong> was the strongest contributor based on {perf_metric} (<span class='{top_color}'>{top_val_fmt}</span>)."

                    # Only show worst if it's different from the best and has a meaningful value
                    if top_segment != worst_segment:
                         segment_insight += f" Performance in <strong>{worst_segment}</strong> (<span class='{worst_color}'>{worst_val_fmt}</span>) may warrant review.</p>"
                    else:
                        segment_insight += "</p>" # Close paragraph if only one segment or all same

                    # Optional: Add total contribution check if possible
                    total_op_segments = sum(commentary_segments.values())
                    total_all_segments = sum(v for v in valid_segments.values() if v is not None) # Sum all numerical segment values
                    if abs(total_all_segments - total_op_segments) > 0.01: # Check if non-operating segments have impact
                         corp_impact = total_all_segments - total_op_segments
                         corp_impact_fmt = format_currency(corp_impact, currency, scale, 1)
                         corp_color = 'positive' if corp_impact >= 0 else 'negative'
                         segment_insight += f"<p>Corporate/Other segments contributed <span class='{corp_color}'>{corp_impact_fmt}</span> to the total.</p>"

                else:
                    # Case where only Head Office/Other/Corporate segments had data
                    segment_insight += f"<p>No core operating segments found with '{perf_metric}' data for comparison. Data may only be available for Corporate/Other segments.</p>"
            else:
                segment_insight += f"<p>No displayable numeric data found for segments using metric '{perf_metric}'.</p>"

        except Exception as e:
            st.warning(f"Error processing segment data: {e}") # Log error for debugging
            segment_insight += "<p>Could not determine top/worst segment due to an error or data format issue.</p>"

        segment_insight += "</div>"
        insights.append(segment_insight)

    # --- AI Recommendations ---
    insights.append("""<div class='metric-card'>
        <h4>AI Recommendations & Next Steps</h4>
        <p><strong>Recommendations (Template):</strong>
            <ul>
                <li>Investigate drivers behind changes in key profitability ratios (ROE, ROA, Margins).</li>
                <li>Analyze trends in segment performance and contribution (if applicable).</li>
                <li>Benchmark leverage (Debt/Equity) and liquidity (Current Ratio) against relevant industry peers.</li>
                <li>For Insurers: Examine CSM movement drivers (New Business vs. Amortization vs. Experience).</li>
                <li>For Banks: Monitor NIM, Efficiency Ratio, and CET1 Ratio trends.</li>
                <li>For Airports: Correlate operational KPIs (passenger traffic) with financial results.</li>
            </ul>
        </p>
        <p><strong>Next Steps (Future Capabilities):</strong>
            <ul>
                <li>Driver-based forecast adjustments using historical data and external factors.</li>
                <li>Scenario analysis (e.g., impact of interest rate changes, traffic volume changes).</li>
                <li>Automated variance analysis against budget or prior periods.</li>
                <li>Natural Language Querying ("Ask the AI") for specific data points or trends.</li>
            </ul>
        </p>
    </div>""")

    return insights

# =============================================
# 📊 VISUALIZATION COMPONENTS
# =============================================
def create_waterfall_chart(df: pd.DataFrame, title: str, y_axis_title: str) -> Optional[go.Figure]:
    df['value'] = pd.to_numeric(df['value'], errors='coerce'); df = df.dropna(subset=['value'])
    if df.empty: st.info(f"No data for {title} waterfall."); return None
    metadata = st.session_state.financial_data.get('metadata', {}); currency = metadata.get('currency', DEFAULT_CURRENCY); scale_val = metadata.get('scale', 1); scale_val = 1 if not isinstance(scale_val, (int, float)) or scale_val == 0 else scale_val
    text_labels = [format_currency(v, currency, scale_val, 1) for v in df['value']]
    font_color = "#e0e0e0" if st.session_state.theme_mode == "dark" else "#212529"
    fig = go.Figure(go.Waterfall(name=title, orientation="v", measure=df['measure'].tolist(), x=df['label'].tolist(), text= text_labels, textposition="outside", textfont=dict(size=12, color=font_color), y=df['value'].tolist(), connector={"line": {"color": "rgb(63, 63, 63)"}}, increasing={"marker": {"color": "#28a745"}}, decreasing={"marker": {"color": "#dc3545"}}, totals={"marker": {"color": "#0d6efd"}} ))
    min_y = df['value'].min(); max_y = df['value'].max(); cumulative_max = 0; current_sum = 0
    for i, measure in enumerate(df['measure']): val = df['value'].iloc[i]; current_sum = val if measure in ['absolute', 'total'] else current_sum + val; cumulative_max = max(cumulative_max, current_sum); min_y = min(min_y, current_sum)
    max_y = max(max_y, cumulative_max); range_padding = (max_y - min_y) * 0.25; final_min_y = min_y - range_padding; final_max_y = max_y + range_padding
    if abs(final_min_y) < abs(min_y * 0.1): final_min_y = min_y - abs(max_y * 0.1)
    if abs(final_max_y) < abs(max_y * 0.1): final_max_y = max_y + abs(min_y * 0.1)
    fig.update_layout(title=title, showlegend=False, height=500, yaxis_title=y_axis_title, xaxis_tickangle=-45, yaxis=dict(range=[final_min_y, final_max_y], tickformat=',.0f'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=font_color))
    return fig

def create_gauge_chart(value: Optional[float], min_val: float, max_val: float, title: str, target: Optional[float] = None, suffix: str = "", higher_is_better: bool = True) -> Optional[go.Figure]:
    gauge_title = title; display_value = value; number_font_color = "#ffffff" if st.session_state.theme_mode == "dark" else "#1a1a1a"; title_font_color = "#f0f0f0" if st.session_state.theme_mode == "dark" else "#3f3f3f"; general_font_color = "#e0e0e0" if st.session_state.theme_mode == "dark" else "#212529"
    delta_config = None; threshold_config = None; steps_config = None; bar_color = "#adb5bd" if st.session_state.theme_mode == "dark" else "#ced4da"
    if value is None or not isinstance(value, (int, float)) or pd.isna(value):
        gauge_title = title + " (N/A)"; display_value = min_val; number_config = {'suffix': suffix, 'font': {'size': 18, 'color': "#adb5bd" if st.session_state.theme_mode == "dark" else "#6c757d"}}; indicator_mode = "gauge"
    else:
        indicator_mode="gauge+number" + ("+delta" if target is not None else ""); number_config = {'suffix': suffix, 'font': {'size': 22, 'color': number_font_color}}; delta_config = {'reference': target, 'suffix': suffix} if target is not None else None
        low_threshold = min_val + (max_val - min_val) * 0.4; high_threshold = min_val + (max_val - min_val) * 0.8; mid_point = target if target is not None else min_val + (max_val - min_val) / 2; effective_target = target if target is not None else mid_point
        if higher_is_better: bar_color = "#198754" if display_value >= high_threshold or (target is not None and display_value >= effective_target) else "#ffc107" if display_value >= low_threshold else "#dc3545"
        else: bar_color = "#198754" if display_value <= low_threshold or (target is not None and display_value <= effective_target) else "#ffc107" if display_value <= high_threshold else "#dc3545"
        steps_config = [{'range': [min_val, low_threshold], 'color': 'rgba(220, 53, 69, 0.1)'}, {'range': [low_threshold, high_threshold], 'color': 'rgba(255, 193, 7, 0.1)'}, {'range': [high_threshold, max_val], 'color': 'rgba(25, 135, 84, 0.1)'}]; threshold_config = {'line': {'color': "#fd7e14", 'width': 3}, 'thickness': 0.75, 'value': target} if target is not None else None
    try:
        fig = go.Figure(go.Indicator(mode=indicator_mode, value=display_value, number=number_config, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': gauge_title, 'font': {'size': 16, 'color': title_font_color}}, delta=delta_config, gauge={'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue", 'ticksuffix': suffix}, 'bar': {'color': bar_color, 'thickness': 0.3}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 1, 'bordercolor': "#adb5bd" if st.session_state.theme_mode == "dark" else "#ced4da", 'steps': steps_config, 'threshold': threshold_config })); fig.update_layout(height=250, margin=dict(t=50, b=10, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=general_font_color) ); return fig
    except Exception as e: print(f"Error creating gauge '{title}': {e}"); print(f"Inputs: value={value}, display_value={display_value}, min_val={min_val}, max_val={max_val}, target={target}, suffix={suffix}, higher_is_better={higher_is_better}"); return None

def display_ifrs17_waterfalls(data: Dict[str, Any]) -> None:
    st.markdown("Detailed IFRS 17 reconciliations based on extracted/sample data."); st.divider()
    metadata = data.get("metadata", {}); currency = metadata.get("currency", DEFAULT_CURRENCY); scale_val = metadata.get('scale'); scale_val = 1 if not isinstance(scale_val, (int, float)) or scale_val == 0 else scale_val
    scale_desc = "M" if scale_val == 1_000_000 else "k" if scale_val == 1000 else ""; y_axis_title = f"Value ({currency}{' ' + scale_desc if scale_desc else ''})"
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("###### CSM Movement Analysis"); st.caption("Shows changes in Contractual Service Margin from start to end of period.")
        csm_movements = data.get("IFRS17 Movements", {}).get("CSM Movement", {});
        if csm_movements and len(csm_movements) > 1:
            df_csm = pd.DataFrame(list(csm_movements.items()), columns=['Item', 'Value']).dropna(subset=['Value'])
            if not df_csm.empty:
                 df_csm_display = df_csm.copy(); df_csm_display['Value'] = df_csm_display['Value'].apply(lambda x: format_currency(x, currency, scale_val, 1))
                 val_col_name = f"Value ({scale_desc}{currency})" if scale_desc else f"Value ({currency})"
                 column_config = { val_col_name: st.column_config.TextColumn(val_col_name, help=f"Values in {scale_desc} {currency}") }
                 st.dataframe(df_csm_display.rename(columns={'Value': val_col_name}).set_index('Item'), use_container_width=True, height=350, column_config=column_config)
                 measures = ['relative'] * len(df_csm);
                 for i, label in enumerate(df_csm['Item']): measures[i] = 'absolute' if 'beginning of period' in label.lower() or i == 0 or 'end of period' in label.lower() or 'closing balance' in label.lower() else 'relative'
                 df_csm['measure'] = measures; df_csm = df_csm.rename(columns={'Item': 'label', 'Value': 'value'})
                 fig_csm = create_waterfall_chart(df_csm, "CSM Movement Waterfall", y_axis_title);
                 if fig_csm: st.plotly_chart(fig_csm, use_container_width=True)
            else: st.info("CSM Movement data empty after cleaning.")
        else: st.info("CSM Movement data unavailable or insufficient.")
    with col2:
        st.markdown("###### Net Income Breakdown"); st.caption("Illustrates components contributing to Net Income.")
        income_breakdown = data.get("IFRS17 Movements", {}).get("Income Breakdown", {})
        if income_breakdown and len(income_breakdown) > 1:
            df_inc = pd.DataFrame(list(income_breakdown.items()), columns=['label', 'value']).dropna(subset=['value'])
            if not df_inc.empty: measures = ['relative'] * len(df_inc); start_labels = ["Insurance Revenue", "Insurance Service Result"]; total_labels = ["Income before", "Net Income for", "Net Income Attributable"];
            for i, label in enumerate(df_inc['label']): label_lower = label.lower(); measures[i] = 'absolute' if i == 0 and any(start in label for start in start_labels) else 'total' if any(total in label_lower for total in total_labels) else 'relative'
            if not any(total in df_inc.iloc[-1]['label'].lower() for total in total_labels): measures[-1] = 'total'
            df_inc['measure'] = measures; fig_inc = create_waterfall_chart(df_inc, "Income Breakdown", y_axis_title)
            if fig_inc: st.plotly_chart(fig_inc, use_container_width=True)
            else: st.info("Income Breakdown data empty.")
        else: st.info("Income breakdown data unavailable.")

def display_key_metrics_and_ratios(data: Dict[str, Any]) -> None:
    st.subheader("📊 Key Performance Indicators"); st.caption("Snapshot of core financial metrics for the reporting period.")
    metadata = data.get("metadata", {}); currency = metadata.get("currency", DEFAULT_CURRENCY); scale = metadata.get("scale"); company_type = metadata.get("company_type", "general")
    income = data.get("Income Statement", {}); balance_sheet = data.get("Balance Sheet", {}); ratios = data.get("Ratios", {}); ifrs_metrics = data.get("IFRS17 Metrics", {}); other_kpis = data.get("Other KPIs", {})
    color_positive_light = "#198754"; color_negative_light = "#dc3545"; color_default_light = "#1a1a1a"
    color_positive_dark = "#4ade80"; color_negative_dark = "#f87171"; color_default_dark = "#ffffff"
    m_cols = st.columns(4)

    def display_markdown_kpi(column, label, value, currency, scale, help_text=None, is_currency=True):
        with column:
            st.markdown(f"**{label}**", unsafe_allow_html=True)
            if is_currency:
                value_formatted = format_currency(value, currency, scale, 1) if value is not None else "N/A"
            else:
                value_formatted = f"{value:,.0f}" if isinstance(value, (int, float)) else "N/A"
            is_negative = isinstance(value, (int, float)) and value < 0
            is_positive = isinstance(value, (int, float)) and value > 0
            is_net_income_label = "Net Income" in label or "Loss" in label
            if st.session_state.theme_mode == "dark":
                color = color_negative_dark if is_negative else (color_positive_dark if is_positive else color_default_dark)
            else:
                if is_net_income_label and is_negative:
                    color = color_negative_light
                else:
                    color = color_default_light
            st.markdown(f"<div class='kpi-value' style='color:{color};'>{value_formatted}</div>", unsafe_allow_html=True)
            if help_text: st.caption(help_text)

    if company_type == 'banking':
        nii = income.get("Net Interest Income"); display_markdown_kpi(m_cols[0], "Net Interest Income", nii, currency, scale)
        total_rev = income.get("Total Revenue"); display_markdown_kpi(m_cols[1], "Total Revenue", total_rev, currency, scale)
    elif company_type == 'airport':
        total_rev = income.get("Total Revenue"); display_markdown_kpi(m_cols[0], "Total Revenue", total_rev, currency, scale)
        ni = income.get("Net Income"); display_markdown_kpi(m_cols[1], "Net Income/(Loss)", ni, currency, scale)
        passengers = other_kpis.get("Total Passengers"); display_markdown_kpi(m_cols[2], "Passengers", passengers, currency, scale, is_currency=False)
        movements = other_kpis.get("Aircraft Movements"); display_markdown_kpi(m_cols[3], "Movements", movements, currency, scale, is_currency=False)
    else:
        ni_key = "Net Income Attributable to Shareholders" if "Net Income Attributable to Shareholders" in income else "Net Income"; ni = income.get(ni_key); display_markdown_kpi(m_cols[0], "Net Income Attrib.", ni, currency, scale)
        rev_key = "Insurance Revenue" if company_type == 'insurance' and "Insurance Revenue" in income else "Net Sales" if "Net Sales" in income else "Revenue"; rev = income.get(rev_key); display_markdown_kpi(m_cols[1], "Revenue", rev, currency, scale)

    if company_type != 'airport':
        equity_key = "Shareholders' Equity" if "Shareholders' Equity" in balance_sheet else "Total Equity"; equity = balance_sheet.get(equity_key); display_markdown_kpi(m_cols[2], "Equity", equity, currency, scale)
        if company_type == "insurance": csm = ifrs_metrics.get("CSM"); display_markdown_kpi(m_cols[3], "CSM", csm, currency, scale, help_text="Contractual Service Margin (IFRS 17)")
        elif company_type == "banking": loans = balance_sheet.get("Total Loans Net"); display_markdown_kpi(m_cols[3], "Total Loans (Net)", loans, currency, scale)
        else: assets = balance_sheet.get("Total Assets"); display_markdown_kpi(m_cols[3], "Total Assets", assets, currency, scale)

    if company_type == 'airport':
        st.divider()
        st.subheader("✈️ Airport Revenue Breakdown")
        st.caption("Breakdown of revenue sources.")
        rev_contracts = income.get("Revenue from contracts with customers")
        rev_lease = income.get("Revenue for lease contracts")
        total_rev = income.get("Total Revenue")
        seg_col1, seg_col2 = st.columns(2)
        with seg_col1:
            st.markdown("##### Revenue from Contracts")
            if rev_contracts is not None:
                 rev_contracts_share = (rev_contracts / total_rev) if total_rev is not None and total_rev != 0 else None
                 st.markdown(f"<div class='segment-kpi-label'>Revenue:</div> <div class='segment-kpi-value'>{format_currency(rev_contracts, currency, scale, 1)}</div>", unsafe_allow_html=True)
                 st.markdown(f"<div class='segment-kpi-label'>Revenue Share:</div> <div class='segment-kpi-value'>{format_percentage(rev_contracts_share) if rev_contracts_share is not None else 'N/A'}</div>", unsafe_allow_html=True)
            else:
                 st.info("Revenue from Contracts data not available.")
        with seg_col2:
            st.markdown("##### Revenue from Leases")
            if rev_lease is not None:
                 rev_lease_share = (rev_lease / total_rev) if total_rev is not None and total_rev != 0 else None
                 st.markdown(f"<div class='segment-kpi-label'>Revenue:</div> <div class='segment-kpi-value'>{format_currency(rev_lease, currency, scale, 1)}</div>", unsafe_allow_html=True)
                 st.markdown(f"<div class='segment-kpi-label'>Revenue Share:</div> <div class='segment-kpi-value'>{format_percentage(rev_lease_share) if rev_lease_share is not None else 'N/A'}</div>", unsafe_allow_html=True)
            else:
                 st.info("Revenue from Leases data not available.")

    st.divider(); st.subheader("📈 Financial Ratios"); st.caption(""" Key ratios provide insights into profitability, efficiency, and solvency/capital adequacy. Gauges show performance against typical ranges or targets (targets are illustrative). """)
    g_cols = st.columns(3); gauge_count = 0
    def display_gauge(ratio_key, title, min_v, max_v, target_v, suffix_v, is_higher_better):
        nonlocal gauge_count; raw_value = ratios.get(ratio_key); display_value = None; target_display = None
        if isinstance(raw_value, (int, float)) and not pd.isna(raw_value):
            if suffix_v == "%": display_value = raw_value * 100.0; target_display = target_v
            else: display_value = raw_value; target_display = target_v
        with g_cols[gauge_count % 3]: fig = create_gauge_chart(display_value, min_v, max_v, title, target_display, suffix_v, is_higher_better);
        if fig: st.plotly_chart(fig, use_container_width=True)
        else: g_cols[gauge_count % 3].container(height=250).write(f"{title}: N/A")
        gauge_count += 1

    ratio_configs = {
        "insurance": [ {"key": "Reported ROE", "title": "Return on Equity", "min": 0, "max": 20, "target": 13, "suffix": "%", "higher_better": True}, {"key": "ROA", "title": "Return on Assets", "min": 0, "max": 2, "target": 1, "suffix": "%", "higher_better": True}, {"key": "Group LICAT Ratio", "title": "LICAT Ratio", "min": 100, "max": 200, "target": 150, "suffix": "%", "higher_better": True}, ],
        "banking": [ {"key": "Net Interest Margin", "title": "Net Interest Margin", "min": 1, "max": 5, "target": 3, "suffix": "%", "higher_better": True}, {"key": "Efficiency Ratio", "title": "Efficiency Ratio", "min": 40, "max": 80, "target": 55, "suffix": "%", "higher_better": False}, {"key": "CET1 Ratio", "title": "CET1 Ratio", "min": 8, "max": 16, "target": 12, "suffix": "%", "higher_better": True}, ],
        "technology": [ {"key": "Gross Margin %", "title": "Gross Margin", "min": 20, "max": 70, "target": 45, "suffix": "%", "higher_better": True}, {"key": "Operating Margin %", "title": "Operating Margin", "min": 0, "max": 50, "target": 25, "suffix": "%", "higher_better": True}, {"key": "ROE", "title": "Return on Equity", "min": 0, "max": 180, "target": 100, "suffix": "%", "higher_better": True}, ],
        "airport": [ {"key": "ROE", "title": "Return on Equity", "min": -20, "max": 20, "target": 5, "suffix": "%", "higher_better": True}, {"key": "ROA", "title": "Return on Assets", "min": -5, "max": 5, "target": 1, "suffix": "%", "higher_better": True}, {"key": "Debt to Equity Ratio", "title": "Debt to Equity", "min": 0, "max": 3, "target": 1.5, "suffix": "x", "higher_better": False}, {"key": "Current Ratio", "title": "Current Ratio", "min": 0, "max": 5, "target": 2, "suffix": "x", "higher_better": True}, {"key": "Debt Service Coverage Ratio", "title": "Debt Service Coverage", "min": -2, "max": 4, "target": 1.5, "suffix": "x", "higher_better": True},],
        "general": [ {"key": "ROE", "title": "Return on Equity", "min": -20, "max": 30, "target": 15, "suffix": "%", "higher_better": True}, {"key": "ROA", "title": "Return on Assets", "min": -5, "max": 15, "target": 5, "suffix": "%", "higher_better": True}, {"key": "Debt to Equity Ratio", "title": "Debt to Equity", "min": 0, "max": 3, "target": 1, "suffix": "x", "higher_better": False}, ]}
    selected_ratios = ratio_configs.get(company_type, ratio_configs["general"])
    displayed_count = 0
    for config in selected_ratios:
        ratio_val = ratios.get(config["key"])
        if ratio_val is None and config["key"] == "ROE" and "Reported ROE" in ratios: ratio_val = ratios.get("Reported ROE")
        if ratio_val is None and config["key"] == "Group LICAT Ratio" and "LICAT Ratio" in ratios: ratio_val = ratios.get("LICAT Ratio")
        if ratio_val is not None:
            if displayed_count < 3: display_gauge(config["key"], config["title"], config["min"], config["max"], config["target"], config["suffix"], config["higher_better"]); displayed_count += 1
            else: break
    while gauge_count < 3: g_cols[gauge_count % 3].container(height=250); gauge_count += 1

def display_segment_map(data: Dict[str, Any]) -> None:
    st.markdown("Geographical and comparative performance of business segments.")
    st.divider()
    segments = data.get("Segments", {}); metadata = data.get("metadata", {});
    currency = metadata.get("currency", DEFAULT_CURRENCY); scale_val = metadata.get("scale", 1);
    if not isinstance(scale_val, (int, float)) or scale_val == 0: scale_val = 1
    scale_desc = "M" if scale_val == 1_000_000 else "k" if scale_val == 1000 else "" if scale_val == 1 else f"x{scale_val}"
    font_color = "#e0e0e0" if st.session_state.theme_mode == "dark" else "#212529"
    company_type = metadata.get("company_type", "general")

    if company_type == 'airport' and not segments:
        st.subheader("✈️ Revenue Breakdown")
        st.caption("Contribution of different revenue streams based on Income Statement data.")
        income = data.get("Income Statement", {})
        rev_contracts = income.get("Revenue from contracts with customers")
        rev_lease = income.get("Revenue for lease contracts")
        bar_chart_data = []
        if rev_contracts is not None: bar_chart_data.append({"Category": "Revenue from Contracts", "Value": rev_contracts})
        if rev_lease is not None: bar_chart_data.append({"Category": "Revenue from Leases", "Value": rev_lease})
        if bar_chart_data:
             df_bar = pd.DataFrame(bar_chart_data);
             df_bar = df_bar.sort_values('Value', ascending=False)
             color_scale = px.colors.sequential.Viridis
             color_midpoint = None
             fig_bar = px.bar(df_bar, x='Category', y='Value', title="Revenue by Source",
                             labels={'Value': f'Revenue ({currency}{scale_desc})'},
                             text='Value', color='Value',
                             color_continuous_scale=color_scale,
                             color_continuous_midpoint=color_midpoint)
             fig_bar.update_traces(texttemplate=f'{currency} %{{y:,.1f}}{scale_desc}', textposition='outside', textfont_size=12)
             fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color=font_color), coloraxis_showscale=False)
             st.plotly_chart(fig_bar, use_container_width=True)
        else:
             st.info("Revenue breakdown data unavailable for chart display.")
        return

    if not segments:
       st.info("Segment data unavailable or not applicable for this company type."); return

    if company_type == 'insurance': perf_metric = "Core Earnings"
    elif company_type == 'banking': perf_metric = "Net Income"
    elif company_type == 'technology': perf_metric = "Net Sales"
    elif company_type == 'airport': perf_metric = "Revenue"
    else: perf_metric = "Net Sales"

    first_seg_data = next(iter(segments.values()), {})
    if not isinstance(first_seg_data, dict) or perf_metric not in first_seg_data:
        if isinstance(first_seg_data, dict) and 'Net Sales' in first_seg_data: perf_metric = 'Net Sales'
        elif isinstance(first_seg_data, dict) and 'Operating Income' in first_seg_data: perf_metric = 'Operating Income'
        elif isinstance(first_seg_data, dict) and 'Net Income' in first_seg_data: perf_metric = 'Net Income'
        elif isinstance(first_seg_data, dict): perf_metric = next(iter(first_seg_data.keys()), 'Value')
        else: perf_metric = 'Value'

    st.subheader("📊 Segment Contribution")
    st.caption(f"Compares the contribution of each segment based on {perf_metric}. Color indicates profitability/value (Green=Positive, Red=Negative).")
    bar_chart_data = []
    for name, metrics in segments.items():
        if isinstance(metrics, dict): value = metrics.get(perf_metric);
        elif isinstance(metrics, (int, float)): value = metrics
        else: value = None
        if isinstance(value, (int, float)): bar_chart_data.append({"Segment": name, "Value": value})

    if bar_chart_data:
        df_bar = pd.DataFrame(bar_chart_data);
        df_bar_display = df_bar[~df_bar['Segment'].str.contains("Head Office|Other|Corporate", na=False, case=False)].copy()
        if df_bar_display.empty: df_bar_display = df_bar
        df_bar_display = df_bar_display.sort_values('Value', ascending=False)
        min_val = df_bar_display['Value'].min(); max_val = df_bar_display['Value'].max()
        if min_val >= 0 :
            color_scale = px.colors.sequential.Greens
            color_midpoint = None
        elif max_val <= 0:
             color_scale = px.colors.sequential.Reds_r
             color_midpoint = None
        else:
             color_scale = px.colors.diverging.RdYlGn
             color_midpoint = 0
        fig_bar = px.bar(df_bar_display, x='Segment', y='Value', title=f"{perf_metric} by Segment",
                       labels={'Value': f'{perf_metric} ({currency}{scale_desc})'},
                       text='Value', color='Value',
                       color_continuous_scale=color_scale,
                       color_continuous_midpoint=color_midpoint)
        fig_bar.update_traces(texttemplate=f'{currency} %{{y:,.1f}}{scale_desc}', textposition='outside', textfont_size=12)
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color=font_color), coloraxis_showscale=False if color_midpoint is None else True)
        st.plotly_chart(fig_bar, use_container_width=True)
    else: st.info(f"No numeric data available for segment contribution bar chart ({perf_metric}).")

    mappable_types = ['insurance', 'banking', 'technology']
    if company_type in mappable_types:
        st.divider(); st.subheader("🗺️ Segment Performance Map")
        st.caption(f"Geographic distribution and performance magnitude ({perf_metric} - size/color). Hover for details.")
        segment_geo_map = { "Sagicor Canada": {"iso_alpha": "CAN", "lat": 56.1304, "lon": -106.3468}, "Sagicor Life USA": {"iso_alpha": "USA", "lat": 38.9637, "lon": -95.7129}, "Sagicor Jamaica": {"iso_alpha": "JAM", "lat": 18.1096, "lon": -77.2975}, "Sagicor Life": {"iso_alpha": "BRB", "lat": 13.1939, "lon": -59.5432}, "Americas": {"iso_alpha": "USA", "lat": 38.9637, "lon": -95.7129}, "Europe": {"iso_alpha": "FRA", "lat": 46.2276, "lon": 2.2137 }, "Greater China": {"iso_alpha": "CHN", "lat": 35.8617, "lon": 104.1954}, "Japan": {"iso_alpha": "JPN", "lat": 36.2048, "lon": 138.2529}, "Rest of Asia Pacific": {"iso_alpha": "AUS", "lat": -25.2744, "lon": 133.7751},"Canadian Personal and Business Banking": {"iso_alpha": "CAN", "lat": 56.1304, "lon": -106.3468},"Canadian Commercial Banking and Wealth Management": {"iso_alpha": "CAN", "lat": 56.1304, "lon": -106.3468},"U.S. Commercial Banking and Wealth Management": {"iso_alpha": "USA", "lat": 38.9637, "lon": -95.7129},"Capital Markets and Direct Financial Services": {"iso_alpha": "CAN", "lat": 56.1304, "lon": -106.3468}}
        segment_map_data = []; values = []
        for name, metrics in segments.items():
            geo_info = segment_geo_map.get(name)
            if geo_info:
                if isinstance(metrics, dict): value = metrics.get(perf_metric); ni_value = metrics.get("Net Income"); asset_value = metrics.get("Total Assets")
                elif isinstance(metrics, (int, float)): value = metrics; ni_value = None; asset_value = None
                else: value = None; ni_value = None; asset_value = None
                if isinstance(value, (int, float)):
                     values.append(abs(value))
                     segment_map_data.append({
                          "Segment": name, "Value": value, "ISO": geo_info["iso_alpha"], "lat": geo_info["lat"], "lon": geo_info["lon"],
                          "Formatted Value": format_currency(value, currency, scale_val, 1),
                          "AbsValue": abs(value), "Metric": perf_metric,
                          "Formatted NI": format_currency(ni_value, currency, scale_val, 1) if ni_value is not None else "N/A",
                          "Formatted Assets": format_currency(asset_value, currency, scale_val, 1) if asset_value is not None else "N/A"
                     })
        if not segment_map_data:
           st.info(f"No mappable geographic data found for metric: '{perf_metric}'.")
        else:
           df_map = pd.DataFrame(segment_map_data); max_abs_value = max(values) if values else 1; min_marker_size = 10; max_marker_size = 45
           df_map['Size'] = df_map['AbsValue'].apply(lambda x: max(min_marker_size, (x / max_abs_value) * max_marker_size if max_abs_value > 0 else min_marker_size)); df_map['DisplayValue'] = df_map['Value']
           hover_template = ( "<b>%{customdata[0]}</b> (%{customdata[1]})<br>" f"<b>{perf_metric}:</b> %{{customdata[3]}}<br>" "Net Income: %{customdata[4]}<br>" "Assets: %{customdata[5]}" "<extra></extra>" )
           try:
                fig_map = px.scatter_geo( df_map, lat='lat', lon='lon', scope='world', size='Size', color='DisplayValue', hover_name='Segment',
                                       custom_data=['Segment', 'ISO', 'Metric', 'Formatted Value', 'Formatted NI', 'Formatted Assets'],
                                       title=f"Segment Performance by {perf_metric} (Color/Size indicate magnitude)",
                                       color_continuous_scale=px.colors.diverging.RdYlGn, color_continuous_midpoint=0, projection="natural earth", size_max=max_marker_size )
                fig_map.update_traces(hovertemplate=hover_template, marker=dict(line=dict(width=0.5, color='#ffffff' if st.session_state.theme_mode=='dark' else '#3f3f3f'), opacity=0.85))
                ocean_color = "#1f77b4"; land_color = "#4a4a4a" if st.session_state.theme_mode == "dark" else "#E5ECF6"; subunit_color = "#6a6a6a" if st.session_state.theme_mode == "dark" else "#B0B9C6";
                fig_map.update_layout( geo=dict(bgcolor='rgba(0,0,0,0)', landcolor=land_color, subunitcolor=subunit_color, showocean=True, oceancolor=ocean_color, lakecolor=ocean_color), height=550, margin={"r":0,"t":50,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=font_color), geo_center=dict(lon=-75, lat=25), geo_projection_scale=1.5 )
                st.plotly_chart(fig_map, use_container_width=True)
           except Exception as e: st.error(f"Could not generate map: {e}")
    elif segments:
         st.info(f"Segment map visualization is typically applicable for Insurance, Banking, or Technology sectors with defined geographic regions. A bar chart comparison is provided above.")

def display_forecasting_charts(historical_data: Optional[Dict[str, Dict[str, Any]]], data: Dict[str, Any]) -> None:
    st.markdown("Basic 3-year forecast using Prophet based on available historical data."); st.caption("Note: Uses default Prophet settings. Actual results depend on data quality and model tuning."); st.divider()
    if not historical_data: st.info("Historical data unavailable for forecasting."); return
    metadata = data.get("metadata", {}); currency = metadata.get("currency", DEFAULT_CURRENCY); scale = metadata.get("scale", 1); scale = 1 if not isinstance(scale, (int, float)) or scale == 0 else scale
    scale_desc = "M" if scale == 1_000_000 else "k" if scale == 1000 else ""; y_axis_title_base = f"Value ({currency}{' ' + scale_desc if scale_desc else ''})"
    valid_metrics = {k: {str(year): val for year, val in v.items() if isinstance(val, (int, float))} for k, v in historical_data.items() if isinstance(v, dict)}; valid_metrics = {k: v for k, v in valid_metrics.items() if len(v) >= 2}
    if not valid_metrics: st.warning("No metrics with sufficient historical data (>= 2 years) for forecasting."); return

    other_kpis = data.get("Other KPIs", {})
    for kpi_name, kpi_value in other_kpis.items():
       if isinstance(kpi_value, (int, float)) and kpi_name not in valid_metrics:
           hist_kpi = data.get("Historical Data", {}).get(kpi_name, {})
           if hist_kpi and isinstance(hist_kpi, dict) and len(hist_kpi) >= 1:
                combined_kpi = {str(year): val for year, val in hist_kpi.items() if isinstance(val, (int, float))}
                current_year = metadata.get("report_date", "N/A").split("-")[0]
                if current_year != "N/A" and current_year not in combined_kpi:
                    combined_kpi[current_year] = kpi_value
                if len(combined_kpi) >= 2:
                    valid_metrics[kpi_name] = combined_kpi

    if not valid_metrics:
        st.warning("No metrics with sufficient historical data (>= 2 years) for forecasting, including KPIs."); return

    metric_to_forecast = st.selectbox("Select metric to forecast", list(valid_metrics.keys()))
    if metric_to_forecast:
        hist_values = valid_metrics[metric_to_forecast]; forecast_df, model = create_prophet_forecast(hist_values, periods=3, freq='Y')
        if forecast_df is not None and model is not None:
            fig = go.Figure(); hist_df_plot = pd.DataFrame(list(hist_values.items()), columns=['ds', 'y']);
            if all(re.match(r"^\d{4}$", str(d)) for d in hist_df_plot['ds']):
                 hist_df_plot['ds'] = pd.to_datetime(hist_df_plot['ds'].astype(str) + '-12-31')
            else:
                 hist_df_plot['ds'] = pd.to_datetime(hist_df_plot['ds'])
            hist_df_plot = hist_df_plot.sort_values('ds'); last_hist_date = hist_df_plot['ds'].max()
            fig.add_trace(go.Scatter(x=hist_df_plot['ds'], y=hist_df_plot['y'], mode='lines+markers', name='Historical', line=dict(color='#0d6efd', width=2.5), marker=dict(size=6)))
            forecast_plot_df = forecast_df[forecast_df['ds'] > last_hist_date]
            fig.add_trace(go.Scatter(x=forecast_plot_df['ds'], y=forecast_plot_df['yhat'], mode='lines', name='Forecast', line=dict(color='#fd7e14', dash='dash', width=2.5)))
            x_fill = pd.concat([hist_df_plot['ds'].tail(1), forecast_plot_df['ds'], forecast_plot_df['ds'][::-1], hist_df_plot['ds'].tail(1)])
            y_fill = pd.concat([hist_df_plot['y'].tail(1), forecast_plot_df['yhat_upper'], forecast_plot_df['yhat_lower'][::-1], hist_df_plot['y'].tail(1)])
            fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill='toself', fillcolor='rgba(253, 126, 20, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False, name='Confidence Interval'))
            font_color = "#e0e0e0" if st.session_state.theme_mode == "dark" else "#212529"
            is_kpi_metric = metric_to_forecast in other_kpis
            current_y_axis_title = metric_to_forecast if is_kpi_metric else y_axis_title_base
            tick_format = ',.0f'
            if not is_kpi_metric:
                 tick_format = ',.0f'

            fig.update_layout(title=f"3-Year Forecast for {metric_to_forecast}", xaxis_title="Date", yaxis_title=current_y_axis_title, hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=font_color)), font=dict(color=font_color), yaxis_tickformat=tick_format)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("View Forecast Data"):
                 future_forecast = forecast_plot_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]; future_forecast['ds'] = future_forecast['ds'].dt.year
                 if is_kpi_metric:
                      for col in ['yhat', 'yhat_lower', 'yhat_upper']: future_forecast[col] = future_forecast[col].apply(lambda x: f"{x:,.0f}")
                 else:
                       for col in ['yhat', 'yhat_lower', 'yhat_upper']: future_forecast[col] = future_forecast[col].apply(lambda x: format_currency(x, currency, 1, 1))
                 st.dataframe(future_forecast.rename(columns={'ds':'Year','yhat':'Forecast','yhat_lower':'Lower Bound','yhat_upper':'Upper Bound'}).set_index('Year'))
        elif model is None: st.warning(f"Could not generate forecast for {metric_to_forecast} (Insufficient data).")
        else: st.error(f"An error occurred during forecasting for {metric_to_forecast}.")
    else: st.info("Please select metric for forecasting.")

# =============================================
# 🏗️ SAMPLE DATA FUNCTIONS
# =============================================
def get_enhanced_sample_data() -> Dict[str, Any]:
    ni_2024 = 97.5; core_earnings_2024 = 90.9; equity_2024_sh = 959.7; equity_2024_total = 1322.5
    assets_2024 = 22767.9; liab_2024 = 21445.5; licat_2024 = 1.39; core_eps_2024 = 0.649
    basic_eps_2024 = 0.696; diluted_eps_2024 = 0.679; core_roe_2024 = 0.096; reported_roe_2024 = 0.103
    book_value_ps_2024 = 7.08
    fin_leverage_2024 = 0.273; dividend_payout_2024 = 0.370;
    assets_2023 = 22384.9
    equity_2023_sh = 970.9
    avg_assets = (assets_2024 + assets_2023) / 2 if assets_2023 and assets_2024 else assets_2024
    roa_2024 = (ni_2024 / avg_assets) if avg_assets and ni_2024 is not None else None
    roe_to_use = reported_roe_2024 if reported_roe_2024 is not None else core_roe_2024
    roe_2023 = 0.55
    data = {
        "metadata": {"company_type": "insurance", "company_name": "Sagicor Financial Company Ltd. (Sample Data FY2024)", "report_date": "2024-12-31", "currency": "USD", "scale": 1_000_000,"extraction_method": "Manual Sample Data (Source: SFCL 2024 AR PDF)", "ifrs17_model_detected": "BBA/GMM Primarily (Inferred)", "warnings": ["Using sample data derived from SFCL 2024 Annual Report PDF content."],},
        "Income Statement": {"Net Income Attributable to Shareholders": ni_2024, "Net Income": ni_2024, "Core Earnings Attributable to Shareholders": core_earnings_2024, "Insurance Service Result": 166.0, "Net Investment Income": 1495.8, "Net insurance finance expenses": -1067.4, "Fees and other income": 153.0, "Gain arising on acquisitions and divestitures": 9.6, "Share of income associates/JV": 4.3, "Other operating expenses": -338.5, "Other interest and finance costs": -232.7, "Insurance Revenue": 1449.9, "Insurance Service Expenses": 1085.3},
        "Balance Sheet": {"Total Assets": assets_2024, "Total Equity": equity_2024_total, "Shareholders' Equity": equity_2024_sh, "Total Liabilities": liab_2024, "Notes and loans payable": 953.9},
        "Cash Flow": {"Operating Cash Flow": 214.2, "Investing Cash Flow": -14.0, "Financing Cash Flow": -64.3, "Free Cash Flow": 214.2 + (-14.0)},
        "IFRS17 Metrics": {"Insurance Service Result": 166.0, "Insurance Finance Income/Expenses": -1067.4, "CSM": 1219.7, "Net CSM to shareholders": 1076.1, "New Business CSM": 166.3},
        "IFRS17 Movements": {"CSM Movement": {"Net CSM, beginning of period": 1278.5, "New insurance business": 166.3,"Expected movements related to finance income or expenses": 66.3,"Insurance experience (losses) gains": -6.9, "CSM recognised for services provided": -154.3,"Change in assumptions and management actions": -58.1, "Impact of markets": 2.8,"Currency impact": -48.9, "Other impact": -26.0, "Net CSM End of Period": 1219.7}, "Income Breakdown": {"Insurance Service Result": 166.0, "Net Investment Income": 1495.8,"Net insurance finance expenses": -1067.4, "Other Income & Gains": 153.0 + 9.6 + 4.3,"Other Expenses & Costs": -338.5 - 232.7, "Income before Taxes": 190.1, "Income taxes": -61.8, "Net Income for the Year": 128.3, "Net Income Attributable to Shareholders": ni_2024 }, "Risk Adjustment Movement": {}, "Loss Component Movement": {}},
        "Ratios": { "Core EPS": core_eps_2024, "Basic EPS": basic_eps_2024, "Diluted EPS": diluted_eps_2024, "Core ROE": core_roe_2024, "Reported ROE": reported_roe_2024, "ROE": roe_to_use, "ROA": roa_2024, "Book Value per share": book_value_ps_2024, "Book value plus Net CSM per share": 15.02, "Financial leverage ratio": fin_leverage_2024, "Core dividend payout ratio": dividend_payout_2024, "Group LICAT Ratio": licat_2024, "LICAT Ratio": licat_2024, "Combined Ratio": None, },
        "Segments": { "Sagicor Canada": {"Core Earnings": 86.9, "Net Income": 96.2, "Total Assets": 7817.8}, "Sagicor Life USA": {"Core Earnings": 40.2, "Net Income": 51.6, "Total Assets": 5591.3}, "Sagicor Jamaica": {"Core Earnings": 31.1, "Net Income": 31.3, "Total Assets": 6390.2}, "Sagicor Life": {"Core Earnings": 26.3, "Net Income": 38.5, "Total Assets": 2652.2}, "Head Office and Other": {"Core Earnings": -93.6, "Net Income": -120.1, "Total Assets": 416.4} },
        "ESG": {"Female Executives (%)": 54.0, "Community Donations (M)": 1.8},
        "Market": {"Market Cap (M)": 583.1, "Dividends per share": 0.240},
        "Historical Data": {"Net Income": {"2022": -164.4, "2023": 532.1, "2024": 97.5},"Total Assets": {"2022": 10621.4, "2023": assets_2023, "2024": assets_2024},"Shareholders' Equity": {"2022": 429.7, "2023": equity_2023_sh, "2024": equity_2024_sh},"Basic EPS": {"2022": -1.150, "2023": 3.740, "2024": 0.696},"Core EPS": {"2023": 0.347, "2024": 0.649},"LICAT Ratio": {"2023": 1.36, "2024": 1.39}, "ROE": {"2023": roe_2023},
                          "Detailed Historical Data": {"Net Income 2024 Reported": 97.5, "Net Income 2023 Reported": 532.1, "Ivari Acquisition Gain 2023": 448.3, "Net Income 2023 Normalized": 83.8}}
    }
    return data

def get_apple_sample_data() -> Dict[str, Any]:
    net_sales_2024 = 391035; products_sales_2024 = 294866; services_sales_2024 = 96169; cost_of_sales_2024 = 210352; gross_margin_2024 = 180683
    r_and_d_2024 = 31370; sg_and_a_2024 = 26097; operating_income_2024 = 123216; other_income_expense_net_2024 = 269; income_before_tax_2024 = 123485
    tax_provision_2024 = 29749; net_income_2024 = 93736; eps_diluted_2024 = 6.08
    total_current_assets_2024 = 152987; total_non_current_assets_2024 = 211993; total_assets_2024 = 364980
    total_current_liabilities_2024 = 176392; total_non_current_liabilities_2024 = 131638; total_liabilities_2024 = 308030; total_equity_2024 = 56950
    cash_ops_2024 = 118254; cash_investing_2024 = 2935; cash_financing_2024 = -121983; ppe_payments_2024 = 9447
    americas_sales_2024 = 167045; europe_sales_2024 = 101328; greater_china_sales_2024 = 66952; japan_sales_2024 = 25052; rest_apac_sales_2024 = 30658
    net_income_2023 = 96995; eps_diluted_2023 = 6.13; total_assets_2023 = 352583; total_equity_2023 = 62146
    gross_margin_pct = gross_margin_2024 / net_sales_2024 if net_sales_2024 else None; operating_margin_pct = operating_income_2024 / net_sales_2024 if net_sales_2024 else None
    net_margin_pct = net_income_2024 / net_sales_2024 if net_sales_2024 else None; avg_equity = (total_equity_2024 + total_equity_2023) / 2 if total_equity_2023 else total_equity_2024
    roe = net_income_2024 / avg_equity if avg_equity else None; avg_assets = (total_assets_2024 + total_assets_2023) / 2 if total_assets_2023 else total_assets_2024
    roa = net_income_2024 / avg_assets if avg_assets else None
    apple_data = {
        "metadata": {"company_type": "technology", "company_name": "Apple Inc. (Sample Data FY2024)", "report_date": "2024-09-28", "currency": "USD", "scale": 1_000_000, "extraction_method": "Manual Sample (2024 10-K Text)", "ifrs17_model_detected": "N/A", "warnings": ["Using sample data derived from Apple 2024 10-K text provided."]},
        "Income Statement": {"Net Sales": net_sales_2024, "Product Sales": products_sales_2024, "Service Sales": services_sales_2024, "Cost of Sales": cost_of_sales_2024, "Gross Margin": gross_margin_2024, "Research and Development": r_and_d_2024, "Selling General and Administrative": sg_and_a_2024, "Operating Income": operating_income_2024, "Other Income/(Expense) Net": other_income_expense_net_2024, "Income Before Tax": income_before_tax_2024, "Provision for Income Taxes": tax_provision_2024, "Net Income": net_income_2024},
        "Balance Sheet": {"Total Current Assets": total_current_assets_2024, "Total Non-Current Assets": total_non_current_assets_2024, "Total Assets": total_assets_2024, "Total Current Liabilities": total_current_liabilities_2024, "Total Non-Current Liabilities": total_non_current_liabilities_2024, "Total Liabilities": total_liabilities_2024, "Total Equity": total_equity_2024, "Shareholders' Equity": total_equity_2024},
        "Cash Flow": {"Operating Cash Flow": cash_ops_2024, "Investing Cash Flow": cash_investing_2024, "Financing Cash Flow": cash_financing_2024, "Free Cash Flow": cash_ops_2024 - ppe_payments_2024 if cash_ops_2024 is not None else None},
        "IFRS17 Metrics": {}, "IFRS17 Movements": {},\
        "Ratios": {"Gross Margin %": gross_margin_pct, "Operating Margin %": operating_margin_pct, "Net Margin %": net_margin_pct, "Diluted EPS": eps_diluted_2024, "ROE": roe, "ROA": roa},
        "Segments": {"Americas": {"Net Sales": americas_sales_2024}, "Europe": {"Net Sales": europe_sales_2024}, "Greater China": {"Net Sales": greater_china_sales_2024}, "Japan": {"Net Sales": japan_sales_2024}, "Rest of Asia Pacific": {"Net Sales": rest_apac_sales_2024}},
        "ESG": {}, "Market": {},\
        "Historical Data": {"Net Income": {"2023": net_income_2023, "2024": net_income_2024}, "Total Assets": {"2023": total_assets_2023, "2024": total_assets_2024}, "Shareholders' Equity": {"2023": total_equity_2023, "2024": total_equity_2024}, "Diluted EPS": {"2023": eps_diluted_2023, "2024": eps_diluted_2024}, "ROE": {"2023": None}},\
        "Notes Data": {} }
    return apple_data

def get_banking_sample_data() -> Dict[str, Any]:
    report_year = 2024; report_currency = "CAD"; report_scale = 1_000_000
    net_interest_income = 13695; non_interest_income = 11911; total_revenue = 25606
    provision_for_credit_losses = 2001; non_interest_expenses = 14439
    income_before_tax = 9166; tax_provision = 2012; net_income = 7154; net_income_common = 6852
    eps_diluted = 7.28
    total_assets = 1041985; allowance_for_credit_losses = 3917; gross_loans = 562203
    total_loans_net = gross_loans - allowance_for_credit_losses
    total_deposits = 764857; total_liabilities = 982978; total_equity = 59007; common_equity = 58735
    cash_ops = 11088; cash_investing = -20751; cash_financing = -2610
    net_interest_margin_pct = 0.0147; efficiency_ratio_pct = 0.564; cet1_ratio_pct = 0.133
    roe = 0.134; roa = 0.0071
    can_pers_bus_ni = 2700; can_comm_wealth_ni = 1900; us_comm_wealth_ni = 500; cap_markets_dfs_ni = 2000
    segments = {"Canadian Personal and Business Banking": {"Net Income": can_pers_bus_ni}, "Canadian Commercial Banking and Wealth Management": {"Net Income": can_comm_wealth_ni}, "U.S. Commercial Banking and Wealth Management": {"Net Income": us_comm_wealth_ni}, "Capital Markets and Direct Financial Services": {"Net Income": cap_markets_dfs_ni}}
    net_income_2023 = 5039; total_assets_2023 = 975690; common_equity_2023 = 52931
    eps_diluted_2023 = 5.17; roe_2023 = 0.103
    banking_data = {
        "metadata": {"company_type": "banking", "company_name": f"CIBC (Sample Data FY{report_year})", "report_date": f"{report_year}-10-31", "currency": report_currency, "scale": report_scale, "extraction_method": "Manual Sample Data (CIBC 2024 AR)", "ifrs17_model_detected": "N/A", "warnings": ["Using sample data manually extracted from CIBC 2024 Annual Report PDF."]},
        "Income Statement": {"Net Interest Income": net_interest_income, "Non-Interest Income": non_interest_income, "Total Revenue": total_revenue, "Provision for Credit Losses": provision_for_credit_losses, "Non-Interest Expenses": non_interest_expenses, "Income Before Tax": income_before_tax, "Provision for Income Taxes": tax_provision, "Net Income": net_income, "Net Income Attributable to Common Shareholders": net_income_common},
        "Balance Sheet": {"Total Assets": total_assets, "Total Loans Net": total_loans_net, "Allowance for Credit Losses": allowance_for_credit_losses, "Gross Loans": gross_loans, "Total Deposits": total_deposits, "Total Liabilities": total_liabilities, "Total Equity": total_equity, "Shareholders' Equity": common_equity},
        "Cash Flow": {"Operating Cash Flow": cash_ops, "Investing Cash Flow": cash_investing, "Financing Cash Flow": cash_financing, "Free Cash Flow": None},
        "IFRS17 Metrics": {}, "IFRS17 Movements": {},\
        "Ratios": {"Net Interest Margin": net_interest_margin_pct, "Efficiency Ratio": efficiency_ratio_pct, "CET1 Ratio": cet1_ratio_pct, "ROE": roe, "ROA": roa, "Diluted EPS": eps_diluted, "Reported ROE": roe},
        "Segments": segments, "ESG": {}, "Market": {},\
        "Historical Data": {"Net Income": {str(report_year-1): net_income_2023, str(report_year): net_income_common}, "Total Assets": {str(report_year-1): total_assets_2023, str(report_year): total_assets}, "Shareholders' Equity": {str(report_year-1): common_equity_2023, str(report_year): common_equity}, "Diluted EPS": {str(report_year-1): eps_diluted_2023, str(report_year): eps_diluted}, "ROE": {str(report_year-1): roe_2023}},\
        "Notes Data": {} }
    return banking_data

def get_gaia_sample_data() -> Dict[str, Any]:
    report_year = 2024
    report_currency = "BBD"
    report_scale = 1
    rev_contracts_2024 = 60755209
    rev_lease_2024 = 17245346
    total_rev_2024 = 80538508
    total_expenses_2024 = 80654464
    net_income_2024 = 245435
    total_assets_2024 = 276949320
    total_liabilities_2024 = 163152956
    total_equity_2024 = 113796364
    shareholders_equity_2024 = 113796364
    current_assets_2024 = 47229453
    current_liabilities_2024 = 88547825
    total_long_term_loans_2024 = 139952368
    net_income_2023 = -5185431
    total_assets_2023 = 269888975
    total_liabilities_2023 = 156338046
    total_equity_2023 = 113550929
    shareholders_equity_2023 = 113550929
    current_assets_2023 = 34027594
    current_liabilities_2023 = 73901465
    net_income_2022 = -27644595
    total_assets_2022 = 282286924
    total_equity_2022 = 118736360
    passengers_2023 = 1788678
    movements_2023 = 26574
    cargo_2023 = 10248222
    mail_2023 = 123123
    passengers_2022 = 988370
    movements_2022 = 19726
    dscr_2023 = 1.74
    dscr_2022 = -1.83
    avg_equity_2024 = (total_equity_2024 + total_equity_2023)/2 if total_equity_2023 else total_equity_2024
    avg_assets_2024 = (total_assets_2024 + total_assets_2023)/2 if total_assets_2023 else total_assets_2024
    roe_2024 = (net_income_2024 / avg_equity_2024) if avg_equity_2024 and net_income_2024 is not None else None
    roa_2024 = (net_income_2024 / avg_assets_2024) if avg_assets_2024 is not None and avg_assets_2024 != 0 and net_income_2024 is not None else None
    debt_equity_ratio_2024 = (total_liabilities_2024 / total_equity_2024) if total_equity_2024 else None
    current_ratio_2024 = (current_assets_2024 / current_liabilities_2024) if current_liabilities_2024 else None
    debt_service_coverage_2024 = None
    roe_2023_calc = (net_income_2023 / ((total_equity_2023 + total_equity_2022)/2)) if total_equity_2022 else (net_income_2023 / total_equity_2023)
    roa_2023_calc = (net_income_2023 / ((total_assets_2023 + total_assets_2022)/2)) if total_assets_2022 else (net_income_2023 / total_assets_2023)
    debt_equity_ratio_2023_calc = (total_liabilities_2023 / total_equity_2023) if total_equity_2023 else None
    current_ratio_2023_calc = (current_assets_2023 / current_liabilities_2023) if current_liabilities_2023 else None

    gaia_sample_data_fy2024 = {
        "metadata": {
            "company_type": "airport",
            "company_name": f"Grantley Adams Intl Airport (GAIA) (Sample Data FY{report_year})",
            "report_date": f"{report_year}-03-31",
            "currency": report_currency,
            "scale": report_scale,
            "extraction_method": f"Manual Sample Data (GAIA {report_year} AR PDF)",
            "ifrs17_model_detected": "N/A",
            "warnings": [f"Using sample data derived from GAIA {report_year} Annual Report PDF content.", f"Financial data in {report_currency}.", "Detailed segment cost breakdown not available in extracted data. Using revenue breakdown.", "Operational KPIs (Passengers, Movements, Cargo, Mail) for 2024 not found in the extracted text; using 2023 values if available."]
        },
        "Income Statement": {
            "Total Revenue": total_rev_2024,
            "Revenue from contracts with customers": rev_contracts_2024,
            "Revenue for lease contracts": rev_lease_2024,
            "Total Expenses": total_expenses_2024,
            "Operating Expenses": 52052616,
            "Net Income": net_income_2024,
        },
        "Balance Sheet": {
            "Total Assets": total_assets_2024,
            "Total Liabilities": total_liabilities_2024,
            "Total Equity": total_equity_2024,
            "Shareholders' Equity": shareholders_equity_2024,
            "Total Long-term loans": total_long_term_loans_2024,
            "Current Assets": current_assets_2024,
            "Current Liabilities": current_liabilities_2024,
        },
        "Cash Flow": {
            "Operating Cash Flow": 20874487,
            "Investing Cash Flow": -9683424,
            "Financing Cash Flow": -3629849,
            "Free Cash Flow": None
        },
        "IFRS17 Metrics": {},
        "IFRS17 Movements": {},
        "Ratios": {
            "ROE": roe_2024,
            "ROA": roa_2024,
            "Debt to Equity Ratio": debt_equity_ratio_2024,
            "Current Ratio": current_ratio_2024,
            "Debt Service Coverage Ratio": debt_service_coverage_2024,
        },
        "Other KPIs": {
            "Total Passengers": passengers_2023,
            "Aircraft Movements": movements_2023,
            "Cargo Handled (kg)": cargo_2023,
            "Mail Handled (kg)": mail_2023,
        },
        "Segments": {},
        "ESG": {},
        "Market": {},
        "Historical Data": {
            "Net Income": {"2023": net_income_2023, str(report_year): net_income_2024},
            "Total Assets": {"2023": total_assets_2023, str(report_year): total_assets_2024},
            "Shareholders' Equity": {"2023": total_equity_2023, str(report_year): shareholders_equity_2024},
            "Total Passengers": {"2023": passengers_2023},
            "Aircraft Movements": {"2023": movements_2023},
            "ROE": {"2023": roe_2023_calc, str(report_year): roe_2024 if roe_2024 is not None else None},
            "ROA": {"2023": roa_2023_calc, str(report_year): roa_2024 if roa_2024 is not None else None},
            "Debt to Equity Ratio": {"2023": debt_equity_ratio_2023_calc, str(report_year): debt_equity_ratio_2024 if debt_equity_ratio_2024 is not None else None},
            "Current Ratio": {"2023": current_ratio_2023_calc, str(report_year): current_ratio_2024 if current_ratio_2024 is not None else None},
            "Debt Service Coverage Ratio": {"2023": dscr_2023},
            "Income Statement": {
                 "2023": {
                      "Revenue from contracts with customers": 50487691,
                      "Revenue for lease contracts": 15816569,
                      "Total Revenue": 71000542,
                 },
                 str(report_year): {
                   "Revenue from contracts with customers": rev_contracts_2024,
                   "Revenue for lease contracts": rev_lease_2024,
                   "Total Revenue": total_rev_2024,
                 }
            }
        },
        "Notes Data": {}
    }
    return gaia_sample_data_fy2024

# =============================================
# 🏗️ APPLICATION STRUCTURE (UI Layer)
# =============================================
def main():
    if 'theme_mode' not in st.session_state: st.session_state.theme_mode = "light"
    if 'current_company_type' not in st.session_state: st.session_state.current_company_type = "insurance"
    if 'financial_data' not in st.session_state:
        if st.session_state.current_company_type == "insurance": st.session_state.financial_data = get_enhanced_sample_data()
        elif st.session_state.current_company_type == "banking": st.session_state.financial_data = get_banking_sample_data()
        elif st.session_state.current_company_type == "technology": st.session_state.financial_data = get_apple_sample_data()
        elif st.session_state.current_company_type == "airport": st.session_state.financial_data = get_gaia_sample_data()
        else: st.session_state.financial_data = get_enhanced_sample_data()
        st.session_state.show_sample_warning = True

    with st.sidebar:
        st.header("🤖 FP&A AI Agent Controls")
        theme_mode_select = st.radio("Select Theme", ["Light", "Dark"], index=0 if st.session_state.theme_mode == "light" else 1, key="theme_select")
        if theme_mode_select.lower() != st.session_state.theme_mode: st.session_state.theme_mode = theme_mode_select.lower(); st.rerun()
        st.divider(); st.header("📂 Data Input & Selection")
        company_type_options = ["Insurance", "Banking", "Technology", "Airport", "Other"]
        try: current_index = [opt.lower() for opt in company_type_options].index(st.session_state.current_company_type)
        except ValueError: current_index = 0
        selected_type = st.selectbox("Company Industry Type", company_type_options, index=current_index, key="company_type_selector")
        selected_type_lower = selected_type.lower()

        if selected_type_lower != st.session_state.current_company_type:
            st.session_state.current_company_type = selected_type_lower
            if selected_type_lower == "insurance": st.session_state.financial_data = get_enhanced_sample_data()
            elif selected_type_lower == "banking": st.session_state.financial_data = get_banking_sample_data()
            elif selected_type_lower == "technology": st.session_state.financial_data = get_apple_sample_data()
            elif selected_type_lower == "airport": st.session_state.financial_data = get_gaia_sample_data()
            else: st.session_state.financial_data = get_enhanced_sample_data()
            st.session_state.show_sample_warning = True; st.rerun()

        uploaded_file = st.file_uploader("Upload Financial Report (PDF)", type="pdf")
        st.caption("Note: Current PDF extraction is basic. Overrides sample if processed.")
        if uploaded_file:
            if st.button("Process Uploaded Report", key="process_pdf"):
                pdf_content = uploaded_file.getvalue()
                with st.spinner("🧠 AI analyzing report..."):
                     extracted_data = extract_financial_data(pdf_content, st.session_state.current_company_type)
                     if any(extracted_data.get(sec) for sec in ["Income Statement", "Balance Sheet", "Ratios", "Other KPIs"]):
                          extracted_data["metadata"]["company_type"] = st.session_state.current_company_type
                          st.session_state.financial_data = extracted_data; st.session_state.show_sample_warning = False
                          st.success("✅ Report analysis complete!"); st.rerun()
                     else: st.error("⚠️ Extraction failed. Displaying sample data."); st.session_state.show_sample_warning = True
        st.divider()

        if st.button(f"Reload Sample {st.session_state.current_company_type.capitalize()} Data", key="reload_sample"):
            if st.session_state.current_company_type == "insurance": st.session_state.financial_data = get_enhanced_sample_data()
            elif st.session_state.current_company_type == "banking": st.session_state.financial_data = get_banking_sample_data()
            elif st.session_state.current_company_type == "technology": st.session_state.financial_data = get_apple_sample_data()
            elif st.session_state.current_company_type == "airport": st.session_state.financial_data = get_gaia_sample_data()
            else: st.session_state.financial_data = get_enhanced_sample_data()
            st.session_state.show_sample_warning = True
            st.success(f"✨ Sample {st.session_state.current_company_type.capitalize()} data loaded!"); st.rerun()

        st.divider(); st.header("⚙️ Analysis Modules")
        _current_data_type = st.session_state.financial_data.get("metadata",{}).get("company_type", "general")
        show_kpis = st.checkbox("Show Overview", True)
        show_ifrs17_details = st.checkbox("Show IFRS 17 Details", True, help="Display IFRS 17 specifics.") if _current_data_type == "insurance" else False
        show_segments = st.checkbox("Show Segment Performance", True, help="Shows bar chart comparison. Map only for specific types.")
        show_ai_insights = st.checkbox("Show AI Insights", True); show_forecasting = st.checkbox("Show Forecasting", True)
        st.divider(); st.header("🚀 Future Capabilities"); st.caption("Planned enhancements:")
        st.text_input("Ask the AI (NLQ)", placeholder="e.g., What drove CSM change?", disabled=True)
        st.selectbox("Scenario Analysis", ["Base Case", "Interest Rate +1%"], disabled=True)
        st.button("Run Variance Analysis", disabled=True); st.button("Generate Report", disabled=True)
        st.caption("Note: Advanced features require backend development.")

    apply_theme(st.session_state.theme_mode)
    data = st.session_state.financial_data; metadata = data.get("metadata", {}); company_name = metadata.get("company_name", "Company"); report_date = metadata.get("report_date", "N/A")
    current_company_type = data.get("metadata",{}).get("company_type", "general")
    currency = metadata.get("currency", DEFAULT_CURRENCY); scale = metadata.get("scale", DEFAULT_SCALE)
    scale_desc = "M" if scale == 1_000_000 else "k" if scale == 1000 else "" if scale == 1 else f"x{scale}"
    font_color = "#e0e0e0" if st.session_state.theme_mode == "dark" else "#212529"

    st.title(f"📊 FP&A AI Agent: {company_name}")
    st.markdown(f"*Analysis for: {report_date} | Type: {current_company_type.capitalize()} | Currency: {currency} {'('+scale_desc+')' if scale_desc else '(as reported)'}*")
    if st.session_state.get('show_sample_warning', False): st.warning(f"ℹ️ Displaying sample data for {current_company_type.capitalize()}. Upload PDF for custom analysis via sidebar.", icon="💡")
    warnings_text = "";
    if warnings_list := metadata.get("warnings"): warnings_text = "; ".join(w for w in warnings_list if not ("Could not determine currency/scale" in w and scale != DEFAULT_SCALE))
    if warnings_text: st.warning(f"Data Notes: {warnings_text}", icon="⚠️")
    st.divider()

    tab_titles = []; tab_content_flags = []
    if show_kpis: tab_titles.append("📊 Overview & KPIs"); tab_content_flags.append("kpis")
    if current_company_type == "insurance" and show_ifrs17_details: tab_titles.append("🌊 IFRS 17 Details"); tab_content_flags.append("ifrs17")
    if show_segments: tab_titles.append("🗺️ Segments"); tab_content_flags.append("segments")
    if show_forecasting: tab_titles.append("🔮 Forecasting"); tab_content_flags.append("forecasting")
    if show_ai_insights: tab_titles.append("💬 AI Insights"); tab_content_flags.append("insights")

    if tab_titles:
        tabs = st.tabs(tab_titles)
        for i, flag in enumerate(tab_content_flags):
            with tabs[i]:
                if flag == "kpis":
                    display_key_metrics_and_ratios(data)
                    hist_data = data.get("Historical Data", {})
                    hist_metric_key = None
                    if current_company_type == 'airport' and 'Total Passengers' in hist_data:
                        hist_metric_key = 'Total Passengers'
                    elif 'Net Income' in hist_data:
                        hist_metric_key = 'Net Income'

                    if hist_metric_key and hist_data.get(hist_metric_key):
                        hist_metric_data = hist_data[hist_metric_key]
                        hist_metric_data = {str(k): v for k, v in hist_metric_data.items() if isinstance(v, (int, float))}
                        if hist_metric_data:
                             try:
                                 hist_metric_items = sorted([(str(k), v) for k, v in hist_metric_data.items()])
                                 hist_metric_df = pd.DataFrame(hist_metric_items, columns=['Year', hist_metric_key])
                                 try: hist_metric_df['Year'] = pd.to_numeric(hist_metric_df['Year'])
                                 except: pass

                                 if not hist_metric_df.empty:
                                      st.divider(); st.subheader(f"Historical {hist_metric_key} Trend")
                                      try:
                                           line_color = "#17A2B8"
                                           y_axis_label = f"{hist_metric_key}"
                                           hover_template_val = '%{y:,.0f}'
                                           if hist_metric_key == 'Net Income':
                                                y_axis_label = f"{hist_metric_key} ({currency}{scale_desc})"
                                                hover_template_val = f'{currency}%{{y:,.1f}}{scale_desc}'

                                           fig_hist = px.line(hist_metric_df, x='Year', y=hist_metric_key, title=f"Historical {hist_metric_key} Trend", markers=True)
                                           fig_hist.update_layout( paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=font_color), xaxis_title="Year", yaxis_title=y_axis_label,
                                                                 yaxis=dict(tickfont=dict(size=12), title_font=dict(size=14), showgrid=True, gridwidth=0.5, gridcolor='#888888' if st.session_state.theme_mode == 'dark' else '#cccccc'),
                                                                 xaxis=dict(tickfont=dict(size=12), title_font=dict(size=14), showgrid=True, gridwidth=0.5, gridcolor='#888888' if st.session_state.theme_mode == 'dark' else '#cccccc', type='category' if isinstance(hist_metric_df['Year'].iloc[0], str) else '-'),
                                                                 legend=dict(font=dict(size=12)) )
                                           fig_hist.update_traces( line=dict(color=line_color, width=2.5), hovertemplate=f"Year: %{{x}}<br>{hist_metric_key}: {hover_template_val}<extra></extra>" )
                                           st.plotly_chart(fig_hist, use_container_width=True)
                                      except Exception as e_plot: st.warning(f"Could not display historical {hist_metric_key} trend chart: {e_plot}")
                             except Exception as e_df: st.warning(f"Could not process historical {hist_metric_key} data: {e_df}")
                elif flag == "ifrs17": display_ifrs17_waterfalls(data)
                elif flag == "segments": display_segment_map(data)
                elif flag == "forecasting": display_forecasting_charts(data.get("Historical Data"), data)
                elif flag == "insights":
                    st.markdown("Automated commentary based on the analyzed data.")
                    insights = generate_ai_insights(data, data.get("Historical Data"))
                    st.caption("Note: Current insights are template-based. Future versions planned with advanced AI analysis.")
                    for insight in insights: st.markdown(insight, unsafe_allow_html=True)
    else:
        st.info("Select analysis modules to display from the sidebar.")

    # --- Export Section (with fix) ---
    st.divider()
    with st.expander("📥 Export Processed Data"):
        try:
            # Define flatten_dict function with correct indentation
            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                if not isinstance(d, dict):
                    # Base case: if input is not a dict, return it with its key
                    return {parent_key if parent_key else 'value': d}
                # Recursive step: if input is a dict
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        # If value is a dict, recursively call flatten_dict
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        # If value is not a dict, add (key, value) to items
                        items.append((new_key, v))
                # Return the dictionary created from all collected items *after* the loop
                return dict(items)

            # Proceed with flattening and export
            data_to_flatten = copy.deepcopy(data)
            export_ready_data = flatten_dict(data_to_flatten)
            export_df = pd.DataFrame(list(export_ready_data.items()), columns=['Metric', 'Value'])
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode('utf-8')

            # Ensure company_name and report_date are available before using them
            file_name_company = company_name.replace(' ','_').lower() if company_name else "export"
            file_name_date = report_date if report_date and report_date != "N/A" else "data"

            st.download_button(
                label="Download Data as CSV",
                data=csv_bytes,
                file_name=f"{file_name_company}_{file_name_date}_fpa_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            # Print the actual error to the console/log for debugging
            print(f"Error during export preparation: {e}")
            # Display a user-friendly message
            st.error(f"Failed to prepare data for export. Error: {e}")


if __name__ == "__main__":
    main()