import os, re, requests, base64
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from bs4 import BeautifulSoup
from datetime import datetime
import pytz

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── LOAD .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── API KEYS — loaded from environment only, no hardcoded keys ────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
FINNHUB_KEY  = os.environ.get("FINNHUB_API_KEY", "")

st.set_page_config(
    page_title="ChartLine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_DEFAULTS = {
    "chat_history": list,
    "rsi_scan_results": list,
    "rsi_selected_stock": lambda: None,
    "rsi_show_invest_panel": lambda: False,
    "rsi_investment_submitted": lambda: False,
    "rsi_investment_amount": lambda: 0.0,
    "active_stock": lambda: "",
    "stock_data_loaded": lambda: False,
    "ai_results": lambda: None,
    "watchlist": list,
    "rsi_last_scan_done": lambda: False,
    "active_page": lambda: "Dashboard",
    "theme": lambda: "dark",
}
for k, fac in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = fac()

def get_logo_b64():
    logo_path = "logo2.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --ink:       #000000;
  --paper:     #ffffff;
  --glass-bg:  rgba(255,255,255,0.04);
  --glass-border: rgba(255,255,255,0.09);
  --green:     #22c55e;
  --green-dim: rgba(34,197,94,0.15);
  --red:       #ef4444;
  --red-dim:   rgba(239,68,68,0.15);
  --gold:      #d4af37;
  --font-display: 'Cormorant Garamond', serif;
  --font-mono:    'DM Mono', monospace;
  --font-hero:    'Bebas Neue', cursive;
  --font-num:     'Inter', sans-serif;
  --r: 12px;
  --r-sm: 8px;
  --ease: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: #000 !important;
  color: rgba(255,255,255,0.88) !important;
  font-family: var(--font-mono) !important;
  font-size: 15px !important;
  letter-spacing: 0.01em;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  opacity: 0.5;
}

#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stSidebar"] {
  background: #080808 !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
  width: 240px !important;
}
[data-testid="stSidebarContent"] { padding: 0 !important; }

.main .block-container {
  padding: 0 2rem 4rem !important;
  max-width: 100% !important;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

h1, h2, h3 {
  font-family: var(--font-display) !important;
  color: #fff !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
}

[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: var(--r) !important;
  padding: 1.1rem 1.3rem !important;
  transition: all 0.28s var(--ease) !important;
  position: relative !important; overflow: hidden !important;
}
[data-testid="stMetric"]::before {
  content: ''; position: absolute; top: 0; left: 0;
  width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
}
[data-testid="stMetric"]:hover {
  background: rgba(255,255,255,0.06) !important;
  border-color: rgba(255,255,255,0.15) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: rgba(255,255,255,0.4) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-num) !important;
  font-size: 1.45rem !important; font-weight: 700 !important;
  color: #fff !important;
  letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--font-num) !important; font-size: 0.82rem !important;
  font-weight: 600 !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button {
  background: rgba(255,255,255,0.05) !important;
  color: rgba(255,255,255,0.75) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 10px !important;
  font-family: var(--font-mono) !important;
  font-size: 1.08rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: none !important;
  padding: 0.85rem 1.1rem !important;
  width: 100% !important;
  text-align: left !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
  transform: none !important;
  margin: 4px 0 !important;
  display: flex !important;
  align-items: center !important;
  cursor: pointer !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
  background: rgba(255,255,255,0.12) !important;
  color: #ffffff !important;
  border-color: rgba(255,255,255,0.28) !important;
  box-shadow: 0 4px 18px rgba(0,0,0,0.5) !important;
  transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:active {
  transform: translateY(0px) !important;
  background: rgba(255,255,255,0.08) !important;
}

.main [data-testid="stButton"] > button,
[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] [data-testid="stButton"] > button {
  background: #fff !important; color: #000 !important;
  border: none !important; border-radius: var(--r-sm) !important;
  font-family: var(--font-mono) !important; font-size: 0.82rem !important;
  font-weight: 500 !important; letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: 0.6rem 1.4rem !important;
  transition: all 0.22s var(--ease-spring) !important;
  position: relative !important; overflow: hidden !important;
}
.main [data-testid="stButton"] > button:hover {
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 6px 28px rgba(255,255,255,0.2) !important;
}
.main [data-testid="stButton"] > button:active {
  transform: translateY(0) scale(0.99) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: var(--r-sm) !important;
  color: #fff !important; font-family: var(--font-mono) !important;
  font-size: 0.9rem !important; letter-spacing: 0.03em;
  transition: all 0.22s var(--ease) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
  border-color: rgba(255,255,255,0.35) !important;
  background: rgba(255,255,255,0.06) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.05) !important;
}
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label {
  font-family: var(--font-mono) !important; font-size: 0.72rem !important;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: rgba(255,255,255,0.4) !important;
}

[data-testid="stTabs"] [role="tablist"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: var(--r-sm) !important; padding: 3px !important; gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
  font-family: var(--font-mono) !important; font-size: 0.76rem !important;
  font-weight: 500 !important; letter-spacing: 0.06em; text-transform: uppercase;
  color: rgba(255,255,255,0.4) !important; border-radius: 6px !important;
  padding: 0.35rem 0.75rem !important; transition: all 0.2s !important; border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: #fff !important; color: #000 !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.4) !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
  color: rgba(255,255,255,0.8) !important; background: rgba(255,255,255,0.05) !important;
}

[data-testid="stExpander"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: var(--r) !important; transition: all 0.22s var(--ease) !important;
}
[data-testid="stExpander"]:hover { border-color: rgba(255,255,255,0.14) !important; }
[data-testid="stExpander"] summary {
  font-family: var(--font-mono) !important; font-size: 0.86rem !important;
  color: rgba(255,255,255,0.75) !important; padding: 0.7rem 1rem !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: var(--r) !important; overflow: hidden;
}
[data-testid="stDataFrame"] th {
  background: rgba(255,255,255,0.05) !important;
  color: rgba(255,255,255,0.5) !important;
  font-family: var(--font-mono) !important; font-size: 0.72rem !important;
  text-transform: uppercase; letter-spacing: 0.1em;
}
[data-testid="stDataFrame"] td {
  color: rgba(255,255,255,0.8) !important;
  font-family: var(--font-num) !important; font-size: 0.86rem !important;
  font-weight: 500 !important;
  border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}
[data-testid="stDataFrame"] tr:hover td { background: rgba(255,255,255,0.04) !important; }

[data-testid="stInfo"], [data-testid="stSuccess"],
[data-testid="stWarning"], [data-testid="stError"] {
  border-radius: var(--r-sm) !important;
  font-family: var(--font-mono) !important; font-size: 0.84rem !important;
}
[data-testid="stInfo"]    { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; }
[data-testid="stSuccess"] { background: rgba(34,197,94,0.08) !important; border: 1px solid rgba(34,197,94,0.2) !important; }
[data-testid="stWarning"] { background: rgba(212,175,55,0.08) !important; border: 1px solid rgba(212,175,55,0.2) !important; }
[data-testid="stError"]   { background: rgba(239,68,68,0.08) !important; border: 1px solid rgba(239,68,68,0.2) !important; }

[data-testid="stProgress"] > div { background: rgba(255,255,255,0.08) !important; border-radius: 4px; }
[data-testid="stProgress"] > div > div { background: #fff !important; border-radius: 4px; }

[data-testid="stChatMessage"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: var(--r) !important; font-family: var(--font-mono) !important;
  font-size: 0.9rem !important; transition: all 0.2s !important;
}
[data-testid="stChatInput"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.1) !important; border-radius: var(--r) !important;
}
[data-testid="stChatInput"] input {
  background: transparent !important; color: #fff !important;
  font-family: var(--font-mono) !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: rgba(255,255,255,0.25) !important;
  box-shadow: 0 0 0 3px rgba(255,255,255,0.04) !important;
}

[data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px dashed rgba(255,255,255,0.15) !important;
  border-radius: var(--r) !important; transition: all 0.22s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(255,255,255,0.3) !important;
  background: rgba(255,255,255,0.04) !important;
}

[data-testid="stSelectbox"] > div > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: var(--r-sm) !important; color: #fff !important;
  font-family: var(--font-mono) !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); }

hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

@keyframes fadeUp     { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn     { from { opacity:0; } to { opacity:1; } }
@keyframes tickerScroll {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
@keyframes countUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.lux-card {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--r);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  padding: 1.5rem;
  transition: all 0.3s var(--ease);
  position: relative; overflow: hidden;
  animation: fadeUp 0.5s var(--ease) both;
}
.lux-card::before {
  content: ''; position: absolute; top: 0; left: -100%;
  width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
  transition: left 0.5s var(--ease);
}
.lux-card:hover::before { left: 100%; }
.lux-card:hover {
  background: rgba(255,255,255,0.04);
  border-color: rgba(255,255,255,0.14);
  transform: translateY(-2px);
  box-shadow: 0 12px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06);
}

.sec-label {
  font-family: var(--font-mono);
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.18em;
  color: rgba(255,255,255,0.3);
  margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.6rem;
}
.sec-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, rgba(255,255,255,0.12), transparent);
}

.lux-title {
  font-family: var(--font-hero) !important;
  font-size: clamp(2.4rem, 4.5vw, 3.8rem) !important;
  letter-spacing: 0.06em !important;
  background: linear-gradient(135deg, #39ff14 0%, #00ff7f 30%, #32cd32 55%, #00cc66 80%, #66ff99 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  color: transparent !important;
  animation: fadeUp 0.7s var(--ease) both;
  line-height: 1.05 !important;
}
.lux-subtitle {
  font-family: var(--font-mono); font-size: 0.82rem;
  color: rgba(255,255,255,0.38); letter-spacing: 0.06em;
  animation: fadeUp 0.7s 0.1s var(--ease) both;
}

.sym-chip {
  display: inline-flex; align-items: center;
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 4px; padding: 0.12rem 0.5rem;
  font-family: var(--font-num); font-size: 0.78rem;
  font-weight: 700; color: rgba(255,255,255,0.7);
  letter-spacing: 0.04em;
}

.skeleton {
  background: linear-gradient(90deg, rgba(255,255,255,0.04) 25%, rgba(255,255,255,0.09) 50%, rgba(255,255,255,0.04) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 6px;
}

.stat-row {
  display: flex; align-items: baseline; gap: 0.5rem;
  animation: countUp 0.4s var(--ease) both;
}
.stat-val { font-family: var(--font-num); font-size: 1.4rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
.stat-lbl { font-family: var(--font-mono); font-size: 0.72rem; color: rgba(255,255,255,0.35); letter-spacing: 0.1em; text-transform: uppercase; }

.js-plotly-plot .plotly { border-radius: var(--r) !important; }

.nav-active > button {
  background: rgba(255,255,255,0.08) !important;
  color: #fff !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.4) !important;
}

.mm-stock-row {
  cursor: pointer;
  transition: background 0.15s;
}
.mm-stock-row:hover {
  background: rgba(255,255,255,0.06) !important;
}

.news-card {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 0.75rem;
  transition: background 0.22s ease, border-color 0.22s ease, transform 0.22s ease;
  position: relative;
  overflow: hidden;
}
.news-card::after {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}
.news-card:hover {
  background: rgba(255,255,255,0.045);
  border-color: rgba(255,255,255,0.16);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.45);
}
.news-card-headline {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.05rem;
  font-weight: 600;
  color: rgba(255,255,255,0.92);
  line-height: 1.45;
  margin-bottom: 0.45rem;
  letter-spacing: -0.01em;
}
.news-card-meta {
  font-family: 'DM Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.28);
  margin-bottom: 0.55rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.news-card-meta::before {
  content: '';
  display: inline-block;
  width: 5px; height: 5px;
  background: rgba(255,255,255,0.2);
  border-radius: 50%;
}
.news-card-summary {
  font-family: 'DM Mono', monospace;
  font-size: 0.8rem;
  color: rgba(255,255,255,0.38);
  line-height: 1.75;
  margin-bottom: 0.65rem;
}
.news-card-link {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  color: rgba(255,255,255,0.35);
  text-decoration: none;
  letter-spacing: 0.07em;
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  transition: color 0.18s ease;
}
.news-card-link:hover { color: rgba(255,255,255,0.75); }

.wl-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 10px;
  padding: 0.7rem 0.95rem;
  margin-bottom: 0.5rem;
  transition: background 0.2s ease, border-color 0.2s ease;
  gap: 0.5rem;
}
.wl-row:hover {
  background: rgba(255,255,255,0.045);
  border-color: rgba(255,255,255,0.13);
}
.wl-ticker {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: 0.9rem;
  color: #fff;
  letter-spacing: 0.03em;
  min-width: 90px;
}
.wl-price {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  font-size: 0.9rem;
  color: rgba(255,255,255,0.82);
  letter-spacing: -0.01em;
  flex: 1;
  text-align: right;
  padding-right: 1rem;
}
.wl-chg {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  font-size: 0.82rem;
  min-width: 80px;
  text-align: right;
  padding-right: 1rem;
}
.wl-meta {
  font-family: 'DM Mono', monospace;
  font-size: 0.7rem;
  color: rgba(255,255,255,0.3);
  min-width: 60px;
  text-align: right;
  padding-right: 1rem;
}

.greeting-block { animation: fadeUp 0.6s ease both; }
.greeting-time {
  font-family: 'DM Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.25);
  margin-bottom: 0.3rem;
}
.greeting-text {
  font-family: 'Cormorant Garamond', serif;
  font-size: clamp(1.4rem, 2.5vw, 1.9rem);
  font-weight: 600;
  color: rgba(255,255,255,0.88);
  letter-spacing: -0.01em;
  margin-bottom: 0.15rem;
}

.chartline-footer {
  position: fixed;
  bottom: 0;
  left: 240px;
  right: 0;
  z-index: 990;
  background: rgba(4, 4, 4, 0.97);
  border-top: 1px solid rgba(255,255,255,0.06);
  padding: 0.55rem 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
.chartline-footer span {
  font-family: 'DM Mono', monospace;
  font-size: 0.72rem;
  font-weight: 500;
  color: rgba(255,255,255,0.28);
  letter-spacing: 0.08em;
  white-space: nowrap;
}

.scan-progress-text {
  font-family: 'DM Mono', monospace;
  font-size: 0.76rem;
  color: rgba(255,255,255,0.45);
  letter-spacing: 0.1em;
  text-align: center;
  margin-top: 0.5rem;
  animation: fadeIn 0.3s ease both;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ────────────────────────────────────────────────────────────────────
def _rma(arr, n):
    r = np.full(len(arr), np.nan)
    if len(arr) < n: return r
    r[n-1] = np.mean(arr[:n]); a = 1/n
    for i in range(n, len(arr)): r[i] = r[i-1]*(1-a) + arr[i]*a
    return r

def wilder_rsi(close: pd.Series, n=14) -> pd.Series:
    idx = close.index; c = close.dropna().values.astype(float)
    if len(c) < n+1: return pd.Series(np.nan, index=idx)
    ch = np.diff(c, prepend=np.nan)
    up = _rma(np.where(ch>0,ch,0.)[1:], n)
    dn = _rma(np.where(ch<0,-ch,0.)[1:], n)
    rs = np.full(len(c), np.nan)
    for i in range(len(up)):
        u,d = up[i], dn[i]
        if np.isnan(u) or np.isnan(d): rs[i+1]=np.nan
        elif d==0: rs[i+1]=100.
        elif u==0: rs[i+1]=0.
        else: rs[i+1] = 100-100/(1+u/d)
    m = min(len(rs), len(idx))
    return pd.Series(rs[:m], index=idx[:m])


def compute_position(sym, amount):
    h = yf.Ticker(sym).history(period="1y", interval="1wk")
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    if h.empty or len(h) < 20:
        raise ValueError(f"Insufficient data for {sym}.")
    cp = float(h["Close"].squeeze().dropna().iloc[-1])
    if cp <= 0 or np.isnan(cp):
        raise ValueError("Bad price data.")
    high       = h["High"].squeeze()
    low        = h["Low"].squeeze()
    close_prev = h["Close"].squeeze().shift(1)
    tr = pd.concat([
        (high - low),
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(14).mean().dropna()
    if atr_series.empty:
        atr = cp * 0.03
    else:
        atr = float(atr_series.iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = cp * 0.03
    sl  = max(round(cp - 1.5 * atr, 2), round(cp * 0.85, 2))
    rd  = round(cp - sl, 2)
    if rd <= 0 or np.isnan(rd):
        raise ValueError("Risk distance is zero or invalid.")
    tp      = round(cp + 2 * rd, 2)
    raw_qty = amount / cp
    if np.isnan(raw_qty) or np.isinf(raw_qty):
        raise ValueError("Cannot calculate quantity — check price data.")
    qty = int(raw_qty)
    if qty < 1:
        raise ValueError(f"Rs.{amount:,.0f} is less than 1 share (Rs.{cp:,.2f}).")
    used = round(qty * cp, 2)
    return {
        "cp":    round(cp, 2),
        "atr":   round(atr, 2),
        "qty":   qty,
        "used":  used,
        "left":  round(amount - used, 2),
        "sl":    sl,
        "tp":    tp,
        "rd":    rd,
        "trisk": round(rd * qty, 2),
        "rpct":  round((rd * qty / amount) * 100, 2),
        "pmax":  round((tp - cp) * qty, 2),
        "lmax":  round(rd * qty, 2),
        "rr":    round((tp - cp) / rd, 2) if rd else 2.0,
    }


# ── AI ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.9, api_key=GROQ_API_KEY)


# ── CONSTANTS ──────────────────────────────────────────────────────────────────
NIFTY50_STOCKS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS","BHARTIARTL.NS","BPCL.NS",
    "BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDUSINDBK.NS",
    "INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS",
    "LTIM.NS","M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS",
    "ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS",
    "SUNPHARMA.NS","TATAMOTORS.NS","TATACONSUM.NS","TATASTEEL.NS","TCS.NS",
    "TECHM.NS","TITAN.NS","ULTRACEMCO.NS","UPL.NS","WIPRO.NS",
]

TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "SENSEX":   "^BSESN",
    **{s.replace(".NS",""): s for s in NIFTY50_STOCKS},
}


# ── DATA FETCHERS ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_ticker_data():
    items = []
    for name, sym in TICKER_MAP.items():
        try:
            h = yf.Ticker(sym).history(period='2d', interval='1d')
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            if len(h) >= 2:
                p, c = float(h['Close'].iloc[-2]), float(h['Close'].iloc[-1])
                chg = ((c - p) / p) * 100
                items.append((name, c, chg))
        except:
            pass
    return items

@st.cache_data(ttl=1800)
def fetch_news(api_key):
    try:
        if api_key:
            r = requests.get(f"https://finnhub.io/api/v1/news?category=general&token={api_key}", timeout=8)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, list) and d:
                    return [{"headline": i.get("headline",""), "source": i.get("source","Finnhub"),
                             "summary": i.get("summary",""), "url": i.get("url","#")} for i in d]
    except:
        pass
    feeds = [
        ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("Moneycontrol",   "https://www.moneycontrol.com/rss/latestnews.xml"),
        ("Reuters",        "https://feeds.reuters.com/reuters/businessNews"),
    ]
    arts = []
    for src, url in feeds:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200: continue
            for item in BeautifulSoup(r.content, "xml").find_all("item")[:6]:
                t = item.find("title"); l = item.find("link"); d = item.find("description")
                raw = re.sub(r"<[^>]+>", "", d.get_text(" ") if d else "").strip()
                if t and l:
                    arts.append({"headline": t.get_text(strip=True), "source": src,
                                 "summary": raw[:280] or "Read more.", "url": l.get_text(strip=True)})
            if len(arts) >= 20: break
        except:
            pass
    return arts


# ── SAFE yfinance FETCH ────────────────────────────────────────────────────────
def safe_fetch(ticker_obj, period, interval):
    """
    Safely fetch yfinance history. Flattens MultiIndex columns.
    Falls back gracefully on Render where some intervals are blocked.
    """
    try:
        df = ticker_obj.history(period=period, interval=interval, timeout=20)
        if df is None or (hasattr(df, 'empty') and df.empty):
            return pd.DataFrame()
        # Flatten MultiIndex columns (common with newer yfinance on Render)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Drop duplicate column names if any
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception:
        return pd.DataFrame()


def safe_fetch_download(sym, period, interval):
    """
    Use yf.download as fallback with proper MultiIndex handling.
    Returns cleaned DataFrame or empty DataFrame.
    """
    try:
        df = yf.download(sym, period=period, interval=interval,
                         auto_adjust=True, progress=False, timeout=20)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception:
        return pd.DataFrame()


def safe_info(ticker_obj):
    """Safely fetch ticker info, returns empty dict on error."""
    try:
        info = ticker_obj.info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


def get_stock_data_robust(sym, period, interval):
    """
    Try Ticker.history first, fall back to yf.download.
    Handles all MultiIndex issues.
    Intraday intervals (1m, 2m, 5m, 15m, 30m, 60m, 90m) are only 
    available for the last 60 days; on Render, 1m is often blocked,
    so we gracefully fall back to coarser intervals.
    """
    tk = yf.Ticker(sym)
    df = safe_fetch(tk, period, interval)
    if df.empty:
        df = safe_fetch_download(sym, period, interval)
    # Ensure standard column names
    rename_map = {}
    for col in df.columns:
        if col.lower() == 'open': rename_map[col] = 'Open'
        elif col.lower() == 'high': rename_map[col] = 'High'
        elif col.lower() == 'low': rename_map[col] = 'Low'
        elif col.lower() == 'close': rename_map[col] = 'Close'
        elif col.lower() == 'volume': rename_map[col] = 'Volume'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ── CHARTS ─────────────────────────────────────────────────────────────────────
def _dark(fig, h=520):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="rgba(255,255,255,0.4)", size=10),
        height=h,
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(bgcolor="rgba(10,10,10,0.8)", bordercolor="rgba(255,255,255,0.08)",
                    borderwidth=1, font=dict(size=10)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(10,10,10,0.95)", bordercolor="rgba(255,255,255,0.12)",
                        font=dict(family="Inter, sans-serif", size=10, color="rgba(255,255,255,0.8)")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)",
                     showline=False, tickfont=dict(size=9, family="Inter, sans-serif"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)",
                     showline=False, tickfont=dict(size=9, family="Inter, sans-serif"))
    return fig

def candle_chart(data, title):
    if data is None or data.empty: return None
    needed = ['Open','High','Low','Close','Volume']
    missing = [c for c in needed if c not in data.columns]
    if missing:
        return None
    d = data[needed].copy()
    d.index = pd.to_datetime(d.index)
    if hasattr(d.index, 'tz') and d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    # Drop rows where OHLC are all NaN
    d.dropna(subset=['Open','High','Low','Close'], inplace=True)
    if d.empty: return None
    d['MA7']  = d['Close'].rolling(7).mean()
    d['MA21'] = d['Close'].rolling(21).mean()
    clr = ['rgba(34,197,94,0.8)' if c >= o else 'rgba(239,68,68,0.8)'
           for c, o in zip(d['Close'], d['Open'])]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.73, 0.27])
    fig.add_trace(go.Candlestick(
        x=d.index, open=d['Open'], high=d['High'], low=d['Low'], close=d['Close'], name='Price',
        increasing=dict(line=dict(color='#22c55e', width=1), fillcolor='rgba(34,197,94,0.75)'),
        decreasing=dict(line=dict(color='#ef4444', width=1), fillcolor='rgba(239,68,68,0.75)'),
        whiskerwidth=0.35), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['MA7'],
                             line=dict(color='rgba(255,255,255,0.5)', width=1.2), name='MA7'), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d['MA21'],
                             line=dict(color='rgba(255,255,255,0.25)', width=1.2), name='MA21'), row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d['Volume'], name='Vol',
                         marker_color=clr, opacity=0.65), row=2, col=1)
    fig = _dark(fig)
    fig.update_layout(
        title=dict(text=title, font=dict(family="Cormorant Garamond", size=13, color="rgba(255,255,255,0.7)")),
        xaxis_rangeslider_visible=False)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])], row=1, col=1)
    return fig

def ma_chart(data, title):
    if data is None or data.empty: return None
    if 'Close' not in data.columns: return None
    d = data[['Close']].copy()
    d.index = pd.to_datetime(d.index)
    if hasattr(d.index, 'tz') and d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    d.dropna(inplace=True)
    if d.empty: return None
    d['MA7']  = d['Close'].rolling(7).mean()
    d['MA21'] = d['Close'].rolling(21).mean()
    d['MA50'] = d['Close'].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d['Close'], name='Price',
                             line=dict(color='rgba(255,255,255,0.6)', width=1.2)))
    fig.add_trace(go.Scatter(x=d.index, y=d['MA7'],   name='MA7',
                             line=dict(color='rgba(255,255,255,0.85)', width=1.8)))
    fig.add_trace(go.Scatter(x=d.index, y=d['MA21'],  name='MA21',
                             line=dict(color='rgba(255,255,255,0.5)', width=1.5)))
    fig.add_trace(go.Scatter(x=d.index, y=d['MA50'],  name='MA50',
                             line=dict(color='rgba(255,255,255,0.25)', width=1.5)))
    return _dark(fig.update_layout(
        title=dict(text=title, font=dict(family="Cormorant Garamond", size=13, color="rgba(255,255,255,0.7)"))), h=370)

def rsi_chart(data, title):
    if data is None or data.empty: return None
    if 'Close' not in data.columns: return None
    d = data[['Close']].copy()
    d.index = pd.to_datetime(d.index)
    if hasattr(d.index, 'tz') and d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    d.dropna(inplace=True)
    if d.empty: return None
    d['RSI'] = wilder_rsi(d['Close'], 14)
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.04)', line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(34,197,94,0.04)',  line_width=0)
    fig.add_trace(go.Scatter(x=d.index, y=d['RSI'], name="RSI",
                             line=dict(color='rgba(255,255,255,0.8)', width=1.8),
                             fill='tozeroy', fillcolor='rgba(255,255,255,0.03)'))
    fig.add_hline(y=70, line=dict(color='rgba(239,68,68,0.45)', width=1, dash='dash'),
                  annotation_text='70', annotation_font_color='rgba(239,68,68,0.6)')
    fig.add_hline(y=30, line=dict(color='rgba(34,197,94,0.45)', width=1, dash='dash'),
                  annotation_text='30', annotation_font_color='rgba(34,197,94,0.6)')
    fig.add_hline(y=50, line=dict(color='rgba(255,255,255,0.1)', width=1, dash='dot'))
    return _dark(fig.update_layout(
        title=dict(text=title, font=dict(family="Cormorant Garamond", size=13, color="rgba(255,255,255,0.7)")),
        yaxis=dict(range=[0, 100])), h=280)


# ── UTILS ──────────────────────────────────────────────────────────────────────
def get_greeting():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    hour = now.hour
    if 5 <= hour < 12:
        return "Good Morning, Trader ☀️", now.strftime("%A, %d %B %Y · %H:%M IST")
    elif 12 <= hour < 17:
        return "Good Afternoon, Trader 🌤️", now.strftime("%A, %d %B %Y · %H:%M IST")
    else:
        return "Good Evening, Trader 🌙", now.strftime("%A, %d %B %Y · %H:%M IST")

def render_ticker_bar():
    items = get_ticker_data()
    if not items: return
    def item_html(name, price, chg):
        col   = "#22c55e" if chg >= 0 else "#ef4444"
        arrow = "▲" if chg >= 0 else "▼"
        sign  = "+" if chg >= 0 else ""
        return (f'<span class="tk-item">'
                f'<span class="tk-name">{name}</span>'
                f'<span class="tk-price">&#8377;{price:,.2f}</span>'
                f'<span class="tk-chg" style="color:{col}">{arrow}{sign}{chg:.2f}%</span>'
                f'</span><span class="tk-sep">·</span>')
    inner   = "".join(item_html(*i) for i in items)
    inner2x = inner + inner
    spd     = max(22, len(items) * 3.5)
    st.markdown(f"""
<style>
.ticker-bar {{
  position: sticky; top: 0; z-index: 999;
  background: rgba(4,4,4,0.96);
  border-bottom: 1px solid rgba(255,255,255,0.07);
  backdrop-filter: blur(12px);
  height: 36px; overflow: hidden; display: flex; align-items: center;
  margin: -1rem -2rem 0.6rem; padding: 0;
  mask-image: linear-gradient(90deg, transparent, black 4%, black 96%, transparent);
  -webkit-mask-image: linear-gradient(90deg, transparent, black 4%, black 96%, transparent);
}}
.ticker-track {{
  display: flex; align-items: center;
  animation: tickerScroll {spd}s linear infinite;
  white-space: nowrap; will-change: transform;
}}
.ticker-track:hover {{ animation-play-state: paused; }}
.tk-item  {{ display: inline-flex; align-items: center; gap: 0.55rem; padding: 0 1.5rem; }}
.tk-name  {{ font-family:'DM Mono',monospace; font-size:0.66rem; color:rgba(255,255,255,0.35); text-transform:uppercase; letter-spacing:0.1em; }}
.tk-price {{ font-family:'Inter',sans-serif; font-size:0.85rem; font-weight:700; color:rgba(255,255,255,0.85); letter-spacing:-0.01em; }}
.tk-chg   {{ font-family:'Inter',sans-serif; font-size:0.72rem; font-weight:600; letter-spacing:-0.01em; }}
.tk-sep   {{ color:rgba(255,255,255,0.08); font-size:0.7rem; }}
</style>
<div class="ticker-bar"><div class="ticker-track">{inner2x}</div></div>
""", unsafe_allow_html=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
PAGES = [
    ("Dashboard", "◈"), ("Analysis",  "◉"), ("AI Predict", "◍"),
    ("Make Money","⊞"), ("AI Chat",   "◎"), ("Watchlist",  "◇"),
    ("News",      "◈"),
]

with st.sidebar:
    logo_b64 = get_logo_b64()
    if logo_b64:
        st.markdown(f"""
<div style="padding:1.5rem 1.2rem 1.4rem;border-bottom:1px solid rgba(255,255,255,0.06);">
  <img src="data:image/png;base64,{logo_b64}"
       style="width:100%;max-width:180px;display:block;margin:0 auto;border-radius:8px;object-fit:contain;" />
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="padding:2rem 1.5rem 1.8rem;border-bottom:1px solid rgba(255,255,255,0.06);">
  <div style="display:flex;align-items:center;gap:0.85rem;">
    <div style="width:34px;height:34px;background:#fff;border-radius:8px;
                display:flex;align-items:center;justify-content:center;
                font-family:'Bebas Neue',cursive;font-size:1.1rem;color:#000;
                box-shadow:0 0 24px rgba(255,255,255,0.15);">C</div>
    <div>
      <div style="font-family:'Bebas Neue',cursive;font-size:1.15rem;letter-spacing:0.12em;color:#fff;">ChartLine</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:rgba(255,255,255,0.3);
                  letter-spacing:0.14em;text-transform:uppercase;">Market Intelligence</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-label" style="margin-left:0.85rem;">Pages</div>', unsafe_allow_html=True)

    for label, icon in PAGES:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.active_page = label
            st.rerun()

    active_page = st.session_state.active_page
    st.markdown(f"""
<script>
(function() {{
  function styleNavButtons() {{
    const activePage = "{active_page}";
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (!sidebar) return;
    const buttons = sidebar.querySelectorAll('[data-testid="stButton"] > button');
    buttons.forEach(btn => {{
      const text = btn.textContent.trim();
      if (text.includes(activePage)) {{
        btn.style.background = 'rgba(255,255,255,0.18)';
        btn.style.color = '#ffffff';
        btn.style.border = '1px solid rgba(255,255,255,0.35)';
        btn.style.boxShadow = '0 4px 20px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.15)';
        btn.style.fontWeight = '700';
      }} else {{
        btn.style.background = 'rgba(255,255,255,0.05)';
        btn.style.color = 'rgba(255,255,255,0.75)';
        btn.style.border = '1px solid rgba(255,255,255,0.12)';
        btn.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';
        btn.style.fontWeight = '600';
      }}
    }});
  }}
  styleNavButtons();
  setTimeout(styleNavButtons, 100);
  setTimeout(styleNavButtons, 400);
}})();
</script>""", unsafe_allow_html=True)

    ist  = pytz.timezone("Asia/Kolkata")
    now  = datetime.now(ist)
    open_ = now.weekday() < 5 and 9*60+15 <= now.hour*60+now.minute <= 15*60+30
    sc, st_ = ("#22c55e","OPEN") if open_ else ("#ef4444","CLOSED")
    st.markdown(f"""
<div style="position:fixed;bottom:1.2rem;left:0;width:240px;padding:0 1rem;">
  <div style="background:rgba(8,8,8,0.98);border:1px solid rgba(255,255,255,0.07);
              border-radius:10px;padding:0.7rem 0.9rem;display:flex;align-items:center;gap:0.7rem;">
    <div style="width:7px;height:7px;background:{sc};border-radius:50%;
                box-shadow:0 0 8px {sc};flex-shrink:0;"></div>
    <div>
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;font-weight:500;
                  color:{sc};letter-spacing:0.1em;">NSE {st_}</div>
      <div style="font-family:'Inter',sans-serif;font-size:0.6rem;font-weight:600;
                  color:rgba(255,255,255,0.3);">{now.strftime('%H:%M:%S IST')}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ── UI HELPERS ─────────────────────────────────────────────────────────────────
def page_title(title, sub=None, tag=None):
    tag_html = (f'<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:rgba(255,255,255,0.3);border:1px solid rgba(255,255,255,0.1);'
                f'border-radius:4px;padding:0.15rem 0.5rem;">{tag}</span>') if tag else ""
    sub_html = f'<p class="lux-subtitle" style="margin-top:0.35rem;">{sub}</p>' if sub else ""
    st.markdown(f"""
<div style="padding:1.2rem 0 0.8rem;animation:fadeUp 0.5s cubic-bezier(0.4,0,0.2,1) both;">
  <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
    <h1 class="lux-title">{title}</h1>{tag_html}
  </div>{sub_html}
</div>""", unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="sec-label" style="margin:1.3rem 0 0.65rem;">{text}</div>', unsafe_allow_html=True)

def empty_state(icon, msg):
    st.markdown(f"""
<div class="lux-card" style="text-align:center;padding:3.5rem 2rem;">
  <div style="font-size:2.5rem;margin-bottom:0.8rem;opacity:0.15;filter:grayscale(1);">{icon}</div>
  <div style="font-family:'DM Mono',monospace;font-size:0.88rem;color:rgba(255,255,255,0.3);">{msg}</div>
</div>""", unsafe_allow_html=True)


# ── RENDER ─────────────────────────────────────────────────────────────────────
render_ticker_bar()
page = st.session_state.active_page


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    page_title("Welcome To Chartline", "Real-time · AI-powered · Precision trading signals", "LIVE")

    greeting_text, greeting_date = get_greeting()
    st.markdown(f"""
<div class="greeting-block" style="margin-bottom:1.2rem;">
  <div class="greeting-time">{greeting_date}</div>
  <div class="greeting-text">{greeting_text}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="position:relative;background:linear-gradient(135deg,rgba(255,255,255,0.03) 0%,rgba(255,255,255,0.015) 100%);
            border:1px solid rgba(255,255,255,0.07);border-radius:16px;
            padding:2rem 2.4rem;margin-bottom:0.9rem;overflow:hidden;">
  <div style="position:absolute;top:0;right:0;width:60%;height:100%;
              background:repeating-linear-gradient(-45deg,transparent,transparent 40px,rgba(255,255,255,0.012) 40px,rgba(255,255,255,0.012) 41px);
              pointer-events:none;"></div>
  <div style="position:absolute;top:-60px;right:-60px;width:280px;height:280px;
              border:1px solid rgba(255,255,255,0.04);border-radius:50%;pointer-events:none;"></div>
  <div style="position:absolute;top:-30px;right:-30px;width:200px;height:200px;
              border:1px solid rgba(255,255,255,0.06);border-radius:50%;pointer-events:none;"></div>
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1.2rem;position:relative;">
    <div style="animation:fadeUp 0.6s ease both;flex:1;min-width:280px;">
      <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:rgba(255,255,255,0.3);
                  letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.55rem;">◈ ChartLine Terminal</div>
      <div style="font-family:'Bebas Neue',cursive;font-size:clamp(1.9rem,3.2vw,2.8rem);
                  letter-spacing:0.06em;color:#fff;line-height:1.05;margin-bottom:0.5rem;">
        Built for those who trade<br>Indian markets</div>
      <p style="font-family:'DM Mono',monospace;font-size:0.82rem;color:rgba(255,255,255,0.35);
                max-width:480px;line-height:1.75;margin:0;">
        AI-powered signals · RSI screening · 5-model ML forecasting ·
        Real-time NSE/BSE data</p>
    </div>
    <div style="display:flex;flex-direction:row;gap:0.7rem;animation:fadeUp 0.6s 0.15s ease both;align-items:flex-start;flex-shrink:0;">
      <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                  border-radius:12px;padding:0.9rem 1.2rem;text-align:center;min-width:88px;">
        <div style="font-family:'Inter',sans-serif;font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">5</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:rgba(255,255,255,0.3);
                    text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">ML Models</div>
      </div>
      <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                  border-radius:12px;padding:0.9rem 1.2rem;text-align:center;min-width:88px;">
        <div style="font-family:'Inter',sans-serif;font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">50</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:rgba(255,255,255,0.3);
                    text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">Nifty Stocks</div>
      </div>
      <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
                  border-radius:12px;padding:0.9rem 1.2rem;text-align:center;min-width:88px;">
        <div style="font-family:'Inter',sans-serif;font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">7</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:rgba(255,255,255,0.3);
                    text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">Pages</div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(212,175,55,0.05);border:1px solid rgba(212,175,55,0.15);
            border-radius:100px;padding:0.7rem 1.1rem;font-family:'DM Mono',monospace;
            font-size:0.78rem;color:rgba(212,175,55,0.7);line-height:1.8;margin-bottom:0.5rem;">
  &#9888; &nbsp;Disclaimer: AI signals are for educational purposes only. Consult a SEBI-registered advisor before investing.
  Past performance does not guarantee future results. Capital at risk.
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="chartline-footer"><span>Copyright &copy; Chartline By Ayush More</span></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS  — FULLY FIXED FOR RENDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analysis":
    page_title("Stock Analysis", "Candlestick · Moving Averages · Wilder's RSI · Live snapshot")

    lc, rc = st.columns([1, 3], gap="large")
    with lc:
        sec("Select Stock")
        sel_opts  = ["— type or pick below —"] + NIFTY50_STOCKS
        sel_idx   = 0
        cur_active = st.session_state.get("active_stock", "")
        if cur_active in NIFTY50_STOCKS:
            sel_idx = NIFTY50_STOCKS.index(cur_active) + 1
        selected_ns = st.selectbox("Nifty 50 Stock", options=sel_opts, index=sel_idx,
                                   key="analysis_select_ns", label_visibility="collapsed")
        run = st.button("Analyse →", use_container_width=True, key="analysis_run_btn")
        st.markdown("""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
            border-radius:8px;padding:0.65rem 0.8rem;margin-top:0.8rem;
            font-family:'DM Mono',monospace;font-size:0.68rem;color:rgba(255,255,255,0.28);line-height:2;">
  &#9679; All 50 Nifty stocks supported<br>&#9679; NSE Exchange only<br>&#9679; Real-time data via yFinance
</div>""", unsafe_allow_html=True)

    if run:
        if selected_ns == "— type or pick below —":
            st.error("Please select a stock from the dropdown.")
        else:
            st.session_state.active_stock      = selected_ns
            st.session_state.stock_data_loaded = True

    with rc:
        cs = st.session_state.get("active_stock", "")
        if not st.session_state.get("stock_data_loaded") or not cs:
            empty_state("◉", "Select a stock from the dropdown and click Analyse →")
        else:
            with st.spinner(f"Fetching {cs}…"):
                tk   = yf.Ticker(cs)
                info = safe_info(tk)

            pe      = info.get('trailingPE')
            pe_s    = f"{pe:.1f}" if isinstance(pe, (int, float)) and not np.isnan(float(pe)) else 'N/A'
            dy      = info.get('dividendYield')
            dy_s    = f"{dy*100:.2f}%" if isinstance(dy, (int, float)) else 'N/A'
            beta    = info.get('beta')
            beta_s  = f"{beta:.2f}" if isinstance(beta, (int, float)) else 'N/A'
            mc      = info.get('marketCap')
            mc_s    = f"Rs.{mc/1e9:.1f}B" if mc else 'N/A'
            hi52    = info.get('fiftyTwoWeekHigh')
            lo52    = info.get('fiftyTwoWeekLow')
            hi52_str = f"Rs.{hi52:,.2f}" if isinstance(hi52, (int, float)) else 'N/A'
            lo52_str = f"Rs.{lo52:,.2f}" if isinstance(lo52, (int, float)) else 'N/A'
            long_name = info.get('longName', cs)
            exchange  = info.get('exchange', 'NSE')
            currency  = info.get('currency', 'INR')
            sector    = info.get('sector', 'N/A')

            st.markdown(f"""
<div class="lux-card" style="margin-bottom:1rem;padding:1.2rem 1.5rem;">
  <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:0.8rem;">
    <span class="sym-chip">{cs}</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:rgba(255,255,255,0.3);">
      {exchange} · {currency} · {sector}
    </span>
  </div>
  <div style="font-family:'Cormorant Garamond',serif;font-size:1.4rem;font-weight:600;color:#fff;">
    {long_name}</div>
  <div style="display:flex;gap:2rem;margin-top:0.9rem;flex-wrap:wrap;">
    <div><div class="sec-label" style="margin:0 0 0.2rem;">Market Cap</div>
      <div style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:700;color:#fff;">{mc_s}</div></div>
    <div><div class="sec-label" style="margin:0 0 0.2rem;">P/E</div>
      <div style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:700;color:#fff;">{pe_s}</div></div>
    <div><div class="sec-label" style="margin:0 0 0.2rem;">Div Yield</div>
      <div style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:700;color:#fff;">{dy_s}</div></div>
    <div><div class="sec-label" style="margin:0 0 0.2rem;">Beta</div>
      <div style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:700;color:#fff;">{beta_s}</div></div>
    <div><div class="sec-label" style="margin:0 0 0.2rem;">52W High</div>
      <div style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:700;color:#22c55e;">{hi52_str}</div></div>
    <div><div class="sec-label" style="margin:0 0 0.2rem;">52W Low</div>
      <div style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:700;color:#ef4444;">{lo52_str}</div></div>
  </div>
</div>""", unsafe_allow_html=True)

            tab_snap, tab_candle, tab_ma, tab_rsi = st.tabs([
                "Today's Snapshot", "Candlestick Chart", "Moving Averages", "Wilder's RSI (14)"])

            # ── TODAY'S SNAPSHOT ──────────────────────────────────────────────
            with tab_snap:
                # Try 1d/1m first, fall back to 5d/1d on failure (Render often blocks 1m)
                snapshot_loaded = False
                
                # Attempt 1: intraday 5m (more reliable than 1m on Render)
                td = get_stock_data_robust(cs, '1d', '5m')
                if not td.empty and 'Close' in td.columns and len(td) >= 2:
                    try:
                        close_vals = td['Close'].dropna()
                        open_vals  = td['Open'].dropna()
                        rtp = float(close_vals.iloc[-1])
                        rto = float(open_vals.iloc[0])
                        rc_ = ((rtp - rto) / rto) * 100 if rto else 0.0
                        high_val = float(td['High'].max()) if 'High' in td.columns else rtp
                        low_val  = float(td['Low'].min())  if 'Low'  in td.columns else rtp
                        vol_val  = int(td['Volume'].sum())  if 'Volume' in td.columns else 0
                        m1,m2,m3,m4,m5 = st.columns(5)
                        m1.metric("Open",    f"Rs.{rto:,.2f}")
                        m2.metric("Current", f"Rs.{rtp:,.2f}", delta=f"{rc_:+.2f}%")
                        m3.metric("High",    f"Rs.{high_val:,.2f}")
                        m4.metric("Low",     f"Rs.{low_val:,.2f}")
                        m5.metric("Volume",  f"{vol_val:,}")
                        snapshot_loaded = True
                    except Exception:
                        snapshot_loaded = False

                # Attempt 2: last daily candle
                if not snapshot_loaded:
                    td_daily = get_stock_data_robust(cs, '5d', '1d')
                    if not td_daily.empty and 'Close' in td_daily.columns:
                        try:
                            close_vals = td_daily['Close'].dropna()
                            open_vals  = td_daily['Open'].dropna()  if 'Open'   in td_daily.columns else close_vals
                            high_vals  = td_daily['High'].dropna()  if 'High'   in td_daily.columns else close_vals
                            low_vals   = td_daily['Low'].dropna()   if 'Low'    in td_daily.columns else close_vals
                            vol_vals   = td_daily['Volume'].dropna() if 'Volume' in td_daily.columns else pd.Series([0])
                            rtp = float(close_vals.iloc[-1])
                            rto = float(open_vals.iloc[-1])
                            rc_ = ((rtp - rto) / rto) * 100 if rto else 0.0
                            m1,m2,m3,m4,m5 = st.columns(5)
                            m1.metric("Open",   f"Rs.{rto:,.2f}")
                            m2.metric("Close",  f"Rs.{rtp:,.2f}", delta=f"{rc_:+.2f}%")
                            m3.metric("High",   f"Rs.{float(high_vals.iloc[-1]):,.2f}")
                            m4.metric("Low",    f"Rs.{float(low_vals.iloc[-1]):,.2f}")
                            m5.metric("Volume", f"{int(vol_vals.iloc[-1]):,}")
                            snapshot_loaded = True
                        except Exception as e:
                            st.info(f"Snapshot unavailable: {e}")
                    else:
                        st.info("No market data available for this stock right now. Please try again later.")

            # ── CANDLESTICK ───────────────────────────────────────────────────
            with tab_candle:
                # Note: 1m data is NOT available on Render reliably — using 5m for "1D" view
                ctabs = st.tabs(["1D","1W","1M","6M","1Y","5Y"])
                cpds  = [
                    ('1d',  '5m'),   # 1D — use 5m (1m often blocked on Render)
                    ('5d',  '15m'),  # 1W
                    ('1mo', '1d'),   # 1M
                    ('6mo', '1d'),   # 6M
                    ('1y',  '1wk'),  # 1Y
                    ('5y',  '1mo'),  # 5Y
                ]
                for ctab, (period, interval) in zip(ctabs, cpds):
                    with ctab:
                        d = get_stock_data_robust(cs, period, interval)
                        if not d.empty and 'Close' in d.columns:
                            fig = candle_chart(d, f"{cs} ({period})")
                            if fig:
                                # Disable weekend rangebreaks for intraday / monthly charts
                                if interval in ('5m', '15m', '1mo'):
                                    fig.update_xaxes(rangebreaks=[])
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"Chart could not be rendered for {period} timeframe.")
                        else:
                            st.info(f"No data available for {period} timeframe. Please try another period.")

            # ── MOVING AVERAGES ───────────────────────────────────────────────
            with tab_ma:
                matabs = st.tabs(["1D","1W","1M","1Y"])
                mapds  = [
                    ('1d',  '5m'),
                    ('5d',  '15m'),
                    ('1mo', '1d'),
                    ('1y',  '1wk'),
                ]
                for matab, (p, i) in zip(matabs, mapds):
                    with matab:
                        d = get_stock_data_robust(cs, p, i)
                        if not d.empty and 'Close' in d.columns:
                            fig = ma_chart(d, f"{cs} — MA")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"Not enough data for MA chart ({p}).")
                        else:
                            st.info(f"No data available for {p} timeframe.")

            # ── RSI ───────────────────────────────────────────────────────────
            with tab_rsi:
                rsitabs = st.tabs(["1D","1W","1M","1Y"])
                rsipds  = [
                    ('1d',  '5m'),
                    ('5d',  '15m'),
                    ('1mo', '1d'),
                    ('1y',  '1wk'),
                ]
                for rsitab, (p, i) in zip(rsitabs, rsipds):
                    with rsitab:
                        d = get_stock_data_robust(cs, p, i)
                        if not d.empty and 'Close' in d.columns:
                            fig = rsi_chart(d, f"{cs} — RSI")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"Not enough data for RSI chart ({p}).")
                        else:
                            st.info(f"No data available for {p} timeframe.")


# ══════════════════════════════════════════════════════════════════════════════
# AI PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AI Predict":
    page_title("AI Predict", "5 ML models trained on 5Y data · Next-day price forecast")

    lc, rc = st.columns([1, 3], gap="large")
    with lc:
        sec("Select Stock")
        ai_sel_opts   = ["— select a stock —"] + NIFTY50_STOCKS
        ai_sel_idx    = 0
        cur_ai        = st.session_state.get("ai_results") or {}
        cur_ai_ticker = cur_ai.get("ticker", "")
        if cur_ai_ticker in NIFTY50_STOCKS:
            ai_sel_idx = NIFTY50_STOCKS.index(cur_ai_ticker) + 1
        ai_sel_ns = st.selectbox("Nifty 50 Stock", options=ai_sel_opts, index=ai_sel_idx,
                                 key="ai_select_ns", label_visibility="collapsed")
        ai_run = st.button("Run AI Analysis →", use_container_width=True, key="ai_run_btn")
        st.markdown("""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
            border-radius:8px;padding:0.7rem;margin-top:0.8rem;
            font-family:'DM Mono',monospace;font-size:0.72rem;color:rgba(255,255,255,0.28);line-height:2.1;">
  &#9702; Linear Regression<br>&#9702; Decision Tree<br>&#9702; Random Forest<br>
  &#9702; Gradient Boosting<br>&#9702; KNN Regressor
</div>""", unsafe_allow_html=True)

    with rc:
        if ai_run:
            if ai_sel_ns == "— select a stock —":
                st.error("Please select a stock from the dropdown.")
                st.stop()
            AISTOCK = ai_sel_ns
            with st.spinner(f"Training 5 ML models on {AISTOCK}…"):
                try:
                    df = get_stock_data_robust(AISTOCK, '5y', '1d')
                    if df.empty: st.error("No data."); st.stop()
                    df = df[['Open','High','Low','Close','Volume']].copy()
                    df.reset_index(inplace=True)
                    df['MA_7']           = df['Close'].rolling(7).mean()
                    df['MA_21']          = df['Close'].rolling(21).mean()
                    df['MA_50']          = df['Close'].rolling(50).mean()
                    df['Daily_Return']   = df['Close'].pct_change()
                    df['High_Low_Range'] = df['High'] - df['Low']
                    df['Volatility']     = df['Close'].rolling(7).std()
                    df['RSI']            = wilder_rsi(df['Close'], 14).values
                    df['Target']         = df['Close'].shift(-1)
                    df.dropna(inplace=True)
                    feats = ['Open','High','Low','Close','Volume','MA_7','MA_21','MA_50',
                             'Daily_Return','High_Low_Range','Volatility','RSI']
                    X = df[feats]; y = df['Target']
                    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, shuffle=False)
                    sc_ = StandardScaler()
                    Xtr_s = sc_.fit_transform(Xtr); Xte_s = sc_.transform(Xte)
                    mdls = {
                        'Linear Regression':   LinearRegression(),
                        'Decision Tree':       DecisionTreeRegressor(random_state=42),
                        'Random Forest':       RandomForestRegressor(n_estimators=100, random_state=42),
                        'Gradient Boosting':   GradientBoostingRegressor(n_estimators=100, random_state=42),
                        'KNN Regressor':       KNeighborsRegressor(n_neighbors=5),
                    }
                    res = []
                    for mn, mo in mdls.items():
                        mo.fit(Xtr_s, ytr); yp = mo.predict(Xte_s)
                        res.append({'Model': mn,
                                    'MAE':  round(mean_absolute_error(yte, yp), 4),
                                    'RMSE': round(np.sqrt(mean_squared_error(yte, yp)), 4),
                                    'R2':   round(r2_score(yte, yp), 4)})
                    rdf    = pd.DataFrame(res).sort_values('R2', ascending=False).reset_index(drop=True)
                    best_n = rdf.iloc[0]['Model']
                    ls     = sc_.transform(X.iloc[-1].values.reshape(1, -1))
                    preds  = {n: round(m.predict(ls)[0], 2) for n, m in mdls.items()}
                    tc     = float(df['Close'].iloc[-1]); bp = preds[best_n]
                    ch     = bp - tc; pct = (ch / tc) * 100
                    st.session_state.ai_results = {
                        "ticker": AISTOCK, "preds": preds, "best": best_n,
                        "tc": tc, "bp": bp, "ch": ch, "pct": pct,
                        "dir": "UP ▲" if ch > 0 else "DOWN ▼"}
                except Exception as e:
                    st.error(f"Error: {e}")

        r = st.session_state.get("ai_results")
        if r:
            st.markdown(f'<span class="sym-chip" style="margin-bottom:1rem;display:inline-block;">'
                        f'{r["ticker"]}</span>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Today's Close", f"Rs.{r['tc']:.2f}")
            c2.metric("Best Forecast", f"Rs.{r['bp']:.2f}", delta=f"{r['ch']:+.2f} ({r['pct']:+.2f}%)")
            c3.metric("Direction",     r['dir'])
            st.divider(); sec("All Model Predictions")
            pc = st.columns(5)
            for i, (n, p) in enumerate(r['preds'].items()):
                pc[i].metric(n, f"Rs.{p:,.2f}", delta=f"{p - r['tc']:+.2f}")
        else:
            empty_state("◍", "Select a stock and click Run AI Analysis →")


# ══════════════════════════════════════════════════════════════════════════════
# MAKE MONEY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Make Money":
    page_title("Make Money Strategy", "Nifty 50 · RSI >= 60 · SMA(20) · Volume breakouts · 1:2 R:R", "NIFTY 50")

    top_left, top_right = st.columns([3, 1], gap="medium")
    with top_left:
        st.markdown("""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
            border-radius:8px;padding:0.55rem 0.9rem;
            font-family:'DM Mono',monospace;font-size:0.72rem;color:rgba(255,255,255,0.3);
            display:flex;gap:1.8rem;flex-wrap:wrap;">
  <span>RSI(14) &ge; 60</span><span>Close &ge; SMA(20)</span>
  <span>Vol &gt; 1.5&times; Avg</span><span>R:R = 1:2</span>
</div>""", unsafe_allow_html=True)
    with top_right:
        scan_btn = st.button("Run Scan →", key="scanner_run_btn", use_container_width=True)

    if scan_btn:
        st.session_state.rsi_show_invest_panel    = False
        st.session_state.rsi_selected_stock       = None
        st.session_state.rsi_investment_submitted = False

        nifty50 = [
            "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","HINDUNILVR.NS","ITC.NS",
            "SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS",
            "SUNPHARMA.NS","TITAN.NS","BAJFINANCE.NS","NESTLEIND.NS","WIPRO.NS","ULTRACEMCO.NS",
            "POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS","ADANIENT.NS","ADANIPORTS.NS",
            "COALINDIA.NS","ONGC.NS","JSWSTEEL.NS","TATASTEEL.NS","BAJAJFINSV.NS","BAJAJ-AUTO.NS",
            "HEROMOTOCO.NS","DIVISLAB.NS","DRREDDY.NS","CIPLA.NS","EICHERMOT.NS","BPCL.NS",
            "BRITANNIA.NS","GRASIM.NS","HINDALCO.NS","INDUSINDBK.NS","M&M.NS","SBILIFE.NS",
            "HDFCLIFE.NS","APOLLOHOSP.NS","TATACONSUM.NS","UPL.NS","LTIM.NS",
        ]
        total = len(nifty50); fresh = []
        pb            = st.progress(0, "Initializing scan…")
        progress_text = st.empty()

        for i, sym in enumerate(nifty50):
            loaded    = i + 1
            pct       = loaded / total
            sym_short = sym.replace('.NS', '')
            pb.progress(pct, f"Scanning {sym_short}…")
            progress_text.markdown(
                f'<div class="scan-progress-text">Loaded {loaded} of {total} — {sym_short}</div>',
                unsafe_allow_html=True)
            try:
                dw = get_stock_data_robust(sym, "2y", "1wk")
                if dw.empty or len(dw) < 30: continue
                cl  = dw["Close"].squeeze().dropna()
                vol = dw["Volume"].squeeze()
                rsi_= wilder_rsi(cl, 14)
                lr  = float(rsi_.iloc[-1])
                if np.isnan(lr): continue
                sma = cl.rolling(20).mean()
                lp  = float(cl.iloc[-1]); ls_ = float(sma.iloc[-1])
                if np.isnan(ls_): continue
                av  = float(vol.rolling(20).mean().iloc[-1]); lv = float(vol.iloc[-1])
                if lr >= 60 and lp >= ls_:
                    sig = "BUY — Volume Breakout" if lv > 1.5 * av else "RSI OK — No Volume"
                    fresh.append((sym.replace(".NS",""), round(lp,2), round(lr,2),
                                  round(ls_,2), int(lv), int(av), sig, sym))
            except:
                pass

        pb.empty(); progress_text.empty()
        fresh.sort(key=lambda x: x[2], reverse=True)
        st.session_state.rsi_scan_results   = fresh
        st.session_state.rsi_last_scan_done = True

    if st.session_state.get("rsi_last_scan_done"):
        res = st.session_state.rsi_scan_results
        if not res:
            st.warning("No stocks passed the criteria.")
        else:
            left_col, right_col = st.columns([3, 2], gap="large")

            with left_col:
                sec("Scan Results — click GO to calculate position")
                for r in res:
                    vr        = round(r[4] / r[5], 2) if r[5] > 0 else 0
                    is_buy    = "BUY" in r[6]
                    sig_label = "Vol Breakout" if is_buy else "RSI OK"
                    sig_color = "#22c55e" if is_buy else "rgba(255,255,255,0.4)"
                    is_sel    = (st.session_state.rsi_selected_stock is not None and
                                 st.session_state.rsi_selected_stock[0] == r[0])
                    row_bg    = "rgba(255,255,255,0.07)" if is_sel else "rgba(255,255,255,0.02)"
                    row_brd   = "rgba(255,255,255,0.18)" if is_sel else "rgba(255,255,255,0.06)"

                    c1,c2,c3,c4,c5,c6 = st.columns([2,2,1.2,1.5,2,1.5])
                    with c1:
                        st.markdown(f"""
<div style="background:{row_bg};border:1px solid {row_brd};border-radius:7px;
            padding:0.5rem 0.7rem;font-family:'Inter',sans-serif;font-size:0.82rem;
            font-weight:700;color:#fff;letter-spacing:0.03em;">{r[0]}</div>""",
                                    unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
<div style="padding:0.5rem 0.3rem;font-family:'Inter',sans-serif;font-size:0.9rem;
            font-weight:600;color:rgba(255,255,255,0.82);">Rs.{r[1]:,.2f}</div>""",
                                    unsafe_allow_html=True)
                    with c3:
                        rsi_col = "#22c55e" if r[2] >= 70 else "rgba(255,255,255,0.75)"
                        st.markdown(f"""
<div style="padding:0.5rem 0.3rem;font-family:'Inter',sans-serif;font-size:0.82rem;
            font-weight:700;color:{rsi_col};">{r[2]}</div>""", unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"""
<div style="padding:0.5rem 0.3rem;font-family:'Inter',sans-serif;font-size:0.75rem;
            font-weight:600;color:rgba(255,255,255,0.4);">{vr}x</div>""",
                                    unsafe_allow_html=True)
                    with c5:
                        st.markdown(f"""
<div style="padding:0.45rem 0.3rem;">
  <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:{sig_color};letter-spacing:0.07em;">
    {'▲ ' if is_buy else '— '}{sig_label}
  </span>
</div>""", unsafe_allow_html=True)
                    with c6:
                        if st.button("GO", key=f"sel_{r[0]}", use_container_width=True):
                            st.session_state.rsi_selected_stock       = r
                            st.session_state.rsi_show_invest_panel    = True
                            st.session_state.rsi_investment_submitted = False
                            st.session_state.rsi_investment_amount    = 0.0
                            st.rerun()

            with right_col:
                if st.session_state.rsi_show_invest_panel and st.session_state.rsi_selected_stock:
                    sel            = st.session_state.rsi_selected_stock
                    dn, sp, sr, fs = sel[0], sel[1], sel[2], sel[7]

                    amt_val = float(st.session_state.rsi_investment_amount) \
                              if st.session_state.rsi_investment_amount > 0 else 10000.0

                    pos_result = None
                    pos_error  = None
                    if st.session_state.rsi_investment_submitted and \
                       st.session_state.rsi_investment_amount > 0:
                        with st.spinner("Calculating position size…"):
                            try:
                                pos_result = compute_position(fs, st.session_state.rsi_investment_amount)
                            except Exception as e:
                                pos_error = str(e)

                    results_html = ""
                    if pos_result:
                        r_ = pos_result
                        results_html = f"""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
            border-radius:10px;padding:0.7rem 0.9rem;margin-top:0.6rem;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.3rem 1.2rem;
              font-family:'DM Mono',monospace;font-size:0.72rem;">
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">Entry</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:#fff;">Rs.{r_['cp']:,.2f}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">Shares</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:#fff;">{r_['qty']}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">Capital</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:#fff;">Rs.{r_['used']:,.0f}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">Leftover</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:600;color:rgba(255,255,255,0.5);">Rs.{r_['left']:,.0f}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">Stop Loss</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:#ef4444;">Rs.{r_['sl']:,.2f}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">Target</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:#22c55e;">Rs.{r_['tp']:,.2f}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">Max Profit</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:#22c55e;">+Rs.{r_['pmax']:,.0f}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">Max Loss</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:#ef4444;">-Rs.{r_['lmax']:,.0f}</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
      <span style="color:rgba(255,255,255,0.35);">% at Risk</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:rgba(212,175,55,0.9);">{r_['rpct']}%</strong>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.25rem 0;">
      <span style="color:rgba(255,255,255,0.35);">R : R</span>
      <strong style="font-family:'Inter',sans-serif;font-weight:700;color:#fff;">1 : {r_['rr']}</strong>
    </div>
  </div>
</div>"""

                    if pos_error and not results_html:
                        results_html = (f'<div style="color:#ef4444;font-family:\'DM Mono\',monospace;'
                                        f'font-size:0.78rem;padding:0.6rem 0;margin-top:0.6rem;">'
                                        f'{pos_error}</div>')

                    st.markdown(f"""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.09);
            border-radius:12px;padding:0.9rem 1rem;">
  <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.7rem;">
    <span class="sym-chip">{dn}</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:rgba(255,255,255,0.3);">
      RSI {sr} &nbsp;·&nbsp; Rs.{sp:,.2f}
    </span>
  </div>
  {results_html}
</div>""", unsafe_allow_html=True)

                    amt = st.number_input(
                        "Investment Amount (Rs.)",
                        min_value=100.0, max_value=100_000_000.0,
                        value=amt_val, step=500.0, format="%.2f", key="invest_amount")

                    c_calc, c_close = st.columns([2, 1])
                    with c_calc:
                        calc_btn = st.button("Calculate →", key="invest_submit", use_container_width=True)
                    with c_close:
                        if st.button("Clear", key="invest_back", use_container_width=True):
                            st.session_state.rsi_show_invest_panel    = False
                            st.session_state.rsi_selected_stock       = None
                            st.session_state.rsi_investment_submitted = False
                            st.rerun()

                    if calc_btn:
                        if amt < sp:
                            st.error(f"Rs.{amt:,.0f} is less than 1 share (Rs.{sp:,.2f}).")
                        else:
                            st.session_state.rsi_investment_amount    = amt
                            st.session_state.rsi_investment_submitted = True
                            st.rerun()
                else:
                    st.markdown("""
<div style="background:rgba(255,255,255,0.015);border:1px dashed rgba(255,255,255,0.08);
            border-radius:10px;padding:2.5rem 1.5rem;text-align:center;margin-top:2.5rem;">
  <div style="font-size:1.8rem;opacity:0.1;margin-bottom:0.6rem;">&#8594;</div>
  <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:rgba(255,255,255,0.2);">
    Select a stock from the list to calculate position size
  </div>
</div>""", unsafe_allow_html=True)
    else:
        empty_state("⊞", "Click Run Scan → to begin screening Nifty 50")


# ══════════════════════════════════════════════════════════════════════════════
# AI CHAT — FULLY FIXED: no st.stop(), proper key handling, no rerun on input
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AI Chat":
    page_title("AI Chat", "Groq LLaMA 3.1 · NSE/BSE · Strategies · Technical & Fundamental analysis")

    # Check API key availability upfront
    if not GROQ_API_KEY:
        st.warning(
            "⚠️ Groq API key not found. Please set `GROQ_API_KEY` in your `.env` file or "
            "Render environment variables and redeploy.",
            icon="⚠️"
        )

    lc, rc = st.columns([1, 3], gap="large")
    with lc:
        sec("Chat Controls")
        st.markdown("""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
            border-radius:8px;padding:0.7rem;margin-bottom:0.9rem;
            font-family:'DM Mono',monospace;font-size:0.72rem;color:rgba(255,255,255,0.28);line-height:2.1;">
  Model: Groq LLaMA 3.1<br>Context: 10 turns<br>
  Expertise: NSE · BSE · F&amp;O<br>Technical &amp; Fundamental
</div>""", unsafe_allow_html=True)
        if st.session_state.chat_history:
            if st.button("Clear Chat →", key="clear_chat_btn", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    with rc:
        # Show welcome card only when chat is empty
        if not st.session_state.chat_history:
            st.markdown("""
<div class="lux-card" style="text-align:center;padding:2.5rem 2rem;margin-bottom:1rem;">
  <div style="font-size:2.2rem;margin-bottom:0.7rem;opacity:0.2;">&#9678;</div>
  <div style="font-family:'Cormorant Garamond',serif;font-size:1.2rem;font-weight:600;color:#fff;margin-bottom:0.3rem;">
    AI Market Expert</div>
  <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:rgba(255,255,255,0.3);">
    Ask about stocks · strategies · RSI · MACD · F&amp;O · valuation</div>
</div>""", unsafe_allow_html=True)

        # Render existing messages
        for msg in st.session_state.chat_history[-20:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input — this must be OUTSIDE any conditional block that uses st.stop()
        # to avoid the input box disappearing
        user_in = st.chat_input("Ask anything about markets…", key="main_chat_input")

        if user_in and user_in.strip():
            # Immediately add user message and show it
            st.session_state.chat_history.append({"role": "user", "content": user_in.strip()})

            if not GROQ_API_KEY:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "⚠️ Groq API key is not configured. Please set GROQ_API_KEY in your .env file or Render environment variables."
                })
            else:
                try:
                    groq_client = get_groq_client()
                    if groq_client is None:
                        raise ValueError("Could not initialize Groq client. Check your GROQ_API_KEY.")

                    sys_msg = SystemMessage(content=(
                        "You are an expert stock market analyst specializing in Indian & US markets. "
                        "You have deep knowledge of NSE, BSE, Nifty 50, Sensex, F&O, intraday, and swing trading. "
                        "Technical analysis expertise: RSI, MACD, Bollinger Bands, Moving Averages, Fibonacci, VWAP. "
                        "Fundamental analysis: P/E ratio, EPS, Market Cap, D/E ratio, ROE, ROCE. "
                        "Always structure your response: Analysis → Strategy → Risk Management → Disclaimer. "
                        "Keep responses under 250 words. Be concise and actionable."))

                    msgs = [sys_msg] + [
                        HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"])
                        for m in st.session_state.chat_history[-10:]
                    ]
                    resp = groq_client.invoke(msgs)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp.content
                    })
                except Exception as e:
                    err_msg = str(e)
                    if "api_key" in err_msg.lower() or "authentication" in err_msg.lower() or "401" in err_msg:
                        friendly = "⚠️ Authentication error with Groq API. Please verify your GROQ_API_KEY is valid and not expired."
                    elif "rate" in err_msg.lower() or "429" in err_msg:
                        friendly = "⚠️ Groq API rate limit reached. Please wait a moment and try again."
                    else:
                        friendly = f"⚠️ Error communicating with AI: {err_msg}"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": friendly
                    })

            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST — FULLY FIXED FOR RENDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Watchlist":
    page_title("Watchlist", "Track stocks in real time · Live prices · Key metrics")

    lc, rc = st.columns([1, 3], gap="large")
    with lc:
        sec("Add Stock")
        wl_sel_opts = ["— pick a stock —"] + NIFTY50_STOCKS
        wl_sel = st.selectbox("Nifty 50 Quick Add", options=wl_sel_opts, index=0,
                              key="wl_select_ns", label_visibility="collapsed")
        add_ = st.button("Add →", use_container_width=True, key="wl_add_btn")
        st.markdown(f"""
<div style="margin-top:0.8rem;font-family:'DM Mono',monospace;font-size:0.72rem;
            color:rgba(255,255,255,0.25);text-align:center;padding:0.5rem;
            background:rgba(255,255,255,0.02);border-radius:6px;">
  {len(st.session_state.watchlist)} stock(s) tracked · NSE only
</div>""", unsafe_allow_html=True)

    if add_:
        to_add = wl_sel if wl_sel != "— pick a stock —" else ""
        if not to_add:
            with rc:
                st.warning("Please select a stock from the dropdown.")
        elif not to_add.endswith(".NS"):
            with rc:
                st.error("Only NSE (.NS) stocks supported.")
        elif to_add in st.session_state.watchlist:
            with rc:
                st.info(f"{to_add} is already in your watchlist.")
        else:
            add_ok  = False
            add_err = ""
            try:
                # Use robust fetch — handles MultiIndex automatically
                h_check = get_stock_data_robust(to_add, '5d', '1d')
                if not h_check.empty and 'Close' in h_check.columns and len(h_check) >= 1:
                    add_ok = True
                else:
                    add_err = f"No market data for {to_add}. It may be unavailable right now."
            except Exception as ex:
                add_err = f"Could not validate {to_add}: {ex}"

            if add_ok:
                st.session_state.watchlist.append(to_add)
                st.rerun()
            else:
                with rc:
                    st.error(add_err)

    with rc:
        if not st.session_state.watchlist:
            empty_state("◇", "Your watchlist is empty — add tickers in the left panel")
        else:
            sec("Your Stocks")
            to_remove = None
            for sym in st.session_state.watchlist:
                fetch_ok = False
                cv = chg = 0.0
                pe_ = mc_ = name = "N/A"
                try:
                    # Use robust fetch for MultiIndex safety
                    h_ = get_stock_data_robust(sym, '5d', '1d')
                    t_ = yf.Ticker(sym)
                    i_ = safe_info(t_)

                    if not h_.empty and 'Close' in h_.columns:
                        close_vals = h_['Close'].dropna()
                        if len(close_vals) >= 2:
                            pv   = float(close_vals.iloc[-2])
                            cv   = float(close_vals.iloc[-1])
                            chg  = ((cv - pv) / pv) * 100 if pv else 0.0
                        elif len(close_vals) == 1:
                            cv   = float(close_vals.iloc[-1])
                            chg  = 0.0
                        fetch_ok = cv > 0

                    mc   = i_.get('marketCap')
                    mc_  = f"Rs.{mc/1e9:.1f}B" if mc else "N/A"
                    pe   = i_.get('trailingPE')
                    pe_  = f"{pe:.1f}" if isinstance(pe, (int, float)) else "N/A"
                    raw_name = i_.get('shortName','') or i_.get('longName','') or sym
                    name = str(raw_name)[:22]
                except Exception:
                    fetch_ok = False

                chg_col   = "#22c55e" if chg >= 0 else "#ef4444"
                chg_arrow = "▲" if chg >= 0 else "▼"
                chg_sign  = "+" if chg >= 0 else ""

                col_main, col_remove = st.columns([10, 1])
                with col_main:
                    if fetch_ok:
                        st.markdown(f"""
<div class="wl-row">
  <div style="display:flex;flex-direction:column;min-width:130px;">
    <span class="wl-ticker">{sym}</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:rgba(255,255,255,0.28);margin-top:1px;">{name}</span>
  </div>
  <div class="wl-price">Rs.{cv:,.2f}</div>
  <div class="wl-chg" style="color:{chg_col};">{chg_arrow} {chg_sign}{abs(chg):.2f}%</div>
  <div class="wl-meta">P/E {pe_}</div>
  <div class="wl-meta">{mc_}</div>
</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div class="wl-row" style="opacity:0.5;">
  <div style="display:flex;flex-direction:column;min-width:130px;">
    <span class="wl-ticker">{sym}</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:rgba(255,255,255,0.28);margin-top:1px;">Data unavailable</span>
  </div>
  <div class="wl-price" style="color:rgba(255,255,255,0.3);">—</div>
  <div class="wl-chg" style="color:rgba(255,255,255,0.25);">— —</div>
  <div class="wl-meta">—</div><div class="wl-meta">—</div>
</div>""", unsafe_allow_html=True)

                with col_remove:
                    st.markdown('<div style="padding-top:0.35rem;">', unsafe_allow_html=True)
                    if st.button("✕", key=f"wl_rm_{sym}", help=f"Remove {sym}"):
                        to_remove = sym
                    st.markdown('</div>', unsafe_allow_html=True)

            if to_remove:
                st.session_state.watchlist.remove(to_remove)
                st.rerun()

            st.markdown("""
<style>
.wl-row + div [data-testid="stButton"] > button {
  background: rgba(255,255,255,0.04) !important; color: rgba(255,255,255,0.35) !important;
  border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 6px !important;
  font-size: 0.72rem !important; padding: 0.3rem 0.5rem !important;
  min-height: 0 !important; height: auto !important; line-height: 1 !important;
  letter-spacing: 0 !important; text-transform: none !important;
  transition: background 0.18s, color 0.18s !important;
}
.wl-row + div [data-testid="stButton"] > button:hover {
  background: rgba(255,255,255,0.1) !important; color: rgba(255,255,255,0.75) !important;
  border-color: rgba(255,255,255,0.18) !important;
  transform: none !important; box-shadow: none !important;
}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NEWS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "News":
    page_title("Market News", "Finnhub · ET Markets · Moneycontrol · Reuters · Yahoo Finance")

    lc, rc = st.columns([1, 3], gap="large")
    with lc:
        sec("Filters")
        srch = st.text_input("Search headlines", placeholder="e.g. Nifty, RBI, earnings…",
                             key="news_srch", label_visibility="collapsed")
        if st.button("Refresh →", use_container_width=True, key="news_refresh"):
            fetch_news.clear(); st.rerun()

    with rc:
        with st.spinner("Loading news…"):
            nl = fetch_news(FINNHUB_KEY)
        if not isinstance(nl, list): nl = []
        if srch:
            nl = [n for n in nl if srch.lower() in n.get('headline','').lower()
                                 or srch.lower() in n.get('summary','').lower()]
        if nl:
            st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
                        f'color:rgba(255,255,255,0.25);margin-bottom:1.2rem;">'
                        f'{min(len(nl),20)} articles</div>', unsafe_allow_html=True)
            for art in nl[:20]:
                hl  = art.get('headline','No Headline')
                src = art.get('source','Unknown')
                sm  = art.get('summary','')
                url = art.get('url','#')
                if not isinstance(url, str) or not url.startswith("http"): url = "#"
                summary_text = sm[:260] + ("…" if len(sm) > 260 else "")
                st.markdown(f"""
<div class="news-card">
  <div class="news-card-meta">{src}</div>
  <div class="news-card-headline">{hl}</div>
  <div class="news-card-summary">{summary_text}</div>
  <a href="{url}" target="_blank" class="news-card-link">Read article &#8594;</a>
</div>""", unsafe_allow_html=True)
        else:
            empty_state("◈", "No articles found — try refreshing")


# ── CURSOR GLOW ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.cursor-glow {
  width:300px; height:300px; border-radius:50%;
  background:radial-gradient(circle, rgba(255,255,255,0.025) 0%, transparent 70%);
  position:fixed; pointer-events:none; z-index:9998;
  transform:translate(-50%,-50%); transition:opacity 0.3s; will-change:transform;
}
</style>
<div class="cursor-glow" id="cglow"></div>
<script>
(function() {
  const glow = document.getElementById('cglow');
  if (glow) {
    document.addEventListener('mousemove', e => {
      glow.style.left = e.clientX + 'px';
      glow.style.top  = e.clientY + 'px';
    });
  }
  document.addEventListener('mouseover', e => {
    const tr = e.target.closest('tr');
    if (tr && tr.parentElement && tr.parentElement.tagName === 'TBODY')
      tr.style.background = 'rgba(255,255,255,0.04)';
  });
  document.addEventListener('mouseout', e => {
    const tr = e.target.closest('tr');
    if (tr && tr.parentElement && tr.parentElement.tagName === 'TBODY')
      tr.style.background = '';
  });
})();
</script>
""", unsafe_allow_html=True)