import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
import os
from datetime import datetime, timedelta
from data_loader import SP500DataLoader 
from strategies import StrategyEngine
import streamlit as st
import os
from datetime import datetime, date
import time

from daily_batch import DailyPredictor

REPORT_FILE = "Rapport_Prediction_LATEST.txt"

def check_and_update_system():
    """
    Cette fonction vérifie si le rapport est à jour.
    S'il est vieux (date != aujourd'hui), elle lance le calcul IA automatiquement.
    """
    should_update = False
    
    # Cas 1 : Le fichier n'existe pas
    if not os.path.exists(REPORT_FILE):
        should_update = True
    else:
        # Cas 2 : Le fichier existe, on regarde sa date de modification
        file_timestamp = os.path.getmtime(REPORT_FILE)
        file_date = datetime.fromtimestamp(file_timestamp).date()
        today = date.today()
        
        # Si le fichier n'est pas d'aujourd'hui, on doit mettre à jour
        if file_date < today:
            # OPTIONNEL : On peut éviter de mettre à jour le week-end si on veut
            # if today.weekday() < 5: (0=Lundi, 4=Vendredi)
            should_update = True

    if should_update:
        # Affichage d'un spinner pendant que l'IA travaille (ça fait pro !)
        with st.spinner('🤖 Démarrage du système autonome... Récupération des données de marché & Calcul des prédictions...'):
            try:
                # --- C'est ici que la magie opère tout seul ---
                bot = DailyPredictor()
                pred_data = bot.generate_prediction()
                bot.save_report(pred_data)
                
                st.success(f"✅ Système mis à jour avec succès ! (Données du {pred_data['date']})")
                time.sleep(1) # Petite pause pour que l'utilisateur voie le message
                st.rerun() # On recharge la page pour afficher les nouvelles données
            except Exception as e:
                st.error(f"⚠️ Échec de la mise à jour automatique : {e}")

# --- PLACE CETTE FONCTION TOUT EN HAUT DE TON CODE ---
check_and_update_system()




# --- CONFIGURATION ---





st.set_page_config(page_title="QueensField AI", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
    .stApp { background-color: #0b0c0e; }
    [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #222; }
    
    /* INPUTS */
    .stSelectbox > div > div, .stNumberInput > div > div, .stTextInput > div > div, .stDateInput > div > div {
        background-color: #13151a !important; border: 1px solid #333 !important; color: white !important;
    }
    
    /* BOUTONS */
    div.stButton > button {
        background-color: #E65100; color: white; border: none; font-weight: 600; height: 42px; text-transform: uppercase; width: 100%;
    }
    div.stButton > button:hover { background-color: #ff5722; box-shadow: 0 0 10px rgba(230, 81, 0, 0.5); }
    
    /* BOUTON SIMULATION (VERT) */
    .sim-btn > button {
        background-color: #00E676 !important;
        color: black !important;
    }
    .sim-btn > button:hover {
        background-color: #00C853 !important;
        box-shadow: 0 0 10px rgba(0, 230, 118, 0.5);
    }

    /* KPIS */
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #161920 0%, #0e1117 100%);
        border: 1px solid #2a2d36; padding: 15px; border-radius: 6px;
    }
    h1, h2, h3 { color: #E65100 !important; font-family: 'Helvetica Neue', sans-serif; }
    
    /* REPORT TERMINAL STYLE */
    .report-box { background-color: #000; border: 1px solid #333; border-left: 5px solid #E65100; padding: 15px; font-family: 'Courier New', monospace; color: #0f0; font-size: 14px; margin-bottom: 20px; }
    
    /* P&L RESULT BOX STYLE */
    .pnl-box {
        background-color: #000;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
    }
    .pnl-text { font-family: 'Roboto', sans-serif; color: #888; font-size: 14px; }
    .pnl-val { font-family: 'Roboto Mono', monospace; color: white; font-size: 22px; font-weight: bold; }
    .pnl-gain { color: #00E676; }
    .pnl-loss { color: #FF1744; }
    </style>
    """, unsafe_allow_html=True)

if 'tickers' not in st.session_state: st.session_state.tickers = ["^GSPC"]
if 'engine' not in st.session_state: st.session_state.engine = None # Pour garder le backtest en mémoire

def get_latest_report():
    list_of_files = glob.glob('Rapport_Prediction_*.txt')
    if not list_of_files: return None
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'r', encoding='utf-8') as f: return f.read()

def create_chart(df, color="#E65100"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color=color, width=2), fill='tozeroy', fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)"))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=True, gridcolor='#222', side='right'))
    return fig



# --- HEADER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("QUEENSFIELD AI TECHNOLOGIES")
    st.caption("ADVANCED MARKET INTELLIGENCE SYSTEM")
with c2:
    st.write("")
    if st.button("📥 EXPORT DATA"): st.toast("Exporting data...")

# --- TABS ---
tab1, tab2 = st.tabs(["MARKET OVERVIEW", "STRATEGY LAB"])

with tab1:
    st.markdown("##### 📡 DAILY INTELLIGENCE REPORT (7:30 AM)")
    report_content = get_latest_report()
    
    if report_content:
        # --- 1. ANALYSE DU TEXTE (PARSING) ---
        lines = report_content.split('\n')
        
        # Valeurs par défaut
        signal = "NEUTRAL"
        pred_return = "0.00%"
        target_price = "0.00"
        prev_close = "0.00"
        low_price = "N/A"
        high_price = "N/A"
        
        # On cherche les infos ligne par ligne
        for line in lines:
            if ">>>" in line: signal = line.replace(">>>", "").replace("<<<", "").strip()
            if "Forecast Return:" in line: pred_return = line.split(":")[-1].strip()
            if "Target Price" in line: target_price = line.split("$")[-1].strip()
            if "Previous Close" in line: prev_close = line.split("$")[-1].strip()
            if "Bearish Case" in line: low_price = line.split("$")[-1].strip()
            if "Bullish Case" in line: high_price = line.split("$")[-1].strip()

        # --- 2. COULEURS ET STYLE ---
        if "VENTE" in signal or "SELL" in signal:
            box_color = "rgba(255, 75, 75, 0.2)" # Rouge transparent
            border_color = "#FF4B4B"
            text_color = "#FF4B4B"
        elif "ACHAT" in signal or "BUY" in signal:
            box_color = "rgba(0, 200, 100, 0.2)" # Vert transparent
            border_color = "#00C864"
            text_color = "#00C864"
        else:
            box_color = "rgba(128, 128, 128, 0.2)"
            border_color = "#888"
            text_color = "#AAA"

        # --- 3. AFFICHAGE VISUEL (DASHBOARD) ---
        col_signal, col_kpi = st.columns([1.5, 3])
        
        with col_signal:
            # Boîte Signal Style "Badge"
            st.markdown(f"""
            <div style="background-color: {box_color}; border: 2px solid {border_color}; border-radius: 10px; padding: 15px; text-align: center;">
                <h4 style="margin:0; color:white; font-size:14px; opacity:0.8; letter-spacing: 1px;">AI SIGNAL</h4>
                <h2 style="margin:5px 0; color:{text_color}; font-size:32px; font-weight:bold;">{signal}</h2>
                <div style="font-size:12px; color:#ddd; border-top:1px solid {border_color}; margin-top:5px; padding-top:5px;">CONFIDENCE: HIGH</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_kpi:
            # KPIs financiers
            k1, k2, k3 = st.columns(3)
            k1.metric("TARGET PRICE", f"${target_price}", delta=pred_return)
            k2.metric("PREV CLOSE", f"${prev_close}")
            k3.metric("RISK RANGE", f"${low_price}", delta_color="off", help=f"Low: {low_price} / High: {high_price}")
            
            # Barre de progression visuelle pour le range (Optionnel mais joli)
            st.caption(f"📉 Bearish: **${low_price}** ....................... 🎯 Target ....................... 📈 Bullish: **${high_price}**")

        # Bouton discret pour voir les logs bruts
        with st.expander("🛠️ VIEW RAW SYSTEM LOGS"):
            st.code(report_content, language="text")

    else:
        st.info("⚠️ No prediction report found. System requires update.")
    
    c_add, c_space, c_time = st.columns([2, 3, 2])
    with c_add:
        new_ticker = st.text_input("WATCHLIST ADD TICKER...", placeholder="Ex: AAPL").upper()
        if new_ticker:
            if st.button(f"ADD {new_ticker}"):
                if new_ticker not in st.session_state.tickers:
                    st.session_state.tickers.append(new_ticker)
                    st.rerun()
    with c_time:
        time_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}
        sel = st.radio("TIMEFRAME", list(time_map.keys()), index=3, horizontal=True, label_visibility="collapsed")
        period_yf = time_map[sel]

    st.markdown("---")
    cols = st.columns(2)
    for i, ticker in enumerate(st.session_state.tickers):
        with cols[i % 2]:
            with st.container(border=True):
                h1, h2 = st.columns([4, 1])
                loader = SP500DataLoader(ticker=ticker)
                df = loader.fetch_data(period=period_yf)
                if not df.empty:
                    last = df['Close'].iloc[-1]
                    chg = df['Close'].pct_change().iloc[-1]
                    col_txt = "#00FF00" if chg >= 0 else "#FF0000"
                    name = "S&P 500" if ticker == "^GSPC" else ticker
                    h1.markdown(f"**{name}**")
                    h1.markdown(f"<span style='color:{col_txt}; font-size:20px'>${last:,.2f} ({chg:+.2%})</span>", unsafe_allow_html=True)
                    if ticker != "^GSPC" and st.button("✕", key=f"d{ticker}"):
                        st.session_state.tickers.remove(ticker)
                        st.rerun()
                    st.plotly_chart(create_chart(df), use_container_width=True)

with tab2:
    st.markdown("### ADVANCED STRATEGIES")
    c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1.2, 1, 1.5])
    with c1: st_type = st.selectbox("ALGORITHM", ["Moving Average (MA)", "Momentum"], label_visibility="collapsed")
    
    params = {}
    if st_type == "Moving Average (MA)":
        code = "MA"
        with c2: params['short_window'] = st.number_input("SHORT MA", value=20, label_visibility="collapsed")
        with c3: params['long_window'] = st.number_input("LONG MA", value=50, label_visibility="collapsed")
    else:
        code = "Momentum"
        with c2: params['period'] = st.number_input("PERIOD", value=14, label_visibility="collapsed")
        with c3: st.write("")

    with c4: enable_ai = st.checkbox("ENABLE AI FORECAST", value=True)
    with c5: bt_per = st.selectbox("PERIOD", ["1y", "2y", "5y", "10y", "max"], index=2, label_visibility="collapsed")
    with c6: run = st.button("EXECUTE", use_container_width=True)

    st.write("---")

    if run:
        with st.spinner('Computing Quantitative Model...'):
            loader = SP500DataLoader(ticker="^GSPC")
            df_s = loader.fetch_data(period=bt_per)
            if not df_s.empty:
                eng = StrategyEngine(df_s, initial_capital=100000)
                res = eng.backtest(strategy_type=code, **params)
                st.session_state.engine = eng # Sauvegarde pour la P&L Sim
                st.session_state.backtest_res = res
            else: st.error("No data.")

    # Affichage des résultats si le backtest existe (soit vient de tourner, soit en mémoire)
    if st.session_state.engine is not None:
        res = st.session_state.backtest_res
        eng = st.session_state.engine
        
        # KPIs
        ret = (res['Cumulative_Strategy'].iloc[-1] / 100000) - 1
        vol = res['Strategy_Returns'].std() * (252 ** 0.5)
        sharpe = (res['Strategy_Returns'].mean() / res['Strategy_Returns'].std()) * (252 ** 0.5)
        dd = ((res['Cumulative_Strategy'] - res['Cumulative_Strategy'].cummax()) / res['Cumulative_Strategy'].cummax()).min()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("TOTAL RETURN", f"{ret:.2%}")
        k2.metric("SHARPE RATIO", f"{sharpe:.2f}")
        k3.metric("VOLATILITY", f"{vol:.2%}")
        k4.metric("MAX DRAWDOWN", f"{dd:.2%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res.index, y=res['Cumulative_Strategy'], name='Strategy', line=dict(color='#00FF00', width=2)))
        fig.add_trace(go.Scatter(x=res.index, y=res['Cumulative_Market'], name='Benchmark', line=dict(color='#444', width=2, dash='dash')))

        if enable_ai:
            future_df = eng.project_future_performance(days=60)
            if future_df is not None:
                fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Upper_95'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Lower_95'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 0, 0.15)', name='95% Confidence Band'))
                fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Expected'], mode='lines', line=dict(color='#00FF00', width=1, dash='dot'), name='Expected AI Path'))

        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), hovermode="x unified", xaxis=dict(showgrid=False, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#222'))
        st.plotly_chart(fig, use_container_width=True)
        
       # --- MODULE P&L SIMULATION (AVEC FRAIS RÉELS) ---
        st.write("---")
        st.markdown("### 💰 REALISTIC P&L SIMULATION")
        
        with st.container(border=True):
            # On divise en 4 colonnes pour ajouter le champ "Fees"
            s1, s2, s3, s4, s5 = st.columns([2, 1.5, 1.5, 1.5, 4])
            
            data_start_date = res.index[0].to_pydatetime()
            
            with s1:
                default_date = max(datetime.now() - timedelta(days=365), data_start_date)
                sim_date = st.date_input("START DATE", value=default_date)
            
            with s2:
                sim_amount = st.number_input("CAPITAL (€)", value=1000, step=100)
                
            with s3:
                # Nouveau champ pour les frais (ex: 0.1% par défaut)
                sim_fees = st.number_input("FEES (%)", value=0.10, step=0.01, min_value=0.0, format="%.2f")
                # On convertit le % en décimal (ex: 0.1 -> 0.001)
                fee_decimal = sim_fees / 100

            with s4:
                st.write("") 
                st.write("")
                sim_run = st.button("CALCULATE", key="sim_btn", use_container_width=True)

            # Initialisation
            final_val = sim_amount
            pnl = 0
            fees_paid = 0
            
            if sim_run:
                # Vérification date
                sim_datetime = pd.to_datetime(sim_date).replace(tzinfo=None)
                data_start_naive = pd.to_datetime(data_start_date).replace(tzinfo=None)

                if sim_datetime < data_start_naive:
                    st.toast(f"⚠️ Simulation adjusted to start of data: {data_start_date.strftime('%Y-%m-%d')}", icon="⚠️")
                
                # Appel du nouveau moteur avec frais
                final_val, pnl, fees_paid = eng.simulate_custom_pnl(sim_date, sim_amount, tx_cost_pct=fee_decimal)
            
            # Affichage Résultat avec Ligne "Fees"
            pnl_color = "#00E676" if pnl >= 0 else "#FF1744" # Vert ou Rouge
            pnl_sign = "+" if pnl >= 0 else ""
            
            
        with s5:
            # Tout ce bloc est décalé vers la droite (une tabulation)
            st.markdown("### Résultat") 
            c1, c2 = st.columns(2)
            c1.metric("INVESTED", f"€{sim_amount:,.2f}")
            c2.metric("FINAL (NET)", f"€{final_val:,.2f}")

            st.markdown("---")

            c3, c4 = st.columns(2)
            c3.metric("FEES PAID", f"-€{fees_paid:,.2f}")
            c4.metric("NET PROFIT", f"€{pnl:,.2f}", delta=f"{pnl:,.2f} €")