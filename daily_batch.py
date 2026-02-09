import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import os

class DailyPredictor:
    def __init__(self, ticker="^GSPC"):
        self.ticker = ticker
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def get_data(self):
        # On charge 2 ans pour avoir assez d'historique
        df = yf.download(self.ticker, period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def prepare_training_data(self, df):
        data = df.copy()
        
        # 1. Calcul des rendements réels
        data['Returns'] = data['Close'].pct_change()
        
        # 2. SHIFT pour l'entraînement (Training)
        # Sur la ligne T, on veut que les features soient celles de T-1, T-2...
        data['Feature_Lag1'] = data['Returns'].shift(1)
        data['Feature_Lag2'] = data['Returns'].shift(2)
        data['Feature_Vol'] = data['Returns'].rolling(5).std().shift(1)
        
        # On supprime les NaN créés par le shift
        data = data.dropna()
        return data

    def generate_prediction(self):
        print(f"📥 Récupération des données pour {self.ticker}...")
        df_raw = self.get_data()
        
        # --- ETAPE 1 : ENTRAÎNEMENT ---
        df_train = self.prepare_training_data(df_raw)
        
        X_train = df_train[['Feature_Lag1', 'Feature_Lag2', 'Feature_Vol']]
        y_train = df_train['Returns']
        
        self.model.fit(X_train, y_train)
        
        # --- ETAPE 2 : PRÉDICTION LIVE (Correction importante) ---
        # Pour prédire AUJOURD'HUI, on ne prend pas la dernière ligne shiftée de X_train (qui date d'avant-hier).
        # On doit construire le vecteur input avec les VRAIES données d'HIER soir.
        
        last_real_returns = df_raw['Close'].pct_change() # Recalcul sur données brutes
        
        # Les features pour "Demain" sont les réalités d'aujourd'hui
        input_lag1 = last_real_returns.iloc[-1]      # Rendement d'hier
        input_lag2 = last_real_returns.iloc[-2]      # Rendement d'avant-hier
        input_vol  = last_real_returns.tail(5).std() # Volatilité récente
        
        # On crée un petit DataFrame pour la prédiction (format attendu par sklearn)
        X_live = pd.DataFrame({
            'Feature_Lag1': [input_lag1],
            'Feature_Lag2': [input_lag2],
            'Feature_Vol':  [input_vol]
        })
        
        # Prédiction
        predicted_return = self.model.predict(X_live)[0]
        
        # --- INTERVALLE DE CONFIANCE (95%) ---
        # Basé sur les erreurs historiques du modèle
        train_preds = self.model.predict(X_train)
        residuals = y_train - train_preds
        std_error = residuals.std()
        margin_of_error = 1.96 * std_error
        
        # --- ETAPE 2 : PRÉDICTION LIVE ---
        # Récupération des données du jour pour le prix de référence
        ticker_live = yf.Ticker(self.ticker)
        todays_data = ticker_live.history(period="1d")

        if not todays_data.empty:
            # On prend l'ouverture (Open) d'aujourd'hui si dispo
            reference_price = todays_data['Open'].iloc[0]
        else:
            # Sinon on prend la clôture d'hier (Fallback)
            reference_price = df_raw['Close'].iloc[-1]
            
        # ... (ton code précédent) ...
        if not todays_data.empty:
            reference_price = todays_data['Open'].iloc[0]
            source = "OPEN DU JOUR (Live)"
        else:
            reference_price = df_raw['Close'].iloc[-1]
            source = "CLOSE DE LA VEILLE (Marché fermé)"
        
        # --- LE MOUCHARD ---
        print(f"------------------------------------------------")
        print(f"🔍 DEBUG: Source utilisée : {source}")
        print(f"💰 PRIX DE RÉFÉRENCE : {reference_price}")
        print(f"------------------------------------------------")
        # -------------------

    
        # Calcul des prix cibles basés sur ce reference_price
        predicted_price = reference_price * (1 + predicted_return)
        lower_price = reference_price * (1 + (predicted_return - margin_of_error))
        upper_price = reference_price * (1 + (predicted_return + margin_of_error))
        
        
        # Signal
        signal = "⚪ ATTENTE (NEUTRAL)"
        if predicted_return > 0.001: 
            signal = "🟢 ACHAT (BUY)"
        elif predicted_return < -0.001:
            signal = "🔴 VENTE (SELL)"
            
        return {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_price": last_closing_price,
            "predicted_return": predicted_return,
            "predicted_price": predicted_price,
            "signal": signal,
            "lower": lower_price,
            "upper": upper_price,
            "confidence": f"{margin_of_error*100:.2f}%"
        }

    def save_report(self, data):
        # On écrase le fichier pour n'avoir toujours que le dernier rapport
        filename = f"Rapport_Prediction_LATEST.txt"
        # On garde aussi un historique si besoin
        history_filename = f"Rapport_Prediction_{datetime.now().strftime('%Y%m%d')}.txt"
        
        content = f"""
==================================================
   QUEENSFIELD AI - MORNING INTELLIGENCE
   Generated: {data['date']}
==================================================

ASSET: S&P 500 (^GSPC)
STATUS: PRE-MARKET ANALYSIS

1. AI SIGNAL
------------
>>> {data['signal']} <<<

Forecast Return: {data['predicted_return']:.2%}

2. PRICE TARGETS (Next Close)
-----------------------------
Previous Close : ${data['last_price']:.2f}
Target Price   : ${data['predicted_price']:.2f}

3. CONFIDENCE INTERVAL (95%)
----------------------------
Bearish Case (Low)  : ${data['lower']:.2f}
Bullish Case (High) : ${data['upper']:.2f}

Model Volatility : +/- {data['confidence']}

==================================================
CONFIDENTIAL - QUEENSFIELD AI TECHNOLOGIES
"""
        # Sauvegarde du fichier "LATEST" pour Streamlit
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
            
        # Sauvegarde de l'historique
        with open(history_filename, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ Rapport généré : {filename}")
        print(content)

if __name__ == "__main__":
    bot = DailyPredictor()
    pred_data = bot.generate_prediction()
    bot.save_report(pred_data)