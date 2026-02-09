import pandas as pd
import numpy as np

class StrategyEngine:
    """
    Moteur de calcul complet : 
    - Génération de Signaux (MA / Momentum)
    - Backtest avec Gestion du Risque (Stop Loss / Take Profit)
    - Projection Monte Carlo (AI Forecast)
    - Simulation P&L avec Frais de transaction réels
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.backtest_data = None # Stocke les résultats pour éviter de recalculer

    def _generate_raw_signals(self, strategy_type, **params):
        """
        Génère les signaux bruts (1 = Achat, 0 = Cash) sans gestion du risque.
        """
        df = self.data.copy()
        df['Signal'] = 0 

        if strategy_type == "MA":
            short_window = params.get('short_window', 50)
            long_window = params.get('long_window', 200)
            df['SMA_Short'] = df['Close'].rolling(window=short_window).mean()
            df['SMA_Long'] = df['Close'].rolling(window=long_window).mean()
            # Signal = 1 si la moyenne courte est au-dessus de la longue
            df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)

        elif strategy_type == "Momentum":
            period = params.get('period', 14)
            df['Momentum'] = df['Close'].pct_change(periods=period)
            # Signal = 1 si le momentum est positif
            df['Signal'] = np.where(df['Momentum'] > 0, 1, 0)
            
        return df

    def backtest(self, strategy_type, stop_loss_pct=0.0, take_profit_pct=0.0, **params):
        """
        Exécute le backtest en boucle (Event-Driven) pour gérer le risque jour par jour.
        """
        df = self._generate_raw_signals(strategy_type, **params)
        
        positions = [] 
        current_position = 0
        current_entry_price = 0.0
        
        prices = df['Close'].values
        raw_signals = df['Signal'].values 
        
        # Boucle jour par jour
        for i in range(len(df)):
            price = prices[i]
            signal = raw_signals[i]
            
            if current_position == 0:
                # Si on est CASH, on regarde si on doit acheter
                if signal == 1:
                    current_position = 1
                    current_entry_price = price
            
            elif current_position == 1:
                # Si on est INVESTI, on vérifie les sorties
                
                # 1. Stop Loss (Protection)
                if stop_loss_pct > 0 and price < current_entry_price * (1 - stop_loss_pct):
                    current_position = 0 
                
                # 2. Take Profit (Prise de gains)
                elif take_profit_pct > 0 and price > current_entry_price * (1 + take_profit_pct):
                    current_position = 0 
                
                # 3. Sortie Stratégie (Le signal devient rouge)
                elif signal == 0:
                    current_position = 0 
            
            positions.append(current_position)

        df['Position'] = positions
        # Shift important : La décision prise à la clôture de J s'applique au rendement de J+1
        df['Position'] = df['Position'].shift(1).fillna(0)
        
        # Calcul des rendements
        df['Market_Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Position'] * df['Market_Returns']
        
        # Calcul des courbes d'équité (Cumulative)
        df['Cumulative_Market'] = (1 + df['Market_Returns']).cumprod() * self.initial_capital
        df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod() * self.initial_capital
        
        self.backtest_data = df # On sauvegarde pour les modules suivants
        
        return df

    def project_future_performance(self, days=60):
        """
        Module AI : Projette un cône de probabilité basé sur la volatilité historique récente.
        """
        if self.backtest_data is None:
            return None

        # On utilise les 180 derniers jours pour calibrer le modèle
        recent_data = self.backtest_data.tail(180)
        
        mu = recent_data['Strategy_Returns'].mean() # Drift (Tendance moyenne)
        sigma = recent_data['Strategy_Returns'].std() # Volatilité (Risque)
        
        last_val = self.backtest_data['Cumulative_Strategy'].iloc[-1]
        last_date = self.backtest_data.index[-1]
        
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days + 1)]
        
        expected_values = []
        upper_values = []
        lower_values = []
        
        # Projection Mathématique (Mouvement Brownien Géométrique simplifié)
        for t in range(1, days + 1):
            drift = (1 + mu) ** t
            volatility_component = sigma * np.sqrt(t) * 1.96 # Intervalle de confiance 95%
            
            mean_val = last_val * drift
            upper_val = mean_val * (1 + volatility_component)
            lower_val = mean_val * (1 - volatility_component)
            
            expected_values.append(mean_val)
            upper_values.append(upper_val)
            lower_values.append(lower_val)
            
        return pd.DataFrame({
            'Expected': expected_values,
            'Upper_95': upper_values,
            'Lower_95': lower_values
        }, index=future_dates)

    def simulate_custom_pnl(self, start_date, investment_amount, tx_cost_pct=0.001):
        """
        Simulateur P&L Réaliste.
        Prend en compte une date de début spécifique ET les frais de transaction.
        
        Args:
            start_date (str/datetime): Date de début de l'investissement.
            investment_amount (float): Montant investi (ex: 1000€).
            tx_cost_pct (float): Frais par ordre (ex: 0.001 pour 0.1%).
            
        Returns:
            final_value, net_pnl, total_fees_paid
        """
        if self.backtest_data is None:
            return investment_amount, 0.0, 0.0
            
        # Conversion et filtrage des données
        start_date = pd.to_datetime(start_date)
        subset = self.backtest_data[self.backtest_data.index >= start_date].copy()
        
        if subset.empty:
            return investment_amount, 0.0, 0.0
            
        # 1. Identification des Transactions (Achat ou Vente)
        # On regarde la différence de position d'un jour à l'autre
        # abs(diff) car passer de 1 à 0 ou 0 à 1 coûte des frais pareil
        trades = subset['Position'].diff().fillna(0).abs()
        
        # Cas particulier : Si on démarre la simulation alors que la stratégie est DÉJÀ en position (1),
        # on doit payer les frais d'entrée immédiatement.
        if subset['Position'].iloc[0] != 0:
            trades.iloc[0] = 1.0
            
        # 2. Calcul du Rendement Net (Rendement Brut - Frais)
        # Chaque jour où il y a un trade, on soustrait le % de frais
        net_returns = subset['Strategy_Returns'] - (trades * tx_cost_pct)
        
        # 3. Croissance du capital (Intérêts composés sur rendement net)
        cumulative_growth = (1 + net_returns).cumprod()
        
        final_value = investment_amount * cumulative_growth.iloc[-1]
        net_pnl = final_value - investment_amount
        
        # 4. Calcul des Frais Totaux (Pour l'affichage)
        # On compare la courbe Brute vs la courbe Nette
        gross_growth = (1 + subset['Strategy_Returns']).cumprod()
        gross_final_value = investment_amount * gross_growth.iloc[-1]
        
        total_fees = gross_final_value - final_value
        
        return final_value, net_pnl, total_fees