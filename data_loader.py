import yfinance as yf
import pandas as pd

class SP500DataLoader:
    """
    Class responsible for fetching and cleaning S&P 500 data directly from Yahoo Finance.
    """
    
    def __init__(self, ticker: str = "^GSPC"):
        """
        Initializes the loader with a specific ticker. 
        Default is ^GSPC for the S&P 500 Index.
        """
        self.ticker = ticker
        self.data = pd.DataFrame()

    def fetch_data(self, period: str = "10y") -> pd.DataFrame:
        """
        Downloads OHLCV data (Open, High, Low, Close, Volume).
        
        Args:
            period (str): The time period to download (e.g., '1y', '5y', '10y', 'max').
            
        Returns:
            pd.DataFrame: Cleaned dataframe with OHLCV data.
        """
        print(f"Fetching data for {self.ticker}...")
        
        # Fetch data using yfinance
        df = yf.download(self.ticker, period=period, interval="1d", progress=False)

        # Fix MultiIndex columns (common issue with yfinance updates)
        # If columns have levels (e.g., Price, Ticker), we drop the Ticker level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure the index is in datetime format
        df.index = pd.to_datetime(df.index)
        
        # Store in the instance
        self.data = df
        
        # Basic validation
        if df.empty:
            print("Warning: Data download failed or returned empty.")
        else:
            print(f"Successfully loaded {len(df)} rows.")

        return self.data

# This block allows you to test this file independently without running Streamlit
if __name__ == "__main__":
    loader = SP500DataLoader()
    df = loader.fetch_data()
    print(df.head())