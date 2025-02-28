# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import yfinance as yf
from Stock import Portfolio



Finance_Stocks ={}          # Global dictionary of 50 companies

def Extract_Data():
    # Define a list of stock tickers

    global Finance_Stocks

    # Predefined list of stock tickers
    tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "META", "NFLX", "NVDA"]

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                # Format the current price as a string with a dollar sign
                current_price = f"${data['Close'][-1]:,.2f}"
                Finance_Stocks[ticker] = {
                    "name": stock.info.get("shortName", "N/A"),
                    "current_price": current_price,
                    "sector": stock.info.get("sector", "N/A")
                }
                print(f"Added {ticker}: {Finance_Stocks[ticker]}")
            else:
                print(f"No data for {ticker}. Skipping.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")


if __name__ == '__main__':



   Extract_Data()
   # Print all entries in Finance_Stocks
   print("\nFinance_Stocks Dictionary:")
   for ticker, details in Finance_Stocks.items():
       print(f"{ticker}: {details}")

   '''
      portfolio = Portfolio()

      # Add stocks
      portfolio.add_stock("AAPL", 10, 150.0, "2024-12-01")
      portfolio.add_stock("GOOGL", 5, 2800.0, "2024-12-01")
      portfolio.add_stock("TSLA", 4, 3000, "2024-11-30")
      portfolio.add_stock("MSFT", 7, 700, "2024-09-25")

      # Print portfolio details
      print("Portfolio Details:")
      for ticker, details in portfolio.stocks.items():
          print(f"{ticker}: {details}")

      # Remove a stock
      portfolio.remove_stock("GOOGL")

      # Update current prices
      portfolio.update_current_price("AAPL", 160.0)
      portfolio.update_current_price("TSLA", 3100.0)
      portfolio.update_current_price("MSFT", 750.0)

      # Display updated portfolio
      print("\nPortfolio Details after updates:")
      for ticker, details in portfolio.stocks.items():
          print(f"{ticker}: {details}")

      # Calculate metrics
      portfolio.calculate_metrics()
  '''