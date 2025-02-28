class Portfolio:
    def __init__(self):
        self.stocks = {}  # Dictionary to hold stock details by ticker symbol

    def add_stock(self, ticker, number_of_shares, purchase_price, purchase_date):
        """
        Add a stock to the portfolio.
        """
        if ticker in self.stocks:
            print(f"Stock {ticker} is already in the portfolio. Update its details instead.")
        else:
            self.stocks[ticker] = {
                "number_of_shares": number_of_shares,
                "purchase_price": purchase_price,
                "purchase_date": purchase_date,
                "current_price": None,  # Ensure current_price is initialized
            }
            print(f"Stock {ticker} added to portfolio.")

    def remove_stock(self, ticker):
        """
        Remove a stock from the portfolio.

        :param ticker: str, Stock ticker symbol to be removed.
        """
        if ticker in self.stocks:
            del self.stocks[ticker]
            print(f"Stock {ticker} removed from portfolio.")
        else:
            print(f"Stock {ticker} not found in the portfolio.")

    def update_current_price(self, ticker, current_price):
        """
        Update the current price of a stock.
        """
        if ticker in self.stocks:
            self.stocks[ticker]["current_price"] = current_price
            print(f"Updated current price for {ticker} to ${current_price:.2f}.")
        else:
            print(f"Stock {ticker} not found in portfolio.")

    def calculate_metrics(self):
        """
        Calculate and display portfolio metrics:
        - Total value
        - Individual stock profit/loss
        """
        total_value = 0
        print("\nPortfolio Metrics:")
        for ticker, details in self.stocks.items():
            if details["current_price"] is not None:
                value = details["number_of_shares"] * details["current_price"]
                profit_loss = value - (details["number_of_shares"] * details["purchase_price"])
                total_value += value
                print(f"{ticker}: Value = ${value:.2f}, Profit/Loss = ${profit_loss:.2f}")
            else:
                print(f"{ticker}: Current price not available.")
        print(f"\nTotal Portfolio Value: ${total_value:.2f}")