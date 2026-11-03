import logging

from crewai.tools import tool

logger = logging.getLogger(__name__)


@tool("Lookup Stock Price")
def get_stock_price(ticker: str) -> float:
    """Look up the latest stock price for a ticker symbol (e.g. 'AAPL').
    Use this whenever a stock price is needed. Returns the price as a float."""
    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
        "AMZN": 185.20,
        "NVDA": 120.75,
    }

    logger.info("Tool called: get_stock_price(ticker=%r)", ticker)
    price = simulated_prices.get(ticker.upper())
    if price is None:
        logger.warning("No simulated price found for ticker %r", ticker.upper())
        raise ValueError(f"Simulated price for ticker '{ticker.upper()}' not found.")
    logger.info("Returning price %.2f for %s", price, ticker.upper())
    return price
