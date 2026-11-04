import logging

from crewai.tools import tool

logger = logging.getLogger(__name__)


@tool("Get Portfolio Holdings")
def get_portfolio_holdings() -> dict[str, int]:
    """Get the user's private portfolio: the number of shares they hold per ticker.
    Use this whenever a question mentions the user's own holdings, position or
    portfolio. Returns a mapping like {'MSFT': 10}. Prices are not included."""
    # Private data: the model can't know this from its training, so it must use the tool
    holdings = {"MSFT": 12, "NVDA": 40, "AAPL": 25}
    logger.info("Tool called: get_portfolio_holdings() -> %s", holdings)
    return holdings
