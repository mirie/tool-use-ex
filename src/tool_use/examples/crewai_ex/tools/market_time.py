import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from crewai.tools import tool

logger = logging.getLogger(__name__)


@tool("Get Current Time")
def get_current_time(timezone: str) -> str:
    """Get the current local date and time in an IANA timezone such as
    'America/New_York' or 'Europe/London'. Use this whenever a question depends on
    the current time, e.g. whether a stock exchange is open right now."""
    logger.info("Tool called: get_current_time(timezone=%r)", timezone)
    try:
        now = datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError:
        raise ValueError(
            f"Unknown timezone '{timezone}'. Use an IANA name like 'America/New_York'."
        ) from None
    result = now.strftime("%A %Y-%m-%d %H:%M %Z")
    logger.info("Returning time %s for %s", result, timezone)
    return result
