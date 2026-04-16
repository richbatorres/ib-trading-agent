"""Agent configuration loaded from .env file."""

import logging
import sys
from dataclasses import dataclass

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("IB_ACCOUNT_ID", "IB_HOST", "IB_PORT", "ENVIRONMENT")
_VALID_ENVIRONMENTS = ("paper", "live")


@dataclass
class AgentConfig:
    """All configuration from .env file."""

    ib_account_id: str
    ib_host: str
    ib_port: int
    environment: str
    email_address: str
    email_smtp_host: str
    email_smtp_port: int
    email_smtp_user: str
    email_smtp_password: str
    max_portfolio_loss_pct: float = 20.0
    max_position_size_pct: float = 25.0
    stop_loss_pct: float = 5.0
    cash_buffer_pct: float = 10.0
    db_url: str = "sqlite:///data/agent.db"
    llm_model: str = "claude-sonnet-4-6"
    max_llm_calls_per_day: int = 10
    market_data_type: int = 4  # 1=real-time, 3=delayed, 4=frozen delayed

    @classmethod
    def from_env(cls, path: str = ".env") -> "AgentConfig":
        """Load configuration from a .env file.

        Validates required fields and environment value.
        Exits with code 1 if validation fails.
        """
        values = dotenv_values(path)

        # Validate required fields
        missing = [f for f in _REQUIRED_FIELDS if not values.get(f)]
        if missing:
            for field in missing:
                logger.error("Missing required configuration: %s", field)
            sys.exit(1)

        # Validate ENVIRONMENT value
        environment = values["ENVIRONMENT"]
        if environment not in _VALID_ENVIRONMENTS:
            logger.error(
                "Invalid ENVIRONMENT value '%s'. Must be 'paper' or 'live'.",
                environment,
            )
            sys.exit(1)

        return cls(
            ib_account_id=values["IB_ACCOUNT_ID"],
            ib_host=values["IB_HOST"],
            ib_port=int(values["IB_PORT"]),
            environment=environment,
            email_address=values.get("EMAIL_ADDRESS", ""),
            email_smtp_host=values.get("EMAIL_SMTP_HOST", ""),
            email_smtp_port=int(values.get("EMAIL_SMTP_PORT", "587")),
            email_smtp_user=values.get("EMAIL_SMTP_USER", ""),
            email_smtp_password=values.get("EMAIL_SMTP_PASSWORD", ""),
            max_portfolio_loss_pct=float(values.get("MAX_PORTFOLIO_LOSS_PCT", "20.0")),
            max_position_size_pct=float(values.get("MAX_POSITION_SIZE_PCT", "25.0")),
            stop_loss_pct=float(values.get("STOP_LOSS_PCT", "5.0")),
            cash_buffer_pct=float(values.get("CASH_BUFFER_PCT", "10.0")),
            db_url=values.get("DB_URL", "sqlite:///data/agent.db"),
            llm_model=values.get("LLM_MODEL", "claude-sonnet-4-6"),
            max_llm_calls_per_day=int(values.get("MAX_LLM_CALLS_PER_DAY", "10")),
            market_data_type=int(values.get("MARKET_DATA_TYPE", "4")),
        )
