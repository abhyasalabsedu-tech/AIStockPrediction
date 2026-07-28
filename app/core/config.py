from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    database_url: str = "postgresql://postgres:password@localhost:5432/stockxai"

    default_ticker: str = "TCS.NS"
    market_data_provider: str = "yfinance"

    env: str = "development"
    cors_origins: str = "http://localhost:3000"
    eval_cron_hour: int = 16
    eval_cron_minute: int = 30

    class Config:
        env_file = ".env"

    @property
    def cors_origin_list(self):
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
