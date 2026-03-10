from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List, Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "ABCat Shop API"
    # default เป็น 127.0.0.1 (localhost only) — ปลอดภัยกว่า 0.0.0.0
    # ถ้า deploy ใน Docker/production ให้ set APP_HOST=0.0.0.0 ใน .env แทน
    APP_HOST: str = "127.0.0.1"  # nosec B104
    APP_PORT: int = 8000
    DEBUG: bool = False

    # Database
    DATABASE_URL: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None

    # Cloudinary
    CLOUDINARY_CLOUD_NAME: Optional[str] = None
    CLOUDINARY_API_KEY: Optional[str] = None
    CLOUDINARY_API_SECRET: Optional[str] = None

    # Firebase
    FIREBASE_SERVICE_ACCOUNT_KEY: str = "app/serviceAccountKey.json"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://your-app.vercel.app",
    ]

    # Upload
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL

        if self.POSTGRES_USER and self.POSTGRES_PASSWORD and self.POSTGRES_DB:
            return (
                f"postgresql://{self.POSTGRES_USER}:"
                f"{self.POSTGRES_PASSWORD}@postgres:5432/"
                f"{self.POSTGRES_DB}"
            )

        raise RuntimeError("DATABASE_URL or POSTGRES_* variables required")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()