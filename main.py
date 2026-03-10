import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.callback_flutter import router as callback_router
from app.api.search_flutter import router as search_router
from app.api.vision import router as vision_router
from app.api.cat_crud_api import router as cat_crud_router
from app.auth.login import router as login_router
from app.auth.register import router as sign_up_router
from app.db.database import create_db_pool, close_db_pool
from app.core.firebase import init_firebase
from app.api.api_favourite import router as api_favourite
from app.api.api_basket import router as api_basket
from app.api.detect_api import router as detect_router
from app.api.recommen_api import router as recommen_router
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    # --- Startup ---
    try:
        await create_db_pool()
    except Exception as e:
        logger.warning("⚠️ Database not ready, continue without DB: %s", e)

    try:
        init_firebase()
        logger.info("✅ Firebase initialized")
    except Exception as e:
        logger.warning("⚠️ Firebase skipped: %s", e)

    logger.info("🚀 App startup complete")
    yield

    # --- Shutdown ---
    try:
        await close_db_pool()
        logger.info("🧹 Database pool closed")
    except Exception as e:                          # nosec B110
        logger.warning("⚠️ Failed to close database pool: %s", e)

    logger.info("🧹 App shutdown complete")


app = FastAPI(
    title="ABC SHOP API",
    version="1.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(callback_router, prefix="/api", tags=["Callback"])
app.include_router(search_router,   prefix="/api", tags=["Search"])
app.include_router(login_router,    prefix="/api", tags=["LogIn"])
app.include_router(sign_up_router,  prefix="/api", tags=["SignUp"])
app.include_router(vision_router,   prefix="/api", tags=["Vision"])
app.include_router(detect_router,   prefix="/api", tags=["Detect"])
app.include_router(api_favourite,   prefix="/api", tags=["Favourites"])
app.include_router(api_basket,      prefix="/api", tags=["Baskets"])
app.include_router(cat_crud_router, prefix="/api", tags=["Cat CRUD"])
app.include_router(recommen_router, prefix="/api", tags=["Recommend"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Cat Shop API is running"}