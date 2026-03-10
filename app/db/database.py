import asyncpg
import asyncio
import os

_pool = None


def get_database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://catuser:Qc6CgariRTv2thkYymNzhLrmKRGE3CPN@dpg-d645vn7pm1nc738ihcng-a.singapore-postgres.render.com:5432/catdb_xt7q"
    )


async def create_db_pool():
    global _pool
    DATABASE_URL = os.getenv("DATABASE_URL")

    for i in range(10):
        try:
            _pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=5,
                server_settings={'client_encoding': 'UTF8'}
            )
            print("✅ Database connected")
            return
        except Exception as e:
            print(f"⏳ Waiting for DB... ({i+1}/10)")
            await asyncio.sleep(2)

    print("❌ Database connection failed")
    raise RuntimeError("Database not ready")

async def get_db_pool() -> asyncpg.Pool:
    if _pool is None:
        await create_db_pool()
    return _pool


async def close_db_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        print("🧹 Database pool closed")
