# search_api.py - Backend API for Search System

from fastapi import APIRouter, HTTPException, Query
import asyncpg
from typing import Optional, List
from pydantic import BaseModel
from app.db.database import get_db_pool

router = APIRouter()


# ============================================================================
# Models
# ============================================================================

class SearchCategoryResponse(BaseModel):
    id: int
    name_en: str
    name_th: str
    category_type: str

class ClothingItemResponse(BaseModel):
    id: int
    image_url: str
    images: dict
    clothing_name: str
    description: str
    category: int
    category_name_en: Optional[str] = None
    category_name_th: Optional[str] = None
    size_category: str
    price: float
    discount_price: Optional[float] = None
    discount_percent: Optional[int] = None
    gender: int
    stock: int
    breed: str
    created_at: str

class PaginatedResponse(BaseModel):
    items: List[ClothingItemResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


# ============================================================================
# Shared SQL fragments (literals only — no user input)
# ============================================================================

# CASE expression สำหรับเรียงลำดับ gender — ใช้ซ้ำหลายที่
_GENDER_ORDER = """
    CASE
        WHEN gender = 0 THEN 0
        WHEN gender = 1 THEN 1
        WHEN gender = 2 THEN 2
        WHEN gender = 3 THEN 3
        ELSE 4
    END
""".strip()

_DISCOUNT_PERCENT = """
    CASE
        WHEN discount_price IS NOT NULL AND discount_price < price
        THEN ROUND(((price - discount_price) / price) * 100, 0)
        ELSE NULL
    END as discount_percent
""".strip()

_DISCOUNT_PERCENT_ALIAS = """
    CASE
        WHEN c.discount_price IS NOT NULL AND c.discount_price < c.price
        THEN ROUND(((c.price - c.discount_price) / c.price) * 100, 0)
        ELSE NULL
    END as discount_percent
""".strip()


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/search/autocomplete")
async def search_autocomplete(
    query: Optional[str] = Query(None, description="Search query (optional, shows all if empty)")
):
    """
    Autocomplete search suggestions from search_category table.
    Returns matching categories based on name_category.
    If no query provided, returns all categories (up to 10).
    """
    # ORDER BY literal ทั้งหมด — ปลอดภัย
    _CATEGORY_ORDER = """
        ORDER BY
            CASE
                WHEN category_type = 'all'      THEN 0
                WHEN category_type = 'season'   THEN 1
                WHEN category_type = 'festival' THEN 2
                WHEN category_type = 'style'    THEN 3
                ELSE 4
            END,
            name_category
        LIMIT 10
    """

    try:
        pool = await get_db_pool()
        async with pool.acquire() as connection:
            if not query or not query.strip():
                sql = (  # nosec B608 — concatenates only module-level constant _CATEGORY_ORDER
                    "SELECT id, name_category, category_type "
                    "FROM search_category "
                    + _CATEGORY_ORDER
                )
                rows = await connection.fetch(sql)
            else:
                sql = (  # nosec B608 — same, value passed via $1 parameter
                    "SELECT id, name_category, category_type "
                    "FROM search_category "
                    "WHERE LOWER(name_category) LIKE LOWER($1) "
                    + _CATEGORY_ORDER
                )

                rows = await connection.fetch(sql, f"%{query}%")

        return [dict(row) for row in rows]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/search/btn/outfit/{category_id}")
async def search_btn_outfit(
    category_id: int,
    gender: Optional[int] = Query(
        None,
        description="Gender filter (0=Unisex, 1=Male, 2=Female, 3=Kitten, None=All)"
    )
):
    """
    Get clothing items by category_id and optional gender filter.
    - category_id: ID from search_category table
    - gender: optional filter
    """
    # Base conditions — literals เท่านั้น
    conditions = ["category_id = $1", "is_active = true"]
    params: list = [category_id]

    if gender is not None:
        conditions.append(f"gender = ${len(params) + 1}")
        params.append(gender)

    # where_clause ประกอบจาก literals + $N placeholders เท่านั้น
    where_clause = " AND ".join(conditions)  # nosec B608

    sql = (  # nosec B608 — concatenates only module-level constants, no user input
           "SELECT id, uuid, image_url, images, clothing_name, description, "
            "category, size_category, price, discount_price, "
            + _DISCOUNT_PERCENT + ", "
            "gender, clothing_like, clothing_seller, stock, breed, "
            "category_id, created_at "
            "FROM cat_clothing "
            "WHERE " + where_clause + " "
            "ORDER BY " + _GENDER_ORDER + ", created_at DESC"
        )

    try:
        pool = await get_db_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(sql, *params)
            print(f"✅ Found {len(rows)} items")

            if not rows:
                print("⚠️ Warning: No items found in database!")
                return []

            return [dict(row) for row in rows]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/search/clothing", response_model=PaginatedResponse)
async def search_clothing_page(
    category_id: Optional[int] = Query(None, description="Category ID from search_category"),
    gender: Optional[int] = Query(
        None,
        description="Gender filter (0=Unisex, 1=Male, 2=Female, 3=Kitten)"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page")
):
    """
    Search clothing items with filtering and pagination.
    - category_id + gender: optional combined filters
    - Neither provided: returns all active items
    """
    # Base condition — literal เท่านั้น
    conditions = ["c.is_active = true"]
    params: list = []
    param_count = 1

    if category_id is not None:
        conditions.append(f"c.category_id = ${param_count}")
        params.append(category_id)
        param_count += 1

    if gender is not None:
        conditions.append(f"c.gender = ${param_count}")
        params.append(gender)
        param_count += 1

    # where_clause ประกอบจาก literals + $N placeholders เท่านั้น
    where_clause = " AND ".join(conditions)  # nosec B608

    count_sql = (  # nosec B608
        "SELECT COUNT(*) FROM cat_clothing c WHERE " + where_clause
    )


    items_sql = (  # nosec B608
        "SELECT c.id, c.uuid, c.image_url, c.images, "
        "c.clothing_name, c.description, c.category_id, "
        "sc.name_en AS category_name_en, sc.name_th AS category_name_th, "
        "c.size_category, c.price, c.discount_price, "
        + _DISCOUNT_PERCENT_ALIAS + ", "
        "c.gender, c.stock, c.breed, c.created_at "
        "FROM cat_clothing c "
        "LEFT JOIN search_category sc ON c.category_id = sc.id "
        "WHERE " + where_clause + " "
        f"ORDER BY c.created_at DESC "
        f"LIMIT ${param_count} OFFSET ${param_count + 1}"
    )

    try:
        pool = await get_db_pool()
        async with pool.acquire() as connection:
            total_count = await connection.fetchval(count_sql, *params)

            offset = (page - 1) * page_size
            total_pages = (total_count + page_size - 1) // page_size

            rows = await connection.fetch(items_sql, *params, page_size, offset)
            items = [dict(row) for row in rows]

        return {
            "items":       items,
            "total":       total_count,
            "page":        page,
            "page_size":   page_size,
            "total_pages": total_pages,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")