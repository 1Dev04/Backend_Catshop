"""
app/api/recommen_api.py

2 endpoints:
  GET /system/recommend/                    → list + pagination (ใช้ cat ล่าสุดของ user)
  GET /system/recommend/detail/{clothing_id} → detail + cat match score

หมายเหตุ:
  - ไม่มี table cat_clothing_recommendations
  - match real-time: cat (size/weight/chest) × cat_clothing
  - vision.py ส่ง recommendations[] มาพร้อม analyze ครั้งแรกแล้ว
    endpoint นี้ใช้เมื่อ user เข้าหน้า recommend โดยตรง หรือ refresh
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from app.db.database import get_db_pool
from app.auth.dependencies import verify_firebase_token

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _serialize(d: dict) -> dict:
    result = {}
    for k, v in d.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        elif hasattr(v, "__float__"):
            result[k] = float(v)
        else:
            result[k] = v
    return result


def _safe_float(value, default: float = 0.0) -> float:
    """float() ที่ไม่ crash เมื่อ value เป็น None"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _calc_size(chest: float) -> str:
    """คำนวณ size จาก chest_cm (fallback เมื่อ size_category เป็น None)"""
    if chest < 28:  return "XS"
    if chest < 32:  return "S"
    if chest < 36:  return "M"
    if chest < 40:  return "L"
    return "XL"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIST — recommendations ของ user พร้อม pagination
#    GET /system/recommend/
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/system/recommend/", response_model=dict)
async def get_recommendations(
    page: int = Query(default=1, ge=1, description="หน้าที่ต้องการ"),
    page_size: int = Query(default=10, ge=1, le=50, description="จำนวนต่อหน้า"),
    user: dict = Depends(verify_firebase_token),
):
    """
    ดึง clothing ที่เหมาะกับแมวล่าสุดของ user
    match_score = size(0.5) + weight_in_range(0.3) + chest_in_range(0.2)
    """
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    offset = (page - 1) * page_size
    pool = await get_db_pool()

    async with pool.acquire() as conn:

        # ── ดึง cat ล่าสุดของ user ───────────────────────────────────────────
        cat = await conn.fetchrow(
            """
            SELECT
                id, cat_color, breed, age, gender,
                weight, size_category,
                chest_cm, neck_cm, body_length_cm,
                confidence, bounding_box, thumbnail_url,
                age_category, body_condition, body_condition_score,
                body_condition_description,
                bmi, waist_cm, back_length_cm, leg_length_cm,
                posture, size_recommendation, size_ranges,
                quality_flag, analysis_version, analysis_method,
                image_cat, detected_at, updated_at
            FROM cat
            WHERE firebase_uid = $1
            ORDER BY detected_at DESC
            LIMIT 1
            """,
            firebase_uid,
        )

        if not cat:
            return {
                "cat": None,
                "items": [],
                "pagination": {
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False,
                },
                "message": "ยังไม่มีข้อมูลแมว กรุณาวิเคราะห์แมวก่อน",
            }

        # ✅ FIX: ใช้ _safe_float แทน float() ตรงๆ ป้องกัน None crash
        weight_val = _safe_float(cat["weight"], default=4.0)
        chest_val  = _safe_float(cat["chest_cm"], default=32.0)

        # ✅ FIX: size_category อาจเป็น None → คำนวณจาก chest แทน
        size = cat["size_category"] or _calc_size(chest_val)

        # ── count สำหรับ pagination ──────────────────────────────────────────
        total: int = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM cat_clothing
            WHERE is_active      = true
              AND size_category  = $1
              AND min_weight    <= $2
              AND max_weight    >= $2
            """,
            size, weight_val,
        )

        # ── fetch + คำนวณ match_score ────────────────────────────────────────
        rows = await conn.fetch(
            """
            SELECT
                id,
                uuid,
                description,
                clothing_name,
                category,
                size_category,
                min_weight,
                max_weight,
                chest_min_cm,
                chest_max_cm,
                price,
                discount_price,
                CASE
                    WHEN discount_price IS NOT NULL AND discount_price < price
                    THEN CONCAT(
                        ROUND(((price - discount_price) / price * 100)::numeric, 0),
                        '%'
                    )
                    ELSE NULL
                END                         AS discount_percent,
                stock,
                image_url,
                images,
                gender,
                breed,
                is_featured,
                clothing_like,

                -- match flags
                (size_category = $1)        AS match_size,
                (min_weight <= $2 AND max_weight >= $2)
                                            AS match_weight,
                (
                    chest_min_cm IS NOT NULL AND chest_max_cm IS NOT NULL
                    AND chest_min_cm <= $3  AND chest_max_cm >= $3
                )                           AS match_chest,

                -- match_score
                ROUND((
                    0.5
                    + CASE WHEN min_weight <= $2 AND max_weight >= $2
                           THEN 0.3 ELSE 0.0 END
                    + CASE WHEN chest_min_cm IS NOT NULL
                                AND chest_max_cm IS NOT NULL
                                AND chest_min_cm <= $3
                                AND chest_max_cm >= $3
                           THEN 0.2 ELSE 0.0 END
                )::numeric, 3)              AS match_score

            FROM cat_clothing
            WHERE is_active      = true
              AND size_category  = $1
              AND min_weight    <= $2
              AND max_weight    >= $2
            ORDER BY match_score DESC, is_featured DESC, clothing_like DESC
            LIMIT $4 OFFSET $5
            """,
            size, weight_val, chest_val, page_size, offset,
        )

    items = [_serialize(dict(r)) for r in rows]
    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        "cat": _serialize(dict(cat)),
        "items": items,
        "pagination": {
            "total":       total,
            "page":        page,
            "page_size":   page_size,
            "total_pages": total_pages,
            "has_next":    page < total_pages,
            "has_prev":    page > 1,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. DETAIL — clothing detail + cat match (หน้า "Learn More")
#    GET /system/recommend/detail/{clothing_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/system/recommend/detail/{clothing_id}", response_model=dict)
async def get_recommendation_detail(
    clothing_id: int,
    user: dict = Depends(verify_firebase_token),
):
    """
    ดึง clothing detail ครบทุก field
    + คำนวณ match score กับ cat ล่าสุดของ user
    """
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    pool = await get_db_pool()
    async with pool.acquire() as conn:

        # ── cat ล่าสุด ───────────────────────────────────────────────────────
        cat = await conn.fetchrow(
            """
            SELECT id, cat_color, breed, age, weight, size_category, chest_cm
            FROM cat
            WHERE firebase_uid = $1
            ORDER BY detected_at DESC
            LIMIT 1
            """,
            firebase_uid,
        )

        # ── clothing full detail ─────────────────────────────────────────────
        clothing = await conn.fetchrow(
            """
            SELECT
                id,
                uuid,
                image_url,
                images,
                clothing_name,
                description,
                category,
                size_category,
                min_weight,
                max_weight,
                chest_min_cm,
                chest_max_cm,
                price,
                discount_price,
                CASE
                    WHEN discount_price IS NOT NULL AND discount_price < price
                    THEN CONCAT(
                        ROUND(((price - discount_price) / price * 100)::numeric, 0),
                        '%'
                    )
                    ELSE NULL
                END     AS discount_percent,
                gender,
                clothing_like,
                clothing_seller,
                stock,
                breed,
                is_featured,
                created_at
            FROM cat_clothing
            WHERE id        = $1
              AND is_active = true
            """,
            clothing_id,
        )

    if not clothing:
        raise HTTPException(
            status_code=404,
            detail=f"Clothing id={clothing_id} not found",
        )

    result = _serialize(dict(clothing))

    # ── คำนวณ match กับ cat (ถ้ามี) ─────────────────────────────────────────
    if cat:
        # ✅ FIX: ใช้ _safe_float ทั้ง weight และ chest_cm
        c_weight = _safe_float(cat["weight"], default=4.0)
        c_chest  = _safe_float(cat["chest_cm"], default=32.0)
        c_size   = cat["size_category"] or _calc_size(c_chest)
        cl       = dict(clothing)

        match_size   = c_size == cl["size_category"]
        match_weight = (
            cl["min_weight"] is not None
            and cl["max_weight"] is not None
            and float(cl["min_weight"]) <= c_weight <= float(cl["max_weight"])
        )
        match_chest = (
            cl["chest_min_cm"] is not None
            and cl["chest_max_cm"] is not None
            and float(cl["chest_min_cm"]) <= c_chest <= float(cl["chest_max_cm"])
        )
        match_score = round(
            (0.5 if match_size   else 0.0)
            + (0.3 if match_weight else 0.0)
            + (0.2 if match_chest  else 0.0),
            3,
        )

        parts = []
        if match_size:   parts.append("ขนาดตรง")
        if match_weight: parts.append("น้ำหนักอยู่ใน range")
        if match_chest:  parts.append("รอบอกพอดี")
        reason = " • ".join(parts) if parts else "ไม่ตรงเกณฑ์"

        result["cat_match"] = {
            "cat_id":       cat["id"],
            "cat_color":    cat["cat_color"],
            "cat_size":     c_size,
            "cat_weight":   c_weight,
            "cat_chest_cm": c_chest,
            "match_score":  match_score,
            "match_size":   match_size,
            "match_weight": match_weight,
            "match_chest":  match_chest,
            "reason":       reason,
        }
    else:
        result["cat_match"] = None

    return {"item": result}  