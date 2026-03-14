"""
app/api/recommen_api.py
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from app.db.database import get_db_pool
from app.auth.dependencies import verify_firebase_token

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _calc_size(chest: float) -> str:
    if chest < 28: return "XS"
    if chest < 32: return "S"
    if chest < 36: return "M"
    if chest < 40: return "L"
    return "XL"


def _resolve_size(cat_row: dict) -> str | None:
    """
    Priority:
      1. size_category  (บันทึกใน DB แล้ว — ถูกต้องที่สุด)
      2. size_recommendation (fallback จาก AI)
      3. คำนวณจาก chest_cm จริง (ถ้ามี)
      4. None → ไม่รู้ size จริงๆ อย่า force
    ไม่ใช้ default chest 32.0 เด็ดขาด
    """
    # 1. size_category
    val = cat_row.get("size_category")
    if val and str(val).strip():
        return str(val).strip()

    # 2. size_recommendation
    val = cat_row.get("size_recommendation")
    if val and str(val).strip():
        return str(val).strip()

    # 3. คำนวณจาก chest_cm ถ้ามีค่าจริง
    chest = cat_row.get("chest_cm")
    if chest is not None:
        try:
            return _calc_size(float(chest))
        except (TypeError, ValueError):
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIST
#    GET /system/recommend/
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/system/recommend/", response_model=dict)
async def get_recommendations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50),
    cat_id: int = Query(default=None, description="ระบุ cat id (optional, default=ล่าสุด)"),
    user: dict = Depends(verify_firebase_token),
):
    """
    match_score = size(0.4) + breed(0.1) + weight(0.3) + chest(0.2)
    - breed IS NULL ใน clothing → ใส่ได้ทุกพันธุ์ → match_breed = true
    - cat breed IS NULL → ไม่กรอง breed เลย
    - chest_cm IS NULL ใน cat → ไม่ใช้ default, skip match_chest
    """
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    offset = (page - 1) * page_size
    pool = await get_db_pool()

    async with pool.acquire() as conn:

        # ── ดึง cat ─────────────────────────────────────────────────────────
        if cat_id is not None:
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
                WHERE id = $1 AND firebase_uid = $2
                """,
                cat_id, firebase_uid,
            )
        else:
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
                    "total": 0, "page": page, "page_size": page_size,
                    "total_pages": 0, "has_next": False, "has_prev": False,
                },
                "message": "ยังไม่มีข้อมูลแมว กรุณาวิเคราะห์แมวก่อน",
            }

        cat_dict   = dict(cat)
        weight_val = _safe_float(cat_dict.get("weight"), default=4.0)
        breed_val  = cat_dict.get("breed")  # None = ไม่กรอง breed

        # ── resolve size (ไม่ใช้ default chest) ──────────────────────────────
        size = _resolve_size(cat_dict)
        if not size:
            return {
                "cat": _serialize(cat_dict),
                "items": [],
                "pagination": {
                    "total": 0, "page": page, "page_size": page_size,
                    "total_pages": 0, "has_next": False, "has_prev": False,
                },
                "message": "ไม่สามารถระบุขนาดแมวได้ กรุณาวิเคราะห์แมวใหม่หรือแก้ไขข้อมูลแมว",
            }

        # ── chest_val: None ถ้าไม่มีค่าจริง ──────────────────────────────────
        # ใช้ None แทน default 32.0 เพื่อให้ query ไม่ match_chest ผิดพลาด
        chest_raw  = cat_dict.get("chest_cm")
        chest_val: float | None = (
            float(chest_raw) if chest_raw is not None else None
        )

        # ── count ─────────────────────────────────────────────────────────────
        total: int = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM cat_clothing
            WHERE is_active     = true
              AND size_category = $1
              AND (
                  $2::text IS NULL          -- cat ไม่รู้ breed → แสดงทั้งหมด
                  OR breed IS NULL          -- clothing ไม่ระบุ breed → ใส่ได้ทุกพันธุ์
                  OR breed = $2             -- breed ตรง
              )
            """,
            size, breed_val,
        )

        # ── fetch + match_score ───────────────────────────────────────────────
        rows = await conn.fetch(
            """
            SELECT
                id, uuid, description, clothing_name, category,
                size_category, min_weight, max_weight,
                chest_min_cm, chest_max_cm,
                price, discount_price,
                CASE
                    WHEN discount_price IS NOT NULL
                    AND  discount_price > 0
                    AND  discount_price < price
                    THEN CONCAT(
                        ROUND(((price - discount_price) / price * 100)::numeric, 0),
                        '%')
                    ELSE NULL
                END AS discount_percent,
                stock, image_url, images, gender, breed,
                is_featured, clothing_like,

                -- match flags
                (size_category = $1) AS match_size,
                (min_weight <= $2 AND max_weight >= $2) AS match_weight,

                -- match_chest: ถ้า cat ไม่มี chest_cm ($3 IS NULL) → false (ไม่นับ)
                (
                    $3::numeric IS NOT NULL
                    AND chest_min_cm IS NOT NULL
                    AND chest_max_cm IS NOT NULL
                    AND chest_min_cm <= $3
                    AND chest_max_cm >= $3
                ) AS match_chest,

                -- breed: NULL clothing = ใส่ได้ทุกพันธุ์ → true
                (breed IS NULL OR $4::text IS NULL OR breed = $4) AS match_breed,

                -- match_score: size(0.4) + breed(0.1) + weight(0.3) + chest(0.2)
                -- chest score: 0.2 ถ้ามีค่าและตรง, 0.0 ถ้าไม่มีค่าหรือไม่ตรง
                ROUND((
                    0.4
                    + CASE WHEN breed IS NULL OR $4::text IS NULL OR breed = $4
                           THEN 0.1 ELSE 0.0 END
                    + CASE WHEN min_weight <= $2 AND max_weight >= $2
                           THEN 0.3 ELSE 0.0 END
                    + CASE
                           WHEN $3::numeric IS NOT NULL
                                AND chest_min_cm IS NOT NULL
                                AND chest_max_cm IS NOT NULL
                                AND chest_min_cm <= $3
                                AND chest_max_cm >= $3
                           THEN 0.2 ELSE 0.0 END
                )::numeric, 3) AS match_score

            FROM cat_clothing
            WHERE is_active     = true
              AND size_category = $1
              AND (
                  $4::text IS NULL
                  OR breed IS NULL
                  OR breed = $4
              )
            ORDER BY match_score DESC, is_featured DESC, clothing_like DESC
            LIMIT $5 OFFSET $6
            """,
            size, weight_val, chest_val, breed_val, page_size, offset,
        )

    items = [_serialize(dict(r)) for r in rows]
    total_pages = max(1, (total + page_size - 1) // page_size)

    return {
        "cat": _serialize(cat_dict),
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
# 2. DETAIL
#    GET /system/recommend/detail/{clothing_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/system/recommend/detail/{clothing_id}", response_model=dict)
async def get_recommendation_detail(
    clothing_id: int,
    user: dict = Depends(verify_firebase_token),
):
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    pool = await get_db_pool()
    async with pool.acquire() as conn:

        cat = await conn.fetchrow(
            """
            SELECT id, cat_color, breed, age, weight,
                   size_category, size_recommendation, chest_cm
            FROM cat
            WHERE firebase_uid = $1
            ORDER BY detected_at DESC
            LIMIT 1
            """,
            firebase_uid,
        )

        clothing = await conn.fetchrow(
            """
            SELECT
                id, uuid, image_url, images, clothing_name, description,
                category, size_category, min_weight, max_weight,
                chest_min_cm, chest_max_cm, price, discount_price,
                CASE
                    WHEN discount_price IS NOT NULL
                    AND  discount_price > 0
                    AND  discount_price < price
                    THEN CONCAT(
                        ROUND(((price - discount_price) / price * 100)::numeric, 0),
                        '%')
                    ELSE NULL
                END AS discount_percent,
                gender, clothing_like, clothing_seller,
                stock, breed, is_featured, created_at
            FROM cat_clothing
            WHERE id = $1 AND is_active = true
            """,
            clothing_id,
        )

    if not clothing:
        raise HTTPException(status_code=404, detail=f"Clothing id={clothing_id} not found")

    result = _serialize(dict(clothing))

    if cat:
        cat_dict = dict(cat)

        # ── resolve size (ไม่ใช้ default chest) ──────────────────────────────
        c_size = _resolve_size(cat_dict)

        # ── chest_val: None ถ้าไม่มีค่าจริง ──────────────────────────────────
        chest_raw = cat_dict.get("chest_cm")
        c_chest: float | None = (
            float(chest_raw) if chest_raw is not None else None
        )

        c_weight = _safe_float(cat_dict.get("weight"), default=4.0)
        c_breed  = cat_dict.get("breed")
        cl       = dict(clothing)

        # match_size: ถ้า resolve size ไม่ได้ → False
        match_size = bool(c_size and c_size == cl["size_category"])

        match_weight = (
            cl["min_weight"] is not None and cl["max_weight"] is not None
            and float(cl["min_weight"]) <= c_weight <= float(cl["max_weight"])
        )

        # match_chest: ถ้า c_chest เป็น None → False (ไม่ใช้ default)
        match_chest = (
            c_chest is not None
            and cl["chest_min_cm"] is not None
            and cl["chest_max_cm"] is not None
            and float(cl["chest_min_cm"]) <= c_chest <= float(cl["chest_max_cm"])
        )

        # breed IS NULL ใน clothing = ใส่ได้ทุกพันธุ์
        match_breed = cl["breed"] is None or c_breed is None or cl["breed"] == c_breed

        match_score = round(
            (0.4 if match_size   else 0.0)
            + (0.1 if match_breed  else 0.0)
            + (0.3 if match_weight else 0.0)
            + (0.2 if match_chest  else 0.0),
            3,
        )

        parts = []
        if match_size:   parts.append("ขนาดตรง")
        if match_breed:  parts.append("พันธุ์ตรง")
        if match_weight: parts.append("น้ำหนักอยู่ใน range")
        if match_chest:  parts.append("รอบอกพอดี")
        reason = " • ".join(parts) if parts else "ไม่ตรงเกณฑ์"

        result["cat_match"] = {
            "cat_id":       cat_dict["id"],
            "cat_color":    cat_dict["cat_color"],
            "cat_size":     c_size,
            "cat_weight":   c_weight,
            "cat_chest_cm": c_chest,      # อาจเป็น None (แสดงตรงๆ ไม่ปลอมค่า)
            "match_score":  match_score,
            "match_size":   match_size,
            "match_breed":  match_breed,
            "match_weight": match_weight,
            "match_chest":  match_chest,
            "reason":       reason,
        }
    else:
        result["cat_match"] = None

    return {"item": result}