# Crud Cat api — asyncpg version (consistent with vision.py)

import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional, List

from app.db.database import get_db_pool
from app.auth.dependencies import verify_firebase_token
from app.utils.response import success_response
from app.services.analysis_cat import analyze_cat

router = APIRouter()


# ─────────────────────────────────────────────
# helper: แปลง asyncpg Record → dict
# ─────────────────────────────────────────────
def _row(record) -> dict:
    return dict(record) if record else {}


def _rows(records) -> list:
    return [dict(r) for r in records]


def _f(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


# ============================================
# CREATE - สร้างข้อมูลแมว
# ============================================
@router.post("/system/cats", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_cat(
    cat: dict,
    user: dict = Depends(verify_firebase_token)
):
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO cat (
                    firebase_uid, cat_color, breed, age, gender,
                    weight, size_category,
                    chest_cm, neck_cm, waist_cm,
                    body_length_cm, back_length_cm, leg_length_cm,
                    confidence, bounding_box,
                    image_cat, thumbnail_url,
                    age_category,
                    body_condition_score, body_condition, body_condition_description,
                    bmi, posture, size_recommendation,
                    size_ranges, quality_flag,
                    analysis_version, analysis_method,
                    detected_at, updated_at
                ) VALUES (
                    $1,  $2,  $3,  $4,  $5,
                    $6,  $7,
                    $8,  $9,  $10,
                    $11, $12, $13,
                    $14, $15::jsonb,
                    $16, $17,
                    $18,
                    $19, $20, $21,
                    $22, $23, $24,
                    $25::jsonb, $26,
                    $27, $28,
                    $29, $30
                ) RETURNING *
                """,
                firebase_uid,
                cat.get("cat_color"),
                cat.get("breed"),
                cat.get("age"),
                cat.get("gender", 0),
                _f(cat.get("weight")),
                cat.get("size_category", "M"),
                _f(cat.get("chest_cm")),
                _f(cat.get("neck_cm")),
                _f(cat.get("waist_cm")),
                _f(cat.get("body_length_cm")),
                _f(cat.get("back_length_cm")),
                _f(cat.get("leg_length_cm")),
                _f(cat.get("confidence", 0.9)),
                json.dumps(cat.get("bounding_box", [])),
                cat.get("image_cat"),
                cat.get("thumbnail_url"),
                cat.get("age_category", "adult"),
                cat.get("body_condition_score"),
                cat.get("body_condition"),
                cat.get("body_condition_description"),
                _f(cat.get("bmi")),
                cat.get("posture"),
                cat.get("size_recommendation"),
                json.dumps(cat.get("size_ranges")) if cat.get("size_ranges") else None,
                cat.get("quality_flag", "good"),
                cat.get("analysis_version", "2.0"),
                cat.get("analysis_method", "manual"),
                datetime.utcnow(),
                datetime.utcnow(),
            )

        return success_response(data=_row(row), message="Cat created successfully")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create cat: {e}")


# ============================================
# SEARCH - ต้องอยู่ก่อน /{cat_id} เสมอ!
# ============================================
@router.get("/system/cats/search", response_model=dict)
async def search_cats(
    breed: Optional[str] = None,
    size_category: Optional[str] = None,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None,
    skip: int = 0,
    limit: int = 100,
    user: dict = Depends(verify_firebase_token)
):
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    # column names เขียนตรงใน code — ไม่รับจาก user input
    conditions = ["firebase_uid = $1"]
    params: list = [firebase_uid]
    idx = 2

    if breed is not None:
        conditions.append(f"breed ILIKE ${idx}")
        params.append(f"%{breed}%")
        idx += 1

    if size_category is not None:
        conditions.append(f"size_category = ${idx}")
        params.append(size_category)
        idx += 1

    if min_weight is not None:
        conditions.append(f"weight >= ${idx}")
        params.append(min_weight)
        idx += 1

    if max_weight is not None:
        conditions.append(f"weight <= ${idx}")
        params.append(max_weight)
        idx += 1

    # where clause ประกอบจาก literals ใน code เท่านั้น — ไม่มี user input ปน
    where_clause = " AND ".join(conditions)  # nosec B608

    count_sql = f"SELECT COUNT(*) FROM cat WHERE {where_clause}"  # nosec B608
    select_sql = (                                                  # nosec B608
        f"SELECT * FROM cat WHERE {where_clause} "
        f"ORDER BY detected_at DESC "
        f"OFFSET ${idx} LIMIT ${idx + 1}"
    )

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval(count_sql, *params)
            cats  = await conn.fetch(select_sql, *params, skip, limit)

        return success_response(
            data={
                "cats":   _rows(cats),
                "total":  total,
                "skip":   skip,
                "limit":  limit,
                "filters": {
                    "breed":         breed,
                    "size_category": size_category,
                    "min_weight":    min_weight,
                    "max_weight":    max_weight,
                },
            },
            message=f"Found {len(cats)} cats matching criteria",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search cats: {e}")


# ============================================
# READ ALL - ดึงแมวทั้งหมดของ User
# ============================================
@router.get("/system/cats", response_model=dict)
async def get_user_cats(
    skip: int = 0,
    limit: int = 100,
    user: dict = Depends(verify_firebase_token)
):
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM cat WHERE firebase_uid = $1", firebase_uid
            )
            cats = await conn.fetch(
                """
                SELECT * FROM cat
                WHERE firebase_uid = $1
                ORDER BY detected_at DESC
                OFFSET $2 LIMIT $3
                """,
                firebase_uid, skip, limit
            )

        return success_response(
            data={
                "cats": _rows(cats),
                "total": total,
                "skip": skip,
                "limit": limit,
            },
            message=f"Retrieved {len(cats)} cats"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cats: {e}")


# ============================================
# READ ONE - ดึงแมวตัวเดียว
# ============================================
@router.get("/system/cats/{cat_id}", response_model=dict)
async def get_cat(
    cat_id: int,
    user: dict = Depends(verify_firebase_token)
):
    firebase_uid = user.get("firebase_uid")

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            cat = await conn.fetchrow(
                "SELECT * FROM cat WHERE id = $1 AND firebase_uid = $2",
                cat_id, firebase_uid
            )

        if not cat:
            raise HTTPException(
                status_code=404,
                detail=f"Cat {cat_id} not found or not owned by you"
            )

        return success_response(data=_row(cat), message="Cat retrieved successfully")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cat: {e}")


# ============================================
# UPDATE - อัปเดตข้อมูลแมว
# ============================================
@router.put("/system/cats/{cat_id}", response_model=dict)
async def update_cat(
    cat_id: int,
    payload: dict,
    user: dict = Depends(verify_firebase_token)
):
    firebase_uid = user.get("firebase_uid")

    ALLOWED_COLUMNS: frozenset = frozenset({
        "cat_color", "breed", "age", "gender", "weight",
        "size_category", "chest_cm", "neck_cm", "waist_cm",
        "body_length_cm", "back_length_cm", "leg_length_cm",
        "confidence", "age_category", "body_condition_score",
        "body_condition", "body_condition_description", "bmi",
        "posture", "size_recommendation", "quality_flag",
        "analysis_version", "analysis_method", "image_cat", "thumbnail_url",
    })

    # กรอง key และ validate ว่าอยู่ใน whitelist จริงๆ
    updates: dict = {
        col: val
        for col, val in payload.items()
        if col in ALLOWED_COLUMNS
    }

    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    # column names มาจาก ALLOWED_COLUMNS (literals) เท่านั้น — ปลอดภัย
    set_parts = [
        f"{col} = ${i + 2}"          # nosec B608
        for i, col in enumerate(updates.keys())
    ]
    set_clause = ", ".join(set_parts)
    values     = list(updates.values())

    update_sql = (
        f"UPDATE cat SET {set_clause}, updated_at = NOW() "  # nosec B608
        f"WHERE id = $1 RETURNING *"
    )

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT id FROM cat WHERE id = $1 AND firebase_uid = $2",
                cat_id, firebase_uid,
            )
            if not exists:
                raise HTTPException(
                    status_code=404,
                    detail=f"Cat {cat_id} not found or not owned by you",
                )

            row = await conn.fetchrow(update_sql, cat_id, *values)

        return success_response(data=_row(row), message="Cat updated successfully")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update cat: {e}")


# ============================================
# DELETE - ลบข้อมูลแมว
# ============================================
@router.delete("/system/cats/{cat_id}", response_model=dict)
async def delete_cat(
    cat_id: int,
    user: dict = Depends(verify_firebase_token)
):
    firebase_uid = user.get("firebase_uid")

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            deleted = await conn.fetchval(
                """
                DELETE FROM cat
                WHERE id = $1 AND firebase_uid = $2
                RETURNING id
                """,
                cat_id, firebase_uid
            )

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Cat {cat_id} not found or not owned by you"
            )

        return success_response(
            data={"id": cat_id, "deleted": True},
            message="Cat deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete cat: {e}")


# ============================================
# ADMIN - ดูแมวทั้งหมดในระบบ
# ============================================
@router.get("/system/admin/cats/all", response_model=dict)
async def get_all_cats_admin(
    skip: int = 0,
    limit: int = 100,
    user: dict = Depends(verify_firebase_token)
):
    # FIX: enforce admin role check — ป้องกัน user ทั่วไปเข้าถึง
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM cat")
            cats = await conn.fetch(
                "SELECT * FROM cat ORDER BY detected_at DESC OFFSET $1 LIMIT $2",
                skip, limit
            )

        return success_response(
            data={
                "cats": _rows(cats),
                "total": total,
                "skip": skip,
                "limit": limit,
            },
            message=f"Retrieved {len(cats)} cats (admin view)"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cats: {e}")


# ============================================
# ANALYSIS SAVE - วิเคราะห์และบันทึกแมว
# ============================================
@router.post("/system/analysis/save", response_model=dict, status_code=status.HTTP_201_CREATED)
async def analyze_and_save_cat(
    image_path: str,
    bounding_box: List[float],
    cat_color: Optional[str] = None,
    breed: str = "unknown",
    age_category: str = "adult",
    image_url: Optional[str] = None,
    thumbnail_url: Optional[str] = None,
    user: dict = Depends(verify_firebase_token)
):
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    try:
        # Step 1: วิเคราะห์แมว
        analysis = analyze_cat(
            image_path=image_path,
            bounding_box=bounding_box,
            firebase_uid=firebase_uid,
            cat_color=cat_color,
            breed=breed,
            age_category=age_category
        )

        m = analysis.get("measurements", {})

        # Step 2: บันทึก DB
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO cat (
                    firebase_uid,
                    cat_color, breed, age_category,
                    weight, size_category,
                    chest_cm, neck_cm, waist_cm,
                    body_length_cm, back_length_cm, leg_length_cm,
                    body_condition_score, body_condition, body_condition_description,
                    bmi, posture,
                    size_recommendation, size_ranges,
                    confidence, quality_flag,
                    bounding_box, image_cat, thumbnail_url,
                    analysis_version, analysis_method,
                    detected_at, updated_at
                ) VALUES (
                    $1,
                    $2,  $3,  $4,
                    $5,  $6,
                    $7,  $8,  $9,
                    $10, $11, $12,
                    $13, $14, $15,
                    $16, $17,
                    $18, $19::jsonb,
                    $20, $21,
                    $22::jsonb, $23, $24,
                    $25, $26,
                    $27, $28
                ) RETURNING *
                """,
                firebase_uid,
                analysis.get("cat_color", "Unknown"),
                analysis.get("breed"),
                analysis.get("age_category", "adult"),
                _f(analysis.get("weight")),
                analysis.get("size_category", "M"),
                _f(m.get("chest_cm")),
                _f(m.get("neck_cm")),
                _f(m.get("waist_cm")),
                _f(m.get("body_length_cm")),
                _f(m.get("back_length_cm")),
                _f(m.get("leg_length_cm")),
                analysis.get("body_condition_score"),
                analysis.get("body_condition"),
                analysis.get("body_condition_description"),
                _f(analysis.get("bmi")),
                analysis.get("posture"),
                analysis.get("size_recommendation"),
                json.dumps(analysis.get("size_ranges")) if analysis.get("size_ranges") else None,
                _f(analysis.get("confidence", 0.9)),
                analysis.get("quality_flag", "good"),
                json.dumps(bounding_box),
                image_url or image_path,
                thumbnail_url,
                analysis.get("analysis_version", "2.0"),
                analysis.get("analysis_method", "gemini_1.5_flash_vision"),
                datetime.utcnow(),
                datetime.utcnow(),
            )

        result = _row(row)
        result["analysis_summary"] = {
            "weight":      analysis.get("weight"),
            "size_category":  analysis.get("size_category"),
            "body_condition": analysis.get("body_condition"),
            "confidence":     analysis.get("confidence"),
        }

        return success_response(
            data=result,
            message=f"✅ วิเคราะห์เสร็จ! น้ำหนัก {analysis.get('weight_kg')} kg, ขนาด {analysis.get('size_category')}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze and save cat: {e}")