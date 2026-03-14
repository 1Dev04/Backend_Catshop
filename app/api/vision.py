"""
app/api/vision.py
POST /vision/analyze-cat

Flow:
  Flutter (ML Kit detect cat ✅)
    → ส่ง image_cat มาที่ Backend
    → Gemini 2.5 Flash วิเคราะห์ขนาด/สี/สายพันธุ์  [analysis_cat.py]
    → INSERT ครบทุก column ใน table `cat`
    → Query cat_clothing ที่ match → recommend
    → คืน JSON กลับ Flutter

Root cause ของ 500:
  analysis_cat.py ส่ง measurements เป็น nested dict:
      analysis["measurements"]["chest_cm"]   ✅
  แต่ vision.py เดิมอ่านแบบ flat:
      measurements = analysis.get("measurements", {})  ← ได้ {} ทุกครั้ง
      measurements.get("chest_cm")                     ← ได้ None ทุกครั้ง
  ผลคือ chest_cm=None insert ลง NOT NULL column → 500
"""

import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from app.auth.dependencies import verify_firebase_token

from app.services.analysis_cat import analyze_cat


from app.db.database import get_db_pool

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _f(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


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


# ─────────────────────────────────────────────────────────────────────────────
# Request model
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeCatRequest(BaseModel):
    image_cat: str
    measurements: dict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/vision/analyze-cat", response_model=dict)
async def analyze_cat_endpoint(
    request: AnalyzeCatRequest,
    user: dict = Depends(verify_firebase_token),
):
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    print(f"\n🐱 analyze-cat | user={firebase_uid[:8]}*** | url={request.image_cat}"
          + (f" | measurements={list(request.measurements.keys())}"
             if request.measurements else ""))

    try:
        # STEP 1
        analysis = analyze_cat(
            image_cat=request.image_cat,
            measurements=request.measurements,  # ✅ ส่งต่อ
        )

        if not analysis.get("is_cat", True):
            return {"is_cat": False, "message": analysis.get("message", "😿 ไม่พบแมวในภาพ")}

        # ✅ flat dict ตรงๆ ไม่มี nested "measurements" แล้ว
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            cat_id = await conn.fetchval(
                """
                INSERT INTO cat (
                    firebase_uid,
                    cat_color, breed, age, gender,
                    weight, size_category,
                    chest_cm, neck_cm, waist_cm,
                    body_length_cm, back_length_cm, leg_length_cm,
                    confidence, bounding_box,
                    image_cat, thumbnail_url,
                    age_category,
                    body_condition_score, body_condition, body_condition_description,
                    bmi, posture,
                    size_recommendation, size_ranges,
                    quality_flag, analysis_version, analysis_method,
                    detected_at, updated_at
                ) VALUES (
                    $1,  $2,  $3,  $4,  $5,
                    $6,  $7,  $8,  $9,  $10,
                    $11, $12, $13, $14, $15::jsonb,
                    $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25::jsonb,
                    $26, $27, $28, $29, $30
                ) RETURNING id
                """,
                firebase_uid,
                analysis.get("cat_color", "Unknown"),
                analysis.get("breed"),
                analysis.get("age", 0),
                analysis.get("gender", 0),
                _f(analysis.get("weight")),
                analysis.get("size_category", "M"),
                # ✅ flat — อ่านตรงจาก analysis
                _f(analysis.get("chest_cm")),
                _f(analysis.get("neck_cm")),
                _f(analysis.get("waist_cm")),
                _f(analysis.get("body_length_cm")),
                _f(analysis.get("back_length_cm")),
                _f(analysis.get("leg_length_cm")),
                _f(analysis.get("confidence", 0.90)),
                json.dumps(analysis.get("bounding_box", [])),
                request.image_cat,
                None,
                analysis.get("age_category", "adult"),
                analysis.get("body_condition_score"),
                analysis.get("body_condition"),
                analysis.get("body_condition_description"),
                analysis.get("bmi"),
                analysis.get("posture"),
                analysis.get("size_recommendation"),
                json.dumps(analysis.get("size_ranges"))
                    if analysis.get("size_ranges") else None,
                analysis.get("quality_flag", "good"),
                analysis.get("analysis_version", "2.1"),
                analysis.get("analysis_method", "gpt-4.1-mini_vision"),
                datetime.utcnow(),
                datetime.utcnow(),
            )

        print(f"✅ Saved → cat.id={cat_id}")

        # STEP 3 — chest_val = None ถ้าไม่มีค่า
        size       = analysis.get("size_category", "M")
        weight_val = _f(analysis.get("weight")) or 0.0
        chest_val  = _f(analysis.get("chest_cm"))  # ✅ None ถ้าไม่มี

        async with pool.acquire() as conn:
            rec_rows = await conn.fetch(
                """
                SELECT
                    id, uuid, clothing_name, category, size_category,
                    price, discount_price,
                    CASE
                        WHEN discount_price IS NOT NULL AND discount_price < price
                        THEN CONCAT(ROUND(((price-discount_price)/price*100)::numeric,0),'%')
                        ELSE NULL
                    END AS discount_percent,
                    stock, image_url, gender, is_featured, clothing_like,
                    ROUND((
                        0.5
                        + CASE WHEN min_weight <= $2 AND max_weight >= $2
                               THEN 0.3 ELSE 0.0 END
                        + CASE WHEN $3::numeric IS NOT NULL   -- ✅ NULL-safe
                                    AND chest_min_cm IS NOT NULL
                                    AND chest_max_cm IS NOT NULL
                                    AND chest_min_cm <= $3
                                    AND chest_max_cm >= $3
                               THEN 0.2 ELSE 0.0 END
                    )::numeric, 3) AS match_score
                FROM cat_clothing
                WHERE is_active = true
                  AND size_category = $1
                  AND min_weight <= $2
                  AND max_weight >= $2
                ORDER BY match_score DESC, is_featured DESC, clothing_like DESC
                LIMIT 20
                """,
                size, weight_val, chest_val,  # chest_val อาจเป็น None
            )

        recommendations = [_serialize(dict(r)) for r in rec_rows]
        print(f"✅ Recommendations: {len(recommendations)} items")

        return {
            "is_cat":         True,
            "message":        "✅ วิเคราะห์แมวสำเร็จ!",
            "db_id":          cat_id,
            "name":           analysis.get("cat_color", "Unknown"),
            "cat_color":      analysis.get("cat_color", "Unknown"),
            "breed":          analysis.get("breed"),
            "age":            analysis.get("age", 0),
            "weight":         _f(analysis.get("weight")) or 0.0,
            "size_category":  size,
            # ✅ flat
            "chest_cm":       _f(analysis.get("chest_cm")),
            "neck_cm":        _f(analysis.get("neck_cm")),
            "body_length_cm": _f(analysis.get("body_length_cm")),
            "waist_cm":       _f(analysis.get("waist_cm")),
            "back_length_cm": _f(analysis.get("back_length_cm")),
            "leg_length_cm":  _f(analysis.get("leg_length_cm")),
            "confidence":     _f(analysis.get("confidence", 0.90)),
            "bounding_box":   analysis.get("bounding_box", []),
            "image_cat":      request.image_cat,
            "image_url":      request.image_cat,
            "thumbnail_url":  None,
            "detected_at":    datetime.utcnow().isoformat() + "Z",
            "gender":                     analysis.get("gender", 0),
            "age_category":               analysis.get("age_category"),
            "body_condition":             analysis.get("body_condition"),
            "body_condition_score":       analysis.get("body_condition_score"),
            "body_condition_description": analysis.get("body_condition_description"),
            "bmi":                        analysis.get("bmi"),
            "posture":                    analysis.get("posture"),
            "size_recommendation":        analysis.get("size_recommendation"),
            "size_ranges":                analysis.get("size_ranges"),
            "quality_flag":               analysis.get("quality_flag"),
            "analysis_method":            analysis.get("analysis_method"),
            "recommendations":            recommendations,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")