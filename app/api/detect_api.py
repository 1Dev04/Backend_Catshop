# app/api/detect_api.py
# POST /api/detect/cat — รับ base64 จาก Flutter ตรงๆ ไม่ต้องใช้ Firebase Storage

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from app.auth.dependencies import verify_firebase_token
from app.services.detect_cat import detect_cat_base64 as _detect_cat

router = APIRouter()


# ── Request / Response ────────────────────────────────────────────────────────

class DetectCatRequest(BaseModel):
    image_base64: str        # base64 encoded image
    mime_type: str = "image/jpeg"


class DetectCatResponse(BaseModel):
    passed:        bool
    is_cat:        bool
    is_single:     bool
    is_real_photo: bool
    reason:        str
    confidence:    float
    message:       str


# ── Reason → Thai message ─────────────────────────────────────────────────────
_REASON_MESSAGE = {
    "passed":         "✅ พบแมว! กดวิเคราะห์ได้เลย",
    "no_cat":         "😿 ไม่พบแมวในภาพ ลองถ่ายใหม่ให้เห็นแมวชัดเจนทั้งตัว",
    "multiple_cats":  "🐱🐱 ตรวจพบแมวมากกว่า 1 ตัว กรุณาถ่ายรูปแมวทีละตัวเท่านั้น",
    "is_dog":         "🐶 ตรวจพบสุนัข ฟีเจอร์นี้รองรับเฉพาะแมวเท่านั้น",
    "non_cat_animal": "🚫 ตรวจพบสัตว์อื่น ฟีเจอร์นี้รองรับเฉพาะแมวเท่านั้น",
    "cartoon":        "🎨 ตรวจพบภาพการ์ตูน/ของเล่น กรุณาใช้รูปถ่ายแมวจริงเท่านั้น",
    "other":          "🤔 ไม่สามารถระบุได้ ลองถ่ายรูปใหม่ให้เห็นแมวชัดเจนขึ้น",
}


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/detect/cat", response_model=DetectCatResponse)
async def detect_cat_endpoint(
    request: DetectCatRequest,
    user: dict = Depends(verify_firebase_token),
):
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token",
        )

    print(f"\n🔍 detect/cat | user={firebase_uid[:8]}*** | mime={request.mime_type} | size={len(request.image_base64)} chars")

    try:
        result = _detect_cat(
            image_base64=request.image_base64,
            mime_type=request.mime_type,
        )

        reason  = result.get("reason", "other")
        message = _REASON_MESSAGE.get(reason, _REASON_MESSAGE["other"])

        return DetectCatResponse(
            passed        = result["passed"],
            is_cat        = result["is_cat"],
            is_single     = result["is_single"],
            is_real_photo = result["is_real_photo"],
            reason        = reason,
            confidence    = result["confidence"],
            message       = message,
        )

    except RuntimeError as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            raise HTTPException(status_code=429, detail=error_msg)
        raise HTTPException(status_code=500, detail=f"Detect failed: {error_msg}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detect failed: {e}")