# app/services/detect_cat.py
# รับ base64 จาก Flutter — auto fallback ถ้า model quota หมด

import os
import re
import json
import uuid
import base64
import logging

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Client ────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.environ["OPEN_API_KEY_DETECT"])

MODEL = "gpt-4.1-mini"

# ── Prompt ────────────────────────────────────────────────────────────────────
DETECT_PROMPT = """
You are a cat image validator. Analyze the image carefully and respond ONLY with raw JSON.
No markdown. No explanation. No extra text.

Schema (return exactly this):
{
  "is_cat": boolean,
  "is_single": boolean,
  "is_real_photo": boolean,
  "reason": string,
  "confidence": number
}

Rules:
- "is_cat": true ONLY if there is a real domestic cat visible (NOT lion/tiger/cheetah/wildcat)
- "is_single": true ONLY if exactly 1 cat is visible in the entire image
- "is_real_photo": true = real photograph | false = cartoon/anime/drawing/plush/figurine/3D render/toy
- "reason": MUST be one of these exact strings:
    "passed"         → single real cat, real photo ✅
    "no_cat"         → no cat found in image
    "multiple_cats"  → 2 or more cats detected
    "is_dog"         → dog detected (not a cat)
    "non_cat_animal" → other animal (rabbit, bird, hamster, etc.)
    "cartoon"        → cartoon / drawing / toy / not a real photo
    "other"          → cannot determine clearly
- "confidence": float 0.0-1.0 how confident you are in the result
"""


# ── JSON Parser ───────────────────────────────────────────────────────────────

def _parse_json_robust(raw_text: str) -> dict:
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find('{')
    if start != -1:
        depth, end = 0, -1
        in_string, escape_next = False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

    if '"is_cat": false' in text or '"is_cat":false' in text:
        return {
            "is_cat": False,
            "is_single": False,
            "is_real_photo": False,
            "reason": "no_cat",
            "confidence": 0.0,
        }

    raise RuntimeError(f"Cannot parse OpenAI detect response. Preview: {text[:200]}")


# ── Quota error check ─────────────────────────────────────────────────────────

def _is_quota_error(error_str: str) -> bool:
    return (
        "insufficient_quota" in error_str
        or "quota" in error_str.lower()
        or "429" in error_str
        or "rate_limit" in error_str.lower()
    )


# ── OpenAI Caller ─────────────────────────────────────────────────────────────

def _call_openai_detect(image_bytes: bytes, mime_type: str) -> str:
    request_id = str(uuid.uuid4())[:8]
    print(f"[detect/{request_id}] 🔍 Calling model: {MODEL}")

    # encode image to base64 data URL
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_image}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": DETECT_PROMPT},
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content.strip()

        if not raw_text:
            raise RuntimeError("Empty response from OpenAI detect")

        print(f"[detect/{request_id}] ✅ OK | model={MODEL} | {len(raw_text)} chars")
        return raw_text

    except Exception as e:
        error_str = str(e)
        print(f"[detect/{request_id}] ❌ {MODEL} failed: {error_str[:150]}")

        if _is_quota_error(error_str):
            raise RuntimeError("OpenAI quota หมด กรุณาลองใหม่ภายหลัง")

        raise RuntimeError(f"OpenAI detect failed [{request_id}]: {error_str}")


# ── Helper: build result dict ─────────────────────────────────────────────────

def _build_result(raw_text: str) -> dict:
    try:
        result = _parse_json_robust(raw_text)
    except RuntimeError as e:
        logger.error(f"detect parse error: {e}")
        return {
            "is_cat": True, "is_single": True, "is_real_photo": True,
            "reason": "other", "confidence": 0.5, "passed": True,
        }

    is_cat     = bool(result.get("is_cat", False))
    is_single  = bool(result.get("is_single", True))
    is_real    = bool(result.get("is_real_photo", True))
    reason     = result.get("reason", "other")
    confidence = float(result.get("confidence", 0.0))
    passed     = is_cat and is_single and is_real

    print(
        f"🔍 detect result: passed={passed} | reason={reason} "
        f"| cat={is_cat} single={is_single} real={is_real} | conf={confidence:.2f}"
    )

    return {
        "is_cat":        is_cat,
        "is_single":     is_single,
        "is_real_photo": is_real,
        "reason":        reason,
        "confidence":    confidence,
        "passed":        passed,
    }


# ── Main Entry Points ─────────────────────────────────────────────────────────

def detect_cat_base64(image_base64: str, mime_type: str = "image/jpeg") -> dict:
    """รับ base64 จาก Flutter โดยตรง"""
    print(f"🔍 detect_cat_base64: mime={mime_type} | size={len(image_base64)} chars")

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        raise RuntimeError(f"Cannot decode base64 image: {e}")

    print(f"✅ Decoded ({len(image_bytes)/1024:.1f} KB)")
    raw_text = _call_openai_detect(image_bytes, mime_type)
    return _build_result(raw_text)


def detect_cat(image_url: str) -> dict:
    """Legacy: รับ URL"""
    import requests
    print(f"🔍 detect_cat: downloading {image_url}")
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot download image for detect: {e}")

    image_bytes = resp.content
    mime_type   = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    print(f"✅ Downloaded ({len(image_bytes)/1024:.1f} KB) | mime={mime_type}")

    raw_text = _call_openai_detect(image_bytes, mime_type)
    return _build_result(raw_text)