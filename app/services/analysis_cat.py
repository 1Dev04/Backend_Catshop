import os
import re
import json
import uuid
import base64
import logging
import requests
import time
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import Optional

load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ["OPEN_API_KEY_ANALYSIS"])
MODEL = "gpt-4.1-mini"


# ── Size helpers (single source of truth) ────────────────────

def _calc_size(chest: Optional[float]) -> str:
    """
    XS: chest < 28
    S : 28 <= chest < 32
    M : 32 <= chest < 36
    L : 36 <= chest < 40
    XL: chest >= 40
    """
    if chest is None: return "M"
    if chest < 28:    return "XS"
    if chest < 32:    return "S"
    if chest < 36:    return "M"
    if chest < 40:    return "L"
    return "XL"


def _size_ranges_from_chest(chest: float) -> dict:
    size = _calc_size(chest)
    table = {
        "XS": dict(chest_min=24, chest_max=27, neck_min=16, neck_max=20, back_length_min=28, back_length_max=34),
        "S":  dict(chest_min=28, chest_max=31, neck_min=18, neck_max=22, back_length_min=32, back_length_max=38),
        "M":  dict(chest_min=32, chest_max=35, neck_min=20, neck_max=24, back_length_min=36, back_length_max=42),
        "L":  dict(chest_min=36, chest_max=39, neck_min=22, neck_max=26, back_length_min=40, back_length_max=46),
        "XL": dict(chest_min=40, chest_max=50, neck_min=24, neck_max=28, back_length_min=44, back_length_max=50),
    }
    return table[size]


# ── Prompt builder ────────────────────────────────────────────

def _build_prompt(measurements: Optional[dict] = None) -> str:
    """
    Build CAT_ANALYSIS_PROMPT
    ถ้ามี measurements จาก user → เพิ่ม MEASUREMENT OVERRIDE section
    ที่บอก AI ให้ใช้ค่าจริงแทนการประมาณจากภาพ
    """

    # ── measurement override block ────────────────────────────
    measurement_block = ""
    forced_size = None

    if measurements:
        chest  = _to_float(measurements.get("chest_cm"))
        neck   = _to_float(measurements.get("neck_cm"))
        waist  = _to_float(measurements.get("waist_cm"))
        body   = _to_float(measurements.get("body_length_cm"))
        back   = _to_float(measurements.get("back_length_cm"))
        leg    = _to_float(measurements.get("leg_length_cm"))
        weight = _to_float(measurements.get("weight"))

        lines = []
        if chest  is not None: lines.append(f"  chest_cm        = {chest} cm  ← PRIMARY")
        if neck   is not None: lines.append(f"  neck_cm         = {neck} cm")
        if waist  is not None: lines.append(f"  waist_cm        = {waist} cm")
        if body   is not None: lines.append(f"  body_length_cm  = {body} cm")
        if back   is not None: lines.append(f"  back_length_cm  = {back} cm")
        if leg    is not None: lines.append(f"  leg_length_cm   = {leg} cm")
        if weight is not None: lines.append(f"  weight          = {weight} kg")

        if lines:
            if chest is not None:
                forced_size = _calc_size(chest)
                ranges      = _size_ranges_from_chest(chest)
                size_note   = (
                    f"\n  → size_category MUST be \"{forced_size}\" "
                    f"(chest={chest}cm → rule: XS<28, S<32, M<36, L<40, XL≥40)"
                    f"\n  → size_ranges MUST be {json.dumps(ranges)}"
                )
            else:
                size_note = ""

            measurement_block = f"""
════════════════════════════════════════════════════════
⚠️  MEASUREMENT OVERRIDE  (user-provided — highest priority)
════════════════════════════════════════════════════════
The owner has physically measured this cat. These values are GROUND TRUTH.
You MUST use them exactly in your JSON response. Do NOT estimate from the image.

{chr(10).join(lines)}
{size_note}

  analysis_method MUST be "user_measurements"

If a field is provided above → copy it verbatim into the JSON.
If a field is NOT provided   → estimate from image as usual.
════════════════════════════════════════════════════════
"""

    prompt = f"""
You are a professional cat analysis AI specialized in pet body measurement and health assessment.

STRICT OUTPUT REQUIREMENTS:
- Return raw JSON only. No markdown. No explanation. No extra text.
- JSON must be complete and properly closed — never truncate.
- Every field must be present. Use null only where explicitly allowed.
{measurement_block}
════════════════════════════════════════════════════════
STEP 1 — REALITY CHECK (evaluate BEFORE anything else)
════════════════════════════════════════════════════════

First, determine the subject_type from one of these exact values:
  "real_cat"         → a live, biological cat (the ONLY type that should proceed)
  "cartoon"          → drawing, illustration, anime, painting, CGI, digital art
  "stuffed_toy"      → plush toy, stuffed animal, cat doll
  "figurine_model"   → resin/plastic/ceramic cat figure, scale model
  "human_in_costume" → person wearing a cat costume, cat ears, cat makeup
  "cat_mask_prop"    → cat mask, cat face prop held/worn by a human
  "printed_image"    → photo of a physical printed paper/poster only (screen photos are ALLOWED)
  "other_animal"     → dog, rabbit, or any non-cat animal
  "no_cat"           → no cat-like subject at all

Key detection rules for REJECTING fake cats:
  ✗ Human body proportions (upright torso, human hands/feet, legs)
  ✗ Fabric texture, stitching seams, button eyes → stuffed toy
  ✗ Uniform surface, no fur texture, painted features → figurine
  ✗ Flat/drawn lines, cel-shading, unrealistic colors → cartoon
  ✗ A person wearing ears/tail/makeup — body is still human
  ✗ Cat face but human body parts visible

Key features of a REAL cat:
  ✓ Actual fur texture with individual hair strands
  ✓ Natural cat body proportions (4 legs, horizontal spine)
  ✓ Realistic eyes with slit pupils or round pupils
  ✓ Natural muscle/fat variation under fur
  ✓ Paws with visible toe beans or claws
  ✓ Natural lighting interaction on fur

════════════════════════════════════════════════════════
STEP 2 — RETURN BASED ON subject_type
════════════════════════════════════════════════════════

If subject_type is NOT "real_cat", return ONLY this and STOP:
{{
  "is_cat": false,
  "subject_type": "<detected type>",
  "message": "<short Thai message explaining rejection>",
  "confidence": 0.95
}}

Message templates by type:
  cartoon          → "ตรวจพบภาพการ์ตูนหรือภาพวาด ไม่ใช่แมวจริง"
  stuffed_toy      → "ตรวจพบตุ๊กตาหรือของเล่นรูปแมว ไม่ใช่แมวจริง"
  figurine_model   → "ตรวจพบโมเดลหรือฟิกเกอร์รูปแมว ไม่ใช่แมวจริง"
  human_in_costume → "ตรวจพบมนุษย์ที่แต่งตัวเป็นแมว ไม่ใช่แมวจริง"
  cat_mask_prop    → "ตรวจพบหน้ากากแมวหรืออุปกรณ์ประกอบฉาก ไม่ใช่แมวจริง"
  other_animal     → "ตรวจพบสัตว์ชนิดอื่น ไม่ใช่แมว"
  no_cat           → "ไม่พบแมวในภาพ"

════════════════════════════════════════════════════════
STEP 3 — FULL ANALYSIS (only if subject_type == "real_cat")
════════════════════════════════════════════════════════

Return this exact schema (all fields required):
{{
  "is_cat": true,
  "subject_type": "real_cat",
  "cat_color": "describe main color(s) e.g. orange tabby, black and white, grey tabby",
  "breed": "always provide best guess — NEVER null",
  "age": 3,
  "gender": 0,

  "weight": 4.5,
  "chest_cm": 32.0,
  "neck_cm": 22.0,
  "waist_cm": 28.0,
  "body_length_cm": 45.0,
  "back_length_cm": 38.0,
  "leg_length_cm": 12.0,

  "body_condition_score": 5,
  "body_condition": "normal",
  "body_condition_description": "Healthy weight, ribs palpable with slight fat cover",

  "posture": "sitting",
  "size_recommendation": "M",
  "size_category": "M",
  "size_ranges": {{
    "chest_min": 32.0,
    "chest_max": 36.0,
    "neck_min": 20.0,
    "neck_max": 24.0,
    "back_length_min": 36.0,
    "back_length_max": 42.0
  }},

  "quality_flag": "good",
  "analysis_method": "ai_estimation",
  "confidence": 0.87
}}

DERIVATION RULES:

age (integer, NEVER null — use 0 if truly unknown):
  Estimate from face, body proportions, coat condition.

gender: 0=unknown/female, 1=male

weight (kg, float): estimate from body volume vs typical domestic cat.

breed (string, NEVER null):
  Always provide best guess. If uncertain → "Domestic Shorthair" or "Mixed Breed".

size_category and size_recommendation (MUST match chest_cm using this rule):
  XS: chest_cm < 28
  S : 28 <= chest_cm < 32
  M : 32 <= chest_cm < 36
  L : 36 <= chest_cm < 40
  XL: chest_cm >= 40

  ⚠️  size_category and size_recommendation MUST be derived from chest_cm above.
      Never set size independently from chest_cm.

size_ranges (derive from size_category):
  XS: chest_min=24  chest_max=27  neck_min=16 neck_max=20 back_min=28 back_max=34
  S : chest_min=28  chest_max=31  neck_min=18 neck_max=22 back_min=32 back_max=38
  M : chest_min=32  chest_max=35  neck_min=20 neck_max=24 back_min=36 back_max=42
  L : chest_min=36  chest_max=39  neck_min=22 neck_max=26 back_min=40 back_max=46
  XL: chest_min=40  chest_max=50  neck_min=24 neck_max=28 back_min=44 back_max=50

body_condition_score: integer 1(emaciated) to 9(obese), 4-5=ideal
body_condition: underweight | normal | overweight | obese
posture: standing | sitting | lying | crouching | other
quality_flag: good | blurry | partial | dark | backlit | other
analysis_method: "user_measurements" if owner measurements were provided, else "ai_estimation"
confidence: 0.0-1.0 reflecting image clarity and full body visibility
"""
    return prompt.strip()


# ── subject_type constants ────────────────────────────────────
FAKE_CAT_TYPES = {
    "cartoon", "stuffed_toy", "figurine_model",
    "human_in_costume", "cat_mask_prop",
    "other_animal", "no_cat",
}


# ── Pydantic Schema ───────────────────────────────────────────
class SizeRanges(BaseModel):
    chest_min: float
    chest_max: float
    neck_min: float
    neck_max: float
    back_length_min: float
    back_length_max: float


class CatAnalysisSchema(BaseModel):
    is_cat: bool
    subject_type: str = "real_cat"
    cat_color: str
    breed: str = "Domestic Shorthair"
    age: int = 0
    gender: int = 0
    weight: float
    chest_cm: float
    neck_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    body_length_cm: Optional[float] = None
    back_length_cm: Optional[float] = None
    leg_length_cm: Optional[float] = None
    body_condition_score: int = Field(..., ge=1, le=9)
    body_condition: str
    body_condition_description: Optional[str] = None
    posture: str = "other"
    size_recommendation: Optional[str] = None
    size_ranges: Optional[SizeRanges] = None
    quality_flag: str = "good"
    analysis_method: str = "ai_estimation"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("weight", "chest_cm", mode="before")
    @classmethod
    def cast_to_float(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert '{v}' to float")

    @field_validator("body_condition_score", mode="before")
    @classmethod
    def clamp_bcs(cls, v):
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 5
        return max(1, min(9, v))

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v):
        if v is None:
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def from_ai(cls, data: dict) -> "CatAnalysisSchema":
        return cls(**data)


# ── Pure Helpers ──────────────────────────────────────────────

def _to_float(value) -> Optional[float]:
    try:
        f = float(value)
        return f if f == f else None   # NaN guard
    except (TypeError, ValueError):
        return None


def _calc_bmi(weight: Optional[float], body_length_cm: Optional[float]) -> Optional[float]:
    if not weight or not body_length_cm or body_length_cm <= 0:
        return None
    return round(weight / ((body_length_cm / 100.0) ** 2), 2)


def _calc_age_category(age: int) -> str:
    if age < 1:   return "kitten"
    if age <= 2:  return "junior"
    if age <= 10: return "adult"
    return "senior"


def _log_parse_error(raw_text: str, error, request_id: str = "") -> None:
    log_path = f"parse_error_{int(time.time())}.log"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"RequestID: {request_id}\nError: {error}\n\nRaw response:\n{raw_text}")
        logger.error(f"[{request_id}] Parse error logged to: {log_path}")
    except Exception as e:
        logger.error(f"[{request_id}] Failed to write parse error log: {e}")


# ── Robust JSON Parser ────────────────────────────────────────

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
        depth = 0
        end = -1
        in_string = False
        escape_next = False
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
            except json.JSONDecodeError as e:
                logger.warning(f"Brace-counted block still invalid: {e}")

    fragment = cleaned[start:] if start != -1 else cleaned
    repaired = _repair_truncated_json(fragment)
    if repaired:
        try:
            result = json.loads(repaired)
            logger.warning("Used truncated JSON repair — result may be incomplete")
            return result
        except json.JSONDecodeError:
            pass

    if '"is_cat": false' in text or '"is_cat":false' in text:
        return {"is_cat": False, "subject_type": "no_cat", "message": "ไม่พบแมวในภาพ"}

    raise RuntimeError(
        f"OpenAI returned invalid JSON. "
        f"Length={len(text)}, Preview: {text[:300]}"
    )


def _repair_truncated_json(text: str) -> Optional[str]:
    try:
        lines = text.split('\n')
        complete_lines = []
        for line in lines:
            s = line.strip()
            if s.endswith(',') or s.endswith('{') or s.endswith('[') or s == '{':
                complete_lines.append(line)
            else:
                break
        if not complete_lines:
            return None
        partial = '\n'.join(complete_lines).rstrip().rstrip(',')
        depth_brace   = partial.count('{') - partial.count('}')
        depth_bracket = partial.count('[') - partial.count(']')
        return partial + ']' * depth_bracket + '}' * depth_brace
    except Exception:
        return None


# ── OpenAI Caller with retry ──────────────────────────────────

def _call_openai_with_retry(
    image_bytes: bytes,
    mime_type: str,
    measurements: Optional[dict] = None,
) -> str:
    request_id = str(uuid.uuid4())[:8]
    max_retries = 3
    base_wait   = 3

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url  = f"data:{mime_type};base64,{b64_image}"

    # build prompt once (with or without measurements)
    prompt = _build_prompt(measurements)

    for attempt in range(max_retries):
        start = time.time()
        try:
            print(f"[{request_id}] 🤖 OpenAI attempt {attempt + 1}/{max_retries}"
                  + (" [user_measurements]" if measurements else " [ai_estimation]"))

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=4000,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content.strip()
            latency  = round(time.time() - start, 2)

            if not raw_text:
                raise RuntimeError("Empty OpenAI response")

            if not raw_text.rstrip().endswith("}"):
                print(f"[{request_id}] ⚠️  Truncated response ({len(raw_text)} chars), retrying...")
                raise RuntimeError("Truncated OpenAI JSON")

            print(f"[{request_id}] ✅ OK | {len(raw_text)} chars | {latency}s")
            return raw_text

        except Exception as e:
            error_str = str(e)
            latency   = round(time.time() - start, 2)
            print(f"[{request_id}] ❌ Attempt {attempt + 1} failed ({latency}s): {error_str}")

            if "insufficient_quota" in error_str or "PerDay" in error_str:
                raise RuntimeError("วันนี้ใช้ quota หมดแล้ว กรุณาลองใหม่ภายหลัง")

            transient = any(kw in error_str.lower() for kw in [
                "truncated", "empty", "429", "rate_limit",
                "deadline", "timeout", "unavailable",
            ])
            if transient and attempt < max_retries - 1:
                wait = base_wait * (attempt + 1)
                print(f"[{request_id}] ⏳ Retrying in {wait}s...")
                time.sleep(wait)
                continue

            raise RuntimeError(f"OpenAI failed [{request_id}]: {error_str}")

    raise RuntimeError(f"OpenAI failed completely after {max_retries} retries [{request_id}]")


# ── Override helper ───────────────────────────────────────────

def _apply_measurement_overrides(result: dict, measurements: dict) -> dict:
    """
    หลัง AI ตอบกลับมาแล้ว — override ค่าที่ user วัดมาเองอีกครั้ง
    เพื่อป้องกัน AI เขียนทับค่าที่ถูกต้อง
    """
    fields = [
        "chest_cm", "neck_cm", "waist_cm",
        "body_length_cm", "back_length_cm", "leg_length_cm", "weight",
    ]
    for field in fields:
        val = _to_float(measurements.get(field))
        if val is not None:
            result[field] = val

    # recalc size จาก chest ที่ override แล้ว
    chest = result.get("chest_cm")
    if chest is not None:
        size = _calc_size(chest)
        result["size_category"]       = size
        result["size_recommendation"] = size
        result["size_ranges"]         = _size_ranges_from_chest(chest)

    result["analysis_method"] = "user_measurements"
    return result


# ── Main ──────────────────────────────────────────────────────

def analyze_cat(
    image_cat: str,
    measurements: Optional[dict] = None,
) -> dict:
    """
    Parameters
    ----------
    image_cat    : URL ของรูปแมว
    measurements : dict ค่าวัดจาก user (optional)
                   keys: chest_cm, neck_cm, waist_cm,
                         body_length_cm, back_length_cm, leg_length_cm, weight
    """
    has_measurements = bool(
        measurements and any(
            _to_float(measurements.get(k)) is not None
            for k in ("chest_cm", "neck_cm", "waist_cm",
                      "body_length_cm", "back_length_cm",
                      "leg_length_cm", "weight")
        )
    )

    # 1. Download ──────────────────────────────────────────────
    print(f"⬇️  Downloading: {image_cat}")
    try:
        resp = requests.get(image_cat, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot download image: {e}")

    image_bytes = resp.content
    mime_type   = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    print(f"✅ Downloaded ({len(image_bytes)/1024:.1f} KB) | mime={mime_type}"
          + (f" | measurements={list(k for k, v in (measurements or {}).items() if _to_float(v) is not None)}"
             if has_measurements else ""))

    # 2. Call OpenAI (prompt รวม measurement hint แล้ว) ────────
    raw_text = _call_openai_with_retry(
        image_bytes,
        mime_type,
        measurements=measurements if has_measurements else None,
    )

    # 3. Parse JSON ────────────────────────────────────────────
    try:
        ai_data: dict = _parse_json_robust(raw_text)
    except RuntimeError as e:
        _log_parse_error(raw_text, e)
        raise

    # 4. FAKE CAT GATE ─────────────────────────────────────────
    if not ai_data.get("is_cat", True):
        subject_type = ai_data.get("subject_type", "no_cat")
        message      = ai_data.get("message", "ไม่พบแมวในภาพ")
        print(f"🚫 Rejected: subject_type={subject_type} | {message}")
        return {
            "is_cat":       False,
            "subject_type": subject_type,
            "message":      message,
            "confidence":   ai_data.get("confidence", 0.95),
        }

    subject_type = ai_data.get("subject_type", "real_cat")
    if subject_type in FAKE_CAT_TYPES:
        fake_messages = {
            "cartoon":          "ตรวจพบภาพการ์ตูนหรือภาพวาด ไม่ใช่แมวจริง",
            "stuffed_toy":      "ตรวจพบตุ๊กตาหรือของเล่นรูปแมว ไม่ใช่แมวจริง",
            "figurine_model":   "ตรวจพบโมเดลหรือฟิกเกอร์รูปแมว ไม่ใช่แมวจริง",
            "human_in_costume": "ตรวจพบมนุษย์ที่แต่งตัวเป็นแมว ไม่ใช่แมวจริง",
            "cat_mask_prop":    "ตรวจพบหน้ากากแมวหรืออุปกรณ์ประกอบฉาก ไม่ใช่แมวจริง",
            "other_animal":     "ตรวจพบสัตว์ชนิดอื่น ไม่ใช่แมว",
            "no_cat":           "ไม่พบแมวในภาพ",
        }
        message = fake_messages.get(subject_type, "ไม่ใช่แมวจริง")
        print(f"🚫 Safety-net reject: subject_type={subject_type} | {message}")
        return {
            "is_cat":       False,
            "subject_type": subject_type,
            "message":      message,
            "confidence":   ai_data.get("confidence", 0.9),
        }

    # 5. Pydantic validation ───────────────────────────────────
    try:
        validated = CatAnalysisSchema.from_ai(ai_data)
    except Exception as e:
        _log_parse_error(raw_text, e)
        raise RuntimeError(f"AI response failed schema validation: {e}")

    # 6. Business logic ────────────────────────────────────────
    weight   = _to_float(validated.weight)
    chest_cm = _to_float(validated.chest_cm)
    body_len = _to_float(validated.body_length_cm)
    age: int = validated.age

    # size คำนวณจาก chest_cm ของ AI ก่อน
    size_category = _calc_size(chest_cm)
    age_category  = _calc_age_category(age)
    bmi           = _calc_bmi(weight, body_len)
    confidence    = validated.confidence if validated.confidence is not None else 0.5

    # 7. Build result ──────────────────────────────────────────
    result = {
        "is_cat":       True,
        "subject_type": "real_cat",
        "message":      "ok",
        "cat_color":    validated.cat_color,
        "breed":        validated.breed,
        "age":          age,
        "gender":       validated.gender,

        "weight":              weight,
        "size_category":       size_category,
        "size_recommendation": validated.size_recommendation or size_category,
        "size_ranges":         (
            validated.size_ranges.model_dump()
            if validated.size_ranges
            else _size_ranges_from_chest(chest_cm) if chest_cm else None
        ),

        "chest_cm":       chest_cm,
        "neck_cm":        _to_float(validated.neck_cm),
        "waist_cm":       _to_float(validated.waist_cm),
        "body_length_cm": body_len,
        "back_length_cm": _to_float(validated.back_length_cm),
        "leg_length_cm":  _to_float(validated.leg_length_cm),

        "age_category":               age_category,
        "body_condition_score":       validated.body_condition_score,
        "body_condition":             validated.body_condition,
        "body_condition_description": validated.body_condition_description,
        "bmi":                        bmi,
        "posture":                    validated.posture,

        "confidence":       confidence,
        "quality_flag":     validated.quality_flag,
        "analysis_version": "2.1",
        "analysis_method":  validated.analysis_method or "ai_estimation",
    }

    # 8. Hard override ─────────────────────────────────────────
    #    ถ้า user ส่ง measurements มา → บังคับใช้ค่าจริงทับผลของ AI
    #    ป้องกัน AI "ลืม" ค่าที่บอกไปใน prompt
    if has_measurements:
        result = _apply_measurement_overrides(result, measurements)
        print(
            f"📐 Measurements applied: chest={result.get('chest_cm')}cm "
            f"→ size={result.get('size_category')}"
        )

    # recalc bmi ถ้า weight / body_len ถูก override
    result["bmi"] = _calc_bmi(result.get("weight"), result.get("body_length_cm"))

    print(
        f"✅ Done: {result['cat_color']} | size={result['size_category']} "
        f"| chest={result['chest_cm']}cm | weight={result['weight']}kg "
        f"| method={result['analysis_method']} | confidence={result['confidence']}"
    )
    return result