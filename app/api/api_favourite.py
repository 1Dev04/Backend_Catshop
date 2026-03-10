from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Optional
import asyncpg
from uuid import UUID

from app.db.database import get_db_pool

router = APIRouter()

# ============================================================================
# Pydantic Models (ใช้ firebase_uid แทน user_id)
# ============================================================================

class FavouriteItem(BaseModel):
    firebase_uid: str  # ✅ เปลี่ยนจาก user_id → firebase_uid
    clothing_uuid: str

class PaginationRequest(BaseModel):
    firebase_uid: str  # ✅ เปลี่ยนจาก user_id
    page: int = 1
    limit: int = 10

# ============================================================================
# GET: ดึงรายการโปรดทั้งหมด
# ============================================================================

@router.get("/get/person-favourite/{firebase_uid}")
async def get_person_favourite(firebase_uid: str):
    """
    Get favourite list for a specific user by firebase_uid
    """
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
                SELECT 
                    uf.id as favourite_id,
                    uf.firebase_uid,
                    uf.clothing_uuid,
                    uf.created_at,
                    c.uuid,
                    c.clothing_name,
                    c.price,
                    c.discount_price,
                    c.stock,
                    c.image_url,
                    c.images,
                    c.category,
                    c.size_category,
                    c.gender,
                    c.breed,
                    c.cat_color,
                    c.description,
                    c.images
                FROM user_favorites uf
                INNER JOIN cat_clothing c ON uf.clothing_uuid = c.uuid
                WHERE uf.firebase_uid = $1
                ORDER BY uf.created_at DESC
            """
            rows = await connection.fetch(query, firebase_uid)
            
            if not rows:
                return []
            
            result = []
            for row in rows:
                item = dict(row)
                if item.get('uuid'):
                    item['uuid'] = str(item['uuid'])
                if item.get('clothing_uuid'):
                    item['clothing_uuid'] = str(item['clothing_uuid'])
                result.append(item)
            
            return result
            
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# GET: นับจำนวนรายการโปรด
# ============================================================================

@router.get("/get/person-favourite/count/{firebase_uid}")
async def get_favourite_count(firebase_uid: str):
    """Get total count of favourite items"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
                SELECT COUNT(*) as total
                FROM user_favorites
                WHERE firebase_uid = $1
            """
            result = await connection.fetchval(query, firebase_uid)
            
            return {"firebase_uid": firebase_uid, "total": result}
            
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# POST: เพิ่มรายการโปรด
# ============================================================================

@router.post("/post/person-favourite")
async def post_person_favourite(data: FavouriteItem):
    """Add an item to user's favourites"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            # เช็คว่ามีอยู่แล้วหรือไม่
            check_query = """
                SELECT id FROM user_favorites
                WHERE firebase_uid = $1 AND clothing_uuid = $2
            """
            existing = await connection.fetchval(
                check_query, 
                data.firebase_uid, 
                UUID(data.clothing_uuid)
            )
            
            if existing:
                raise HTTPException(status_code=400, detail="Item already in favourites")
            
            # เพิ่มรายการใหม่
            insert_query = """
                INSERT INTO user_favorites (firebase_uid, clothing_uuid)
                VALUES ($1, $2)
                RETURNING id, firebase_uid, clothing_uuid, created_at
            """
            result = await connection.fetchrow(
                insert_query, 
                data.firebase_uid, 
                UUID(data.clothing_uuid)
            )
            
            response = dict(result)
            if response.get('clothing_uuid'):
                response['clothing_uuid'] = str(response['clothing_uuid'])
            
            return {"message": "Added to favourites successfully", "data": response}
            
    except HTTPException:
        raise
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# DELETE: ลบรายการโปรด
# ============================================================================

@router.delete("/del/person-favourite")
async def del_person_favourite(
    firebase_uid: str = Body(...), 
    clothing_uuid: str = Body(...)
):
    """Remove an item from user's favourites"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            delete_query = """
                DELETE FROM user_favorites
                WHERE firebase_uid = $1 AND clothing_uuid = $2
                RETURNING id
            """
            deleted_id = await connection.fetchval(
                delete_query, 
                firebase_uid, 
                UUID(clothing_uuid)
            )
            
            if not deleted_id:
                raise HTTPException(status_code=404, detail="Favourite item not found")
            
            return {"message": "Removed from favourites successfully", "deleted_id": deleted_id}
            
    except HTTPException:
        raise
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# GET: เช็คว่าอยู่ใน Favourite หรือไม่
# ============================================================================

@router.get("/get/check-favourite/{firebase_uid}/{clothing_uuid}")
async def check_favourite(firebase_uid: str, clothing_uuid: str):
    """Check if an item is in user's favourites"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
                SELECT id FROM user_favorites
                WHERE firebase_uid = $1 AND clothing_uuid = $2
            """
            result = await connection.fetchval(query, firebase_uid, UUID(clothing_uuid))
            
            return {
                "firebase_uid": firebase_uid,
                "clothing_uuid": clothing_uuid,
                "is_favourite": result is not None
            }
            
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 