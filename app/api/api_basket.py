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

class BasketItem(BaseModel):
    firebase_uid: str 
    clothing_uuid: str
    quantity: int = 1

class UpdateQuantity(BaseModel):
    firebase_uid: str  
    clothing_uuid: str
    quantity: int

# ============================================================================
# GET: ดึงตะกร้าสินค้าทั้งหมด
# ============================================================================

@router.get("/get/person-baskets/{firebase_uid}")
async def get_person_baskets(firebase_uid: str):
    """Get shopping basket for a specific user by firebase_uid"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
                SELECT 
                    ub.id as basket_id,
                    ub.firebase_uid,
                    ub.clothing_uuid,
                    ub.quantity,
                    ub.created_at,
                    ub.updated_at,
                    c.uuid,
                    c.clothing_name,
                    c.price,
                    c.discount_price,
                    c.stock,
                    c.image_url,
                    c.category,
                    c.size_category,
                    c.gender,
                    c.breed,
                    c.description,
                    c.images,
                    CASE 
                        WHEN c.discount_price > 0 THEN c.discount_price * ub.quantity
                        ELSE c.price * ub.quantity
                    END as total_price
                FROM user_baskets ub
                INNER JOIN cat_clothing c ON ub.clothing_uuid = c.uuid
                WHERE ub.firebase_uid = $1
                ORDER BY ub.created_at DESC
            """
            rows = await connection.fetch(query, firebase_uid)
            
            if not rows:
                return {
                    "items": [],
                    "summary": {
                        "total_items": 0,
                        "total_quantity": 0,
                        "total_price": 0.0
                    }
                }
            
            items = []
            for row in rows:
                item = dict(row)
                if item.get('uuid'):
                    item['uuid'] = str(item['uuid'])
                if item.get('clothing_uuid'):
                    item['clothing_uuid'] = str(item['clothing_uuid'])
                items.append(item)
            
            total_items = len(items)
            total_quantity = sum(item['quantity'] for item in items)
            total_price = sum(float(item['total_price']) for item in items)
            
            return {
                "items": items,
                "summary": {
                    "total_items": total_items,
                    "total_quantity": total_quantity,
                    "total_price": round(total_price, 2)
                }
            }
            
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# GET: นับจำนวนสินค้าในตะกร้า
# ============================================================================

@router.get("/get/person-baskets/count/{firebase_uid}")
async def get_basket_count(firebase_uid: str):
    """Get total count and quantity of items in basket"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
                SELECT 
                    COUNT(*) as total_items,
                    COALESCE(SUM(quantity), 0) as total_quantity
                FROM user_baskets
                WHERE firebase_uid = $1
            """
            result = await connection.fetchrow(query, firebase_uid)
            
            return {
                "firebase_uid": firebase_uid,
                "total_items": result['total_items'],
                "total_quantity": result['total_quantity']
            }
            
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# POST: เพิ่มสินค้าลงตะกร้า
# ============================================================================

@router.post("/post/person-baskets")
async def post_person_baskets(data: BasketItem):
    """Add an item to user's shopping basket"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            # เช็คว่ามีอยู่แล้วหรือไม่
            check_query = """
                SELECT id, quantity FROM user_baskets
                WHERE firebase_uid = $1 AND clothing_uuid = $2
            """
            existing = await connection.fetchrow(
                check_query, 
                data.firebase_uid, 
                UUID(data.clothing_uuid)
            )
            
            if existing:
                # ถ้ามีอยู่แล้ว ให้เพิ่มจำนวน
                update_query = """
                    UPDATE user_baskets
                    SET quantity = quantity + $1, updated_at = NOW()
                    WHERE firebase_uid = $2 AND clothing_uuid = $3
                    RETURNING id, firebase_uid, clothing_uuid, quantity, created_at, updated_at
                """
                result = await connection.fetchrow(
                    update_query,
                    data.quantity,
                    data.firebase_uid,
                    UUID(data.clothing_uuid)
                )
                
                response = dict(result)
                if response.get('clothing_uuid'):
                    response['clothing_uuid'] = str(response['clothing_uuid'])
                
                return {"message": "Updated quantity in basket", "data": response}
            else:
                # เพิ่มรายการใหม่
                insert_query = """
                    INSERT INTO user_baskets (firebase_uid, clothing_uuid, quantity)
                    VALUES ($1, $2, $3)
                    RETURNING id, firebase_uid, clothing_uuid, quantity, created_at, updated_at
                """
                result = await connection.fetchrow(
                    insert_query, 
                    data.firebase_uid, 
                    UUID(data.clothing_uuid),
                    data.quantity
                )
                
                response = dict(result)
                if response.get('clothing_uuid'):
                    response['clothing_uuid'] = str(response['clothing_uuid'])
                
                return {"message": "Added to basket successfully", "data": response}
            
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# PUT: อัพเดทจำนวนสินค้า
# ============================================================================

@router.put("/put/person-baskets/quantity")
async def update_basket_quantity(data: UpdateQuantity):
    """Update quantity of an item in basket"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            if data.quantity <= 0:
                # ลบออกถ้าจำนวนเป็น 0
                delete_query = """
                    DELETE FROM user_baskets
                    WHERE firebase_uid = $1 AND clothing_uuid = $2
                    RETURNING id
                """
                deleted_id = await connection.fetchval(
                    delete_query,
                    data.firebase_uid,
                    UUID(data.clothing_uuid)
                )
                
                if not deleted_id:
                    raise HTTPException(status_code=404, detail="Item not found in basket")
                
                return {"message": "Item removed from basket", "deleted_id": deleted_id}
            else:
                # อัพเดทจำนวน
                update_query = """
                    UPDATE user_baskets
                    SET quantity = $1, updated_at = NOW()
                    WHERE firebase_uid = $2 AND clothing_uuid = $3
                    RETURNING id, firebase_uid, clothing_uuid, quantity, created_at, updated_at
                """
                result = await connection.fetchrow(
                    update_query,
                    data.quantity,
                    data.firebase_uid,
                    UUID(data.clothing_uuid)
                )
                
                if not result:
                    raise HTTPException(status_code=404, detail="Item not found in basket")
                
                response = dict(result)
                if response.get('clothing_uuid'):
                    response['clothing_uuid'] = str(response['clothing_uuid'])
                
                return {"message": "Quantity updated successfully", "data": response}
            
    except HTTPException:
        raise
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# DELETE: ลบสินค้าออกจากตะกร้า
# ============================================================================

@router.delete("/del/person-baskets")
async def del_person_baskets(
    firebase_uid: str = Body(...), 
    clothing_uuid: str = Body(...)
):
    """Remove an item from user's basket"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            delete_query = """
                DELETE FROM user_baskets
                WHERE firebase_uid = $1 AND clothing_uuid = $2
                RETURNING id
            """
            deleted_id = await connection.fetchval(
                delete_query, 
                firebase_uid, 
                UUID(clothing_uuid)
            )
            
            if not deleted_id:
                raise HTTPException(status_code=404, detail="Basket item not found")
            
            return {"message": "Removed from basket successfully", "deleted_id": deleted_id}
            
    except HTTPException:
        raise
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# DELETE: ล้างตะกร้าทั้งหมด
# ============================================================================

@router.delete("/del/person-baskets/clear/{firebase_uid}")
async def clear_all_baskets(firebase_uid: str):
    """Clear all items from user's basket"""
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            delete_query = """
                DELETE FROM user_baskets
                WHERE firebase_uid = $1
                RETURNING id
            """
            deleted_ids = await connection.fetch(delete_query, firebase_uid)
            
            return {
                "message": "Basket cleared successfully",
                "deleted_count": len(deleted_ids)
            }
            
    except asyncpg.PostgresError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")