from fastapi import APIRouter, HTTPException

import asyncpg




from app.db.database import get_db_pool


from fastapi import APIRouter

router = APIRouter()

@router.get("/home-advertiment")
async def get_home_advertiment():
    """
    Get clothing items for home page advertisement
    
    **No Authentication Required**
    """
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
    SELECT 
        id,
        uuid,
        image_url,
        images,
        clothing_name,
        description,
        category,
        size_category,
        price,
        discount_price,
        CASE 
            WHEN discount_price IS NOT NULL AND discount_price < price 
            THEN ROUND(((price - discount_price) / price) * 100, 0)
            ELSE NULL
        END as discount_percent,
        gender,
        clothing_like,
        clothing_seller,
        stock,
        breed,
        created_at
    FROM cat_clothing
    WHERE is_active = true
    ORDER BY RANDOM()
    LIMIT 10;
            """
            
            rows = await connection.fetch(query)
            
            if not rows:
                return []
            
            result = [dict(row) for row in rows]
            return result
            
    except asyncpg.PostgresError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/home-advertiment/{item_id}")
async def get_home_advertiment_detail(item_id: int):
    """
    Get single clothing item detail
    """
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
                  SELECT 
        id,
        uuid,
        image_url,
        images,
        clothing_name,
        description,
        category,
        size_category,
        price,
        discount_price,
        CASE 
            WHEN discount_price IS NOT NULL AND discount_price < price 
            THEN CONCAT(ROUND(((price - discount_price) / price) * 100, 0), '%')
            ELSE NULL
        END as discount_percent,
        gender,
        clothing_like,
        clothing_seller,
        stock,
        breed,
        created_at
    FROM cat_clothing
    WHERE id = $1
    AND is_active = true
            """
            
            row = await connection.fetchrow(query, item_id)
            
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Clothing item with id {item_id} not found"
                )
            
            return dict(row)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/clothing-shop/like")
async def get_clothing_shop_like():
    """
     Get all clothing items for clothing shop page (liked view)
    """
    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
               	    SELECT 
                    id,
                    uuid,
                    image_url,
                    images,
                    clothing_name,
                    description,
                    category,
                    size_category,
                    price,
                    discount_price,
                    CASE 
                        WHEN discount_price IS NOT NULL AND discount_price < price 
                        THEN ROUND(((price - discount_price) / price) * 100, 0)
                        ELSE NULL
                    END as discount_percent,
                    gender,
                    clothing_like,
                    clothing_seller,
                    stock,
                    breed,
                    created_at
                FROM cat_clothing
                WHERE is_active = true
                ORDER BY clothing_like DESC
                LIMIT 10;
            """
            
            rows = await connection.fetch(query)
            
            if not rows:
                return []
            
            result = [dict(row) for row in rows]
            return result
            
    except asyncpg.PostgresError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/clothing-shop/seller")
async def get_clothing_shop_seller():
    """
        Get all clothing items for clothing shop page (seller view)
    """

    try:
        pool = await get_db_pool()
        
        async with pool.acquire() as connection:
            query = """
                SELECT 
                    id,
                    uuid,
                    image_url,
                    images,
                    clothing_name,
                    description,
                    category,
                    size_category,
                    price,
                    discount_price,
                    CASE 
                        WHEN discount_price IS NOT NULL AND discount_price < price 
                        THEN ROUND(((price - discount_price) / price) * 100, 0)
                        ELSE NULL
                    END as discount_percent,
                    gender,
                    clothing_like,
                    clothing_seller,
                    stock,
                    breed,
                    created_at
                FROM cat_clothing
                WHERE is_active = true
                ORDER BY clothing_seller DESC
                LIMIT 10;
            """
            
            rows = await connection.fetch(query)
            
            if not rows:
                return []
            
            result = [dict(row) for row in rows]
            return result
            
    except asyncpg.PostgresError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
@router.get("/notifications/messages")
async def get_notifications_messages():
    try: 
        pool = await get_db_pool()

        async with pool.acquire() as connection:
            query = """ 
            SELECT
                id,
                uuid,
                image_url,
                images,
                clothing_name,
                description,
                category,
                size_category,
                CONCAT(price, ' THB') as price,
                CONCAT(discount_price, ' THB') as discount_price,
                CONCAT(ROUND(((price - discount_price) / price) * 100, 0), '%') as discount_percent,
                gender,
                clothing_like,
                clothing_seller,
                stock,
                breed,
                created_at
                FROM cat_clothing
                WHERE is_active = true
                AND discount_price IS NOT NULL
                AND discount_price < price
                AND discount_price > 0 
                ORDER BY discount_percent DESC
                LIMIT 5       
                """
            rows = await connection.fetch(query)

            if not rows:
                return []
            result = [dict(row) for row in rows]
            return result
    except asyncpg.PostgresError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
@router.get("/notifications/messages/{item_id}")
async def get_notifications_messages_detail(item_id: int):
  
    try:
        pool = await get_db_pool()

        async with pool.acquire() as connection:
            query = """
            SELECT
                id,
                uuid,
                image_url,
                images,
                clothing_name,
                description,
                category,
                size_category,
                CONCAT(price, ' THB') as price,
                CONCAT(discount_price, ' THB') as discount_price,
                CONCAT(ROUND(((price - discount_price) / price) * 100, 0), '%') as discount_percent,
                gender,
                clothing_like,
                clothing_seller,
                stock,
                breed,
                created_at
                FROM cat_clothing
                WHERE id = $1
  				AND is_active = true
 				AND discount_price IS NOT NULL
                AND discount_price > 0 
  				AND discount_price < price   
               
            """

            row = await connection.fetchrow(query, item_id)

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Clothing item with id {item_id} not found"
                )

            return dict(row)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@router.get("/notifications/news")
async def get_notifications_news():
    try: 
        pool = await get_db_pool()

        async with pool.acquire() as connection:
            query = """ 
             SELECT
                id,
                uuid,
                image_url,
                images,
                clothing_name,
                description,
                category,
                size_category,
                CONCAT(price, ' THB') AS price,
                gender,
                clothing_like,
                clothing_seller,
                stock,
                breed,
                created_at
                FROM cat_clothing
                WHERE is_active = true
                AND discount_price = 0
                ORDER BY created_at DESC
                LIMIT 5;

                """
            rows = await connection.fetch(query)

            if not rows:
                return []
            result = [dict(row) for row in rows]
            return result
    except asyncpg.PostgresError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
@router.get("/notifications/news/{item_id}")
async def get_notifications_news_detail(item_id: int):
    try:
        pool = await get_db_pool()

        async with pool.acquire() as connection:
            query = """
                SELECT
                    id,
                    uuid,
                    image_url,
                    images,
                    clothing_name,
                    description,
                    category,
                    size_category,
                    price,
                    gender,
                    clothing_like,
                    clothing_seller,
                    stock,
                    breed,
                    created_at
                FROM cat_clothing
                WHERE id = $1
                 AND discount_price = 0
                  AND is_active = true
            """

            row = await connection.fetchrow(query, item_id)

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Clothing item with id {item_id} not found"
                )

            return dict(row)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )






# ============================================
# CAT DETECTION & PREDICTION ENDPOINTS (Auth Required)
# ============================================

# @router.post("/cats/detect", response_model=dict)
# async def detect_cat_endpoint(
#     file: UploadFile = File(...),
#     user: dict = Depends(verify_firebase_token)
# ):
#     """
#     Detect if uploaded image contains a cat
    
#     **Authentication:** Firebase ID Token required
    
#     **Body:**
#     - file: Image file (JPEG, PNG, WebP)
    
#     **Returns:**
#     - is_cat: bool
#     - is_cartoon: bool 
#     - is_blur: bool
#     - is_dark: bool
#     - distance: str ("close", "medium", "far")
#     - confidence: float
#     - image_url: str (Cloudinary URL)
#     - thumbnail_url: str (Cloudinary thumbnail)
#     """
#     try: 
#         # 1. Upload image to Cloudinary
#         upload_result = await upload_image_to_cloudinary(file, user["uid"])
#         image_url = upload_result["secure_url"]
        
#         # 2. Detect cat using ChatGPT Vision
#         detection_result = await detect_cat(image_url)
        
#         # 3. Combine results
#         response_data = {
#             **detection_result,
#             "image_url": upload_result["secure_url"],
#             "thumbnail_url": upload_result["thumbnail_url"],
#             "image_width": upload_result["width"],
#             "image_height": upload_result["height"]
#         }
        
#         # 4. Return response
#         return success_response(
#             data=response_data,
#             message="Cat detection completed"
#         )
    
#     except HTTPException as e:
#         raise e
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Detection failed: {str(e)}"
#         )


# @router.post("/cats/predict", response_model=dict)
# async def predict_cat_endpoint(
#     file: UploadFile = File(...),
#     user: dict = Depends(verify_firebase_token)
# ):
#     """
#     Predict cat information from uploaded image
    
#     **Authentication:** Firebase ID Token required
    
#     **Body:**
#     - file: Image file (JPEG, PNG, WebP)
    
#     **Returns:**
#     - breed: str
#     - weight_estimate: str
#     - age_estimate: str
#     - color: str
#     - size_category: str (XS, S, M, L, XL)
#     - chest_cm: float
#     - neck_cm: float
#     - body_length_cm: float
#     - description: str
#     - confidence: float
#     - image_url: str (Cloudinary URL)
#     - thumbnail_url: str (Cloudinary thumbnail)
#     """
#     try: 
#         # 1. Upload Image to Cloudinary
#         upload_result = await upload_image_to_cloudinary(file, user["uid"])
#         image_url = upload_result["secure_url"]
        
#         # 2. Predict cat info using ChatGPT Vision
#         prediction_result = await predict_cat(image_url)
        
#         # 3. Combine results
#         response_data = {
#             **prediction_result,
#             "image_url": upload_result["secure_url"],
#             "thumbnail_url": upload_result["thumbnail_url"],
#             "image_width": upload_result["width"],
#             "image_height": upload_result["height"]
#         }
        
#         # 4. Return response
#         return success_response(
#             data=response_data,
#             message="Cat prediction completed"
#         )
    
#     except HTTPException as e:
#         raise e
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Prediction failed: {str(e)}"
#         )


# @router.get("/cats/me", response_model=dict)
# async def get_current_user(
#     user: dict = Depends(verify_firebase_token)
# ):
#     """
#     Get current user info from Firebase token 
    
#     **Authentication:** Firebase ID Token required
    
#     **Returns:**
#     - uid: Firebase user ID
#     - email: User email
#     - name: User display name
#     - picture: User profile picture URL
#     """
#     return success_response(
#         data=user,
#         message="User info retrieved"
#     )

