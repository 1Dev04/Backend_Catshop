from fastapi import APIRouter, Depends, HTTPException
from app.db.database import get_db_pool
from app.auth.dependencies import verify_firebase_token
from fastapi import APIRouter

router = APIRouter()

@router.post("/auth/login")
async def login(
    user_info: dict = Depends(verify_firebase_token),
    db = Depends(get_db_pool),
):
    firebase_uid = user_info["firebase_uid"]
    firebase_email = user_info.get("email")
    
    print(f"📝 Logging in user: {firebase_email} (UID: {firebase_uid})")

    try:
        async with db.acquire() as conn:
          
            await conn.execute(
                """
                INSERT INTO person_login_catshop (firebase_uid, firebase_user, firebase_login)
                VALUES ($1, $2, true)
                ON CONFLICT (firebase_uid) 
                DO UPDATE SET 
                    firebase_login = true,
                    firebase_user = $2,
                    updated_at = NOW()
                """,
                firebase_uid,
                firebase_email,
            )
            print(f"✅ User {firebase_email} logged in successfully")
            
    except Exception as e:
        print(f"❌ DB error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail="Database unavailable"
        )

    return {
        "status": "success",
        "message": "Login successful",
        "user": {
            "uid": firebase_uid,
            "email": firebase_email,
        }
    }