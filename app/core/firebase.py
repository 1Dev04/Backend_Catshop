import os
import firebase_admin
from firebase_admin import credentials

# Public Google OAuth2 endpoint — ไม่ใช่ secret, เป็น standard constant
GOOGLE_TOKEN_URI: str = "https://oauth2.googleapis.com/token"  # nosec B105


def init_firebase() -> None:
    """Initialize Firebase Admin SDK"""

    if firebase_admin._apps:
        print("✅ Firebase already initialized")
        return

    print("🔍 Reading Firebase credentials from environment...")

    project_id    = os.getenv("FIREBASE_PROJECT_ID")
    client_email  = os.getenv("FIREBASE_CLIENT_EMAIL")
    private_key   = os.getenv("FIREBASE_PRIVATE_KEY")

    print(f"   - FIREBASE_PROJECT_ID:    {'✅' if project_id    else '❌ MISSING'}")
    print(f"   - FIREBASE_CLIENT_EMAIL:  {'✅' if client_email  else '❌ MISSING'}")
    print(f"   - FIREBASE_PRIVATE_KEY:   {'✅' if private_key   else '❌ MISSING'}")

    if not all([project_id, client_email, private_key]):
        missing = [
            name for name, val in {
                "FIREBASE_PROJECT_ID":   project_id,
                "FIREBASE_CLIENT_EMAIL": client_email,
                "FIREBASE_PRIVATE_KEY":  private_key,
            }.items()
            if not val
        ]

        error_msg = f"Missing Firebase ENV variables: {', '.join(missing)}"
        print(f"❌ {error_msg}")
        print("\n💡 How to fix:")
        print("   1. Go to Render Dashboard → Your Service → Environment")
        print("   2. Add the missing environment variables above")
        print("   3. Get values from Firebase Console:")
        print("      https://console.firebase.google.com/")
        print("      → Project Settings → Service Accounts → Generate new private key")

        raise RuntimeError(error_msg)

    try:
        cleaned_private_key = (
            private_key
            .replace("\\n", "\n")
            .strip('"')
            .strip("'")
        )

        print("🔍 Creating Firebase credentials...")

        cred = credentials.Certificate({
            "type":         "service_account",
            "project_id":   project_id,
            "client_email": client_email,
            "private_key":  cleaned_private_key,
            "token_uri":    GOOGLE_TOKEN_URI,   # nosec B105 — public OAuth2 endpoint
        })

        print("🔍 Initializing Firebase Admin SDK...")
        firebase_admin.initialize_app(cred)

        print(f"✅ Firebase initialized successfully for project: {project_id}")

    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        raise