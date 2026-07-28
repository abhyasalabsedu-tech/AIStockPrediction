from fastapi import Header, HTTPException
from app.core.config import get_settings

settings = get_settings()


def require_admin(authorization: str = Header(default="")):
    """Simple bearer-token check for admin-triggered endpoints (train/ingest).
    Not enterprise-grade auth -- just enough to stop a random internet visitor
    from triggering training or burning your Gemini quota via ingest."""
    expected = f"Bearer {settings.admin_token}"
    if not authorization or authorization != expected:
        raise HTTPException(401, "Invalid or missing admin token.")
    return True
