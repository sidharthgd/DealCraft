"""
Google Identity Platform authentication service.
"""
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import auth, credentials
from app.core.config import settings

logger = logging.getLogger(__name__)

class FirebaseAuthService:
    """Service for Firebase/Google Identity Platform authentication."""
    
    def __init__(self):
        self.app = None
        self.is_initialized = False
        
        logger.info(f"=== FIREBASE INIT DEBUG ===")
        logger.info(f"FIREBASE_PROJECT_ID: {settings.FIREBASE_PROJECT_ID}")
        logger.info(f"GCS_CREDENTIALS_PATH: {settings.GCS_CREDENTIALS_PATH}")
        
        # Initialize Firebase Admin SDK
        if settings.FIREBASE_PROJECT_ID:
            try:
                # Check if Firebase is already initialized
                try:
                    self.app = firebase_admin.get_app()
                    logger.info("Firebase Admin SDK already initialized")
                except ValueError:
                    logger.info("Firebase not initialized yet - attempting to initialize")
                    # Initialize with default credentials (for Cloud Run)
                    # or with explicit credentials if provided
                    if settings.GCS_CREDENTIALS_PATH:
                        logger.info("Using explicit credentials file")
                        cred = credentials.Certificate(settings.GCS_CREDENTIALS_PATH)
                        self.app = firebase_admin.initialize_app(cred, {
                            'projectId': settings.FIREBASE_PROJECT_ID
                        })
                    else:
                        logger.info("Using default credentials (Cloud Run)")
                        # Use default credentials (works on Cloud Run)
                        self.app = firebase_admin.initialize_app(options={
                            'projectId': settings.FIREBASE_PROJECT_ID
                        })
                    
                    logger.info(f"Firebase Admin SDK initialized for project: {settings.FIREBASE_PROJECT_ID}")
                
                self.is_initialized = True
                logger.info("Firebase initialization successful")
                logger.info("Firebase Admin SDK ready for token verification")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Firebase Admin SDK: {e}")
                logger.warning("Falling back to development mode - authentication disabled")
                self.is_initialized = False
        else:
            logger.warning("Firebase project ID not configured - authentication disabled")
            logger.warning("Running in development mode")
            self.is_initialized = False
        
        logger.info(f"Final is_initialized: {self.is_initialized}")
    
    def is_available(self) -> bool:
        """Check if Firebase authentication is available."""
        return self.is_initialized
    
    async def verify_token(self, id_token: str) -> Dict[str, Any]:
        """
        Verify a Firebase ID token.
        
        Args:
            id_token: The Firebase ID token to verify
            
        Returns:
            The decoded token payload
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        if not self.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(id_token)
            
            logger.info(f"Token verified for user: {decoded_token.get('uid')}")
            
            return {
                'uid': decoded_token.get('uid'),
                'email': decoded_token.get('email'),
                'email_verified': decoded_token.get('email_verified', False),
                'name': decoded_token.get('name'),
                'picture': decoded_token.get('picture'),
                'auth_time': decoded_token.get('auth_time'),
                'firebase': decoded_token.get('firebase', {})
            }
            
        except auth.ExpiredIdTokenError:
            logger.warning("Expired ID token provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except auth.RevokedIdTokenError:
            logger.warning("Revoked ID token provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        except auth.InvalidIdTokenError as e:
            logger.warning(f"Invalid ID token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication verification failed"
            )
    
    async def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Get user information by UID.
        
        Args:
            uid: The user's Firebase UID
            
        Returns:
            User information dictionary or None if not found
        """
        if not self.is_available():
            return None
        
        try:
            user_record = auth.get_user(uid)
            
            return {
                'uid': user_record.uid,
                'email': user_record.email,
                'email_verified': user_record.email_verified,
                'display_name': user_record.display_name,
                'photo_url': user_record.photo_url,
                'disabled': user_record.disabled,
                'creation_timestamp': user_record.user_metadata.creation_timestamp,
                'last_sign_in_timestamp': user_record.user_metadata.last_sign_in_timestamp
            }
            
        except auth.UserNotFoundError:
            logger.warning(f"User not found: {uid}")
            return None
        except Exception as e:
            logger.error(f"Failed to get user {uid}: {e}")
            return None
    
    async def create_custom_token(self, uid: str, additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a custom token for a user.
        
        Args:
            uid: The user's UID
            additional_claims: Optional additional claims to include
            
        Returns:
            Custom token string
        """
        if not self.is_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service not available"
            )
        
        try:
            custom_token = auth.create_custom_token(uid, additional_claims)
            logger.info(f"Custom token created for user: {uid}")
            return custom_token.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to create custom token for {uid}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create authentication token"
            )


# HTTP Bearer token scheme for FastAPI
security = HTTPBearer()

# Global auth service instance
_auth_service = None

def get_auth_service() -> FirebaseAuthService:
    """Get the Firebase authentication service."""
    global _auth_service
    if _auth_service is None:
        _auth_service = FirebaseAuthService()
    return _auth_service


# FastAPI dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    FastAPI dependency to get the current authenticated user.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Current user information
        
    Raises:
        HTTPException: If authentication fails
    """
    auth_service = get_auth_service()
    
    logger.info(f"=== AUTH DEBUG ===")
    logger.info(f"Auth service available: {auth_service.is_available()}")
    logger.info(f"Credentials provided: {credentials is not None}")
    
    if not auth_service.is_available():
        # Allow fallback ONLY if explicitly enabled (development)
        if settings.ALLOW_AUTH_FALLBACK:
            logger.warning("Authentication not configured - using default user (development fallback enabled)")
            return {
                'uid': 'default-user',
                'email': 'test@example.com',
                'email_verified': True,
                'name': 'Test User'
            }
        logger.error("Authentication service unavailable and fallback disabled")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    if credentials is None:
        logger.error("No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    
    # Extract token from credentials
    token = credentials.credentials
    logger.info(f"Token received: {token[:20]}..." if token else "No token")
    
    # Only verify token if auth service is available
    if auth_service.is_available():
        logger.info("Auth service available - verifying token")
        user_info = await auth_service.verify_token(token)

        # Ensure a corresponding User row exists in the database for FK integrity
        try:
            from sqlalchemy import select
            from app.core.database import AsyncSessionLocal
            from app.models.user import User

            async with AsyncSessionLocal() as session:
                existing = (await session.execute(select(User).where(User.id == user_info.get("uid")))).scalar_one_or_none()
                if existing is None:
                    provisioned = User(
                        id=user_info.get("uid"),
                        email=user_info.get("email") or f"{user_info.get('uid')}@users.noreply",
                        name=user_info.get("name") or (user_info.get("email") or user_info.get("uid"))
                    )
                    session.add(provisioned)
                    await session.commit()
        except Exception as e:
            logger.warning(f"Failed to ensure user exists: {e}")

        return user_info
    else:
        # Fallback to default user even with token
        logger.warning("Auth service unavailable - using default user despite token")
        return {
            'uid': 'default-user',
            'email': 'test@example.com',
            'email_verified': True,
            'name': 'Test User'
        }


# Optional dependency for authentication (allows unauthenticated access)
async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = security) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to optionally get the current authenticated user.
    Returns None if no authentication is provided.
    
    Args:
        credentials: Optional HTTP authorization credentials
        
    Returns:
        Current user information or None
    """
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None