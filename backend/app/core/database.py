from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings
from sqlalchemy import text  # Added for raw SQL execution


class Base(DeclarativeBase):
    pass


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,
    future=True,
    pool_size=2,  # Only 2 connections per instance (instead of 5)
    max_overflow=0,
    # Add connection pool settings for Cloud SQL
    pool_pre_ping=True,
    pool_recycle=300,
    pool_timeout=30,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Alias for consistency with import expectations
async def get_async_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    async with engine.begin() as conn:
        # Import all models here to ensure they are created
        from app.models import user, deal, document, memo, tag
        await conn.run_sync(Base.metadata.create_all)

        # --- Schema patches -------------------------------------------------
        # Ensure newer columns that are not covered by simple `create_all()`
        # exist in already-created tables. This makes local development and
        # CI environments resilient when the schema evolves but Alembic
        # migrations have not been run yet.
        # Currently required for the "category" column on the "documents"
        # table which causes 500 errors on upload if missing.
        try:
            await conn.execute(text("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS category VARCHAR NULL
            """))
        except Exception as e:
            # Log but don't fail startup – the app can continue using the
            # existing schema while the developer investigates.
            print(f"[init_db] Failed to ensure 'category' column exists: {e}")

        # Ensure newer 'storage_path' column exists for documents
        try:
            await conn.execute(text(
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS storage_path VARCHAR NULL
                """
            ))
        except Exception as e:
            print(f"[init_db] Failed to ensure 'storage_path' column exists: {e}")
    
    # Create default user if it doesn't exist
    async with AsyncSessionLocal() as session:
        try:
            from app.models.user import User
            from sqlalchemy import select
            
            # Check if default user exists
            result = await session.execute(select(User).where(User.id == "default-user"))
            existing_user = result.scalar_one_or_none()
            
            if not existing_user:
                # Create default user
                default_user = User(
                    id="default-user",
                    email="default@dealcraft.ai",
                    name="Default User"
                )
                session.add(default_user)
                await session.commit()
                print("Created default user")
            else:
                print("Default user already exists")
        except Exception as e:
            await session.rollback()
            print(f"Error creating default user: {e}")
            raise 
