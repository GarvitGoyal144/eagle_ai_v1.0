from motor.motor_asyncio import AsyncIOMotorClient

from app.config.settings import settings


class MongoDB:
    """
    MongoDB connection manager.

    Responsibilities:
    - Create MongoDB client
    - Connect on application startup
    - Disconnect on shutdown
    - Provide database instance
    """

    client: AsyncIOMotorClient | None = None

    database = None

    async def connect(self):
        """Connect to MongoDB."""
        self.client = AsyncIOMotorClient(settings.MONGO_URI)

        self.database = self.client[settings.DATABASE_NAME]

        print("✅ Connected to MongoDB")

    async def disconnect(self):
        """Close MongoDB connection."""

        if self.client:
            self.client.close()
            print("❌ MongoDB connection closed")


mongodb = MongoDB()