from pymongo import MongoClient

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

    client: MongoClient | None = None
    database = None

    async def connect(self):
        """Connect to MongoDB."""
        try:
            import certifi
            self.client = MongoClient(settings.MONGO_URI, tlsCAFile=certifi.where())
            # Quick ping test
            self.client.admin.command('ping')
        except Exception:
            try:
                self.client = MongoClient(settings.MONGO_URI, tlsAllowInvalidCertificates=True)
            except Exception:
                self.client = MongoClient(settings.MONGO_URI)

        self.database = self.client[settings.DATABASE_NAME]

        print("✅ Connected to MongoDB")

    async def disconnect(self):
        """Close MongoDB connection."""

        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
            self.database = None
            print("❌ MongoDB connection closed")

    def is_connected(self) -> bool:
        """Ping MongoDB to verify live connection."""
        if self.client is None:
            return False
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False


mongodb = MongoDB()