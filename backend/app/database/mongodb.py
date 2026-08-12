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
        mongo_kwargs = {
            "tlsAllowInvalidCertificates": True,
            "serverSelectionTimeoutMS": 5000,
        }
        try:
            import certifi
            mongo_kwargs["tlsCAFile"] = certifi.where()
        except Exception:
            pass

        try:
            self.client = MongoClient(settings.MONGO_URI, **mongo_kwargs)
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