"""Database package — re-exports the MongoDB singleton."""

from app.database.mongodb import mongodb

__all__ = ["mongodb"]