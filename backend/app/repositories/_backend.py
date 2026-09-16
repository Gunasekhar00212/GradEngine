"""Shared repository backend helpers for MongoDB and local fallback storage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from os import getenv
from typing import Any

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover - optional dependency
    MongoClient = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deep_merge(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(target)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class MemoryCollectionBackend:
    """Small in-memory fallback so the app keeps running without MongoDB."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}

    def upsert(self, key_field: str, key_value: str, document: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(document)
        payload[key_field] = key_value
        self._documents[key_value] = payload
        return deepcopy(payload)

    def find_one(self, key_field: str, key_value: str) -> dict[str, Any] | None:
        document = self._documents.get(key_value)
        return deepcopy(document) if document is not None else None

    def merge(self, key_field: str, key_value: str, updates: dict[str, Any]) -> dict[str, Any]:
        current = self._documents.get(key_value, {key_field: key_value})
        merged = _deep_merge(current, updates)
        self._documents[key_value] = merged
        return deepcopy(merged)

    def delete(self, key_field: str, key_value: str) -> None:
        self._documents.pop(key_value, None)


@dataclass(slots=True)
class MongoCollectionBackend:
    """Thin adapter over a PyMongo collection."""

    collection: Any

    def upsert(self, key_field: str, key_value: str, document: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(document)
        payload[key_field] = key_value
        payload.setdefault("updated_at", _now_iso())
        self.collection.replace_one({key_field: key_value}, payload, upsert=True)
        return deepcopy(payload)

    def find_one(self, key_field: str, key_value: str) -> dict[str, Any] | None:
        document = self.collection.find_one({key_field: key_value})
        if not document:
            return None
        document.pop("_id", None)
        return document

    def merge(self, key_field: str, key_value: str, updates: dict[str, Any]) -> dict[str, Any]:
        current = self.find_one(key_field, key_value) or {key_field: key_value}
        merged = _deep_merge(current, updates)
        merged.setdefault("updated_at", _now_iso())
        self.collection.replace_one({key_field: key_value}, merged, upsert=True)
        return deepcopy(merged)

    def delete(self, key_field: str, key_value: str) -> None:
        self.collection.delete_one({key_field: key_value})


_BACKEND_CACHE: dict[str, MemoryCollectionBackend | MongoCollectionBackend] = {}


def get_collection_backend(collection_name: str) -> MemoryCollectionBackend | MongoCollectionBackend:
    """Return a Mongo-backed collection when available, otherwise a memory fallback."""

    cached = _BACKEND_CACHE.get(collection_name)
    if cached is not None:
        return cached

    uri = getenv("MONGODB_URI", "").strip()
    database_name = getenv("MONGODB_DATABASE", "gradengine").strip() or "gradengine"

    if MongoClient is None or not uri:
        backend: MemoryCollectionBackend | MongoCollectionBackend = MemoryCollectionBackend()
        _BACKEND_CACHE[collection_name] = backend
        return backend

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=1500)
        collection = client[database_name][collection_name]
        backend = MongoCollectionBackend(collection)
    except Exception:
        backend = MemoryCollectionBackend()

    _BACKEND_CACHE[collection_name] = backend
    return backend
