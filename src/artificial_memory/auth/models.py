from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    username: str
    email: str
    hashed_password: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_login: datetime | None = None


class APIKey(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    user_id: int
    name: str
    key_hash: str
    key_prefix: str  # First 8 chars for identification
    scopes: list[str] = Field(default_factory=list)
    is_active: bool = True
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: datetime | None = None


class Session(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str | None = None
    user_id: int
    token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.now)
    ip_address: str | None = None
    user_agent: str | None = None


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password with salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return pwd_hash.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify a password against hash."""
    pwd_hash, _ = hash_password(password, salt)
    return pwd_hash == hashed


def generate_api_key() -> tuple[str, str]:
    """Generate API key and its hash."""
    key = f"am_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash


def hash_api_key(key: str) -> str:
    """Hash an API key."""
    return hashlib.sha256(key.encode()).hexdigest()
