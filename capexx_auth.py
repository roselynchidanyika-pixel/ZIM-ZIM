"""
CAPEXX AI AGENT - Demonstration Authentication
==============================================
Royal-blue and white demo login.

- Seeded ADMIN:   username `FraiserXX`   password `M251232@1`
- Seeded ADMIN:   username `admin`       password `admin123`
- Any visitor can register a username + password -> stored as a GUEST.
- Status is displayed as GUEST or ADMIN after login.

SECURITY NOTE (demonstration grade only):
- Passwords are never stored as plain text; they are salted SHA-256 hashes.
- Real credentials must never live in source code. For production use a real
  identity provider (OAuth, SSO, database with password hashing).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from typing import Any, Dict, List, Optional

USERS_FILE = "users.json"

# Seeded demo admins. These are DEMO credentials - NOT for production.
SEED_ADMINS = {
    "FraiserXX": "M251232@1",
    "admin": "admin123",
}


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _new_salt() -> str:
    return secrets.token_hex(16)


def _default_users() -> Dict[str, Dict[str, Any]]:
    users: Dict[str, Dict[str, Any]] = {}
    for u, pw in SEED_ADMINS.items():
        salt = _new_salt()
        users[u] = {"role": "ADMIN", "salt": salt,
                    "hash": _hash(pw, salt),
                    "created": "seeded"}
    return users


def load_users() -> Dict[str, Dict[str, Any]]:
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("users file corrupt")
        # ensure seeded admins always exist
        for u, pw in SEED_ADMINS.items():
            if u not in data:
                salt = _new_salt()
                data[u] = {"role": "ADMIN", "salt": salt,
                           "hash": _hash(pw, salt), "created": "seeded"}
        return data
    except Exception:
        return _default_users()


def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as fh:
            json.dump(users, fh, indent=2)
    except Exception as ex:
        # Non-fatal: filesystem may be read-only on some hosts.
        print(f"[capexx_auth] could not persist users: {ex}")


def register(username: str, password: str) -> Dict[str, Any]:
    username = (username or "").strip()
    if len(username) < 3:
        return {"ok": False, "message": "Username must be at least 3 characters."}
    if not password or len(password) < 4:
        return {"ok": False, "message": "Password must be at least 4 characters."}
    users = load_users()
    if username in users:
        if username in SEED_ADMINS:
            return {"ok": False, "message": "That username is reserved."}
        return {"ok": False, "message": "Username already exists."}
    salt = _new_salt()
    users[username] = {"role": "GUEST", "salt": salt,
                       "hash": _hash(password, salt),
                       "created": "registered"}
    save_users(users)
    return {"ok": True, "message": f"Guest account '{username}' created.",
            "role": "GUEST"}


def verify(username: str, password: str) -> Dict[str, Any]:
    username = (username or "").strip()
    users = load_users()
    rec = users.get(username)
    if not rec:
        return {"ok": False, "message": "Unknown username."}
    if _hash(password or "", rec.get("salt", "")) != rec.get("hash"):
        return {"ok": False, "message": "Incorrect password."}
    return {"ok": True, "message": "Authenticated.",
            "role": rec.get("role", "GUEST")}


def usernames() -> List[str]:
    return sorted(load_users().keys())