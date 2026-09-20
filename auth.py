"""RFC Securities authentication.

A simple hashed local user file (``rfc_users.json``) is used so the platform
can run standalone. Seed demo accounts are created on first use.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

USERS_FILE = Path(__file__).resolve().parents[1] / "rfc_users.json"
DEMO_ACCOUNTS = [
    ("rfc", "rfc2024", "Admin"),
    ("demo", "rfc2024", "Analyst"),
]


def _hash(password: str, salt: str) -> str:
    return hashlib.sha256((salt + "::" + password).encode("utf-8")).hexdigest()


def _load() -> dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _ensure_seed() -> None:
    users = _load()
    changed = False
    for user, pw, role in DEMO_ACCOUNTS:
        if user in users:
            continue
        salt = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
        users[user] = {"salt": salt, "hash": _hash(pw, salt), "role": role}
        changed = True
    if changed:
        _save(users)


def verify(username: str, password: str) -> dict:
    _ensure_seed()
    users = _load()
    rec = users.get((username or "").strip())
    if not rec:
        return {"ok": False, "message": "Unknown username. Register first or "
                                        "use the demo credentials shown."}
    if _hash(password or "", rec["salt"]) == rec["hash"]:
        return {"ok": True, "username": (username or "").strip(),
                "role": rec.get("role", "Guest")}
    return {"ok": False, "message": "Incorrect password."}


def register(username: str, password: str) -> dict:
    _ensure_seed()
    user = (username or "").strip()
    if not user or not password:
        return {"ok": False, "message": "Username and password are required."}
    if len(password) < 6:
        return {"ok": False, "message": "Password must be at least 6 characters."}
    users = _load()
    if user in users:
        return {"ok": False, "message": "That username is already taken."}
    salt = hashlib.sha256(os.urandom(16)).hexdigest()[:16]
    users[user] = {"salt": salt, "hash": _hash(password, salt),
                   "role": "Guest"}
    _save(users)
    return {"ok": True, "message": f"Account '{user}' created — sign in now."}