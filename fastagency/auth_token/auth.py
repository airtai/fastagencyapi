import hashlib
import re
import secrets
import string
import uuid
from datetime import datetime, timedelta
from typing import Union

from fastapi import HTTPException
from pydantic import BaseModel

from ..db.base import DefaultDB


def generate_auth_token(length: int = 32) -> str:
    characters = string.ascii_letters + string.digits
    token = "".join(secrets.choice(characters) for _ in range(length))
    return token


def hash_auth_token(token: str) -> str:
    # Generate a random salt
    salt = secrets.token_bytes(16)

    # Combine salt and token
    salted_token = salt + token.encode("utf-8")

    # Use SHA-256 for hashing
    hashed_token = hashlib.sha256(salted_token).hexdigest()

    # Return the salt and hashed token
    return salt.hex() + ":" + hashed_token


def verify_auth_token(token: str, stored_hash: str) -> bool:
    if ":" not in stored_hash:
        return False
    # Split the stored hash into salt and hash
    salt, hash_value = stored_hash.split(":")

    # Convert salt back to bytes
    salt_bytes = bytes.fromhex(salt)

    # Combine salt and token
    salted_token = salt_bytes + token.encode("utf-8")

    # Hash the salted token
    computed_hash = hashlib.sha256(salted_token).hexdigest()

    # Compare the computed hash with the stored hash
    return computed_hash == hash_value


class DeploymentAuthToken(BaseModel):
    auth_token: str


async def parse_expiry(expiry: str) -> datetime:
    match = re.match(r"(\d+)([d])", expiry)
    if not match:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid expiry format - {expiry}; expected format: <number>d",
        )

    value, unit = match.groups()
    value = int(value)

    td = timedelta(days=value)

    d = datetime.utcnow()
    expires_at = d + td
    if expires_at <= d:
        raise HTTPException(status_code=400, detail="Expiry date cannot be in the past")
    return expires_at


async def create_deployment_auth_token(
    user_uuid: Union[str, uuid.UUID],
    deployment_uuid: Union[str, uuid.UUID],
    name: str = "Default deployment token",
    expiry: str = "99999d",
) -> DeploymentAuthToken:
    user = await DefaultDB.frontend().get_user(user_uuid=user_uuid)
    deployment = await DefaultDB.backend().find_model(model_uuid=deployment_uuid)

    if user["uuid"] != deployment["user_uuid"]:
        raise HTTPException(
            status_code=403, detail="User does not have access to this deployment"
        )
    expires_at = await parse_expiry(expiry)

    auth_token = generate_auth_token()
    hashed_token = hash_auth_token(auth_token)

    await DefaultDB.backend().create_auth_token(
        auth_token_uuid=uuid.uuid4(),
        name=name,
        user_uuid=user_uuid,
        deployment_uuid=deployment_uuid,
        hashed_auth_token=hashed_token,
        expiry=expiry,
        expires_at=expires_at,
    )

    return DeploymentAuthToken(auth_token=auth_token)
