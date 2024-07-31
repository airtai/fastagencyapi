import uuid
from datetime import datetime
from typing import Any, Dict, Union
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import fastagency.app
import fastagency.auth_token.auth
import fastagency.db
import fastagency.db.inmemory
import fastagency.db.prisma
from fastagency.app import app
from fastagency.auth_token.auth import (
    create_deployment_auth_token,
    generate_auth_token,
    hash_auth_token,
    parse_expiry,
    verify_auth_token,
)

client = TestClient(app)


def test_generate_auth_token() -> None:
    token = generate_auth_token()
    assert isinstance(token, str)
    assert len(token) == 32


def test_hash_auth_token() -> None:
    token = generate_auth_token()
    hashed_token = hash_auth_token(token)
    assert isinstance(hashed_token, str)
    assert len(hashed_token) == 97
    assert ":" in hashed_token


def test_verify_auth_token() -> None:
    token = generate_auth_token()
    hashed_token = hash_auth_token(token)
    assert verify_auth_token(token, hashed_token)
    assert not verify_auth_token(token, "wrong_hash")
    assert not verify_auth_token("wrong_token", hashed_token)
    assert not verify_auth_token("wrong_token", "wrong_hash")


@pytest.mark.asyncio()
async def test_parse_expiry() -> None:
    expiry = await parse_expiry("1d")
    assert expiry is not None
    assert isinstance(expiry, datetime)
    assert expiry > datetime.utcnow()


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "expiry_str, expected",  # noqa: PT006
    [
        ("1", "Invalid expiry format - 1; expected format: <number>d"),
        ("1h", "Invalid expiry format - 1h; expected format: <number>d"),
        ("1w", "Invalid expiry format - 1w; expected format: <number>d"),
        ("1m", "Invalid expiry format - 1m; expected format: <number>d"),
        ("1y", "Invalid expiry format - 1y; expected format: <number>d"),
        ("0d", "Expiry date cannot be in the past"),
        ("-1d", "Invalid expiry format - -1d; expected format: <number>d"),
    ],
)
async def test_parse_expiry_with_invalid_expiry(expiry_str: str, expected: str) -> None:
    with pytest.raises(HTTPException) as e:
        await parse_expiry(expiry_str)
    assert e.value.status_code == 400
    assert e.value.detail == expected


@pytest.mark.db()
@pytest.mark.asyncio()
async def test_create_deployment_token(
    user_uuid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment_uuid = uuid.uuid4()

    async def mock_find_model(*args: Any, **kwargs: Any) -> Dict[str, Union[str, UUID]]:
        return {
            "user_uuid": user_uuid,
            "uuid": deployment_uuid,
        }

    monkeypatch.setattr(
        fastagency.db.inmemory.InMemoryBackendDB,
        "find_model",
        mock_find_model,
    )

    token = await create_deployment_auth_token(user_uuid, deployment_uuid)
    assert isinstance(token.auth_token, str)
    assert len(token.auth_token) == 32, token.auth_token


@pytest.mark.db()
@pytest.mark.asyncio()
async def test_create_deployment_token_with_wrong_user_uuid(
    user_uuid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment_uuid = uuid.uuid4()

    async def mock_find_model(*args: Any, **kwargs: Any) -> Dict[str, Union[str, UUID]]:
        return {
            "user_uuid": "random_wrong_uuid",
            "uuid": deployment_uuid,
        }

    monkeypatch.setattr(
        fastagency.db.inmemory.InMemoryBackendDB,
        "find_model",
        mock_find_model,
    )

    with pytest.raises(HTTPException) as e:
        await create_deployment_auth_token(user_uuid, deployment_uuid)

    assert e.value.status_code == 403
    assert e.value.detail == "User does not have access to this deployment"


@pytest.mark.db()
@pytest.mark.asyncio()
async def test_create_deployment_auth_token_route(
    user_uuid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment_uuid = uuid.uuid4()

    async def mock_find_model(*args: Any, **kwargs: Any) -> Dict[str, Union[str, UUID]]:
        return {
            "user_uuid": user_uuid,
            "uuid": deployment_uuid,
        }

    monkeypatch.setattr(
        fastagency.db.inmemory.InMemoryBackendDB,
        "find_model",
        mock_find_model,
    )

    response = client.post(
        f"/user/{user_uuid}/deployment/{deployment_uuid}",
        json={"name": "Test token", "expiry": "99d"},
    )
    assert response.status_code == 200
    assert "auth_token" in response.json()
    assert response.json()["auth_token"] is not None


@pytest.mark.db()
@pytest.mark.asyncio()
async def test_get_all_deployment_auth_tokens(
    user_uuid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment_uuid = uuid.uuid4()

    async def mock_find_model(*args: Any, **kwargs: Any) -> Dict[str, Union[str, UUID]]:
        return {
            "user_uuid": user_uuid,
            "uuid": deployment_uuid,
        }

    monkeypatch.setattr(
        fastagency.db.inmemory.InMemoryBackendDB,
        "find_model",
        mock_find_model,
    )

    response = client.post(
        f"/user/{user_uuid}/deployment/{deployment_uuid}",
        json={"name": "Test token", "expiry": "99d"},
    )
    assert response.status_code == 200

    monkeypatch.setattr(
        fastagency.db.inmemory.InMemoryBackendDB,
        "find_model",
        mock_find_model,
    )
    response = client.get(f"/user/{user_uuid}/deployment/{deployment_uuid}")
    assert response.status_code == 200
    response_json = response.json()
    assert len(response_json) == 1
    assert "uuid" in response_json[0]
    assert response_json[0]["name"] == "Test token"
    assert response_json[0]["expiry"] == "99d"


@pytest.mark.db()
@pytest.mark.asyncio()
async def test_delete_deployment_auth_token(
    user_uuid: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment_uuid = uuid.uuid4()

    async def mock_find_model(*args: Any, **kwargs: Any) -> Dict[str, Union[str, UUID]]:
        return {
            "user_uuid": user_uuid,
            "uuid": deployment_uuid,
        }

    monkeypatch.setattr(
        fastagency.db.inmemory.InMemoryBackendDB,
        "find_model",
        mock_find_model,
    )

    response = client.post(
        f"/user/{user_uuid}/deployment/{deployment_uuid}",
        json={"name": "Test token", "expiry": "99d"},
    )
    assert response.status_code == 200

    monkeypatch.setattr(
        fastagency.db.inmemory.InMemoryBackendDB,
        "find_model",
        mock_find_model,
    )
    response = client.get(f"/user/{user_uuid}/deployment/{deployment_uuid}")
    assert len(response.json()) == 1
    auth_token_uuid = str(response.json()[0]["uuid"])

    response = client.delete(
        url=f"/user/{user_uuid}/deployment/{deployment_uuid}/{auth_token_uuid}"
    )
    assert response.status_code == 200
