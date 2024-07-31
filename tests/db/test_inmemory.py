import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Union
from uuid import UUID

import pytest

import fastagency.db
import fastagency.db.inmemory
from fastagency.db.base import DefaultDB, KeyNotFoundError
from fastagency.db.inmemory import InMemoryBackendDB, InMemoryFrontendDB
from fastagency.models.llms.azure import AzureOAIAPIKey


@pytest.mark.asyncio()
class TestInMemoryFrontendDB:
    async def test_set(self) -> None:
        frontend_db = InMemoryFrontendDB()
        backend_db = InMemoryBackendDB()
        with DefaultDB.set(backend_db=backend_db, frontend_db=frontend_db):
            assert DefaultDB._frontend_db == frontend_db
            assert DefaultDB._backend_db == backend_db

    async def test_db(self) -> None:
        frontend_db = InMemoryFrontendDB()
        backend_db = InMemoryBackendDB()
        with DefaultDB.set(backend_db=backend_db, frontend_db=frontend_db):
            assert DefaultDB.frontend() == frontend_db
            assert DefaultDB.backend() == backend_db

    async def test_create_user_get_user(self) -> None:
        frontend_db = InMemoryFrontendDB()

        random_id = random.randint(1, 1_000_000)
        generated_uuid = uuid.uuid4()
        email = f"user{random_id}@airt.ai"
        username = f"user{random_id}"

        user_uuid = await frontend_db._create_user(generated_uuid, email, username)
        assert user_uuid == generated_uuid

        user = await frontend_db.get_user(user_uuid)
        assert user["uuid"] == str(user_uuid)
        assert user["email"] == email
        assert user["username"] == username

    async def test_user_exception(self) -> None:
        frontend_db = InMemoryFrontendDB()
        user_uuid = uuid.uuid4()
        with pytest.raises(KeyNotFoundError) as e:
            await frontend_db.get_user(user_uuid)
        assert f"user_uuid {user_uuid} not found" == str(e.value)


@pytest.mark.db()
@pytest.mark.asyncio()
class TestInMemoryBackendDB:
    async def test_model_CRUD(self) -> None:  # noqa: N802
        # Setup
        frontend_db = InMemoryFrontendDB()
        backend_db = InMemoryBackendDB()
        random_id = random.randint(1, 1_000_000)
        user_uuid = await frontend_db._create_user(
            uuid.uuid4(), f"user{random_id}@airt.ai", f"user{random_id}"
        )
        model_uuid = uuid.uuid4()
        azure_oai_api_key = AzureOAIAPIKey(api_key="whatever", name="who cares?")

        # Tests
        model = await backend_db.create_model(
            user_uuid=user_uuid,
            model_uuid=model_uuid,
            type_name="secret",
            model_name="AzureOAIAPIKey",
            json_str=azure_oai_api_key.model_dump_json(),
        )
        assert model["uuid"] == str(model_uuid)
        assert model["user_uuid"] == str(user_uuid)
        assert model["type_name"] == "secret"
        assert model["model_name"] == "AzureOAIAPIKey"
        assert model["json_str"] == azure_oai_api_key.model_dump()

        found_model = await backend_db.find_model(model_uuid)
        assert found_model["uuid"] == str(model_uuid)

        many_model = await backend_db.find_many_model(user_uuid)
        assert len(many_model) == 1
        assert many_model[0]["uuid"] == str(model_uuid)

        updated_model = await backend_db.update_model(
            model_uuid=model_uuid,
            user_uuid=user_uuid,
            type_name="secret",
            model_name="AzureOAIAPIKey2",
            json_str=azure_oai_api_key.model_dump_json(),
        )
        assert updated_model["uuid"] == str(model_uuid)
        assert updated_model["model_name"] == "AzureOAIAPIKey2"

        deleted_model = await backend_db.delete_model(model_uuid)
        assert deleted_model["uuid"] == str(model_uuid)

    async def test_auth_token_CRUD(self, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: N802
        # Setup
        frontend_db = InMemoryFrontendDB()
        backend_db = InMemoryBackendDB()
        random_id = random.randint(1, 1_000_000)
        user_uuid = await frontend_db._create_user(
            uuid.uuid4(), f"user{random_id}@airt.ai", f"user{random_id}"
        )
        deployment_uuid = uuid.uuid4()
        auth_token_uuid = uuid.uuid4()

        async def mock_find_model(
            *args: Any, **kwargs: Any
        ) -> Dict[str, Union[str, UUID]]:
            return {
                "user_uuid": user_uuid,
                "uuid": deployment_uuid,
            }

        monkeypatch.setattr(
            fastagency.db.inmemory.InMemoryBackendDB,
            "find_model",
            mock_find_model,
        )

        # Tests
        auth_token = await backend_db.create_auth_token(
            auth_token_uuid=auth_token_uuid,
            name="Test token",
            user_uuid=user_uuid,
            deployment_uuid=deployment_uuid,
            hashed_auth_token="whatever",
            expiry="99d",
            expires_at=datetime.utcnow() + timedelta(days=99),
        )
        assert auth_token["uuid"] == str(auth_token_uuid)
        assert auth_token["name"] == "Test token"

        many_auth_token = await backend_db.find_many_auth_token(
            user_uuid, deployment_uuid
        )
        assert len(many_auth_token) == 1
        assert many_auth_token[0]["uuid"] == str(auth_token_uuid)

        deleted_auth_token = await backend_db.delete_auth_token(
            auth_token_uuid, deployment_uuid, user_uuid
        )
        assert deleted_auth_token["uuid"] == str(auth_token_uuid)

    async def test_model_exception(self) -> None:
        backend_db = InMemoryBackendDB()
        model_uuid = uuid.uuid4()
        user_uuid = uuid.uuid4()
        with pytest.raises(KeyNotFoundError) as e:
            await backend_db.find_model(model_uuid)
        assert f"model_uuid {model_uuid} not found" == str(e.value)

        with pytest.raises(KeyNotFoundError) as e:
            await backend_db.update_model(
                model_uuid=model_uuid,
                user_uuid=user_uuid,
                type_name="secret",
                model_name="AzureOAIAPIKey2",
                json_str="[]",
            )
        assert f"model_uuid {model_uuid} not found" == str(e.value)

        with pytest.raises(KeyNotFoundError) as e:
            await backend_db.delete_model(model_uuid)
        assert f"model_uuid {model_uuid} not found" == str(e.value)

    async def test_auth_token_exception(self) -> None:
        backend_db = InMemoryBackendDB()
        auth_token_uuid = uuid.uuid4()
        deployment_uuid = uuid.uuid4()
        user_uuid = uuid.uuid4()

        with pytest.raises(KeyNotFoundError) as e:
            await backend_db.delete_auth_token(
                auth_token_uuid, deployment_uuid, user_uuid
            )
        assert f"auth_token_uuid {auth_token_uuid} not found" == str(e.value)
