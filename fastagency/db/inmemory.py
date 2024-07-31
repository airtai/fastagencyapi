import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from .base import BackendDBProtocol, FrontendDBProtocol, KeyNotFoundError


class InMemoryBackendDB(BackendDBProtocol):
    def __init__(self) -> None:
        """In memory backend database."""
        self._models: List[Dict[str, Any]] = []
        self._auth_tokens: List[Dict[str, Any]] = []

    async def create_model(
        self,
        model_uuid: Union[str, UUID],
        user_uuid: Union[str, UUID],
        type_name: str,
        model_name: str,
        json_str: str,
    ) -> Dict[str, Any]:
        model = {
            "uuid": str(model_uuid),
            "user_uuid": str(user_uuid),
            "type_name": type_name,
            "model_name": model_name,
            "json_str": json.loads(json_str),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        self._models.append(model)
        return model

    async def find_model(self, model_uuid: Union[str, UUID]) -> Dict[str, Any]:
        for model in self._models:
            if model["uuid"] == str(model_uuid):
                return model
        raise KeyNotFoundError(f"model_uuid {model_uuid} not found")

    async def find_many_model(
        self, user_uuid: Union[str, UUID], type_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [model for model in self._models if model["user_uuid"] == str(user_uuid)]

    async def update_model(
        self,
        model_uuid: Union[str, UUID],
        user_uuid: Union[str, UUID],
        type_name: str,
        model_name: str,
        json_str: str,
    ) -> Dict[str, Any]:
        for model in self._models:
            if model["uuid"] == str(model_uuid):
                model["type_name"] = type_name
                model["model_name"] = model_name
                model["json_str"] = json.loads(json_str)
                model["updated_at"] = datetime.now()
                return model
        raise KeyNotFoundError(f"model_uuid {model_uuid} not found")

    async def delete_model(self, model_uuid: Union[str, UUID]) -> Dict[str, Any]:
        for model in self._models:
            if model["uuid"] == str(model_uuid):
                self._models.remove(model)
                return model
        raise KeyNotFoundError(f"model_uuid {model_uuid} not found")

    async def create_auth_token(
        self,
        auth_token_uuid: Union[str, UUID],
        name: str,
        user_uuid: Union[str, UUID],
        deployment_uuid: Union[str, UUID],
        hashed_auth_token: str,
        expiry: str,
        expires_at: datetime,
    ) -> Dict[str, Any]:
        auth_token = {
            "uuid": str(auth_token_uuid),
            "name": name,
            "user_uuid": str(user_uuid),
            "deployment_uuid": str(deployment_uuid),
            "hashed_auth_token": hashed_auth_token,
            "expiry": expiry,
            "expires_at": expires_at,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        self._auth_tokens.append(auth_token)
        return auth_token

    async def find_many_auth_token(
        self, user_uuid: Union[str, UUID], deployment_uuid: Union[str, UUID]
    ) -> List[Dict[str, Any]]:
        return [
            auth_token
            for auth_token in self._auth_tokens
            if auth_token["user_uuid"] == str(user_uuid)
            and auth_token["deployment_uuid"] == str(deployment_uuid)
        ]

    async def delete_auth_token(
        self,
        auth_token_uuid: Union[str, UUID],
        deployment_uuid: Union[str, UUID],
        user_uuid: Union[str, UUID],
    ) -> Dict[str, Any]:
        for auth_token in self._auth_tokens:
            if (
                auth_token["uuid"] == str(auth_token_uuid)
                and auth_token["user_uuid"] == str(user_uuid)
                and auth_token["deployment_uuid"] == str(deployment_uuid)
            ):
                self._auth_tokens.remove(auth_token)
                return auth_token
        raise KeyNotFoundError(f"auth_token_uuid {auth_token_uuid} not found")


class InMemoryFrontendDB(FrontendDBProtocol):
    def __init__(self) -> None:
        """In memory frontend database."""
        self._users: List[Dict[str, Any]] = []

    async def get_user(self, user_uuid: Union[str, UUID]) -> Any:
        for user in self._users:
            if user["uuid"] == str(user_uuid):
                return user
        raise KeyNotFoundError(f"user_uuid {user_uuid} not found")

    async def _create_user(
        self, user_uuid: Union[str, UUID], email: str, username: str
    ) -> Union[str, UUID]:
        """Only to create user in testing."""
        self._users.append(
            {
                "uuid": str(user_uuid),
                "email": email,
                "username": username,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
        )
        return user_uuid
