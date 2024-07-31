from contextlib import contextmanager
from datetime import datetime
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)
from uuid import UUID


class KeyNotFoundError(ValueError):
    pass


class KeyExistsError(ValueError):
    pass


@runtime_checkable
class BackendDBProtocol(Protocol):
    async def create_model(
        self,
        model_uuid: Union[str, UUID],
        user_uuid: Union[str, UUID],
        type_name: str,
        model_name: str,
        json_str: str,
    ) -> Dict[str, Any]: ...

    async def find_model(self, model_uuid: Union[str, UUID]) -> Dict[str, Any]: ...

    async def find_many_model(
        self, user_uuid: Union[str, UUID], type_name: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...

    async def update_model(
        self,
        model_uuid: Union[str, UUID],
        user_uuid: Union[str, UUID],
        type_name: str,
        model_name: str,
        json_str: str,
    ) -> Dict[str, Any]: ...

    async def delete_model(self, model_uuid: Union[str, UUID]) -> Dict[str, Any]: ...

    async def create_auth_token(
        self,
        auth_token_uuid: Union[str, UUID],
        name: str,
        user_uuid: Union[str, UUID],
        deployment_uuid: Union[str, UUID],
        hashed_auth_token: str,
        expiry: str,
        expires_at: datetime,
    ) -> Dict[str, Any]: ...

    async def find_many_auth_token(
        self, user_uuid: Union[str, UUID], deployment_uuid: Union[str, UUID]
    ) -> List[Dict[str, Any]]: ...

    async def delete_auth_token(
        self,
        auth_token_uuid: Union[str, UUID],
        deployment_uuid: Union[str, UUID],
        user_uuid: Union[str, UUID],
    ) -> Dict[str, Any]: ...


@runtime_checkable
class FrontendDBProtocol(Protocol):
    async def get_user(self, user_uuid: Union[str, UUID]) -> Dict[str, Any]: ...

    async def _create_user(
        self, user_uuid: Union[str, UUID], email: str, username: str
    ) -> Union[str, UUID]: ...


class DefaultDB:
    _backend_db: Optional[BackendDBProtocol] = None
    _frontend_db: Optional[FrontendDBProtocol] = None

    @staticmethod
    @contextmanager
    def set(
        *,
        backend_db: BackendDBProtocol,
        frontend_db: FrontendDBProtocol,
    ) -> Generator[None, None, None]:
        old_backend_default = DefaultDB._backend_db
        old_frontend_default = DefaultDB._frontend_db
        try:
            DefaultDB._backend_db = backend_db
            DefaultDB._frontend_db = frontend_db
            yield
        finally:
            DefaultDB._backend_db = old_backend_default
            DefaultDB._frontend_db = old_frontend_default

    @staticmethod
    def backend() -> BackendDBProtocol:
        return DefaultDB._backend_db  # type: ignore[return-value]

    @staticmethod
    def frontend() -> FrontendDBProtocol:
        return DefaultDB._frontend_db  # type: ignore[return-value]
