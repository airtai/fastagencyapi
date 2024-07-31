import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from uuid import UUID

from asyncer import asyncify
from fastapi import BackgroundTasks, HTTPException

from fastagency.saas_app_generator import (
    InvalidFlyTokenError,
    InvalidGHTokenError,
    SaasAppGenerator,
)

from .auth_token.auth import create_deployment_auth_token
from .db.base import DefaultDB
from .models.base import Model, ObjectReference
from .models.registry import Registry

T = TypeVar("T", bound=Model)


async def get_model_by_uuid(model_uuid: Union[str, UUID]) -> Model:
    model_dict = await DefaultDB.backend().find_model(model_uuid=model_uuid)

    registry = Registry.get_default()
    model = registry.validate(
        type=model_dict["type_name"],
        name=model_dict["model_name"],
        model=model_dict["json_str"],
    )

    return model


async def get_model_by_ref(model_ref: ObjectReference) -> Model:
    return await get_model_by_uuid(model_ref.uuid)


async def validate_tokens_and_create_gh_repo(
    model: Dict[str, Any],
    model_uuid: str,
) -> SaasAppGenerator:
    found_gh_token = await DefaultDB.backend().find_model(
        model_uuid=model["gh_token"]["uuid"]
    )
    found_fly_token = await DefaultDB.backend().find_model(
        model_uuid=model["fly_token"]["uuid"]
    )

    found_gh_token_uuid = found_gh_token["json_str"]["gh_token"]
    found_fly_token_uuid = found_fly_token["json_str"]["fly_token"]

    saas_app = SaasAppGenerator(
        fly_api_token=found_fly_token_uuid,
        github_token=found_gh_token_uuid,
        app_name=model["name"],
        repo_name=model["repo_name"],
        fly_app_name=model["fly_app_name"],
        fastagency_deployment_uuid=model_uuid,
    )

    saas_app.validate_tokens()
    saas_app.create_new_repository()
    return saas_app


async def deploy_saas_app(
    saas_app: SaasAppGenerator,
    user_uuid: str,
    model_uuid: str,
    type_name: str,
    model_name: str,
) -> None:
    deployment_auth_token = await create_deployment_auth_token(user_uuid, model_uuid)
    saas_app.deployment_auth_token = deployment_auth_token.auth_token
    saas_app.developer_uuid = user_uuid

    await asyncify(saas_app.execute)()

    found_model = await DefaultDB.backend().find_model(model_uuid=model_uuid)
    found_model["json_str"]["app_deploy_status"] = "completed"
    await DefaultDB.backend().update_model(
        model_uuid=found_model["uuid"],
        user_uuid=user_uuid,
        type_name=type_name,
        model_name=model_name,
        json_str=json.dumps(found_model["json_str"]),
    )


async def add_model_to_user(
    user_uuid: str,
    type_name: str,
    model_name: str,
    model_uuid: str,
    model: Dict[str, Any],
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    try:
        registry = Registry.get_default()
        validated_model = registry.validate(type_name, model_name, model)

        validated_model_dict = validated_model.model_dump()
        validated_model_json = validated_model.model_dump_json()
        saas_app = None

        if type_name == "deployment":
            saas_app = await validate_tokens_and_create_gh_repo(
                validated_model_dict, model_uuid
            )

            validated_model_dict["app_deploy_status"] = "inprogress"
            validated_model_dict["gh_repo_url"] = saas_app.gh_repo_url

            updated_validated_model_dict = json.loads(validated_model_json)
            updated_validated_model_dict["app_deploy_status"] = "inprogress"
            updated_validated_model_dict["gh_repo_url"] = saas_app.gh_repo_url
            validated_model_json = json.dumps(updated_validated_model_dict)

        await DefaultDB.frontend().get_user(user_uuid=user_uuid)
        await DefaultDB.backend().create_model(
            model_uuid=model_uuid,
            user_uuid=user_uuid,
            type_name=type_name,
            model_name=model_name,
            json_str=validated_model_json,
        )

        if saas_app is not None:
            background_tasks.add_task(
                deploy_saas_app,
                saas_app,
                user_uuid,
                model_uuid,
                type_name,
                model_name,
            )

        return validated_model_dict

    except InvalidGHTokenError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    except InvalidFlyTokenError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    except Exception as e:
        msg = "Oops! Something went wrong. Please try again later."
        raise HTTPException(status_code=422, detail=msg) from e


async def create_model(
    cls: Type[T],
    type_name: str,
    user_uuid: Union[str, UUID],
    background_tasks: Optional[BackgroundTasks] = None,
    **kwargs: Any,
) -> Tuple[UUID, Dict[str, Any]]:
    model = cls(**kwargs)
    model_uuid = uuid.uuid4()

    validated_model = await add_model_to_user(
        user_uuid=str(user_uuid),
        type_name=type_name,
        model_name=cls.__name__,  # type: ignore [attr-defined]
        model_uuid=str(model_uuid),
        model=model.model_dump(),
        background_tasks=background_tasks,  # type: ignore[arg-type]
    )
    return model_uuid, validated_model


async def create_model_ref(
    cls: Type[T],
    type_name: str,
    user_uuid: Union[str, UUID],
    background_tasks: Optional[BackgroundTasks] = None,
    **kwargs: Any,
) -> ObjectReference:
    model_uuid, _ = await create_model(
        cls,
        type_name,
        user_uuid,
        background_tasks,
        **kwargs,
    )

    model_ref = cls.get_reference_model()(uuid=model_uuid)

    return model_ref


async def get_all_models_for_user(
    user_uuid: Union[str, UUID],
    type_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    models = await DefaultDB.backend().find_many_model(
        user_uuid=user_uuid, type_name=type_name
    )

    return models  # type: ignore[no-any-return]


async def create_autogen(
    model_ref: ObjectReference,
    user_uuid: Union[str, UUID],
    **kwargs: Any,
) -> Any:
    user_id = UUID(user_uuid) if isinstance(user_uuid, str) else user_uuid
    model_id = (
        UUID(model_ref.uuid)  # type: ignore[arg-type]
        if isinstance(model_ref.uuid, str)
        else model_ref.uuid
    )
    model = await get_model_by_ref(model_ref)

    return await model.create_autogen(model_id=model_id, user_id=user_id, **kwargs)
