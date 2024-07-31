import json
import logging
from os import environ
from typing import (
    Annotated,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from uuid import UUID

import httpx
import yaml
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Path
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, ValidationError

from .auth_token.auth import DeploymentAuthToken, create_deployment_auth_token
from .db.base import DefaultDB, KeyNotFoundError
from .db.prisma import fastapi_lifespan
from .helpers import (
    add_model_to_user,
    create_model,
    get_all_models_for_user,
)
from .models.registry import Registry, Schemas
from .models.toolboxes.toolbox import Toolbox

logging.basicConfig(level=logging.INFO)


app = FastAPI(lifespan=fastapi_lifespan)


@app.middleware("http")
async def handle_keynotfounderror_middleware(
    request: Request, call_next: Callable[[Request], Coroutine[Any, Any, Response]]
) -> Response:
    try:
        return await call_next(request)
    except KeyNotFoundError as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})


@app.get("/models/schemas")
async def get_models_schemas() -> Schemas:
    schemas = Registry.get_default().get_schemas()
    return schemas


async def validate_toolbox(toolbox: Toolbox) -> None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(toolbox.openapi_url)  # type: ignore[arg-type]
    except Exception as e:
        raise HTTPException(status_code=422, detail="OpenAPI URL is invalid") from e

    if not (resp.status_code >= 200 and resp.status_code < 400):
        raise HTTPException(
            status_code=422, detail=f"OpenAPI URL returns error code {resp.status_code}"
        )

    try:
        if "yaml" in toolbox.openapi_url or "yml" in toolbox.openapi_url:  # type: ignore [operator]
            openapi_spec = yaml.safe_load(resp.text)
        else:
            openapi_spec = resp.json()

        if "openapi" not in openapi_spec:
            raise HTTPException(
                status_code=422,
                detail="OpenAPI URL does not contain a valid OpenAPI spec",
            )
    except Exception as e:
        raise HTTPException(
            status_code=422, detail="OpenAPI URL does not contain a valid OpenAPI spec"
        ) from e


@app.post("/models/{type}/{name}/validate")
async def validate_model(type: str, name: str, model: Dict[str, Any]) -> Dict[str, Any]:
    try:
        validated_model = Registry.get_default().validate(type, name, model)
        if isinstance(validated_model, Toolbox):
            await validate_toolbox(validated_model)
        return validated_model.model_dump()
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json())) from e


@app.post("/user/{user_uuid}/models/secret/{name}/{model_uuid}/validate")
async def validate_secret_model(
    user_uuid: UUID, name: str, model_uuid: UUID, model: Dict[str, Any]
) -> Dict[str, Any]:
    type: str = "secret"

    found_model = await DefaultDB.backend().find_model(model_uuid=model_uuid)
    if "api_key" in found_model["json_str"]:
        model["api_key"] = found_model["json_str"]["api_key"]
    try:
        validated_model = Registry.get_default().validate(type, name, model)
        return validated_model.model_dump()
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=json.loads(e.json())) from e


# new routes by Harish


async def mask(value: str) -> str:
    return value[:3] + "*" * (len(value) - 7) + value[-4:]


@app.get("/user/{user_uuid}/models")
async def get_all_models(
    user_uuid: str,
    type_name: Optional[str] = None,
) -> List[Any]:
    ret_val_without_mask = await get_all_models_for_user(
        user_uuid=user_uuid, type_name=type_name
    )

    ret_val = []
    for model in ret_val_without_mask:
        if model["type_name"] == "secret":
            for k in ["api_key", "gh_token", "fly_token"]:
                if k in model["json_str"]:
                    model["json_str"][k] = await mask(model["json_str"][k])
        ret_val.append(model)

    return ret_val  # type: ignore[no-any-return]


@app.post("/user/{user_uuid}/models/{type_name}/{model_name}/{model_uuid}")
async def add_model(
    user_uuid: str,
    type_name: str,
    model_name: str,
    model_uuid: str,
    model: Dict[str, Any],
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    return await add_model_to_user(
        user_uuid=user_uuid,
        type_name=type_name,
        model_name=model_name,
        model_uuid=model_uuid,
        model=model,
        background_tasks=background_tasks,
    )


async def create_toolbox_for_new_user(user_uuid: Union[str, UUID]) -> Dict[str, Any]:
    await DefaultDB.frontend().get_user(user_uuid=user_uuid)  # type: ignore[arg-type]

    domain = environ.get("DOMAIN", "localhost")
    toolbox_openapi_url = (
        "https://weather.tools.staging.fastagency.ai/openapi.json"
        if "staging" in domain or "localhost" in domain
        else "https://weather.tools.fastagency.ai/openapi.json"
    )

    # Check if default weather toolbox already exists
    models = await get_all_models_for_user(user_uuid=user_uuid, type_name="toolbox")
    if models:
        raise HTTPException(status_code=400, detail="Weather toolbox already exists")

    _, validated_model = await create_model(
        cls=Toolbox,
        type_name="toolbox",
        user_uuid=user_uuid,
        name="WeatherToolbox",
        openapi_url=toolbox_openapi_url,
    )
    return validated_model


@app.get("/user/{user_uuid}/setup")
async def setup_user(user_uuid: str) -> Dict[str, Any]:
    """Setup user after creating.

    This function is called after the user is created.
    Currently it sets up weather toolbox for the user.
    """
    return await create_toolbox_for_new_user(user_uuid)


@app.put("/user/{user_uuid}/models/{type_name}/{model_name}/{model_uuid}")
async def update_model(
    user_uuid: str,
    type_name: str,
    model_name: str,
    model_uuid: str,
    model: Dict[str, Any],
) -> Dict[str, Any]:
    registry = Registry.get_default()
    validated_model = registry.validate(type_name, model_name, model)

    found_model = await DefaultDB.backend().find_model(model_uuid=model_uuid)
    await DefaultDB.backend().update_model(
        model_uuid=found_model["uuid"],
        user_uuid=user_uuid,
        type_name=type_name,
        model_name=model_name,
        json_str=validated_model.model_dump_json(),
    )

    return validated_model.model_dump()


@app.delete("/user/{user_uuid}/models/{type_name}/{model_uuid}")
async def models_delete(
    user_uuid: str, type_name: str, model_uuid: str
) -> Dict[str, Any]:
    found_model = await DefaultDB.backend().find_model(model_uuid=model_uuid)
    model = await DefaultDB.backend().delete_model(model_uuid=found_model["uuid"])
    return model["json_str"]  # type: ignore


def get_azure_llm_client() -> Tuple[AsyncAzureOpenAI, str]:
    azure_gpt35_model = environ["AZURE_GPT35_MODEL"]
    api_key = environ["AZURE_OPENAI_API_KEY"]
    azure_endpoint = environ["AZURE_API_ENDPOINT"]
    api_version = environ["AZURE_API_VERSION"]

    aclient = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,  # type: ignore
        api_version=api_version,
    )

    return aclient, azure_gpt35_model


# todo: fix monkeypatching in testing and remove global variables
try:
    aclient, azure_gpt35_model = get_azure_llm_client()
except Exception:
    aclient = None  # type: ignore[assignment]
    azure_gpt35_model = None  # type: ignore[assignment]

SYSTEM_PROMPT = """I am developing a chat application where users specify a task for the application to accomplish.
Generate a concise, professional name for the chat that directly reflects the essence of the task.
The name should consist of 2-3 words and be no more than 25 characters in total.
It should be immediately recognizable, meaningful, and sound natural to users, making it easy to identify the chat's
purpose at a glance. Please provide only the chat name in your response, with no additional text or explanation.
The name should use clear, user-friendly terminology that precisely captures the task's intent.

Note:
- I will tip you $1000 every time you generate a chat name that is 1-3 words long and up to 25 characters.
- Your chat name MUST be pertinent to the given task and avoids generic words such as "Forge" and "Hub".

Task Name:
{task_name}

Chat Name:
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_chat_name",
            "description": "Use this tool to generate a chat name based on the task description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_name": {
                        "type": "string",
                        "description": "The name of the chat",
                    },
                },
                "required": ["chat_name"],
            },
        },
    },
]


async def generate_chat_name(
    team_name: str,
    chat_id: int,
    chat_name: str,
) -> Dict[str, Union[Optional[str], int]]:
    return {
        "team_status": "inprogress",
        "team_name": team_name,
        "team_id": chat_id,
        "customer_brief": "Some customer brief",
        "conversation_name": chat_name,
    }


class ChatRequest(BaseModel):
    chat_id: int
    message: List[Dict[str, str]]
    user_id: int


@app.post("/user/{user_uuid}/chat/{model_name}/{model_uuid}")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    message = request.message[0]["content"]
    chat_id = request.chat_id
    user_id = request.user_id
    team_name = f"{user_id}_{chat_id}"

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(task_name=message)}
        ]
        completion = await aclient.chat.completions.create(
            model=azure_gpt35_model,
            messages=messages,
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {"name": "generate_chat_name"},
            },
        )  # type: ignore

        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                if function_name == "generate_chat_name":
                    return await generate_chat_name(  # type: ignore[return-value]
                        team_name=team_name,
                        chat_id=chat_id,
                        **function_args,
                    )

    except Exception:
        logging.error("Unable to generate chat name: ", exc_info=True)

    default_response = {
        "team_status": "inprogress",
        "team_name": team_name,
        "team_id": chat_id,
        "customer_brief": "Some customer brief",
        "conversation_name": message,
    }

    return default_response


@app.post("/deployment/{deployment_uuid}/chat")
async def deployment_chat(deployment_uuid: str) -> Dict[str, Any]:
    found_model = await DefaultDB.backend().find_model(model_uuid=deployment_uuid)
    team_name = found_model["json_str"]["name"]
    team_uuid = found_model["json_str"]["team"]["uuid"]

    return {
        "team_status": "inprogress",
        "team_name": team_name,
        "team_uuid": team_uuid,
        "conversation_name": "New Chat",
    }


@app.get("/deployment/{deployment_uuid}/ping")
async def deployment_ping() -> Dict[str, str]:
    return {
        "status": "ok",
    }


@app.post("/user/{user_uuid}/deployment/{deployment_uuid}")
async def create_auth_token(
    user_uuid: Annotated[str, Path(title="User UUID")],
    deployment_uuid: Annotated[str, Path(title="Deployment UUID")],
    name: Annotated[str, Body(title="Name of the auth token")],
    expiry: Annotated[str, Body(title="Expiry of the auth token")],
) -> DeploymentAuthToken:
    return await create_deployment_auth_token(
        user_uuid=user_uuid, deployment_uuid=deployment_uuid, name=name, expiry=expiry
    )


class DeploymentAuthTokenInfo(BaseModel):
    uuid: UUID
    name: str
    expiry: str


@app.get("/user/{user_uuid}/deployment/{deployment_uuid}")
async def get_all_deployment_auth_tokens(
    user_uuid: str, deployment_uuid: str
) -> List[DeploymentAuthTokenInfo]:
    user = await DefaultDB.frontend().get_user(user_uuid=user_uuid)
    deployment = await DefaultDB.backend().find_model(model_uuid=deployment_uuid)

    if user["uuid"] != deployment["user_uuid"]:
        raise HTTPException(  # pragma: no cover
            status_code=403, detail="User does not have access to this deployment"
        )

    auth_tokens = await DefaultDB.backend().find_many_auth_token(
        user_uuid=user_uuid, deployment_uuid=deployment_uuid
    )
    return [
        DeploymentAuthTokenInfo(
            uuid=auth_token["uuid"],
            name=auth_token["name"],
            expiry=auth_token["expiry"],
        )
        for auth_token in auth_tokens
    ]


@app.delete("/user/{user_uuid}/deployment/{deployment_uuid}/{auth_token_uuid}")
async def delete_deployment_auth_token(
    user_uuid: str,
    deployment_uuid: str,
    auth_token_uuid: str,
) -> DeploymentAuthTokenInfo:
    user = await DefaultDB.frontend().get_user(user_uuid=user_uuid)
    deployment = await DefaultDB.backend().find_model(model_uuid=deployment_uuid)

    if user["uuid"] != deployment["user_uuid"]:
        raise HTTPException(  # pragma: no cover
            status_code=403, detail="User does not have access to this deployment"
        )

    auth_token = await DefaultDB.backend().delete_auth_token(
        auth_token_uuid=auth_token_uuid,
        deployment_uuid=deployment_uuid,
        user_uuid=user_uuid,
    )
    return DeploymentAuthTokenInfo(
        uuid=auth_token["uuid"],  # type: ignore[union-attr]
        name=auth_token["name"],  # type: ignore[union-attr]
        expiry=auth_token["expiry"],  # type: ignore[union-attr]
    )
