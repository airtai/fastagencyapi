import contextlib
import os
import random
import socket
import threading
import time
import uuid
from platform import system
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Optional,
    TypeVar,
)

import openai
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI, Path
from pydantic import BaseModel

from fastagency.db.base import DefaultDB
from fastagency.db.inmemory import InMemoryBackendDB, InMemoryFrontendDB
from fastagency.helpers import create_autogen, create_model_ref, get_model_by_ref
from fastagency.models.agents.assistant import AssistantAgent
from fastagency.models.agents.user_proxy import UserProxyAgent
from fastagency.models.agents.web_surfer import BingAPIKey, WebSurferAgent
from fastagency.models.agents.web_surfer_autogen import WebSurferChat
from fastagency.models.base import ObjectReference
from fastagency.models.llms.anthropic import Anthropic, AnthropicAPIKey
from fastagency.models.llms.azure import AzureOAI, AzureOAIAPIKey
from fastagency.models.llms.openai import OpenAI, OpenAIAPIKey
from fastagency.models.llms.together import TogetherAI, TogetherAIAPIKey
from fastagency.models.teams.two_agent_teams import TwoAgentTeam
from fastagency.models.toolboxes.toolbox import OpenAPIAuth, Toolbox

from .helpers import add_random_sufix, expand_fixture, get_by_tag, tag, tag_list

F = TypeVar("F", bound=Callable[..., Any])


@pytest_asyncio.fixture(scope="session", autouse=True)  # type: ignore[misc]
async def set_default_db() -> AsyncGenerator[None, None]:
    backend_db = InMemoryBackendDB()
    frontend_db = InMemoryFrontendDB()

    with (
        DefaultDB.set(backend_db=backend_db, frontend_db=frontend_db),
    ):
        yield


@pytest_asyncio.fixture(scope="session")  # type: ignore[misc]
async def user_uuid() -> AsyncIterator[str]:
    try:
        random_id = random.randint(1, 1_000_000)
        generated_uuid = uuid.uuid4()
        email = f"user{random_id}@airt.ai"
        username = f"user{random_id}"

        await DefaultDB.frontend()._create_user(
            user_uuid=generated_uuid, email=email, username=username
        )
        user = await DefaultDB.frontend().get_user(user_uuid=generated_uuid)

        yield user["uuid"]
    finally:
        pass


################################################################################
###
# Fixtures for LLMs
###
################################################################################


def azure_model_llm_config(model_env_name: str) -> Dict[str, Any]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", default="*" * 64)
    api_base = os.getenv(
        "AZURE_API_ENDPOINT", default="https://my-deployment.openai.azure.com"
    )

    def get_default_model_name(model_env_name: str) -> str:
        if model_env_name == "AZURE_GPT35_MODEL":
            return "gpt-35-turbo-16k"
        elif model_env_name == "AZURE_GPT4_MODEL":
            return "gpt-4"
        elif model_env_name == "AZURE_GPT4o_MODEL":
            return "gpt-4o"
        else:
            raise ValueError(f"Unknown model_env_name: {model_env_name}")

    default_model_env_name = get_default_model_name(model_env_name)
    gpt_model_name = os.getenv(model_env_name, default=default_model_env_name)

    openai.api_type = "azure"
    openai.api_version = os.getenv("AZURE_API_VERSION", default="2024-02-01")

    config_list = [
        {
            "model": gpt_model_name,
            "api_key": api_key,
            "base_url": api_base,
            "api_type": openai.api_type,
            "api_version": openai.api_version,
        }
    ]

    llm_config = {
        "config_list": config_list,
        "temperature": 0.8,
    }

    return llm_config


@tag("llm_config")
@pytest.fixture()
def azure_gpt35_turbo_16k_llm_config() -> Dict[str, Any]:
    return azure_model_llm_config("AZURE_GPT35_MODEL")


@tag("llm_config")
@pytest.fixture()
def azure_gpt4_llm_config() -> Dict[str, Any]:
    return azure_model_llm_config("AZURE_GPT4_MODEL")


@tag("llm_config")
@pytest.fixture()
def azure_gpt4o_llm_config() -> Dict[str, Any]:
    return azure_model_llm_config("AZURE_GPT4o_MODEL")


def openai_llm_config(model: str) -> Dict[str, Any]:
    zeros = "0" * 20
    api_key = os.getenv("OPENAI_API_KEY", default=f"sk-{zeros}T3BlbkFJ{zeros}")

    config_list = [
        {
            "model": model,
            "api_key": api_key,
        }
    ]

    llm_config = {
        "config_list": config_list,
        "temperature": 0.8,
    }

    return llm_config


@tag("llm_config")
@pytest.fixture()
def openai_gpt35_turbo_16k_llm_config() -> Dict[str, Any]:
    return openai_llm_config("gpt-3.5-turbo")


# @tag("llm_config")
# @pytest.fixture()
# def openai_gpt4_llm_config() -> Dict[str, Any]:
#     return openai_llm_config("gpt-4")


@tag("llm-key")
@pytest_asyncio.fixture()
async def azure_oai_key_ref(
    user_uuid: str, azure_gpt35_turbo_16k_llm_config: Dict[str, Any]
) -> ObjectReference:
    api_key = azure_gpt35_turbo_16k_llm_config["config_list"][0]["api_key"]
    return await create_model_ref(
        AzureOAIAPIKey,
        "secret",
        user_uuid=user_uuid,
        name=add_random_sufix("azure_oai_key"),
        api_key=api_key,
    )


@tag("llm", "noapi", "weather-llm")
@pytest_asyncio.fixture()
async def azure_oai_gpt35_ref(
    user_uuid: str,
    azure_gpt35_turbo_16k_llm_config: Dict[str, Any],
    azure_oai_key_ref: ObjectReference,
) -> ObjectReference:
    kwargs = azure_gpt35_turbo_16k_llm_config["config_list"][0].copy()
    kwargs.pop("api_key")
    temperature = azure_gpt35_turbo_16k_llm_config["temperature"]
    return await create_model_ref(
        AzureOAI,
        "llm",
        user_uuid=user_uuid,
        name=add_random_sufix("azure_oai"),
        api_key=azure_oai_key_ref,
        temperature=temperature,
        **kwargs,
    )


@tag("llm")
@pytest_asyncio.fixture()
async def azure_oai_gpt4_ref(
    user_uuid: str,
    azure_gpt4_llm_config: Dict[str, Any],
    azure_oai_key_ref: ObjectReference,
) -> ObjectReference:
    kwargs = azure_gpt4_llm_config["config_list"][0].copy()
    kwargs.pop("api_key")
    temperature = azure_gpt4_llm_config["temperature"]
    return await create_model_ref(
        AzureOAI,
        "llm",
        user_uuid=user_uuid,
        name=add_random_sufix("azure_oai"),
        api_key=azure_oai_key_ref,
        temperature=temperature,
        **kwargs,
    )


@tag("llm", "websurfer-llm")
@pytest_asyncio.fixture()
async def azure_oai_gpt4o_ref(
    user_uuid: str,
    azure_gpt4o_llm_config: Dict[str, Any],
    azure_oai_key_ref: ObjectReference,
) -> ObjectReference:
    kwargs = azure_gpt4o_llm_config["config_list"][0].copy()
    kwargs.pop("api_key")
    temperature = azure_gpt4o_llm_config["temperature"]
    return await create_model_ref(
        AzureOAI,
        "llm",
        user_uuid=user_uuid,
        name=add_random_sufix("azure_oai"),
        api_key=azure_oai_key_ref,
        temperature=temperature,
        **kwargs,
    )


async def openai_oai_key_ref(
    user_uuid: str, openai_llm_config: Dict[str, Any]
) -> ObjectReference:
    api_key = openai_llm_config["config_list"][0]["api_key"]
    model = openai_llm_config["config_list"][0]["model"]
    return await create_model_ref(
        OpenAIAPIKey,
        "secret",
        user_uuid=user_uuid,
        name=add_random_sufix("openai_oai_key"),
        api_key=api_key,
        model=model,
    )


@tag("llm-key")
@pytest_asyncio.fixture()
async def openai_oai_key_gpt35_ref(
    user_uuid: str, openai_gpt35_turbo_16k_llm_config: Dict[str, Any]
) -> ObjectReference:
    return await openai_oai_key_ref(user_uuid, openai_gpt35_turbo_16k_llm_config)


# @tag("llm-key")
# @pytest_asyncio.fixture()
# async def openai_oai_key_gpt4_ref(
#     user_uuid: str, openai_gpt4_llm_config: Dict[str, Any]
# ) -> ObjectReference:
#     return await openai_oai_key_ref(user_uuid, openai_gpt4_llm_config)


async def openai_oai_ref(
    user_uuid: str,
    openai_llm_config: Dict[str, Any],
    openai_oai_key_ref: ObjectReference,
) -> ObjectReference:
    kwargs = openai_llm_config["config_list"][0].copy()
    kwargs.pop("api_key")
    temperature = openai_llm_config["temperature"]
    return await create_model_ref(
        OpenAI,
        "llm",
        user_uuid=user_uuid,
        name=add_random_sufix("azure_oai"),
        api_key=openai_oai_key_ref,
        temperature=temperature,
        **kwargs,
    )


@tag("llm", "noapi", "weather-llm", "openai-llm")
@pytest_asyncio.fixture()
async def openai_oai_gpt35_ref(
    user_uuid: str,
    openai_gpt35_turbo_16k_llm_config: Dict[str, Any],
    openai_oai_key_gpt35_ref: ObjectReference,
) -> ObjectReference:
    return await openai_oai_ref(
        user_uuid, openai_gpt35_turbo_16k_llm_config, openai_oai_key_gpt35_ref
    )


# @tag("openai-llm")
# @pytest_asyncio.fixture()
# async def openai_oai_gpt4_ref(
#     user_uuid: str,
#     openai_gpt4_llm_config: Dict[str, Any],
#     openai_oai_key_gpt4_ref: ObjectReference,
# ) -> ObjectReference:
#     return await openai_oai_ref(
#         user_uuid, openai_gpt4_llm_config, openai_oai_key_gpt4_ref
#     )


@tag("llm-key")
@pytest_asyncio.fixture()
async def anthropic_key_ref(user_uuid: str) -> ObjectReference:
    api_key = os.getenv(
        "ANTHROPIC_API_KEY",
        default="sk-ant-api03-" + "_" * 95,
    )

    return await create_model_ref(
        AnthropicAPIKey,
        "secret",
        user_uuid=user_uuid,
        name=add_random_sufix("anthropic_api_key"),
        api_key=api_key,
    )


@tag("llm", "weather-llm")
@pytest_asyncio.fixture()
async def anthropic_ref(
    user_uuid: str,
    anthropic_key_ref: ObjectReference,
) -> ObjectReference:
    return await create_model_ref(
        Anthropic,
        "llm",
        user_uuid=user_uuid,
        name=add_random_sufix("anthropic_api"),
        api_key=anthropic_key_ref,
    )


@tag("llm-key")
@pytest_asyncio.fixture()
async def together_ai_key_ref(user_uuid: str) -> ObjectReference:
    api_key = os.getenv(
        "TOGETHER_API_KEY",
        default="*" * 64,
    )

    return await create_model_ref(
        TogetherAIAPIKey,
        "secret",
        user_uuid=user_uuid,
        name=add_random_sufix("togetherai_api_key"),
        api_key=api_key,
    )


@tag("llm", "noapi")
@pytest_asyncio.fixture()
async def togetherai_ref(
    user_uuid: str,
    together_ai_key_ref: ObjectReference,
) -> ObjectReference:
    return await create_model_ref(
        TogetherAI,
        "llm",
        user_uuid=user_uuid,
        name=add_random_sufix("togetherai"),
        api_key=together_ai_key_ref,
        model="Mixtral-8x7B Instruct v0.1",
    )


################################################################################
###
# Fixtures for Toolkit
###
################################################################################


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


def create_fastapi_app(host: str, port: int) -> FastAPI:
    app = FastAPI(
        servers=[
            {"url": f"http://{host}:{port}", "description": "Local development server"}
        ]
    )

    @app.get("/")
    def read_root() -> Dict[str, str]:
        return {"Hello": "World"}

    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: Optional[str] = None) -> Dict[str, Any]:
        return {"item_id": item_id, "q": q}

    @app.post("/items")
    async def create_item(item: Item) -> Item:
        return item

    return app


def create_weather_fastapi_app(host: str, port: int) -> FastAPI:
    app = FastAPI(
        title="Weather",
        servers=[
            {"url": f"http://{host}:{port}", "description": "Local development server"}
        ],
    )

    @app.get("/forecast/{city}", description="Get the weather forecast for a city")
    def forecast(
        city: Annotated[str, Path(description="name of the city")],
    ) -> str:
        return f"Weather in {city} is sunny"

    return app


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]  # type: ignore [no-any-return]


def run_server(app: FastAPI, host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


class Server(uvicorn.Server):  # type: ignore [misc]
    def install_signal_handlers(self) -> None:
        pass

    @contextlib.contextmanager
    def run_in_thread(self) -> Iterator[None]:
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


@pytest.fixture(scope="session")
def fastapi_openapi_url() -> Iterator[str]:
    host = "127.0.0.1"
    port = find_free_port()
    app = create_fastapi_app(host, port)
    openapi_url = f"http://{host}:{port}/openapi.json"

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = Server(config=config)
    with server.run_in_thread():
        time.sleep(1 if system() != "Windows" else 5)  # let the server start

        yield openapi_url


@pytest.fixture(scope="session")
def weather_fastapi_openapi_url() -> Iterator[str]:
    host = "127.0.0.1"
    port = find_free_port()
    app = create_weather_fastapi_app(host, port)
    openapi_url = f"http://{host}:{port}/openapi.json"

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = Server(config=config)
    with server.run_in_thread():
        time.sleep(1 if system() != "Windows" else 5)  # let the server start

        yield openapi_url


@tag("toolbox", "items")
@pytest_asyncio.fixture()  # type: ignore[misc]
async def toolbox_ref(user_uuid: str, fastapi_openapi_url: str) -> ObjectReference:
    openapi_auth = await create_model_ref(
        OpenAPIAuth,
        "secret",
        user_uuid,
        name="openapi_auth_secret",
        username="test",
        password="password",  # pragma: allowlist secret
    )

    toolbox = await create_model_ref(
        Toolbox,
        "toolbox",
        user_uuid,
        name="test_toolbox",
        openapi_url=fastapi_openapi_url,
        openapi_auth=openapi_auth,
    )

    return toolbox


@tag("toolbox", "weather")
@pytest_asyncio.fixture()  # type: ignore[misc]
async def weather_toolbox_ref(
    user_uuid: str, weather_fastapi_openapi_url: str
) -> ObjectReference:
    openapi_auth = await create_model_ref(
        OpenAPIAuth,
        "secret",
        user_uuid,
        name="openapi_auth_secret",
        username="test",
        password="password",  # pragma: allowlist secret
    )

    toolbox = await create_model_ref(
        Toolbox,
        "toolbox",
        user_uuid,
        name="test_toolbox",
        openapi_url=weather_fastapi_openapi_url,
        openapi_auth=openapi_auth,
    )

    return toolbox


################################################################################
###
# Fixtures for Agents
###
################################################################################


@tag_list("assistant", "noapi")
@expand_fixture(
    dst_fixture_prefix="assistant_noapi",
    src_fixtures_names=get_by_tag("llm", "noapi"),
    placeholder_name="llm_ref",
)
async def placeholder_assistant_noapi_ref(
    user_uuid: str, llm_ref: ObjectReference
) -> ObjectReference:
    return await create_model_ref(
        AssistantAgent,
        "agent",
        user_uuid=user_uuid,
        name=add_random_sufix("assistant"),
        llm=llm_ref,
    )


# @pytest_asyncio.fixture()
# async def assistant_noapi_openai_oai_gpt4_ref(
#     user_uuid: str, openai_oai_gpt4_ref: ObjectReference
# ) -> ObjectReference:
#     return await create_model_ref(
#         AssistantAgent,
#         "agent",
#         user_uuid=user_uuid,
#         name=add_random_sufix("assistant"),
#         llm=openai_oai_gpt4_ref,
#     )


@tag_list("assistant", "weather")
@expand_fixture(
    dst_fixture_prefix="assistant_weather",
    src_fixtures_names=get_by_tag("weather-llm"),
    placeholder_name="llm_ref",
)
async def placeholder_assistant_weatherapi_ref(
    user_uuid: str, llm_ref: ObjectReference, weather_toolbox_ref: ObjectReference
) -> ObjectReference:
    return await create_model_ref(
        AssistantAgent,
        "agent",
        user_uuid=user_uuid,
        name=add_random_sufix("assistant_weather"),
        llm=llm_ref,
        toolbox_1=weather_toolbox_ref,
        system_message="You are a helpful assistant with access to Weather API. After you successfully answer the question asked and there are no new questions, terminate the chat by outputting 'TERMINATE'",
    )


@pytest_asyncio.fixture()
async def bing_api_key_ref(user_uuid: str) -> ObjectReference:
    api_key = os.getenv(
        "BING_API_KEY",
        default="*" * 64,
    )
    return await create_model_ref(
        BingAPIKey,
        "secret",
        user_uuid=user_uuid,
        name=add_random_sufix("bing_api_key"),
        api_key=api_key,
    )


@tag_list("websurfer")
@expand_fixture(
    dst_fixture_prefix="websurfer",
    src_fixtures_names=get_by_tag("websurfer-llm"),
    placeholder_name="llm_ref",
)
async def placeholder_websurfer_ref(
    user_uuid: str, llm_ref: ObjectReference, bing_api_key_ref: ObjectReference
) -> ObjectReference:
    return await create_model_ref(
        WebSurferAgent,
        "agent",
        user_uuid=user_uuid,
        name=add_random_sufix("websurfer"),
        llm=llm_ref,
        summarizer_llm=llm_ref,
        bing_api_key=bing_api_key_ref,
    )


@tag_list("websurfer-chat")
@expand_fixture(
    dst_fixture_prefix="websurfer_chat",
    src_fixtures_names=get_by_tag("websurfer"),
    placeholder_name="websurfer_ref",
)
async def placeholder_websurfer_chat(
    user_uuid: str, websurfer_ref: ObjectReference, bing_api_key_ref: ObjectReference
) -> WebSurferChat:
    websurfer_model: WebSurferAgent = await get_model_by_ref(websurfer_ref)  # type: ignore [assignment]
    llm_config = await create_autogen(websurfer_model.llm, user_uuid)
    summarizer_llm_config = await create_autogen(
        websurfer_model.summarizer_llm, user_uuid
    )

    bing_api_key = (
        await create_autogen(websurfer_model.bing_api_key, user_uuid)
        if websurfer_model.bing_api_key
        else None
    )

    viewport_size = websurfer_model.viewport_size

    return WebSurferChat(
        name_prefix=websurfer_model.name,
        llm_config=llm_config,
        summarizer_llm_config=summarizer_llm_config,
        viewport_size=viewport_size,
        bing_api_key=bing_api_key,
    )


@pytest_asyncio.fixture()
async def user_proxy_agent_ref(user_uuid: str) -> ObjectReference:
    return await create_model_ref(
        UserProxyAgent,
        "agent",
        user_uuid=user_uuid,
        name=add_random_sufix("user_proxy_agent"),
        max_consecutive_auto_reply=10,
        human_input_mode="NEVER",
    )


################################################################################
###
# Fixtures for Two Agent Teams
###
################################################################################


@tag_list("team", "noapi")
@expand_fixture(
    dst_fixture_prefix="two_agent_team_noapi",
    src_fixtures_names=get_by_tag("assistant", "noapi"),
    placeholder_name="assistant_ref",
)
async def placeholder_team_noapi_ref(
    user_uuid: str,
    assistant_ref: ObjectReference,
    user_proxy_agent_ref: ObjectReference,
) -> ObjectReference:
    return await create_model_ref(
        TwoAgentTeam,
        "team",
        user_uuid=user_uuid,
        name=add_random_sufix("two_agent_team_noapi"),
        initial_agent=user_proxy_agent_ref,
        secondary_agent=assistant_ref,
        human_input_mode="NEVER",
    )


@tag_list("team", "weather")
@expand_fixture(
    dst_fixture_prefix="two_agent_team_weatherapi",
    src_fixtures_names=get_by_tag("assistant", "weather"),
    placeholder_name="assistant_ref",
)
async def placeholder_team_weatherapi_ref(
    user_uuid: str,
    assistant_ref: ObjectReference,
    user_proxy_agent_ref: ObjectReference,
) -> ObjectReference:
    return await create_model_ref(
        TwoAgentTeam,
        "team",
        user_uuid=user_uuid,
        name=add_random_sufix("two_agent_team_weather"),
        initial_agent=user_proxy_agent_ref,
        secondary_agent=assistant_ref,
        human_input_mode="NEVER",
    )


# FastAPI app for testing

################################################################################
###
# Fixtures for application
###
################################################################################
