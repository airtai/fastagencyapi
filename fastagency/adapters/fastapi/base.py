## IONats part

import asyncio
import json
import re
from typing import Any, Callable, Optional, Union
from uuid import UUID, uuid4

import nats
import requests
import websockets
from fastapi import APIRouter, WebSocket
from pydantic import BaseModel

from fastagency.logging import get_logger

from ...base import UI, ProviderProtocol, WorkflowsProtocol
from ...messages import (
    AskingMessage,
    IOMessage,
    MessageProcessorMixin,
)
from ..nats import InitiateWorkflowModel, InputResponseModel, NatsProvider

logger = get_logger(__name__)


class InititateChatModel(BaseModel):
    workflow_name: str
    params: dict[str, Any]


class WorkflowInfo(BaseModel):
    name: str
    description: str

class FastAPIAdapter(MessageProcessorMixin):
    def __init__(
        self,
        provider: NatsProvider,
        *,
        user: Optional[str] = None,
        password: Optional[str] = None,
        super_conversation: Optional["FastAPIAdapter"] = None,
        initiate_chat_path: str = "/fastagency/initiate_chat",
        discovery_path: str = "/fastagency/discovery",
        ws_path: str = "/fastagency/ws",
    ) -> None:
        """Provider for NATS.

        Args:
            provider (NatsProvider): The provider.
            user (Optional[str], optional): The user. Defaults to None.
            password (Optional[str], optional): The password. Defaults to None.
            super_conversation (Optional["FastAPIProvider"], optional): The super conversation. Defaults to None.
        """
        self.provider = provider

        self.user = user
        self.password = password

        self.router = self.setup_routes()

        self.initiate_chat_path = initiate_chat_path
        self.discovery_path = discovery_path
        self.ws_path = ws_path

    def setup_routes(self) -> APIRouter:
        router = APIRouter()

        @router.post(self.initiate_chat_path)
        async def initiate_chat(
            initiate_chat: InititateChatModel,
        ) -> InitiateWorkflowModel:
            user_id: UUID = uuid4()
            workflow_uuid: UUID = uuid4()
            init_msg = InitiateWorkflowModel(
                user_id=user_id,
                workflow_uuid=workflow_uuid,
                params=initiate_chat.params,
                name=initiate_chat.workflow_name,
            )

            nc = await nats.connect(
                self.provider.nats_url,
                user=self.provider.user,
                password=self.provider.password,
            )
            await nc.publish(
                "chat.server.initiate_chat",
                init_msg.model_dump_json().encode(),
            )

            return init_msg

        @router.get(self.discovery_path)
        def discovery() -> list[WorkflowInfo]:
            names = self.provider.names
            descriptions = [self.provider.get_description(name) for name in names]
            return [
                WorkflowInfo(name=name, description=description)
                for name, description in zip(names, descriptions)
            ]
        
        @router.websocket(self.ws_path)
        async def websocket_endpoint(websocket: WebSocket):
            # read and write from NATS
            raise NotImplementedError("websocket_endpoint")
            # await websocket.accept()
            # while True:
            #     data = await websocket.receive_text()
            #     await websocket.send_text(f"Message text was: {data}")

        return router

    def visit_default(self, message: IOMessage) -> Optional[str]:
        raise NotImplementedError(f"visit_{message.type}")

    def process_message(self, message: IOMessage) -> Optional[str]:
        try:
            return self.visit(message)
        except Exception as e:
            logger.error(f"Error in process_message: {e}", stack_info=True)
            raise

    def create_subconversation(self) -> "FastAPIAdapter":
        return self

    @classmethod
    def create_provider(
        cls,
        fastapi_url: str,
        fastapi_user: Optional[str] = None,
        fastapi_password: Optional[str] = None,
        nats_url: Optional[str] = None,
        nats_user: Optional[str] = None,
        nats_password: Optional[str] = None,
    ) -> ProviderProtocol:
        return FastAPIProvider(
            fastapi_url=fastapi_url,
            fastapi_user=fastapi_user,
            fastapi_password=fastapi_password,
            nats_url=nats_url,
            nats_user=nats_user,
            nats_password=nats_password,
        )


class FastAPIProvider(ProviderProtocol):
    def __init__(
        self,
        fastapi_url: str,
        fastapi_user: Optional[str] = None,
        fastapi_password: Optional[str] = None,
        # nats_url: Optional[str] = None,
        # nats_user: Optional[str] = None,
        # nats_password: Optional[str] = None,
        initiate_chat_path: str = "/fastagency/initiate_chat",
        discovery_path: str = "/fastagency/discovery",
        ws_path: str = "/fastagency/ws",
        # todo: we need security context
    ) -> None:
        """Initialize the fastapi workflows."""
        self._workflows: dict[
            str, tuple[Callable[[WorkflowsProtocol, UI, str, str], str], str]
        ] = {}

        self.fastapi_url = (
            fastapi_url[:-1] if fastapi_url.endswith("/") else fastapi_url
        )
        self.fastapi_user = fastapi_user
        self.fastapi_password = fastapi_password

        # self.nats_url = nats_url or "ws://localhost:9222"
        if not self.nats_url.startswith(
            "ws://"  # nosemgrep
        ) and not self.nats_url.startswith("wss://"):
            raise ValueError(
                f"NATS URL must start with ws:// or wss:// but got {self.nats_url}"  # nosemgrep
            )
        # self.nats_user = nats_user
        # self.nats_password = nats_password

        self.is_broker_running: bool = False

        self.initiate_chat_path = initiate_chat_path
        self.discovery_path = discovery_path

    def _send_initiate_chat_msg(
        self, workflow_name: str, params: dict[str, Any]
    ) -> InitiateWorkflowModel:
        msg = InititateChatModel(workflow_name=workflow_name, params=params)

        payload = msg.model_dump()

        resp = requests.post(
            f"{self.fastapi_url}{self.initiate_chat_path}", json=payload, timeout=5
        )
        retval = InitiateWorkflowModel(**resp.json())
        return retval

    async def _create_connect_message(self) -> str:
        connect_msg = {
            "verbose": False,
            "pedantic": False,
            "tls_required": False,
            "user": self.nats_user,
            "pass": self.nats_password,
        }
        return f"CONNECT {json.dumps(connect_msg)}\r\n"

    async def _publish_websocket_message(
        self,
        websocket: websockets.WebSocketClientProtocol,
        subject: str,
        message: InputResponseModel,
    ) -> None:
        payload = message.model_dump_json()
        # todo: reformat message to match the adapter
        raise NotImplementedError("publish_websocket_message")
        # protocol_message = f"PUB {subject} {len(payload)}\r\n{payload}\r\n"
        # await websocket.send(protocol_message)
        # logger.info(f"Message sent to topic {subject}: {message}")

    async def _parse_websocket_message(
        self, message: Union[bytes, str]
    ) -> Optional[dict[str, Any]]:
        message_str = message.decode() if isinstance(message, bytes) else message

        # todo: parse to match the adapter
        raise NotImplementedError("parse_websocket_message")
        # if not message_str.startswith("MSG"):
        #     return None

        # pattern = r"\r\n(.*?)\r\n"
        # json_str = re.search(pattern, message_str, re.DOTALL).group(1)  # type: ignore [union-attr]
        # return json.loads(json_str)  # type: ignore [no-any-return]

    async def _run_websocket_subscriber(
        self, ui: UI, from_server_subject: str, to_server_subject: str
    ) -> None:
        # connect_url = self.nats_url
        connect_url = f"{self.fastapi_url}{self.ws_path}"
        async with websockets.connect(connect_url) as websocket:
            connect_message = await self._create_connect_message()
            await websocket.send(connect_message)

            message = f"SUB {from_server_subject} 1\r\n"
            await websocket.send(message)
            logger.info(f"Subscribed to topic {from_server_subject}")

            while True:
                response = await websocket.recv()

                parsed_message = await self._parse_websocket_message(response)
                if parsed_message is None:
                    continue
                logger.info(f"Received message: {parsed_message}")

                iomessage = (
                    IOMessage.create(**{"type": "error", "long": parsed_message["msg"]})
                    if parsed_message.get("error")
                    else IOMessage.create(**parsed_message)
                )
                if isinstance(iomessage, AskingMessage):
                    processed_message = ui.process_message(iomessage)
                    input_response = InputResponseModel(
                        msg=processed_message, question_id=iomessage.uuid
                    )
                    logger.debug(f"Processed response: {input_response}")
                    await self._publish_websocket_message(
                        websocket, to_server_subject, input_response
                    )
                else:
                    ui.process_message(iomessage)

    def run(self, name: str, ui: UI, **kwargs: Any) -> str:
        initiate_workflow = self._send_initiate_chat_msg(name, params=kwargs)
        user_id = initiate_workflow.user_id
        workflow_uuid = initiate_workflow.workflow_uuid

        _from_server_subject = f"chat.client.messages.{user_id}.{workflow_uuid}"
        _to_server_subject = f"chat.server.messages.{user_id}.{workflow_uuid}"

        async def _setup_and_run() -> None:
            await self._run_websocket_subscriber(
                ui, _from_server_subject, _to_server_subject
            )

        async def run_lifespan() -> None:
            if not self.is_broker_running:
                self.is_broker_running = True
                await _setup_and_run()
            else:
                await _setup_and_run()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(run_lifespan())

        return "FastAPIWorkflows.run() completed"

    def _get_workflow_info(self) -> list[dict[str, str]]:
        resp = requests.get(f"{self.fastapi_url}/{self.discovery_path}", timeout=5)
        return resp.json()  # type: ignore [no-any-return]

    def _get_names(self) -> list[str]:
        return [workflow["name"] for workflow in self._get_workflow_info()]

    def _get_description(self, name: str) -> str:
        return next(
            workflow["description"]
            for workflow in self._get_workflow_info()
            if workflow["name"] == name
        )

    @property
    def names(self) -> list[str]:
        return self._get_names()

    def get_description(self, name: str) -> str:
        return self._get_description(name)
