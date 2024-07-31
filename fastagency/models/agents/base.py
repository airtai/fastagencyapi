from typing import Annotated, List, Optional, Union
from uuid import UUID

from pydantic import Field
from typing_extensions import TypeAlias

from fastagency.openapi.client import Client

from ..base import Model
from ..registry import Registry
from ..toolboxes.toolbox import ToolboxRef

__all__ = ["AgentBaseModel"]

# Agents can work with any LLM, so we construct a union of all LLM references
llm_type_refs: TypeAlias = Union[  # type: ignore[valid-type]
    tuple(Registry.get_default().get_models_refs_by_type("llm"))
]


class AgentBaseModel(Model):
    llm: Annotated[
        llm_type_refs,
        Field(
            title="LLM",
            description="LLM used by the agent for producing responses",
        ),
    ]

    toolbox_1: Annotated[
        Optional[ToolboxRef],
        Field(
            title="Toolbox",
            description="Toolbox used by the agent for producing responses",
        ),
    ] = None

    toolbox_2: Annotated[
        Optional[ToolboxRef],
        Field(
            title="Toolbox",
            description="Toolbox used by the agent for producing responses",
        ),
    ] = None

    toolbox_3: Annotated[
        Optional[ToolboxRef],
        Field(
            title="Toolbox",
            description="Toolbox used by the agent for producing responses",
        ),
    ] = None

    async def get_clients_from_toolboxes(self, user_id: UUID) -> List[Client]:
        clients: List[Client] = []
        for i in range(3):
            toolbox_property = getattr(self, f"toolbox_{i+1}")
            if toolbox_property is None:
                continue

            toolbox_model = await toolbox_property.get_data_model().from_db(
                toolbox_property.uuid
            )
            client = await toolbox_model.create_autogen(toolbox_property.uuid, user_id)
            clients.append(client)
        return clients
