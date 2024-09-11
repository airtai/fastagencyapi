import json
import os
from collections.abc import Iterator
from uuid import uuid4

import mesop as me

from fastagency.core.base import AskingMessage, WorkflowCompleted, Workflows
from fastagency.core.io.mesop.base import MesopMessage
from fastagency.core.io.mesop.components.inputs import input_prompt, input_user_feedback
from fastagency.core.io.mesop.components.ui_common import header
from fastagency.core.io.mesop.data_model import Conversation, State
from fastagency.core.io.mesop.message import message_box
from fastagency.core.io.mesop.send_prompt import (
    send_prompt_to_autogen,
    send_user_feedback_to_autogen,
)
from fastagency.core.io.mesop.styles import (
    CHAT_STARTER_STYLE,
    PAST_CHATS_HIDE_STYLE,
    PAST_CHATS_SHOW_STYLE,
    ROOT_BOX_STYLE,
    STYLESHEETS,
)
from fastagency.logging import get_logger

# Get the logger
logger = get_logger(__name__)


def get_workflows() -> Workflows:
    import_string = os.environ.get("IMPORT_STRING", None)
    if import_string is None:
        raise ValueError("No import string provided")

    # import app using import string
    # app = import_from_string(import_string)
    # todo workaround for failing import, fixme
    from examples.cli.main_mesop import app

    # get workflows from the app
    wf = app.wf

    return wf


SECURITY_POLICY = me.SecurityPolicy(allowed_iframe_parents=["https://huggingface.co"])


@me.page(  # type: ignore[misc]
    path="/",
    stylesheets=STYLESHEETS,
    security_policy=SECURITY_POLICY,
)
def home_page() -> None:
    state = me.state(State)
    with me.box(style=ROOT_BOX_STYLE):
        past_conversations_box()
        if state.in_conversation:
            conversation_box()
        else:
            conversation_starter_box()


def past_conversations_box() -> None:
    def select_past_conversation(ev: me.ClickEvent) -> None:
        id = ev.key
        state = me.state(State)
        conversations_with_id = list(
            filter(lambda c: c.id == id, state.past_conversations)
        )
        conversation = conversations_with_id[0]
        state.conversation = conversation
        state.in_conversation = True

    def on_show_hide(ev: me.ClickEvent) -> None:
        state.hide_past = not state.hide_past

    def on_start_new_conversation(ev: me.ClickEvent) -> None:
        state.in_conversation = False

    state = me.state(State)
    style = PAST_CHATS_HIDE_STYLE if state.hide_past else PAST_CHATS_SHOW_STYLE
    with me.box(style=style):
        with me.box(
            style=me.Style(
                flex_direction="row",
                width="100%",
                padding=me.Padding(top="16px"),
                justify_content="space-between",
            )
        ):
            with me.content_button(
                on_click=on_show_hide, disabled=not state.past_conversations
            ):
                me.icon("menu")
            with me.content_button(
                on_click=on_start_new_conversation,
                disabled=not state.conversation.completed,
            ):
                me.icon("rate_review")
        if not state.hide_past:
            for conversation in state.past_conversations:
                with me.box(
                    key=conversation.id,  # they are GUIDs so should not clash with anything other on the page
                    on_click=select_past_conversation,
                    style=me.Style(
                        padding=me.Padding.all(16),
                        border_radius=16,
                    ),
                ):
                    me.text(
                        text=conversation.title,
                    )


def conversation_starter_box() -> None:
    with me.box(style=CHAT_STARTER_STYLE):
        header()
        with me.box(
            style=me.Style(
                width="min(680px, 100%)",
                # margin=me.Margin.symmetric(horizontal="auto", vertical=36),
            )
        ):
            me.text(
                "Enter a prompt to chat with FastAgency team",
                style=me.Style(font_size=20, margin=me.Margin(bottom=24)),
            )
            input_prompt(send_prompt)


def _handle_message(state: State, message: MesopMessage) -> None:
    conversation = state.conversation
    messages = conversation.messages
    level = message.conversation.level
    conversation_id = message.conversation.id
    io_message = message.io_message
    message_dict = io_message.model_dump()
    message_string = json.dumps(
        {"level": level, "conversationId": conversation_id, "io_message": message_dict}
    )
    messages.append(message_string)
    conversation.messages = list(messages)
    if isinstance(io_message, AskingMessage):
        conversation.waiting_for_feedback = True
        conversation.completed = False
    if isinstance(io_message, WorkflowCompleted):
        conversation.completed = True
        conversation.waiting_for_feedback = False
        if not conversation.is_from_the_past:
            uuid: str = uuid4().hex
            becomme_past = Conversation(
                id=uuid,
                title=conversation.title,
                messages=conversation.messages,
                completed=True,
                is_from_the_past=True,
                waiting_for_feedback=False,
            )
            state.past_conversations.insert(0, becomme_past)


def send_prompt(e: me.ClickEvent) -> Iterator[None]:
    wf = get_workflows()
    name = wf.names[0]

    state = me.state(State)
    # me.navigate("/conversation")
    prompt = state.prompt
    state.prompt = ""
    conversation = Conversation(
        title=prompt, completed=False, waiting_for_feedback=False
    )
    state.conversation = conversation
    state.in_conversation = True
    yield
    responses = send_prompt_to_autogen(prompt=prompt, wf=wf, name=name)
    for message in responses:
        state = me.state(State)
        _handle_message(state, message)
        yield
        me.scroll_into_view(key="end_of_messages")
        yield
    yield


def conversation_box() -> None:
    state = me.state(State)
    conversation = state.conversation
    with me.box(style=CHAT_STARTER_STYLE):
        header()
        messages = conversation.messages
        with me.box(
            style=me.Style(
                overflow_y="auto",
            )
        ):
            for message in messages:
                message_box(message)
            if messages:
                me.box(
                    key="end_of_messages",
                    style=me.Style(margin=me.Margin(bottom="50vh")),
                )
        if conversation.waiting_for_feedback:
            input_user_feedback(on_user_feedback)


def on_user_feedback(e: me.ClickEvent) -> Iterator[None]:
    state = me.state(State)
    conversation = state.conversation
    feedback = conversation.feedback
    conversation.feedback = ""
    conversation.waiting_for_feedback = False
    yield
    me.scroll_into_view(key="end_of_messages")
    yield
    responses = send_user_feedback_to_autogen(feedback)
    for message in responses:
        state = me.state(State)
        _handle_message(state, message)
        yield
        me.scroll_into_view(key="end_of_messages")
        yield
    yield
