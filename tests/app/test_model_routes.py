import random
import uuid
from typing import List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fastagency.app import app, mask
from fastagency.db.base import DefaultDB
from fastagency.models.llms.azure import AzureOAIAPIKey
from fastagency.saas_app_generator import SaasAppGenerator

client = TestClient(app)


class Function:
    def __init__(self, arguments: str, name: str):
        """Function class."""
        self.arguments = arguments
        self.name = name


class ChatCompletionMessageToolCall:
    def __init__(self, id: str, function: Function, type: str):
        """ChatCompletionMessageToolCall class."""
        self.id = id
        self.function = function
        self.type = type


class ChatCompletionMessage:
    def __init__(
        self,
        content: Optional[str],
        role: str,
        function_call: Optional[str],
        tool_calls: List[ChatCompletionMessageToolCall],
    ):
        """ChatCompletionMessage class."""
        self.content = content
        self.role = role
        self.function_call = function_call
        self.tool_calls = tool_calls


class Choice:
    def __init__(self, message: ChatCompletionMessage):
        """Choice class."""
        self.message = message


class MockChatCompletion:
    def __init__(self, id: str, choices: List[Choice]):
        """MockChatCompletion class."""
        self.id = id
        self.choices = choices


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "api_key,expected",  # noqa: PT006
    [("whatever", "wha*ever"), ("some_other_key", "som*******_key")],
)
async def test_mask(api_key: str, expected: str) -> None:
    assert await mask(api_key) == expected


@pytest.mark.db()
class TestModelRoutes:
    @pytest.mark.asyncio()
    async def test_get_all_models(
        self, user_uuid: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_uuid = str(uuid.uuid4())
        azure_oai_api_key = AzureOAIAPIKey(api_key="whatever", name="whatever")
        type_name = "secret"
        model_name = "AzureOAIAPIKey"

        # Create model
        response = client.post(
            f"/user/{user_uuid}/models/{type_name}/{model_name}/{key_uuid}",
            json=azure_oai_api_key.model_dump(),
        )
        assert response.status_code == 200

        response = client.get(f"/user/{user_uuid}/models")
        assert response.status_code == 200

        expected = [
            {
                "json_str": {
                    "api_key": "wha*ever",  # pragma: allowlist secret
                    "name": "whatever",
                },
                "uuid": key_uuid,
                "type_name": "secret",
                "model_name": "AzureOAIAPIKey",
                "user_uuid": user_uuid,
            }
        ]
        actual = response.json()
        assert len(actual) == len(expected)
        for i in range(len(expected)):
            for key in expected[i]:
                assert actual[i][key] == expected[i][key]

    @pytest.mark.asyncio()
    async def test_setup_user(self) -> None:
        random_id = random.randint(1, 1_000_000)
        user_uuid = await DefaultDB.frontend()._create_user(
            user_uuid=uuid.uuid4(),
            email=f"user{random_id}@airt.ai",
            username=f"user{random_id}",
        )
        # Call setup route for user
        response = client.get(f"/user/{user_uuid}/setup")
        assert response.status_code == 200, response.text
        expected_setup = {
            "name": "WeatherToolbox",
            "openapi_url": "https://weather.tools.staging.fastagency.ai/openapi.json",
            "openapi_auth": None,
        }
        actual = response.json()
        assert actual == expected_setup

        # Call get all models route to check for the newly added weather toolbox
        response = client.get(
            f"/user/{user_uuid}/models", params={"type_name": "toolbox"}
        )
        assert response.status_code == 200
        expected_toolbox_model = {
            "user_uuid": str(user_uuid),
            "type_name": "toolbox",
            "model_name": "Toolbox",
            "json_str": {
                "name": "WeatherToolbox",
                "openapi_url": "https://weather.tools.staging.fastagency.ai/openapi.json",
                "openapi_auth": None,
            },
        }
        actual_toolbox_model = next(
            iter(
                [
                    model
                    for model in response.json()
                    if model["json_str"]["name"] == "WeatherToolbox"
                ]
            )
        )

        for key, value in expected_toolbox_model.items():
            assert actual_toolbox_model[key] == value

        # Call the setup route again and check the response
        response = client.get(f"/user/{user_uuid}/setup")
        assert response.status_code == 400
        expected_setup_again = {"detail": "Weather toolbox already exists"}
        actual = response.json()
        assert actual == expected_setup_again

    @pytest.mark.asyncio()
    async def test_add_model(self, user_uuid: str) -> None:
        model_uuid = str(uuid.uuid4())
        azure_oai_api_key = AzureOAIAPIKey(api_key="whatever", name="who cares?")
        response = client.post(
            f"/user/{user_uuid}/models/secret/AzureOAIAPIKey/{model_uuid}",
            json=azure_oai_api_key.model_dump(),
        )

        assert response.status_code == 200
        expected = {
            "api_key": "whatever",  # pragma: allowlist secret
            "name": "who cares?",
        }
        actual = response.json()
        assert actual == expected

    @pytest.mark.asyncio()
    async def test_add_model_deployment(self, user_uuid: str) -> None:
        team_uuid = str(uuid.uuid4())
        deployment_uuid = str(uuid.uuid4())
        gh_token_uuid = str(uuid.uuid4())
        fly_token_uuid = str(uuid.uuid4())

        model = {
            "name": "name",
            "repo_name": "repo_name",
            "fly_app_name": "test the deployment name char",  # within the character limit. Max 30
            "team": {"uuid": team_uuid, "type": "team", "name": "TwoAgentTeam"},
            "gh_token": {
                "uuid": gh_token_uuid,
                "type": "secret",
                "name": "GitHubToken",
            },
            "fly_token": {"uuid": fly_token_uuid, "type": "secret", "name": "FlyToken"},
            "uuid": deployment_uuid,
            "type_name": "deployment",
            "model_name": "Deployment",
        }
        type_name = "deployment"
        model_name = "Deployment"
        model_uuid = str(uuid.uuid4())

        # Mock the background task
        fly_api_token = "some-token"
        fastagency_deployment_uuid = "some-uuid"
        github_token = "some-github-token"
        app_name = "test fastagency template"
        repo_name = "test-fastagency-template"
        fly_app_name = "test-fastagency-template"
        saas_app = SaasAppGenerator(
            fly_api_token,
            fastagency_deployment_uuid,
            github_token,
            app_name,
            repo_name,
            fly_app_name,
        )
        saas_app.gh_repo_url = "https://some-git-url"
        with (
            patch(
                "fastagency.helpers.validate_tokens_and_create_gh_repo",
                return_value=saas_app,
            ) as mock_task,
            patch("fastagency.helpers.deploy_saas_app"),
        ):
            response = client.post(
                f"/user/{user_uuid}/models/{type_name}/{model_name}/{model_uuid}",
                json=model,
            )
            mock_task.assert_called_once()

        assert response.status_code == 200
        expected = {
            "name": "name",
            "repo_name": "repo_name",
            "fly_app_name": "test the deployment name char",
            "team": {"type": "team", "name": "TwoAgentTeam", "uuid": team_uuid},
            "gh_token": {
                "type": "secret",
                "name": "GitHubToken",
                "uuid": gh_token_uuid,
            },
            "fly_token": {"type": "secret", "name": "FlyToken", "uuid": fly_token_uuid},
            "app_deploy_status": "inprogress",
            "gh_repo_url": "https://some-git-url",
        }

        actual = response.json()
        assert actual == expected

    @pytest.mark.asyncio()
    async def test_add_model_deployment_with_long_name(self, user_uuid: str) -> None:
        team_uuid = str(uuid.uuid4())
        deployment_uuid = str(uuid.uuid4())
        gh_token_uuid = str(uuid.uuid4())
        fly_token_uuid = str(uuid.uuid4())

        model = {
            "name": "name",
            "repo_name": "repo_name",
            "fly_app_name": "test the deployment name charc",  # beyond the character limit. Max 30
            "team": {"uuid": team_uuid, "type": "team", "name": "TwoAgentTeam"},
            "gh_token": {
                "uuid": gh_token_uuid,
                "type": "secret",
                "name": "GitHubToken",
            },
            "fly_token": {"uuid": fly_token_uuid, "type": "secret", "name": "FlyToken"},
            "uuid": deployment_uuid,
            "type_name": "deployment",
            "model_name": "Deployment",
        }
        type_name = "deployment"
        model_name = "Deployment"
        model_uuid = str(uuid.uuid4())

        response = client.post(
            f"/user/{user_uuid}/models/{type_name}/{model_name}/{model_uuid}",
            json=model,
        )

        assert response.status_code != 200

    @pytest.mark.asyncio()
    async def test_background_task_not_called_on_error(self, user_uuid: str) -> None:
        team_uuid = str(uuid.uuid4())
        deployment_uuid = str(uuid.uuid4())
        gh_token_uuid = str(uuid.uuid4())
        fly_token_uuid = str(uuid.uuid4())

        model = {
            "name": "name",
            "repo_name": "repo_name",
            "fly_app_name": "Test",
            "team": {"uuid": team_uuid, "type": "team", "name": "TwoAgentTeam"},
            "gh_token": {
                "uuid": gh_token_uuid,
                "type": "secret",
                "name": "GitHubToken",
            },
            "fly_token": {"uuid": fly_token_uuid, "type": "secret", "name": "FlyToken"},
            "uuid": deployment_uuid,
            "type_name": "deployment",
            "model_name": "Deployment",
        }
        type_name = "deployment"
        model_name = "Deployment"
        model_uuid = uuid.uuid4()

        with (
            patch(
                "fastagency.app.DefaultDB._frontend_db.get_user",
                side_effect=Exception(),
            ),
            patch(
                "fastagency.db.prisma.PrismaBackendDB._get_db_connection",
                side_effect=Exception(),
            ),
            patch("fastagency.helpers.deploy_saas_app") as mock_task,
        ):
            response = client.post(
                f"/user/{user_uuid}/models/{type_name}/{model_name}/{model_uuid}",
                json=model,
            )

        mock_task.assert_not_called()
        assert response.status_code != 200

    @pytest.mark.asyncio()
    async def test_update_model(
        self, user_uuid: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_uuid = str(uuid.uuid4())
        azure_oai_api_key = AzureOAIAPIKey(api_key="who cares", name="whatever")
        type_name = "secret"
        model_name = "AzureOAIAPIKey"

        # Create model
        response = client.post(
            f"/user/{user_uuid}/models/{type_name}/{model_name}/{key_uuid}",
            json=azure_oai_api_key.model_dump(),
        )
        assert response.status_code == 200

        response = client.put(
            f"/user/{user_uuid}/models/secret/AzureOAIAPIKey/{key_uuid}",
            json=azure_oai_api_key.model_dump(),
        )

        assert response.status_code == 200
        expected = {
            "api_key": "who cares",  # pragma: allowlist secret
            "name": "whatever",
        }
        actual = response.json()
        assert actual == expected

    @pytest.mark.asyncio()
    async def test_update_model_deployment(self, user_uuid: str) -> None:
        team_uuid = str(uuid.uuid4())
        deployment_uuid = str(uuid.uuid4())
        gh_token_uuid = str(uuid.uuid4())
        fly_token_uuid = str(uuid.uuid4())
        model = {
            "name": "name",
            "repo_name": "repo_name",
            "fly_app_name": "Test",
            "team": {"uuid": team_uuid, "type": "team", "name": "TwoAgentTeam"},
            "gh_token": {
                "uuid": gh_token_uuid,
                "type": "secret",
                "name": "GitHubToken",
            },
            "fly_token": {"uuid": fly_token_uuid, "type": "secret", "name": "FlyToken"},
            "uuid": deployment_uuid,
            "type_name": "deployment",
            "model_name": "Deployment",
        }
        type_name = "deployment"
        model_name = "Deployment"

        model_uuid = str(uuid.uuid4())
        # Mock the background task
        fly_api_token = "some-token"
        fastagency_deployment_uuid = "some-uuid"
        github_token = "some-github-token"
        app_name = "test fastagency template"
        repo_name = "test-fastagency-template"
        fly_app_name = "test-fastagency-template"
        saas_app = SaasAppGenerator(
            fly_api_token,
            fastagency_deployment_uuid,
            github_token,
            app_name,
            repo_name,
            fly_app_name,
        )
        saas_app.gh_repo_url = "https://some-git-url"
        with (
            patch(
                "fastagency.helpers.validate_tokens_and_create_gh_repo",
                return_value=saas_app,
            ) as mock_task,
            patch("fastagency.helpers.deploy_saas_app"),
        ):
            response = client.post(
                f"/user/{user_uuid}/models/{type_name}/{model_name}/{model_uuid}",
                json=model,
            )
            mock_task.assert_called_once()

        assert response.status_code == 200
        # Update deployment
        new_gh_token_uuid = str(uuid.uuid4())
        model = {
            "name": "name",
            "repo_name": "repo_name",
            "fly_app_name": "Test",
            "team": {"uuid": team_uuid, "type": "team", "name": "TwoAgentTeam"},
            "gh_token": {
                "uuid": new_gh_token_uuid,
                "type": "secret",
                "name": "GitHubToken",
            },
            "fly_token": {"uuid": fly_token_uuid, "type": "secret", "name": "FlyToken"},
            "uuid": deployment_uuid,
            "type_name": "deployment",
            "model_name": "Deployment",
        }
        response = client.put(
            f"/user/{user_uuid}/models/deployment/Deployment/{model_uuid}",
            json=model,
        )

        assert response.status_code == 200
        expected = {
            "name": "name",
            "repo_name": "repo_name",
            "fly_app_name": "Test",
            "team": {
                "type": "team",
                "name": "TwoAgentTeam",
                "uuid": team_uuid,
            },
            "gh_token": {
                "type": "secret",
                "name": "GitHubToken",
                "uuid": new_gh_token_uuid,
            },
            "fly_token": {
                "type": "secret",
                "name": "FlyToken",
                "uuid": fly_token_uuid,
            },
        }

        actual = response.json()
        assert actual == expected

    @pytest.mark.asyncio()
    async def test_delete_model(
        self, user_uuid: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key_uuid = str(uuid.uuid4())
        azure_oai_api_key = AzureOAIAPIKey(api_key="whatever", name="whatever")
        type_name = "secret"
        model_name = "AzureOAIAPIKey"

        # Create model
        response = client.post(
            f"/user/{user_uuid}/models/{type_name}/{model_name}/{key_uuid}",
            json=azure_oai_api_key.model_dump(),
        )
        assert response.status_code == 200

        response = client.delete(f"/user/{user_uuid}/models/secret/{key_uuid}")

        assert response.status_code == 200
        expected = {
            "api_key": "whatever",  # pragma: allowlist secret
            "name": "whatever",
        }
        actual = response.json()
        assert actual == expected

    @pytest.mark.llm()
    @pytest.mark.asyncio()
    async def test_chat_with_no_function_calling(
        self, user_uuid: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_uuid = str(uuid.uuid4())
        model_name = "MultiAgentTeam"
        # Mocking the aclient.chat.completions.create function
        mock_create = AsyncMock()
        monkeypatch.setattr(
            "fastagency.app.aclient.chat.completions.create", mock_create
        )

        # Define the mock return value
        mock_create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(tool_calls=None))]
        )

        # Define the request body
        request_body = {
            "message": [{"role": "user", "content": "Hello"}],
            "chat_id": 123,
            "user_id": 456,
        }

        # Define the expected response
        expected_response = {
            "team_status": "inprogress",
            "team_name": "456_123",
            "team_id": 123,
            "customer_brief": "Some customer brief",
            "conversation_name": "Hello",
        }

        response = client.post(
            f"/user/{user_uuid}/chat/{model_name}/{model_uuid}", json=request_body
        )

        # Assert the status code and the response body
        assert response.status_code == 200
        assert response.json() == expected_response

        # Assert the mock was called with the correct arguments
        mock_create.assert_called_once()

    @pytest.mark.llm()
    @pytest.mark.asyncio()
    async def test_chat_error(
        self, user_uuid: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_uuid = str(uuid.uuid4())
        model_name = "MultiAgentTeam"

        mock_create = AsyncMock()
        monkeypatch.setattr(
            "fastagency.app.aclient.chat.completions.create", mock_create
        )
        mock_create.side_effect = Exception("Error creating chat completion")

        # Define the request body
        request_body = {
            "message": [{"role": "user", "content": "Hello"}],
            "chat_id": 123,
            "user_id": 456,
        }

        # Define the expected response
        expected_response = {
            "team_status": "inprogress",
            "team_name": "456_123",
            "team_id": 123,
            "customer_brief": "Some customer brief",
            "conversation_name": "Hello",
        }

        response = client.post(
            f"/user/{user_uuid}/chat/{model_name}/{model_uuid}", json=request_body
        )

        # Assert the status code and the response body
        assert response.status_code == 200
        assert response.json() == expected_response

        # Assert the mock was called with the correct arguments
        mock_create.assert_called_once()

    @pytest.mark.llm()
    @pytest.mark.asyncio()
    async def test_chat_with_function_calling(
        self, user_uuid: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model_uuid = str(uuid.uuid4())
        model_name = "MultiAgentTeam"

        mock_create = AsyncMock()
        monkeypatch.setattr(
            "fastagency.app.aclient.chat.completions.create", mock_create
        )

        function = Function(
            arguments='{\n  "chat_name": "Calculate 2 * 2"\n}',
            name="generate_chat_name",
        )
        tool_call = ChatCompletionMessageToolCall(
            id="1", function=function, type="function"
        )
        message = ChatCompletionMessage(
            content=None, role="assistant", function_call=None, tool_calls=[tool_call]
        )
        choice = Choice(message=message)
        chat_completion = MockChatCompletion(id="1", choices=[choice])

        mock_create.return_value = chat_completion

        # Define the request body
        request_body = {
            "message": [{"role": "user", "content": "Hello"}],
            "chat_id": 123,
            "user_id": 456,
        }

        # Define the expected response
        expected_response = {
            "team_status": "inprogress",
            "team_name": "456_123",
            "team_id": 123,
            "customer_brief": "Some customer brief",
            "conversation_name": "Calculate 2 * 2",
        }

        response = client.post(
            f"/user/{user_uuid}/chat/{model_name}/{model_uuid}", json=request_body
        )

        # Assert the status code and the response body
        assert response.status_code == 200
        assert response.json() == expected_response

        # Assert the mock was called with the correct arguments
        mock_create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_ping(self) -> None:
        deployment_uuid = str(uuid.uuid4())
        response = client.get(f"/deployment/{deployment_uuid}/ping")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
