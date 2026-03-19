from orgemage.catalog import FederatedModelCatalog
from orgemage.models import AgentCapabilities, DownstreamAgentConfig, ModelOption


def test_federated_model_catalog_resolves_composite_values() -> None:
    catalog = FederatedModelCatalog(
        [
            DownstreamAgentConfig(
                agent_id="codex",
                name="Codex",
                command="codex",
                models=[ModelOption(value="gpt-5-codex", name="GPT-5 Codex")],
                capabilities=AgentCapabilities(),
            ),
            DownstreamAgentConfig(
                agent_id="gemini",
                name="Gemini",
                command="gemini",
                models=[ModelOption(value="gemini-2.5-pro", name="Gemini 2.5 Pro")],
                capabilities=AgentCapabilities(),
            ),
        ]
    )

    options = catalog.northbound_model_options()

    assert [item["value"] for item in options] == [
        "gemini::gemini-2.5-pro",
        "codex::gpt-5-codex",
    ]
    assert options[0]["metadata"]["degraded"] is True
    resolved = catalog.resolve("codex::gpt-5-codex")
    assert resolved.agent.agent_id == "codex"
    assert resolved.option.value == "gpt-5-codex"


def test_federated_model_catalog_prefers_discovered_models_and_tracks_refresh_state() -> (
    None
):
    catalog = FederatedModelCatalog(
        [
            DownstreamAgentConfig(
                agent_id="codex",
                name="Codex",
                command="codex",
                models=[ModelOption(value="bootstrap-model", name="Bootstrap Model")],
                capabilities=AgentCapabilities(),
            )
        ]
    )

    catalog.record_discovery(
        agent_id="codex",
        config_options=[
            {
                "id": "model",
                "category": "model",
                "type": "select",
                "options": [
                    {
                        "value": "discovered-a",
                        "name": "Discovered A",
                        "description": "Primary",
                    },
                    {"value": "discovered-b", "name": "Discovered B"},
                ],
            }
        ],
        capabilities={"commands": ["read", "edit"]},
        command_advertisements=["read", "edit"],
    )

    options = catalog.northbound_model_options()

    assert [item["value"] for item in options] == [
        "codex::discovered-a",
        "codex::discovered-b",
    ]
    assert all(item["metadata"]["source"] == "discovery" for item in options)
    assert all(item["metadata"]["trusted"] is True for item in options)
    assert catalog.runtime_state("codex").command_advertisements == ["read", "edit"]
    assert catalog.resolve("codex::discovered-b").option.name == "Discovered B"


def test_federated_model_catalog_uses_bootstrap_fallback_after_failed_discovery() -> (
    None
):
    catalog = FederatedModelCatalog(
        [
            DownstreamAgentConfig(
                agent_id="codex",
                name="Codex",
                command="codex",
                models=[ModelOption(value="bootstrap-model", name="Bootstrap Model")],
                capabilities=AgentCapabilities(),
            )
        ]
    )

    catalog.record_discovery_failure("codex", "temporary timeout")
    options = catalog.northbound_model_options()

    assert options == [
        {
            "value": "codex::bootstrap-model",
            "name": "Bootstrap Model (Codex)",
            "description": "",
            "category": "model",
            "metadata": {
                "agent_id": "codex",
                "source": "bootstrap",
                "trusted": False,
                "degraded": True,
                "last_refresh_started_at": catalog.runtime_state(
                    "codex"
                ).last_refresh_started_at,
                "last_refresh_succeeded_at": None,
                "last_refresh_failed_at": catalog.runtime_state(
                    "codex"
                ).last_refresh_failed_at,
                "last_refresh_error": "temporary timeout",
            },
        }
    ]


def test_bootstrap_catalog_uses_codex_acp_default() -> None:
    from orgemage import cli

    agents = cli._default_agents()

    codex = next(agent for agent in agents if agent.agent_id == "codex")
    assert codex.command == "codex-acp"
