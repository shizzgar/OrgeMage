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

    assert [item["value"] for item in options] == ["gemini::gemini-2.5-pro", "codex::gpt-5-codex"]
    resolved = catalog.resolve("codex::gpt-5-codex")
    assert resolved.agent.agent_id == "codex"
    assert resolved.option.value == "gpt-5-codex"
