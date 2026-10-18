from pathlib import Path


SCRIPT = Path("examples/twitter_simulation_vllm_with_edges.py")


def test_vllm_edge_aware_example_uses_data_aware_generator():
    source = SCRIPT.read_text()

    assert "from oasis.social_agent.agents_generator import generate_agents" in source
    assert "await generate_agents(" in source
    assert "twitter=infra" in source
    assert "generate_twitter_agent_graph" not in source
