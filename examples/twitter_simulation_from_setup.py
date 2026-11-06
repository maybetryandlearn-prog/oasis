"""Run phase two of the filtered Twitter simulation.

Phase one must already have created ``artifacts/twitter_setup``. This script
then posts the configured propagation post as a manual first action and lets
the selected OASIS agents act with LLM actions for subsequent timesteps.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
from pathlib import Path

import pandas as pd
import oasis
from oasis import ActionType, LLMAction, ManualAction
from oasis.social_agent.agents_generator import generate_twitter_agent_graph

from twitter_setup_until_llm import create_model, load_config


DEFAULT_CONFIG = Path("./twitter_setup_config.json")
DEFAULT_ARTIFACTS = Path("./artifacts/twitter_setup")


def add_profile_edges(agent_graph, profile_csv: Path) -> None:
    profile = pd.read_csv(profile_csv)
    for source_id, value in enumerate(profile["following_agentid_list"]):
        for target_id in ast.literal_eval(value):
            agent_graph.add_edge(source_id, target_id)


async def run(config_path: Path, artifacts_dir: Path, timesteps: int) -> None:
    config = load_config(config_path)
    profile_csv = artifacts_dir / "oasis_profiles.csv"
    database_path = artifacts_dir / "twitter_setup.db"
    if not profile_csv.exists() or not database_path.exists():
        raise FileNotFoundError(
            "Run twitter_setup_until_llm.py before this simulation script."
        )

    available_actions = ActionType.get_default_twitter_actions()
    model = create_model(config["model"])
    agent_graph = await generate_twitter_agent_graph(
        profile_path=str(profile_csv),
        model=model,
        available_actions=available_actions,
    )
    add_profile_edges(agent_graph, profile_csv)

    author_name = config.get("post_author_username") or config["usernames"][0]
    author_id = config["usernames"].index(author_name)
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=str(database_path),
    )

    try:
        await env.reset()

        # First action: seed the propagation experiment with the author's post.
        await env.step({
            env.agent_graph.get_agent(author_id): ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": config["propagation_post"]},
            )
        })

        for timestep in range(1, timesteps + 1):
            actions = {
                agent: LLMAction()
                for _, agent in env.agent_graph.get_agents()
            }
            await env.step(actions)
            graph_path = artifacts_dir / f"simulation_timestep_{timestep}.png"
            # This records the evolving follow graph; posts/reposts remain in DB.
            env.agent_graph.visualize(str(graph_path), with_labels=True)
            print(f"Completed timestep {timestep}/{timesteps}")
    finally:
        await env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--timesteps", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(run(args.config, args.artifacts_dir, args.timesteps))


if __name__ == "__main__":
    main()
