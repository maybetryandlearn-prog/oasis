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


class CostGuard:
    def __init__(self, config: dict, ledger_path: Path):
        self.threshold = float(
            config.get("cost_guard", {}).get("max_cost_usd", 0.50)
        )
        self.actual_cost = 0.0
        self.responses = 0
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def record_response(self, agent_id: int, timestep: int, response) -> bool:
        usage = getattr(response, "usage", None)
        cost = getattr(usage, "cost", 0) if usage is not None else 0
        if not cost:
            info = getattr(response, "info", {}) or {}
            info_usage = info.get("usage", {}) if isinstance(info, dict) else {}
            cost = info_usage.get("cost", 0) if isinstance(info_usage, dict) else 0
        cost = float(cost or 0)
        self.responses += 1
        self.actual_cost += cost
        self.write({
            "event": "response",
            "agent_id": agent_id,
            "timestep": timestep,
            "cost_usd": cost,
            "total_cost_usd": self.actual_cost,
        })
        return self.actual_cost >= self.threshold

    def write(self, record: dict) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")


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
    guard = CostGuard(config, artifacts_dir / "cost_ledger.jsonl")
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

        stop_simulation = False
        for timestep in range(1, timesteps + 1):
            # Match OasisEnv.step() without losing each LLM response, which
            # lets the ledger record usage when CAMEL exposes it.
            await env.platform.update_rec_table()
            actions = {}
            for agent_id, agent in env.agent_graph.get_agents():
                actions[agent] = LLMAction()
            for agent in actions:
                response = await agent.perform_action_by_llm()
                if guard.record_response(agent.social_agent_id, timestep, response):
                    print(f"Cost threshold reached: ${guard.actual_cost:.6f}")
                    stop_simulation = True
                    break
            env.platform.sandbox_clock.time_step += 1
            graph_path = artifacts_dir / f"simulation_timestep_{timestep}.png"
            # This records the evolving follow graph; posts/reposts remain in DB.
            env.agent_graph.visualize(str(graph_path), with_labels=True)
            print(f"Completed timestep {timestep}/{timesteps}")
            if stop_simulation:
                break
        print(
            f"Cost ledger: {guard.ledger_path}; "
            f"actual reported cost: ${guard.actual_cost:.6f}"
        )
        if guard.responses and guard.actual_cost == 0:
            print("Warning: no response usage.cost was exposed by the model wrapper.")
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
