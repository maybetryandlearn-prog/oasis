"""Prepare a filtered OASIS Twitter simulation without calling an LLM.

Run from the OASIS checkout, for example:

    python examples/twitter_setup_until_llm.py

Edit the three experiment inputs below, or pass their paths with CLI flags.
The script writes an OASIS SQLite database, an input CSV, a manifest, and a
PNG of the selected follow graph into ``./artifacts/twitter_setup``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform


DEFAULT_USERS_CSV = Path("../../data/users.csv")
DEFAULT_EDGES_CSV = Path("../../data/edges_usernames.csv")
DEFAULT_OUTPUT_DIR = Path("./artifacts/twitter_setup")
DEFAULT_CONFIG = Path("./twitter_setup_config.json")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    required = {"usernames", "propagation_post", "model"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    if not config["usernames"]:
        raise ValueError("Config 'usernames' must contain at least one username")
    return config


def create_model(model_config: dict):
    """Create an OpenAI-compatible model; no network request is made here."""
    if not model_config.get("api_key"):
        return None
    if model_config.get("platform", "openai") != "openai":
        raise ValueError("This setup script currently supports platform='openai'")
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=model_config["model_name"],
        api_key=model_config["api_key"],
        url=model_config.get("endpoint"),
    )


def load_descriptions(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    return {item["username"]: item["description"] for item in records}


def build_oasis_profile_csv(
    usernames: list[str],
    descriptions: dict[str, str],
    users_csv: Path,
    edges_csv: Path,
    output_csv: Path,
) -> tuple[int, int]:
    """Convert username-based data into the CSV expected by ``generate_agents``."""
    selected = list(dict.fromkeys(usernames))
    if not selected:
        raise ValueError("USERNAMES must contain at least one username")

    with users_csv.open(newline="", encoding="utf-8-sig") as file:
        users = {row["username"]: row for row in csv.DictReader(file)}
    missing_users = sorted(set(selected) - users.keys())
    if missing_users:
        raise ValueError(f"Usernames missing from {users_csv}: {missing_users}")

    missing_descriptions = sorted(set(selected) - descriptions.keys())
    if missing_descriptions:
        raise ValueError(
            f"Descriptions missing from descriptions JSON: {missing_descriptions}"
        )

    index = {username: agent_id for agent_id, username in enumerate(selected)}
    following: dict[str, list[int]] = {username: [] for username in selected}
    with edges_csv.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.reader(file):
            if len(row) < 2 or row[0] not in index or row[1] not in index:
                continue
            following[row[0]].append(index[row[1]])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "username",
        "name",
        "description",
        "user_char",
        "following_agentid_list",
        "previous_tweets",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for username in selected:
            user = users[username]
            writer.writerow(
                {
                    "username": username,
                    "name": user.get("name") or username,
                    "description": descriptions[username],
                    "user_char": descriptions[username],
                    "following_agentid_list": repr(following[username]),
                    "previous_tweets": repr([]),
                }
            )
    return len(selected), sum(map(len, following.values()))


async def prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    usernames = config["usernames"]
    descriptions = load_descriptions(Path("../../data/accountsummeries.json"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_csv = args.output_dir / "oasis_profiles.csv"
    database_path = args.output_dir / "twitter_setup.db"
    graph_png = args.output_dir / "follow_graph.png"
    manifest_path = args.output_dir / "manifest.json"

    node_count, edge_count = build_oasis_profile_csv(
        usernames, descriptions, args.users, args.edges, profile_csv
    )

    channel = Channel()
    platform = Platform(
        db_path=str(database_path),
        channel=channel,
        sandbox_clock=Clock(k=60),
        start_time=datetime.now(timezone.utc),
        recsys_type="twitter",
    )
    try:
        # Model construction is local only; no LLM call occurs until an agent
        # performs an action. Leaving api_key empty keeps this setup offline.
        agent_graph = await generate_agents(
            agent_info_path=str(profile_csv),
            channel=channel,
            start_time=0,
            model=create_model(config["model"]),
            recsys_type="twitter",
            twitter=platform,
        )
        agent_graph.visualize(str(graph_png), with_labels=True)

        author = config.get("post_author_username") or usernames[0]
        if author not in usernames:
            raise ValueError("post_author_username must be in usernames")
        author_id = usernames.index(author)
        platform.db.execute(
            "INSERT INTO post (user_id, content, created_at, num_likes, num_dislikes) "
            "VALUES (?, ?, ?, 0, 0)",
            (author_id, config["propagation_post"], 0),
        )
        platform.db.commit()
    finally:
        platform.db_cursor.close()
        platform.db.close()

    manifest_path.write_text(
        json.dumps(
            {
                "database": str(database_path),
                "profile_csv": str(profile_csv),
                "graph_png": str(graph_png),
                "propagation_post": config["propagation_post"],
                "post_author_username": author,
                "agent_count": node_count,
                "follow_edge_count": edge_count,
                "llm_called": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Created {node_count} agents and {edge_count} follow edges")
    print(f"Database: {database_path}")
    print(f"Graph:    {graph_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS_CSV)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES_CSV)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(prepare(parse_args()))
