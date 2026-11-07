"""Prepare a filtered OASIS Twitter simulation without calling an LLM.

Run from the OASIS checkout, for example:

    python examples/twitter_setup_until_llm.py

Edit the three experiment inputs below, or pass their paths with CLI flags.
The script writes an OASIS SQLite database, an input CSV, a manifest, and a
PNG of the selected follow graph into ``./artifacts/twitter_setup``.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
import igraph as ig
import shutil


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
    # Match OASIS's bundled CSV schema. The generator reads by column name,
    # but this makes the adapter easier to compare with OASIS examples.
    fieldnames = [
        "user_id",
        "name",
        "username",
        "following_agentid_list",
        "previous_tweets",
        "user_char",
        "description",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for username in selected:
            user = users[username]
            writer.writerow(
                {
                    "user_id": user.get("user_id", ""),
                    "name": user.get("name") or username,
                    "username": username,
                    "following_agentid_list": repr(following[username]),
                    "previous_tweets": repr([]),
                    "user_char": descriptions[username],
                    "description": user.get("description", ""),
                }
            )
    return len(selected), sum(map(len, following.values()))


def validate_agent_graph(agent_graph, profile_csv: Path, author_id: int) -> None:
    """Fail loudly if OASIS dropped any profile-defined follow edges."""
    profile = pd.read_csv(profile_csv)
    expected = {
        (agent_id, followee_id)
        for agent_id, value in enumerate(profile["following_agentid_list"])
        for followee_id in ast.literal_eval(value)
    }
    actual = set(agent_graph.graph.get_edgelist())
    missing = expected - actual
    unexpected = actual - expected

    if missing or unexpected:

        def names(edges):
            return [
                f"{profile.iloc[source]['username']} -> "
                f"{profile.iloc[target]['username']}"
                for source, target in sorted(edges)
            ]

        expected_followers = {
            source for source, target in expected if target == author_id
        }
        actual_followers = {source for source, target in actual if target == author_id}
        raise RuntimeError(
            "OASIS AgentGraph edge mismatch.\n"
            f"Expected: {len(expected)}; actual: {len(actual)}.\n"
            f"Missing: {names(missing)}\n"
            f"Unexpected: {names(unexpected)}\n"
            "Expected direct followers of author: "
            f"{[profile.iloc[i]['username'] for i in sorted(expected_followers)]}\n"
            "Actual direct followers of author: "
            f"{[profile.iloc[i]['username'] for i in sorted(actual_followers)]}"
        )


def visualize_propagation_bfs(
    agent_graph, root_agent_id: int, output_path: Path
) -> None:
    """Save a username-labelled BFS tree rooted at the propagation author.

    OASIS stores ``follower -> followee`` edges. Therefore incoming graph
    neighbors are the accounts that can receive a post from the current node.
    """
    graph = agent_graph.graph
    distances = {root_agent_id: 0}
    tree_edges: list[tuple[int, int]] = []
    queue = deque([root_agent_id])
    while queue:
        node = queue.popleft()
        for follower in graph.neighbors(node, mode="in"):
            if follower not in distances:
                distances[follower] = distances[node] + 1
                # Reverse the stored follower -> followee edge so the image
                # shows information flowing author -> follower.
                tree_edges.append((node, follower))
                queue.append(follower)

    reachable = sorted(distances)
    if not reachable:
        raise ValueError("Propagation author is not present in the agent graph")
    old_to_new = {old: new for new, old in enumerate(reachable)}
    tree_edge_set = set(tree_edges)
    # Include all relationships among reachable nodes. The stored edge is
    # follower -> followee, so reverse it for propagation-oriented drawing.
    all_edges = {
        (edge.target, edge.source)
        for edge in graph.es
        if edge.source in old_to_new and edge.target in old_to_new
    }
    cross_edges = sorted(all_edges - tree_edge_set)
    drawing_edges = tree_edges + cross_edges
    subgraph = graph.__class__(n=len(reachable), directed=True)
    subgraph.add_edges(
        [
            (old_to_new[parent], old_to_new[follower])
            for parent, follower in drawing_edges
        ]
    )
    labels = [agent_graph.get_agent(old).user_info.name for old in reachable]
    levels: dict[int, list[int]] = {}
    for old, new in old_to_new.items():
        levels.setdefault(distances[old], []).append(new)

    # A deterministic layered layout makes the BFS distance visually clear.
    coords = [(0.0, 0.0)] * len(reachable)
    for level, nodes in levels.items():
        for index, vertex_id in enumerate(nodes):
            coords[vertex_id] = (float(level), -(index - (len(nodes) - 1) / 2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ig.plot(
        subgraph,
        target=str(output_path),
        layout=coords,
        vertex_label=labels,
        vertex_label_dist=2.0,
        vertex_label_angle=-math.pi / 2,
        edge_color=["#555555"] * len(tree_edges) + ["#4c78a8"] * len(cross_edges),
        vertex_size=28,
        vertex_color="#f74f1b",
        vertex_frame_width=2,
        edge_arrow_size=0.5,
        bbox=(1800, max(900, 220 * max(levels) + 300)),
        margin=100,
    )


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
        author = config.get("post_author_username") or usernames[0]
        if author not in usernames:
            raise ValueError("post_author_username must be in usernames")
        author_id = usernames.index(author)
        validate_agent_graph(agent_graph, profile_csv, author_id)
        visualize_propagation_bfs(agent_graph, author_id, graph_png)
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
    shutil.rmtree("./artifacts", ignore_errors=True)
    asyncio.run(prepare(parse_args()))
