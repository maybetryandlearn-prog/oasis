# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Run a Twitter simulation with VLLM and CSV-defined follow edges.

Unlike ``twitter_simulation_vllm.py``, this example uses the data-aware
``generate_agents`` initializer.  That initializer loads users, historical
posts, and ``following_agentid_list`` into the platform database.
"""

import asyncio
import os
import random

from camel.models import ModelFactory, ModelManager
from camel.types import ModelPlatformType

from oasis import ActionType
from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform


DEFAULT_PROFILE_PATH = (
    "data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv")
DEFAULT_DB_PATH = "./data/twitter_simulation_vllm_with_edges.db"


async def running(
    profile_path: str = DEFAULT_PROFILE_PATH,
    db_path: str = DEFAULT_DB_PATH,
    num_timesteps: int = 4,
    clock_factor: int = 60,
) -> None:
    """Run the VLLM simulation using the relationships in ``profile_path``."""
    model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="qwen-2",
        url="http://10.109.28.7:8080/v1",
    )
    model_2 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="qwen-2",
        url="http://10.109.27.103:8080/v1",
    )
    model_manager = ModelManager(
        models=[model_1, model_2],
        scheduling_strategy="round_robin",
    )

    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    channel = Channel()
    clock = Clock(k=clock_factor)
    infra = Platform(
        db_path=db_path,
        channel=channel,
        sandbox_clock=clock,
        start_time=0,
        recsys_type="twitter",
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )
    platform_task = asyncio.create_task(infra.running())

    try:
        agent_graph = await generate_agents(
            agent_info_path=profile_path,
            channel=channel,
            start_time=0,
            model=model_manager,
            recsys_type="twitter",
            twitter=infra,
            available_actions=ActionType.get_default_twitter_actions(),
        )

        for timestep in range(1, num_timesteps + 1):
            clock.time_step = timestep * 3
            await infra.update_rec_table()
            tasks = []
            for _, agent in agent_graph.get_agents():
                if (agent.user_info.is_controllable is False
                        and random.random() < 1.0):
                    tasks.append(agent.perform_action_by_llm())
            await asyncio.gather(*tasks)
    finally:
        await channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await platform_task


if __name__ == "__main__":
    asyncio.run(running())
