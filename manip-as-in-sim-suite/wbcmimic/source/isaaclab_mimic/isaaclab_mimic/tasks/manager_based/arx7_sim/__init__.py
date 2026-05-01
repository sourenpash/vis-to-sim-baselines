# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##
gym.register(
    id="Isaac-X7-PickToothpasteIntoCupAndPush-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.arx7_sim.envs:X7PickToothpasteIntoCupAndPushMimicEEControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7_pick_toothpaste_into_cup_and_push:X7PickToothpasteIntoCupAndPushMimicCfg",
    },
)

gym.register(
    id="Isaac-X7-PickToothpasteIntoCupAndPush-Joint-Mimic-Annotate-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.arx7_sim.envs:X7PickToothpasteIntoCupAndPushMimicJointWbcControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7_pick_toothpaste_into_cup_and_push:X7PickToothpasteIntoCupAndPushJointWbcMimicCfgAnnotated",
    },
)

gym.register(
    id="Isaac-X7-PickToothpasteIntoCupAndPush-Joint-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.arx7_sim.envs:X7PickToothpasteIntoCupAndPushMimicJointWbcControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7_pick_toothpaste_into_cup_and_push:X7PickToothpasteIntoCupAndPushJointWbcMimicCfg",
    },
)

gym.register(
    id="Isaac-X7-PickToothpasteIntoCupAndPush-Joint-Mimic-MP-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.arx7_sim.envs:X7PickToothpasteIntoCupAndPushMimicJointWbcControlMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7_pick_toothpaste_into_cup_and_push:X7PickToothpasteIntoCupAndPushJointWbcMimicCfg",
    },
)


# gym.register(
#     id="Isaac-X7-PickToothpasteIntoCupAndPush-Joint-Mimic-MP-v0",
#     entry_point="isaaclab_mimic.tasks.manager_based.arx7_sim.envs:X7PickToothpasteIntoCupAndPushMimicEEControlEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.x7_pick_toothpaste_into_cup_and_push:X7PickToothpasteIntoCupAndPushMimicCfg",
#     },
# )

gym.register(
    id="Isaac-X7-PickToothpasteIntoCupAndPush-Wbc-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.arx7_sim.envs:X7PickToothpasteIntoCupAndPushMimicJointWbcControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7_pick_toothpaste_into_cup_and_push:X7PickToothpasteIntoCupAndPushJointWbcMimicCfg",
    },
)

gym.register(
    id="Isaac-X7-PickToothpasteIntoCupAndPush-Wbc-Mimic-Annotated-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.arx7_sim.envs:X7PickToothpasteIntoCupAndPushMimicJointWbcControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7_pick_toothpaste_into_cup_and_push:X7PickToothpasteIntoCupAndPushJointWbcMimicCfgAnnotated",
    },
)

gym.register(
    id="Isaac-X7-PickToothpasteIntoCupAndPush-Joint-Mimic-rotate-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.arx7_sim.envs:X7PickToothpasteIntoCupAndPushMimicJointWbcControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7_pick_toothpaste_into_cup_and_push:X7PickToothpasteIntoCupAndPushJointRotateWbcMimicCfg",
    },
)


if __name__ == "__main__":
    env = gym.make("Isaac-X7-PourWater-Mimic-v0")
    env.reset()
    env.close()
