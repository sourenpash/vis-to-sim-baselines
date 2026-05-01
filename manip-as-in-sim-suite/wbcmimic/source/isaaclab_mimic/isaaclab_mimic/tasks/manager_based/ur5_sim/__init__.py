# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-UR5-PutBowlInMicroWaveAndClose-OneCamera-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5PutBowlInMicroWaveAndCloseMimicEEControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_put_bowl_in_microwave_and_close:UR5PutBowlInMicroWaveAndCloseOneCameraMimicEnvCfg",
    },
)

gym.register(
    id="Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-OneCamera-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5PutBowlInMicroWaveAndCloseMimicJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_put_bowl_in_microwave_and_close:UR5PutBowlInMicroWaveAndCloseJointWbcGoHomeOneCameraMimicCfg",
    },
)

gym.register(
    id="Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-OneCamera-Mimic-MP-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5PutBowlInMicroWaveAndCloseMimicJointControlMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_put_bowl_in_microwave_and_close:UR5PutBowlInMicroWaveAndCloseJointWbcGoHomeOneCameraMimicCfg",
    },
)

gym.register(
    id="Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-Retry-Place-OneCamera-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5PutBowlInMicroWaveAndCloseMimicJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_put_bowl_in_microwave_and_close:UR5PutBowlInMicroWaveAndCloseJointWbcGoHomeRetryPlaceOneCameraMimicCfg",
    },
)

gym.register(
    id="Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-Retry-Place-OneCamera-NoCrop-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5PutBowlInMicroWaveAndCloseMimicJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_put_bowl_in_microwave_and_close:UR5PutBowlInMicroWaveAndCloseJointWbcGoHomeRetryPlaceNoCropOneCameraMimicCfg",
    },
)


gym.register(
    id="Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-Retry-Place-OneCamera-Mimic-MP-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5PutBowlInMicroWaveAndCloseMimicJointControlMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_put_bowl_in_microwave_and_close:UR5PutBowlInMicroWaveAndCloseJointWbcGoHomeRetryPlaceOneCameraMimicCfg",
    },
)

gym.register(
    id="Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-GoHome-Retry-Place-OneCamera-NoCrop-Mimic-MP-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5PutBowlInMicroWaveAndCloseMimicJointControlMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_put_bowl_in_microwave_and_close:UR5PutBowlInMicroWaveAndCloseJointWbcGoHomeRetryPlaceNoCropOneCameraMimicCfg",
    },
)

gym.register(
    id="Isaac-UR5-PutBowlInMicroWaveAndClose-Joint-Mimic-Annotated-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5PutBowlInMicroWaveAndCloseMimicJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_put_bowl_in_microwave_and_close:UR5PutBowlInMicroWaveAndCloseJointWbcMimicAnnotateCfg",
    },
)


# ================ Clean Plate ==================

gym.register(
    id="Isaac-UR5-CleanPlate-Joint-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5CleanPlateEEControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_clean_plate:UR5CleanPlateMimicEnvCfg",
    },
)

gym.register(
    id="Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-Annotate-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5CleanPlateMimicJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_clean_plate:UR5CleanPlateJointWbcAnnotateMimicCfg",
    },
)

gym.register(
    id="Isaac-UR5-CleanPlate-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5CleanPlateEEControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_clean_plate:UR5CleanPlateMimicEnvCfg",
    },
)

gym.register(
    id="Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5CleanPlateMimicJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_clean_plate:UR5CleanPlateJointWbcGoHomeMimicCfg",
    },
)


gym.register(
    id="Isaac-UR5-CleanPlate-Joint-GoHome-Mimic-MP-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5CleanPlateMimicJointControlMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_clean_plate:UR5CleanPlateJointWbcGoHomeMimicCfg",
    },
)

gym.register(
    id="Isaac-UR5-CleanPlate-Joint-GoHome-Retry-Mimic-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5CleanPlateMimicJointControlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_clean_plate:UR5CleanPlateJointWbcGoHomeRetryPlaceMimicCfg",
    },
)


gym.register(
    id="Isaac-UR5-CleanPlate-Joint-GoHome-Retry-Mimic-MP-v0",
    entry_point="isaaclab_mimic.tasks.manager_based.ur5_sim.envs:UR5CleanPlateMimicJointControlMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5_clean_plate:UR5CleanPlateJointWbcGoHomeRetryPlaceMimicCfg",
    },
)
