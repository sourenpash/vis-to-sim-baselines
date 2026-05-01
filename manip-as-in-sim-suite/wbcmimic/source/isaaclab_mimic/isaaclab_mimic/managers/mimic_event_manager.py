# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import TYPE_CHECKING, Type

import torch
from isaaclab.managers import ManagerBase, ManagerTermBaseCfg
from isaaclab.utils import configclass
from loguru import logger as lgr

if TYPE_CHECKING:
    from ..envs import ManagerBasedRLMimicEnv

from dataclasses import MISSING
from functools import reduce
from typing import Callable, Dict, List, Sequence, Tuple

from isaaclab.envs.mimic_env_cfg import SubTaskConfig


def empty_func(env, env_ids, **kwargs):
    """A placeholder function that does nothing.

    Args:
        env: The environment object.
        env_ids: The environment indices.
        **kwargs: Additional keyword arguments.
    """
    pass


class TriggerBase(ManagerBase):
    def __init__(self, cfg: "MimicEventTermCfg", env: "ManagerBasedRLMimicEnv"):
        super().__init__(cfg, env)
        self.cfg: "MimicEventTermCfg"
        self.env: "ManagerBasedRLMimicEnv"

    def _prepare_terms(self):
        pass

    def active_terms(self):
        return

    @abstractmethod
    def __call__(self, subtask_idx: torch.Tensor, trigged_times: torch.Tensor):
        """Check if the event should be triggered.

        Args:
            subtask_idx (torch.Tensor): The index of the subtask.
            trigged_times (torch.Tensor): The number of times the event has been triggered.

        Returns:
            bool: True if the event should be triggered, False otherwise.
        """
        pass


class TriggerRandomPerStep(TriggerBase):
    def __init__(self, cfg: "MimicEventTermCfg", env: "ManagerBasedRLMimicEnv"):
        super().__init__(cfg, env)

    def __call__(
        self,
        subtask_idx: torch.Tensor,
        trigged_times: torch.Tensor,
        subtask_term_signal_ids: List[int],
        prob: float,
        max_trigged_times: int = 1,
    ):
        """Check if the event should be triggered."""
        env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        mask = torch.zeros_like(subtask_idx, dtype=torch.bool)
        for subtask_id in subtask_term_signal_ids:
            mask = torch.logical_or(mask, subtask_idx == subtask_id)
        mask = torch.logical_and(mask, trigged_times[:, subtask_id] < max_trigged_times)
        env_ids = env_ids[mask]
        mask = torch.rand_like(env_ids, dtype=torch.float) < prob
        return env_ids[mask]


@configclass
class MimicEventTermCfg(ManagerTermBaseCfg):
    """Configuration for a mimic event term."""

    func: Callable[..., None] = empty_func
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input.
    """

    trigger_class: Type[TriggerBase] = TriggerRandomPerStep
    """When the event should be triggered."""

    trigger_params: Dict = {}
    # subtask_term_signal: str = MISSING
    # """During which subtask the event should be triggered."""

    # prob: float = 0.001
    # """The probability of the event being triggered."""

    continue_subtask_term_signal: str = MISSING

    eef_name: str = MISSING


class MimicEventManager(ManagerBase):
    def __init__(self, cfg: object, env: "ManagerBasedRLMimicEnv"):
        self._reseted: Dict[str, Dict[str, torch.Tensor]] = {}
        self._term_cfgs = []
        self._term_name_to_cfg = {}
        self._eef_name_to_terms: Dict[
            str, List[Tuple[Callable, Dict, TriggerBase, MimicEventTermCfg]]
        ] = {}
        super().__init__(cfg, env)
        self._env: "ManagerBasedRLMimicEnv"
        self.one_step_most_num_event: int = getattr(cfg, "one_step_most_num_event", 1)

    def _prepare_terms(self):
        for eef_name, subtask_term_signals in self._env.subtasks.items():
            self._reseted[eef_name] = {}
            self._eef_name_to_terms[eef_name] = []

        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        self._term_name_to_cfg = cfg_items

        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue

            # check for valid config type
            if not isinstance(term_cfg, MimicEventTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type MimicEventTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            self._eef_name_to_terms[term_cfg.eef_name].append(
                (
                    term_cfg.func,
                    term_cfg.params,
                    term_cfg.trigger_class(term_cfg, self._env),
                    term_cfg,
                )
            )
            self._reseted[eef_name][str(term_cfg.trigger_class.__name__)] = torch.zeros(
                (self._env.num_envs, len(subtask_term_signals)),
                dtype=torch.int,
                device=self._env.device,
            )

    @property
    def active_terms(self):
        return self._term_name_to_cfg

    def pre_reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # TODO: not handle seed of reset
        pass

    def post_reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        for eef_name in self._env.subtasks:
            for key in self._reseted[eef_name]:
                self._reseted[eef_name][key][env_ids, :] = 0
        return {}

    def pre_step(self):
        pass

    def safe_check(self, env_ids, eef_name: str) -> bool:
        r = True
        for env_id in env_ids:
            idx = int(env_id)
            r = (
                r
                and (idx in self._env.data_generator.current_eef_subtask_indices)
                and (
                    eef_name
                    in self._env.data_generator.current_eef_subtask_indices[idx]
                )
            )
        return r

    def post_step(self):
        # FIXME: There may contains bug in dual arm setting.
        # only support wbc now
        subtask_terms = self._env.obs_buf["subtask_terms"]

        for eef_name in self._env.subtasks:
            current_subtask_id = torch.zeros(
                (self._env.num_envs,), dtype=torch.int, device=self._env.device
            )
            for env_id in self._env.data_generator.current_eef_subtask_indices.keys():
                current_subtask_id[env_id] = (
                    self._env.data_generator.current_eef_subtask_indices[env_id][
                        eef_name
                    ]
                )

            if not self.safe_check(range(self._env.num_envs), eef_name):
                continue
            # for subtask_idx in self._subtask_idx_to_funcs[eef_name]:
            local_one_step_most_num_event = 0
            for i in range(len(self._eef_name_to_terms[eef_name])):
                func, params, trigger_class, term_cfg = self._eef_name_to_terms[
                    eef_name
                ][i]
                env_ids = trigger_class(
                    current_subtask_id,
                    self._reseted[eef_name][str(trigger_class.__class__.__name__)],
                    **term_cfg.trigger_params,
                )
                func(self._env, env_ids, **params)
                continue_subtask_term_signal_idx = self._env.subtask_term_signal_to_idx[
                    eef_name
                ][term_cfg.continue_subtask_term_signal]
                self._reseted[eef_name][str(trigger_class.__class__.__name__)][
                    env_ids, current_subtask_id[env_ids]
                ] += 1
                if self._env.data_generator is not None:
                    for env_id in env_ids:
                        self._env.data_generator.current_eef_subtask_indices[
                            int(env_id)
                        ][eef_name] = continue_subtask_term_signal_idx
                        if hasattr(self._env, "success_term") and hasattr(
                            self._env.success_term, "_subtask_history"
                        ):
                            self._env.success_term._subtask_history[eef_name][
                                env_id, continue_subtask_term_signal_idx:
                            ] = 0
                            lgr.trace(
                                "Cleared subtask history for",
                                eef_name,
                                "at index",
                                continue_subtask_term_signal_idx,
                            )
                        self._env.data_generator.current_eef_subtask_step_indices[
                            int(env_id)
                        ][eef_name] = 0
                        del self._env.data_generator.current_eef_subtask_trajectories[
                            int(env_id)
                        ][eef_name]
                        self._env.data_generator.current_eef_subtask_trajectories[
                            int(env_id)
                        ][eef_name] = self._env.data_generator.generate_trajectory(
                            int(env_id),
                            eef_name,
                            self._env.data_generator.current_eef_subtask_indices[
                                int(env_id)
                            ][eef_name],
                            self._env.data_generator.randomized_subtask_boundaries[
                                int(env_id)
                            ],
                            self._env.data_generator.runtime_subtask_constraints_dict[
                                int(env_id)
                            ],
                            self._env.data_generator.current_eef_selected_src_demo_indices[
                                int(env_id)
                            ],
                            self._env.data_generator.current_eef_subtask_trajectories[
                                int(env_id)
                            ],
                            wbc=True,
                        )
                else:
                    lgr.warning(
                        "Warning: No data generator registered. Skipping mimic event."
                    )
                local_one_step_most_num_event += 1 if len(env_ids) > 0 else 0
                if local_one_step_most_num_event >= self.one_step_most_num_event:
                    break

    def __str__(self):
        return "Use Mimic Event Manager"
