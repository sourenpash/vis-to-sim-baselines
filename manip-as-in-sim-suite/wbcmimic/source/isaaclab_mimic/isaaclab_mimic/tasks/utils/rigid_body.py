# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
import omni.log
import omni.physics.tensors.impl.api as physx
from isaaclab.assets import RigidObject, RigidObjectCfg, RigidObjectData
from pxr import UsdPhysics


class RigidObjectInitDataAuto(RigidObject):

    def __init__(self, cfg: RigidObjectCfg):
        super().__init__(cfg)

    def _initialize_impl(self):
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(
                f"Failed to find prim for expression: '{self.cfg.prim_path}'."
            )
        template_prim_path = template_prim.GetPath().pathString

        # find rigid root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a rigid body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'USD RigidBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single rigid body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one rigid body in the prim path tree."
            )

        # resolve root prim back into regex expression
        root_prim_path = root_prims[0].GetPath().pathString
        root_prim_path_expr = (
            self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        )
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_rigid_body_view(
            root_prim_path_expr.replace(".*", "*")
        )

        # check if the rigid body was created
        if self._root_physx_view._backend is None:
            raise RuntimeError(
                f"Failed to create rigid body at: {self.cfg.prim_path}. Please check PhysX logs."
            )

        # log information about the rigid body
        omni.log.info(
            f"Rigid body initialized at: {self.cfg.prim_path} with root '{root_prim_path_expr}'."
        )
        omni.log.info(f"Number of instances: {self.num_instances}")
        omni.log.info(f"Number of bodies: {self.num_bodies}")
        omni.log.info(f"Body names: {self.body_names}")

        # container for data access
        self._data = RigidObjectData(self.root_physx_view, self.device)

        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the rigid body data
        self.update(0.0)
