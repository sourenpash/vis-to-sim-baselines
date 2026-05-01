# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0


class WbcSolveFailedException(Exception):
    pass


class CollisionError(Exception):
    pass


class WbcStepOverMaxStepException(Exception):
    pass
