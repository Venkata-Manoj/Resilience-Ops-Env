# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Resilience Ops Env Environment Implementation.

Thin wrapper — all logic lives in models.py per project convention.
"""

try:
    from ..models import ResilienceOpsEnvironment
except ImportError:
    from models import ResilienceOpsEnvironment

__all__ = ["ResilienceOpsEnvironment"]
