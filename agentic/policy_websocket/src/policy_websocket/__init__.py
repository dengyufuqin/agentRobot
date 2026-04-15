"""
policy_websocket — WebSocket-based policy client/server for robot learning.

Provides BasePolicy interface, WebsocketClientPolicy, WebsocketPolicyServer,
and ActionChunkBroker. Compatible with openpi, robocasa, and other robot envs.
"""

from policy_websocket.base_policy import BasePolicy
from policy_websocket.websocket_client import WebsocketClientPolicy
from policy_websocket.websocket_server import WebsocketPolicyServer
from policy_websocket.action_chunk_broker import ActionChunkBroker
from policy_websocket.action_checker import ActionSanityChecker
from policy_websocket.eval_preflight import EvalPreflightChecker, PreflightVerdict
from policy_websocket.eval_registry import (
    EvalConfig,
    Readiness,
    lookup as lookup_eval_config,
    list_ready,
    print_matrix,
)

__all__ = [
    "BasePolicy",
    "WebsocketClientPolicy",
    "WebsocketPolicyServer",
    "ActionChunkBroker",
    "ActionSanityChecker",
    "EvalPreflightChecker",
    "PreflightVerdict",
    "EvalConfig",
    "Readiness",
    "lookup_eval_config",
    "list_ready",
    "print_matrix",
]
