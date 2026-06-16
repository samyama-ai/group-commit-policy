"""group_commit: when is adaptive group-commit tuning worth it? A closed-loop, self-clocking view.

Pre-registration: dbms_cloud/daily/06-optimal-group-commit-policy/HYPOTHESIS.md (frozen 2026-06-16).
"""
from .theory import (
    sqrt_rule_T, openloop_overhead, openloop_overhead_opt,
    closed_loop_fixed_point, offline_opt, offline_opt_bruteforce,
    online_cost_abstraction,
)
from .device import FlushModel
from .sim import simulate

__all__ = [
    "sqrt_rule_T", "openloop_overhead", "openloop_overhead_opt",
    "closed_loop_fixed_point", "offline_opt", "offline_opt_bruteforce", "online_cost_abstraction",
    "FlushModel", "simulate",
]
