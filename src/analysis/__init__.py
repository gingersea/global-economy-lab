"""
Analysis sub-package.
"""

from src.analysis.cycle_position import (
    get_cycle_position,
    get_current_phase,
    PHASE_LABELS,
)
from src.analysis.correlation import (
    build_return_matrix,
    compute_correlation_matrix,
    compute_rolling_correlation,
)
from src.analysis.event_study import (
    analyze_event,
    analyze_preset_event,
    list_preset_events,
    PRESET_EVENTS,
)
from src.analysis.dashboard_data import build_dashboard_bundle

__all__ = [
    "get_cycle_position",
    "get_current_phase",
    "PHASE_LABELS",
    "build_return_matrix",
    "compute_correlation_matrix",
    "compute_rolling_correlation",
    "analyze_event",
    "analyze_preset_event",
    "list_preset_events",
    "PRESET_EVENTS",
    "build_dashboard_bundle",
]
