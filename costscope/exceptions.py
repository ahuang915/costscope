class CostScopeError(Exception):
    pass


class EstimationCancelled(CostScopeError):
    """Raised when the user declines to proceed after seeing the cost estimate."""
