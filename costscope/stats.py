import math
from dataclasses import dataclass


_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
}
_T_TABLE_99 = {
    1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032,
    6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
    11: 3.106, 12: 3.055, 13: 3.012, 14: 2.977, 15: 2.947,
    16: 2.921, 17: 2.898, 18: 2.878, 19: 2.861, 20: 2.845,
    21: 2.831, 22: 2.819, 23: 2.807, 24: 2.797, 25: 2.787,
    26: 2.779, 27: 2.771, 28: 2.763, 29: 2.756,
}
_T_TABLE_90 = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
    6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
    11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
    16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725,
    21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708,
    26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699,
}
_Z_BY_CONFIDENCE = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def t_critical(df: int, confidence: float = 0.95) -> float:
    table = {0.90: _T_TABLE_90, 0.95: _T_TABLE_95, 0.99: _T_TABLE_99}.get(confidence)
    if table is None:
        return _Z_BY_CONFIDENCE.get(confidence, 1.960)
    if df < 1:
        return table[1]
    return table.get(df, _Z_BY_CONFIDENCE[confidence])


@dataclass
class CostEstimate:
    mean_per_call: float
    stdev_per_call: float
    sample_size: int
    total_calls: int
    confidence: float = 0.95

    @property
    def total_estimate(self) -> float:
        return self.mean_per_call * self.total_calls

    @property
    def margin(self) -> float:
        if self.sample_size < 2:
            return float("inf")
        df = self.sample_size - 1
        t = t_critical(df, self.confidence)
        se_per_call = self.stdev_per_call / math.sqrt(self.sample_size)
        if self.total_calls > self.sample_size:
            se_per_call *= math.sqrt(1 - self.sample_size / self.total_calls)
        return t * se_per_call * self.total_calls

    @property
    def lower(self) -> float:
        return max(0.0, self.total_estimate - self.margin)

    @property
    def upper(self) -> float:
        return self.total_estimate + self.margin

    @property
    def relative_margin(self) -> float:
        if self.total_estimate <= 0:
            return float("inf")
        return self.margin / self.total_estimate


def compute_estimate(
    costs: list[float],
    total_calls: int,
    confidence: float = 0.95,
) -> CostEstimate:
    n = len(costs)
    if n == 0:
        raise ValueError("Need at least one sample to estimate")
    mean = sum(costs) / n
    if n < 2:
        stdev = 0.0
    else:
        var = sum((c - mean) ** 2 for c in costs) / (n - 1)
        stdev = math.sqrt(var)
    return CostEstimate(
        mean_per_call=mean,
        stdev_per_call=stdev,
        sample_size=n,
        total_calls=total_calls,
        confidence=confidence,
    )
