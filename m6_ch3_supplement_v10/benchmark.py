"""Fixed forward-only measurement contract; CUDA operations injected for tests."""
import statistics
import time


def measure(forward, synchronize, reset_peak, memory, batch, clock=time.perf_counter, checkpoint=lambda: None):
    if batch not in (1, 16):
        raise ValueError('fixed measurement batch')
    for _ in range(50):
        forward()
    synchronize()
    resident = memory()
    reset_peak()
    groups = []
    for group in range(5):
        checkpoint()
        synchronize()
        start = clock()
        for _ in range(100):
            forward()
        synchronize()
        elapsed = clock()-start
        if elapsed <= 0:
            raise ValueError('invalid clock interval')
        groups.append(dict(group=group, calls=100, wall_seconds=elapsed,
                           batch_ms=elapsed*10, sample_ms=elapsed*10/batch,
                           samples_per_second=batch*100/elapsed))
    values = [g['batch_ms'] for g in groups]
    peak = memory()
    return dict(warmup_calls=50, timed_calls=500, groups=groups, batch=batch,
                median_batch_ms=statistics.median(values), min_batch_ms=min(values),
                max_batch_ms=max(values), stdev_batch_ms=statistics.pstdev(values),
                median_sample_ms=statistics.median(values)/batch,
                median_samples_per_second=batch*1000/statistics.median(values),
                resident=resident, peak=peak,
                incremental_peak_allocated=peak.get('max_allocated', 0)-resident.get('allocated', 0),
                backward=0, Adam=0)
