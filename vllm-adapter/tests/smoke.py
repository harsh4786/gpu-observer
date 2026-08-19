import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpu_observer_semantic import semantic_emitter


class NewRequest:
    req_id = "chatcmpl-semantic-smoke"


class CachedRequests:
    req_ids = []
    num_output_tokens = []


class SchedulerOutput:
    scheduled_new_reqs = [NewRequest()]
    scheduled_cached_reqs = CachedRequests()
    num_scheduled_tokens = {"chatcmpl-semantic-smoke": 44}
    total_num_scheduled_tokens = 44
    finished_req_ids = set()


class KvCacheManager:
    usage = 0.25


class Scheduler:
    kv_cache_manager = KvCacheManager()

    @staticmethod
    def get_request_counts():
        return 1, 2


assert semantic_emitter.enabled
scheduler_output = SchedulerOutput()
step_id = semantic_emitter.begin(scheduler_output, Scheduler())
assert step_id == 1
assert scheduler_output._gpu_observer_step_id == step_id
semantic_emitter.packed_layout(
    step_id,
    ["chatcmpl-semantic-smoke"],
    [44],
)
semantic_emitter.end(step_id, 0)
print("semantic smoke emitted step=1 packed_rows=[0,44)")
