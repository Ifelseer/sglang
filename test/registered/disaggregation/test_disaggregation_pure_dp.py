"""PD with a plain `--dp-size 2` prefill: two replicas, no attention DP.

The prefill dispatches by bootstrap room (follow_bootstrap_room, its default)
and the KV sender checks the room against its own rank. Under plain DP that
rank is the replica index; `attn_dp_rank` is 0 on both replicas, so keying on
it rejected every room that landed on replica 1.
"""

import re
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_cuda_ci(est_time=240, stage="extra-b", runner_config="4-gpu-b200")

PREFILL_DP_SIZE = 2


class TestDisaggregationPureDP(PDDisaggregationServerBase):
    decode_base_gpu_id = PREFILL_DP_SIZE

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.launch_all()

    @classmethod
    def start_prefill(cls):
        # No --nccl-port: PortArgs pins one port per server and both replicas
        # would bind it.
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--dp-size",
            str(PREFILL_DP_SIZE),
            "--enable-metrics",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices_for(
            range(PREFILL_DP_SIZE)
        )
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    def test_rooms_of_both_parities_complete(self):
        """Rooms are random, so about half land on replica 1; all must finish."""

        def one(i: int):
            resp = requests.post(
                f"{self.lb_url}/generate",
                json={
                    "text": f"Request {i}: count from one to ten.",
                    "sampling_params": {
                        "max_new_tokens": 16,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                },
                timeout=120,
            )
            return resp.status_code, resp.text[:300]

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(one, range(32)))
        failed = [r for r in results if r[0] != 200]
        self.assertEqual(failed, [], f"{len(failed)}/32 failed: {failed[:3]}")

        served = self._prefill_transfers_by_rank()
        self.assertEqual(set(served), set(range(PREFILL_DP_SIZE)), served)

    def _prefill_transfers_by_rank(self) -> dict[int, float]:
        """Finished KV sends per prefill replica, from the scheduler metrics."""
        text = requests.get(f"{self.prefill_url}/metrics", timeout=10).text
        served: dict[int, float] = {}
        for line in text.splitlines():
            if not line.startswith("sglang:kv_transfer_latency_ms_count{"):
                continue
            match = re.search(r'dp_rank="(\d+)"', line)
            value = float(line.rsplit(" ", 1)[1])
            if match and value > 0:
                rank = int(match.group(1))
                served[rank] = served.get(rank, 0.0) + value
        return served


if __name__ == "__main__":
    unittest.main()
