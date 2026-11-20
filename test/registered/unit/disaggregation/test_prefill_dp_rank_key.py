"""The prefill dp rank a KV sender reports and checks rooms against.

The bootstrap server keys prefill ranks by the replica under plain `--dp-size`
and by the attention-DP rank otherwise (`_handle_route`); the sender must use
the same numbering, or every room for replica 1 fails its parity check.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.test.test_utils import CustomTestCase


def _manager(*, system_dp_size, system_dp_rank, attn_dp_size, attn_dp_rank):
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.system_dp_size = system_dp_size
    mgr.system_dp_rank = system_dp_rank
    mgr.attn_dp_size = attn_dp_size
    mgr.attn_dp_rank = attn_dp_rank
    return mgr


class TestPrefillDpRankKey(CustomTestCase):
    def test_plain_dp_uses_the_replica_rank(self):
        mgr = _manager(
            system_dp_size=2, system_dp_rank=1, attn_dp_size=1, attn_dp_rank=0
        )
        self.assertEqual(mgr.prefill_dp_rank, 1)

    def test_attention_dp_uses_the_attention_rank(self):
        mgr = _manager(
            system_dp_size=1, system_dp_rank=0, attn_dp_size=4, attn_dp_rank=3
        )
        self.assertEqual(mgr.prefill_dp_rank, 3)

    @patch("sglang.srt.disaggregation.common.conn.requests.post")
    def test_registration_sends_the_replica_rank(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        sender = CommonKVSender.__new__(CommonKVSender)
        sender.kv_mgr = _manager(
            system_dp_size=2, system_dp_rank=1, attn_dp_size=1, attn_dp_rank=0
        )
        sender.bootstrap_room = 7
        sender.bootstrap_server_url = "127.0.0.1:1"
        sender._register_prefill_dp_rank()
        self.assertEqual(
            mock_post.call_args.kwargs["json"], {"bootstrap_room": 7, "dp_rank": 1}
        )


if __name__ == "__main__":
    unittest.main()
