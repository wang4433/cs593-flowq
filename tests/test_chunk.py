"""Checks for action-chunked Flow BC (flowq/chunk.py)."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from flowq.chunk import ChunkFlow, load, load_chunks, save
from flowq.data import KEYS
from flowq.flow import DTYPE, Flow


class Chunk(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_chunk_one_is_the_stage1_flow(self):
        """Same seed gives the same initial weights and the same loss as Flow."""
        torch.manual_seed(3); single = Flow(5, hidden=16, max_velocity=6.)
        torch.manual_seed(3); chunked = ChunkFlow(5, 1, hidden=16, max_velocity=6.)
        s = torch.randn(32, 5, dtype=DTYPE)
        a = torch.cat([torch.rand(32, 6, dtype=DTYPE)*2-1, torch.randint(2, (32, 1)).to(DTYPE)*2-1], -1)
        torch.manual_seed(7); expected = single.loss(s, a)
        torch.manual_seed(7); actual = chunked.loss(s, a[:, None])[0]
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_sample_is_a_valid_command_chunk(self):
        flow = ChunkFlow(5, 4, hidden=16, max_velocity=6.)
        plan = flow.sample(torch.randn(3, 6, dtype=DTYPE), generator=torch.Generator().manual_seed(1))
        self.assertEqual(plan.shape, (3, 4, 7))
        self.assertTrue((plan[..., :6].abs() < 1).all())
        self.assertTrue(torch.isin(plan[..., 6], torch.tensor([-1., 0., 1.], dtype=DTYPE)).all())

    def test_chunks_stop_at_demonstration_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'demos.hdf5'
            with h5py.File(path, 'w') as f:
                for name, n, value in [('demo_0', 3, .1), ('demo_1', 2, .5)]:
                    for key in KEYS:
                        f.create_dataset(f'data/{name}/obs/{key}', data=np.zeros((n, 1)))
                    actions = np.full((n, 7), value); actions[:, 0] = value+np.arange(n)/100
                    f.create_dataset(f'data/{name}/actions', data=actions)
            states, chunks = load_chunks(dict(dataset=str(path)), ['demo_0', 'demo_1'], 4)
        self.assertEqual(states.shape, (5, len(KEYS)))
        self.assertEqual(chunks.shape, (5, 4, 7))
        # Last step of demo_0: its own command repeated, never demo_1's.
        torch.testing.assert_close(chunks[2, :, 0], torch.full((4,), .12, dtype=DTYPE))
        torch.testing.assert_close(chunks[0, :, 0], torch.tensor([.10, .11, .12, .12], dtype=DTYPE))

    def test_checkpoint_roundtrip_with_ema(self):
        flow = ChunkFlow(5, 2, hidden=16, layers=3, gripper_hidden=8, gripper_layers=1)
        ema = ChunkFlow(5, 2, hidden=16, layers=3, gripper_hidden=8, gripper_layers=1)
        cfg = dict(task='square')
        with tempfile.TemporaryDirectory() as tmp:
            save(flow, cfg, {}, 10, Path(tmp)/'flow.pt', ema)
            raw, ck = load(Path(tmp)/'flow.pt', cfg)
            averaged, _ = load(Path(tmp)/'flow.pt', cfg, ema=True)
            save(flow, cfg, {}, 10, Path(tmp)/'plain.pt')
            with self.assertRaises(ValueError):
                load(Path(tmp)/'plain.pt', cfg, ema=True)
        self.assertEqual(ck['updates'], 10)
        for loaded, source in [(raw, flow), (averaged, ema)]:
            for a, b in zip(loaded.parameters(), source.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))


if __name__ == '__main__':
    unittest.main()
