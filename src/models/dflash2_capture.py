"""Observe native vLLM 0.30 DFlash2 lattices without changing path selection."""

from pathlib import Path
from types import MethodType
import numpy as np


class CaptureExtension:
    """RPC methods mixed into the actual vLLM GPU worker."""

    def install_lattice_capture(self):
        proposer = self.model_runner.speculator
        assert type(proposer).__name__ == 'DFlash2Speculator'
        original = proposer._sample_path
        proposer.prefix_capture = None

        def observe(this, candidate_ids, scores, num_reqs):
            original(candidate_ids, scores, num_reqs)
            if this.prefix_capture is None:
                return
            assert num_reqs == 1, 'The study requires synchronous single-request decoding.'
            length = this.num_speculative_steps
            this.prefix_capture.append({
                'candidate_ids': candidate_ids[:1].detach().cpu().numpy().copy()[0],
                'pair_scores': scores[:1].float().detach().cpu().numpy().copy()[0],
                'native_tokens': this.draft_tokens[:1, :length].detach().cpu().numpy().copy()[0],
                'sample_positions': this.sample_pos[:length].detach().cpu().numpy().copy(),
                'anchor_id': this.input_buffers.input_ids[this._anchor_indices[:1]].detach().cpu().numpy().copy()[0],
            })

        proposer._sample_path = MethodType(observe, proposer)
        return {'proposer': type(proposer).__name__, 'length': proposer.num_speculative_steps,
                'width': proposer.selector_top_k}

    def begin_lattice_capture(self):
        self.model_runner.speculator.prefix_capture = []

    def end_lattice_capture(self, path):
        proposer = self.model_runner.speculator
        records = proposer.prefix_capture
        proposer.prefix_capture = None
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if records:
            np.savez_compressed(target, **{key: np.stack([r[key] for r in records])
                                          for key in records[0]})
        else:
            np.savez_compressed(target)
        return {'rounds': len(records), 'path': str(target)}
