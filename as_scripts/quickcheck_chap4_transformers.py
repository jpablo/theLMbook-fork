"""
Quick shape/grad check for experiments/chap4-transformers.py

Usage examples:
  python as_scripts/quickcheck_chap4_transformers.py
  python as_scripts/quickcheck_chap4_transformers.py --vocab-size 100 --emb-dim 16 --num-heads 4 --num-blocks 3 --batch 4 --seq 12
  python as_scripts/quickcheck_chap4_transformers.py --no-backward  # forward only
"""

from __future__ import annotations

import argparse

from torch import Tensor

from experiments.chap4_transformers import DecoderLanguageModel

import torch
import torch.nn.functional as F


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=50)
    p.add_argument("--emb-dim", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=1)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--pad-idx", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    # maximum sequence length
    p.add_argument("--seq", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-backward", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    model = DecoderLanguageModel(
        vocab_size=args.vocab_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        pad_idx=args.pad_idx,
    )

    # a 1 x 4 tensor correspond to a single text with 4 tokens ("the small red house")
    tokens: Tensor = torch.randint(0, args.vocab_size, (args.batch, args.seq))
    logits = model(tokens)
    print("forward logits shape:", tuple(logits.shape))

    if not args.no_backward:
        target = torch.randint(0, args.vocab_size, (args.batch * args.seq,))
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), target)
        loss.backward()
        print("backward ok, loss:", float(loss))


if __name__ == "__main__":
    main()

