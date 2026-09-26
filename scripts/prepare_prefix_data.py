#!/usr/bin/env python3
"""Construct pinned, prompt-only discovery/validation/final-test datasets."""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi


SEED = 20260926
SOURCES = (
    ("gsm8k", "math", "openai/gsm8k", "main", "test", "question"),
    ("mbpp", "code", "google-research-datasets/mbpp", "full", "test", "text"),
    ("alpaca_eval", "general_instructions", "tatsu-lab/alpaca_eval", "alpaca_eval", "eval", "instruction"),
)
PARTITIONS = (("discovery", 50), ("validation", 50), ("final_test", 200))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    filenames = [f"{name}.jsonl" for name, _ in PARTITIONS] + ["manifest.json"]
    if any((args.output / name).exists() for name in filenames):
        parser.error("Output already contains split files or a manifest; choose a new directory.")

    api = HfApi()
    rng = np.random.default_rng(SEED)
    seen_prompts = set()
    records = {name: [] for name, _ in PARTITIONS}
    manifest = {
        "seed": SEED,
        "rng": "numpy.random.default_rng / PCG64",
        "numpy_version": np.__version__,
        "source_order": [source[0] for source in SOURCES],
        "partition_sizes_per_domain": dict(PARTITIONS),
        "row_id_convention": "Zero-based row index in the pinned source configuration and split.",
        "duplicate_policy": (
            "Exact prompt-string equality, across all domains and partitions. "
            "Keep the first occurrence encountered in source-order permutations; "
            "replace skipped duplicates with the next eligible row of that source permutation."
        ),
        "prompt_policy": "Read and export only the declared prompt column, without stripping or rewriting it.",
        "sources": [],
    }

    for source_name, domain, dataset_id, config, source_split, prompt_field in SOURCES:
        revision = api.dataset_info(dataset_id, revision="main").sha
        if not revision:
            raise RuntimeError(f"No resolved revision for {dataset_id}")
        dataset = load_dataset(dataset_id, name=config, split=source_split, revision=revision)
        # Drop answer/reference columns before reading any row values.
        prompts = dataset.select_columns([prompt_field])
        selected = []
        skipped_duplicates = 0
        for index in rng.permutation(len(prompts)):
            row_id = int(index)
            prompt = prompts[row_id][prompt_field]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing or invalid prompt in {dataset_id}, row {row_id}")
            if prompt in seen_prompts:
                skipped_duplicates += 1
                continue
            seen_prompts.add(prompt)
            selected.append((row_id, prompt))
            if len(selected) == 300:
                break
        if len(selected) != 300:
            raise RuntimeError(f"{dataset_id} supplies only {len(selected)} eligible unique prompts")

        source_manifest = {
            "name": source_name,
            "domain": domain,
            "dataset_id": dataset_id,
            "revision": revision,
            "config": config,
            "source_split": source_split,
            "prompt_field": prompt_field,
            "source_row_count": len(prompts),
            "duplicates_skipped_before_selection_complete": skipped_duplicates,
            "selected_row_ids": {},
        }
        offset = 0
        for partition, size in PARTITIONS:
            subset = selected[offset:offset + size]
            source_manifest["selected_row_ids"][partition] = [row_id for row_id, _ in subset]
            for row_id, prompt in subset:
                records[partition].append({
                    "prompt_id": f"{source_name}:{row_id}",
                    "domain": domain,
                    "split": partition,
                    "dataset_id": dataset_id,
                    "dataset_revision": revision,
                    "dataset_config": config,
                    "source_split": source_split,
                    "source_row_id": row_id,
                    "prompt_field": prompt_field,
                    "prompt": prompt,
                })
            offset += size
        manifest["sources"].append(source_manifest)

    # All three cohorts are constructed before writing any deliverable.
    for partition, _ in PARTITIONS:
        with (args.output / f"{partition}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records[partition]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved 150 discovery, 150 validation, and 600 final-test prompts to {args.output}")


if __name__ == "__main__":
    main()
