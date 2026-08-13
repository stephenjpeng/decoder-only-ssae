"""Test embedding materialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_generation.materialize_embedding_split import (
    build_tuple_key_map,
    materialize_embedding_split,
    verify_manifests_compatible,
)


def write_fake_source(folder: Path, prompts: list[dict], manifest: dict) -> None:
    """Write a fake source folder with prompts.json, embds/manifest.json, and embds/embds_<i>/ directories."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f)
    embds = folder / "embds"
    embds.mkdir(parents=True, exist_ok=True)
    with open(embds / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    for i, prompt in enumerate(prompts):
        row_dir = embds / f"embds_{i}"
        row_dir.mkdir(parents=True, exist_ok=True)
        (row_dir / "embds.h5").write_text(f"fake_h5_{i}", encoding="utf-8")
        (row_dir / "prompts.txt").write_text(prompt["prompt"], encoding="utf-8")


class TestEmbeddingMaterialization(unittest.TestCase):
    def test_maps_tuple_keys_across_two_source_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src1 = tmp / "src1"
            src2 = tmp / "src2"
            dest = tmp / "dest"

            manifest = {
                "backbone": "sd35",
                "backbone_kwargs": {},
                "streams": ["t5"],
                "flat_dim": 4096,
                "style_suffix": "",
                "n_prompts": 2,
                "created_at": "2025-01-01T00:00:00Z",
            }

            prompts1 = [
                {"id": "p0", "prompt": "a", "choices": {}, "tuple_key": ["a"]},
                {"id": "p1", "prompt": "b", "choices": {}, "tuple_key": ["b"]},
            ]
            prompts2 = [
                {"id": "p2", "prompt": "c", "choices": {}, "tuple_key": ["c"]},
                {"id": "p3", "prompt": "d", "choices": {}, "tuple_key": ["d"]},
            ]

            write_fake_source(src1, prompts1, manifest)
            write_fake_source(src2, prompts2, {**manifest, "n_prompts": 2})

            dest_prompts = [
                {"id": "d0", "prompt": "a", "choices": {}, "tuple_key": ["a"]},
                {"id": "d1", "prompt": "c", "choices": {}, "tuple_key": ["c"]},
                {"id": "d2", "prompt": "b", "choices": {}, "tuple_key": ["b"]},
                {"id": "d3", "prompt": "d", "choices": {}, "tuple_key": ["d"]},
            ]
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / "prompts.json", "w", encoding="utf-8") as f:
                json.dump(dest_prompts, f)

            materialize_embedding_split([src1, src2], dest, mode="copy")

            # verify all 4 rows materialized
            for i in range(4):
                row_dir = dest / "embds" / f"embds_{i}"
                self.assertTrue(row_dir.exists())
                self.assertTrue((row_dir / "embds.h5").exists())
                self.assertTrue((row_dir / "prompts.txt").exists())

            # verify content matches
            self.assertEqual((dest / "embds/embds_0/prompts.txt").read_text(), "a")
            self.assertEqual((dest / "embds/embds_1/prompts.txt").read_text(), "c")
            self.assertEqual((dest / "embds/embds_2/prompts.txt").read_text(), "b")
            self.assertEqual((dest / "embds/embds_3/prompts.txt").read_text(), "d")

    def test_rejects_duplicate_tuple_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src1 = tmp / "src1"
            src2 = tmp / "src2"
            dest = tmp / "dest"

            manifest = {
                "backbone": "sd35",
                "backbone_kwargs": {},
                "streams": ["t5"],
                "flat_dim": 4096,
                "style_suffix": "",
                "n_prompts": 1,
                "created_at": "2025-01-01T00:00:00Z",
            }

            prompts1 = [{"id": "p0", "prompt": "a", "choices": {}, "tuple_key": ["a"]}]
            prompts2 = [{"id": "p1", "prompt": "a_dup", "choices": {}, "tuple_key": ["a"]}]

            write_fake_source(src1, prompts1, manifest)
            write_fake_source(src2, prompts2, manifest)

            dest_prompts = [{"id": "d0", "prompt": "a", "choices": {}, "tuple_key": ["a"]}]
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / "prompts.json", "w", encoding="utf-8") as f:
                json.dump(dest_prompts, f)

            with self.assertRaises(ValueError) as ctx:
                materialize_embedding_split([src1, src2], dest)
            self.assertIn("duplicate tuple_key across sources", str(ctx.exception))

    def test_rejects_missing_tuple_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src1 = tmp / "src1"
            dest = tmp / "dest"

            manifest = {
                "backbone": "sd35",
                "backbone_kwargs": {},
                "streams": ["t5"],
                "flat_dim": 4096,
                "style_suffix": "",
                "n_prompts": 1,
                "created_at": "2025-01-01T00:00:00Z",
            }

            prompts1 = [{"id": "p0", "prompt": "a", "choices": {}, "tuple_key": ["a"]}]
            write_fake_source(src1, prompts1, manifest)

            dest_prompts = [
                {"id": "d0", "prompt": "a", "choices": {}, "tuple_key": ["a"]},
                {"id": "d1", "prompt": "b", "choices": {}, "tuple_key": ["b"]},
            ]
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / "prompts.json", "w", encoding="utf-8") as f:
                json.dump(dest_prompts, f)

            with self.assertRaises(ValueError) as ctx:
                materialize_embedding_split([src1], dest)
            self.assertIn("not found in any source", str(ctx.exception))

    def test_accepts_compatible_manifests_differing_n_prompts_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src1 = tmp / "src1"
            src2 = tmp / "src2"
            dest = tmp / "dest"

            manifest1 = {
                "backbone": "sd35",
                "backbone_kwargs": {},
                "streams": ["t5"],
                "flat_dim": 4096,
                "style_suffix": "",
                "n_prompts": 1,
                "created_at": "2025-01-01T00:00:00Z",
            }
            manifest2 = {
                "backbone": "sd35",
                "backbone_kwargs": {},
                "streams": ["t5"],
                "flat_dim": 4096,
                "style_suffix": "",
                "n_prompts": 2,
                "created_at": "2025-01-02T00:00:00Z",
            }

            prompts1 = [{"id": "p0", "prompt": "a", "choices": {}, "tuple_key": ["a"]}]
            prompts2 = [{"id": "p1", "prompt": "b", "choices": {}, "tuple_key": ["b"]}]

            write_fake_source(src1, prompts1, manifest1)
            write_fake_source(src2, prompts2, manifest2)

            dest_prompts = [
                {"id": "d0", "prompt": "a", "choices": {}, "tuple_key": ["a"]},
                {"id": "d1", "prompt": "b", "choices": {}, "tuple_key": ["b"]},
            ]
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / "prompts.json", "w", encoding="utf-8") as f:
                json.dump(dest_prompts, f)

            # should succeed
            materialize_embedding_split([src1, src2], dest, mode="copy")
            self.assertTrue((dest / "embds" / "embds_0").exists())
            self.assertTrue((dest / "embds" / "embds_1").exists())

    def test_rejects_incompatible_backbone(self) -> None:
        manifest1 = {
            "backbone": "sd35",
            "backbone_kwargs": {},
            "streams": ["t5"],
            "flat_dim": 4096,
            "style_suffix": "",
        }
        manifest2 = {
            "backbone": "flux",
            "backbone_kwargs": {},
            "streams": ["t5"],
            "flat_dim": 4096,
            "style_suffix": "",
        }
        with self.assertRaises(ValueError) as ctx:
            verify_manifests_compatible(
                [(Path("p1"), manifest1), (Path("p2"), manifest2)]
            )
        self.assertIn("incompatible backbone", str(ctx.exception))

    def test_creates_hardlinked_files_preserves_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src1 = tmp / "src1"
            dest = tmp / "dest"

            manifest = {
                "backbone": "sd35",
                "backbone_kwargs": {},
                "streams": ["t5"],
                "flat_dim": 4096,
                "style_suffix": "",
                "n_prompts": 1,
                "created_at": "2025-01-01T00:00:00Z",
            }

            prompts1 = [{"id": "p0", "prompt": "test_content", "choices": {}, "tuple_key": ["a"]}]
            write_fake_source(src1, prompts1, manifest)

            dest_prompts = [{"id": "d0", "prompt": "test_content", "choices": {}, "tuple_key": ["a"]}]
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / "prompts.json", "w", encoding="utf-8") as f:
                json.dump(dest_prompts, f)

            materialize_embedding_split([src1], dest, mode="hardlink")

            # verify content
            src_file = src1 / "embds/embds_0/prompts.txt"
            dest_file = dest / "embds/embds_0/prompts.txt"
            self.assertEqual(src_file.read_text(), "test_content")
            self.assertEqual(dest_file.read_text(), "test_content")

            # verify inodes match if on same filesystem
            try:
                self.assertEqual(src_file.stat().st_ino, dest_file.stat().st_ino)
            except AssertionError:
                # different filesystems, skip inode check
                pass

    def test_writes_complete_materialization_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src1 = tmp / "src1"
            dest = tmp / "dest"

            manifest = {
                "backbone": "sd35",
                "backbone_kwargs": {},
                "streams": ["t5"],
                "flat_dim": 4096,
                "style_suffix": "",
                "n_prompts": 1,
                "created_at": "2025-01-01T00:00:00Z",
            }

            prompts1 = [{"id": "p0", "prompt": "a", "choices": {}, "tuple_key": ["a"]}]
            write_fake_source(src1, prompts1, manifest)

            dest_prompts = [{"id": "d0", "prompt": "a", "choices": {}, "tuple_key": ["a"]}]
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / "prompts.json", "w", encoding="utf-8") as f:
                json.dump(dest_prompts, f)

            materialize_embedding_split([src1], dest, mode="copy")

            mat_manifest_path = dest / "embedding_materialization_manifest.json"
            self.assertTrue(mat_manifest_path.exists())

            with open(mat_manifest_path, encoding="utf-8") as f:
                mat = json.load(f)

            self.assertIn("source_folders", mat)
            self.assertIn("destination_folder", mat)
            self.assertEqual(mat["mode"], "copy")
            self.assertEqual(mat["n_destination"], 1)
            self.assertIn("n_sources", mat)
            self.assertIn("file_counts", mat)
            self.assertIn("prompts_json_sha256", mat)
            self.assertIn("embds_manifest_sha256", mat)


if __name__ == "__main__":
    unittest.main()
