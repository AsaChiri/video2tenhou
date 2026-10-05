# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Tiny human-input fixtures exercise the refinement recipe without model training."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from video2tenhou.train import classifier_context as cc


def _png(path: Path, pixels: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), pixels)
    return path


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Create reviewed training inputs with isolated frame acquisition."""
    pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    video = tmp_path / "recording.mp4"
    video.write_bytes(b"synthetic recording identity")
    work = tmp_path / "work"
    anchors = tmp_path / "train"
    anchor = _png(anchors / "1m/hand_TL_40_00.png", cc.to_crop(pixels))
    _png(anchors / "none/pond_BR_99_00.png", cc.to_crop(pixels))
    labels = [
        {
            "kind": "hand",
            "corner": "TL",
            "t": 39.6,
            "boxes": [
                {
                    "tile": "1m",
                    "quad": [[10, 12], [30, 12], [30, 32], [10, 32]],
                    "sideways": True,
                },
                {"tile": "X", "quad": [[0, 0], [20, 0], [20, 20], [0, 20]]},
                {"tile": "none", "quad": [[0, 0], [20, 0], [20, 20], [0, 20]]},
            ],
        },
        {"kind": "pond", "corner": "BR", "t": 99.0, "boxes": []},
        {"kind": "hand", "corner": "TR", "t": 40.2, "boxes": []},
    ]
    hands = [{"t_start": i * 10, "t_end": i * 10 + 9.8} for i in range(5)]
    monkeypatch.setattr(cc, "load_labels", lambda _: labels)
    monkeypatch.setattr(cc, "hand_table", lambda *_: hands)
    monkeypatch.setattr(
        cc.Calibration, "load", lambda *_: SimpleNamespace(data={"fixture": True})
    )
    monkeypatch.setattr(cc, "fit_path", lambda _: tmp_path / "no-fit.json")
    monkeypatch.setattr(cc, "region_upright", lambda frame, *_: (frame, np.eye(3)))
    _png(work / video.stem / "frames/39.600.png", pixels)
    manifest = tmp_path / "human.jsonl"
    human = {
        "source": "C:\\reviewed\\hand_TL_40.png",
        "split": "train",
        "origin": "human",
        "reviewed": True,
    }
    manifest.write_text(json.dumps(human) + "\n")
    return SimpleNamespace(
        video=video,
        work=work,
        anchors=anchors,
        anchor=anchor,
        labels=labels,
        manifest=manifest,
        human=human,
        output=tmp_path / "context",
        pixels=pixels,
    )


def _build(corpus: SimpleNamespace) -> dict:
    return cc.build_context_dataset(
        corpus.anchors,
        corpus.manifest,
        corpus.output,
        video=corpus.video,
        work=corpus.work,
    )


def test_context_build_preserves_anchors_and_original_timestamp_split(
    corpus: SimpleNamespace,
) -> None:
    original = corpus.anchor.read_bytes()
    report = _build(corpus)
    rows = [
        json.loads(line)
        for line in (corpus.output / "manifest.jsonl").read_text().splitlines()
    ]
    assert report["complete"]
    assert report["counts"] == {
        "anchors": 2,
        "contexts": 1,
        "context_hand": 1,
    }
    assert corpus.anchor.read_bytes() == original
    assert [row["kind"] for row in rows] == ["anchor", "anchor", "context"]
    assert rows[0]["t"] == 39.6
    assert rows[0]["group"] == "recording:hand:3"
    assert rows[1]["tile"] == "none"
    assert rows[1]["group"] == "recording:hand:None"
    # Rounding 39.6 to 40 would leak this TRAIN row into held-out hand 4.
    assert report["annotations"]["hand_TL_40"] == {"t": 39.6, "hand": 3}
    assert rows[2]["box"] == [7.0, 7.0, 27.0, 27.0]
    assert rows[2]["sideways"]
    saved = cv2.imread(rows[2]["image"])
    assert saved is not None
    assert np.array_equal(saved, corpus.pixels[5:39, 3:37])
    assert report["video_sha256"] == cc.file_hash(corpus.video)
    with pytest.raises(FileExistsError):
        _build(corpus)


@pytest.mark.parametrize(
    "bad", ["heldout_anchor", "heldout_context", "pseudo", "unreviewed"]
)
def test_context_build_rejects_split_or_human_provenance_violation(
    corpus: SimpleNamespace, bad: str
) -> None:
    if bad == "heldout_anchor":
        corpus.labels[0]["t"] = 40.2  # Still the same rounded filename.
        corpus.manifest.write_text("")
    elif bad == "heldout_context":
        corpus.human["source"] = "hand_TR_40.png"
    elif bad == "pseudo":
        corpus.human["origin"] = "teacher"
    else:
        corpus.human["reviewed"] = False
    if bad != "heldout_anchor":
        corpus.manifest.write_text(json.dumps(corpus.human) + "\n")
    with pytest.raises(ValueError, match=r"held-out|human TRAIN"):
        _build(corpus)
    assert not corpus.output.exists()


def test_context_augmentation_uses_recorded_draw_order_and_human_rotation() -> None:
    image = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)
    calls, values = [], iter([0.1, 0.2, 0.05, -0.05])

    class RecordedDraws:
        """Supply the recorded recipe draws, failing on extra random values."""

        def uniform(self, a: float, b: float) -> float:
            """Record the requested interval and consume one fixture value."""
            calls.append((a, b))
            return next(values)

        def choice[Item](self, seq: Sequence[Item]) -> Item:
            """Use the counterclockwise rotation from the recorded recipe."""
            return seq[1]

    rng = RecordedDraws()
    row = {"kind": "context", "box": [10, 10, 30, 30], "sideways": True}
    result = cc.context_crop(image, row, rng)
    assert np.array_equal(
        result, cv2.rotate(image[5:33, 9:33], cv2.ROTATE_90_COUNTERCLOCKWISE)
    )
    assert calls == [(0, 0.2), (0, 0.2), (-0.06, 0.06), (-0.06, 0.06)]
    assert cc.context_crop(image, {"kind": "anchor"}, rng) is image


def test_anchor_teacher_mask_and_context_gradients() -> None:
    images = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    logits = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0]], requires_grad=True
    )
    labels, mask = torch.tensor([0, 1, 2]), torch.tensor([True, False, True])
    seen = []

    def teacher(x: torch.Tensor) -> torch.Tensor:
        seen.append((x.clone(), torch.is_grad_enabled()))
        return x

    ce = nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
    loss = cc.refinement_loss(logits, labels, mask, images, teacher)
    expected = ce + 16 * nn.functional.kl_div(
        nn.functional.log_softmax(logits[mask] / 2, dim=1),
        nn.functional.softmax(images[mask] / 2, dim=1),
        reduction="batchmean",
    )
    assert torch.allclose(loss, expected)
    assert len(seen) == 1
    assert not seen[0][1]
    assert torch.equal(seen[0][0], images[mask])
    ce_gradient = torch.autograd.grad(ce, logits, retain_graph=True)[0]
    loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad[1], ce_gradient[1])
    assert not torch.allclose(logits.grad[0], ce_gradient[0])
    assert torch.allclose(
        cc.refinement_loss(
            logits, labels, ~torch.ones(3, dtype=torch.bool), images, teacher
        ),
        ce,
    )
    assert len(seen) == 1


def test_frozen_batchnorm_retains_affine_learning() -> None:
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(0.1))
    bn = model[0]
    assert isinstance(bn, nn.BatchNorm1d)
    assert bn.running_mean is not None
    assert bn.running_var is not None
    assert bn.num_batches_tracked is not None
    assert bn.weight is not None
    mean, var, count = (
        bn.running_mean.clone(),
        bn.running_var.clone(),
        bn.num_batches_tracked.clone(),
    )
    cc.freeze_batchnorm_statistics(model)
    model(torch.ones(4, 3)).sum().backward()
    assert model.training
    assert model[1].training
    assert not bn.training
    assert torch.equal(bn.running_mean, mean)
    assert torch.equal(bn.running_var, var)
    assert torch.equal(bn.num_batches_tracked, count)
    assert bn.weight.grad is not None
    assert bn.weight.grad.abs().sum() > 0


def test_training_checkpoint_contract_with_tiny_cpu_model(
    corpus: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _build(corpus)
    validation = tmp_path / "val"
    _png(validation / "1m/hand_TR_40_00.png", cc.to_crop(corpus.pixels))
    base = tmp_path / "base"
    base.mkdir()

    def tiny_model(n: int) -> nn.Module:
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, n))

    monkeypatch.setattr(cc, "make_model", tiny_model)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    torch.save(tiny_model(len(cc.CLASSES)).state_dict(), base / "weights.pt")
    (base / "meta.json").write_text(
        json.dumps({"classes": cc.CLASSES, "temperature": 0.7})
    )
    original = {p.name: p.read_bytes() for p in base.iterdir()}
    output = tmp_path / "refined"
    report = cc.train_refinement(corpus.output, validation, base, output, device="cpu")
    assert report["complete"]
    assert len(report["epochs"]) == 3
    assert {p.name: p.read_bytes() for p in base.iterdir()} == original
    for epoch in (1, 2, 3):
        checkpoint = output / f"epoch_{epoch}"
        meta = json.loads((checkpoint / "meta.json").read_text())
        assert meta["temperature"] == 0.7
        assert meta["classes"] == cc.CLASSES
        assert meta["refinement"]["epoch"] == epoch
        proof = json.loads((checkpoint / "provenance.json").read_text())
        assert proof["checkpoint_complete"]
        assert proof["checkpoint_epoch"] == epoch
        assert proof["epochs"][-1]["weights_sha256"] == cc.file_hash(
            checkpoint / "weights.pt"
        )
    with pytest.raises(FileExistsError):
        cc.train_refinement(corpus.output, validation, base, output, device="cpu")
    corpus.anchor.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="Training image changed"):
        cc.train_refinement(
            corpus.output, validation, base, tmp_path / "rejected", device="cpu"
        )
