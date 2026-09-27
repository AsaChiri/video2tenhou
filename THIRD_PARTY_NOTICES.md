# Third-party notices

## Tile artwork

The UI uses [FluffyStuff's riichi-mahjong-tiles](https://github.com/FluffyStuff/riichi-mahjong-tiles).
Its CC0 dedication and source details are retained with the vendored SVG assets.

## Models and inference

- [LibreYOLO 1.5.0](https://raw.githubusercontent.com/LibreYOLO/libreyolo/v1.5.0/LICENSE)
  publishes its core software under MIT. Its pretrained model families carry
  separate terms; the core license does not cover every checkpoint.
- The specific [LibreYOLO9-S base](https://huggingface.co/LibreYOLO/LibreYOLO9s)
  identifies MIT weights from MultimediaTechLab/YOLO's `v1.0-alpha` release,
  `v9-s.pt`, copyright 2024 Kin-Yiu Wong and Hao-Tang Tsui. The conversion changes
  state-dictionary keys, not learned parameters. Keep the base notice and exact
  checkpoint identity with derivative detector weights.
- The classifier uses PyTorch/torchvision ResNet18 and ImageNet initialization
  during training. Inference loads project-trained local weights. Models are
  not included in the Python distribution.

Each separately distributed model bundle retains its training recipe, input and
base-checkpoint provenance, selected weights digest and applicable notices.

FFmpeg and Tesseract are external installations. Python dependency versions are
recorded in `uv.lock`; their own licenses remain applicable. Distribution of
broadcast-derived calibration templates and regression images requires rights
to the underlying material. Record those permissions alongside asset provenance.
