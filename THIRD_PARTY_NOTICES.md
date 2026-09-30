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

FFmpeg is an external installation. Python dependency versions are
recorded in `uv.lock`; their own licenses remain applicable. Distribution of
broadcast-derived calibration templates and regression images requires rights
to the underlying material. Record those permissions alongside asset provenance.

## Local web runtime

The local server uses Starlette and Uvicorn (BSD-3-Clause), AnyIO (MIT),
and h11 (MIT). Their installed distributions retain the upstream license files;
exact dependency versions are recorded in `uv.lock`. Starlette and Uvicorn
are copyright Encode OSS Ltd; AnyIO is copyright Alex Grönholm; h11 is
copyright Nathaniel J. Smith and other contributors.

Outbound score queries and HTTP integration tests use HTTPX (BSD-3-Clause),
copyright Encode OSS Ltd. Its installed distribution retains the upstream license.

## Frontend runtime

The bundled Vue runtime and its @vue packages use the following MIT license.
Frontend dependencies and exact versions are recorded in `frontend/package-lock.json`.

The MIT License (MIT)

Copyright (c) 2018-present, Yuxi (Evan) You

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
