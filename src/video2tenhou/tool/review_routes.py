# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Starlette endpoints for recording-scoped evidence and corrections."""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING

import cv2
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import FileResponse, Response
from starlette.routing import Mount, Route

from .http import REVIEW_BODY_LIMIT, json_response, read_json_body
from .workflow import BUSY

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.requests import Request

    from .review_state import ReviewState
    from .workflow import Workspace

ANALYSIS_JOBS = ("prepare", "analyze")


class ReviewRoutes:
    """Bind individual review endpoints to the application-owned workspace."""

    def __init__(self, workspace: Workspace) -> None:
        """Keep all project lookup and job exclusion scoped to this workspace."""
        self.workspace = workspace

    def state(self, request: Request) -> ReviewState:
        """Resolve the project key captured by Starlette's parent mount."""
        return self.workspace.review_state(request.path_params["key"])

    def check_write(self, *, allow_review_job: bool) -> None:
        """Reject conflicting work before receiving an optional request body.

        Answers may be saved while hands are updating; nothing changes review data
        while a recording is prepared or analyzed.
        """
        running = self.workspace.busy()
        if running is not None and (
            running.kind in ANALYSIS_JOBS or not allow_review_job
        ):
            raise HTTPException(
                HTTPStatus.CONFLICT, f"{BUSY[running.kind]}. Wait for it to finish."
            )

    async def mutate(
        self,
        request: Request,
        operation: Callable[[ReviewState, dict], object],
        *,
        allow_review_job: bool = False,
    ) -> Response:
        """Bound incoming JSON and recheck job exclusion atomically before writing."""
        await run_in_threadpool(self.check_write, allow_review_job=allow_review_job)
        body = await read_json_body(request, REVIEW_BODY_LIMIT)

        def apply() -> Response:
            with self.workspace.lock:
                self.check_write(allow_review_job=allow_review_job)
                return json_response(operation(self.state(request), body))

        return await run_in_threadpool(apply)

    async def start(
        self, request: Request, start: Callable[[str, dict], dict]
    ) -> Response:
        """Bound the body of a job request; the workspace claims its job slot."""
        await run_in_threadpool(self.check_write, allow_review_job=False)
        body = await read_json_body(request, REVIEW_BODY_LIMIT)
        key = request.path_params["key"]
        return json_response(await run_in_threadpool(start, key, body))

    def hands(self, request: Request) -> Response:
        """List the recording's hands and whether each awaits an update."""
        return json_response(self.state(request).hand_summary())

    def items(self, request: Request) -> Response:
        """List open review questions across the recording."""
        return json_response(self.state(request).all_items())

    def read(self, request: Request) -> Response:
        """Read tiles only when the analysis pipeline is idle."""
        running = self.workspace.busy()
        if running is not None and running.kind in ANALYSIS_JOBS:
            raise HTTPException(
                HTTPStatus.CONFLICT,
                f"{BUSY[running.kind]}. Tile labeling is available when it finishes.",
            )
        query = request.query_params
        return json_response(
            self.state(request).read(float(query["t"]), query["region"])
        )

    def context(self, request: Request) -> Response:
        """Return evidence surrounding one player's turn."""
        query = request.query_params
        return json_response(
            self.state(request).context(
                int(query["hand"]), query["seat"], float(query["t"])
            )
        )

    def facts(self, request: Request) -> Response:
        """List saved answers, optionally scoped to a hand."""
        hand = request.query_params.get("hand")
        return json_response(
            self.state(request).facts(int(hand) if hand is not None else None)
        )

    def calibration(self, request: Request) -> Response:
        """Return the current recording geometry and its border checks."""
        return json_response(self.state(request).calib())

    def hand(self, request: Request) -> Response:
        """Return a hand's reconstruction and the answers it could not apply."""
        return json_response(self.state(request).hand_view(request.path_params["hand"]))

    def frame(self, request: Request) -> Response:
        """Render one calibrated evidence frame."""
        query = request.query_params
        data = self.state(request).render(
            float(query["t"]),
            query.get("region", "frame"),
            float(query.get("scale", "1")),
        )
        return Response(data, media_type="image/jpeg")

    def clip(self, request: Request) -> Response:
        """Stream a recording-scoped evidence clip through Starlette."""
        query = request.query_params
        path = self.state(request).clip(
            float(query["t0"]), float(query["t1"]), query.get("region", "frame")
        )
        return FileResponse(path, media_type="video/mp4")

    def plate(self, request: Request) -> Response:
        """Encode the prepared calibration plate as JPEG without building one."""
        image = self.state(request).plate()
        if image is None:
            raise HTTPException(
                HTTPStatus.CONFLICT,
                "Prepare the recording to show the table preview.",
            )
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Unable to encode the calibration plate.")
        return Response(buffer.tobytes(), media_type="image/jpeg")

    async def add_fact(self, request: Request) -> Response:
        """Save an answer even while hands are updating."""
        return await self.mutate(
            request,
            lambda state, body: state.add_fact(body),
            allow_review_job=True,
        )

    async def delete_fact(self, request: Request) -> Response:
        """Delete one saved answer even while hands are updating."""
        return await self.mutate(
            request,
            lambda state, body: {"deleted": state.delete_fact(float(body["ts"]))},
            allow_review_job=True,
        )

    async def label(self, request: Request) -> Response:
        """Save one corrected tile label."""
        return await self.mutate(
            request, lambda state, body: {"saved": state.save_label(body).name}
        )

    async def rebuild(self, request: Request) -> Response:
        """Start updating pending hands, all hands, or the listed hands."""
        return await self.start(
            request, lambda key, body: self.workspace.rebuild(key, body.get("hands"))
        )

    async def save_calibration(self, request: Request) -> Response:
        """Persist updated recording geometry."""
        return await self.mutate(request, lambda state, body: state.save_calib(body))

    async def fit_calibration(self, request: Request) -> Response:
        """Start measuring the table again."""
        return await self.start(
            request, lambda key, _body: self.workspace.calibrate(key, "fit")
        )

    async def check_calibration(self, request: Request) -> Response:
        """Start checking the saved borders."""
        return await self.start(
            request, lambda key, _body: self.workspace.calibrate(key, "check")
        )


def review_routes(workspace: Workspace) -> Mount:
    """Let Starlette own review path matching, conversion and allowed methods."""
    endpoints = ReviewRoutes(workspace)
    return Mount(
        "/review/{key}/api",
        routes=[
            Route("/hands", endpoints.hands),
            Route("/items", endpoints.items),
            Route("/read", endpoints.read),
            Route("/context", endpoints.context),
            Route("/facts", endpoints.facts),
            Route("/facts", endpoints.add_fact, methods=["POST"]),
            Route("/facts/delete", endpoints.delete_fact, methods=["POST"]),
            Route("/calib", endpoints.calibration),
            Route("/calib", endpoints.save_calibration, methods=["POST"]),
            Route("/calib/fit", endpoints.fit_calibration, methods=["POST"]),
            Route("/calib/check", endpoints.check_calibration, methods=["POST"]),
            Route("/rebuild", endpoints.rebuild, methods=["POST"]),
            Route("/hand/{hand:int}", endpoints.hand),
            Route("/frame", endpoints.frame),
            Route("/clip", endpoints.clip),
            Route("/plate", endpoints.plate),
            Route("/label", endpoints.label, methods=["POST"]),
        ],
    )
