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

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.requests import Request

    from .review_state import ReviewState
    from .workflow import Workspace


class ReviewRoutes:
    """Bind individual review endpoints to the application-owned workspace."""

    def __init__(self, workspace: Workspace) -> None:
        """Keep all project lookup and job exclusion scoped to this workspace."""
        self.workspace = workspace

    def state(self, request: Request) -> ReviewState:
        """Resolve the project key captured by Starlette's parent mount."""
        return self.workspace.review_state(request.path_params["key"])

    def analysis_running(self) -> bool:
        """Inspect the shared pipeline exclusion state under its lock."""
        with self.workspace.lock:
            return any(
                project.get("job", {}).get("running")
                for project in self.workspace.projects.values()
            )

    def check_write(self, *, allow_review_job: bool) -> None:
        """Reject conflicting work before receiving an optional request body."""
        with self.workspace.lock:
            if self.analysis_running():
                raise HTTPException(
                    HTTPStatus.CONFLICT,
                    "Analysis is running. Wait until it finishes before "
                    "changing review data.",
                )
            if not allow_review_job and any(
                any(job.get("running") for job in state.jobs.values())
                for state in self.workspace.states.values()
            ):
                raise HTTPException(
                    HTTPStatus.CONFLICT,
                    "A review job is running. Wait for it to finish first.",
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

    def revision(self, request: Request) -> Response:
        """Report externally rebuilt review artifacts."""
        return json_response(self.state(request).revision())

    def hands(self, request: Request) -> Response:
        """List the recording's hands and their current review state."""
        return json_response(self.state(request).hand_summary())

    def decode_all_status(self, request: Request) -> Response:
        """Return progress for a full recording rebuild."""
        return json_response(self.state(request).redecode_all_status())

    def decode_pending_status(self, request: Request) -> Response:
        """Return progress for rebuilding pending hands."""
        return json_response(self.state(request).redecode_pending_status())

    def items(self, request: Request) -> Response:
        """List review questions across the recording."""
        return json_response(self.state(request).all_items())

    def read(self, request: Request) -> Response:
        """Read tiles only when the analysis pipeline is idle."""
        if self.analysis_running():
            raise HTTPException(
                HTTPStatus.CONFLICT,
                "Analysis is running. Tile labeling becomes available "
                "when it finishes.",
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
        """Return the current recording geometry."""
        return json_response(self.state(request).calib())

    def calibration_job(self, request: Request) -> Response:
        """Return calibration fitting or checking progress."""
        key = request.query_params.get("key", "calib")
        return json_response(self.state(request).jobs.get(key) or {"running": False})

    def decode_status(self, request: Request) -> Response:
        """Return rebuild progress for the hand captured by Starlette."""
        return json_response(
            self.state(request).redecode_status(request.path_params["hand"])
        )

    def hand(self, request: Request) -> Response:
        """Return a hand's reconstruction alongside its saved answers."""
        state = self.state(request)
        hand = request.path_params["hand"]
        decoded = state.review_decode(hand)
        return json_response(
            {"entry": state.hands[hand], "decode": decoded, "facts": state.facts(hand)}
        )

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
        """Encode the current calibration plate as JPEG."""
        image = self.state(request).plate()
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            message = "Unable to encode the calibration plate."
            raise RuntimeError(message)
        return Response(buffer.tobytes(), media_type="image/jpeg")

    async def add_fact(self, request: Request) -> Response:
        """Save an answer even while a review rebuild is running."""
        return await self.mutate(
            request,
            lambda state, body: state.add_fact(body),
            allow_review_job=True,
        )

    async def delete_fact(self, request: Request) -> Response:
        """Delete one saved answer without blocking an active review rebuild."""
        return await self.mutate(
            request,
            lambda state, body: {"deleted": state.delete_fact(float(body["ts"]))},
            allow_review_job=True,
        )

    async def label(self, request: Request) -> Response:
        """Save one corrected tile label."""
        return await self.mutate(
            request, lambda state, body: {"saved": str(state.save_label(body))}
        )

    async def decode_all(self, request: Request) -> Response:
        """Start a recording-wide rebuild."""
        return await self.mutate(
            request, lambda state, _body: state.start_redecode_all()
        )

    async def decode_pending(self, request: Request) -> Response:
        """Start rebuilding only hands with pending changes."""
        return await self.mutate(
            request, lambda state, _body: state.start_redecode_pending()
        )

    async def save_calibration(self, request: Request) -> Response:
        """Persist updated recording geometry."""
        return await self.mutate(request, lambda state, body: state.save_calib(body))

    async def fit_calibration(self, request: Request) -> Response:
        """Start calibration fitting under the shared job exclusion policy."""
        return await self.mutate(
            request, lambda state, _body: state.start_job("calib", state.run_calib_fit)
        )

    async def check_calibration(self, request: Request) -> Response:
        """Start calibration checking under the shared job exclusion policy."""
        return await self.mutate(
            request,
            lambda state, _body: state.start_job("check", state.run_calib_check),
        )

    async def decode(self, request: Request) -> Response:
        """Start rebuilding the hand captured by Starlette."""
        return await self.mutate(
            request,
            lambda state, _body: state.start_redecode(request.path_params["hand"]),
        )


def review_routes(workspace: Workspace) -> Mount:
    """Let Starlette own review path matching, conversion and allowed methods."""
    endpoints = ReviewRoutes(workspace)
    return Mount(
        "/review/{key}/api",
        routes=[
            Route("/revision", endpoints.revision),
            Route("/hands", endpoints.hands),
            Route("/decode_all", endpoints.decode_all_status),
            Route("/decode_all", endpoints.decode_all, methods=["POST"]),
            Route("/decode_pending", endpoints.decode_pending_status),
            Route("/decode_pending", endpoints.decode_pending, methods=["POST"]),
            Route("/items", endpoints.items),
            Route("/read", endpoints.read),
            Route("/context", endpoints.context),
            Route("/facts", endpoints.facts),
            Route("/facts", endpoints.add_fact, methods=["POST"]),
            Route("/facts/delete", endpoints.delete_fact, methods=["POST"]),
            Route("/calib", endpoints.calibration),
            Route("/calib", endpoints.save_calibration, methods=["POST"]),
            Route("/calib/job", endpoints.calibration_job),
            Route("/calib/fit", endpoints.fit_calibration, methods=["POST"]),
            Route("/calib/check", endpoints.check_calibration, methods=["POST"]),
            Route("/decode/{hand:int}", endpoints.decode_status),
            Route("/decode/{hand:int}", endpoints.decode, methods=["POST"]),
            Route("/hand/{hand:int}", endpoints.hand),
            Route("/frame", endpoints.frame),
            Route("/clip", endpoints.clip),
            Route("/plate", endpoints.plate),
            Route("/label", endpoints.label, methods=["POST"]),
        ],
    )
