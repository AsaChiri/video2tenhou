"""Project-scoped review API dispatch; no listener or global recording state."""

import cv2


def get_review(handler, state, path, q):
    if path == "/api/revision":
        return handler._json(state.revision())
    if path == "/api/hands":
        return handler._json(state.hand_summary())
    if path == "/api/decode_all":
        return handler._json(state.redecode_all_status())
    if path == "/api/decode_pending":
        return handler._json(state.redecode_pending_status())
    if path.startswith("/api/decode/"):
        return handler._json(state.redecode_status(int(path.rsplit("/", 1)[1])))
    if path == "/api/items":
        return handler._json(state.all_items())
    if path.startswith("/api/hand/"):
        i = int(path.rsplit("/", 1)[1])
        d = state.review_decode(i)
        return handler._json(
            {"entry": state.hands[i], "decode": d, "facts": state.facts(i)}
        )
    if path == "/api/frame":
        data = state.render(
            float(q["t"]), q.get("region", "frame"), float(q.get("scale", "1"))
        )
        return handler._bytes(data, "image/jpeg")
    if path == "/api/read":
        return handler._json(state.read(float(q["t"]), q["region"]))
    if path == "/api/context":
        return handler._json(state.context(int(q["hand"]), q["seat"], float(q["t"])))
    if path == "/api/clip":
        p = state.clip(float(q["t0"]), float(q["t1"]), q.get("region", "frame"))
        return handler._bytes(p.read_bytes(), "video/mp4")
    if path == "/api/facts":
        return handler._json(state.facts(int(q["hand"]) if "hand" in q else None))
    if path == "/api/calib":
        return handler._json(state.calib())
    if path == "/api/calib/job":
        return handler._json(
            state.jobs.get(q.get("key", "calib")) or {"running": False}
        )
    if path == "/api/plate":
        img = state.plate()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return handler._bytes(buf.tobytes(), "image/jpeg")
    return handler._json({"error": "Unknown review endpoint."}, 404)


def post_review(handler, state, path, body):
    if path == "/api/facts":
        return handler._json(state.add_fact(body))
    if path == "/api/facts/delete":
        return handler._json({"deleted": state.delete_fact(float(body["ts"]))})
    if path == "/api/label":
        return handler._json({"saved": str(state.save_label(body))})
    if path == "/api/decode_all":
        return handler._json(state.start_redecode_all())
    if path == "/api/decode_pending":
        return handler._json(state.start_redecode_pending())
    if path.startswith("/api/decode/"):
        return handler._json(state.start_redecode(int(path.rsplit("/", 1)[1])))
    if path == "/api/calib":
        return handler._json(state.save_calib(body))
    if path == "/api/calib/fit":
        return handler._json(state.start_job("calib", state.run_calib_fit))
    if path == "/api/calib/check":
        return handler._json(state.start_job("check", state.run_calib_check))
    return handler._json({"error": "Unknown review endpoint."}, 404)
