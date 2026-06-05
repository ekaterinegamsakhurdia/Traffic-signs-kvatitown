import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Raw AprilTag 36h11 decoder — requires only base OpenCV (no contrib/aruco).
# Used as a fallback when cv2.aruco and pupil_apriltags are both unavailable.
# ---------------------------------------------------------------------------

# How many consecutive frames a tag must be seen before it is reported.
TAG_CONFIRM_FRAMES = 2

# 36-bit canonical codes (k=0 rotation) for every tag ID used by this project.
# Generated with:
#   d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
#   img = cv2.aruco.generateImageMarker(d, tag_id, 80)
#   _, bw = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#   row_centres = np.arange(8) * 10 + 5
#   col_centres = np.arange(8) * 10 + 5
#   bits = bw[np.ix_(row_centres, col_centres)]
#   inner = bits[1:7, 1:7]
#   code = int(inner.ravel().dot(1 << np.arange(35, -1, -1, dtype=np.int64)))
_CODES_36H11 = {
    0x4e20b5a64: 9,
    0x61d897f2c: 10,
    0xab3469ffc: 11,
    0xf52925b81: 20,
    0x3913d5345: 24,
    0xd79b1a3b5: 25,
    0xdfbe768d:  26,
    0xad6a1caf9: 39,
}

_DST_PTS = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], dtype=np.float32)


def _order_corners(pts):
    """Sort 4 points into TL, TR, BR, BL order."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL: smallest x+y
        pts[np.argmin(d)],   # TR: smallest y-x
        pts[np.argmax(s)],   # BR: largest x+y
        pts[np.argmax(d)],   # BL: largest y-x
    ])


def _decode_warped(bw80):
    """
    bw80: 80x80 binary image (values 0 or 1).
    Samples the 8x8 cell grid (each cell 10x10 pixels, sampled at centre).
    AprilTag 36h11 layout: 1-cell black border + 6x6 data bits.
    Tries all 4 rotations of the inner grid against _CODES_36H11.
    Returns tag_id (int) or None.
    """
    row_centres = np.arange(8) * 10 + 5   # [5, 15, 25, 35, 45, 55, 65, 75]
    col_centres = np.arange(8) * 10 + 5
    bits = bw80[np.ix_(row_centres, col_centres)]   # shape (8, 8)

    # Outer ring must be all black (0) — if not, this is not a valid tag.
    border = np.concatenate([
        bits[0, :], bits[7, :],
        bits[1:7, 0], bits[1:7, 7],
    ])
    if np.any(border != 0):
        return None

    inner = bits[1:7, 1:7]   # 6x6 data bits

    for k in range(4):
        rotated = np.rot90(inner, k=k)
        code = int(rotated.ravel().dot(1 << np.arange(35, -1, -1, dtype=np.int64)))
        if code in _CODES_36H11:
            return _CODES_36H11[code]

    return None


def _raw_detect(gray):
    """
    Detect AprilTag 36h11 markers using only base OpenCV (no contrib/aruco).
    Returns list of {"tag_id": int, "corners": (4,2) float32 array}.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5,
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    max_area = h * w * 0.5

    tags = []
    seen = set()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > max_area:
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        src = _order_corners(approx.reshape(4, 2))
        M   = cv2.getPerspectiveTransform(src, _DST_PTS)
        warped = cv2.warpPerspective(gray, M, (80, 80))

        _, bw = cv2.threshold(warped, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        tag_id = _decode_warped(bw)
        if tag_id is not None and tag_id not in seen:
            seen.add(tag_id)
            tags.append({"tag_id": tag_id, "corners": src})

    return tags


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class AprilTagDetector:
    def __init__(self, family="tag36h11"):
        self.backend  = None
        self.detector = None

        # Temporal confirmation buffer: tag_id → consecutive-seen-frame count.
        # A tag must be seen for TAG_CONFIRM_FRAMES frames before being reported.
        self._tag_buffer: dict = {}

        # ── Backend 1: opencv-contrib aruco (fast, simulation-friendly) ──────
        try:
            aruco      = cv2.aruco
            dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
            params     = aruco.DetectorParameters()
            params.adaptiveThreshWinSizeMin  = 3
            params.adaptiveThreshWinSizeMax  = 53
            params.adaptiveThreshWinSizeStep = 8
            params.cornerRefinementMethod    = aruco.CORNER_REFINE_SUBPIX

            self.detector = aruco.ArucoDetector(dictionary, params)
            self.backend  = "opencv"
            return
        except Exception:
            pass

        # ── Backend 2: pupil_apriltags ────────────────────────────────────────
        try:
            from pupil_apriltags import Detector
            self.detector = Detector(families=family)
            self.backend  = "pupil_apriltags"
            return
        except Exception:
            pass

        # ── Backend 3: raw OpenCV fallback (always available) ─────────────────
        # No external library required. Slower and less robust than the above,
        # but works on the physical robot even without opencv-contrib installed.
        self.backend = "raw"
        # self.detector stays None; _raw_detect() is called directly.

    @property
    def ready(self):
        # Always True now: "raw" is always available as the final fallback.
        return self.backend is not None

    # ------------------------------------------------------------------
    # Detection entry points
    # ------------------------------------------------------------------

    def detect(self, frame_rgb):
        """Full-frame detection."""
        if not self.ready:
            return []

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        return self._detect_gray(gray, offset_x=0, offset_y=0)

    def detect_inside_signs(self, frame_rgb, detections):
        """Crop to sign bounding boxes and detect only in the tag region."""
        if not self.ready:
            return []

        found = []

        for bbox, score, cls_id in detections:
            if cls_id != 2:   # 2 = sign
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            h = y2 - y1
            w = x2 - x1
            if h <= 15 or w <= 15:
                continue

            # AprilTag lives in the top ~58% of the sign bounding box.
            tag_y1 = y1
            tag_y2 = y1 + int(h * 0.58)

            pad = 8
            x1p = max(0, x1 - pad)
            x2p = min(frame_rgb.shape[1], x2 + pad)
            y1p = max(0, tag_y1 - pad)
            y2p = min(frame_rgb.shape[0], tag_y2 + pad)

            crop = frame_rgb[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)

            for t in self._detect_gray(gray, offset_x=x1p, offset_y=y1p):
                t["source"] = "sign_crop"
                found.append(t)

        return found

    def detect_combined(self, frame_rgb, detections=None):
        """
        Merge full-frame and sign-crop detections, deduplicate by largest area,
        apply temporal confirmation, and return confirmed tags.
        """
        detections = detections or []

        full_tags = self.detect(frame_rgb)
        for t in full_tags:
            t["source"] = "full_frame"

        crop_tags = self.detect_inside_signs(frame_rgb, detections)

        # Prefer sign-crop detections (more precise); keep largest area per ID.
        all_tags = crop_tags + full_tags
        unique: dict = {}
        for t in all_tags:
            tid = int(t["id"])
            if tid not in unique or t["area"] > unique[tid]["area"]:
                unique[tid] = t

        # ── Temporal confirmation ──────────────────────────────────────────
        # Remove tags from buffer that are no longer visible this frame.
        current_ids = set(unique.keys())
        for tid in list(self._tag_buffer.keys()):
            if tid not in current_ids:
                del self._tag_buffer[tid]

        # Increment counter for every currently visible tag.
        for tid in current_ids:
            self._tag_buffer[tid] = self._tag_buffer.get(tid, 0) + 1

        # Only return tags that have been seen for TAG_CONFIRM_FRAMES frames.
        confirmed = [
            tag for tid, tag in unique.items()
            if self._tag_buffer.get(tid, 0) >= TAG_CONFIRM_FRAMES
        ]

        return confirmed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_gray(self, gray, offset_x=0, offset_y=0):
        """Route to the active backend and return a list of tag dicts."""

        if self.backend == "opencv":
            corners, ids, _ = self.detector.detectMarkers(gray)
            if ids is None:
                return []
            tags = []
            for c, tag_id in zip(corners, ids.flatten()):
                pts = c.reshape(-1, 2).astype(float)
                pts[:, 0] += offset_x
                pts[:, 1] += offset_y
                tags.append(self._make_tag(int(tag_id), pts))
            return tags

        if self.backend == "pupil_apriltags":
            raw = self.detector.detect(gray)
            tags = []
            for d in raw:
                pts = np.asarray(d.corners, dtype=float)
                pts[:, 0] += offset_x
                pts[:, 1] += offset_y
                tags.append(self._make_tag(int(d.tag_id), pts))
            return tags

        # backend == "raw"
        raw = _raw_detect(gray)
        tags = []
        for r in raw:
            pts = r["corners"].astype(float)
            pts[:, 0] += offset_x
            pts[:, 1] += offset_y
            tags.append(self._make_tag(int(r["tag_id"]), pts))
        return tags

    def _make_tag(self, tag_id, pts):
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        return {
            "id":      int(tag_id),
            "corners": [(float(x), float(y)) for x, y in pts],
            "center":  (cx, cy),
            "area":    float(cv2.contourArea(pts.astype(np.float32))),
            "source":  "unknown",
        }


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_tags(bgr, tags):
    for tag in tags:
        pts = np.asarray(tag["corners"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(bgr, [pts], True, (255, 120, 0), 2)
        cx, cy = map(int, tag["center"])
        cv2.putText(
            bgr,
            f"APRILTAG {tag['id']} {tag.get('source', '')}",
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )