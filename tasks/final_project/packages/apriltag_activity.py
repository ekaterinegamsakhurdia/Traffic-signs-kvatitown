import cv2
import numpy as np


class AprilTagDetector:
    def __init__(self, family="tag36h11"):
        self.backend = None
        self.detector = None

        try:
            aruco = cv2.aruco
            dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
            params = aruco.DetectorParameters()
            params.adaptiveThreshWinSizeMin = 3
            params.adaptiveThreshWinSizeMax = 53
            params.adaptiveThreshWinSizeStep = 8
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

            self.detector = aruco.ArucoDetector(dictionary, params)
            self.backend = "opencv"
            return
        except Exception:
            pass

        try:
            from pupil_apriltags import Detector
            self.detector = Detector(families=family)
            self.backend = "pupil_apriltags"
        except Exception:
            self.detector = None
            self.backend = None

    @property
    def ready(self):
        return self.detector is not None

    def detect(self, frame_rgb):
        if not self.ready:
            return []

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        return self._detect_gray(gray, offset_x=0, offset_y=0)

    def detect_inside_signs(self, frame_rgb, detections):

        if not self.ready:
            return []

        found = []

        for bbox, score, cls_id in detections:
            # In your model: 0 duckie, 1 truck, 2 sign
            if cls_id != 2:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]

            h = y2 - y1
            w = x2 - x1

            if h <= 15 or w <= 15:
                continue

            # The AprilTag lives in the top ~58% of the sign bounding box.
            # FIX: tag_y2 is now used as the actual bottom of the crop region.
            tag_y1 = y1
            tag_y2 = y1 + int(h * 0.58)

            pad = 8
            x1p = max(0, x1 - pad)
            x2p = min(frame_rgb.shape[1], x2 + pad)
            y1p = max(0, tag_y1 - pad)
            y2p = min(frame_rgb.shape[0], tag_y2 + pad)  # was y2 + pad — now restricted to tag region

            crop = frame_rgb[y1p:y2p, x1p:x2p]

            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)

            tags = self._detect_gray(gray, offset_x=x1p, offset_y=y1p)

            for t in tags:
                t["source"] = "sign_crop"
                found.append(t)

        return found

    def detect_combined(self, frame_rgb, detections=None):
        detections = detections or []

        full_tags = self.detect(frame_rgb)
        for t in full_tags:
            t["source"] = "full_frame"

        crop_tags = self.detect_inside_signs(frame_rgb, detections)

        # Prefer cropped sign detections because they are related to actual sign objects.
        all_tags = crop_tags + full_tags

        unique = {}
        for t in all_tags:
            tag_id = int(t["id"])
            if tag_id not in unique or t["area"] > unique[tag_id]["area"]:
                unique[tag_id] = t

        return list(unique.values())

    def _detect_gray(self, gray, offset_x=0, offset_y=0):
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

        detections = self.detector.detect(gray)

        tags = []
        for d in detections:
            pts = np.asarray(d.corners, dtype=float)
            pts[:, 0] += offset_x
            pts[:, 1] += offset_y
            tags.append(self._make_tag(int(d.tag_id), pts))

        return tags

    def _make_tag(self, tag_id, pts):
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        return {
            "id": int(tag_id),
            "corners": [(float(x), float(y)) for x, y in pts],
            "center": (cx, cy),
            "area": float(cv2.contourArea(pts.astype(np.float32))),
            "source": "unknown",
        }


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