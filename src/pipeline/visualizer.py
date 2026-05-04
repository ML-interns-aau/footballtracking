import cv2
import supervision as sv
import numpy as np


class PipelineVisualizer:
    """
    Enhanced visualizer:
    - Colour-coded team boxes (red / blue / yellow for referee / white for ball)
    - Filled team badge circle above each player box
    - Referee gets yellow 'REF' label and box
    - Ball trail drawn as fading white/yellow dots
    - Predicted ball position shown with a dashed circle
    - Top-left HUD: frame index, active player count, active ball speed
    """

    TEAM_COLORS_BGR = {
        0:          (40,  40, 230),   # BGR red  → Team 0
        1:          (230, 100, 40),   # BGR blue → Team 1
        -1:         (150, 150, 150),  # unknown / grey
        -2:         (40,  230, 230),  # referee  → yellow
        -3:         (200, 50, 200),   # GK0      → purple
        -4:         (50, 200, 50),    # GK1      → green
        "ball":     (255, 255, 255),  # white
    }

    BADGE_COLORS_BGR = {
        0:  (40,  40, 230),
        1:  (230, 100, 40),
        -1: (150, 150, 150),
        -2: (40,  230, 230),
        -3: (200, 50, 200),
        -4: (50, 200, 50),
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_dashed_rect(img, pt1, pt2, color, thickness=2, gap=8):
        x1, y1 = pt1
        x2, y2 = pt2
        pts = [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]
        for lx1, ly1, lx2, ly2 in pts:
            dx = lx2 - lx1
            dy = ly2 - ly1
            length = max(1, int(np.hypot(dx, dy)))
            for i in range(0, length, gap * 2):
                s = i / length
                e = min((i + gap) / length, 1.0)
                sx, sy = int(lx1 + s * dx), int(ly1 + s * dy)
                ex, ey = int(lx1 + e * dx), int(ly1 + e * dy)
                cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    @staticmethod
    def _put_text_with_shadow(img, text, org, font_scale=0.45, thickness=1, color=(255, 255, 255)):
        cv2.putText(img, text, (org[0] + 1, org[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, org,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Ball trail
    # ------------------------------------------------------------------
    def draw_ball_trail(self, frame: np.ndarray, trail: list[tuple[float, float]], is_predicted: bool):
        """Draw fading dot trail behind the ball."""
        n = len(trail)
        for i, (cx, cy) in enumerate(trail):
            alpha = (i + 1) / max(n, 1)
            radius = max(2, int(4 * alpha))
            color = (180, 180, 50) if is_predicted else (255, 255, 255)
            intensity = int(255 * alpha)
            dot_color = tuple(int(c * intensity / 255) for c in color)
            cv2.circle(frame, (int(cx), int(cy)), radius, dot_color, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Main annotation pass
    # ------------------------------------------------------------------
    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        team_ids: np.ndarray,
        speed_estimator,
        ball_trail: list[tuple[float, float]] = None,
        ball_speed_kmh: float = 0.0,
        ball_is_predicted: bool = False,
        frame_idx: int = 0,
    ) -> np.ndarray:

        annotated = frame.copy()
        player_count = 0

        # --- Draw ball trail first (background layer) ---
        if ball_trail:
            self.draw_ball_trail(annotated, ball_trail, ball_is_predicted)

        # --- Per-detection annotation ---
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # ---- Ball ----
            if class_id == 32:
                color = self.TEAM_COLORS_BGR["ball"]
                if ball_is_predicted:
                    self._draw_dashed_rect(annotated, (x1, y1), (x2, y2), (100, 200, 100), 1)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated, (cx, cy), 6, color, -1, cv2.LINE_AA)
                self._put_text_with_shadow(annotated, "Ball", (x1, y1 - 6))
                continue

            # ---- Player / Referee ----
            if class_id != 0 or tracker_id is None:
                continue

            tid = int(team_ids[i])

            # Box colour
            box_color = self.TEAM_COLORS_BGR.get(tid, self.TEAM_COLORS_BGR[-1])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Badge circle above box
            badge_cx = (x1 + x2) // 2
            badge_cy = max(y1 - 12, 12)
            badge_col = self.BADGE_COLORS_BGR.get(tid, self.BADGE_COLORS_BGR[-1])
            cv2.circle(annotated, (badge_cx, badge_cy), 9, badge_col, -1, cv2.LINE_AA)
            cv2.circle(annotated, (badge_cx, badge_cy), 9, (255, 255, 255), 1, cv2.LINE_AA)

            # Label
            if tid == -2:
                label = f"REF #{tracker_id}"
                self._put_text_with_shadow(annotated, label, (x1, y1 - 18), color=(40, 230, 230))
            elif tid == -3:
                label = f"GK0 #{tracker_id}"
                self._put_text_with_shadow(annotated, label, (x1, y1 - 18), color=(200, 50, 200))
            elif tid == -4:
                label = f"GK1 #{tracker_id}"
                self._put_text_with_shadow(annotated, label, (x1, y1 - 18), color=(50, 200, 50))
            else:
                player_count += 1
                speed, dist, (x_m, y_m) = speed_estimator.get_stats(tracker_id)
                team_label = f"T{tid}|#{tracker_id}"
                self._put_text_with_shadow(annotated, team_label, (x1, y1 - 18))
                # Speed & distance below box
                info_y = y2 + 15
                self._put_text_with_shadow(annotated, f"{speed:.1f}km/h  {dist:.0f}m",
                                           (x1, info_y), font_scale=0.38)

        # --- HUD top-left panel ---
        self._draw_hud(annotated, frame_idx, player_count, ball_speed_kmh, ball_is_predicted)

        return annotated

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def _draw_hud(self, frame, frame_idx, player_count, ball_speed, ball_predicted):
        panel_w, panel_h = 240, 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), (80, 80, 80), 1)

        lines = [
            f"Frame: {frame_idx}",
            f"Players tracked: {player_count}",
            f"Ball speed: {ball_speed:.1f} km/h" + (" (est)" if ball_predicted else ""),
        ]
        for j, line in enumerate(lines):
            self._put_text_with_shadow(frame, line, (16, 28 + j * 19), font_scale=0.42)
