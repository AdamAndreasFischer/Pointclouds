from __future__ import annotations

import argparse
import struct
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from natnet import DataFrame, NatNetClient
from natnet.data_frame import RigidBody

from calibrate_orbbec import Estimate_charuco_pose, start_orbbec_pipeline

"""
Static (tripod) hand-eye calibration between Motive rigid body and Orbbec camera.

Workflow:
  1. Place the Charuco board flat on a stable surface.
  2. Mount the camera + mocap markers on a tripod.
  3. Move the tripod to a position with a clear board view and wait 1-2 s to settle.
  4. Press SPACE to capture. The script collects --frames-per-pose raw frames,
     checks that both board and mocap are stable, then accepts or rejects the pose.
  5. Move to a new position/tilt and repeat.
  6. Aim for 15-25 poses with good diversity (vary distance, tilt, and direction).
     Press Q when done (≥5 pairs needed, ≥15 recommended).

Why this beats the dynamic version:
  - Averaging N raw frames per static pose reduces board-detection noise significantly.
  - A variance gate rejects any capture where the camera was still moving.
  - Mocap readings are averaged over the same window, eliminating timestamp jitter.
  - No EMA smoothing contamination across different poses.

Pose diversity tips:
  - Tilt the camera toward the board at several different angles.
  - Move the tripod to at least 4-5 distinct XY positions.
  - Include at least one pose where the camera is rolled ~45°.
  - Vary distance from the board (0.5 m – 1.5 m typical).
"""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Invalid zero quaternion from mocap")
    x, y, z, w = q / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz),   1 - 2*(xx+zz),  2*(yz-wx)],
        [  2*(xz-wy),    2*(yz+wx),  1 - 2*(xx+yy)],
    ], dtype=np.float64)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def svd_mean_rotation(R_list: list[np.ndarray]) -> np.ndarray:
    """Geodesic mean of rotation matrices via SVD."""
    M = np.mean(np.stack(R_list), axis=0)
    U, _, Vt = np.linalg.svd(M)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return R_mean


# ---------------------------------------------------------------------------
# Mocap listener (same as dynamic version)
# ---------------------------------------------------------------------------

@dataclass
class MocapPose:
    T_rigid2world: np.ndarray
    stamp: float


class MocapListener:
    def __init__(self, server_ip: str, client_ip: str,
                 rigid_body_id: int | None = None, use_multicast: bool = False):
        self._latest: MocapPose | None = None
        self._lock = threading.Lock()
        self._rigid_body_id = rigid_body_id
        self.client = NatNetClient(
            server_ip_address=server_ip,
            local_ip_address=client_ip,
            use_multicast=use_multicast,
        )
        self.client.on_data_frame_received_event.handlers.append(self._on_frame)

    def _extract_id(self, rb: RigidBody) -> int | None:
        for key in ("id_num", "id", "id_", "rigid_body_id"):
            if hasattr(rb, key):
                return int(getattr(rb, key))
        return None

    def _on_frame(self, data_frame: DataFrame):
        if not isinstance(data_frame, DataFrame):
            return
        rigid_bodies = getattr(data_frame, "rigid_bodies", None)
        if not rigid_bodies:
            return
        selected_rb = None
        if self._rigid_body_id is None:
            selected_rb = rigid_bodies[0]
        else:
            for rb in rigid_bodies:
                if self._extract_id(rb) == self._rigid_body_id:
                    selected_rb = rb
                    break
        if selected_rb is None:
            return
        if hasattr(selected_rb, "tracking_valid") and not bool(selected_rb.tracking_valid):
            return
        pos = np.array(selected_rb.pos, dtype=np.float64)
        qx, qy, qz, qw = selected_rb.rot
        R = quat_xyzw_to_rot(qx, qy, qz, qw)
        with self._lock:
            self._latest = MocapPose(T_rigid2world=make_T(R, pos), stamp=time.time())

    def get_latest(self) -> MocapPose | None:
        with self._lock:
            return self._latest


# ---------------------------------------------------------------------------
# Raw (unsmoothed) board detection — avoids EMA contamination between poses
# ---------------------------------------------------------------------------

def detect_board_raw(board_estimator: Estimate_charuco_pose,
                     frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Single-frame board->camera pose estimation with no EMA smoothing.
    Uses solvePnPRansac directly. Returns (R_3x3, tvec_3x1) or (None, None).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, _, _ = board_estimator.charuco_detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return None, None

    obj_pts, im_pts = cv2.aruco.Board.matchImagePoints(
        board_estimator.board,
        detectedCorners=charuco_corners,
        detectedIds=charuco_ids,
    )
    if obj_pts is None or obj_pts.shape[0] < 4:
        return None, None
    if not (np.isfinite(obj_pts).all() and np.isfinite(im_pts).all()):
        return None, None

    K = np.array(board_estimator.color_intrinsics["camera_matrix"])
    D = np.array(board_estimator.color_intrinsics["distortion_coefficients"])
    retval, rvec, tvec, _ = cv2.solvePnPRansac(
        objectPoints=obj_pts, imagePoints=im_pts,
        cameraMatrix=K, distCoeffs=D,
    )
    if not retval or rvec is None or tvec is None:
        return None, None
    if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return None, None

    R_mat, _ = cv2.Rodrigues(rvec)
    return R_mat, tvec


# ---------------------------------------------------------------------------
# Static pose capture
# ---------------------------------------------------------------------------

def capture_static_pose(
    board_estimator: Estimate_charuco_pose,
    pipeline,
    mocap_listener: MocapListener,
    n_frames: int,
    warmup: int,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    """
    Collect n_frames raw board+mocap poses from a stationary camera.

    Skips the first `warmup` frames to let the camera settle after the SPACE keypress.
    Returns (T_board2cam_mean, T_mocap_mean, board_tvec_std_m, mocap_tvec_std_m)
    or (None, None, 0, 0) on failure.
    """
    R_cam_list: list[np.ndarray] = []
    t_cam_list: list[np.ndarray] = []
    T_mocap_list: list[np.ndarray] = []

    collected = 0
    attempts = 0
    max_attempts = (n_frames + warmup) * 8

    while collected < (n_frames + warmup) and attempts < max_attempts:
        attempts += 1

        try:
            mocap_listener.client.update_sync()
        except (BlockingIOError, struct.error):
            pass

        frame, _ = board_estimator.get_camera_stream(pipeline)
        if frame is None:
            continue

        R_cb, t_cb = detect_board_raw(board_estimator, frame)
        if R_cb is None or t_cb is None:
            continue

        mocap_pose = mocap_listener.get_latest()
        if mocap_pose is None:
            continue
        if not np.isfinite(mocap_pose.T_rigid2world).all():
            continue

        collected += 1
        if collected <= warmup:
            continue  # discard settling frames

        R_cam_list.append(R_cb)
        t_cam_list.append(t_cb.reshape(3))
        T_mocap_list.append(mocap_pose.T_rigid2world.copy())

    if len(R_cam_list) < max(5, n_frames // 3):
        return None, None, 0.0, 0.0

    R_mean_cam = svd_mean_rotation(R_cam_list)
    t_arr_cam = np.stack(t_cam_list)
    t_mean_cam = t_arr_cam.mean(axis=0)
    board_std = float(t_arr_cam.std(axis=0).max())

    R_mean_mocap = svd_mean_rotation([T[:3, :3] for T in T_mocap_list])
    t_arr_mocap = np.stack([T[:3, 3] for T in T_mocap_list])
    t_mean_mocap = t_arr_mocap.mean(axis=0)
    mocap_std = float(t_arr_mocap.std(axis=0).max())

    return make_T(R_mean_cam, t_mean_cam), make_T(R_mean_mocap, t_mean_mocap), board_std, mocap_std


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    n_collected: int,
    n_target: int,
    board_ok: bool,
    mocap_ok: bool,
    status_msg: str = "",
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    green, red, white, cyan = (0, 220, 0), (0, 60, 220), (255, 255, 255), (220, 220, 0)

    cv2.putText(frame, f"Poses: {n_collected}/{n_target}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
    cv2.putText(frame, f"Board: {'OK' if board_ok else 'NOT DETECTED'}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, green if board_ok else red, 2)
    cv2.putText(frame, f"Mocap: {'OK' if mocap_ok else 'NO SIGNAL'}", (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, green if mocap_ok else red, 2)
    cv2.putText(frame, "SPACE=capture  Q=finish", (w - 310, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    if status_msg:
        cv2.putText(frame, status_msg, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, cyan, 2)


def check_pose_diversity(
    R_list: list[np.ndarray], t_list: list[np.ndarray]
) -> tuple[float, float]:
    """Return (max_rotation_spread_deg, translation_spread_m) across collected poses."""
    if len(R_list) < 2:
        return 0.0, 0.0
    t_arr = np.stack([t.reshape(3) for t in t_list])
    trans_spread = float(np.max(np.linalg.norm(t_arr - t_arr.mean(0), axis=1)))
    R0 = R_list[0]
    max_angle = 0.0
    for R in R_list[1:]:
        cos_theta = np.clip((np.trace(R0.T @ R) - 1) / 2, -1.0, 1.0)
        max_angle = max(max_angle, float(np.degrees(np.arccos(cos_theta))))
    return max_angle, trans_spread


def get_method(name: str) -> int:
    return {
        "park": cv2.CALIB_HAND_EYE_PARK,
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }[name]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Static (tripod) hand-eye calibration: Motive rigid body -> Orbbec camera"
    )
    p.add_argument("--server-ip", type=str, required=True, help="Motive/NatNet server IP")
    p.add_argument("--client-ip", type=str, required=True, help="This machine's IP for NatNet")
    p.add_argument("--use-multicast", action="store_true")
    p.add_argument("--rigid-body-id", type=int, default=None, help="Motive rigid body ID (default: first)")
    p.add_argument("--num-poses", type=int, default=20,
                   help="Number of static pose pairs to collect")
    p.add_argument("--frames-per-pose", type=int, default=40,
                   help="Raw frames averaged per static pose (higher = less noise, slower capture)")
    p.add_argument("--warmup-frames", type=int, default=10,
                   help="Frames to discard at the start of each capture (camera settling)")
    p.add_argument("--max-board-std-mm", type=float, default=0.3,
                   help="Reject pose if board tvec std dev exceeds this (mm). "
                        "Increase if captures are too often rejected.")
    p.add_argument("--max-mocap-std-mm", type=float, default=0.8,
                   help="Reject pose if mocap tvec std dev exceeds this (mm).")
    p.add_argument("--max-output-trans-m", type=float, default=2.0,
                   help="Sanity-check on result translation magnitude")
    p.add_argument("--method", type=str, default="park",
                   choices=["park", "tsai", "horaud", "andreff", "daniilidis"])
    p.add_argument("--save-path", type=str, default="mocap_rigidbody_to_camera.npy")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    max_board_std_m = args.max_board_std_mm / 1000.0
    max_mocap_std_m = args.max_mocap_std_mm / 1000.0

    print("Starting Orbbec + NatNet for static hand-eye calibration...")
    pipeline = None

    try:
        pipeline = start_orbbec_pipeline()
        board_estimator = Estimate_charuco_pose()
        mocap_listener = MocapListener(
            server_ip=args.server_ip,
            client_ip=args.client_ip,
            rigid_body_id=args.rigid_body_id,
            use_multicast=args.use_multicast,
        )

        R_gripper2base: list[np.ndarray] = []
        t_gripper2base: list[np.ndarray] = []
        R_target2cam: list[np.ndarray] = []
        t_target2cam: list[np.ndarray] = []

        method = get_method(args.method)

        print(f"\n=== Static calibration — target: {args.num_poses} poses ===")
        print(f"  {args.frames_per_pose} frames averaged per pose, "
              f"{args.warmup_frames} warmup frames discarded")
        print(f"  Board std limit: {args.max_board_std_mm:.1f} mm | "
              f"Mocap std limit: {args.max_mocap_std_mm:.1f} mm")
        print("\nControls: SPACE = capture pose | Q = finish early (need ≥5 pairs)\n")

        status_msg = "Ready — position the camera, wait for it to settle, then press SPACE"

        with mocap_listener.client:
            while len(R_gripper2base) < args.num_poses:

                # Drive mocap UDP receive
                try:
                    mocap_listener.client.update_sync()
                except (BlockingIOError, struct.error):
                    pass

                # Live preview with EMA smoothing (visual feedback only)
                frame, _ = board_estimator.get_camera_stream(pipeline)
                if frame is None:
                    cv2.waitKey(1)
                    continue

                R_live, t_live = board_estimator.detect_board(frame, debug=False)
                board_ok = R_live is not None

                if board_ok:
                    try:
                        rvec_live, _ = cv2.Rodrigues(R_live)
                        K = np.array(board_estimator.color_intrinsics["camera_matrix"])
                        D = np.array(board_estimator.color_intrinsics["distortion_coefficients"])
                        cv2.drawFrameAxes(frame, K, D, rvec_live, t_live, 0.1)
                    except Exception:
                        pass

                mocap_ok = mocap_listener.get_latest() is not None
                draw_overlay(frame, len(R_gripper2base), args.num_poses,
                             board_ok, mocap_ok, status_msg)
                cv2.imshow("Hand-Eye Calibration (static)", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    if len(R_gripper2base) >= 5:
                        print("Finishing early on user request...")
                        break
                    print(f"  Need ≥5 pairs to run calibration (have {len(R_gripper2base)}). Keep collecting.")
                    continue

                if key != ord(" "):
                    continue

                # --- SPACE pressed: capture static pose ---
                if not board_ok:
                    status_msg = "REJECTED: board not visible — reposition and try again"
                    print(f"  {status_msg}")
                    continue
                if not mocap_ok:
                    status_msg = "REJECTED: no mocap signal"
                    print(f"  {status_msg}")
                    continue

                pose_idx = len(R_gripper2base) + 1
                print(f"\n[Pose {pose_idx}/{args.num_poses}] Capturing "
                      f"{args.frames_per_pose} frames ({args.warmup_frames} warmup)... hold still!")

                # Flash "hold still" before the blocking capture
                status_msg = "Capturing — HOLD STILL"
                draw_overlay(frame, len(R_gripper2base), args.num_poses,
                             board_ok, mocap_ok, status_msg)
                cv2.imshow("Hand-Eye Calibration (static)", frame)
                cv2.waitKey(1)

                T_b2cam, T_mocap, board_std, mocap_std = capture_static_pose(
                    board_estimator, pipeline, mocap_listener,
                    n_frames=args.frames_per_pose,
                    warmup=args.warmup_frames,
                )

                if T_b2cam is None:
                    status_msg = "FAILED: not enough board detections — try again"
                    print(f"  {status_msg}")
                    continue

                board_std_mm = board_std * 1000
                mocap_std_mm = mocap_std * 1000

                if board_std > max_board_std_m:
                    status_msg = (f"REJECTED: board std = {board_std_mm:.1f} mm "
                                  f"(limit {args.max_board_std_mm:.0f} mm) — camera was moving")
                    print(f"  {status_msg}")
                    continue

                if mocap_std > max_mocap_std_m:
                    status_msg = (f"REJECTED: mocap std = {mocap_std_mm:.2f} mm "
                                  f"(limit {args.max_mocap_std_mm:.1f} mm) — tripod was vibrating")
                    print(f"  {status_msg}")
                    continue

                t_board = T_b2cam[:3, 3]
                if np.any(np.abs(t_board) > 1.5):
                    status_msg = f"REJECTED: implausible board position {t_board.round(3)}"
                    print(f"  {status_msg}")
                    continue

                R_gripper2base.append(T_mocap[:3, :3].astype(np.float64))
                t_gripper2base.append(T_mocap[:3, 3].reshape(3, 1).astype(np.float64))
                R_target2cam.append(T_b2cam[:3, :3].astype(np.float64))
                t_target2cam.append(T_b2cam[:3, 3].reshape(3, 1).astype(np.float64))

                n = len(R_gripper2base)
                status_msg = (f"Accepted #{n}  "
                              f"board_std={board_std_mm:.1f} mm  "
                              f"mocap_std={mocap_std_mm:.2f} mm")
                print(f"  {status_msg}")
                remaining = args.num_poses - n
                if remaining > 0:
                    print(f"  Move to next position. {remaining} more needed.\n")

        # -------------------------------------------------------------------
        # Calibration
        # -------------------------------------------------------------------

        if len(R_gripper2base) < 5:
            raise RuntimeError("Too few valid samples. Need at least 5, ideally 15+.")

        rot_spread, trans_spread = check_pose_diversity(R_gripper2base, t_gripper2base)
        print(f"\nPose diversity: rotation span = {rot_spread:.1f}°, "
              f"translation span = {trans_spread * 100:.1f} cm")
        if rot_spread < 30:
            print("  WARNING: low rotation diversity — tilt the camera more between captures")
        if trans_spread < 0.05:
            print("  WARNING: low translation diversity — move the tripod further between positions")

        print("\nRunning hand-eye calibration...")
        R_cam2rigid, t_cam2rigid = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=method,
        )

        T_cam2rigid = make_T(R_cam2rigid, t_cam2rigid)
        T_rigid2cam = invert_T(T_cam2rigid)

        if not np.isfinite(T_rigid2cam).all():
            raise RuntimeError(
                "Hand-eye produced non-finite matrix. "
                "Collect more poses with better rotational diversity."
            )

        trans_norm = float(np.linalg.norm(T_rigid2cam[:3, 3]))
        if trans_norm > args.max_output_trans_m:
            raise RuntimeError(
                f"Implausible translation magnitude ({trans_norm:.3f} m). "
                "Check that the board was visible in all accepted captures."
            )

        np.save(args.save_path, T_rigid2cam)

        print("\n=== Calibration result ===")
        print("T_rigidbody_to_camera (mocap rigid-body frame -> camera optical frame):")
        print(T_rigid2cam)
        print(f"\nTranslation magnitude: {trans_norm * 100:.2f} cm")
        print(f"Saved to: {args.save_path}")
        print("\nUsage:")
        print("  T_world_to_camera = T_world_to_rigidbody @ T_rigidbody_to_camera")

    finally:
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
