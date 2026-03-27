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

from calibrate_orbbec import Estimate_charuco_pose


"""
Quick hand-eye calibration between:
  - Mocap rigid body (on camera) from Motive/NatNet
  - Camera optical frame from Charuco board pose estimation

This uses cv2.calibrateHandEye with:
  - gripper2base := rigid_body2mocap_world  (from NatNet)
  - target2cam   := charuco_board2camera    (from Estimate_charuco_pose)

Result from OpenCV is cam2rigid. We also return rigid2cam (its inverse), which is
usually what you want to transform mocap rigid-body frame points into camera frame.
"""


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
	"""Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
	q = np.array([qx, qy, qz, qw], dtype=np.float64)
	n = np.linalg.norm(q)
	if n < 1e-12:
		raise ValueError("Invalid zero quaternion from mocap")
	x, y, z, w = q / n

	xx, yy, zz = x * x, y * y, z * z
	xy, xz, yz = x * y, x * z, y * z
	wx, wy, wz = w * x, w * y, w * z

	return np.array(
		[
			[1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
			[2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
			[2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
		],
		dtype=np.float64,
	)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
	T = np.eye(4, dtype=np.float64)
	T[:3, :3] = R
	T[:3, 3] = t.reshape(3)
	return T


def invert_T(T: np.ndarray) -> np.ndarray:
	R = T[:3, :3]
	t = T[:3, 3]
	T_inv = np.eye(4, dtype=np.float64)
	T_inv[:3, :3] = R.T
	T_inv[:3, 3] = -R.T @ t
	return T_inv


@dataclass
class MocapPose:
	T_rigid2world: np.ndarray
	stamp: float


class MocapListener:
	def __init__(
		self,
		server_ip: str,
		client_ip: str,
		rigid_body_id: int | None = None,
		use_multicast: bool = False,
	):
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
				rb_id = self._extract_id(rb)
				if rb_id == self._rigid_body_id:
					selected_rb = rb
					break

		if selected_rb is None:
			return

		if hasattr(selected_rb, "tracking_valid") and (not bool(selected_rb.tracking_valid)):
			return

		pos = np.array(selected_rb.pos, dtype=np.float64)
		qx, qy, qz, qw = selected_rb.rot
		R = quat_xyzw_to_rot(qx, qy, qz, qw)
		T = make_T(R, pos)

		with self._lock:
			self._latest = MocapPose(T_rigid2world=T, stamp=time.time())

	def get_latest(self) -> MocapPose | None:
		with self._lock:
			return self._latest


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Hand-eye calibration: Motive rigid body -> Orbbec camera")
	parser.add_argument("--server-ip", default="192.168.1.155",type=str, required=True, help="Motive/NatNet server IP")
	parser.add_argument("--client-ip",default="192.168.1.234" ,type=str, required=True, help="This machine's IP for NatNet")
	parser.add_argument(
		"--use-multicast",
		action="store_true",
		help="Enable multicast mode (must match Motive streaming setting)",
	)
	parser.add_argument("--rigid-body-id", type=int, default=None, help="Optional Motive rigid body id")
	parser.add_argument("--num-samples", type=int, default=200, help="Number of valid pose pairs")
	parser.add_argument("--max-seconds", type=float, default=120.0, help="Maximum collection time")
	parser.add_argument("--min-rot-step-deg", type=float, default=1.0, help="Min relative mocap rotation between accepted samples")
	parser.add_argument("--min-trans-step-m", type=float, default=0.005, help="Min relative mocap translation between accepted samples")
	parser.add_argument("--max-output-trans-m", type=float, default=2.0, help="Reject result if |t_rigidbody_to_camera| exceeds this")
	parser.add_argument(
		"--method",
		type=str,
		default="park",
		choices=["park", "tsai", "horaud", "andreff", "daniilidis"],
		help="OpenCV hand-eye method",
	)
	parser.add_argument(
		"--save-path",
		type=str,
		default="mocap_rigidbody_to_camera.npy",
		help="Where to save 4x4 rigidbody->camera transform",
	)
	return parser.parse_args()


def get_method(method_name: str) -> int:
	methods = {
		"park": cv2.CALIB_HAND_EYE_PARK,
		"tsai": cv2.CALIB_HAND_EYE_TSAI,
		"horaud": cv2.CALIB_HAND_EYE_HORAUD,
		"andreff": cv2.CALIB_HAND_EYE_ANDREFF,
		"daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
	}
	return methods[method_name]


def relative_motion(T_prev: np.ndarray, T_curr: np.ndarray) -> tuple[float, float]:
	"""Return (rotation_deg, translation_m) for T_prev^-1 * T_curr."""
	T_rel = invert_T(T_prev) @ T_curr
	R_rel = T_rel[:3, :3]
	t_rel = T_rel[:3, 3]
	cos_theta = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
	rot_deg = float(np.degrees(np.arccos(cos_theta)))
	trans_m = float(np.linalg.norm(t_rel))
	return rot_deg, trans_m


def main():
	args = parse_args()

	print("Starting Orbbec + NatNet for hand-eye calibration...")
	board_estimator = Estimate_charuco_pose()
	mocap_listener = MocapListener(
		server_ip=args.server_ip,
		client_ip=args.client_ip,
		rigid_body_id=args.rigid_body_id,
		use_multicast=args.use_multicast,
	)

	R_gripper2base = []  # rigid_body -> mocap_world
	t_gripper2base = []
	R_target2cam = []    # charuco_board -> camera
	t_target2cam = []
	last_accepted_rigid_pose = None

	method = get_method(args.method)
	t0 = time.time()

	try:
		with mocap_listener.client:
			print("Collecting paired poses. Move camera to diverse poses while board is static...")
			if args.rigid_body_id is None:
				print("Warning: --rigid-body-id not set. Using first tracked rigid body in stream.")
			print("Tip: hold each pose for a short moment instead of moving continuously.")
			while len(R_gripper2base) < args.num_samples and (time.time() - t0) < args.max_seconds:
				try:
					mocap_listener.client.update_sync()
				except (BlockingIOError, struct.error):
					# NatNet UDP can occasionally return transient socket/packet issues.
					# Ignore one bad iteration and keep collecting.
					time.sleep(0.002)
					continue

				T_b2cam, _ = board_estimator.get_camera_pose()
				if T_b2cam is None:
					continue
				

				mocap_pose = mocap_listener.get_latest()
				if mocap_pose is None:
					time.sleep(0.002)
					continue

				camera_data = board_estimator.get_camera_stream()
				if camera_data is None:
					continue
				frame, _ = camera_data
				T_b2cam = board_estimator.get_camera_pose_one_frame(frame)
				if T_b2cam is None:
					continue

				T_rigid2world = mocap_pose.T_rigid2world
				if not np.isfinite(T_rigid2world).all() or not np.isfinite(T_b2cam).all():
					continue

				if last_accepted_rigid_pose is not None:
					rot_step_deg, trans_step_m = relative_motion(last_accepted_rigid_pose, T_rigid2world)
					if rot_step_deg < args.min_rot_step_deg and trans_step_m < args.min_trans_step_m:
						continue
				print(f"target_to_cam {T_b2cam[:3, 3]}")
				if np.any(T_b2cam[:3, 3].astype(np.float64)>1.2) or np.any(T_b2cam[:3, 3].astype(np.float64)<-1.2):
					continue
				R_gripper2base.append(T_rigid2world[:3, :3].astype(np.float64))
				t_gripper2base.append(T_rigid2world[:3, 3].reshape(3, 1).astype(np.float64))
				R_target2cam.append(T_b2cam[:3, :3].astype(np.float64))
				t_target2cam.append(T_b2cam[:3, 3].reshape(3, 1).astype(np.float64))
				last_accepted_rigid_pose = T_rigid2world.copy()
				
				print(f"Collected {len(R_gripper2base)}/{args.num_samples}")

		if len(R_gripper2base) < 5:
			raise RuntimeError("Too few valid samples. Need at least 5, ideally 20+.")

		R_cam2rigid, t_cam2rigid = cv2.calibrateHandEye(
			R_gripper2base,
			t_gripper2base,
			R_target2cam,
			t_target2cam,
			method=method,
		)

		T_cam2rigid = make_T(R_cam2rigid, t_cam2rigid)
		T_rigid2cam = invert_T(T_cam2rigid)

		if not np.isfinite(T_rigid2cam).all():
			raise RuntimeError("Hand-eye produced non-finite matrix. Re-capture with better pose diversity.")

		trans_norm = float(np.linalg.norm(T_rigid2cam[:3, 3]))
		if trans_norm > args.max_output_trans_m:
			raise RuntimeError(
				f"Implausible translation magnitude ({trans_norm:.3f} m). "
				"Sampling likely unsynchronized/degenerate. Re-run with slower stops and fixed board."
			)

		np.save(args.save_path, T_rigid2cam)

		print("\n=== Calibration result ===")
		print("T_rigidbody_to_camera (mocap rigid-body frame -> camera optical frame):")
		print(T_rigid2cam)
		print(f"Saved to: {args.save_path}")

		print("\nUse it as:")
		print("T_world_to_camera = T_world_to_rigidbody @ T_rigidbody_to_camera")

	finally:
		try:
			if getattr(board_estimator, "pipeline", None) is not None:
				board_estimator.pipeline.stop()
		except Exception:
			pass


if __name__ == "__main__":
	main()

