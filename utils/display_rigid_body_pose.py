"""Live display of a Motive rigid body pose via NatNet."""

import argparse
import struct
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from natnet import DataFrame, NatNetClient
from natnet.data_frame import RigidBody


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live-display the pose of a Motive rigid body in the terminal."
    )
    parser.add_argument(
        "--rigid-body-id",
        type=int,
        default=26,
        help="Rigid body ID to track (default: 5).",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="192.168.125.86",
        help="NatNet/Motive server IP.",
    )
    parser.add_argument(
        "--client-ip",
        type=str,
        default="192.168.125.57",
        help="Local client IP.",
    )
    parser.add_argument(
        "--use-multicast",
        action="store_false",
        help="Enable multicast mode (must match Motive streaming setting).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=30.0,
        help="Display refresh rate in Hz (default: 30).",
    )
    return parser.parse_args()


def _extract_rb_id(rb: RigidBody):
    for key in ("id_num", "id", "id_", "rigid_body_id"):
        if hasattr(rb, key):
            return int(getattr(rb, key))
    return None


class PoseDisplay:
    def __init__(self, target_id: int):
        self.target_id = target_id
        self.latest_pose: np.ndarray | None = None
        self.frame_count = 0
        self.last_display = 0.0

    def callback(self, msg):
        if not isinstance(msg, DataFrame):
            return

        rigid_bodies = getattr(msg, "rigid_bodies", None)
        if not rigid_bodies:
            return

        for rb in rigid_bodies:
            if _extract_rb_id(rb) != self.target_id:
                continue

            x, y, z = rb.pos
            qx, qy, qz, qw = rb.rot
            self.latest_pose = np.array([x, y, z, qx, qy, qz, qw], dtype=np.float64)
            self.frame_count += 1
            break

    def render(self, interval: float):
        now = time.monotonic()
        if now - self.last_display < interval:
            return
        self.last_display = now

        print("\033[2J\033[H", end="")  # clear screen, move cursor home

        print(f"=== Motive rigid body — ID {self.target_id} ===\n")

        if self.latest_pose is None:
            print("  Waiting for data ...")
            return

        x, y, z, qx, qy, qz, qw = self.latest_pose

        r = Rotation.from_quat([qx, qy, qz, qw])
        roll_deg, pitch_deg, yaw_deg = r.as_euler("xyz", degrees=True)

        print(f"  Position (m)")
        print(f"    x = {x:+.4f}")
        print(f"    y = {y:+.4f}")
        print(f"    z = {z:+.4f}")
        print()
        print(f"  Orientation — quaternion")
        print(f"    qx = {qx:+.4f}")
        print(f"    qy = {qy:+.4f}")
        print(f"    qz = {qz:+.4f}")
        print(f"    qw = {qw:+.4f}")
        print()
        print(f"  Orientation — Euler XYZ (deg)")
        print(f"    roll  = {roll_deg:+8.3f}")
        print(f"    pitch = {pitch_deg:+8.3f}")
        print(f"    yaw   = {yaw_deg:+8.3f}")
        print()
        print(f"  Frames received: {self.frame_count}")
        print()
        print("  Press Ctrl-C to quit.")


def main():
    args = parse_args()
    interval = 1.0 / max(args.rate_hz, 0.1)

    display = PoseDisplay(target_id=args.rigid_body_id)

    print(f"Connecting to {args.server_ip} …")
    client = NatNetClient(
        server_ip_address=args.server_ip,
        local_ip_address=args.client_ip,
        use_multicast=args.use_multicast,
    )
    client.on_data_frame_received_event.handlers.append(display.callback)

    stop_event = threading.Event()

    def poll_loop():
        with client:
            while not stop_event.is_set():
                try:
                    client.update_sync()
                except (BlockingIOError, struct.error):
                    time.sleep(0.001)

    thread = threading.Thread(target=poll_loop, daemon=True)
    thread.start()

    print(f"Tracking rigid body ID {args.rigid_body_id}. Ctrl-C to stop.")
    try:
        while True:
            display.render(interval)
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
