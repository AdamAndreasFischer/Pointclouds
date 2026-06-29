#!/usr/bin/env bash

set -u


ROOT_DIR="/home/adamfi/codes"
POSE_LISTENER="$ROOT_DIR/Pointclouds/src/natnet_pose_listener_multi_cam.py"
CAPTURE_PCD="$ROOT_DIR/Pointclouds/src/capture_point_cloud_multi_cam.py"
TRANSFORM_POSES="$ROOT_DIR/Pointclouds/src/Multi_stage_icp_multi_cam.py"
CAMERA_DATA="$ROOT_DIR/Pointclouds/utils/camera_data_robot_lab.yml"
CALIBRATION_DIR="$ROOT_DIR/Pointclouds/Orbbec_calibrations_mocaplab"

CLOUD_FOLDER="$ROOT_DIR/Pointclouds/pointclouds/vis-seq-robotlab-17-step2"

NATNET_IP="192.168.125.86"
CLIENT_IP="192.168.125.57"
#NATNET_IP="192.168.12.183"
#CLIENT_IP="192.168.12.201"
CAMERA_IDS="1,2,3"
CAMERA_IDS_EXPLICIT=false
RIGID_BODY_IDS=""
SERIALS=""
DEVICE_INDEXES=""
NUM_CLOUDS=5
MIN_POINTS=600000
POSE_TIMEOUT_SEC=0
USE_MULTICAST=false
RUN_UFO=true
SCP_IMAGES=true
SKIP_PREFLIGHT=false
PREVIEW_CAPTURE=false
PYTHON_CMD="${PYTHON_CMD:-python3}"


usage() {
	echo "Usage: $0 [options]"
	echo
	echo "Options:"
	echo "  --root_dir <path>             Output dataset directory"
	echo "  --camera-data <path>          camera_data.yml path"
	echo "  --calibration-dir <path>      Directory with orbbec{N}.npy calibration files"
	echo "  --camera-ids <ids>            Comma-separated camera IDs (default: auto-detected)"
	echo "  --rb-ids <map>                Rigid body overrides, e.g. 1:4,2:5,3:6,4:7"
	echo "  --serials <map>               Serial overrides, e.g. 1:SERIAL_A,2:SERIAL_B"
	echo "  --device-indexes <map>        0-based Orbbec device indexes, e.g. 1:0,2:1,3:2,4:3"
	echo "  --num-pcds <n>                Point clouds per camera pose (default: 5)"
	echo "  --min-points <n>              Minimum accepted point count per cloud"
	echo "  --server-ip <ip>              NatNet/Motive server IP"
	echo "  --client-ip <ip>              Local NatNet client IP"

	echo "  --timeout-sec <seconds>       NatNet pose timeout, 0 waits forever"
	echo "  --use-multicast               Enable NatNet multicast mode"
	echo "  --preview                     Preview captured clouds in Open3D"
	echo "  --no-ufo                      Do not run UFO map after transformation"
	echo "  --skip-preflight              Skip camera_data validation before capture"
	echo "  -h, --help                    Show this help"
}


while [[ $# -gt 0 ]]; do
	case "$1" in
		--root_dir|--root-dir)
			CLOUD_FOLDER="${2:-}"
			shift 2
			;;
		--camera-data)
			CAMERA_DATA="${2:-}"
			shift 2
			;;
		--calibration-dir)
			CALIBRATION_DIR="${2:-}"
			shift 2
			;;
		--camera-ids)
			CAMERA_IDS="${2:-}"
			CAMERA_IDS_EXPLICIT=true
			shift 2
			;;
		--rb-ids|--rigid-body-ids)
			RIGID_BODY_IDS="${2:-}"
			shift 2
			;;
		--serials)
			SERIALS="${2:-}"
			shift 2
			;;
		--device-indexes)
			DEVICE_INDEXES="${2:-}"
			shift 2
			;;
		--num-pcds|--num_pcds)
			NUM_CLOUDS="${2:-}"
			shift 2
			;;
		--min-points)
			MIN_POINTS="${2:-}"
			shift 2
			;;
		--server-ip)
			NATNET_IP="${2:-}"
			shift 2
			;;
		--client-ip)
			CLIENT_IP="${2:-}"
			shift 2
			;;
		--timeout-sec)
			POSE_TIMEOUT_SEC="${2:-}"
			shift 2
			;;
		--use-multicast)
			USE_MULTICAST=true
			shift
			;;
		--preview)
			PREVIEW_CAPTURE=true
			shift
			;;
		--no-ufo)
			RUN_UFO=false
			shift
			;;
		--skip-preflight)
			SKIP_PREFLIGHT=true
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			usage
			exit 1
			;;
	esac
done

if [[ -z "$CLOUD_FOLDER" ]]; then
	echo "--root_dir must be non-empty."
	exit 1
fi

echo "NatNet server IP: $NATNET_IP"
echo "NatNet client IP: $CLIENT_IP"
echo "Dataset folder: $CLOUD_FOLDER"


run_python_script() {
	local script_path="$1"
	local step_name="$2"
	shift 2

	if [[ ! -f "$script_path" ]]; then
		echo "[ERROR] Could not find $step_name script: $script_path"
		return 1
	fi

	echo
	echo "========== $step_name =========="
	"$PYTHON_CMD" "$script_path" "$@"
	local exit_code=$?

	if [[ $exit_code -ne 0 ]]; then
		echo "[ERROR] $step_name failed with exit code $exit_code"
		return $exit_code
	fi

	echo "[OK] $step_name finished"
	return 0
}


next_capture_index() {
	local max_index=0
	local base
	local index
	shopt -s nullglob
	for dir in "$CLOUD_FOLDER"/Cloud_pose*_camera_*; do
		base="$(basename "$dir")"
		if [[ "$base" =~ ^Cloud_pose0*([0-9]+)_camera_[0-9]+$ ]]; then
			index="${BASH_REMATCH[1]}"
			if (( 10#$index > max_index )); then
				max_index=$((10#$index))
			fi
		fi
	done
	shopt -u nullglob
	echo $((max_index + 1))
}


ALL_CAMERA_IDS=""

# Detect connected Orbbec cameras via get_serial.py and match against camera_data.yml.
# Sets CAMERA_IDS, SERIALS, DEVICE_INDEXES and accumulates into ALL_CAMERA_IDS.
detect_cameras() {
	local get_serial="$ROOT_DIR/Pointclouds/utils/get_serial.py"
	echo "Detecting connected cameras..."
	local serial_output
	serial_output=$("$PYTHON_CMD" "$get_serial" 2>/dev/null) || {
		echo "[WARN] Camera detection failed, using CAMERA_IDS=$CAMERA_IDS"
		return
	}
	if [[ -z "$serial_output" ]]; then
		echo "[WARN] No cameras detected, using CAMERA_IDS=$CAMERA_IDS"
		return
	fi

	local detected_ids=()
	local serials_map=()
	local indexes_map=()

	while IFS= read -r line; do
		if [[ "$line" =~ ^Device\ ([0-9]+):\ serial\ =\ (.+)$ ]]; then
			local dev_idx="${BASH_REMATCH[1]}"
			local serial="${BASH_REMATCH[2]//[[:space:]]/}"
			local cam_id
			cam_id=$(awk -v serial="$serial" '
				/^#/ { next }
				/^camera_[0-9]+:/ { match($0, /[0-9]+/); cam = substr($0, RSTART, RLENGTH) }
				/serial/ && index($0, serial) { print cam; exit }
			' "$CAMERA_DATA")
			if [[ -n "$cam_id" ]]; then
				detected_ids+=("$cam_id")
				serials_map+=("${cam_id}:${serial}")
				indexes_map+=("${cam_id}:${dev_idx}")
				echo "  Camera $cam_id: serial=$serial, device_index=$dev_idx"
			else
				echo "  Unknown serial $serial (device_index=$dev_idx) — not in camera_data.yml"
			fi
		fi
	done <<< "$serial_output"

	if [[ ${#detected_ids[@]} -eq 0 ]]; then
		echo "[WARN] No known cameras detected, keeping CAMERA_IDS=$CAMERA_IDS"
		return
	fi

	CAMERA_IDS=$(IFS=','; echo "${detected_ids[*]}")
	SERIALS=$(IFS=','; echo "${serials_map[*]}")
	DEVICE_INDEXES=$(IFS=','; echo "${indexes_map[*]}")

	if [[ -z "$ALL_CAMERA_IDS" ]]; then
		ALL_CAMERA_IDS="$CAMERA_IDS"
	else
		ALL_CAMERA_IDS="${ALL_CAMERA_IDS},${CAMERA_IDS}"
	fi
}


build_args() {
	COMMON_ARGS=(--root_dir "$CLOUD_FOLDER" --camera-data "$CAMERA_DATA" --camera-ids "$CAMERA_IDS")

	POSE_ARGS=("${COMMON_ARGS[@]}" --server-ip "$NATNET_IP" --client-ip "$CLIENT_IP" --timeout-sec "$POSE_TIMEOUT_SEC")
	if [[ -n "$RIGID_BODY_IDS" ]]; then
		POSE_ARGS+=(--rigid-body-ids "$RIGID_BODY_IDS")
	fi
	if [[ "$USE_MULTICAST" == true ]]; then
		POSE_ARGS+=(--use-multicast)
	fi

	CAPTURE_ARGS=("${COMMON_ARGS[@]}" --num_pcds "$NUM_CLOUDS" --min-points "$MIN_POINTS")
	if [[ -n "$SERIALS" ]]; then
		CAPTURE_ARGS+=(--serials "$SERIALS")
	fi
	if [[ -n "$DEVICE_INDEXES" ]]; then
		CAPTURE_ARGS+=(--device-indexes "$DEVICE_INDEXES")
	fi
	if [[ "$PREVIEW_CAPTURE" == true ]]; then
		CAPTURE_ARGS+=(--preview)
	fi

	TRANSFORM_ARGS=("${COMMON_ARGS[@]}" --num-pcds "$NUM_CLOUDS" --calibration-dir "$CALIBRATION_DIR")
}


run_capture_loop() {
	while true; do
		echo
		echo "#############################################"
		echo "Capture $cycle"
		echo "#############################################"

		run_python_script "$POSE_LISTENER" "NatNet multi-camera pose capture" "${POSE_ARGS[@]}" --capture-index "$cycle" || {
			read -r -p "Pose capture failed. Continue anyway? [y/N]: " continue_anyway
			if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
				echo "Stopping."
				exit 1
			fi
		}

		read -r -p "Press Enter to run multi-camera point-cloud capture (or type q to quit): " start_capture
		if [[ "$start_capture" =~ ^[Qq]$ ]]; then
			echo "Exiting capture loop."
			break
		fi

		if run_python_script "$CAPTURE_PCD" "Multi-camera point-cloud capture" "${CAPTURE_ARGS[@]}" --capture-index "$cycle"; then
			captured_any=true
		else
			read -r -p "Capture failed. Continue to next capture? [y/N]: " continue_next
			if [[ ! "$continue_next" =~ ^[Yy]$ ]]; then
				echo "Stopping."
				exit 1
			fi
		fi

		read -r -p "Capture another multi-camera pose + cloud set? [Y/n]: " do_next
		if [[ "$do_next" =~ ^[Nn]$ ]]; then
			echo "Done capturing."
			break
		fi

		cycle=$((cycle + 1))
	done
}


if [[ "$CAMERA_IDS_EXPLICIT" != true ]]; then
	detect_cameras
fi
# If detection was skipped or found nothing, seed ALL_CAMERA_IDS from the current value
[[ -z "$ALL_CAMERA_IDS" ]] && ALL_CAMERA_IDS="$CAMERA_IDS"
echo "Camera IDs: $CAMERA_IDS"

build_args

if [[ "$SKIP_PREFLIGHT" != true ]]; then
	run_python_script "$POSE_LISTENER" "NatNet multi-camera config validation" "${POSE_ARGS[@]}" --validate-config || exit 1
	run_python_script "$CAPTURE_PCD" "Point-cloud multi-camera config validation" "${CAPTURE_ARGS[@]}" --validate-config || exit 1
	run_python_script "$TRANSFORM_POSES" "Calibration transform config validation" "${TRANSFORM_ARGS[@]}" --validate-config || exit 1
fi

if [[ -d "$CLOUD_FOLDER" ]]; then
	echo "$CLOUD_FOLDER already exists."
	read -r -p "Directory already exists. Continue? [y/N]: " continue_existing
	if [[ "$continue_existing" =~ ^[Nn]$|^$ ]]; then
		echo "Stopping."
		exit 0
	fi
else
	echo "$CLOUD_FOLDER created"
fi

mkdir -p "$CLOUD_FOLDER"

echo
echo "Interactive multi-camera pose + point-cloud capture"
echo "- NatNet stores one pose for each requested camera"
echo "- Point-cloud capture stores $NUM_CLOUDS clouds for each requested camera"
echo "- Calibration transforms are written directly; ICP is skipped"

cycle="$(next_capture_index)"
captured_any=false
transform_failed=false

run_capture_loop

# Offer to capture for additional cameras (e.g. after swapping hardware)
while true; do
	echo
	read -r -p "Capture poses and point clouds for additional cameras? [y/N]: " add_more
	[[ ! "$add_more" =~ ^[Yy]$ ]] && break
	echo "Connect the new cameras, then press Enter to detect..."
	read -r
	detect_cameras
	echo "Camera IDs: $CAMERA_IDS"
	build_args
	cycle=$((cycle + 1))
	run_capture_loop
done

# Final transform uses all camera IDs captured across all detection rounds
TRANSFORM_ARGS=(--root_dir "$CLOUD_FOLDER" --camera-data "$CAMERA_DATA" --camera-ids "$ALL_CAMERA_IDS" --num-pcds "$NUM_CLOUDS" --calibration-dir "$CALIBRATION_DIR")

if [[ "$captured_any" == true ]]; then
	run_python_script "$TRANSFORM_POSES" "Calibration-only multi-camera transform" "${TRANSFORM_ARGS[@]}" || {
		echo "Calibration transform failed."
		transform_failed=true
	}
else
	echo "No successful captures found, skipping calibration transform and UFO map."
fi

if [[ "$captured_any" == true && "$transform_failed" != true && "$RUN_UFO" == true ]]; then
	echo
	echo "Creating UFO map"

	if [[ -d "$CLOUD_FOLDER/clouds" ]]; then
		find "$CLOUD_FOLDER/clouds" -maxdepth 1 -type f -name 'cloud_*.pcd' -delete
	fi

	pushd "$ROOT_DIR/ufo" >/dev/null || exit 1
	./lib/map/apps/manipulation/ply_to_pcd.bash "$CLOUD_FOLDER"
	sed -i "s|^dataset_path *=.*|dataset_path = \"$CLOUD_FOLDER\"|" ./lib/map/apps/manipulation/config.toml
	./build/lib/map/apps/manipulation/UFOManipulation ./lib/map/apps/manipulation/config.toml
	popd >/dev/null || exit 1
fi

if [[ "$captured_any" == true && "$transform_failed" != true && "$RUN_UFO" == true && "$SCP_IMAGES" == true ]]; then
	echo
	echo "Copying images to dataset folder and sending to workstation"
	cp /home/adamfi/codes/UFO_TEST/rgb.png "$CLOUD_FOLDER"
	cp /home/adamfi/codes/UFO_TEST/depth.png "$CLOUD_FOLDER"
	#scp /home/adamfi/codes/UFO_TEST/rgb.png adamfi@192.168.125.160:/home/adamfi/codes/visual-mm-planning/data/sim-to-real-imgs/rgb.png
	#scp /home/adamfi/codes/UFO_TEST/depth.png adamfi@192.168.125.160:/home/adamfi/codes/visual-mm-planning/data/sim-to-real-imgs/depth.png
	scp /home/adamfi/codes/UFO_TEST/rgb.png adamfi@192.168.125.160:/home/adamfi/codes/base-pose-sequencing/data/sim-to-real-imgs/rgb.png
	scp /home/adamfi/codes/UFO_TEST/depth.png adamfi@192.168.125.160:/home/adamfi/codes/base-pose-sequencing/data/sim-to-real-imgs/depth.png
fi

if [[ "$transform_failed" == true ]]; then
	exit 1
fi
