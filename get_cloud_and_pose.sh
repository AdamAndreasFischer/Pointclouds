#!/usr/bin/env bash

set -u

ROOT_DIR="/home/adamfi/codes"
POSE_LISTENER="$ROOT_DIR/Pointclouds/src/ros_pose_listener.py"
CAPTURE_PCD="$ROOT_DIR/Pointclouds/src/capture_point_cloud.py"
MULTI_STAGE_ICP="$ROOT_DIR/Pointclouds/src/Multi_stage_icp.py"

mkdir -p "$ROOT_DIR/cloud_folder"
CLOUD_FOLDER="$ROOT_DIR/cloud_folder"

NUM_CLOUDS=5
icp_failed=false

PYTHON_CMD="python3"

run_python_script() {
	local script_path="$1"
	local step_name="$2"

	if [[ ! -f "$script_path" ]]; then
		echo "[ERROR] Could not find $step_name script: $script_path"
		return 1
	fi
	echo
	echo "========== $step_name =========="
	if [[ "$script_path" == "$CAPTURE_PCD" ]]; then
		"$PYTHON_CMD" "$script_path" --root_dir "$CLOUD_FOLDER" --num_pcds "$NUM_CLOUDS"
	else
	    "$PYTHON_CMD" "$script_path" --root_dir "$CLOUD_FOLDER"
	fi
	local exit_code=$?

	if [[ $exit_code -ne 0 ]]; then
		echo "[ERROR] $step_name failed with exit code $exit_code"
		return $exit_code
	fi

	echo "[OK] $step_name finished"
	return 0
}

echo "Interactive pose + point-cloud capture"
echo "- First, pose listener runs and stores one pose"
echo "- When you press Enter, point-cloud capture starts"
echo "- After each capture, you can optionally run Multi-stage ICP"

cycle=1
captured_any=false
while true; do
	echo
	echo "#############################################"
	echo "Cycle $cycle"
	echo "#############################################"

	run_python_script "$POSE_LISTENER" "Pose listener" || {
		read -r -p "Pose listener failed. Continue anyway? [y/N]: " continue_anyway
		if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
			echo "Stopping."
			exit 1
		fi
	}

	read -r -p "Press Enter to run point-cloud capture (or type q to quit): " start_capture
	if [[ "$start_capture" =~ ^[Qq]$ ]]; then
		echo "Exiting."
		break
	fi

	if run_python_script "$CAPTURE_PCD" "Point-cloud capture"; then
		captured_any=true
	else
		read -r -p "Capture failed. Continue to next cycle? [y/N]: " continue_next
		if [[ ! "$continue_next" =~ ^[Yy]$ ]]; then
			echo "Stopping."
			exit 1
		fi
	fi

	read -r -p "Capture another pose + cloud? [Y/n]: " do_next
	if [[ "$do_next" =~ ^[Nn]$ ]]; then
		echo "Done."
		break
	fi

	cycle=$((cycle + 1))
done

if [[ "$captured_any" == true ]]; then
	read -r -p "Run Multi_stage_icp.py on all captured poses/clouds now? [y/N]: " run_icp
	if [[ "$run_icp" =~ ^[Yy]$ ]]; then
		run_python_script "$MULTI_STAGE_ICP" "Multi-stage ICP" || {
			echo "ICP failed."
			icp_failed=true
			exit 1
		}
	fi
else
	echo "No successful captures found, skipping registration prompt."
fi

if [[ "$icp_failed" != "true" ]]; then
	echo "Creating UFO map"
	echo
	echo "Transforming poses"
	python3 "$ROOT_DIR/ufo/lib/map/apps/manipulation/numpy_to_tsv.py" "$CLOUD_FOLDER" "$NUM_CLOUDS" True True

	pushd "$ROOT_DIR/ufo" >/dev/null || exit 1
	./lib/map/apps/manipulation/ply_to_pcd.bash "$CLOUD_FOLDER"

	sed -i "s|^dataset_path *=.*|dataset_path = \"$CLOUD_FOLDER\"|" ./lib/map/apps/manipulation/config.toml
	./build/lib/map/apps/manipulation/UFOManipulation ./lib/map/apps/manipulation/config.toml
	popd >/dev/null || exit 1
fi