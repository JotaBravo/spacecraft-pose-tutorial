#!/bin/bash














#popular-finch-300 invincible-bug-944 debonair-hog-651 whimsical-donkey-574 ambitious-sheep-254 gregarious-cow-970 puzzled-pig-225 rumbling-bear-361 bustling-bat-711 nebulous-grub-837 ambitious-mare-46 beautiful-shrimp-650 



declare -a run_names=(
unruly-bat-706 nebulous-robin-61 dashing-bear-620)


# Function to execute python script for a given run ID
execute_python_script() {
    local run_id="$1"
    python test.py --run "$run_id"
}

# Iterate through each hardcoded RunName
for run_name in "${run_names[@]}"; do
    # Search for the corresponding run ID in the mlruns folder
    run_ids=$(find mlruns -name 'meta.yaml' -exec grep -l "$run_name" {} + | xargs grep -Eo '[0-9a-f]{32}' | sort -u)
    if [ -z "$run_ids" ]; then
        echo "Error: RunName '$run_name' not found in mlruns folder."
    else
        for run_id in $run_ids; do
            echo "Executing python script for RunName: $run_name (Run ID: $run_id)"
            execute_python_script "$run_id"
        done
    fi
done
In this vers