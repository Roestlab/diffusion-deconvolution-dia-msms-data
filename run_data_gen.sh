#!/bin/bash

module load gcc arrow/16.1.0

source py310/bin/activate

# Define ranges for outer and inner loops
num_runs=(1 2 3 4 5 6)
isolation_windows=({0..53})

# Outer loop over num_runs
for num in ${num_runs[@]}; do
    # Inner loop over isolation_windows
    for iso_win_idx in ${isolation_windows[@]}; do

        # Prepare a job script for each python task
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dq_datagen_${num}_${iso_win_idx}
#SBATCH --account=def-hroest
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=800G
#SBATCH --output=dquartic_${num}_${iso_win_idx}_%j.log
#SBATCH --error=dquartic_${num}_${iso_win_idx}_%j.err

module load gcc arrow/16.1.0
source py310/bin/activate

dquartic generate-train-data \
    --isolation_window_index=$iso_win_idx \
    --window-size=340 \
    --sliding-step=20 \
    --ms1-fixed-mz-size=50 \
    --ms2-fixed-mz-size=30000 \
    --batch-size=100 \
    --num-chunks=1 \
    --threads=1 \
    "data/searle_2018/dia_narrow/23aug2017_hela_serum_timecourse_4mz_narrow_$num.sqMass" \
    "data/searle_2018/dia_narrow/ms_data_slices_narrow_${num}_isowin_${iso_win_idx}.parquet"

EOF

        # Optional: Print a message for tracking
        echo "Submitted job for num=$num and iso_win_idx=$iso_win_idx"

    done
done
