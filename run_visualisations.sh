#!/bin/bash

set -e  # Exit on error
set -m  # Job control enabled

# === Config ===
MODEL_PATH_WITH="/auto/archive/tcga/sn666/trained_models/PATHS/ablations/enrichment_strategy/luad/residual_fusion/luad_paths_0"
# MODEL_PATH_WITH="/home/sn666/dissertation/benchmarking/PATHS/backup/kirp_paths_0"
MODEL_PATH_WITHOUT="/home/sn666/dissertation/benchmarking/PATHS/models/luad_paths_0"
CONFIG_PATH="/home/sn666/dissertation/config/train_config/stnet_pancancer_image_grid.yaml"
TRANSCRIPTOMICS_CHECKPOINT="/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/stnet_pancancer_highest_mag/epoch=5-step=17874.ckpt"
TRANSCRIPTOMICS_CHECKPOINT_VISUALISATION="/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/stnet_pancancer_highest_mag/epoch=5-step=17874.ckpt"
WSI_DIR="/auto/archive/tcga/tcga/wsi/luad"
CHOSEN_GENE_LIST="/home/sn666/dissertation/config/visualisation_genes/pancancer.txt"
TEST_LIST="/home/sn666/dissertation/benchmarking/PATHS/models/luad_paths_0/test_slide_ids.txt"
OUT_PATH="/auto/archive/tcga/sn666/visualisations/luad_9June"
GENES_TO_VISUALISE="ABI1,ABL2,ACKR3,ACSL3"

MAX_JOBS=3

# === Handle Ctrl+C ===
cleanup() {
    echo "Caught SIGINT. Killing all subprocesses..."
    pkill -P $$ || true  # kill all child processes
    exit 1
}
trap cleanup SIGINT

# === Slide processing function ===
process_slide() {
    base_filename="$1"
    SVS_FILE="$WSI_DIR/${base_filename}.svs"

    echo "[$base_filename] WITH transcriptomics..."
    python heatmap_visualise_transcriptomics.py \
        -m "$MODEL_PATH_WITH" \
        -s "$SVS_FILE" \
        -c "$CONFIG_PATH" \
        -g "$CHOSEN_GENE_LIST" \
        -o "$OUT_PATH" \
        -t "$TRANSCRIPTOMICS_CHECKPOINT" \
        -u "$TRANSCRIPTOMICS_CHECKPOINT_VISUALISATION" \
        -v "$GENES_TO_VISUALISE"

    echo "[$base_filename] WITHOUT transcriptomics..."
    python heatmap_visualise_transcriptomics.py \
        -m "$MODEL_PATH_WITHOUT" \
        -s "$SVS_FILE" \
        -c "$CONFIG_PATH" \
        -g "$CHOSEN_GENE_LIST" \
        -t "$TRANSCRIPTOMICS_CHECKPOINT" \
        -u "$TRANSCRIPTOMICS_CHECKPOINT_VISUALISATION" \
        -o "$OUT_PATH" \
        -v "$GENES_TO_VISUALISE"
}

export -f process_slide
export MODEL_PATH_WITH MODEL_PATH_WITHOUT CONFIG_PATH TRANSCRIPTOMICS_CHECKPOINT TRANSCRIPTOMICS_CHECKPOINT_VISUALISATION WSI_DIR CHOSEN_GENE_LIST OUT_PATH GENES_TO_VISUALISE

# === Parallel execution with Ctrl+C safety ===
cat "$TEST_LIST" | xargs -n 1 -P "$MAX_JOBS" -I {} bash -c 'process_slide "$@"' _ {}