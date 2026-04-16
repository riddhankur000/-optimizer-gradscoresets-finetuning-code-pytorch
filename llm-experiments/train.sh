#!/bin/bash
# Sequential Training Wrapper Script
# Automatically sets up and runs sequential training using config.yaml

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# Print header
print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║    Sequential Multi-Task Training with config.yaml        ║"
    echo "║          Support: AdamW & Muon, Single/Multi GPU          ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
print_usage() {
    echo -e "${YELLOW}Usage: bash train.sh [OPTIONS]${NC}"
    echo ""
    echo "OPTIONS:"
    echo "  --config PATH       Path to config.yaml (default: ./config.yaml)"
    echo "  --optimizer OPT     Override optimizer: adamw or muon"
    echo "  --gpu PROFILE       Override GPU profile: gpu_0, gpu_1, gpu_multi, or cpu"
    echo "  --help              Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  # Default: use settings from config.yaml"
    echo "  bash train.sh"
    echo ""
    echo "  # Override to use Muon optimizer"
    echo "  bash train.sh --optimizer muon"
    echo ""
    echo "  # Use multi-GPU training"
    echo "  bash train.sh --gpu gpu_multi"
    echo ""
    echo "  # Custom config file and optimizer"
    echo "  bash train.sh --config ./my_config.yaml --optimizer adamw"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --optimizer)
            OVERRIDE_OPTIMIZER="$2"
            shift 2
            ;;
        --gpu)
            OVERRIDE_GPU="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Print header
print_header

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Error: Config file not found: $CONFIG_FILE${NC}"
    echo ""
    echo -e "Please provide a valid config.yaml file."
    echo -e "Copy config.yaml.example to config.yaml and customize it."
    exit 1
fi

echo -e "${GREEN}✓ Config file found: $CONFIG_FILE${NC}"
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Error: Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python --version)${NC}"
echo ""

# Extract active profiles from config
echo -e "${BLUE}Parsing configuration...${NC}"

# Extract optimizer and gpu from config
OPTIMIZER=$(grep -A 2 "active_profiles:" "$CONFIG_FILE" | grep "optimizer:" | awk '{print $2}' | tr -d '"')
GPU=$(grep -A 2 "active_profiles:" "$CONFIG_FILE" | grep "gpu:" | awk '{print $2}' | tr -d '"')

# Override if specified
if [ -n "$OVERRIDE_OPTIMIZER" ]; then
    OPTIMIZER="$OVERRIDE_OPTIMIZER"
    echo -e "${YELLOW}! Overriding optimizer: $OPTIMIZER${NC}"
fi

if [ -n "$OVERRIDE_GPU" ]; then
    GPU="$OVERRIDE_GPU"
    echo -e "${YELLOW}! Overriding GPU profile: $GPU${NC}"
fi

# Default values if not found
OPTIMIZER=${OPTIMIZER:-"adamw"}
GPU=${GPU:-"gpu_0"}

echo -e "${GREEN}✓ Active Optimizer: $OPTIMIZER${NC}"
echo -e "${GREEN}✓ Active GPU Profile: $GPU${NC}"
echo ""

# Check for dataset
DATASET_PATH=$(grep -A 5 "dataset_config:" "$CONFIG_FILE" | grep "dataset_path:" | awk -F': ' '{print $2}' | tr -d '"')
if [ -z "$DATASET_PATH" ]; then
    echo -e "${RED}✗ Error: dataset_path not found in config${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset path: $DATASET_PATH${NC}"

if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${YELLOW}⚠ Warning: Dataset folder not found at $DATASET_PATH${NC}"
    echo -e "Make sure your dataset is prepared before training starts."
fi

echo ""

# Extract device_ids from GPU profile in config
echo -e "${BLUE}Extracting GPU device configuration...${NC}"

# Use sed and grep to extract device_ids for the specified GPU profile
# Format: under gpu_profiles, find the specific profile section and get device_ids
DEVICE_IDS=$(sed -n "/gpu_profiles:/,/## /p" "$CONFIG_FILE" | sed -n "/$GPU:/,/^  [^ ]/p" | grep "device_ids:" | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'")

if [ -z "$DEVICE_IDS" ] || [ "$DEVICE_IDS" = "None" ]; then
    if [ "$GPU" != "cpu" ]; then
        echo -e "${YELLOW}⚠ Warning: No device_ids found for GPU profile '$GPU'${NC}"
    fi
    # For CPU training, no CUDA_VISIBLE_DEVICES needed
    if [ "$GPU" = "cpu" ]; then
        export CUDA_VISIBLE_DEVICES=""
        echo -e "${GREEN}✓ CPU mode selected${NC}"
    fi
else
    # Set CUDA_VISIBLE_DEVICES to the specified device IDs
    export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"
    echo -e "${GREEN}✓ GPU Device IDs: $DEVICE_IDS${NC}"
    echo -e "${GREEN}✓ CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES${NC}"
fi

# Print training info
echo -e "${BLUE}═════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Training Configuration:${NC}"
echo -e "${BLUE}═════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Config File:      ${GREEN}$CONFIG_FILE${NC}"
echo -e "  Optimizer:        ${GREEN}$OPTIMIZER${NC}"
echo -e "  GPU Profile:      ${GREEN}$GPU${NC}"
echo -e "  Dataset:          ${GREEN}$DATASET_PATH${NC}"
echo ""
echo -e "${BLUE}═════════════════════════════════════════════════════════════${NC}"
echo ""

# Run training
echo -e "${BLUE}Starting training...${NC}"
echo ""

cd "$SCRIPT_DIR"

# Set environment variables to prevent bitsandbytes compatibility issues
# (not needed for regular LoRA, only for 8-bit quantization)
export BITSANDBYTES_NOWELCOME=1
export CUDA_LAUNCH_BLOCKING=1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nemo_muon_env

python -u colm/train/train_sequential_from_config.py "$CONFIG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo -e "${BLUE}═════════════════════════════════════════════════════════════${NC}"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo ""
    OUTPUT_DIR=$(grep -A 20 "training_config:" "$CONFIG_FILE" | grep "output_dir:" | awk -F': ' '{print $2}' | tr -d '"')
    if [ -n "$OUTPUT_DIR" ]; then
        echo -e "Results saved to: ${GREEN}$OUTPUT_DIR${NC}"
    fi
else
    echo -e "${RED}✗ Training failed with exit code $TRAIN_EXIT_CODE${NC}"
fi

echo -e "${BLUE}═════════════════════════════════════════════════════════════${NC}"
echo ""

exit $TRAIN_EXIT_CODE
