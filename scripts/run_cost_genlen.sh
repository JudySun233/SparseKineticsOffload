SPARSE_ARG=${SPARSE_ARG:-dense}

TASK=${TASK:-results/aime24/offload}
python3 kinetics_cost_models/long_CoT/cost_model_offload_genlen.py "$TASK" "$SPARSE_ARG"

# TASK=${TASK:-results/aime24/no-offload}
# python3 kinetics_cost_models/long_CoT/cost_model_genlen.py "$TASK" "$SPARSE_ARG"
