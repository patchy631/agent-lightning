# Search for the best hyper-parameters
set -e
set -x

# Generate all combinations and shuffle them
combinations=$(for lr in 3e-5 1e-4 3e-4; do
  for gs in 8 16; do
    for loss in ppo importance_sampling; do
      for seed in 0 17 42; do
        echo "$lr $gs $loss $seed"
      done
    done
  done
done | shuf)

# Run each combination
while read -r lr gs loss seed; do
  echo "Running with LR=$lr, GS=$gs, LOSS_FN=$loss, SEED=$seed"
  LEARNING_RATE=$lr GROUP_SIZE=$gs LOSS_FN=$loss SEED=$seed \
  uv run --no-sync dotenv run python q20_train_grid.py
done <<< "$combinations"
