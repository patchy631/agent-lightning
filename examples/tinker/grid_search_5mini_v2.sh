# Search for the best hyper-parameters
# set -e
# set -x

# Generate all combinations and shuffle them
combinations=$(for lr in 1e-5 3e-5 1e-4; do
  for gs in 16; do
    for loss in ppo importance_sampling; do
      for seed in 0 17 42; do
        for rank in 8 16; do
          for bs in 8 16 32; do
            echo "$lr $gs $loss $seed $rank $bs"
          done
        done
      done
    done
  done
done | shuf)

# Run each combination
while read -r lr gs loss seed rank bs; do
  echo "Running with LR=$lr, GS=$gs, LOSS_FN=$loss, SEED=$seed, RANK=$rank BS=$bs"
  LEARNING_RATE=$lr GROUP_SIZE=$gs LOSS_FN=$loss SEED=$seed RANK=$rank BATCH_SIZE=$bs \
  uv run --no-sync dotenv run python q20_train_grid_5mini_v2.py --port 4755
done <<< "$combinations"
