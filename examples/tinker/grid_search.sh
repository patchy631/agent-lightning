# Search for the best hyper-parameters

for lr in 3e-5 1e-4 3e-4; do
    for gs in 8 16; do
        for loss in ppo importance_sampling; do
            for seed in 42 43; do
                LEARNING_RATE=$lr GROUP_SIZE=$gs LOSS_FN=$loss SEED=$seed uv run --no-sync dotenv run python q20_train_grid.py
            done
        done
    done
done
