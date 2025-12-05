from edgesim_env import EdgeSimEnv


def main():
    # Same dataset you used in scenario_1cloud3edge.py
    dataset_path = "datasets/sample_dataset.json"

    # One episode = 600 steps (seconds)
    env = EdgeSimEnv(dataset_path=dataset_path, episode_steps=600)

    state0 = env.reset()
    print("Initial state:", state0)

    # Dummy action (ignored inside env.step for now)
    action = 0
    next_state, reward, done, info = env.step(action)

    print("\n=== Episode summary ===")
    print("Next state [E, C, L]:", next_state)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)


if __name__ == "__main__":
    main()
