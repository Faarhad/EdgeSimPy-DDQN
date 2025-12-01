from edgesim_env import EdgeSimEnv

def main():
    # Path to your dataset JSON (the same one you used in scenario_1cloud3edge.py)
    dataset_path = "datasets/sample_dataset.json"

    env = EdgeSimEnv(dataset_path=dataset_path, episode_steps=600)

    state = env.reset()
    print("Initial state:", state)

    # Dummy action (we ignore it inside env.step for now)
    action = 0

    next_state, reward, done, info = env.step(action)

    print("Next state:", next_state)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)


if __name__ == "__main__":
    main()
