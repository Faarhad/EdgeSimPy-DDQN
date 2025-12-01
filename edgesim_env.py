import numpy as np
import pandas as pd

from edge_sim_py import Simulator, EdgeServer, Service


class EdgeSimEnv:
    """
    Minimal RL-style wrapper around your 1-cloud + 3-edge EdgeSimPy scenario.

    For now:
      - Action is ignored (we'll add real actions later).
      - One call to step() = run a full 600-step simulation.
      - State = [total_energy, total_cost, avg_latency].
      - Reward = negative weighted sum of energy, cost, latency.
    """

    def __init__(self, dataset_path: str, episode_steps: int = 600):
        self.dataset_path = dataset_path
        self.episode_steps = episode_steps

        # Reward weights (you can tune these later)
        self.alpha_E = 1.0     # energy weight
        self.alpha_C = 1.0     # cost weight
        self.alpha_L = 1.0     # latency weight

        # These will be filled in reset()
        self.simulator = None
        self.cloud_server = None
        self.edge_servers = None
        self.TRACKED_IDS = None

    # -----------------------------
    # 1) Internal helper: stopping criterion
    # -----------------------------
    def _stopping_criterion(self, model: object) -> bool:
        return model.schedule.steps >= self.episode_steps

    # -----------------------------
    # 2) Internal helper: build simulator and topology
    #    (this is basically your scenario_1cloud3edge.py refactored)
    # -----------------------------
    def _build_simulator(self):
        # 2.1 Create simulator
        self.simulator = Simulator(
            dump_interval=10,
            tick_unit="seconds",
            tick_duration=1,
            stopping_criterion=self._stopping_criterion,
            resource_management_algorithm=None,  # we will attach algo later if needed
        )

        # 2.2 Initialize with dataset (for now: treat JSON as ADT)
        self.simulator.initialize(input_file=self.dataset_path)

        # 2.3 Select 1 cloud + 3 edge nodes
        all_servers = sorted(EdgeServer.all(), key=lambda s: s.id)
        if len(all_servers) < 4:
            raise RuntimeError("Dataset has fewer than 4 edge servers")

        self.cloud_server = all_servers[0]
        self.edge_servers = all_servers[1:4]

        # (optional) capacity scaling: make node 1 behave like cloud
        self.cloud_server.name = "cloudNode"
        self.cloud_server.cpu *= 4
        self.cloud_server.memory *= 4
        self.cloud_server.disk *= 4

        # 2.4 Energy model calibration (P_idle / P_max)
        if getattr(self.cloud_server, "energy_model", None) is not None:
            self.cloud_server.energy_model["idle_power"] = 250.0
            self.cloud_server.energy_model["max_power"] = 500.0

        for es in self.edge_servers:
            if getattr(es, "energy_model", None) is not None:
                es.energy_model["idle_power"] = 60.0
                es.energy_model["max_power"] = 120.0

        # 2.5 Pick 4 microservices to track (for future actions, not used yet)
        all_services = Service.all()
        if len(all_services) < 4:
            raise RuntimeError("Dataset has fewer than 4 services")

        tracked_services = all_services[:4]
        self.TRACKED_IDS = {srv.id for srv in tracked_services}

        # 2.6 Attach a very simple baseline placement algorithm
        #     (same logic as in your script, just refactored)
        def my_baseline_algorithm(parameters: dict):
            candidates = [self.cloud_server] + list(self.edge_servers)
            for service in Service.all():
                if service.id not in self.TRACKED_IDS:
                    continue
                if service.server is not None or service.being_provisioned:
                    continue
                for host in candidates:
                    if host.has_capacity_to_host(service=service):
                        service.provision(target_server=host)
                        break

        self.simulator.resource_management_algorithm = my_baseline_algorithm

    # -----------------------------
    # 3) reset(): start a fresh episode
    # -----------------------------
    def reset(self):
        """
        Build a fresh simulator and return an initial dummy state.
        For now, the "initial state" is just zeros; the real state is observed after step().
        """
        # Clear any global EdgeSimPy state if needed (depends on library design)
        # In many frameworks, we may need to reset registries; but we'll assume fresh process here.

        # Build a brand new simulator for this episode
        self._build_simulator()

        # Initial state can be zero vector; we haven't run anything yet.
        # Later we can encode capacities, roles, etc.
        initial_state = np.zeros(3, dtype=np.float32)  # [energy, cost, latency] placeholder
        return initial_state

    # -----------------------------
    # 4) step(action): run one full simulation under (currently ignored) action
    # -----------------------------
    def step(self, action: int):
        """
        For now, 'action' is ignored — we always run the same baseline algorithm.
        We:
          - run the simulator for episode_steps
          - collect energy, cost, latency
          - build next_state and reward
          - mark episode as done=True
        """
        # 4.1 Run the simulation
        self.simulator.run_model()

        # 4.2 Collect edge server logs
        edge_logs = pd.DataFrame(self.simulator.agent_metrics["EdgeServer"])

        # Energy: sum of power * delta_t over all nodes and steps
        if "Power Consumption" in edge_logs.columns:
            # tick_duration=1 second ⇒ energy in Joules ~ sum(Power[W] * 1s)
            total_energy = edge_logs["Power Consumption"].sum()
        else:
            total_energy = 0.0

        # Cost: if you already added a 'Cost' column in your script, use it.
        # For now, we assume no cost column → set cost=0 and mark TODO.
        if "Cost" in edge_logs.columns:
            total_cost = edge_logs["Cost"].sum()
        else:
            total_cost = 0.0  # TODO: plug your cost model here

        # 4.3 Collect latency from 'User' metrics (mean over all requests)
        user_logs = pd.DataFrame(self.simulator.agent_metrics["User"])
        avg_latency = 0.0
        if "Delays" in user_logs.columns and not user_logs.empty:
            import ast
            latencies = []
            for _, row in user_logs.iterrows():
                raw = row["Delays"]
                if isinstance(raw, dict):
                    delays_dict = raw
                else:
                    try:
                        delays_dict = ast.literal_eval(str(raw))
                    except Exception:
                        continue
                for _, delay in delays_dict.items():
                    latencies.append(float(delay))
            if latencies:
                avg_latency = float(np.mean(latencies))

        # 4.4 Build next_state: [energy, cost, latency]
        next_state = np.array(
            [total_energy, total_cost, avg_latency],
            dtype=np.float32,
        )

        # 4.5 Compute reward (negative multi-objective)
        # You can tune alpha_E, alpha_C, alpha_L later
        reward = (
            - self.alpha_E * total_energy
            - self.alpha_C * total_cost
            - self.alpha_L * avg_latency
        )

        done = True  # one episode = one simulator run
        info = {
            "total_energy": total_energy,
            "total_cost": total_cost,
            "avg_latency": avg_latency,
        }

        return next_state, reward, done, info
