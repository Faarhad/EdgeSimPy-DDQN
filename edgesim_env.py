import ast
import numpy as np
import pandas as pd

from edge_sim_py import Simulator, EdgeServer, Service
from edge_sim_py.components.power_models.servers.linear_server_power_model import (
    LinearServerPowerModel,
)


class EdgeSimEnv:
    """
    RL-style wrapper around your calibrated 1-cloud + 3-edge EdgeSimPy scenario.

    - Topology, energy model and cost model are aligned with scenario_1cloud3edge.py.
    - One call to step() runs a full simulation episode (episode_steps).
    - State = [total_energy, total_cost, avg_latency].
    - Reward = negative weighted sum of energy, cost, latency.
    """

    def __init__(self, dataset_path: str, episode_steps: int = 600):
        self.dataset_path = dataset_path
        self.episode_steps = episode_steps

        # Reward weights (tune later)
        self.alpha_E = 1.0  # energy weight
        self.alpha_C = 1.0  # cost weight
        self.alpha_L = 1.0  # latency weight

        # Will be filled in reset() / _build_simulator()
        self.simulator = None
        self.cloud_server = None
        self.edge_servers = None
        self.TRACKED_IDS = None

        # Node roles and cost parameters (per node id)
        self.ROLE = {}
        self.CPU_PRICE = {}
        self.RAM_PRICE = {}
        self.BASE_COST = {}

    # -----------------------------
    # 1) Internal helper: stopping criterion
    # -----------------------------
    def _stopping_criterion(self, model: object) -> bool:
        # Simulate for a fixed number of steps (e.g., 600 seconds)
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
            resource_management_algorithm=None,  # will be set below
        )

        # 2.2 Initialize with dataset (ADT-like JSON)
        self.simulator.initialize(input_file=self.dataset_path)

        # 2.3 Select 1 cloud + 3 edge nodes
        all_servers = sorted(EdgeServer.all(), key=lambda s: s.id)
        if len(all_servers) < 4:
            raise RuntimeError("Dataset has fewer than 4 edge servers")

        self.cloud_server = all_servers[0]
        self.edge_servers = all_servers[1:4]

        # Assign roles
        self.ROLE = {}
        self.ROLE[self.cloud_server.id] = "cloud"
        for es in self.edge_servers:
            self.ROLE[es.id] = "edge"

        # 2.4 Capacity scaling: make node 1 behave like a "cloud"
        self.cloud_server.name = "cloudNode"
        self.cloud_server.cpu *= 4
        self.cloud_server.memory *= 4
        self.cloud_server.disk *= 4

        # 2.5 Energy model calibration (P_idle / P_max) using LinearServerPowerModel
        for srv in [self.cloud_server] + list(self.edge_servers):
            if srv.power_model is None:
                srv.power_model = LinearServerPowerModel
            if srv.power_model_parameters is None:
                srv.power_model_parameters = {}

        def set_Pidle_Pmax(server, P_idle, P_max):
            """
            Map (P_idle, P_max) to EdgeSimPy's linear model params:
            P_idle = static_power_percentage * max_power_consumption
            """
            server.power_model_parameters["max_power_consumption"] = float(P_max)
            server.power_model_parameters["static_power_percentage"] = float(P_idle) / float(P_max)

        # Cloud more power-hungry
        set_Pidle_Pmax(self.cloud_server, P_idle=250.0, P_max=500.0)

        # Edges lighter
        for es in self.edge_servers:
            set_Pidle_Pmax(es, P_idle=60.0, P_max=120.0)

        # 2.6 Cost model parameters (per node) – same structure as scenario_1cloud3edge
        self.CPU_PRICE = {}
        self.RAM_PRICE = {}
        self.BASE_COST = {}

        for srv in [self.cloud_server] + list(self.edge_servers):
            if self.ROLE.get(srv.id) == "cloud":
                # Cloud tariffs
                self.CPU_PRICE[srv.id] = 0.000020  # cost per CPU unit per second
                self.RAM_PRICE[srv.id] = 0.000010  # cost per RAM unit per second
                self.BASE_COST[srv.id] = 0.00050   # fixed cost per second
            else:
                # Edge tariffs
                self.CPU_PRICE[srv.id] = 0.000010
                self.RAM_PRICE[srv.id] = 0.000005
                self.BASE_COST[srv.id] = 0.00025   # fixed cost per second

        # 2.7 Pick 4 microservices to track
        all_services = Service.all()
        if len(all_services) < 4:
            raise RuntimeError("Dataset has fewer than 4 services")

        tracked_services = all_services[:4]
        self.TRACKED_IDS = {srv.id for srv in tracked_services}

        # Optionally, scale resource demand of the tracked services (as in scenario_1cloud3edge)
        for s in tracked_services:
            if hasattr(s, "cpu_demand"):
                s.cpu_demand *= 2.0
            if hasattr(s, "memory_demand"):
                s.memory_demand *= 1.5
            if hasattr(s, "disk_demand"):
                s.disk_demand *= 1.2

        # 2.8 Attach a simple baseline placement algorithm
        def my_baseline_algorithm(parameters: dict):
            """
            Baseline:
            - Only manage the 4 tracked microservices.
            - If a tracked service is not placed, try to place it on one of
              {cloud, edge1, edge2, edge3} with enough capacity.
            - Order is deterministic here; you can randomise if desired.
            """
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
        # Build a brand new simulator for this episode
        self._build_simulator()

        # Initial state can be zero vector; we haven't run anything yet.
        initial_state = np.zeros(3, dtype=np.float32)  # [energy, cost, latency]
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

        # 4.2 Collect edge server logs and restrict to our 1 cloud + 3 edge nodes
        edge_logs = pd.DataFrame(self.simulator.agent_metrics["EdgeServer"])
        selected_ids = [self.cloud_server.id] + [es.id for es in self.edge_servers]
        edge_logs = edge_logs[edge_logs["Instance ID"].isin(selected_ids)].copy()

        # 4.2.1 Compute cost per node per step using the calibrated cost model
        delta_t = getattr(self.simulator, "tick_duration", 1)

        if "CPU Demand" in edge_logs.columns and "RAM Demand" in edge_logs.columns:
            def compute_cost_row(row):
                node_id = row["Instance ID"]
                cpu_demand = row["CPU Demand"]
                ram_demand = row["RAM Demand"]

                cpu_price = self.CPU_PRICE.get(node_id, 0.0)
                ram_price = self.RAM_PRICE.get(node_id, 0.0)
                base_cost = self.BASE_COST.get(node_id, 0.0)

                per_second_cost = base_cost + cpu_demand * cpu_price + ram_demand * ram_price
                return per_second_cost * delta_t

            edge_logs["Cost"] = edge_logs.apply(compute_cost_row, axis=1)
            total_cost = float(edge_logs["Cost"].sum())
        else:
            total_cost = 0.0

        # 4.2.2 Energy: sum of power * delta_t over all selected nodes and steps
        if "Power Consumption" in edge_logs.columns:
            total_energy = float(edge_logs["Power Consumption"].sum() * delta_t)
        else:
            total_energy = 0.0

        # 4.3 Collect latency from 'User' metrics (flattened delays)
        user_logs = pd.DataFrame(self.simulator.agent_metrics["User"])
        avg_latency = 0.0

        if not user_logs.empty and "Delays" in user_logs.columns:
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
                    try:
                        latencies.append(float(delay))
                    except Exception:
                        continue

            if latencies:
                avg_latency = float(np.mean(latencies))

        # 4.4 Build next_state: [energy, cost, latency]
        next_state = np.array(
            [total_energy, total_cost, avg_latency],
            dtype=np.float32,
        )

        # 4.5 Compute reward (negative multi-objective)
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
