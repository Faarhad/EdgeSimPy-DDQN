import random
import pandas as pd
import ast

from edge_sim_py import Simulator, EdgeServer, Service


# -------------------------------
# 1. Stopping criterion
# -------------------------------
def stopping_criterion(model: object) -> bool:
    # Simulate 600 steps = 600 seconds (tick_duration=1, tick_unit="seconds")
    return model.schedule.steps >= 600


# -------------------------------
# 2. Create simulator
# -------------------------------
simulator = Simulator(
    dump_interval=10,
    tick_unit="seconds",
    tick_duration=1,
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=None,  # we will assign later
)

# -------------------------------
# 3. Load an official tutorial dataset
#    (no need to download locally)
# -------------------------------
simulator.initialize(
    input_file=(
        "datasets/sample_dataset.json"
    )
)

print("Dataset loaded.")
print("Total EdgeServers in dataset:", EdgeServer.count())
print("Total Services in dataset:", Service.count())

# -------------------------------
# 4. Select 1 cloud + 3 edge nodes
# -------------------------------
all_servers = sorted(EdgeServer.all(), key=lambda s: s.id)

if len(all_servers) < 4:
    raise RuntimeError("Dataset has fewer than 4 edge servers – pick another dataset.")

cloud_server = all_servers[0]
edge_servers = all_servers[1:4]  # next three

ROLE = {}
ROLE[cloud_server.id] = "cloud"
for es in edge_servers:
    ROLE[es.id] = "edge"

print("\n=== Node roles ===")
print(f"Cloud server ID: {cloud_server.id}")
print("Edge servers IDs:", [es.id for es in edge_servers])

print("\n=== Original power models ===")
print("Cloud server:",
      cloud_server.power_model,
      cloud_server.power_model_parameters)

for es in edge_servers:
    print(f"Edge server {es.id}:",
          es.power_model,
          es.power_model_parameters)
    
    

from edge_sim_py.components.power_models.servers.linear_server_power_model import (
    LinearServerPowerModel,
)

# Make sure all 4 nodes have a power model
for srv in [cloud_server] + edge_servers:
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


# Example numbers – change them to match your paper’s calibration
# Cloud more power-hungry:
set_Pidle_Pmax(cloud_server, P_idle=250.0, P_max=500.0)

# Edges lighter:
for es in edge_servers:
    set_Pidle_Pmax(es, P_idle=60.0, P_max=120.0)


def get_Pidle_Pmax(server):
    params = server.power_model_parameters or {}
    if "max_power_consumption" in params and "static_power_percentage" in params:
        P_max = params["max_power_consumption"]
        P_idle = params["static_power_percentage"] * P_max
        return P_idle, P_max
    return None, None

print("\n=== Calibrated energy parameters (P_idle, P_max) ===")
print("Cloud server:", get_Pidle_Pmax(cloud_server))
for es in edge_servers:
    print(f"Edge server {es.id}:", get_Pidle_Pmax(es))


# -------------------------------
# Cost model parameters (per-node)
# -------------------------------

# Unit prices per node (e.g., $ per unit of CPU-second and RAM-unit-second)
CPU_PRICE = {}
RAM_PRICE = {}
BASE_COST = {} 

# Example: cloud more expensive per unit than edge
for srv in [cloud_server] + edge_servers:
    if ROLE.get(srv.id) == "cloud":
        # Cloud tariffs (you can tune these)
        CPU_PRICE[srv.id] = 0.000020  # cost per CPU unit per second
        RAM_PRICE[srv.id] = 0.000010  # cost per RAM unit per second
        BASE_COST[srv.id] = 0.00050   #Fixed cost for having a cloud node "on"
    else:
        # Edge tariffs (typically lower or different)
        CPU_PRICE[srv.id] = 0.000010
        RAM_PRICE[srv.id] = 0.000005
        BASE_COST[srv.id] = 0.00025  # per second

# -------------------------------
# 5. Select 4 microservices to track
# -------------------------------
all_services = Service.all()
if len(all_services) < 4:
    raise RuntimeError("Dataset has fewer than 4 services – cannot build 4-microservice scenario.")

tracked_services = all_services[:4]
TRACKED_IDS = {srv.id for srv in tracked_services}

print("\nTracked service IDs:", TRACKED_IDS)


from edge_sim_py import EdgeServer, Service

# -------------------------------
# 6(new). Manipulate topology and workload
# -------------------------------
print("\n=== Before manipulation (capacities) ===")
for s in [cloud_server] + edge_servers:
    print(f"Server {s.id}: CPU={s.cpu}, RAM={s.memory}, Disk={s.disk}")

# 6.1 Make node 1 behave like a CLOUD
# Simple rule: multiply its capacity by a factor
cloud_server.name = "cloudNode"
cloud_server.cpu *= 4
cloud_server.memory *= 4
cloud_server.disk *= 4

print("\nCloud-like server after scaling:")
print(f"Server {cloud_server.id}: CPU={cloud_server.cpu}, RAM={cloud_server.memory}, Disk={cloud_server.disk}")

# 6.3 Approximate network distance via coordinates (if supported)

def safe_set_coordinates(server, x, y):
    # Different versions may store coordinates under different attribute names.
    # We'll try a few common ones.
    if hasattr(server, "coordinates"):
        server.coordinates = (x, y)
    elif hasattr(server, "location"):
        server.location = (x, y)
    else:
        # If no coordinate attribute, just ignore
        pass

print("\nSetting logical coordinates for cloud/edge nodes (if supported)...")

# Put cloud "far away"
safe_set_coordinates(cloud_server, 1000, 1000)

# Put edges closer to origin (representing proximity to users)
for idx, es in enumerate(edge_servers):
    safe_set_coordinates(es, 100 * (idx + 1), 0)

print("\n=== Tracked services before scaling ===")
for s in tracked_services:
    print(f"Service {s.id} attrs: {s.__dict__.keys()}")

# 6.4 Scale resource demand of our 4 tracked services
for s in tracked_services:
    # These attribute names are the most likely; adjust if you see slightly different names.
    if hasattr(s, "cpu_demand"):
        s.cpu_demand *= 2.0  # double CPU demand
    if hasattr(s, "ram_demand"):
        s.memory_demand *= 1.5  # increase RAM demand by 50%
    if hasattr(s, "disk_demand"):
        s.disk_demand *= 1.2  # slight increase in disk

print("\nTracked services after scaling:")
for s in tracked_services:
    # We guard against missing attributes
    cpu_d = getattr(s, "cpu_demand", None)
    memory_d = getattr(s, "memory_demand", None)
    disk_d = getattr(s, "disk_demand", None)
    print(f"Service {s.id}: cpu_demand={cpu_d}, ram_demand={memory_d}, disk_demand={disk_d}")


# -------------------------------
# 6(old). Define placement policy for this scenario
#    (baseline heuristic - later replaced by DDQN)
# -------------------------------
def my_algorithm(parameters: dict):
    """
    Simple baseline:
    - Consider only the 4 tracked microservices.
    - If a tracked service is not hosted, try to place it on
      one of {cloud, edge1, edge2, edge3} with enough capacity.
    - Choose host in a random order (this is where DDQN will go later).
    """

    # Build candidate host list once
    candidates = [cloud_server] + edge_servers

    for service in Service.all():
        # Only manage our 4 "microservices of interest"
        if service.id not in TRACKED_IDS:
            continue

        # Skip if already placed or being provisioned
        if service.server is not None or service.being_provisioned:
            continue

        # Try to place on one of the 4 nodes
        for host in random.sample(candidates, len(candidates)):
            if host.has_capacity_to_host(service=service):
                service.provision(target_server=host)
                break


# Attach our algorithm to the simulator
simulator.resource_management_algorithm = my_algorithm

# -------------------------------
# 7. Run the simulation
# -------------------------------
print("\nRunning 1-cloud + 3-edge scenario simulation...")
simulator.run_model()
print("Simulation finished. Total steps:", simulator.schedule.steps)

# -------------------------------
# 8. Inspect & save edge server logs
# -------------------------------

edge_logs = pd.DataFrame(simulator.agent_metrics["EdgeServer"])
print("\nAvailable metric groups:", simulator.agent_metrics.keys())
edge_logs.to_csv("scenario_1cloud3edge_edge_logs.csv", index=False)
print("\nSaved scenario_1cloud3edge_edge_logs.csv in the project folder.")

# access to latency log
app_logs = pd.DataFrame(simulator.agent_metrics["User"])
app_logs.to_csv("scenario_1cloud3edge_app_logs.csv", index=False)
print("\nSaved scenario_1cloud3edge_app_logs.csv")
print("User log columns:", app_logs.columns)
print(app_logs.head())
# -------------------------------
# Cost model computation
# -------------------------------

# Duration of one simulation step (seconds)
delta_t = simulator.tick_duration  # we set tick_duration=1, but keep it generic

def compute_cost_row(row):
    """
    Compute instantaneous monetary cost for a single node at a single time step.
    Cost_j(t) = base_j + p_j * CPU_Demand_j(t) + eta_j * RAM_Demand_j(t)
    """
    node_id = row["Instance ID"]
    cpu_demand = row["CPU Demand"]
    ram_demand = row["RAM Demand"]

    cpu_price = CPU_PRICE.get(node_id, 0.0)
    ram_price = RAM_PRICE.get(node_id, 0.0)
    base_cost = BASE_COST.get(node_id, 0.0)

    per_second_cost = base_cost + cpu_demand * cpu_price + ram_demand * ram_price

    # cost for this time step (already per-second, so multiply by delta_t)
    return per_second_cost * delta_t

# Per-step node cost
edge_logs["Cost"] = edge_logs.apply(compute_cost_row, axis=1)

# Total accumulated cost per node
total_cost_per_node = (
    edge_logs.groupby("Instance ID")["Cost"]
    .sum()
    .reset_index()
    .rename(columns={"Cost": "Total Cost"})
)

print("\nTotal monetary cost per node:")
print(total_cost_per_node)


# -------------------------------
# Extract latency per request (flattened)
# -------------------------------
user_logs = pd.DataFrame(simulator.agent_metrics["User"])

def expand_delays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the 'Delays' dict into one row per (user, request).
    Each entry in 'Delays' looks like: {'1': 10, '2': 15, ...}
    """
    rows = []
    for _, row in df.iterrows():
        raw = row["Delays"]

        # Parse safely (can be dict or string)
        if isinstance(raw, dict):
            delays_dict = raw
        else:
            try:
                delays_dict = ast.literal_eval(str(raw))
            except Exception:
                continue

        for req_id, delay in delays_dict.items():
            rows.append({
                "Time Step": row["Time Step"],
                "User ID": row["Instance ID"],
                "Request ID": int(req_id),
                "Latency": float(delay),  # delay in ms or ticks depending on simulator
            })

    return pd.DataFrame(rows)


latency_df = expand_delays(user_logs)

if latency_df.empty:
    print("\nWARNING: No delays found in User metrics.")
else:
    print("\nLatency extracted successfully.")
    print(latency_df.head())

# -------------------------------
# Aggregate latency per time step
# -------------------------------
if not latency_df.empty:
    latency_by_step = (
        latency_df.groupby("Time Step")["Latency"]
        .mean()   # you may choose .quantile(0.95) instead
        .reset_index()
        .rename(columns={"Latency": "Avg_Appl_Latency_on_Timestep"})
    )
else:
    latency_by_step = pd.DataFrame(columns=["Time Step", "Avg_Appl_Latency_on_Timestep"])

# Merge with edge logs on time step
merged_df = pd.merge(edge_logs, latency_by_step, on="Time Step", how="left")

# Save extended logs (with per-step cost)
edge_logs.to_csv("scenario_1cloud3edge_edge_logs_with_cost.csv", index=False)
print("\nSaved scenario_1cloud3edge_edge_logs_with_cost.csv in the project folder.")

# Save per-node total cost
total_cost_per_node.to_csv("scenario_1cloud3edge_cost_per_node.csv", index=False)
print("Saved scenario_1cloud3edge_cost_per_node.csv in the project folder.")

selected_ids = [cloud_server.id] + [es.id for es in edge_servers]
filtered_logs = edge_logs[edge_logs["Instance ID"].isin(selected_ids)]

print("\nFiltered logs only for 1 cloud + 3 edge nodes:")

filtered_logs.to_csv("scenario_filtered_1cloud3edge.csv", index=False)
print("\nSaved filtered log file.")


merged_df.to_csv("scenario_1cloud3edge_metrics.csv", index=False)
print("\nSaved unified scenario_1cloud3edge_metrics.csv")