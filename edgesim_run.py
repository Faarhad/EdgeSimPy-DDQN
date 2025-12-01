import random

from edge_sim_py import Simulator, Service, EdgeServer


# 1) When should the simulation stop?
def stopping_criterion(model: object) -> bool:
    # Stop after 600 steps (e.g., 600 seconds)
    return model.schedule.steps >= 600


# 2) Very simple placement policy (example baseline)c
def my_algorithm(parameters: dict):
    # At each time step, we try to provision all services
    for service in Service.all():
        # Shuffle edge servers randomly
        edge_servers = random.sample(EdgeServer.all(), EdgeServer.count())

        for edge_server in edge_servers:
            if edge_server.has_capacity_to_host(service=service):
                service.provision(target_server=edge_server)
                break


# 3) Instantiate the simulator
simulator = Simulator(
    dump_interval=10,
    tick_unit="seconds",
    tick_duration=1,
    stopping_criterion=stopping_criterion,
    resource_management_algorithm=my_algorithm,
)

# 4) Load dataset
simulator.initialize(input_file="datasets/sample_dataset.json")

print("Running simulation...")
simulator.run_model()
print("Simulation finished. Total steps:", simulator.schedule.steps)
