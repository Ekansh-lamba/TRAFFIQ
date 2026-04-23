import traci
import time

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
SUMO_CONFIG = "junction.sumocfg"

traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start"])
print("Connected to SUMO")

# 🔥 GET REAL TLS IDS FROM SUMO
tls_ids = traci.trafficlight.getIDList()
print("Traffic lights found:", tls_ids)

if len(tls_ids) == 0:
    raise RuntimeError("No traffic lights found in the network!")

TLS_ID = tls_ids[0]  # use the first one
print("Using TLS ID:", TLS_ID)

for step in range(200):
    traci.simulationStep()

    if step % 30 == 0:
        current_phase = traci.trafficlight.getPhase(TLS_ID)

        logic = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
        num_phases = len(logic.phases)

        next_phase = (current_phase + 1) % num_phases
        traci.trafficlight.setPhase(TLS_ID, next_phase)

    phase = traci.trafficlight.getPhase(TLS_ID)
    lanes = traci.trafficlight.getControlledLanes(TLS_ID)
    total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

    print(f"Step {step:03d} | Phase {phase} | Queue {total_queue}")
    time.sleep(0.1)


traci.close()
print("Simulation ended")
