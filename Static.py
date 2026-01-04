"""
TABULAR Q-LEARNING TRAFFIC SIGNAL CONTROLLER

- Controls the traffic light via Q-learning.
- Faster simulation: STEP_LENGTH = 1.0s, TOTAL_STEPS = 3000
- MIN_GREEN_STEPS = 5 (seconds)

Metrics (same as static):
 - Total delay (veh-seconds)
 - Average queue length
 - Max queue length
 - Vehicles completed (throughput)
 - Average travel time per vehicle (s)
 - 90th percentile travel time (s)
 - Max travel time (s)
"""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# USER CONFIG
# -----------------------
SUMO_CONFIG_PATH = r"C:\Users\ASUS\Desktop\tracii - Copy (2)\tracii\RL.sumocfg"
STEP_LENGTH = 1.0
TOTAL_STEPS = 3000

# Optional: if SUMO_HOME not in environment, set it here:
# os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

# ----------  Validate SUMO_HOME and add tools ----------
if 'SUMO_HOME' not in os.environ:
    sys.exit("ERROR: Please declare environment variable 'SUMO_HOME'.")
sumo_tools_dir = os.path.join(os.environ['SUMO_HOME'], 'tools')
if not os.path.isdir(sumo_tools_dir):
    sys.exit(f"ERROR: SUMO tools directory not found at: {sumo_tools_dir}")
sys.path.append(sumo_tools_dir)

# ----------  Validate SUMO config exists ----------
if not os.path.isfile(SUMO_CONFIG_PATH):
    sys.exit(f"ERROR: SUMO config file not found: {SUMO_CONFIG_PATH}")

# Use full path to sumo-gui (or sumo) binary
if os.name == 'nt':
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
else:
    sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')

if not os.path.isfile(sumo_binary):
    alt = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    if os.path.isfile(alt):
        sumo_binary = alt
    else:
        sys.exit(f"ERROR: SUMO binary not found in {os.path.join(os.environ['SUMO_HOME'], 'bin')}")

# ----------  Import traci ----------
try:
    import traci
except Exception as e:
    sys.exit(f"ERROR importing traci: {e}")

# ---------- Sumo start configuration ----------
Sumo_config = [
    sumo_binary,
    "-c", SUMO_CONFIG_PATH,
    "--step-length", str(STEP_LENGTH),
    "--lateral-resolution", "0"
]

# -------------------------
# RL + simulation variables
# -------------------------
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]       # 0 = keep, 1 = switch

Q_table = {}
MIN_GREEN_STEPS = 5   # 5 seconds min green
last_switch_step = -MIN_GREEN_STEPS

DETECTORS = {
    "Node1_2_EB_0": "Node1_2_EB_0",
    "Node1_2_EB_1": "Node1_2_EB_1",
    "Node1_2_EB_2": "Node1_2_EB_2",
    "Node2_7_SB_0": "Node2_7_SB_0",
    "Node2_7_SB_1": "Node2_7_SB_1",
    "Node2_7_SB_2": "Node2_7_SB_2",
}
TRAFFIC_LIGHT_ID = "Node2"

# -------------------------
# Utility & RL functions
# -------------------------
def safe_lanearea_getLastStepVehicleNumber(detector_id):
    try:
        return int(traci.lanearea.getLastStepVehicleNumber(detector_id))
    except Exception:
        return 0

def safe_trafficlight_getPhase(tls_id):
    try:
        return int(traci.trafficlight.getPhase(tls_id))
    except Exception:
        return 0

def get_state():
    q_EB_0 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node1_2_EB_0"])
    q_EB_1 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node1_2_EB_1"])
    q_EB_2 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node1_2_EB_2"])
    q_SB_0 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node2_7_SB_0"])
    q_SB_1 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node2_7_SB_1"])
    q_SB_2 = safe_lanearea_getLastStepVehicleNumber(DETECTORS["Node2_7_SB_2"])
    current_phase = safe_trafficlight_getPhase(TRAFFIC_LIGHT_ID)
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)

def get_reward(state):
    total_queue = sum(state[:-1])
    return -float(total_queue)

def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return float(np.max(Q_table[s]))

def update_Q_table(old_state, action, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    old_q = float(Q_table[old_state][action])
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return int(np.argmax(Q_table[state]))

def apply_action(action, tls_id=TRAFFIC_LIGHT_ID, current_simulation_step=0):
    global last_switch_step
    if action == 0:
        return
    elif action == 1:
        try:
            if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
                program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                num_phases = len(program.phases)
                next_phase = (safe_trafficlight_getPhase(tls_id) + 1) % num_phases
                traci.trafficlight.setPhase(tls_id, next_phase)
                last_switch_step = current_simulation_step
        except Exception:
            return

def compute_metrics(queue_history, travel_times, step_length=STEP_LENGTH):
    if queue_history:
        total_delay = sum(q * step_length for q in queue_history)
        avg_queue = float(sum(queue_history)) / len(queue_history)
        max_queue = max(queue_history)
    else:
        total_delay = 0.0
        avg_queue = 0.0
        max_queue = 0

    throughput = len(travel_times)

    if travel_times:
        arr = np.array(travel_times, dtype=np.float32)
        avg_tt = float(arr.mean())
        max_tt = float(arr.max())
        p90_tt = float(np.percentile(arr, 90))
    else:
        avg_tt = 0.0
        max_tt = 0.0
        p90_tt = 0.0

    return total_delay, avg_queue, max_queue, throughput, avg_tt, p90_tt, max_tt

# -------------------------
# Simulation + Training Loop
# -------------------------
step_history = []
time_history = []
reward_history = []
queue_history = []
cumulative_reward = 0.0

depart_times = {}   # vehID -> depart_time (s)
travel_times = []   # finished vehicle travel times

try:
    print("Starting SUMO via TraCI (Q-Learning)...")
    traci.start(Sumo_config)

    # Speed up GUI if available
    try:
        traci.gui.setSchema("View #0", "real world")
        traci.gui.setSpeed("View #0", 5)
    except Exception:
        pass

    print("\n=== Starting Q-Learning Training ===")
    for step in range(TOTAL_STEPS):
        current_time = step * STEP_LENGTH

        state = get_state()
        action = get_action_from_policy(state)
        apply_action(action, tls_id=TRAFFIC_LIGHT_ID, current_simulation_step=step)

        traci.simulationStep()

        # Track vehicle departures/arrivals
        departed_ids = traci.simulation.getDepartedIDList()
        for vid in departed_ids:
            depart_times[vid] = current_time

        arrived_ids = traci.simulation.getArrivedIDList()
        for vid in arrived_ids:
            dt = current_time - depart_times.pop(vid, current_time)
            travel_times.append(dt)

        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward

        update_Q_table(state, action, reward, new_state)

        step_history.append(step)
        time_history.append(current_time)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))

        if step % 200 == 0:
            print(f"Step {step}, Time={current_time:.1f}s, State={state}, "
                  f"Action={action}, Reward={reward:.2f}, CumReward={cumulative_reward:.2f}, "
                  f"Finished={len(travel_times)}")

        # Early stop if no more vehicles
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"No more vehicles expected; stopping at step {step}.")
                break
        except Exception:
            pass

    print("\nSimulation finished, closing TraCI...")
except Exception as e:
    print(f"Exception during simulation: {e}")
finally:
    try:
        traci.close()
    except Exception:
        pass

# -------------------------
# After-run reporting
# -------------------------
print("Training Completed. Static/Baseline Model Summary:\n")

if len(time_history) > 0:
    total_delay, avg_queue, max_queue, throughput, avg_tt, p90_tt, max_tt = compute_metrics(
        queue_history, travel_times
    )

    print("\n=== STATIC METRICS ===")
    print(f"Total delay (veh-seconds): {total_delay:.2f}")
    print(f"Average queue length (vehicles): {avg_queue:.2f}")
    print(f"Max queue length (vehicles): {max_queue}")
    print(f"Vehicles completed (throughput): {throughput}")
    print(f"Average travel time per vehicle (s): {avg_tt:.2f}")
    print(f"90th percentile travel time (s): {p90_tt:.2f}")
    print(f"Max travel time (s): {max_tt:.2f}")

    np.savez("results_ql.npz",
             time=np.array(time_history, dtype=np.float32),
             queue=np.array(queue_history, dtype=np.float32),
             reward=np.array(reward_history, dtype=np.float32),
             travel_times=np.array(travel_times, dtype=np.float32))

    # Optional plots
    plt.figure(figsize=(10, 5))
    plt.plot(time_history, queue_history, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Total Queue Length (vehicles)")
    plt.title("Static/Baseline: Cumulative Reward Over Time")
    plt.grid(True)
    plt.show()
