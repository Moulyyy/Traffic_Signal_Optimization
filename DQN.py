"""
DEEP Q-NETWORK (DQN) TRAFFIC SIGNAL CONTROLLER

- Controls the traffic light using a neural network.
- Faster simulation: STEP_LENGTH = 1.0s, TOTAL_STEPS = 3000
- MIN_GREEN_STEPS = 5

Metrics (same as others):
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# SUMO Path Setup
# -------------------------
SUMO_CONFIG_PATH = r"C:\Users\ASUS\Desktop\tracii\tracii\RL.sumocfg"
STEP_LENGTH = 1.0
TOTAL_STEPS = 3000

if 'SUMO_HOME' not in os.environ:
    sys.exit("❌ ERROR: Please declare environment variable 'SUMO_HOME' before running this script.")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if not os.path.isdir(tools):
    sys.exit(f"❌ ERROR: SUMO tools not found at: {tools}")
sys.path.append(tools)

if os.name == 'nt':
    sumo_gui_path = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
else:
    sumo_gui_path = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')

if not os.path.isfile(sumo_gui_path):
    fallback = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
    if os.path.isfile(fallback):
        print("⚠️ sumo-gui not found — running in headless mode (no GUI).")
        sumo_binary = fallback
    else:
        sys.exit("❌ ERROR: Neither sumo-gui nor sumo binary found in SUMO_HOME/bin.")
else:
    sumo_binary = sumo_gui_path

if not os.path.isfile(SUMO_CONFIG_PATH):
    sys.exit(f"❌ ERROR: SUMO config not found at {SUMO_CONFIG_PATH}")

print("\n===== SUMO CONFIGURATION (DQN) =====")
print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")
print(f"SUMO binary: {sumo_binary}")
print(f"SUMO config: {SUMO_CONFIG_PATH}")
print(f"Working directory: {os.getcwd()}")
print("==============================\n")

import traci

Sumo_config = [
    sumo_binary,
    "-c", SUMO_CONFIG_PATH,
    "--step-length", str(STEP_LENGTH),
    "--lateral-resolution", "0"
]

# -------------------------
# RL Hyperparameters
# -------------------------
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]  # 0 = keep, 1 = switch
MIN_GREEN_STEPS = 5
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# DQN Model Definition
# -------------------------
def build_model(state_size, action_size):
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

def to_array(state_tuple):
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

state_size = 7
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)

# -------------------------
# Utility Functions
# -------------------------
DETECTORS = [
    "Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
    "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"
]
TLS_ID = "Node2"

def safe_get_detector(det_id):
    try:
        return traci.lanearea.getLastStepVehicleNumber(det_id)
    except Exception:
        return 0

def safe_get_phase(tls_id):
    try:
        return traci.trafficlight.getPhase(tls_id)
    except Exception:
        return 0

def get_state():
    q_vals = [safe_get_detector(d) for d in DETECTORS]
    current_phase = safe_get_phase(TLS_ID)
    return (*q_vals, current_phase)

def get_reward(state):
    total_queue = sum(state[:-1])
    return -float(total_queue)

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        Q_values = dqn_model.predict(to_array(state), verbose=0)[0]
        return int(np.argmax(Q_values))

def apply_action(action, tls_id=TLS_ID, current_step=0):
    global last_switch_step
    if action == 0:
        return
    try:
        if current_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (safe_get_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_step
    except Exception:
        pass

def update_DQN(old_state, action, reward, new_state):
    old_array = to_array(old_state)
    new_array = to_array(new_state)
    Q_old = dqn_model.predict(old_array, verbose=0)[0]
    Q_new = dqn_model.predict(new_array, verbose=0)[0]
    best_future_q = np.max(Q_new)
    Q_old[action] = Q_old[action] + ALPHA * (reward + GAMMA * best_future_q - Q_old[action])
    dqn_model.fit(old_array, np.array([Q_old]), verbose=0)

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

depart_times = {}
travel_times = []

print("\n🚦 Starting SUMO (DQN)...\n")

try:
    traci.start(Sumo_config)
    print("✅ SUMO started successfully.")

    # Speed up GUI if available
    try:
        traci.gui.setSchema("View #0", "real world")
        traci.gui.setSpeed("View #0", 5)
    except Exception:
        pass
    
    for step in range(TOTAL_STEPS):
        current_time = step * STEP_LENGTH
        state = get_state()
        action = get_action_from_policy(state)
        apply_action(action, tls_id=TLS_ID, current_step=step)

        traci.simulationStep()

        # Departures and arrivals for travel time
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
        update_DQN(state, action, reward, new_state)

        step_history.append(step)
        time_history.append(current_time)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))

        if step % 200 == 0:
            Q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
            print(f"Step {step}, Time={current_time:.1f}s, State={state}, Action={action}, "
                  f"Reward={reward:.2f}, CumReward={cumulative_reward:.2f}, "
                  f"Finished={len(travel_times)}")

        # Early stop if no more vehicles
        try:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print(f"No more vehicles expected; stopping at step {step}.")
                break
        except Exception:
            pass

except Exception as e:
    print("❌ Simulation Error:", e)

finally:
    traci.close()
    print("✅ SUMO connection closed.\n")

# -------------------------
# Results & Visualization
# -------------------------
print("Training Completed. DQN Model Summary:\n")
dqn_model.summary()

if len(time_history) > 0:
    total_delay, avg_queue, max_queue, throughput, avg_tt, p90_tt, max_tt = compute_metrics(
        queue_history, travel_times
    )

    print("\n=== DQN METRICS ===")
    print(f"Total delay (veh-seconds): {total_delay:.2f}")
    print(f"Average queue length (vehicles): {avg_queue:.2f}")
    print(f"Max queue length (vehicles): {max_queue}")
    print(f"Vehicles completed (throughput): {throughput}")
    print(f"Average travel time per vehicle (s): {avg_tt:.2f}")
    print(f"90th percentile travel time (s): {p90_tt:.2f}")
    print(f"Max travel time (s): {max_tt:.2f}")

    np.savez("results_dqn.npz",
             time=np.array(time_history, dtype=np.float32),
             queue=np.array(queue_history, dtype=np.float32),
             reward=np.array(reward_history, dtype=np.float32),
             travel_times=np.array(travel_times, dtype=np.float32))

    plt.figure(figsize=(10, 5))
    plt.plot(time_history, queue_history, label="Total Queue Length")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Queue Length (vehicles)")
    plt.title("DQN: Queue Length Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()
