import gym
from gym import spaces
import numpy as np
import random
from collections import deque
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler 
from scapy.all import IP, Ether, sendp, RandIP, RandShort 
import threading 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score
import time

# Define DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class TrafficPreventionEnv(gym.Env):
    def __init__(self, learning_rate=0.001, rate_limit=100, quarantine_duration=10):
        super(TrafficPreventionEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        self.state = None
        self.models = [SVR() for _ in range(self.action_space.n)]
        self.memory = deque(maxlen=5000)

        # DQN model initialization
        self.dqn_model = DQN(11, 3)
        self.target_dqn_model = DQN(11, 3)
        self.target_update_freq = 100
        self.steps = 0
        self.dqn_optimizer = optim.Adam(self.dqn_model.parameters(), lr=learning_rate)

        self.rate_limit = rate_limit
        self.ip_packet_count = {}
        self.burst_threshold = 10

        self.scaler = MinMaxScaler()
        self.scaler_fitted = False

        self.packet_count = 0
        self.total_actions = 0
        self.correct_predictions = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.predictions = []
        self.true_labels = []

        self.quarantine_duration = quarantine_duration
        self.quarantine_timestamps = {}

    def reset(self):
        self.packet_count = 0
        self.ip_packet_count.clear()
        self.state = np.zeros(11, dtype=np.float32)
        return self.get_observation()

    def get_observation(self):
        return self.state

    def step(self, action, real_label):
        self.total_actions += 1
        self.predictions.append(action)
        self.true_labels.append(real_label)

        if action == real_label:
            reward = 10  # Correct action
            self.correct_predictions += 1
            if action == 0:
                self.true_negatives += 1
            else:
                self.true_positives += 1
        elif action == 1:  # Drop
            reward = -10 if real_label == 1 else -1  # Higher penalty for dropping malicious
            if real_label == 0:  # False positive
                self.false_positives += 1
        else:  # Quarantine
            reward = -5 if real_label == 0 else 5  # Encourage quarantine for malicious
            if real_label == 0:  # False negative
                self.false_negatives += 1

        next_state = self.get_observation()

        if len(self.memory) > 32:
            self.replay()

        return next_state, reward, False, {}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        batch_size = 64
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            X = np.array([state]).astype(np.float32)
            y = np.array([reward]).astype(np.float32)

            try:
                self.models[action].fit(X.reshape(-1, 11), y)
            except Exception as e:
                print(f"Error fitting model {action}: {e}")

    def process_packet(self, packet):
        try:
            if IP in packet:
                src_ip = packet[IP].src

                # Check if the IP is currently quarantined
                current_time = time.time()
                if src_ip in self.quarantine_timestamps:
                    if current_time < self.quarantine_timestamps[src_ip]:
                        print(f"Traffic from {src_ip} is currently quarantined. Dropping packet.")
                        return
                    else:
                        del self.quarantine_timestamps[src_ip]

                # Rate limiting
                self.packet_count += 1
                if src_ip not in self.ip_packet_count:
                    self.ip_packet_count[src_ip] = [0, current_time]

                if current_time - self.ip_packet_count[src_ip][1] < 5:
                    self.ip_packet_count[src_ip][0] += 1
                else:
                    self.ip_packet_count[src_ip] = [1, current_time]

                if self.ip_packet_count[src_ip][0] > self.rate_limit:
                    print(f"Rate limit exceeded for {src_ip}. Dropping packet.")
                    return

                if self.ip_packet_count[src_ip][0] > self.burst_threshold:
                    print(f"Burst detected from {src_ip}. Quarantining traffic.")
                    self.quarantine_timestamps[src_ip] = current_time + self.quarantine_duration
                    return

                # Feature extraction
                features = [
                    len(packet),
                    *list(map(int, src_ip.split('.'))),
                    *list(map(int, packet[IP].dst.split('.'))),
                    packet[IP].proto,
                    0  # Placeholder for additional features
                ]

                if len(features) != 11:
                    print(f"Feature count mismatch: {len(features)} features extracted.")
                    return

                # Handle scaling
                if not self.scaler_fitted:
                    self.scaler.fit([features])
                    self.scaler_fitted = True
                    scaled_features = self.scaler.transform([features])  # Ensure this is done after fitting
                else:
                    scaled_features = self.scaler.transform([features])

                # Ensure scaled_features is not None
                if scaled_features is None or len(scaled_features) == 0:
                    print("Scaled features are invalid. Skipping packet.")
                    return

                self.state = scaled_features[0]

                action_values = []
                for a in range(self.action_space.n):
                    try:
                        if hasattr(self.models[a], 'support_'):
                            pred_value = self.models[a].predict(self.state.reshape(1, -1))
                            action_values.append(pred_value[0] if pred_value.size > 0 else float('-inf'))
                        else:
                            action_values.append(float('-inf'))
                    except Exception as e:
                        print(f"Error predicting with model {a}: {e}")
                        action_values.append(float('-inf'))

                # DQN action selection
                state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
                dqn_action_values = self.dqn_model(state_tensor).detach().numpy().flatten()
                combined_action_values = dqn_action_values + np.array(action_values)

                # Epsilon-greedy action selection
                epsilon = 0.1  # Exploration rate
                if random.random() < epsilon:
                    action = random.choice(range(self.action_space.n))  # Explore
                else:
                    action = np.argmax(combined_action_values)  # Exploit

                # Simulate the prevention action
                if action == 0:
                    print(f"Allowing traffic: {packet.summary()}")
                elif action == 1:
                    print(f"Quarantining traffic from {packet.summary()}")
                    return
                elif action == 2:
                    print(f"Dropping traffic: {src_ip}.") 
                    return

                real_label = self.label_traffic(packet)
                next_state, reward, done, _ = self.step(action, real_label)
                self.remember(self.state, action, reward, next_state, done)
                print(f"Packet processed: {packet.summary()}, Action: {action}, Reward: {reward}")

            else:
                print(f"Non-IP packet ignored: {packet.summary()}")
        except Exception as e:
            print(f"Error processing packet: {e}")

    def label_traffic(self, packet):
        if random.random() < 0.2:  # 20% chance of being malicious
            return 1  # Malicious
        return 0  # Normal

    def print_metrics(self):
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision = precision_score(self.true_labels, self.predictions, average='binary', zero_division=0)
        recall = recall_score(self.true_labels, self.predictions, average='binary', zero_division=0)

        print(f"Total Actions: {self.total_actions}")
        print(f"Correct Predictions: {self.correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}")
        #print(f"Precision: {precision:.2f}")
        #print(f"Recall: {recall:.2f}")
        #print(f"False Positives: {self.false_positives}")
        #print(f"False Negatives: {self.false_negatives}")
        #print(f"True Positives: {self.true_positives}")
        #print(f"True Negatives: {self.true_negatives}")

# Function to generate traffic
# Function to generate traffic
def generate_traffic(num_packets):
    packets = []
    for _ in range(num_packets):
        src_ip = str(RandIP())  # Convert RandIP to string
        dst_ip = str(RandIP())  # Convert RandIP to string
        packet = Ether()/IP(src=src_ip, dst=dst_ip)/("X" * RandShort())
        packets.append(packet)
    return packets

# Function to start traffic simulation
def start_traffic_simulation(env):
    # Generate traffic
    packets = generate_traffic(200)  # Generate 200 packets
    for packet in packets:
        env.process_packet(packet)

# Main Execution Block
if __name__ == "__main__":
    env = TrafficPreventionEnv()
    env.reset()

    traffic_simulation_thread = threading.Thread(target=start_traffic_simulation, args=(env,))
    traffic_simulation_thread.start()
    traffic_simulation_thread.join()

    env.print_metrics()

# Save accuracy to a text file
    accuracy = accuracy_score(env.true_labels, env.predictions)
    with open('accuracys_200.txt', 'w') as f:
        f.write(f"Accuracy_200: {accuracy * 100:.2f}%\n")  # Save accuracy with two decimal points
    print(f"Accuracy saved to accuracy.txt")

    