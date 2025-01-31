import gym #environment for rl agents
from gym import spaces #used for action spaces
import numpy as np #numerical computation
import random #generating random numbers
from collections import deque #Used for appending and popping elements
from sklearn.svm import SVR #For regression
from sklearn.preprocessing import MinMaxScaler #normalize input features
from scapy.all import sniff, IP, Ether, sendp #real time extraction
import time #sleep and time manupulation
import threading
import torch #neural network
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, accuracy_score
from gtts import gTTS
from playsound import playsound
from pygame import mixer



def announce(text, filename):
    #tts = gTTS(text=text, lang='en')
    #tts.save(filename)
    #playsound(filename)
    print(text)
    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()

# Define DQN Model
class DQN(nn.Module): #custom class for dqn model
    def __init__(self, input_dim, output_dim): #[1200, 1, 500, 80] #packet size #packet type #traffic rate #destination port
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) #input*W + b
        self.fc2 = nn.Linear(128, 256) #
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)
        

    def forward(self, x):
         x = torch.relu(self.fc1(x)) #negitive values to 0
         x = torch.relu(self.fc2(x))
         x = torch.relu(self.fc3(x))
         x = torch.relu(self.fc4(x))
         return self.fc5(x)
        

# Initialize the TrafficPrevention class
class TrafficPreventionEnv(gym.Env):
    def __init__(self, rate_limit=100, quarantine_duration=10):
        super(TrafficPreventionEnv, self).__init__()

        # Define action space: 0 = allow, 1 = drop, 2 = quarantine
        self.action_space = spaces.Discrete(3)

        #11 features as float32
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32) #features used for evaluating.

        self.state = None #current state
        self.models = [SVR() for _ in range(self.action_space.n)] #support vector environment
        self.memory = deque(maxlen=2000) #storing previous experiences.

        # DQN instance initialization
        self.dqn_model = DQN(11, 3)
        self.target_dqn_model = DQN(11, 3)
        self.target_update_freq = 100
        self.steps = 0
        #11 features are packet size, source ip, destination ip, source port, destination port, protocol type, syn flag, ack flag
        # packet count from source ip, #time since last packet from source, #Quarantine status 
        self.dqn_optimizer = optim.Adam(self.dqn_model.parameters(), lr=0.001)  #weights and biases
        self.dqn_memory = deque(maxlen=2000) #agent's experiences

        # Rate limiting threshold
        self.rate_limit = rate_limit 
        self.ip_packet_count = {} 
        self.burst_threshold = 10

        # Normalizer
        self.scaler = MinMaxScaler()
        self.scaler_fitted = False 

        # Packet counting
        self.packet_count = 0 
        self.start_time = time.time()

        # Metrics for evaluation
        self.total_actions = 0 
        self.correct_predictions = 0 
        self.false_positives = 0 
        self.false_negatives = 0 
        self.true_positives = 0 
        self.true_negatives = 0 
        self.predictions = [] 
        self.true_labels = [] 

        # Quarantine settings
        self.quarantine_duration = quarantine_duration  # Duration to quarantine
        self.quarantine_timestamps = {}  
#This blocks resets the environment state
    def reset(self):
        self.packet_count = 0
        self.start_time = time.time() #updates the start time to the current time.
        self.ip_packet_count.clear() #clears stored counts
        self.state = np.zeros(11, dtype=np.float32) #sets the stateof an array to 0 with 11 features
        return self.get_observation() #retrve this state

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
            reward = -5
            if real_label == 0:  # False positive
                self.false_positives += 1
        else:  # Quarantine
            reward = -2
            if real_label == 0:  # False negative
                self.false_negatives += 1

        next_state = self.get_observation() #next state

        # Fit the model only if there are sufficient samples
        if len(self.memory) > 32:
            self.replay()

        state_tensor = torch.FloatTensor(self.state).unsqueeze(0) #batch dimensions
        dqn_action_values = self.dqn_model(state_tensor).detach().numpy().flatten() 

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        target_action_values = self.target_dqn_model(next_state_tensor).detach().numpy().flatten() 

        target_q_value = reward + 0.99 * np.max(target_action_values)
        y = np.array([target_q_value]).astype(np.float32)

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_dqn_model.load_state_dict(self.dqn_model.state_dict())  

        return next_state, reward, False, {}

    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done)) #feedback

#used for training the models based on past experiences
    def replay(self):
        batch_size = 64
        if len(self.memory) < batch_size: #checks if there are 32 experiences
            return #else exit quickly

        minibatch = random.sample(self.memory, batch_size) #selects that random 32 experiences.

        for state, action, reward, next_state, done in minibatch: #iterates over these experiences.
            X = np.array([state]).astype(np.float32) 
            y = np.array([reward]).astype(np.float32) 

            try:
                self.models[action].fit(X.reshape(-1, 11), y)  # Reshape for SVR input
            except Exception as e:
                print(f"Error fitting model {action}: {e}")

    def process_packet(self, packet): #handling incoming packets
        try:
            if IP in packet: 
                src_ip = packet[IP].src #extract the source ip

                # Check if the IP is currently quarantined
                current_time = time.time()  
                if src_ip in self.quarantine_timestamps: 
                    if current_time < self.quarantine_timestamps[src_ip]:

                        print(f"Traffic from {src_ip} is currently quarantined. Dropping packet.")
                        return
                    else:
                        # Remove IP from quarantine if duration has passed
                        del self.quarantine_timestamps[src_ip]

                # Rate limiting
                self.packet_count += 1 #increments the counter that tracks total no of packets processed
                if src_ip not in self.ip_packet_count: 
                    self.ip_packet_count[src_ip] = [0, current_time] 
   
                #This code tracks how many packets come from a specific source IP address in a short timeframe
                if current_time - self.ip_packet_count[src_ip][1] < 5: 
                    self.ip_packet_count[src_ip][0] += 1
                else:
                    self.ip_packet_count[src_ip] = [1, current_time] #One packet has been recived

                if self.ip_packet_count[src_ip][0] > self.rate_limit:
                    print(f"Rate limit exceeded for {src_ip}. Dropping packet.")
                    return

                if self.ip_packet_count[src_ip][0] > self.burst_threshold:
                    print(f"Burst detected from {src_ip}. Quarantining traffic.")
                    self.quarantine_timestamps[src_ip] = current_time + self.quarantine_duration  # Set quarantine duration
                    return

                # Feature extraction
                features = [
                    len(packet), 
                    *list(map(int, src_ip.split('.'))), #splits the source ip
                    *list(map(int, packet[IP].dst.split('.'))), #splits the destination ip
                    packet[IP].proto, #protocol number used in the packet.
                    0  # Placeholder for additional features
                ]
#validation check to ensure that the number of extracted features matches the expected number of features
                if len(features) != 11:
                    print(f"Feature count mismatch: {len(features)} features extracted.")
                    return

                if not self.scaler_fitted: 
                    self.scaler.fit([features]) #if not, the line fits the data in feaures
                    self.scaler_fitted = True 
                    
                else:
                    scaled_features = self.scaler.transform([features])
                    self.state = scaled_features[0]

                # Make predictions using SVR models
                action_values = [] 
                for a in range(self.action_space.n):
                    try:
                        # Check if the model is fitted before predicting
                        if hasattr(self.models[a], 'support_'):  # Ensure the model is fitted
                            action_values.append(self.models[a].predict(self.state.reshape(1, -1))[0]) 
                            #the code makes it's prediction.
                        else:
                            action_values.append(float('-inf'))  # Not fitted
                    except Exception as e:
                        print(f"Error predicting with model {a}: {e}")
                        action_values.append(float('-inf'))

                # Use DQN for action decision
                state_tensor = torch.FloatTensor(self.state).unsqueeze(0) #convert numPy to torch
                dqn_action_values = self.dqn_model(state_tensor).detach().numpy().flatten()

                # Stack SVR and DQN predictions
                combined_action_values = dqn_action_values + np.array(action_values)

                action = np.argmax(combined_action_values)

                # Simulate the prevention action
                if action == 0:
                    print(f"Allowing traffic: {packet.summary()}")
                elif action == 1:
                    print(f"Dropping traffic: {packet.summary()}")
                    return
                elif action == 2:
                    print(f"Quarantining traffic from {src_ip}.")
                    return

                # Determine real label based on more sophisticated logic
                real_label = self.label_traffic(packet) #says wheather the upcoming packet is normal or mal
                next_state, reward, done, _ = self.step(action, real_label) #learns from the prev steps
                self.remember(self.state, action, reward, next_state, done)
                self.dqn_memory.append((self.state, action, reward, next_state, done))  # Store in DQN memory
                print(f"Packet processed: {packet.summary()}, Action: {action}, Reward: {reward}")
            else:
                print(f"Non-IP packet ignored: {packet.summary()}")
        except Exception as e:
            print(f"Error processing packet: {e}")

    def label_traffic(self, packet):
        # Implement logic to determine if the traffic is normal or malicious
        if random.random() < 0.2:  # 20% chance of being malicious
            return 1  # Malicious
        return 0  # Normal

    def print_metrics(self):
        accuracy = accuracy_score(self.true_labels, self.predictions)
        precision = precision_score(self.true_labels, self.predictions, average='binary', zero_division=0)
        recall = recall_score(self.true_labels, self.predictions, average='binary', zero_division=0)

        print(f"Total Actions: {self.total_actions}")
        print(f"Correct Predictions: {self.correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
        print(f"False Positives: {self.false_positives}, False Negatives: {self.false_negatives}")
        print(f"True Positives: {self.true_positives}, True Negatives: {self.true_negatives}")

def generate_traffic():#simulates network traffic
    while True:
        
        source_ips = ["192.168.56.1", "172.18.114.8"]  
        destination_ips = ["192.168.5.1", "192.168.47.1"] 

        
        
        packet = Ether()/IP(src=src_ip, dst=dst_ip)/("X" * random.randint(50, 1500))  
        sendp(packet, iface="eth0")  #generated packet will be sent to eth0 

        time.sleep(random.uniform(0.1, 1)) #pause the execution of the script for a random amount of time between 0.1 and 1 second.      

# Sniffing thread
def start_sniffing(env, stop_event):
    print("Starting packet sniffing...")
    sniff(prn=env.process_packet, store=0, stop_filter=lambda x: stop_event.is_set())

# Main function
if __name__ == "__main__":
    announce("Intrusion detection started", "intrusion_start.mp3")
    env = TrafficPreventionEnv()
    stop_event = threading.Event()

    # Start the packet sniffing in a separate thread
    sniff_thread = threading.Thread(target=start_sniffing, args=(env, stop_event))
    sniff_thread.start()

    # Run for 60 seconds
    time.sleep(60)

    # Stop the sniffing thread
    print("Stopping packet sniffing...")
    stop_event.set()  # Set the event to stop sniffing

    sniff_thread.join(timeout=1)  # Wait for the thread to finish

    # Print evaluation metrics
    env.print_metrics()

    # Save accuracy to a text file
    accuracy = accuracy_score(env.true_labels, env.predictions)
    with open('accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")  # Save accuracy with two decimal points
    print(f"Accuracy saved to accuracy.txt")

    announce("Intrusion detection ended", "intrusion_end.mp3")



    
