# # -*- coding: utf-8 -*-
# """
# Created on Mon Apr  7 13:21:56 2025

# @author: gozalid
# """

# # import numpy as np
# # import random
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # # Environment: Simulated Grid-Based City
# # GRID_SIZE = 10  # 10x10 city grid
# # NUM_CUSTOMERS = 5  # Number of customer delivery locations
# # NUM_VEHICLES = 2  # Number of delivery vehicles

# # # Generate random customer locations
# # customer_locations = {i: (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for i in range(NUM_CUSTOMERS)}
# # depot = (0, 0)  # Vehicle depot at (0,0)

# # # Neural Network for Value Function Approximation
# # class ValueFunctionNN(nn.Module):
# #     def __init__(self):
# #         super(ValueFunctionNN, self).__init__()
# #         self.fc1 = nn.Linear(4, 16)
# #         self.fc2 = nn.Linear(16, 8)
# #         self.fc3 = nn.Linear(8, 1)

# #     def forward(self, x):
# #         x = torch.relu(self.fc1(x))
# #         x = torch.relu(self.fc2(x))
# #         return self.fc3(x)

# # value_function = ValueFunctionNN()
# # optimizer = optim.Adam(value_function.parameters(), lr=0.01)

# # # Distance function (Manhattan Distance for simplicity)
# # def distance(a, b):
# #     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# # # ADP Training Loop
# # def train_adp(num_episodes=1000):
# #     for episode in range(num_episodes):
# #         vehicle_state = {v: depot for v in range(NUM_VEHICLES)}  # Reset vehicles to depot
# #         pending_customers = set(customer_locations.keys())

# #         total_reward = 0
# #         while pending_customers:
# #             # Select a random vehicle and a random customer
# #             vehicle = random.choice(list(vehicle_state.keys()))
# #             customer = random.choice(list(pending_customers))
            
# #             # Compute state features
# #             veh_x, veh_y = vehicle_state[vehicle]
# #             cust_x, cust_y = customer_locations[customer]
# #             state = torch.tensor([veh_x, veh_y, cust_x, cust_y], dtype=torch.float32)
            
# #             # Predict value function (Q-value approximation)
# #             predicted_value = value_function(state)
            
# #             # Compute reward (negative travel time)
# #             reward = -distance(vehicle_state[vehicle], customer_locations[customer])
# #             total_reward += reward
            
# #             # Update vehicle state
# #             vehicle_state[vehicle] = customer_locations[customer]
# #             pending_customers.remove(customer)

# #             # Compute loss and update value function
# #             loss = (reward + predicted_value) ** 2
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
        
# #         if episode % 100 == 0:
# #             print(f"Episode {episode}: Total Reward = {total_reward}")

# # # Train ADP
# # train_adp(num_episodes=1000)

# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from queue import PriorityQueue
# from typing import Dict
# import ast

# # Environment Parameters
# GRID_SIZE = 100
# NUM_BUSES = 10  
# BATTERY_CAPACITY = 100  
# CHARGING_STATIONS = [(3, 3), (7, 7)]  
# DEPOTS = [(0, 0), (9, 9)]  

# # Traffic congestion model (simple)
# def traffic_delay(location):
#     # Simulate congestion with a delay factor
#     congestion_factor = random.uniform(1, 2)  # Random congestion factor
#     return congestion_factor

# # Demand prediction model (simple)
# def predict_demand(current_time):
#     # Simple demand prediction based on time of day
#     if 8 <= current_time <= 9 or 17 <= current_time <= 18:  # Peak hours
#         return random.randint(3, 5)  # High demand during peak hours
#     else:
#         return random.randint(1, 3)  # Low demand during off-peak hours
    
# def dict_to_str(state: Dict) -> str:
#     """Convert dictionary to string key for value function."""
#     return str(list(state.items()))

# def convert_dict(keystr: str) -> Dict:
#     """Convert the key back to dictionary"""
#     convert = ast.literal_eval(keystr)
#     return dict(convert)

# # Neural Network for Value Function Approximation
# class ValueFunctionNN(nn.Module):
#     def __init__(self):
#         super(ValueFunctionNN, self).__init__()
#         self.fc1 = nn.Linear(8, 16)  
#         self.fc2 = nn.Linear(16, 8)
#         self.fc3 = nn.Linear(8, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# value_function = ValueFunctionNN()
# optimizer = optim.Adam(value_function.parameters(), lr=0.01)

# # Distance function
# def distance(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# # Battery consumption function
# def battery_consumption(start, end):
#     return distance(start, end) * 1  

# # Generate real-time trip requests
# def generate_trip():
#     start = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
#     end = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
#     return {"start": start, "end": end, "request_time": random.randint(0, 50)}

# # ADP Training with Traffic & Demand Prediction
# def train_adp(num_episodes=500):
#     reward_hist = []
#     for episode in range(num_episodes):
#         bus_state = {v: {"location": random.choice(DEPOTS), "battery": BATTERY_CAPACITY} for v in range(NUM_BUSES)}
#         trip_queue = PriorityQueue()
        
#         # Generate random trip requests
#         for _ in range(5):
#             trip = generate_trip()
#             trip_queue.put((trip["request_time"], dict_to_str(trip)))
#         total_reward = 0
#         current_time = 0  # Track time for demand prediction
        
#         while not trip_queue.empty():
#             _, trip = trip_queue.get()
#             trip = convert_dict(trip)
#             bus = random.choice(list(bus_state.keys()))

#             bus_x, bus_y = bus_state[bus]["location"]
#             trip_x, trip_y = trip["start"]
#             battery = bus_state[bus]["battery"]
            
#             # Predict traffic congestion
#             congestion_factor = traffic_delay(bus_state[bus]["location"])
            
#             # Predict future demand
#             demand = predict_demand(current_time)
            
#             state = torch.tensor([bus_x, bus_y, trip_x, trip_y, battery, trip["request_time"], congestion_factor, demand], dtype=torch.float32)

#             predicted_value = value_function(state)

#             # Apply congestion delay to travel time
#             travel_time = -distance(bus_state[bus]["location"], trip["start"]) * congestion_factor
#             battery_used = battery_consumption(bus_state[bus]["location"], trip["start"])
#             battery_penalty = -20 if battery_used > bus_state[bus]["battery"] else 0  
#             reward = travel_time + battery_penalty
#             total_reward += reward
#             reward_hist.append(total_reward)
#             # Update bus state
#             bus_state[bus]["location"] = trip["end"]
#             bus_state[bus]["battery"] -= battery_used

#             # If battery is low, charge at the nearest station
#             if bus_state[bus]["battery"] < 20:
#                 nearest_charging = min(CHARGING_STATIONS, key=lambda x: distance(bus_state[bus]["location"], x))
#                 bus_state[bus]["location"] = nearest_charging
#                 bus_state[bus]["battery"] = BATTERY_CAPACITY  

#             # Loss and backpropagation
#             loss = (reward + predicted_value) ** 2
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             current_time += 1  # Increment time after each trip

#         if episode % 100 == 0:
#             print(f"Episode {episode}: Total Reward = {total_reward}")
#     return reward_hist

# # Train Real-Time ADP Model with Traffic & Demand
# r_hist = train_adp(num_episodes=500)

# # Save the trained model
# torch.save(value_function.state_dict(), 'trained_value_function.pth')

# # Load the trained model
# value_function = ValueFunctionNN()
# value_function.load_state_dict(torch.load('trained_value_function.pth'))
# value_function.eval()  # Set the model to evaluation mode

# import matplotlib.pyplot as plt

# # Train the model and get the reward history
# reward_history = train_adp(num_episodes=500)

# # Plot the total reward per episode
# plt.figure(figsize=(10, 6))
# plt.plot(reward_history)
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.title('Total Reward over Training Episodes')
# plt.grid(True)
# plt.show()

# # def visualize_schedule(bus_state, trip_queue):
# #     # This function will simulate the bus schedules and visualize the results
# #     plt.figure(figsize=(15, 10))
    
# #     # Plot logic here (e.g., for each bus, plot trips, idle time, and charging time)
    
# #     plt.xlabel('Time (minutes)')
# #     plt.ylabel('Bus ID')
# #     plt.title('Bus Scheduling Results')
# #     plt.grid(True)
# #     plt.show()

# # # Example of how you might call it after running the training
# # visualize_schedule(bus_state, trip_queue)


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from scipy import stats

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class Environment:
    """Environment for EV scheduling with stochastic transit delays and demand"""
    
    def __init__(self, num_buses=10, num_routes=5, max_charge=100, 
                 charge_rate=10, discharge_rate_per_km=0.5, max_delay=30):
        # Environment parameters
        self.num_buses = num_buses
        self.num_routes = num_routes
        self.max_charge = max_charge  # Maximum battery charge level (kWh)
        self.charge_rate = charge_rate  # kWh per time step when charging
        self.discharge_rate_per_km = discharge_rate_per_km  # kWh per km
        self.max_delay = max_delay  # Maximum possible delay in minutes
        
        # Route properties
        self.route_distances = np.random.uniform(5, 20, num_routes)  # km
        self.route_times = self.route_distances * 3  # minutes (assuming 20 km/h)
        
        # Bus depot locations (for simplicity, assume single depot)
        self.depot_location = np.array([0, 0])
        
        # Time-dependent traffic patterns (24 hours)
        self.traffic_patterns = self._generate_traffic_patterns()
        
        # Initialize state
        self.reset()
    
    def _generate_traffic_patterns(self):
        """Generate 24-hour traffic patterns with AM and PM peaks"""
        hours = np.arange(24)
        # Morning peak (8 AM) and evening peak (5 PM)
        pattern = 1 + 0.5 * np.exp(-0.5 * ((hours - 8) / 2) ** 2) + 0.7 * np.exp(-0.5 * ((hours - 17) / 2) ** 2)
        return pattern
    
    def reset(self):
        """Reset environment to initial state"""
        # Bus properties: [location_x, location_y, charge_level, assigned_route, status]
        # Status: 0=idle at depot, 1=in service, 2=charging
        self.buses = np.zeros((self.num_buses, 5))
        self.buses[:, 2] = np.random.uniform(0.5, 1.0, self.num_buses) * self.max_charge  # Random initial charge
        
        # Time tracking
        self.current_hour = 8  # Start at 8 AM
        self.time_step = 0
        
        # Demand and delay tracking
        self.current_demand = self._generate_demand()
        self.current_delays = self._generate_delays()
        
        return self._get_state()
    
    def _generate_demand(self):
        """Generate stochastic demand for each route based on time of day"""
        base_demand = np.random.poisson(10, self.num_routes)
        time_factor = 1 + 0.5 * np.sin(np.pi * self.current_hour / 12)
        return base_demand * time_factor
    
    def _generate_delays(self):
        """Generate stochastic delays for each route based on traffic patterns"""
        # Use log-normal distribution for delay modeling
        traffic_factor = self.traffic_patterns[int(self.current_hour)]
        mean_delays = traffic_factor * np.random.uniform(1, 5, self.num_routes)
        
        # Log-normal distribution ensures delays are positive and can have occasional large values
        delays = np.random.lognormal(mean=np.log(mean_delays), sigma=0.5)
        return np.minimum(delays, self.max_delay)  # Cap at max_delay
    
    def _get_state(self):
        """Return current state representation"""
        # Flatten bus properties
        bus_states = self.buses.flatten()
        
        # Current time (sin and cos encoding for cyclical nature)
        time_sin = np.sin(2 * np.pi * self.current_hour / 24)
        time_cos = np.cos(2 * np.pi * self.current_hour / 24)
        
        # Current demand and delays
        demand = self.current_demand / 20.0  # Normalize
        delays = self.current_delays / self.max_delay  # Normalize
        
        # Combine all state components
        state = np.concatenate([
            bus_states,
            [time_sin, time_cos],
            demand,
            delays
        ])
        
        return state
    
    def step(self, action):
        """
        Take action and return new state, reward, done
        Action is a matrix of assignments: [bus_id, route_id] or [bus_id, -1] for charging
        """
        # Process each bus assignment
        rewards = 0
        
        for bus_id, route_id in action:
            bus_id = int(bus_id)
            route_id = int(route_id)
            
            # Current bus state
            bus = self.buses[bus_id]
            charge_level = bus[2]
            
            if route_id == -1:  # Charging action
                # Set bus to charging status
                self.buses[bus_id, 3] = -1
                self.buses[bus_id, 4] = 2
                
                # Increase charge level
                new_charge = min(charge_level + self.charge_rate, self.max_charge)
                self.buses[bus_id, 2] = new_charge
                
                # Small penalty for charging instead of serving
                rewards -= 1
                
            elif 0 <= route_id < self.num_routes:  # Assign to route
                route_distance = self.route_distances[route_id]
                required_charge = route_distance * self.discharge_rate_per_km
                
                # Check if bus has enough charge
                if charge_level >= required_charge:
                    # Set bus to in-service status
                    self.buses[bus_id, 3] = route_id
                    self.buses[bus_id, 4] = 1
                    
                    # Decrease charge level
                    self.buses[bus_id, 2] = charge_level - required_charge
                    
                    # Calculate service reward based on demand and delays
                    demand_served = min(1.0, self.current_demand[route_id] / 10.0)
                    delay_penalty = self.current_delays[route_id] / self.max_delay
                    
                    # Reward: balance between serving demand and avoiding delays
                    route_reward = 10 * demand_served - 5 * delay_penalty
                    rewards += route_reward
                else:
                    # Not enough charge - bus stays idle with penalty
                    self.buses[bus_id, 4] = 0
                    rewards -= 5
            
            else:  # Invalid route, bus stays idle
                self.buses[bus_id, 4] = 0
                rewards -= 1
        
        # Update time
        self.time_step += 1
        self.current_hour = (self.current_hour + 1) % 24
        
        # Update stochastic elements
        self.current_demand = self._generate_demand()
        self.current_delays = self._generate_delays()
        
        # Check if episode is done (e.g., after 24 hours)
        done = self.time_step >= 24
        
        return self._get_state(), rewards, done

class ADPAgent:
    """Approximate Dynamic Programming agent using neural networks"""
    
    def __init__(self, state_size, action_size, num_buses, num_routes):
        self.state_size = state_size
        self.action_size = action_size
        self.num_buses = num_buses
        self.num_routes = num_routes
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Create value function approximator (Q-network)
        self.model = self._build_model()
        
        # Target network for stability
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Neural network to approximate Q-value function"""
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            # Explore: random assignment of buses to routes or charging
            actions = []
            for i in range(self.num_buses):
                # Each bus can be assigned to a route (0 to num_routes-1) or to charging (-1)
                route = random.randint(-1, self.num_routes - 1)
                actions.append([i, route])
            return np.array(actions)
        
        # Exploit: use model to predict best assignments
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        
        # Convert 1D action values to bus-route assignments
        actions = self._decode_action(act_values[0])
        return actions
    
    def _decode_action(self, action_values):
        """Convert action values to bus-route assignments"""
        # Reshape action values to a matrix [buses × (routes + charging)]
        action_matrix = action_values.reshape(self.num_buses, self.num_routes + 1)
        
        assignments = []
        for bus_id in range(self.num_buses):
            # Get best action for this bus (either a route or charging=-1)
            best_action = np.argmax(action_matrix[bus_id])
            if best_action == self.num_routes:  # Last option is charging
                route_id = -1
            else:
                route_id = best_action
            
            assignments.append([bus_id, route_id])
        
        return np.array(assignments)
    
    def _encode_action(self, assignments):
        """Convert bus-route assignments to flat action index"""
        action_idx = 0
        for bus_id, route_id in assignments:
            # Convert route_id (-1 for charging) to 0...num_routes index
            if route_id == -1:
                route_idx = self.num_routes
            else:
                route_idx = route_id
            
            # Encode the assignment
            position = bus_id * (self.num_routes + 1) + route_idx
            action_idx |= (1 << position)
        
        return action_idx
    
    def replay(self, batch_size):
        """Train model using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Calculate future Q-value using target network
                next_q_values = self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target = reward + self.gamma * np.max(next_q_values)
            
            # Get current Q-values
            current_q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            
            # Update the Q-value for the action taken
            action_idx = self._encode_action(action)
            current_q_values[0][action_idx] = target
            
            # Train the model
            self.model.fit(state.reshape(1, -1), current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent():
    """Train the ADP agent"""
    # Environment and agent setup
    num_buses = 5
    num_routes = 3
    env = Environment(num_buses=num_buses, num_routes=num_routes)
    
    # Calculate state and action sizes
    state_size = num_buses * 5 + 2 + num_routes * 2  # bus states + time encoding + demand + delays
    action_size = 2 ** (num_buses * (num_routes + 1))  # Binary encoding of all possible assignments
    
    agent = ADPAgent(state_size, action_size, num_buses, num_routes)
    
    # Training parameters
    episodes = 500
    batch_size = 32
    
    # Tracking metrics
    rewards_history = []
    avg_rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
        
        # Train the agent
        agent.replay(batch_size)
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Track rewards
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
        avg_rewards_history.append(avg_reward)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}, Avg Reward: {avg_reward}, Epsilon: {agent.epsilon:.2f}")
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label='Reward')
    plt.plot(avg_rewards_history, label='Avg Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    return env, agent, rewards_history

def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate the trained agent"""
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Use greedy policy (no exploration)
            original_epsilon = agent.epsilon
            agent.epsilon = 0
            action = agent.act(state)
            agent.epsilon = original_epsilon
            
            # Take action
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Evaluation Episode {episode}: Total Reward = {total_reward}")
    
    print(f"Average Evaluation Reward: {np.mean(total_rewards)}")
    
    return total_rewards

# Run the training process
if __name__ == "__main__":
    env, agent, rewards_history = train_agent()
    eval_rewards = evaluate_agent(env, agent)
    
    # Additional analysis
    print("Final agent performance statistics:")
    print(f"- Mean reward: {np.mean(rewards_history[-100:])}")
    print(f"- Std dev reward: {np.std(rewards_history[-100:])}")
    print(f"- Min reward: {np.min(rewards_history[-100:])}")
    print(f"- Max reward: {np.max(rewards_history[-100:])}")
    
    # Save the trained model
    agent.model.save("ev_scheduling_adp_model.h5")
    print("Model saved to ev_scheduling_adp_model.h5")