# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 20:38:49 2025

@author: gozalid
"""

import heapq
from collections import defaultdict, deque
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import time
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class ScheduleState:
    """Immutable state representation for better caching and comparison"""
    path: tuple
    total_duration: float
    battery_level: float
    visited_trips: frozenset
    last_recharge_pos: int
    charging_cycle: int
    total_idle_time: float
    
    def __hash__(self):
        return hash((self.path, self.total_duration, self.battery_level, self.charging_cycle, self.total_idle_time, self.visited_trips))

class ImprovedEVScheduler:
    """
    Improved Electric Vehicle Scheduler with optimizations for large-scale datasets
    """
    
    def __init__(self, max_iterations=50, d_max=350, charging_efficiency=0.9):
        self.max_iterations = max_iterations
        self.D_MAX = d_max
        self.charging_efficiency = charging_efficiency
        self.T_MAX = 1355
        
        # Caching and memoization
        self.state_cache = {}
        self.reachability_cache = {}
        self.distance_cache = {}
        
        # Performance tracking
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'pruned_branches': 0,
            'total_evaluations': 0
        }
    
    def constructive_scheduler_improved(self, df2, gamma_lists, phi_lists, charging_stations):
        """
        Main scheduler with improvements for efficiency and scalability
        """
        print("Starting improved constructive scheduling...")
        start_time = time.time()
        
        # Preprocess and create efficient data structures
        gamma, phi = self._preprocess_data(gamma_lists, phi_lists, df2)
        #print(len(gamma))
        #print(len(phi))
        # Create spatial and temporal indices
        spatial_index = self._build_spatial_index(gamma)
        temporal_index = self._build_temporal_index(df2)
        # print(f"spatial = {spatial_index}")
        # print(f"temporal = {temporal_index}")
        schedules_tab = {}
        schedules = []
        
        # Get trip IDs more efficiently
        trip_ids = df2.loc[df2.type == 'trip', 'ID'].tolist()
        sorted_trips = df2[df2['ID'].isin(trip_ids)].sort_values('dep_time').ID.tolist()
        
        # Use set for faster lookups
        unassigned_trips = set(sorted_trips)
        
        vehicle_count = 0
        
        while unassigned_trips:
            vehicle_count += 1
            print(f"Processing vehicle {vehicle_count}, remaining trips: {len(unassigned_trips)}")
            
            # Select best starting trip using heuristic
            start_trip = self._select_best_start_trip(unassigned_trips, temporal_index, spatial_index)
            start_node = (0, start_trip)
            # Use improved search with pruning and caching
            total_completed, best_path = self._improved_constructive_search(
                start_node, gamma, phi, charging_stations, 
                spatial_index, temporal_index, unassigned_trips, vehicle_count
            )
            
            if best_path:
                schedules_tab[tuple(best_path)] = total_completed
                schedules.append(best_path)
                
                # Update unassigned trips efficiently
                completed_trips = self._extract_trip_ids(best_path)
                unassigned_trips -= completed_trips
                
                # Update gamma and phi efficiently
                gamma, phi = self._update_arc_sets(gamma, phi, completed_trips)
            else:
                # Fallback: assign single trip if no path found
                single_trip = unassigned_trips.pop()
                single_path = [(0, single_trip), (single_trip, 0)]
                schedules_tab[tuple(single_path)] = 1
                schedules.append(single_path)
        
        end_time = time.time()
        print(f"Scheduling completed in {end_time - start_time:.2f} seconds")
        print(f"Total vehicles used: {vehicle_count}")
        print(f"Cache performance: {self.stats['cache_hits']} hits, {self.stats['cache_misses']} misses")
        
        return schedules_tab, schedules
    
    def _preprocess_data(self, gamma_lists, phi_lists, df2):
        """Preprocess data for efficient access"""
        gamma = gamma_lists.copy()
        phi = phi_lists.copy()
        
        # Add battery consumption estimates to arcs
        for arc, data in gamma.items():
            if 'battery_consumption' not in data:
                # Estimate battery consumption based on duration (you can improve this)
                data['battery_consumption'] = data['duration']
        
        for arc, data in phi.items():
            if 'battery_consumption' not in data:
                data['battery_consumption'] = data['duration']  # Charging arcs consume less
        
        return gamma, phi
    
    def _build_spatial_index(self, gamma):
        """Build spatial index for efficient neighbor lookup"""
        spatial_index = defaultdict(list)
        for (origin, dest), data in gamma.items():
            spatial_index[origin].append((dest, data))
        
        # Sort by duration for each origin
        for origin in spatial_index:
            spatial_index[origin].sort(key=lambda x: x[1]['duration'])
        
        return spatial_index
    
    def _build_temporal_index(self, df2):
        """Build temporal index for time-based scheduling"""
        temporal_index = {}
        trip_data = df2[df2.type == 'trip']
        
        for _, row in trip_data.iterrows():
            temporal_index[row['ID']] = {
                'dep_time': row['dep_time'],
                'arr_time': row.get('arr_time', row['dep_time'] + 30),  # Default 30 min
                'priority': row.get('priority', 1)
            }
        
        return temporal_index
    
    def _select_best_start_trip(self, unassigned_trips, temporal_index, spatial_index):
        """Select best starting trip using heuristic"""
        if not unassigned_trips:
            return None
        
        # Score trips based on multiple criteria
        trip_scores = {}
        
        for trip_id in unassigned_trips:
            score = 0
            
            # Temporal urgency (earlier trips get higher priority)
            dep_time = temporal_index.get(trip_id, {}).get('dep_time', float('inf'))
            score += max(0, self.D_MAX - dep_time)  # Earlier = higher score
            
            # Connectivity (trips with more outgoing connections)
            outgoing_connections = len(spatial_index.get(trip_id, []))
            score += outgoing_connections * 10
            
            # Priority if available
            priority = temporal_index.get(trip_id, {}).get('priority', 1)
            score += priority * 50
            
            trip_scores[trip_id] = score
        # print(f"TRIP = {trip_scores}")
        # print(f"MAX = {max(trip_scores.items(), key=lambda x: x[1])}")
        # Return trip with highest score
        return max(trip_scores.items(), key=lambda x: x[1])[0]
    
    def _improved_constructive_search(self, start_node, gamma, phi, charging_stations, 
                                    spatial_index, temporal_index, unassigned_trips, vehicle_count):
        """
        Improved search with pruning, caching, and better battery management
        """
        # Use priority queue for best-first search
        initial_state = ScheduleState(
            path=(start_node,),
            total_duration=gamma.get(start_node, {}).get('duration', 0),
            battery_level=self.D_MAX,
            visited_trips=frozenset([start_node[1]]),
            last_recharge_pos=0,
            charging_cycle=1,
            total_idle_time=0
        )
        
        # Priority queue: (negative_score, state_id, state)
        pq = [(-self._evaluate_state(initial_state, unassigned_trips, vehicle_count), 0, initial_state)]
        state_counter = 1
        best_state = initial_state
        best_score = self._evaluate_state(initial_state, unassigned_trips, vehicle_count)
        
        iterations = 0
        max_queue_size = 1000  # Limit memory usage
        #print(f"GAMMA = {gamma}")
        while pq and iterations < self.max_iterations:
            iterations += 1
            
            if len(pq) > max_queue_size:
                # Keep only best states to manage memory
                pq = heapq.nsmallest(max_queue_size // 2, pq)
                heapq.heapify(pq)
            
            neg_score, _, current_state = heapq.heappop(pq)
            current_score = -neg_score
            
            # Check if this is the best state so far
            if current_score > best_score:
                best_state = current_state
                best_score = current_score
            # Generate successors
            successors = self._generate_successors(
                current_state, gamma, phi, charging_stations, 
                spatial_index, temporal_index, unassigned_trips
            )
            # print(f"SUCCESSORS = {successors}")
            
            for successor in successors:
                if self._is_feasible_state(successor):
                    successor_score = self._evaluate_state(successor, unassigned_trips, vehicle_count)
                    heapq.heappush(pq, (-successor_score, state_counter, successor))
                    state_counter += 1
        
        # Convert best state back to path format
        return len(best_state.visited_trips), list(best_state.path)
    
    def _generate_successors(self, state, gamma, phi, charging_stations, 
                           spatial_index, temporal_index, unassigned_trips):
        """Generate successor states efficiently"""
        successors = []
        current_location = state.path[-1][1] if state.path else 0
        
        # Check if charging is needed
        # if state.battery_level < self.D_MAX * 0.2:  # 20% threshold
        # if state.total_duration >= self.D_MAX * state.charging_cycle:
        possible_recharge = self._generate_charging_successors(state, phi, charging_stations)
        # if current_location[0] == 75:
        #     print(f"CHARGING POSSIBLE?? {possible_recharge}")
        if possible_recharge != None and possible_recharge not in successors:
            successors.append(possible_recharge)
        # print(f"CHARGING ADDED = {successors}")
        # Generate trip successors
        trip_successors = self._generate_trip_successors(
            state, gamma, phi, charging_stations, spatial_index, temporal_index, unassigned_trips
        )
        successors.extend(trip_successors)
        # print(f"TRIP ADDED = {successors}")
        # Generate depot return if needed
        if self._should_return_to_depot(state):
            depot_successor = self._generate_depot_successor(state)
            if depot_successor:
                successors.append(depot_successor)
        # print(f"TRIPS ADDED = {successors}")
        return successors
    # def _is_feasible_recharge(self, state, phi, charging_stations):
    #     successors = []
    #     current_location = state.path[-1][1]
        
    def _generate_charging_successors(self, state, phi, charging_stations):
        """Generate states for charging options"""
        successors = None
        current_location = state.path[-1]
        if current_location[0] == 11:
            print("-"*100)
            for cs_arc in phi:
                if cs_arc[0] == 11:
                    print(f"ARC = {cs_arc}")
            print("-"*100)
        for station in charging_stations:
            arc = (current_location[0], current_location[1], station)
            if arc in phi and arc not in state.path:
                # new_path = state.path + (arc,)
                new_path = state.path[:-1] + (arc,)
                # new_battery = min(self.D_MAX, state.battery_level + 
                #                 self.D_MAX * self.charging_efficiency)
                new_battery = self.D_MAX
                
                successor = ScheduleState(
                    path=new_path,
                    total_duration=state.total_duration + phi[arc]['duration'],
                    battery_level=new_battery,
                    visited_trips=state.visited_trips,
                    last_recharge_pos=len(new_path) - 1,
                    charging_cycle=state.charging_cycle+1,
                    total_idle_time=state.total_idle_time
                )
                successors = successor
        # print(f"CHARGING = {successors}")
        return successors
    
    def _calculate_idle(self, arc, path, phi, temporal_index):
        if arc[1] in temporal_index and arc[0] in temporal_index:
            # print(f"PATH = {path}")
            if len(path) == 2:
                return temporal_index.get(arc[1]).get("dep_time") - temporal_index.get(arc[0], 0).get("arr_time") 
            else:
                # print(f"DURRRR!!! {phi[path].get('duration')}")
                return temporal_index.get(arc[1], 0).get("dep_time") - temporal_index.get(arc[0], 0).get("arr_time") + (phi[path].get("duration") - 50)
        else:
            # print(f"PATH = {path}")
            return 0
    
    def _generate_trip_successors(self, state, gamma, phi, charging_stations, spatial_index, temporal_index, unassigned_trips):
        """Generate states for trip options"""
        successors = []
        current_location = state.path[-1][1]
        
        # Get available trips from current location
        available_trips = spatial_index.get(current_location, [])
        
        for dest, arc_data in available_trips:
            if dest in unassigned_trips and dest not in state.visited_trips:
                arc = (current_location, dest)
                idle_arc = self._calculate_idle(arc, state.path[-1], phi, temporal_index)
                battery_needed = arc_data.get('battery_consumption', arc_data['duration'])
                # print(f"Cost of trip {dest} = {battery_needed}\nbattery level = {state.battery_level}")
                # Check if we have enough battery
                if state.battery_level >= battery_needed:
                # if state.total_duration <= self.D_MAX * state.charging_cycle:
                    new_path = state.path + (arc,)
                    new_visited = state.visited_trips | {dest}
                    
                    successor = ScheduleState(
                        path=new_path,
                        total_duration=state.total_duration + arc_data['duration'],
                        battery_level=state.battery_level - battery_needed,
                        visited_trips=new_visited,
                        last_recharge_pos=state.last_recharge_pos,
                        charging_cycle=state.charging_cycle, 
                        total_idle_time=idle_arc
                    )
                    successors.append(successor)
                # else:
                #     # Generate charging successors
                #     charging_successors = self._generate_charging_successors(
                #         state, phi, charging_stations
                #     )
                #     print(f"CURRENT BATTERY = {state.battery_level}\nCHARGING = {charging_successors}")
                #     if charging_successors:
                #         successors.append(charging_successors)
        # print(successors)
        return successors
    
    def _generate_depot_successor(self, state):
        """Generate depot return successor"""
        current_location = state.path[-1][1]
        if current_location != 0:
            depot_arc = (current_location, 0)
            new_path = state.path + (depot_arc,)
            
            return ScheduleState(
                path=new_path,
                total_duration=state.total_duration,
                battery_level=state.battery_level,
                visited_trips=state.visited_trips,
                last_recharge_pos=state.last_recharge_pos,
                charging_cycle=state.charging_cycle,
                total_idle_time=state.total_idle_time
            )
        return None
    
    def _should_return_to_depot(self, state):
        """Determine if vehicle should return to depot"""
        # Return to depot if battery is very low or no more feasible trips
        return (state.battery_level < self.D_MAX * 0.1 or 
                len(state.visited_trips) > 20)  # Arbitrary limit
    
    def _is_feasible_state(self, state):
        """Check if state is feasible"""
        return (state != None and state.battery_level >= 0 and
                len(state.path) < 100 and state.charging_cycle < 4)  # Reasonable path length limit
    
    def _evaluate_state(self, state, unassigned_trips, vehicle_count):
        """Evaluate state quality"""
        score = 0
        print(f"STATE = {state}")
        # Reward for trips completed
        score += len(state.visited_trips)
        
        # Penalty for duration
        score += state.charging_cycle
        
        score -= (state.total_idle_time * 1/(vehicle_count * self.T_MAX))
        #score -= (len(unassigned_trips))
        # # Bonus for battery efficiency
        # battery_utilization = (state.battery_level) / self.D_MAX
        # score += battery_utilization * 20
        
        # # Penalty for long paths (complexity)
        # score -= len(state.path) * 0.5
        # print(f"EVALUATE STATE = {state} with score = {score}")
        return score
    
    def _extract_trip_ids(self, path):
        """Extract trip IDs from path"""
        trip_ids = set()
        for arc in path:
            if len(arc) >= 2 and arc[1] != 0:  # Not depot
                trip_ids.add(arc[1])
        return trip_ids
    
    def _update_arc_sets(self, gamma, phi, completed_trips):
        """Update arc sets by removing completed trips"""
        new_gamma = {
            arc: data for arc, data in gamma.items() 
            if arc[0] not in completed_trips and arc[1] not in completed_trips
        }
        
        new_phi = {
            arc: data for arc, data in phi.items()
            if arc[0] not in completed_trips and arc[1] not in completed_trips  
        }
        
        return new_gamma, new_phi

# Utility functions for backward compatibility
def flatten_link_names(vector_representation):
    """Flatten link names from vector representation"""
    flattened = []
    for schedule in vector_representation:
        for arc in schedule:
            if isinstance(arc, (tuple, list)) and len(arc) >= 2:
                flattened.extend([arc[0], arc[1]])
    return list(set(flattened))

def vectorSchRepresentation(schedules):
    """
    Transform (i,j)/ (i,j,k) -> [i,j]/ [i,j,k]
    """
    all_schs = []
#     print(schedules)
    for bus in schedules:
        schs = []
        for pair in bus:
            if len(pair) < 3:
                schs.append(pair[1])
            else:
                schs += [pair[2]] + [pair[1]]

        all_schs.append(schs)
    return all_schs

# Example usage and comparison
def run_comparison(df2, gamma_lists, phi_lists, charging_stations):
    """
    Run comparison between original and improved scheduler
    """
    print("="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Original scheduler (you would call your original function here)
    # original_start = time.time()
    # original_schedules_tab, original_schedules = constructiveScheduler(
    #     df2, gamma_lists, phi_lists, charging_stations
    # )
    # original_time = time.time() - original_start
    
    # Improved scheduler
    improved_start = time.time()
    scheduler = ImprovedEVScheduler(max_iterations=100)
    improved_schedules_tab, improved_schedules = scheduler.constructive_scheduler_improved(
        df2, gamma_lists, phi_lists, charging_stations
    )
    improved_time = time.time() - improved_start
    
    print(f"Improved scheduler:")
    print(f"  - Execution time: {improved_time:.2f} seconds")
    print(f"  - Schedules generated: {len(improved_schedules)}")
    print(f"  - Total trips scheduled: {sum(improved_schedules_tab.values())}")
    print(f"  - Cache performance: {scheduler.stats}")
    
    return improved_schedules_tab, improved_schedules



# # For small datasets (< 100 trips)
# scheduler = ImprovedEVScheduler(max_iterations=50, d_max=350)

# # For large datasets (1000+ trips)  
# scheduler = ImprovedEVScheduler(max_iterations=200, d_max=350)
# schedules_tab, schedules = scheduler.constructive_scheduler_improved(
#     df2, gamma_lists, phi_lists, charging_stations
# )