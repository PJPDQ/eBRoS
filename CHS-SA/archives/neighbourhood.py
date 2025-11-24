# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 09:51:52 2025

@author: gozalid
"""


# def find_best_insert(route, trip):
#     if not route:
#         return 1, 0
#     best_pos = None
#     best_cost = float('inf')
#     for pos in range(1, len(route)):
#         if route[pos] == 0:
#             continue
#         # Simple cost estimation (can be improved)
#         prev_node = route[pos - 1]
#         next_node = route[pos] if pos < len(route) else 0
#         can_reach = trip in arc_from.get(prev_node, set())
#         can_leave = next_node in arc_from.get(trip, set()) or next_node == 0
#         if can_reach and can_leave:
#     # Estimate insertion cost
#     cost = self._estimate_insertion_cost(prev_node, trip, next_node)
#     if cost < best_cost:
#         best_cost = cost
#         best_pos = pos

#     return (best_pos, best_cost) if best_pos is not None else None
# for trip in after_cs:
#     best_route_idx = None
#     best_insert_cost = float('inf')
#     for i, route in enumerate(neighbor):
#         if i == route_idx:
#             continue
#         insertion =
        
import copy
import random
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import heapq
from dataclasses import dataclass
import time
from numpy.random import rand
from tqdm import tqdm
import math
@dataclass
class NeighborhoodStats:
    """Track performance statistics for neighborhood operations"""
    operation_counts: Dict[str, int]
    operation_success: Dict[str, int]
    operation_times: Dict[str, float]
    total_calls: int
    
    def __post_init__(self):
        if not hasattr(self, 'operation_counts'):
            self.operation_counts = defaultdict(int)
            self.operation_success = defaultdict(int)
            self.operation_times = defaultdict(float)
            self.total_calls = 0

class ImprovedNeighborhoodGenerator:
    """
    Enhanced neighborhood generation with efficient operations for large-scale datasets
    """
    
    def __init__(self, trips_df, arcs, recharge_arcs, D_MAX=350, max_it=10):
        self.trips_df = trips_df
        self.arcs = arcs
        self.recharge_arcs = recharge_arcs
        self.max_it = max_it
        self.D_MAX = D_MAX
        
        # Pre-compute frequently used data structures
        self._precompute_indices()
        
        # Performance tracking
        self.stats = NeighborhoodStats(
            operation_counts=defaultdict(int),
            operation_success=defaultdict(int),
            operation_times=defaultdict(float),
            total_calls=0
        )
        
        # Adaptive weights for operation selection
        self.operation_weights = {
            'swap_recharge': 1.0,
            'swap_cs': 1.0,
            'insert_schedules': 1.0,
            'early_return': 1.0,
            # 'route_optimization': 1.0,
            'merge_routes': 1.0,
            'split_routes': 1.0,
            'relocate_trips': 1.0
        }
    
    def _precompute_indices(self):
        """Precompute frequently used indices and mappings"""
        # Trip and charging station sets
        self.trip_nodes = set(self.trips_df.loc[self.trips_df.type == 'trip', 'ID'])
        self.cs_nodes = set(self.trips_df.loc[self.trips_df.type == 'cs', 'ID'])
        self.non_trip_nodes = set(self.trips_df.loc[self.trips_df.type != 'trip', 'ID'])
        
        # Spatial connectivity mapping
        self.arc_from = defaultdict(set)
        self.arc_to = defaultdict(set)
        for (origin, dest), data in self.arcs.items():
            self.arc_from[origin].add(dest)
            self.arc_to[dest].add(origin)
        
        # Recharge arc mapping
        self.recharge_from = defaultdict(set)
        for (origin, dest, cs), data in self.recharge_arcs.items():
            self.recharge_from[(origin, cs)].add(dest)
        
        # Trip timing information
        self.trip_times = {}
        trip_data = self.trips_df[self.trips_df.type == 'trip']
        for _, row in trip_data.iterrows():
            self.trip_times[row['ID']] = {
                'dep_time': row.get('dep_time', 0),
                'duration': row.get('duration', 30)
            }
    
    def get_neighbors_enhanced(self, states: List[List], adaptive=True):
        """
        Enhanced neighbor generation with multiple sophisticated operations
        """
        start_time = time.time()
        self.stats.total_calls += 1
        
        # Select operation based on adaptive weights or randomly
        if adaptive:
            operation = self._select_operation_adaptively()
        else:
            operation = random.choice([
                'swap_recharge', 'swap_cs', 'insert_schedules', 'merge_routes', 
                "early_return", "relocate_trips", "split_routes"
                # 'early_return', 'route_optimization', 'merge_routes'
            ])
        
        print(f"Applying operation: {operation}")
        
        # Apply selected operation
        op_start = time.time()
        try:
            if operation == 'swap_recharge':
                neighbor = self._swap_recharge_enhanced(states)
            elif operation == 'swap_cs':
                neighbor = self._swap_cs_enhanced(states)
            elif operation == 'insert_schedules':
                neighbor = self._insert_schedules_enhanced(states)
            elif operation == 'early_return':
                neighbor = self._early_return_enhanced(states)
            # elif operation == 'route_optimization':
            #     neighbor = self._route_optimization(states)
            elif operation == 'merge_routes':
                neighbor = self._merge_routes(states)
            elif operation == 'split_routes':
                neighbor = self._split_routes(states)
            elif operation == 'relocate_trips':
                neighbor = self._relocate_trips(states)
            else:
                neighbor = copy.deepcopy(states)
            
            # Track success
            print(f"isvalid = {self._is_valid_solution(neighbor)}")
            if self._is_valid_solution(neighbor):
                self.stats.operation_success[operation] += 1
            else:
                neighbor = copy.deepcopy(states)
            
        except Exception as e:
            print(f"Error in {operation}: {e}")
            neighbor = copy.deepcopy(states)
        
        # Update statistics
        op_time = time.time() - op_start
        self.stats.operation_counts[operation] += 1
        self.stats.operation_times[operation] += op_time
        
        # Update adaptive weights
        if adaptive:
            self._update_operation_weights(operation, self._solution_quality(neighbor) - self._solution_quality(states))
        
        total_time = time.time() - start_time
        print(f"Operation {operation} completed in {total_time:.3f}s")
        
        return neighbor
    
    def _select_operation_adaptively(self):
        """Select operation based on historical performance"""
        operations = list(self.operation_weights.keys())
        weights = [self.operation_weights[op] for op in operations]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(operations)
        
        probabilities = [w / total_weight for w in weights]
        return np.random.choice(operations, p=probabilities)
    
    def _optimal_trip_insertion(self, neighbor):
        """
        Perform optimal trip insertion by finding unassigned trips and inserting them
        at the best possible positions across all routes, considering both cost and feasibility.
        """
        # Find all trips that should be assigned
        assigned_trips = set()
        for route in neighbor:
            assigned_trips.update(node for node in route if node in self.trip_nodes)
        
        unassigned_trips = self.trip_nodes - assigned_trips
        
        # If no unassigned trips, try to improve existing assignments
        if not unassigned_trips:
            return self._improve_existing_assignments(neighbor)
        
        # Sort unassigned trips by urgency/priority (earliest departure time first)
        unassigned_list = list(unassigned_trips)
        unassigned_list.sort(key=lambda t: self.trip_times.get(t, {}).get('dep_time', 0))
        
        # Insert each trip at its best position
        for trip in unassigned_list:
            best_route_idx = None
            best_position = None
            best_cost = float('inf')
            best_route_after_insertion = None
            
            # Evaluate insertion in each existing route
            for route_idx, route in enumerate(neighbor):
                insertion_result = self._find_best_insertion_with_charging(route, trip)
                
                if insertion_result is not None:
                    position, cost, new_route = insertion_result
                    
                    # Consider multiple factors for insertion cost
                    total_cost = self._calculate_total_insertion_cost(
                        route, new_route, cost, route_idx, len(neighbor)
                    )
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_route_idx = route_idx
                        best_position = position
                        best_route_after_insertion = new_route
            
            # Insert trip in best route or create new route
            if best_route_idx is not None and best_cost < self._new_route_threshold(neighbor):
                neighbor[best_route_idx] = best_route_after_insertion
            else:
                # Create new single-trip route if insertion cost is too high
                new_route = self._create_single_trip_route(trip)
                neighbor.append(new_route)
        
        return neighbor
    
    def _find_best_insertion_with_charging(self, route, trip):
        """
        Find best insertion position considering charging requirements and route feasibility.
        Returns (position, cost, new_route) or None if insertion is not feasible.
        """
        if not route or len(route) < 2:
            # Simple case: insert in minimal route
            return 1, 0, [0, trip, 0]
        
        best_position = None
        best_cost = float('inf')
        best_route = None
        
        # Try inserting at each valid position
        for pos in range(1, len(route)):
            if route[pos] == 0:  # Don't insert after final depot
                continue
            
            # Create temporary route with trip inserted
            temp_route = route[:pos] + [trip] + route[pos:]
            
            # Check if route needs charging station insertion
            feasible_route = self._ensure_charging_feasibility(temp_route)
            
            if feasible_route is not None:
                # Calculate insertion cost
                cost = self._calculate_route_cost_difference(route, feasible_route)
                
                if cost < best_cost:
                    best_cost = cost
                    best_position = pos
                    best_route = feasible_route
        
        return (best_position, best_cost, best_route) if best_position is not None else None
    
    def _ensure_charging_feasibility(self, route):
        """
        Ensure route is feasible by inserting charging stations where needed.
        Returns modified route or None if infeasible.
        """
        if not route or len(route) < 2:
            return route
        
        feasible_route = [route[0]]  # Start with depot
        current_battery = 350  # D_MAX battery capacity
        
        for i in range(1, len(route)):
            current_node = route[i]
            prev_node = feasible_route[-1]
            
            # Calculate required energy for this segment
            arc_key = (prev_node, current_node)
            energy_needed = self.arcs.get(arc_key, {}).get('duration', 30) #* 0.8  # Energy consumption rate
            
            # Check if charging is needed
            if current_battery + energy_needed > self.D_MAX:  # Safety margin
                # Find suitable charging station
                charging_station = self._find_charging_station(prev_node, current_node)
                
                if charging_station is None:
                    return None  # Infeasible - no charging option available
                
                # Insert charging station
                feasible_route.append(charging_station)
                current_battery = 0#350  # Recharge to full
                
                # Recalculate energy for segment from charging station
                energy_needed = self.arcs.get((charging_station, current_node), {}).get('duration', 30) #* 0.8
            
            # Check final feasibility
            if current_battery + energy_needed > self.D_MAX:
                return None  # Still infeasible
            
            feasible_route.append(current_node)
            current_battery -= energy_needed
        
        return feasible_route
    
    def _find_charging_station(self, from_node, to_node):
        """
        Find a suitable charging station between from_node and to_node.
        """
        # Get all charging stations reachable from from_node
        reachable_cs = self.recharge_from.get(from_node, set())
        
        # Filter for those that can reach to_node
        suitable_cs = []
        for cs in reachable_cs:
            if to_node in self.arc_from.get(cs, set()) or to_node == 0:
                suitable_cs.append(cs)
        
        if not suitable_cs:
            return None
        
        # Select charging station with minimum detour cost
        best_cs = None
        best_detour_cost = float('inf')
        
        for cs in suitable_cs:
            # Calculate detour cost
            direct_cost = self.arcs.get((from_node, to_node), {}).get('duration', 1000)
            detour_cost = (self.arcs.get((from_node, cs), {}).get('duration', 500) +
                          self.arcs.get((cs, to_node), {}).get('duration', 500))
            
            total_detour = detour_cost - direct_cost
            
            if total_detour < best_detour_cost:
                best_detour_cost = total_detour
                best_cs = cs
        
        return best_cs
    
    def _calculate_total_insertion_cost(self, original_route, new_route, base_cost, route_idx, total_routes):
        """
        Calculate comprehensive insertion cost considering multiple factors.
        """
        # Base insertion cost
        total_cost = base_cost
        
        # Route length penalty (prefer balanced routes)
        route_length_penalty = len(new_route) * 2
        total_cost += route_length_penalty
        
        # Vehicle utilization bonus (prefer using existing vehicles)
        if len(original_route) <= 3:  # Nearly empty route
            total_cost -= 50  # Bonus for utilizing underused vehicle
        
        # Charging station penalty (prefer routes with fewer charging stops)
        cs_count = sum(1 for node in new_route if node in self.cs_nodes)
        total_cost += cs_count * 20
        
        # Time window compatibility (if available)
        if hasattr(self, '_check_time_feasibility'):
            if not self._check_time_feasibility(new_route):
                total_cost += 200  # Heavy penalty for time infeasibility
        
        return total_cost
    
    def _improve_existing_assignments(self, neighbor):
        """
        Improve existing trip assignments when no unassigned trips exist.
        """
        if len(neighbor) < 2:
            return neighbor
        
        # Find the most promising trip to relocate
        relocation_candidates = []
        
        for route_idx, route in enumerate(neighbor):
            trip_nodes = [node for node in route if node in self.trip_nodes]
            
            for trip in trip_nodes:
                # Calculate current trip cost in its route
                current_cost = self._calculate_trip_cost_in_route(route, trip)
                
                # Find best alternative position
                best_alternative_cost = float('inf')
                best_alternative_route = None
                
                for alt_route_idx, alt_route in enumerate(neighbor):
                    if alt_route_idx == route_idx:
                        continue
                    
                    insertion = self._find_best_insertion_with_charging(alt_route, trip)
                    if insertion and insertion[1] < best_alternative_cost:
                        best_alternative_cost = insertion[1]
                        best_alternative_route = alt_route_idx
                
                # Check if relocation would be beneficial
                if best_alternative_cost < current_cost: #* 0.8:  # 20% improvement threshold
                    relocation_candidates.append((
                        trip, route_idx, best_alternative_route, 
                        current_cost - best_alternative_cost
                    ))
        
        # Apply best relocation if any good candidates exist
        if relocation_candidates:
            # Sort by improvement potential
            relocation_candidates.sort(key=lambda x: x[3], reverse=True)
            
            trip, from_route, to_route, improvement = relocation_candidates[0]
            
            # Remove trip from original route
            original_route = [node for node in neighbor[from_route] if node != trip]
            neighbor[from_route] = self._clean_route(original_route)
            
            # Insert trip in target route
            insertion = self._find_best_insertion_with_charging(neighbor[to_route], trip)
            if insertion:
                pos, cost, new_route = insertion
                neighbor[to_route] = new_route
        
        return neighbor
    
    def _calculate_trip_cost_in_route(self, route, trip):
        """Calculate the cost contribution of a specific trip in its current route."""
        trip_pos = route.index(trip)
        
        if trip_pos == 0 or trip_pos == len(route) - 1:
            return 0  # Shouldn't happen for valid routes
        
        prev_node = route[trip_pos - 1]
        next_node = route[trip_pos + 1]
        
        # Cost of current path: prev -> trip -> next
        current_cost = (self.arcs.get((prev_node, trip), {}).get('duration', 100) +
                       self.arcs.get((trip, next_node), {}).get('duration', 100))
        
        # Cost if trip were removed: prev -> next
        direct_cost = self.arcs.get((prev_node, next_node), {}).get('duration', 200)
        
        return current_cost - direct_cost
    
    def _new_route_threshold(self, neighbor):
        """Dynamic threshold for creating new routes vs inserting in existing ones."""
        # Higher threshold when we have fewer routes (prefer to fill existing routes)
        base_threshold = 150
        route_count_factor = len(neighbor) * 10
        return base_threshold + route_count_factor
    
    def _create_single_trip_route(self, trip):
        """Create a new route containing only the specified trip."""
        # Check if trip needs charging to complete
        trip_energy = self.trip_times.get(trip, {}).get('duration', 30) * 1.5  # Round trip estimation
        
        if trip_energy > 300:  # Needs charging
            # Find charging station accessible from depot
            depot_reachable_cs = self.recharge_from.get(0, set())
            suitable_cs = [cs for cs in depot_reachable_cs 
                          if trip in self.arc_from.get(cs, set())]
            
            if suitable_cs:
                charging_station = random.choice(suitable_cs)
                return [0, charging_station, trip, 0]
        
        return [0, trip, 0]
    
    def _calculate_route_cost_difference(self, original_route, new_route):
        """Calculate the cost difference between original and new route."""
        original_cost = self._estimate_route_cost(original_route)
        new_cost = self._estimate_route_cost(new_route)
        return new_cost - original_cost
    
    def _estimate_route_cost(self, route):
        """Estimate total cost/duration of a route."""
        if len(route) < 2:
            return 0
        
        total_cost = 0
        for i in range(len(route) - 1):
            arc_cost = self.arcs.get((route[i], route[i+1]), {}).get('duration', 100)
            total_cost += arc_cost
        
        # Add penalty for charging stations
        cs_penalty = sum(20 for node in route if node in self.cs_nodes)
        
        return total_cost + cs_penalty
    def _swap_recharge_enhanced(self, states):
        """Enhanced recharge swapping with better route reconstruction"""
        neighbor = copy.deepcopy(states)
        
        # Find routes with charging stations
        recharge_routes = [(i, route) for i, route in enumerate(neighbor) 
                          if any(node in self.cs_nodes for node in route)]
        
        if not recharge_routes:
            return neighbor
        
        # Select route with charging station
        route_idx, route = random.choice(recharge_routes)
        cs_positions = [i for i, node in enumerate(route) if node in self.cs_nodes]
        
        if not cs_positions:
            return neighbor
        
        cs_pos = random.choice(cs_positions)
        cs_node = route[cs_pos]
        
        # Strategy 1: Remove charging station and split route
        if random.random() < 0.5 and cs_pos > 0 and cs_pos < len(route) - 1:
            # Split route at charging station
            before_cs = route[:cs_pos]  # Return to depot
            after_cs = route[cs_pos + 1:]  # Remaining trips
            
            # Update original route
            neighbor[route_idx] = before_cs
            
            # Try to reassign remaining trips
            if after_cs and after_cs != [0]:
                self._reassign_trips_efficiently(neighbor, after_cs, route_idx)
        
        # Strategy 2: Replace charging station
        else:
            if cs_pos > 0 and cs_pos < len(route) - 1:
                prev_node = route[cs_pos - 1]
                next_node = route[cs_pos + 1]
                
                # Find alternative charging stations
                alternative_cs = [cs for cs in self.cs_nodes 
                                if cs != cs_node and cs in self.recharge_from[prev_node]]
                
                if alternative_cs:
                    new_cs = random.choice(alternative_cs)
                    neighbor[route_idx][cs_pos] = new_cs
        
        return neighbor
    
    def _swap_cs_enhanced(self, states):
        """Enhanced charging station swapping with connectivity checks"""
        neighbor = copy.deepcopy(states)
        
        # Find routes with charging stations
        cs_routes = [(i, route, [j for j, node in enumerate(route) if node in self.cs_nodes])
                    for i, route in enumerate(neighbor)]
        cs_routes = [(i, route, positions) for i, route, positions in cs_routes if positions]
        
        if not cs_routes:
            return neighbor
        
        route_idx, route, cs_positions = random.choice(cs_routes)
        cs_pos = random.choice(cs_positions)
        current_cs = route[cs_pos]
        
        # Get context nodes
        prev_node = route[cs_pos - 1] if cs_pos > 0 else 0
        next_node = route[cs_pos + 1] if cs_pos < len(route) - 1 else 0
        
        # Find compatible charging stations
        compatible_cs = []
        for cs in self.cs_nodes:
            if cs != current_cs:
                # Check connectivity
                can_reach_cs = cs in self.recharge_from.get(prev_node, set())
                can_leave_cs = next_node in self.arc_from.get(cs, set()) or next_node == 0
                
                if can_reach_cs and can_leave_cs:
                    compatible_cs.append(cs)
        
        if compatible_cs:
            new_cs = random.choice(compatible_cs)
            neighbor[route_idx][cs_pos] = new_cs
        
        return neighbor
    
    def _insert_schedules_enhanced(self, states):
        """Enhanced schedule insertion with better optimization"""
        neighbor = copy.deepcopy(states)
        
        if len(neighbor) < 2:
            return neighbor
        
        # Find shortest and longest routes for balancing
        route_lengths = [(i, len([n for n in route if n in self.trip_nodes])) 
                        for i, route in enumerate(neighbor)]
        route_lengths.sort(key=lambda x: x[1])
        
        shortest_idx, shortest_len = route_lengths[0]
        longest_idx, longest_len = route_lengths[-1]
        
        # Balance routes if difference is significant
        if longest_len - shortest_len > 2:
            return self._balance_routes(neighbor, shortest_idx, longest_idx)
        
        # Otherwise, try to insert trips optimally
        return self._optimal_trip_insertion(neighbor)
    
    def _balance_routes(self, neighbor, short_idx, long_idx):
        """Balance routes by moving trips between them"""
        short_route = neighbor[short_idx]
        long_route = neighbor[long_idx]
        
        # Find movable trips from long route
        long_trips = [node for node in long_route if node in self.trip_nodes]
        if len(long_trips) <= 1:
            return neighbor
        
        # Select trip to move (prefer middle trips to maintain connectivity)
        if len(long_trips) > 2:
            trip_to_move = long_trips[len(long_trips) // 2]
        else:
            trip_to_move = random.choice(long_trips)
        
        # Remove trip from long route
        trip_pos = long_route.index(trip_to_move)
        new_long_route = long_route[:trip_pos] + long_route[trip_pos + 1:]
        
        # Clean up consecutive depot visits
        new_long_route = self._clean_route(new_long_route)
        
        # Try to insert trip into short route
        best_insertion = self._find_best_insertion(short_route, trip_to_move)
        if best_insertion is not None:
            pos, cost = best_insertion
            new_short_route = short_route[:pos] + [trip_to_move] + short_route[pos:]
            
            neighbor[short_idx] = new_short_route
            neighbor[long_idx] = new_long_route
        
        return neighbor
    
    def _early_return_enhanced(self, states):
        """Enhanced early return with intelligent trip reassignment"""
        neighbor = copy.deepcopy(states)
        
        if not neighbor:
            return neighbor
        
        # Select route for early termination
        eligible_routes = [i for i, route in enumerate(neighbor) 
                          if len([n for n in route if n in self.trip_nodes]) > 1]
        
        if not eligible_routes:
            return neighbor
        
        route_idx = random.choice(eligible_routes)
        route = neighbor[route_idx]
        
        # Find good truncation point
        trip_positions = [i for i, node in enumerate(route) if node in self.trip_nodes]
        if len(trip_positions) < 2:
            return neighbor
        
        # Truncate at random position (but not the last trip)
        truncate_pos = random.choice(trip_positions[:-1])
        truncated_route = route[:truncate_pos + 1]
        removed_trips = [node for node in route[truncate_pos + 1:] if node in self.trip_nodes]
        
        # Update original route
        neighbor[route_idx] = truncated_route
        
        # Intelligently reassign removed trips
        if removed_trips:
            self._reassign_trips_efficiently(neighbor, removed_trips, route_idx)
        
        return neighbor
    
    def _route_optimization(self, states):
        """Local route optimization using 2-opt like improvements"""
        neighbor = copy.deepcopy(states)
        
        if not neighbor:
            return neighbor
        
        # Select route to optimize
        route_idx = random.randint(0, len(neighbor) - 1)
        route = neighbor[route_idx]
        
        # Extract trip sequence
        trip_sequence = [node for node in route if node in self.trip_nodes]
        
        if len(trip_sequence) < 3:
            return neighbor
        
        # Try 2-opt style improvement
        improved_sequence = self._two_opt_improvement(trip_sequence)
        
        # Reconstruct route with charging stations if needed
        if improved_sequence != trip_sequence:
            new_route = self._reconstruct_route_with_charging(improved_sequence)
            neighbor[route_idx] = new_route
        
        return neighbor
    
    def _merge_routes(self, states):
        """Merge compatible short routes"""
        neighbor = copy.deepcopy(states)
        
        if len(neighbor) < 2:
            return neighbor
        
        # Find short routes that can be merged
        short_routes = [(i, route) for i, route in enumerate(neighbor)
                       if len([n for n in route if n in self.trip_nodes]) <= 2]
        all_routes = [(i, route) for i, route in enumerate(neighbor)]
        if len(short_routes) < 2:
            prob = rand()
            if prob > 0.4:
                return neighbor
            else: 
                short_routes = all_routes
        
        # Try to merge two random short routes
        route1_idx, route1 = random.choice(short_routes)
        remaining_routes = [r for r in all_routes if r[0] != route1_idx]
        if not remaining_routes:
            return neighbor
        
        # route2_idx, route2 = random.choice(remaining_routes)
        
        # Merge routes
        trips1 = [n for n in route1 if n in self.trip_nodes]
        route2_idx, merged_route = self.merge_trips(trips1, remaining_routes)
        # trips2 = [n for n in route2 if n in self.trip_nodes]
        
        # merged_trips = trips1 + trips2
        # route2_idx, merged_route = self._reconstruct_route_with_charging(new_trip1)
        
        # Update neighbor
        neighbor[route1_idx] = merged_route
        neighbor.pop(route2_idx)
        
        return neighbor
    
    def sort_trips(self, trip_sequence):
        """
        Sort provided trip sequence

        Parameters
        ----------
        trip_sequence : List
            A small sequence of trip to be sorted based on departure time.

        Returns
        -------
        res : TYPE
            A sorted trip sequence from the departure time.

        """
        res = []
        trip_sorted = self.trips_df.sort_values(by=['dep_time'])
        for idx, row in trip_sorted[['ID', 'dep_time']].iterrows():
            if row['ID'] in self.trip_nodes and idx in trip_sequence:
                res.append(row['ID'])
        return res
    
    def merge_trips(self, trips, routes):
        if not routes:
            return
        numTrips = len(trips)
        new_trips = trips
        tripidx = -1
        for idx, route in routes:
            # print(f"proposed route = {route}")
            temp_merged = trips + route
            temp_merged = self.sort_trips(temp_merged)
            # print(f"temp = {temp_merged}")
            if not self._is_valid_sequence(temp_merged):
                continue
            else:
                res = self._reconstruct_route_with_charging(temp_merged)
                if len(res) >= numTrips:
                    numTrips = len(res)
                    new_trips = res
                    tripidx = idx
            # print(f"tripidx = {tripidx}..\n new_trips = {new_trips}")
        return tripidx, new_trips
                    
    def _split_routes(self, states):
        """Split long routes into smaller ones"""
        neighbor = copy.deepcopy(states)
        
        # Find long routes
        long_routes = [(i, route) for i, route in enumerate(neighbor)
                      if len([n for n in route if n in self.trip_nodes]) > 4]
        
        if not long_routes:
            return neighbor
        
        route_idx, route = random.choice(long_routes)
        trip_sequence = [node for node in route if node in self.trip_nodes]
        
        # Split approximately in the middle
        split_point = len(trip_sequence) // 2
        
        first_half = trip_sequence[:split_point]
        second_half = trip_sequence[split_point:]
        
        # Reconstruct both routes
        route1 = self._reconstruct_route_with_charging(first_half)
        route2 = self._reconstruct_route_with_charging(second_half)
        
        # Update neighbor
        neighbor[route_idx] = route1
        neighbor.append(route2)
        
        return neighbor
    
    def _relocate_trips(self, states):
        """Relocate trips between routes for better efficiency"""
        neighbor = copy.deepcopy(states)
    
        if len(neighbor) < 2:
            return neighbor
    
        # Select source and target routes
        source_idx = random.randint(0, len(neighbor) - 1)
        #target_idx = random.randint(0, len(neighbor) - 1)
        
        #if source_idx == target_idx:
        #    return neighbor
    
        source_trips = [n for n in neighbor[source_idx] if n in self.trip_nodes]
        # print(f"source = {source_trips}")
        if not source_trips:
            # print(neighbor)
            return neighbor
    
        # Select trip to relocate
        trip_to_move = random.choice(source_trips)
        # print(f"trip to move = {trip_to_move}")
        # Remove from source
        new_source = [n for n in neighbor[source_idx] if n != trip_to_move]
        # print(f"new = {new_source}")
        new_source = self._clean_route(new_source)
        # print(f"new_source clean = {new_source}")
        candidate_targets = [(idx, route) for idx, route in enumerate(neighbor) if idx != source_idx and 
                             self._is_valid_sequence(self.sort_trips(route+[trip_to_move]))]
        # print(f"new targets = {candidate_targets}")
        if len(candidate_targets) < 1:
            # print(neighbor)
            return neighbor
        # target_idx = random.randint(0, len(candidate_targets) - 1)
        # Add to target at best position
        best_insertion = self._find_best_insertion_2(candidate_targets, trip_to_move)
        # print(f"BEST insertion = {best_insertion}")
        if best_insertion is not None:
            pos, cost = best_insertion
            route = [(idx, cand) for idx, cand in candidate_targets if idx == pos]
            new_target = self.merge_trips([trip_to_move], route)#neighbor[target_idx][:pos] + [trip_to_move] + neighbor[target_idx][pos:]
    
            neighbor[source_idx] = new_source
            neighbor[pos] = new_target[1]
        # print(neighbor)
        return neighbor
    # def _relocate_trips(self, states):
    #     """Relocate trips between routes for better efficiency"""
    #     neighbor = copy.deepcopy(states)
        
    #     if len(neighbor) < 2:
    #         return neighbor
        
    #     # Select source and target routes
    #     source_idx = random.randint(0, len(neighbor) - 1)
    #     target_idx = random.randint(0, len(neighbor) - 1)
        
    #     if source_idx == target_idx:
    #         return neighbor
        
    #     source_trips = [n for n in neighbor[source_idx] if n in self.trip_nodes]
        
    #     if not source_trips:
    #         return neighbor
        
    #     # Select trip to relocate
    #     trip_to_move = random.choice(source_trips)
        
    #     # Remove from source
    #     new_source = [n for n in neighbor[source_idx] if n != trip_to_move]
    #     new_source = self._clean_route(new_source)
        
    #     # Add to target at best position
    #     best_insertion = self._find_best_insertion(neighbor[target_idx], trip_to_move)
        
    #     if best_insertion is not None:
    #         pos, cost = best_insertion
    #         new_target = neighbor[target_idx][:pos] + [trip_to_move] + neighbor[target_idx][pos:]
            
    #         neighbor[source_idx] = new_source
    #         neighbor[target_idx] = new_target
        
    #     return neighbor
    
    # Helper methods
    def _reassign_trips_efficiently(self, neighbor, trips, exclude_idx):
        """Efficiently reassign trips to existing routes"""
        for trip in trips:
            best_route_idx = None
            best_insertion_cost = float('inf')
            
            for i, route in enumerate(neighbor):
                if i == exclude_idx:
                    continue
                
                insertion = self._find_best_insertion(route, trip)
                # print(f"INSERTION = {insertion}")
                if insertion and insertion[1] < best_insertion_cost:
                    best_route_idx = i
                    best_insertion_cost = insertion[1]
            
            if best_route_idx is not None:
                pos = self._find_best_insertion(neighbor[best_route_idx], trip)[0]
                neighbor[best_route_idx].insert(pos, trip)
            else:
                # Create new single-trip route
                neighbor.append([trip])
    
    def _find_best_insertion_2(self, routes, trip):
        """Find best position to insert trip in route"""
        if not routes:
            return 1  # Insert between depot visits
        
        best_pos = None
        best_cost = float('inf')
        # new_route = route + [trip]
        # new_route = self.sort_trips(new_route)
        ret_idx, new_route = self.merge_trips([trip], routes)
        if ret_idx != -1:
            best_pos = ret_idx
            best_cost = len(new_route)
                # best_roster = new_route
        # for pos in range(1, len(route)):
        #     # if len(route) < 1:  # Don't insert after final depot
        #     #     continue
                
        #     # Simple cost estimation (can be improved)
        #     prev_node = route[pos - 1]
        #     next_node = route[pos] if pos < len(route) else 0
            
        #     # Check connectivity
        #     can_reach = trip in self.arc_from.get(prev_node, set())
        #     can_leave = next_node in self.arc_from.get(trip, set()) or next_node == 0
        #     print(f"CAN REACH - {can_reach}")
        #     print(f"CAN LEAVE = {can_leave}")
        #     if can_reach and can_leave:
        #         # Estimate insertion cost
        #         cost = self._estimate_insertion_cost(prev_node, trip, next_node)
        #         if cost < best_cost:
        #             best_cost = cost
        #             best_pos = pos
        # print(f"BEST = {best_pos}...{best_cost}")
        return (best_pos, best_cost) if best_pos is not None else None
    
    def _find_best_insertion(self, route, trip):
        """Find best position to insert trip in route"""
        if not route:
            return 1  # Insert between depot visits
        
        best_pos = None
        best_cost = float('inf')
        for pos in range(1, len(route)):
            if len(route) < 1:  # Don't insert after final depot
                continue
                
            # Simple cost estimation (can be improved)
            prev_node = route[pos - 1]
            next_node = route[pos] if pos < len(route) else 0
            
            # Check connectivity
            can_reach = trip in self.arc_from.get(prev_node, set())
            can_leave = next_node in self.arc_from.get(trip, set()) or next_node == 0
            # print(f"CAN REACH - {can_reach}")
            # print(f"CAN LEAVE = {can_leave}")
            if can_reach and can_leave:
                # Estimate insertion cost
                cost = self._estimate_insertion_cost(prev_node, trip, next_node)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
        # print(f"BEST = {best_pos}...{best_cost}")
        return (best_pos, best_cost) if best_pos is not None else None
    
    def _estimate_insertion_cost(self, prev_node, new_node, next_node):
        """Estimate cost of inserting new_node between prev_node and next_node"""
        # Simple distance-based estimation (can be improved with actual arc costs)
        cost1 = self.arcs.get((prev_node, new_node), {}).get('duration', 1000)
        cost2 = self.arcs.get((new_node, next_node), {}).get('duration', 1000)
        removed_cost = self.arcs.get((prev_node, next_node), {}).get('duration', 0)
        
        return cost1 + cost2 - removed_cost
    
    def _two_opt_improvement(self, trip_sequence):
        """Apply 2-opt improvement to trip sequence"""
        if len(trip_sequence) < 4:
            return trip_sequence
        
        improved = trip_sequence.copy()
        
        for i in range(len(trip_sequence) - 2):
            for j in range(i + 2, len(trip_sequence)):
                # Try reversing segment between i and j
                new_sequence = (trip_sequence[:i+1] + 
                              trip_sequence[i+1:j+1][::-1] + 
                              trip_sequence[j+1:])
                
                # Check if improvement is valid (connectivity maintained)
                if self._is_sequence_feasible(new_sequence):
                    if self._estimate_sequence_cost(new_sequence) < self._estimate_sequence_cost(improved):
                        improved = new_sequence
        
        return improved
    
    def _is_sequence_feasible(self, sequence):
        """Check if trip sequence is feasible (has required arcs)"""
        for i in range(len(sequence) - 1):
            if (sequence[i], sequence[i+1]) not in self.arcs:
                return False
        return True
    
    def _estimate_sequence_cost(self, sequence):
        """Estimate cost of trip sequence"""
        total_cost = 0
        for i in range(len(sequence) - 1):
            arc_cost = self.arcs.get((sequence[i], sequence[i+1]), {}).get('duration', 1000)
            total_cost += arc_cost
        return total_cost
    
    # def _reconstruct_route(self, trip, route):
    #     if not route:
    #         return
    #     total_duration = 0
        
    #     new_route = route
    #     for r in route:
    #         arc = (r, trip)
    #         if self.arcs.get(arc) == None:
    #             break
    #         else:
    #             duration = self.arcs.get(arc)['duration']
    #             if total_duration + duration > 350:
    #                 available_cs = [cs for cs in self.cs_nodes 
    #                                 if self.recharge_arcs.get((arc[0], arc[1], cs))]
    #             if available_cs:
    #                 css = [(cs, self.recharge_arcs.get((arc[0], arc[1], cs))['duration']) for cs in available_cs]
    #                 css.sort(key=lambda x: x[1], reverse=True)
    #                 cs = css[0]
    #                 route.append(cs[0])
    #                 route.append(trip)
    #                 current_battery = 0#350  # Recharged
    #             total_duration += 
    #             new_route.append(r)
    #             new_route.append(trip)
            
    def _reconstruct_route_with_charging(self, trip_sequence):
        """Reconstruct complete route with charging stations as needed"""
        if not trip_sequence:
            return
        
        route = [trip_sequence[0]]  # Start at depot
        # current_battery = 350  # Assume full battery (D_MAX)
        current_duration = 0
        # print(f"beginning route = {route}")
        for trip in trip_sequence[1:]:
            # Estimate battery consumption
            if route:
                arc = (route[-1], trip)
                consumption = self.arcs.get(arc, {}).get('duration', 30) #* 0.8
                # Check if charging is needed
                # if current_battery - consumption < 50:  # Safety threshold
                if current_duration + consumption > self.D_MAX: #350:
                    # Find nearest charging station
                    available_cs = [cs for cs in self.cs_nodes 
                                    if self.recharge_arcs.get((arc[0], arc[1], cs))]
                                   # if cs in self.recharge_from.get((prev_node, cs), set())]
                    # print(f"available cs = {available_cs}")
                    if available_cs:
                        # cs = random.choice(available_cs)  # Can be optimized
                        css = [(cs, self.recharge_arcs.get((arc[0], arc[1], cs))['duration']) for cs in available_cs]
                        css.sort(key=lambda x: x[1], reverse=True)
                        cs = css[0]
                        route.append(cs[0])
                        route.append(trip)
                        # current_battery = 350  # Recharged
                        current_battery = 0#350  # Recharged
                else:
                    route.append(trip)
                current_duration += consumption
                # print(f"next battery = {current_duration}")
        # route.append(0)  # Return to depot
        return route
    
    def _clean_route(self, route):
        """Remove consecutive depot visits and clean up route"""
        if not route:
            return
        
        cleaned = []
        prev_node = None
        
        for node in route:
            if node != prev_node or node != 0 or node not in self.cs_nodes:  # Avoid consecutive depots
                cleaned.append(node)
                prev_node = node
        
        # # Ensure route starts and ends at depot
        # if not cleaned or cleaned[0] != 0:
        #     cleaned.insert(0, 0)
        # if not cleaned or cleaned[-1] != 0:
        #     cleaned.append(0)
        
        return cleaned
    
    def _is_valid_sequence(self, trip_sequence):
        route_trips = [node for node in trip_sequence if node in self.trip_nodes]
        feasible = set()
        feasible.add(route_trips[0])
        for idx, pair in enumerate(route_trips[:-1]):
            if (pair, list(route_trips)[idx+1]) in self.arcs:
                feasible.add(list(route_trips)[idx+1])
        if len(route_trips) != len(feasible) and len(trip_sequence) > 1:  # Check for overlaps
            return False
        else:
            return True
    
    def _is_valid_solution(self, states):
        """Check if solution is valid"""
        try:
            all_trips = set()
            for route in states:
                route_trips = [node for node in route if node in self.trip_nodes]
                feasible = set()
                feasible.add(route_trips[0])
                for idx, pair in enumerate(route_trips[:-1]):
                    if (pair, list(route_trips)[idx+1]) in self.arcs:
                        feasible.add(list(route_trips)[idx+1])
                if len(route_trips) != len(feasible) and len(route) > 1:  # Check for overlaps
                    return False
                all_trips.update(route_trips)
            return all_trips == self.trip_nodes
        except:
            return False
    
    def _solution_quality(self, states):
        """Estimate solution quality"""
        if not self._is_valid_solution(states):
            return -1000
        
        # Simple quality metric: prefer fewer vehicles and shorter routes
        num_vehicles = len(states)
        total_duration = sum(len(route) for route in states)
        
        return -num_vehicles * 100 - total_duration
    
    def _update_operation_weights(self, operation, improvement):
        """Update operation weights based on performance"""
        current_weight = self.operation_weights[operation]
        
        if improvement > 0:
            # Increase weight for successful operations
            self.operation_weights[operation] = min(2.0, current_weight * 1.1)
        elif improvement < 0:
            # Decrease weight for unsuccessful operations
            self.operation_weights[operation] = max(0.1, current_weight * 0.9)
    
    def get_performance_report(self):
        """Get performance statistics"""
        report = {
            'total_calls': self.stats.total_calls,
            'operation_success_rates': {},
            'average_times': {},
            'current_weights': self.operation_weights.copy()
        }
        
        for op in self.stats.operation_counts:
            success_rate = (self.stats.operation_success[op] / 
                           self.stats.operation_counts[op] if self.stats.operation_counts[op] > 0 else 0)
            avg_time = (self.stats.operation_times[op] / 
                       self.stats.operation_counts[op] if self.stats.operation_counts[op] > 0 else 0)
            
            report['operation_success_rates'][op] = success_rate
            report['average_times'][op] = avg_time
        
        return report
    

def get_total_gap(schedules, all_nodes, recharge_arc):
    total_waiting = 0
    for schedule in schedules:
        for idx in range(len(schedule)-1):
            i = schedule[idx]
            j = schedule[idx+1]
#             print(all_nodes.iloc[i, 1])
            if all_nodes.iloc[i, 1] == 'cs': ## time difference between the end of charging to the next departure
                h = schedule[idx-1]
                start = all_nodes.loc[all_nodes['ID'] == j].iloc[0]['dep_time']
                end = all_nodes.loc[all_nodes['ID'] == h].iloc[0]['arr_time']
                gap_time = start - end
                total_waiting += gap_time
                # print(f"Total Waiting = {total_waiting}\nwaiting = {recharge_arc[(h,j,i)]['duration']}\nj = {all_nodes.iloc[j, 3]}...\ni = {all_nodes.iloc[i, 4]} h = {all_nodes.iloc[h, 4]}")
                # total_waiting += all_nodes.iloc[j, 5] - all_nodes.iloc[h, 6]#recharge_arc[(h,j,i)]['duration']
            elif all_nodes.iloc[j, 1] == 'cs': ## going to charging only cost deadhead time to the charging station
                pass
            elif all_nodes.iloc[j, 1] == 'depot':
                pass
            else:
                #print(all_nodes.iloc[j, 5])
                start = all_nodes.loc[all_nodes['ID'] == j].iloc[0]['dep_time']
                end = all_nodes.loc[all_nodes['ID'] == i].iloc[0]['arr_time']
                gap_time = start - end
                # print(f"Total Waiting = {total_waiting}\nj = {all_nodes.iloc[j, 3]}...\ni = {all_nodes.iloc[i, 4]}")
                # gap_time = all_nodes.iloc[j, 5] - all_nodes.iloc[i, 6] # next_departure - prev_arrival
                total_waiting += gap_time
    return total_waiting

def get_unique_gaps(solution_spaces, all_nodes, recharge_arc):
    unique_nbuses = {}
    for space in solution_spaces:
        if len(space) not in unique_nbuses:
            tot_gap = get_total_gap(space, all_nodes, recharge_arc)
            unique_nbuses[len(space)] = tot_gap
        else:
            tot_gap = get_total_gap(space, all_nodes, recharge_arc)
            prev_gap = unique_nbuses[len(space)]
            if prev_gap > tot_gap:
                unique_nbuses[len(space)] = tot_gap
    return unique_nbuses

def get_pareto(states, prev_state, all_nodes, arcs, recharge_arc, solution_spaces):
    """
    Calculates cost/fitness for the solution/route.
    Cost function is defined as 
        - the number of buses dispatch and,
        - the waiting time (the time difference between the next and previous tasks 
            recharging task comes with deadhead from the terminals to the departure time 
            of the next tasks)
    """
    curr_nbuses = len(states) #Obj1
    prev_nbuses = len(prev_state)
    # prev_buses = [len(prev_state) for prev_state in solution_spaces]
    # prev_gaps = [get_total_gap(prev, all_nodes, recharge_arc) for prev in solution_spaces]
    prev_gaps = get_unique_gaps(solution_spaces, all_nodes, recharge_arc)
    if curr_nbuses > prev_nbuses:
        prob = rand()
        if prob > 0.4:
            gap = min(prev_gaps.values())
            numBuses = [key for key, val in prev_gaps.items() if val == gap]
            return numBuses[0]
        else:
            prev_gaps[curr_nbuses] = get_total_gap(states, all_nodes, recharge_arc)
            return random.choice(list(prev_gaps.keys()))
    else:
        return curr_nbuses
    
def annealing(initial_state, trips_df, arcs, recharge_arc, D_MAX, runs=500):
    
    """Peforms simulated annealing to find a solution"""
    initial_temp = runs
    alpha = 0.95
    current_temp = initial_temp
    # Start by initializing the current state with the initial state
    solution = initial_state
    same_solution = 0
    same_cost_diff = 0
    temperature = []
    it = 0
    cost_diffs = []
    costs = []
    generator = ImprovedNeighborhoodGenerator(trips_df, arcs, recharge_arc, D_MAX)
    best_costs = []
    reheat_threshold = 100
    reheat_counter = 0
    solution_spaces = [solution]
    best_cost, best_soln = get_pareto(solution, solution, trips_df, arcs, recharge_arc, solution_spaces), solution
    curr_cost, curr_solution = best_cost, best_soln
    temp, temp_cost = curr_solution, curr_cost
    pbar = tqdm(total = runs+1)
    while current_temp > 0 and it < runs:
        print(f"Iteration {it+1}...")
        # neighbor = get_neighbors(curr_solution, trips_df, arcs, recharge_arc)
        neighbor = generator.get_neighbors_enhanced(curr_solution)
        # print(f"proposed solution = {neighbor}")
        solution_spaces.append(neighbor)
        # Check if neighbor is best so far
        # neighbor_cost = get_cost(neighbor, curr_solution, trips_df, arcs, recharge_arc, solution_spaces)
        neighbor_cost = get_pareto(neighbor, curr_solution, trips_df, arcs, recharge_arc, solution_spaces)
        print(f"NEIGHBOUR = {neighbor_cost}\nCURR = {curr_cost}\ndiff = {-float(neighbor_cost - curr_cost)}")
        cost_diff = neighbor_cost - curr_cost
        cost_diffs.append(cost_diff)
        it += 1
        accept_neighbor = False
        if cost_diff < 0:
            accept_neighbor = True
            same_solution = 0
            same_cost_diff = 0
        else:
            try:
                acceptance_prob = math.exp(-cost_diff / current_temp)
                if rand() < acceptance_prob:
                    accept_neighbor = True
                    same_solution = 0
                    same_cost_diff = 0
                else:
                    same_cost_diff+=1
                    same_solution+=1
            except (OverflowError, ZeroDivisionError):
                same_cost_diff+=1
                same_solution+=1
        
        if accept_neighbor:
            curr_cost, curr_solution = neighbor_cost, copy.deepcopy(neighbor)
            if (curr_cost - best_cost) < 0:
                best_cost, best_soln = curr_cost, copy.deepcopy(curr_solution)
        
        if same_solution > reheat_threshold and reheat_counter < 3:
            current_temp = initial_temp * 0.5
            reheat_counter += 1
            same_cost_diff = 0
            same_solution = 0
        costs.append(curr_cost)
        best_costs.append(best_cost)
        temperature.append(current_temp)
        # decrement the temperature
        current_temp = current_temp*alpha
        print('-'*100)
        pbar.update(1)
    pbar.close()
    return best_soln, best_cost, cost_diffs, temperature, it, costs, solution_spaces, best_costs