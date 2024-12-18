import numpy as np
import cv2
import pickle
import random
import heapq
import time
from PIL import Image
from math import sqrt

DEBUG_MODE = True

# VIDEO SETTINGS
FPS = 30
VIDEO_DURATION = 10
SIZE_X = 700
SIZE_Y = 500

RENDER_ARROWS = True

CAR_SIZE = 5
ROAD_WIDTH = 10
CAR_COLOR = [255, 255, 255]

# Nodes
NODE_COLOR = [115, 204, 55]
NODE_RADIUS = 10

class Simulation:
    def __init__(self):
        self.vehicles = []
        self.road_nodes = []

        self.render_nodes = True
        self.render_vehicles = True
    
    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)
    
    def add_road_node(self, node):
        self.road_nodes.append(node)
    
    def update(self, dt):
        for vehicle in self.vehicles:
            vehicle.update(dt)

        for node in self.road_nodes:
            node.update(dt)
    
    def render(self):
        canvas = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)

        if self.render_nodes:
            for node in self.road_nodes:
                node.render(canvas)
        
        if self.render_vehicles:
            for vehicle in self.vehicles:
                vehicle.render(canvas)
        
        return canvas
    
    def save_state(self, filename='simulation_state.pkl'):
        """Save the current simulation state to a file"""
        state = {
            'vehicles': self.vehicles,
            'road_nodes': self.road_nodes,
            'render_nodes': self.render_nodes,
            'render_vehicles': self.render_vehicles
        }
        try:
            with open(filename, 'wb') as f:
                pickle.dump(state, f)
            return True
        except Exception as e:
            print(f"Error saving simulation state: {e}")
            return False

    def load_state(self, filename='simulation_state.pkl'):
        """Load simulation state from a file"""
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                self.vehicles = state['vehicles']
                self.road_nodes = state['road_nodes']
                self.render_nodes = state['render_nodes']
                self.render_vehicles = state['render_vehicles']
            return True
        except Exception as e:
            print(f"Error loading simulation state: {e}")
            return False

class RoadNode():
    def __init__(self, x, y, label=None, timing_list : list=None):
        self.x = x
        self.y = y
        self.connections = []
        self.label = label
        self.color = NODE_COLOR

        self.timing_list = timing_list  # [green_time, yellow_time, red_time]
        self.time_accumulated = 0
        self.stop = False

        self.speed_lims = [] # to be implemented
        

    def update(self, dt):
        if self.timing_list is None:
            return
        
        if self.time_accumulated < self.timing_list[0]:
            # Green
            self.stop = False
            self.color = [0, 255, 0]

        elif self.time_accumulated < self.timing_list[0] + self.timing_list[1]:
            # Yellow
            self.stop = True
            self.color = [255, 255, 0]

        else:
            # Red
            self.stop = True
            self.color = [255, 0, 0]
        
        self.time_accumulated = (self.time_accumulated + dt) % sum(self.timing_list)
        
    def render(self, canvas):
        cv2.circle(canvas, (int(self.x), int(self.y)), NODE_RADIUS, self.color, 1)
        
        for node in self.connections:
            # Draw the line connecting nodes
            start = (int(self.x), int(self.y))
            end = (int(node.x), int(node.y))
            cv2.line(canvas, start, end, self.color, 1)

            if RENDER_ARROWS:
                midpoint = np.array([(self.x + node.x) / 2, (self.y + node.y) / 2])
                
                line_vector = np.array([node.x - self.x, node.y - self.y])
                line_length = np.linalg.norm(line_vector)
                
                if line_length == 0:
                    continue
                
                unit_vector = line_vector / line_length
                
                perp_vector = np.array([-unit_vector[1], unit_vector[0]])
                
                arrow_size = 10
                arrow_tip = midpoint
                left_point = arrow_tip - arrow_size * (unit_vector + 0.5 * perp_vector)
                right_point = arrow_tip - arrow_size * (unit_vector - 0.5 * perp_vector)
                
                arrow_tip = (int(arrow_tip[0]), int(arrow_tip[1]))
                left_arrow = (int(left_point[0]), int(left_point[1]))
                right_arrow = (int(right_point[0]), int(right_point[1]))
                
                cv2.line(canvas, arrow_tip, left_arrow, self.color, 1)
                cv2.line(canvas, arrow_tip, right_arrow, self.color, 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return f"Node {self.label} at ({self.x}, {self.y})"

class Vehicle():
    def __init__(self, simulation, x, y, speed=0, max_speed=10, acceleration=0.2, deceleration=0.1):
        self.simulation = simulation
        self.x = x
        self.y = y
        self.speed = speed
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.angle = 0
        self.target_speed = max_speed

        self.braking_distance = (self.speed ** 2) / (2 * self.deceleration)

        self.node_stack = []

    def update_braking_distance(self):
        self.braking_distance = (self.speed ** 2) / (2 * self.deceleration)

    def a_star_stack(self, destination: RoadNode):
        def heuristic(node, goal):
            '''Euclidean'''
            return sqrt(abs((node.x - goal.x) ** 2 + (node.y - goal.y) ** 2))
        
        open_set = []
        g_scores = {node: float('inf') for node in self.simulation.road_nodes}
        parents = {}

        start_node = min(
            self.simulation.road_nodes, 
            key=lambda n: np.hypot(n.x - self.x, n.y - self.y)
        )
        
        g_scores[start_node] = 0
        heapq.heappush(open_set, (0 + heuristic(start_node, destination), start_node))

        while open_set:
            _, current_node = heapq.heappop(open_set)
            
            if current_node == destination:
                # Reconstruct path
                path = []
                while current_node in parents:
                    path.append(current_node)
                    current_node = parents[current_node]
                path.reverse()  # Start to destination
                self.node_stack = path
                return
            
            for neighbor in current_node.connections:
                tentative_g_score = g_scores[current_node] + np.hypot(
                    neighbor.x - current_node.x, 
                    neighbor.y - current_node.y
                )
                if tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    parents[neighbor] = current_node
                    heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, destination), neighbor))
    
    def accelerate(self, dt):
        if self.speed < self.target_speed:
            self.speed = min(self.speed + self.acceleration * dt, self.max_speed)
            self.update_braking_distance()

    def decelerate(self, dt):
        if self.speed > self.target_speed:
            new_speed = self.speed - (self.deceleration * dt)
            self.speed = max(new_speed, self.target_speed)
            self.update_braking_distance()

    def get_next_node_dist(self):
        target_node = self.node_stack[0]
        return np.hypot(target_node.x - self.x, target_node.y - self.y)

    def update_angle(self):
        target_node = self.node_stack[0]
        self.angle = np.arctan2(target_node.y - self.y, target_node.x - self.x)

    def update_accelerations(self, distance, dt):
        if len(self.node_stack) == 0:
            raise Exception("No nodes in stack")

        target_node = self.node_stack[0]
        
        safe_braking_distance = self.braking_distance * 1.5

        if target_node.stop and distance < safe_braking_distance:
            self.target_speed = 0
        elif distance < safe_braking_distance:
            self.target_speed = self.max_speed * (distance / safe_braking_distance)
        else:
            self.target_speed = self.max_speed

        if self.speed < self.target_speed:
            self.accelerate(dt)
        elif self.speed > self.target_speed:
            self.decelerate(dt)
    
    def move(self, dt):
        self.x += self.speed * np.cos(self.angle) * dt
        self.y += self.speed * np.sin(self.angle) * dt

    def check_node_arrival(self, dist, dt):
        '''Check if the vehicle is within a "reasonable" distance of the node'''
        distance = dist
        print(f"{distance:.2f} from next node")

        target_node = self.node_stack[0]
        
        if target_node.stop and distance < self.braking_distance * 1.5:
            self.target_speed = 0
            return  # keep node in stack until light changes
        
        if distance < self.speed * dt and (not target_node.stop or self.speed < 0.1):
            self.node_stack.pop(0)

    
    def update(self, dt):
        if DEBUG_MODE: print(f"\nGoal: {'None' if len(self.node_stack) == 0 else self.node_stack[0]}")

        if len(self.node_stack) == 0:
            if DEBUG_MODE: print("performing A* so that there's nodes in stack")
            final_destination = random.choice(self.simulation.road_nodes)
            self.a_star_stack(final_destination)
            return

        dist = self.get_next_node_dist()

        self.update_angle()
        self.update_accelerations(dist, dt)
        self.move(dt)

        self.check_node_arrival(dist, dt)

    def render(self, canvas):
        # Draw the vehicle as a circle
        cv2.circle(canvas, (int(self.x), int(self.y)), CAR_SIZE, CAR_COLOR, -1)
        cv2.putText(canvas, f"{self.speed:.2f}", (int(self.x), int(self.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CAR_COLOR, 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

def conv_to_frames(lst):
    return [x * FPS for x in lst]

def create_sim(save=True):
    sim = Simulation()

    A = RoadNode(10, SIZE_Y//2 - 10, label='A')
    B = RoadNode(10, SIZE_Y//2 + 10, label='B')
    C = RoadNode(SIZE_X//2 - 20, SIZE_Y//2 - 10, label='C')
    D = RoadNode(SIZE_X//2 - 20, SIZE_Y//2 + 10, label='D', timing_list=conv_to_frames([5, 1, 5]))
    E = RoadNode(SIZE_X//2 + 20, SIZE_Y//2 - 10, label='E', timing_list=conv_to_frames([5, 1, 5]))
    F = RoadNode(SIZE_X//2 + 20, SIZE_Y//2 + 10, label='F')
    G = RoadNode(SIZE_X//2 - 10, SIZE_Y//2 - 20, label='G')
    H = RoadNode(SIZE_X//2 + 10, SIZE_Y//2 - 20, label='H')
    I = RoadNode(SIZE_X - 10, SIZE_Y//2 - 10, label='I')
    J = RoadNode(SIZE_X - 10, SIZE_Y//2 + 10, label='J')
    K = RoadNode(SIZE_X//2 + 10, 10, label='K')
    L = RoadNode(SIZE_X//2 - 10, 10, label='L')

    A.connections = [B]
    B.connections = [D]
    C.connections = [A]
    D.connections = [F, H]
    E.connections = [C, H]
    F.connections = [J]
    G.connections = [C, F]
    H.connections = [K]
    I.connections = [E]
    J.connections = [I]
    K.connections = [L]
    L.connections = [G]

    sim.add_road_node(A)
    sim.add_road_node(B)  
    sim.add_road_node(C)
    sim.add_road_node(D)
    sim.add_road_node(E)  
    sim.add_road_node(F)
    sim.add_road_node(G)
    sim.add_road_node(H)
    sim.add_road_node(I)
    sim.add_road_node(J)
    sim.add_road_node(K)
    sim.add_road_node(L)

    if save:
        sim.save_state("three_way_intersection.pkl")
    return sim

def create_video(system, file_output, rate):
    total_frames = FPS * VIDEO_DURATION
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(fr"{file_output}.mp4", fourcc, FPS, (SIZE_X, SIZE_Y))
    
    # Add time tracking
    start_time = time.time()
    last_frame_time = start_time

    print("Creating video: ")
    for frame_num in range(total_frames):
        frame = system.render()
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        system.update(rate)
        
        # Calculate and display progress with time
        current_time = time.time()
        frame_time = current_time - last_frame_time
        last_frame_time = current_time
        
        progress = (frame_num + 1) / total_frames
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        percent = progress * 100
        print(f'\r[{bar}] {percent:.1f}% ({frame_time*1000:.1f}ms/frame)', end='', flush=True)
    
    print("\nVideo creation complete!")
    video.release()

if __name__ == "__main__":
    mysim = create_sim(save=False)

    mysim.add_vehicle(Vehicle(mysim, 10, SIZE_Y//2 - 10))
    mysim.add_vehicle(Vehicle(mysim, SIZE_X//2 - 20, SIZE_Y//2 - 10))
    mysim.add_vehicle(Vehicle(mysim, SIZE_X//2 + 10, 10))

    create_video(mysim, r"videos\\three_way_intersection_1", rate=1)