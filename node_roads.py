import numpy as np
import cv2
import pickle
import random
import heapq
from PIL import Image

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
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connections = []
        self.color = NODE_COLOR

    def render(self, canvas):
        # Draw the node as a circle
        cv2.circle(canvas, (int(self.x), int(self.y)), NODE_RADIUS, self.color, 1)
        
        for node in self.connections:
            # Draw the line connecting nodes
            start = (int(self.x), int(self.y))
            end = (int(node.x), int(node.y))
            cv2.line(canvas, start, end, self.color, 1)

            if RENDER_ARROWS:
                midpoint = np.array([(self.x + node.x) / 2, (self.y + node.y) / 2])
                
                # Calculate the vector representing the direction of the line
                line_vector = np.array([node.x - self.x, node.y - self.y])
                line_length = np.linalg.norm(line_vector)
                
                if line_length == 0:
                    continue  # Avoid division by zero for degenerate cases
                
                # Normalize to get the unit direction vector
                unit_vector = line_vector / line_length
                
                # Perpendicular vector to the direction vector
                perp_vector = np.array([-unit_vector[1], unit_vector[0]])
                
                # Scale the arrowhead to a suitable size
                arrow_size = 10  # Length of the arrowhead
                arrow_tip = midpoint  # Place the arrowhead at the midpoint
                left_point = arrow_tip - arrow_size * (unit_vector + 0.5 * perp_vector)
                right_point = arrow_tip - arrow_size * (unit_vector - 0.5 * perp_vector)
                
                # Convert points to integer tuple coordinates for OpenCV
                arrow_tip = (int(arrow_tip[0]), int(arrow_tip[1]))
                left_arrow = (int(left_point[0]), int(left_point[1]))
                right_arrow = (int(right_point[0]), int(right_point[1]))
                
                # Draw the arrowhead
                cv2.line(canvas, arrow_tip, left_arrow, self.color, 1)
                cv2.line(canvas, arrow_tip, right_arrow, self.color, 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

class Vehicle():
    def __init__(self, simulation, x, y, speed=0, max_speed=10, acceleration=1, deceleration=2):
        self.simulation = simulation
        self.x = x
        self.y = y
        self.speed = speed
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.angle = 0
        self.target_speed = max_speed

        self.node_stack = []
        self.current_destination = None

    def a_star_stack(self, destination: RoadNode):
        def heuristic(node, goal):
            return abs(node.x - goal.x) + abs(node.y - goal.y)
        
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
    
    def decelerate(self, dt):
        if self.speed > self.target_speed:
            self.speed = max(self.speed - self.deceleration * dt, 0)
    
    def move(self, dt):
        self.x += self.speed * np.cos(self.angle) * dt
        self.y += self.speed * np.sin(self.angle) * dt
    
    def update(self, dt):
        if self.current_destination is None:
            if len(self.node_stack) > 0:
                self.current_destination = self.node_stack.pop(0)
            else:
                final_destination = random.choice(self.simulation.road_nodes)
                self.a_star_stack(final_destination)

        self.move(dt)

    def render(self, canvas):
        # Draw the vehicle as a circle
        cv2.circle(canvas, (int(self.x), int(self.y)), CAR_SIZE, CAR_COLOR, -1)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

def create_sim(save=True):
    sim = Simulation()

    A = RoadNode(10, SIZE_Y//2 - 10)
    B = RoadNode(10, SIZE_Y//2 + 10)
    C = RoadNode(SIZE_X//2 - 20, SIZE_Y//2 - 10)
    D = RoadNode(SIZE_X//2 - 20, SIZE_Y//2 + 10)
    E = RoadNode(SIZE_X//2 + 20, SIZE_Y//2 - 10)
    F = RoadNode(SIZE_X//2 + 20, SIZE_Y//2 + 10)
    G = RoadNode(SIZE_X//2 - 10, SIZE_Y//2 - 20)
    H = RoadNode(SIZE_X//2 + 10, SIZE_Y//2 - 20)
    I = RoadNode(SIZE_X - 10, SIZE_Y//2 - 10)
    J = RoadNode(SIZE_X - 10, SIZE_Y//2 + 10)
    K = RoadNode(SIZE_X//2 + 10, 10)
    L = RoadNode(SIZE_X//2 - 10, 10)

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

if __name__ == "__main__":
    mysim = Simulation()

    mysim.load_state("three_way_intersection.pkl")

    a = Vehicle(mysim, 10, SIZE_Y//2 - 10)

    mysim.add_vehicle(a)

    a.a_star_stack(mysim.road_nodes[9])

    [print(chr(mysim.road_nodes.index(item) + ord("A")))for item in a.node_stack]

    mysim.update(1)

    frame = mysim.render()

    cv2.imshow('Simulation Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()