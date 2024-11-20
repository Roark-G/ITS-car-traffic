import numpy as np
import cv2
from PIL import Image

# VIDEO SETTINGS
FPS = 30
VIDEO_DURATION = 10
SIZE_X = 1920
SIZE_Y = 1080

# SETTINGS
NUM_CARS = 100
SPEED = 3
LOOKAHEAD = 5
LOOKAHEAD_ANGLE = np.pi / 4
PULL_EFFECT_STRENGTH = 0.4

# RENDERING
CAR_COLOR = [255, 255, 255]
ROAD_COLOR = [100, 114, 122]



class CarSystem:
    def __init__(self, num_cars):
        self.num_cars = num_cars
        # [ x , y , dx , dy , target]
        self.cars = np.zeros((num_cars, 5), dtype=np.float32)
        self.cars[:, :2] = np.random.rand(num_cars, 2) * [SIZE_X, SIZE_Y]
        orientations = np.random.rand(num_cars) * 2 * np.pi
        self.cars[:, 2] = np.cos(orientations) * SPEED
        self.cars[:, 3] = np.sin(orientations) * SPEED

    def load_network(self, file_loc="road_network\Simple.png"):
        network = np.array(Image.open(file_loc).convert("L"))
        network = 255 - network
        self.network = np.where(network > 0, True, False)

    # def stay_on_road(self):
    #     # forward vector
    #     forward_vector = self.agents[:, 2:] * LOOKAHEAD

    #     # right vector
    #     right_vector = np.zeros_like(forward_vector)
    #     right_vector[:, 0] = (forward_vector[:, 0] * np.cos(LOOKAHEAD_ANGLE) - 
    #                         forward_vector[:, 1] * np.sin(LOOKAHEAD_ANGLE))
    #     right_vector[:, 1] = (forward_vector[:, 0] * np.sin(LOOKAHEAD_ANGLE) + 
    #                         forward_vector[:, 1] * np.cos(LOOKAHEAD_ANGLE))

    #     # left vector
    #     left_vector = np.zeros_like(forward_vector)
    #     left_vector[:, 0] = (forward_vector[:, 0] * np.cos(-LOOKAHEAD_ANGLE) - 
    #                         forward_vector[:, 1] * np.sin(-LOOKAHEAD_ANGLE))
    #     left_vector[:, 1] = (forward_vector[:, 0] * np.sin(-LOOKAHEAD_ANGLE) + 
    #                         forward_vector[:, 1] * np.cos(-LOOKAHEAD_ANGLE))

    #     # check in bounds
    #     def in_bounds(positions):
    #         return (0 <= positions[:, 0]) & (positions[:, 0] < SIZE_Y) & \
    #                (0 <= positions[:, 1]) & (positions[:, 1] < SIZE_X)

    #     forward_pos, right_pos, left_pos = self.agents[:, :2] + forward_vector, self.agents[:, :2] + right_vector, self.agents[:, :2] + left_vector
    #     forward_mask, right_mask, left_mask = in_bounds(forward_pos), in_bounds(right_pos), in_bounds(left_pos)

    #     forward_strength, right_strength, left_strength = np.zeros(self.agents.shape[0]), np.zeros(self.agents.shape[0]), np.zeros(self.agents.shape[0])

    #     forward_strength[forward_mask] = self.pheromone_map[
    #         forward_pos[forward_mask, 0].astype(int),
    #         forward_pos[forward_mask, 1].astype(int)
    #     ]

    #     right_strength[right_mask] = self.pheromone_map[
    #         right_pos[right_mask, 0].astype(int),
    #         right_pos[right_mask, 1].astype(int)
    #     ]

    #     left_strength[left_mask] = self.pheromone_map[
    #         left_pos[left_mask, 0].astype(int),
    #         left_pos[left_mask, 1].astype(int)
    #     ]

    #     turn_influence = (
    #         forward_strength[:, np.newaxis] * forward_vector +
    #         right_strength[:, np.newaxis] * right_vector +
    #         left_strength[:, np.newaxis] * left_vector
    #     )

    #     direction_adjustment = turn_influence * PULL_EFFECT_STRENGTH

    #     # Update velocity vectors, maintaining the same speed
    #     new_velocity = self.agents[:, 2:] + direction_adjustment
    #     speed_magnitude = np.linalg.norm(new_velocity, axis=1, keepdims=True)
    #     self.agents[:, 2:] = (new_velocity / speed_magnitude) * SPEED

    def update(self):
        self.cars[:, :2] += self.cars[:, 2:4]

        self.cars[:, 2:4] *= np.where((self.cars[:, :2] < 0) | (self.cars[:, :2] > [SIZE_X, SIZE_Y]), -1, 1)
         
    def render_roads(self, img):
        img[self.network] = ROAD_COLOR

    def render_cars(self, img):
        img[np.clip(self.cars[:, 1].astype(int), 0, SIZE_Y - 1),
            np.clip(self.cars[:, 0].astype(int), 0, SIZE_X - 1)] = CAR_COLOR

    def render_frame(self, make_img=False):
            img = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)

            self.render_roads(img)
            self.render_cars(img)
            
            self.update()

            return Image.fromarray(img) if make_img else img

def create_video(system, file_output):
    total_frames = FPS * VIDEO_DURATION
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(fr"videos\{file_output}.mp4", fourcc, FPS, (SIZE_X, SIZE_Y))

    print("Creating video: ")
    for frame_num in range(total_frames):
        frame = system.render_frame()
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        system.update()
        
        # Calculate and display progress
        progress = (frame_num + 1) / total_frames
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        percent = progress * 100
        print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)
    
    print("\nVideo creation complete!")
    video.release()

if __name__ == "__main__":
    system = CarSystem(NUM_CARS)
    system.load_network()
    # print(system.network)
    # print(system.network.shape)

    create_video(system, r"test")