import numpy as np
import cv2
from PIL import Image


#CONSTANTS
SIZE_X = 1920
SIZE_Y = 1080

NUM_CARS = 100
SPEED = 10

# RENDERING
CAR_COLOR = [255, 255, 255]

# VIDEO SETTINGS
FPS = 30
VIDEO_DURATION = 10


class CarSystem:
    def __init__(self, num_cars):
        self.num_cars = num_cars
        # [ x , y , dx , dy , target]
        self.cars = np.zeros((num_cars, 5), dtype=np.float32)
        self.cars[:, :2] = np.random.rand(num_cars, 2) * [SIZE_X, SIZE_Y]
        orientations = np.random.rand(num_cars) * 2 * np.pi
        self.cars[:, 2] = np.cos(orientations) * SPEED
        self.cars[:, 3] = np.sin(orientations) * SPEED
        
    def update(self):
        self.cars[:, :2] += self.cars[:, 2:4]

        self.cars[:, 2:4] *= np.where((self.cars[:, :2] < 0) | (self.cars[:, :2] > [SIZE_X, SIZE_Y]), -1, 1)

    def render_frame(self, make_img=False):
            img = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)

            # Draw cars
            img[np.clip(self.cars[:, 1].astype(int), 0, SIZE_Y - 1),
                np.clip(self.cars[:, 0].astype(int), 0, SIZE_X - 1)] = CAR_COLOR
            
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
    create_video(CarSystem(NUM_CARS), r"test")