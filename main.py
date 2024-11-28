import numpy as np
import cv2
from PIL import Image

# VIDEO SETTINGS
FPS = 30
VIDEO_DURATION = 10
SIZE_X = 1920
SIZE_Y = 1080

CAR_SIZE = 10
ROAD_WIDTH = 10
CAR_COLOR = [255, 255, 255]
ROAD_LINE = [245, 230, 27]
ROAD_PAVEMENT = [110, 109, 104]

cars = []
roads = []

class Road:
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

        self.angle = np.arctan2(end_y - start_y, end_x - start_x)

        roads.append(self)

    def get_nearest_point(self, x, y):
        road_vec = np.array([self.end_x - self.start_x, self.end_y - self.start_y])
        point_vec = np.array([x - self.start_x, y - self.start_y])
    
        road_length = np.linalg.norm(road_vec)
        road_unit = road_vec / road_length
        projection_length = np.dot(point_vec, road_unit)

        projection_length = max(0, min(road_length, projection_length))
        
        # Calculate nearest point
        nearest_x = self.start_x + road_unit[0] * projection_length
        nearest_y = self.start_y + road_unit[1] * projection_length
        
        return nearest_x, nearest_y, self.angle

class Car:
    def __init__(self, x, y, angle, speed):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.current_road = None

        cars.append(self)

    def find_nearest_road(self):
        nearest_dist = float('inf')
        nearest_road = None
        
        for road in roads:
            x, y, _ = road.get_nearest_point(self.x, self.y)
            dist = np.sqrt((x - self.x)**2 + (y - self.y)**2)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_road = road
        
        return nearest_road

    def move(self):
        self.current_road = self.find_nearest_road()
        
        nearest_x, nearest_y, road_angle = self.current_road.get_nearest_point(self.x, self.y)

        self.angle = road_angle

        self.x += self.speed * np.cos(self.angle)
        self.y += self.speed * np.sin(self.angle)

    def render(self, img):
        rect_points = cv2.boxPoints((
            (self.x, self.y),  # Center point
            (CAR_SIZE * 2, CAR_SIZE),  # Size (length x width)
            np.degrees(self.angle)  # Angle in degrees
        ))
        rect_points = np.int32(rect_points)
        
        cv2.fillPoly(img, [rect_points], color=CAR_COLOR)
        return img

def update():
    for car in cars:
        car.move()

def render_frame():
    img = np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)

    for road in roads:
        cv2.line(img, (int(road.start_x), int(road.start_y)), (int(road.end_x), int(road.end_y)), ROAD_PAVEMENT, 15)        
        cv2.line(img, (int(road.start_x), int(road.start_y)), (int(road.end_x), int(road.end_y)), ROAD_LINE)
        
    for car in cars:
        car.render(img)
    
    return img

def create_video(file_output, update_funct, render_frame_funct):
    total_frames = FPS * VIDEO_DURATION
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"videos\\{file_output}.mp4", fourcc, FPS, (SIZE_X, SIZE_Y))

    print("Creating video:")
    for frame_num in range(total_frames):
        frame = render_frame_funct()
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        update_funct()
        
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
    Car(0, 20, np.pi/4, 5)
    Car(SIZE_X, 60, np.pi/4, 5)
    print(Road(0, 20, SIZE_X, 40))
    print(Road(SIZE_X, 60, 0, 40))
    
    create_video("test2", update, render_frame)