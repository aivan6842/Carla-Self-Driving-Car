import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import random
import time
import cv2
import argparse
import logging
import os

IM_HEIGHT = 480
IM_WIDTH = 640
MODEL = keras.models.load_model('nonBalanced-15-Epoch')


def process_image(image, vehicle):
    i = np.array(image.raw_data)
    i = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) ##since rgba
    cv2.imshow('', i)
    cv2.waitKey(1)
    grey = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) ##convert to grey scale

    resized_img = cv2.resize(grey, (50, 50))
    print(resized_img.shape)

    predSteer = MODEL.predict([resized_img.reshape(-1, 50, 50, 1)])
    print(predSteer)

    vehicle.apply_controls(carla.VehicleControl(throttle=0.5, np.argmax(predSteer)))





def delete_all_cars(world, client):
    all = world.get_actors()
    vehicles = all.filter("vehicle.*")
    sensors = all.filter("sensor.*")
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
    client.apply_batch([carla.command.DestroyActor(x) for x in sensors])

    print("Destroyed all vehicles")


def main():

    actors_list = []

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter("model3")[0]

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    actors_list.append(vehicle)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "110")
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    camera = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
    camera.listen(lambda image: process_image(image, vehicle))
    actors_list.append(camera)

    try:
        while True:
            world_snapshot = world.wait_for_tick()

            if vehicle.is_at_traffic_light():
                tl = vehicle.get_traffic_light()
                for x in tl.get_group_traffic_lights():
                    state = carla.TrafficLightState.Green
                    x.set_state(state)
    finally:
        delete_all_cars(world, client)


if __name__ == '__main__':
    main()
