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
FILE_NAME = "dataSquare.npy"


def delete_all_cars(world, client):
    all = world.get_actors()
    vehicles = all.filter("vehicle.*")
    sensors = all.filter("sensor.*")
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
    client.apply_batch([carla.command.DestroyActor(x) for x in sensors])

    print("Destroyed all vehicles")


def process_image(image, vehicle, training_data):
    i = np.array(image.raw_data)
    i = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) ##since rgba
    grey = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) ##convert to grey scale

    resized_img = cv2.resize(grey, (50, 50))

    vehicle_control = vehicle.get_control()
    controls = np.array([vehicle_control.throttle,
                vehicle_control.steer,
                vehicle_control.brake])


    training_data.append(np.array([resized_img, controls]))

    return i



def main():

        actors_list = []
        training_data = []


        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter("model3")[0]



        spawn_point = random.choice(world.get_map().get_spawn_points())

        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)
        actors_list.append(vehicle)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        camera_bp.set_attribute("fov", "110")
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        camera = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
        camera.listen(lambda image: process_image(image, vehicle, training_data))
        actors_list.append(camera)

        try:
            while True:
                world_snapshot = world.wait_for_tick()

                if vehicle.is_at_traffic_light():
                    tl = vehicle.get_traffic_light()
                    for x in tl.get_group_traffic_lights():
                        state = carla.TrafficLightState.Green
                        x.set_state(state)

                if len(training_data) % 1000 == 0 and len(training_data) != 0:
                    print("1000!")
        finally:
            delete_all_cars(world, client)
            np.save(FILE_NAME, training_data)




if __name__ == "__main__":
    main()
