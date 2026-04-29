import irsim

env = irsim.make("lidar_world.yaml")
# env = irsim.make('lidar_world_noise.yaml')

for _i in range(3000):
    env.step()
    env.render(0.05)

    print("CHECK")
    print(env.robot)
    print(env.obstacle_list)
    print(env.robot.sensors)

    if env.done():
        break

env.end(3)
