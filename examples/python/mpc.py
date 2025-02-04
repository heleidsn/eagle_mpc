import numpy as np
import time
import copy
import crocoddyl
import sys
import example_robot_data


import eagle_mpc
from eagle_mpc.utils.path import EAGLE_MPC_YAML_DIR
from eagle_mpc.utils.simulator import AerialSimulator
from eagle_mpc.utils.plots import PlotControlsGroup, showPlots

# Trajectory
dt = 10  # ms
useSquash = True  # False
robotName = 'hexacopter370_flying_arm_3'
trajectoryName = 'hover'
# trajectoryName = 'displacement'
mpcName = 'carrot'

WITHDISPLAY = 'display' in sys.argv


trajectory = eagle_mpc.Trajectory()
trajectory.autoSetup(EAGLE_MPC_YAML_DIR + "/" + robotName +
                     "/trajectories/" + trajectoryName + ".yaml")
problem = trajectory.createProblem(dt, useSquash, "IntegratedActionModelEuler")

if useSquash:
    solver = eagle_mpc.SolverSbFDDP(problem, trajectory.squash)
else:
    solver = crocoddyl.SolverBoxFDDP(problem)

solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.solve([], [], maxiter=400)

mpcPath = EAGLE_MPC_YAML_DIR + "/" + robotName + "/mpc/mpc.yaml"
if mpcName == 'rail':
    mpcController = eagle_mpc.RailMpc(solver.xs, dt, mpcPath)
elif mpcName == 'weighted':
    mpcController = eagle_mpc.WeightedMpc(trajectory, dt, mpcPath)
else:
    mpcController = eagle_mpc.CarrotMpc(trajectory, solver.xs, dt, mpcPath)

mpcController.updateProblem(0)
mpcController.solver.solve(
    solver.xs[:mpcController.problem.T + 1], solver.us[:mpcController.problem.T])
mpcController.solver.convergence_init = 1e-3

dtSimulator = 2
simulator = AerialSimulator(mpcController.robot_model,
                            mpcController.platform_params, dtSimulator, solver.xs[0])
t = 0
updateTime = []
solveTime = []

for i in range(0, int(problem.T * dt * 1.2)):
    mpcController.problem.x0 = simulator.states[-1]
    start = time.time()
    mpcController.updateProblem(int(t))
    end = time.time()
    updateTime.append(end - start)
    start = time.time()
    mpcController.solver.solve(
        mpcController.solver.xs, mpcController.solver.us, mpcController.iters)
    end = time.time()
    # print('solve time: ', end - start)
    solveTime.append(end - start)
    control = np.copy(mpcController.solver.us_squash[0])
    simulator.simulateStep(control)
    t += dtSimulator

time = np.array([i * dt / 1000.0 for i in range(len(solver.us))])
# us_plot_tg = np.vstack(solver.us_squash).T
# PlotControlsGroup(us_plot_tg, time, 6)

time_mpc = np.array([i * dt / 1000.0 for i in range(len(simulator.controls))])
us_plot_mpc = np.vstack(simulator.controls).T
PlotControlsGroup(us_plot_mpc, time_mpc, 6)
# showPlots()

# print(simulator.states[-1])
print("Average update time: ", sum(updateTime) / len(updateTime))
print("Average solving time: ", sum(solveTime) / len(solveTime))

if WITHDISPLAY:
    robot = example_robot_data.load(trajectory.robot_model.name)

    display = crocoddyl.MeshcatDisplay(robot)
    display.rate = -1
    display.freq = 1

    # # Define a 90-degree rotation matrix around the Y-axis
    # rotation_90_y = np.array([
    #     [0, 0, 1, 0],  # X-axis becomes Z-axis
    #     [0, 1, 0, 0],  # Y-axis remains Y-axis
    #     [-1, 0, 0, 0],  # Z-axis becomes -X-axis
    #     [0, 0, 0, 1]
    # ])

    # # Define the translation matrix (position of the camera)
    # translation = np.array([
    #     [1, 0, 0, 10],  # X-axis translation
    #     [0, 1, 0, 100],  # Y-axis translation
    #     [0, 0, 1, 10],   # Z-axis translation (zoom level)
    #     [0, 0, 0, 1]
    # ])

    # # Combine rotation and translation into one transform matrix
    # camera_position = translation @ rotation_90_y

    # # Apply the transformation to set the initial camera view
    # display.viewer["/Cameras/default"].set_transform(camera_position)

    # Define the camera position, target, and up vector
    camera_position = np.array([-2, -1, 3])
    target_position = np.array([0, 0, 0])
    up_vector = np.array([0, 0, 1])

    # Apply the camera view using look_at
    display.viewer["/Cameras/default"].look_at(
        camera_position, target_position, up_vector)

    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
