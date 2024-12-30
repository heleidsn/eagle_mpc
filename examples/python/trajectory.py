import sys
import time

import crocoddyl
import example_robot_data
import numpy as np


import eagle_mpc
from eagle_mpc.utils.path import EAGLE_MPC_YAML_DIR

WITHDISPLAY = 'display' in sys.argv

dt = 10  # ms
useSquash = True
# useSquash = False

robotName = 'hexacopter370_flying_arm_3'
trajectoryName = 'eagle_catch_nc'

trajectory = eagle_mpc.Trajectory()
trajectory.autoSetup(EAGLE_MPC_YAML_DIR + "/" + robotName +
                     "/trajectories/" + trajectoryName + ".yaml")
problem = trajectory.createProblem(dt, useSquash, "IntegratedActionModelEuler")

if useSquash:
    solver = eagle_mpc.SolverSbFDDP(problem, trajectory.squash)
else:
    solver = crocoddyl.SolverBoxFDDP(problem)

solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.solve([], [], maxiter=100)

# if WITHDISPLAY:
#     robot = example_robot_data.load(trajectory.robot_model.name)
#     display = crocoddyl.MeshcatDisplay(robot)

#     # display = crocoddyl.GepettoDisplay(robot)
#     display.displayFromSolver(solver)

if WITHDISPLAY:
    robot = example_robot_data.load(trajectory.robot_model.name)

    rate = -1
    freq = 1
    cameraTF = [-0.03, 4.4, 2.3, 0, 0.7071, 0, 0.7071]

    try:
        import gepetto

        gepetto.corbaserver.Client()

        display = crocoddyl.GepettoDisplay(
            robot, rate, freq, cameraTF, floor=False)
    except Exception as e:
        display = crocoddyl.MeshcatDisplay(robot, rate, freq, cameraTF)
        print("Failed to load gepetto: {}".format(e))

    while True:
        display.displayFromSolver(solver)
        # time.sleep(10.0)
