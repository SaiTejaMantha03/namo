import argparse
import json
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data


DEFAULT_CONFIG_PATH = Path(__file__).with_name("scenario_config.json")


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cell_center(cell_x: int, cell_y: int, cell_size: float) -> tuple[float, float]:
    return (cell_x + 0.5) * cell_size, (cell_y + 0.5) * cell_size


def world_to_cell(position_xy: tuple[float, float], cell_size: float) -> tuple[int, int]:
    return int(position_xy[0] // cell_size), int(position_xy[1] // cell_size)


def create_box(position, half_extents, mass, color):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def create_cylinder(position, radius, height, mass, color, friction):
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color
    )
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )
    p.changeDynamics(body_id, -1, lateralFriction=friction, linearDamping=0.2, angularDamping=0.9)
    return body_id


def build_world(config: dict):
    world_cfg = config["world"]
    robot_cfg = config["robot"]
    box_cfg = config["box"]
    corridor_cfg = config["corridor"]
    render_cfg = config["render"]

    grid_size = world_cfg["grid_size"]
    cell_size = world_cfg["cell_size"]
    wall_height = world_cfg["wall_height"]
    wall_thickness = world_cfg["wall_thickness"]

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(world_cfg["time_step"])
    p.loadURDF("plane.urdf")

    floor_half = grid_size * cell_size * 0.5
    boundary_specs = [
        ([floor_half, -wall_thickness, wall_height * 0.5], [floor_half, wall_thickness, wall_height * 0.5]),
        ([floor_half, grid_size + wall_thickness, wall_height * 0.5], [floor_half, wall_thickness, wall_height * 0.5]),
        ([-wall_thickness, floor_half, wall_height * 0.5], [wall_thickness, floor_half, wall_height * 0.5]),
        ([grid_size + wall_thickness, floor_half, wall_height * 0.5], [wall_thickness, floor_half, wall_height * 0.5]),
    ]
    for position, half_extents in boundary_specs:
        create_box(position, half_extents, 0, render_cfg["boundary_color"])

    corridor_row = corridor_cfg["row"]
    corridor_y = cell_center(0, corridor_row, cell_size)[1]
    left_opening = corridor_cfg["left_opening_x"]
    right_opening = corridor_cfg["right_opening_x"]
    corridor_half_length = (right_opening - left_opening) * 0.5
    corridor_center_x = (left_opening + right_opening) * 0.5

    wall_offset = corridor_cfg["wall_offset"]
    wall_half_extents = [corridor_half_length, wall_thickness, wall_height * 0.5]
    create_box(
        [corridor_center_x, corridor_y - wall_offset, wall_height * 0.5],
        wall_half_extents,
        0,
        render_cfg["wall_color"],
    )
    create_box(
        [corridor_center_x, corridor_y + wall_offset, wall_height * 0.5],
        wall_half_extents,
        0,
        render_cfg["wall_color"],
    )

    robot_a_start = cell_center(*robot_cfg["start_a_cell"], cell_size)
    robot_b_start = cell_center(*robot_cfg["start_b_cell"], cell_size)
    box_start = cell_center(*box_cfg["start_cell"], cell_size)

    robot_a = create_cylinder(
        [robot_a_start[0], robot_a_start[1], robot_cfg["height"] * 0.5],
        robot_cfg["radius"],
        robot_cfg["height"],
        robot_cfg["mass"],
        render_cfg["robot_a_color"],
        robot_cfg["friction"],
    )
    robot_b = create_cylinder(
        [robot_b_start[0], robot_b_start[1], robot_cfg["height"] * 0.5],
        robot_cfg["radius"],
        robot_cfg["height"],
        robot_cfg["mass"],
        render_cfg["robot_b_color"],
        robot_cfg["friction"],
    )

    box = create_box(
        [box_start[0], box_start[1], box_cfg["height"] * 0.5],
        [box_cfg["half_extent"], box_cfg["half_extent"], box_cfg["height"] * 0.5],
        box_cfg["mass"],
        render_cfg["box_color"],
    )
    p.changeDynamics(
        box,
        -1,
        lateralFriction=box_cfg["friction"],
        rollingFriction=0.0,
        spinningFriction=0.0,
    )

    return {
        "robot_a": robot_a,
        "robot_b": robot_b,
        "box": box,
        "goals": {
            robot_a: cell_center(*robot_cfg["goal_a_cell"], cell_size),
            robot_b: cell_center(*robot_cfg["goal_b_cell"], cell_size),
        },
    }


def drive_robot(robot_id: int, goal_xy: tuple[float, float], speed: float = 1.2):
    position, _ = p.getBasePositionAndOrientation(robot_id)
    dx = goal_xy[0] - position[0]
    dy = goal_xy[1] - position[1]
    distance = math.hypot(dx, dy)

    if distance < 0.08:
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        return distance

    scale = min(speed, distance * 1.4) / max(distance, 1e-6)
    vx = dx * scale
    vy = dy * scale
    yaw = math.atan2(dy, dx)
    orientation = p.getQuaternionFromEuler([0, 0, yaw])
    p.resetBasePositionAndOrientation(robot_id, [position[0], position[1], position[2]], orientation)
    p.resetBaseVelocity(robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, 0])
    return distance


def collect_contacts(a_id: int, b_id: int, box_id: int):
    return {
        "robot_robot": len(p.getContactPoints(a_id, b_id)),
        "robot_a_box": len(p.getContactPoints(a_id, box_id)),
        "robot_b_box": len(p.getContactPoints(b_id, box_id)),
    }


def run(gui: bool, steps: int, config: dict):
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)

    world_cfg = config["world"]
    robot_cfg = config["robot"]
    cell_size = world_cfg["cell_size"]

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=config["render"]["camera_distance"],
            cameraYaw=0,
            cameraPitch=-75,
            cameraTargetPosition=[world_cfg["grid_size"] * 0.5, world_cfg["grid_size"] * 0.5, 0.0],
        )

    world = build_world(config)
    robot_a = world["robot_a"]
    robot_b = world["robot_b"]
    box_id = world["box"]
    goals = world["goals"]

    print("Scenario: A ---> corridor <--- B")
    print("Expected outcome: two robots contest the same one-cell corridor and can deadlock.")
    print(f"Config file: {config['meta']['name']}")

    stalled_steps = 0
    for step in range(steps):
        dist_a = drive_robot(robot_a, goals[robot_a], speed=robot_cfg["speed"])
        dist_b = drive_robot(robot_b, goals[robot_b], speed=robot_cfg["speed"])
        p.stepSimulation()

        contacts = collect_contacts(robot_a, robot_b, box_id)
        moving = dist_a > 0.08 or dist_b > 0.08
        blocked = contacts["robot_robot"] > 0 or (contacts["robot_a_box"] > 0 and contacts["robot_b_box"] > 0)

        if moving and blocked:
            stalled_steps += 1
        else:
            stalled_steps = 0

        if step % config["logging"]["print_every_steps"] == 0:
            box_pos, _ = p.getBasePositionAndOrientation(box_id)
            pos_a, _ = p.getBasePositionAndOrientation(robot_a)
            pos_b, _ = p.getBasePositionAndOrientation(robot_b)
            print(
                f"step={step:04d} "
                f"A=({pos_a[0]:.2f},{pos_a[1]:.2f}) cell={world_to_cell((pos_a[0], pos_a[1]), cell_size)} "
                f"B=({pos_b[0]:.2f},{pos_b[1]:.2f}) cell={world_to_cell((pos_b[0], pos_b[1]), cell_size)} "
                f"box=({box_pos[0]:.2f},{box_pos[1]:.2f}) cell={world_to_cell((box_pos[0], box_pos[1]), cell_size)} "
                f"contacts={contacts}"
            )

        if stalled_steps > config["deadlock"]["stall_steps"]:
            print(f"Deadlock reproduced near step {step}.")
            break

        if gui:
            time.sleep(world_cfg["time_step"])

    box_pos, _ = p.getBasePositionAndOrientation(box_id)
    pos_a, _ = p.getBasePositionAndOrientation(robot_a)
    pos_b, _ = p.getBasePositionAndOrientation(robot_b)
    print(
        "Final state:",
        {
            "robot_a": (round(pos_a[0], 2), round(pos_a[1], 2)),
            "robot_b": (round(pos_b[0], 2), round(pos_b[1], 2)),
            "box": (round(box_pos[0], 2), round(box_pos[1], 2)),
        },
    )
    p.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal NAMO deadlock prototype in PyBullet.")
    parser.add_argument("--gui", action="store_true", help="Run with the PyBullet GUI.")
    parser.add_argument("--steps", type=int, default=None, help="Number of simulation steps.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the scenario JSON config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    steps = args.steps if args.steps is not None else config["world"]["sim_steps"]
    run(gui=args.gui, steps=steps, config=config)
