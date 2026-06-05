import time
import math
import argparse
import sys
import subprocess
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from decision.unet_decision_pipeline import UNetDecisionPipeline
from maps.namo_environments import WarehouseEnvironment

def cell_center(cell_x, cell_y, cell_size=1.0):
    return (cell_x + 0.5) * cell_size, (cell_y + 0.5) * cell_size

def create_box(position, half_extents, color):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(
        baseMass=1.5,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )

def create_static_wall(position, half_extents):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.3, 0.3, 0.3, 1.0])
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )

def create_robot(position, color):
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.25, height=0.3)
    visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.25, length=0.3, rgbaColor=color)
    body_id = p.createMultiBody(
        baseMass=2.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )
    p.changeDynamics(body_id, -1, lateralFriction=1.0, linearDamping=0.2, angularDamping=0.9)
    return body_id

def drive_robot(robot_id, target_xy, speed=1.2):
    position, _ = p.getBasePositionAndOrientation(robot_id)
    dx = target_xy[0] - position[0]
    dy = target_xy[1] - position[1]
    distance = math.hypot(dx, dy)

    if distance < 0.12:
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        return distance

    scale = min(speed, distance * 2.0) / max(distance, 1e-6)
    vx = dx * scale
    vy = dy * scale
    yaw = math.atan2(dy, dx)
    orientation = p.getQuaternionFromEuler([0, 0, yaw])
    p.resetBasePositionAndOrientation(robot_id, [position[0], position[1], position[2]], orientation)
    p.resetBaseVelocity(robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, 0])
    return distance

class NAMO3DSimulator:
    def __init__(self, size, grid, start, goal, obstacle, decision, gui=False):
        self.size = size
        self.grid = grid
        self.start = start
        self.goal = goal
        self.obstacle = obstacle
        self.decision = decision
        self.gui = gui
        
    def find_clearing_direction(self):
        ox, oy = self.obstacle
        directions = [
            ((ox, oy + 1), (ox, oy - 1)), # Push South, approach from North
            ((ox, oy - 1), (ox, oy + 1)), # Push North, approach from South
            ((ox + 1, oy), (ox - 1, oy)), # Push East, approach from West
            ((ox - 1, oy), (ox + 1, oy))  # Push West, approach from East
        ]
        
        for clear_cell, approach_cell in directions:
            cx, cy = clear_cell
            ax, ay = approach_cell
            if 0 <= cx < self.size and 0 <= cy < self.size:
                if 0 <= ax < self.size and 0 <= ay < self.size:
                    if self.grid[cy, cx] == 0 and self.grid[ay, ax] == 0:
                        return clear_cell, approach_cell
        return (ox, oy + 1), (ox, oy - 1)

    def run(self):
        mode = p.GUI if self.gui else p.DIRECT
        p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        p.loadURDF("plane.urdf")
        
        cell_size = 1.0
        wall_height = 0.8
        
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=self.size * 1.3,
                cameraYaw=0,
                cameraPitch=-75,
                cameraTargetPosition=[self.size * 0.5, self.size * 0.5, 0.0],
            )
            
        # Spawn layout
        box_id = None
        for r in range(self.size):
            for c in range(self.size):
                x, y = cell_center(c, r, cell_size)
                val = self.grid[r, c]
                if val == 1:
                    create_static_wall([x, y, wall_height*0.5], [0.5, 0.5, wall_height*0.5])
                elif val == 2:
                    box_id = create_box([x, y, 0.2], [0.35, 0.35, 0.2], [0.95, 0.6, 0.1, 1.0])
                    p.changeDynamics(box_id, -1, lateralFriction=0.8)
                    
        # Spawn robot
        s_x, s_y = cell_center(self.start[0], self.start[1])
        robot_id = create_robot([s_x, s_y, 0.15], [0.2, 0.6, 0.9, 1.0])
        
        # Call UNet Decision Pipeline to generate and save the visual heatmap plot
        project_dir = Path(__file__).resolve().parent.parent
        weights_path = str(project_dir / "models" / "namo_unet.pth")
        pipeline = UNetDecisionPipeline(weights_path)
        
        plot_filename = f"sim_{self.size}x{self.size}_decision.png"
        plot_save_path = str(project_dir / "decision" / plot_filename)
        
        pipeline.evaluate_decision(
            self.grid, self.start, self.goal, self.obstacle, 
            push_cost=4.0, save_plot_path=plot_save_path
        )
        print(f" -> Automatically generated and saved plot to: decision/{plot_filename}")

        print(f"\n[Simulation] Executing NAMO {self.decision} on {self.size}x{self.size} grid...")

        # Dynamic Waypoint Generation
        waypoints = []
        if self.decision == "REMOVE":
            clear_cell, approach_cell = self.find_clearing_direction()
            
            grid_bypass = self.grid.copy()
            grid_bypass[self.obstacle[1], self.obstacle[0]] = 2
            _, path_to_approach = pipeline.risk_aware_a_star(grid_bypass, np.zeros((self.size, self.size)), self.start, approach_cell)
            
            waypoints.extend([cell_center(wp[0], wp[1]) for wp in path_to_approach[1:]])
            waypoints.append(cell_center(self.obstacle[0], self.obstacle[1]))
            waypoints.append(cell_center(clear_cell[0], clear_cell[1]))
            
            grid_after_push = self.grid.copy()
            grid_after_push[self.obstacle[1], self.obstacle[0]] = 0
            grid_after_push[clear_cell[1], clear_cell[0]] = 2
            
            _, path_to_goal = pipeline.risk_aware_a_star(grid_after_push, np.zeros((self.size, self.size)), clear_cell, self.goal)
            waypoints.extend([cell_center(wp[0], wp[1]) for wp in path_to_goal[1:]])
            
        else:
            grid_bypass = self.grid.copy()
            grid_bypass[self.obstacle[1], self.obstacle[0]] = 2
            _, path = pipeline.risk_aware_a_star(grid_bypass, np.zeros((self.size, self.size)), self.start, self.goal)
            waypoints = [cell_center(wp[0], wp[1]) for wp in path[1:]]

        # Drive sequence
        current_wp_idx = 0
        for step in range(500):
            if current_wp_idx >= len(waypoints):
                print(f" -> Goal reached dynamically at step {step}!")
                break
                
            target = waypoints[current_wp_idx]
            dist = drive_robot(robot_id, target)
            p.stepSimulation()
            if self.gui: time.sleep(0.01)
            
            if dist < 0.18:
                current_wp_idx += 1
                
        p.disconnect()

def simulate_env(env_name, gui=False):
    project_dir = Path(__file__).resolve().parent.parent
    weights_path = str(project_dir / "models" / "namo_unet.pth")
    pipeline = UNetDecisionPipeline(weights_path)
    
    if env_name == "3x3":
        g3 = np.zeros((3, 3), dtype=int)
        g3[0, :] = 1; g3[2, :] = 1; g3[1, 1] = 2
        decision_3, _, _ = pipeline.evaluate_decision(g3, (0, 1), (2, 1), (1, 1))
        sim = NAMO3DSimulator(3, g3, (0, 1), (2, 1), (1, 1), decision_3, gui)
        sim.run()
        
    elif env_name == "5x5":
        g5 = np.zeros((5, 5), dtype=int)
        g5[0, :] = 1; g5[4, :] = 1; g5[:, 0] = 1; g5[:, 4] = 1
        g5[1, 1] = 1; g5[1, 2] = 1; g5[1, 3] = 1
        g5[3, 1] = 1; g5[3, 2] = 1; g5[3, 3] = 1
        g5[2, 2] = 2
        decision_5, _, _ = pipeline.evaluate_decision(g5, (1, 2), (3, 2), (2, 2))
        sim = NAMO3DSimulator(5, g5, (1, 2), (3, 2), (2, 2), decision_5, gui)
        sim.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="all", choices=["3x3", "5x5", "all"])
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    
    python_bin = sys.executable
    script_path = __file__
    
    if args.env == "all":
        # Launch 3x3 and 5x5 as separate subprocesses to prevent OpenGL context crashes on Mac
        print("Launching 3x3 Simulation Subprocess...")
        cmd3 = [python_bin, script_path, "--env", "3x3"]
        if args.gui: cmd3.append("--gui")
        subprocess.run(cmd3)
        
        print("\nLaunching 5x5 Simulation Subprocess...")
        cmd5 = [python_bin, script_path, "--env", "5x5"]
        if args.gui: cmd5.append("--gui")
        subprocess.run(cmd5)
    else:
        simulate_env(args.env, args.gui)

if __name__ == "__main__":
    main()
