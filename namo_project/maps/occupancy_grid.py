import numpy as np

def generate_deadlock_map(grid_size):
    """
    Generates a hardcoded occupancy grid map of shape (grid_size, grid_size)
    representing a corridor deadlock scenario.
    
    Grid legend:
    0 = Free Cell (.)
    1 = Wall (#)
    2 = Movable Box (X)
    3 = Robot A (A)
    4 = Robot B (B)
    """
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    if grid_size == 1:
        # A 1x1 grid has only one cell. If it has a deadlock, it is occupied by an obstacle/box.
        grid[0, 0] = 2
        
    elif grid_size == 3:
        # 3x3 layout:
        # # # #
        # A X B
        # # # #
        grid[0, :] = 1
        grid[2, :] = 1
        grid[1, 0] = 3  # Robot A
        grid[1, 1] = 2  # Movable Box
        grid[1, 2] = 4  # Robot B
        
    elif grid_size == 5:
        # 5x5 layout:
        # # # # #
        # # . . #
        # A . X B
        # # . . #
        # # # # #
        grid[0, :] = 1
        grid[4, :] = 1
        grid[:, 0] = 1
        grid[:, 4] = 1
        
        # Walls to narrow down the corridor at row 2
        grid[1, 1] = 1
        grid[1, 3] = 1
        grid[3, 1] = 1
        grid[3, 3] = 1
        
        grid[2, 1] = 3  # Robot A
        grid[2, 2] = 2  # Movable Box
        grid[2, 3] = 4  # Robot B
        
    elif grid_size == 10:
        # 10x10 layout (corridor at row 4)
        grid[0, :] = 1
        grid[9, :] = 1
        grid[:, 0] = 1
        grid[:, 9] = 1
        for col in range(2, 8):
            grid[3, col] = 1
            grid[5, col] = 1
        grid[4, 1] = 3  # Robot A
        grid[4, 8] = 4  # Robot B
        grid[4, 4] = 2  # Movable Box
        
    else:
        # Generic border wall fallback
        grid[0, :] = 1
        grid[grid_size - 1, :] = 1
        grid[:, 0] = 1
        grid[:, grid_size - 1] = 1
        mid = grid_size // 2
        grid[mid, 1] = 3
        grid[mid, mid] = 2
        grid[mid, grid_size - 2] = 4
        
    return grid

def print_grid_ascii(grid_2d, size):
    """
    Prints an ASCII visualization of the 2D grid.
    """
    symbols = {
        0: ".",   # Free
        1: "#",   # Wall
        2: "X",   # Movable box
        3: "A",   # Robot A
        4: "B"    # Robot B
    }
    
    print(f"\n--- {size}x{size} NAMO Deadlock Scenario Grid ---")
    for r in range(size):
        row_str = []
        for c in range(size):
            val = grid_2d[r, c]
            row_str.append(symbols.get(val, "?"))
        print(" ".join(row_str))
    print("-" * (2 * size + 8))

if __name__ == "__main__":
    for size in [1, 3, 5, 10]:
        grid = generate_deadlock_map(size)
        print_grid_ascii(grid, size)
