from .planner_base import PathPlannerBase 
from ..utils.pose import Pose
from ..utils.motion import Path

from math import sqrt
from queue import PriorityQueue
from sys import float_info
import time

# implementation
class AstarKevinGrid:
    def __init__(self, grid, diagonal_motion, heuristic):
        self.grid = grid
        self.diagonal_motion = diagonal_motion
        self.heuristics = {"euclidean" : self.euclidean_heuristic, "manhattan" : self.manhattan_heuristic}
        
        if heuristic not in self.heuristics:
            raise NotImplementedError(heuristic + " heuristic is not supported")

        self.heuristic = self.heuristics[heuristic]

        # map to keep track of path inheritance
        self.came_from = {}

    def euclidean_heuristic(self, start_gridcell, end_gridcell):
        start_x, start_y = self.grid.grid_to_world(start_gridcell)
        end_x, end_y = self.grid.grid_to_world(end_gridcell)

        return sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

    def manhattan_heuristic(self, start_gridcell, end_gridcell):
        start_row_idx, start_col_idx = start_gridcell
        end_row_idx, end_col_idx = end_gridcell

        return abs(end_row_idx - start_row_idx) + abs(end_col_idx - start_col_idx)

    def get_valid_neighbors(self, gridcell):
        neighbors = []

        if self.diagonal_motion:
            modifiers = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        else:
            modifiers = [(0,-1), (0,1), (-1,0), (1,0)]

        for delta in modifiers:            
            test_gridcell = (gridcell[0] + delta[0], gridcell[1] + delta[1])
            if self.grid.is_in_bounds(test_gridcell) and not self.grid.is_occupied(test_gridcell):
                neighbors.append(test_gridcell)

        return neighbors

    # take in a start and goal Pose
    def plan(self, start, goal):

        # cost for taking a step to a new cell (tunable)
        step_cost = 1

        # do all planning in gridspace
        start_gridcell = self.grid.world_to_grid((start.x, start.y))
        goal_gridcell = self.grid.world_to_grid((goal.x, goal.y))


        # cost from start node to specified node
        # lack of presence in this dict indicates an infinite cost
        g_scores = {start_gridcell : 0, goal_gridcell : float_info.max}

        # keep track of which cells are in the queue (this is searchable whereas the queue is not)
        open_set = set()
        open_set.add(start_gridcell)

        # priority queue for expanding cells
        gridcell_queue = PriorityQueue()
        gridcell_queue.put((0,start_gridcell))

        while not gridcell_queue.empty():
            _, current_gridcell = gridcell_queue.get()
            open_set.remove(current_gridcell)

            # if we've reached the goal, generate our path
            if current_gridcell[0] == goal_gridcell[0] and current_gridcell[1] == goal_gridcell[1]:
                return self.backtrace(current_gridcell)

            # for every neighbor
            for neighbor_gridcell in self.get_valid_neighbors(current_gridcell):

                # cost for a cell is the value in that cell plus the cost of the single step to get there
                neighbor_cell_cost = self.grid.data[neighbor_gridcell] + step_cost

                # if neighbor not in g_scores it means that we haven't expanded yet (g_score is implicitly infinite)
                tentative_g_score = g_scores[current_gridcell] + neighbor_cell_cost
                if neighbor_gridcell not in g_scores or tentative_g_score < g_scores[neighbor_gridcell]:
                    g_scores[neighbor_gridcell] = tentative_g_score

                    # save parent for backtracing final path
                    self.came_from[neighbor_gridcell] = current_gridcell
                    
                    if neighbor_gridcell not in open_set:  
                        # sort gridcell queue by f = g + h
                        gridcell_queue.put((g_scores[neighbor_gridcell] + self.heuristic(neighbor_gridcell, goal_gridcell), neighbor_gridcell))
                        open_set.add(neighbor_gridcell)

        raise RuntimeError("Planning failed")
    
    def backtrace(self, gridcell):
        poses = [self.gridcell_to_pose(gridcell)]

        while gridcell in self.came_from:
            gridcell = self.came_from[gridcell]
            poses.insert(0, self.gridcell_to_pose(gridcell))

        return Path(poses=poses)

    def gridcell_to_pose(self, gridcell):
        x, y = self.grid.grid_to_world(gridcell)

        return Pose(x, y)
    

# factory
class AstarKevinPlanner(PathPlannerBase):
    def __init__(self, **planner_config):
        super().__init__()

        self.impl = None

        if planner_config.get("grid", None):
            self.impl = AstarKevinGrid(**planner_config)
        else:
            raise NotImplementedError("Kevin's A* implementation only works on grids")

    def plan(self, start, goal):
        start_time = time.time()
        self.latest_path = self.impl.plan(start, goal)
        self.planning_time = time.time() - start_time
        return self.latest_path