import numpy as np
from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet

class ArcAgent:
    def __init__(self):
        """
        You may add additional variables to this init method. Be aware that it gets called only once
        and then the make_predictions method will get called several times.
        """
        """
                making rule sets so that i can apply them easier later. 
                this is giving the agent a 'helping hand'
                """
        self.debug = True
        self.current_rule = None
        self.rules = [
            "rotate_90","rotate_180","rotate_270","flip_up_down",
            "flip_left_right","transpose","crop_nonzero","left_right_and",
            "left_right_or","left_right_nor","top_bottom_and","top_bottom_or",
            "top_bottom_nor","no_change",  ### above are the rules from Milestone A/B
            ##Module 1 = geometry, Module 2 = color maps, Module 3 = cropping

            ##milestone C additions
            "tile", "scale", "mirror_tile", #Module 4: tiling & scaling;  # 62c24649
            "sep_and", "sep_or", "sep_xor", "sep_nor",  #Module 6: partition on a divider line, then SEARCH the combination rule (and/or/xor/...) against training
            "sep_nand", "sep_first_only", "sep_last_only",
            "sep_overlay"

        ]
        self.factor = None   #(rows, cols) scale factor, grabbing from the training pairs so we can rescale later



    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Write the code in this method to solve the incoming ArcProblem.
        Your agent will receive 1 problem at a time.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer (even on the public problems).

        Also, if you return more than 3 predictions in the list it
        is considered an ERROR and the test will be automatically
        marked as INCORRECT.

        A maximum of three distinct predictions is returned.

        agent tries 3 general rules:
        1. rotate the grid 180 degrees.
        2. crop empty space around the non-black object.
        3. learn a color replacement rule.

        then, a rule is only used if it works for every training example.
        """

        predictions, training  = [], []

        #grab the training examples the same way we did in Milestone A
        for pair in arc_problem.training_set():
            train_input = pair.get_input_data().data()
            train_output = pair.get_output_data().data()
            training.append((train_input, train_output))

        test_input = arc_problem.test_set().get_input_data().data()
        self.current_problem = arc_problem.problem_name()
        self.current_rule = "setup"
        self.log(f"##### {self.current_problem}: {len(training)} training pairs, test input shape {test_input.shape}#####")

        self.factor = self.find_factor_value(training)
        self.current_rule = "setup"
        self.log("inferred factor:", self.factor)

        #testing out each of the rules defined earlier one by one
        for rule_name in self.rules:
            if self.rule_works(training, rule_name): #check if the rules by themself give the correct answer
                answer = self.apply_rule(rule_name, test_input)
                self.log("PASSES every training pair by itself")
                self.add_answer(predictions, answer)
            else:
                color_map = self.learn_color_map(training, rule_name)
                if color_map is not None:
                    answer = self.apply_rule(rule_name, test_input)
                    if answer is not None:
                        answer = self.change_out_colors(answer, color_map)
                        self.add_answer(predictions, answer)

        return predictions[:3]

    def apply_rule(self, rule_name, grid):
        """
        apply each rule to the grid and if it works, return it
        """
        self.current_rule = rule_name
        if rule_name == "no_change":
            return grid.copy()
        if rule_name == "rotate_90":
            return np.rot90(grid, 1)
        if rule_name == "rotate_180":
            return np.rot90(grid, 2)
        if rule_name == "rotate_270":
            return np.rot90(grid, 3)
        if rule_name == "flip_up_down":
            return np.flipud(grid)
        if rule_name == "flip_left_right":
            return np.fliplr(grid)
        if rule_name == "transpose":
            return grid.T
        if rule_name == "crop_nonzero":
            return self.crop_nonzero(grid)
        if rule_name.startswith("left_right_"):
            operation = rule_name.replace("left_right_", "")
            return self.compare_halves(grid, "left_right", operation)
        if rule_name.startswith("top_bottom_"):
            operation = rule_name.replace("top_bottom_", "")
            return self.compare_halves(grid, "top_bottom", operation)

        if rule_name in ("tile", "scale", "mirror_tile"):
            return self.resize_grid_by_factor(grid, rule_name)
        if rule_name.startswith("sep_"):
            print('placeholder for 3428a4f5')
            return self.combine_grids(grid, rule_name.replace("sep_", ""))


        return None

    def rule_works(self, training, rule_name):
        """
        check if the rule gives the right output, and if so, reutrn true
        """
        for train_input, train_output in training:
            answer = self.apply_rule(rule_name, train_input)
            if answer is None:
                return False
            if not np.array_equal(answer, train_output):
                return False

        return True

    def add_answer(self, predictions, answer):
        """
        add the answer to the predictions list as long as it isnt empty
        """
        if answer is None:
            return
        for existing in predictions:
            if np.array_equal(existing, answer):
                return

        predictions.append(answer)

####TO DO
#write the functions to test the other rules like
####comparing halves of the training image s
    def crop_nonzero(self, grid):
        """Cut away the empty (0) border around everything else."""
        nonzero_cells = np.argwhere(grid != 0)

        if len(nonzero_cells) == 0:
            return None

        first_row, first_col = nonzero_cells.min(axis=0)
        last_row, last_col = nonzero_cells.max(axis=0)

        return grid[first_row:last_row + 1, first_col:last_col + 1].copy()
####crop nonzero
    def compare_halves(self, grid, direction, operation):
        """
        split the grid into two parts and see if they compare at all through ands ors nors
        """
        rows, cols = grid.shape
        if direction == "left_right":
            size = cols // 2
            if size == 0:
                return None
            first_half = grid[:, :size]
            second_half = grid[:, cols - size:]
        else:
            size = rows // 2
            if size == 0:
                return None
            first_half = grid[:size, :]
            second_half = grid[rows - size:, :]

        first_filled = first_half != 0
        second_filled = second_half != 0
        if operation == "and":  #filled in BOTH halves
            result = first_filled & second_filled
        elif operation == "or":  #filled in EITHER half
            result = first_filled | second_filled
        else:  #"nor": filled in NEITHER half
            result = ~(first_filled | second_filled)
        return result.astype(int)

####learning colors???
    #how to learn when colors are just swapped
    def learn_color_map(self, training, rule_name):
        """
        learns basic color swapping, like every yellow becomes a red, etc.
        """
        color_map = {}

        for train_input, train_output in training:
            #answer = self.apply_rule(rule_name, training)
            #print(f'train_input {train_input}, training: {training}')
            answer = self.apply_rule(rule_name, train_input)

            #print(f'ANSWER: {answer}, FLATTENED: {answer.flatten(), TRAIN FLATTTENED: {train_output.flatten()}')
            if answer is None or answer.shape != train_output.shape:
                return None
            for old_color, new_color in zip(answer.flatten(), train_output.flatten()):
                old_color = int(old_color)
                new_color = int(new_color)

                if old_color in color_map and color_map[old_color] != new_color:
                    return None  ##
                color_map[old_color] = new_color
        return color_map

    def change_out_colors(self, grid, color_map):
        """
        change the colors according to the color mpa that was learned earlier
        """
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color

        return result

    ###################################
    #MILESTONE C FUNCTIONS
    ###################################
    def log(self, *things):
        """print only when debugging is on"""
        if not self.debug:
            return
        print(f"  [{self.current_rule}]", *things)

    def background(self, grid):
        #step 1: if there is any black, treat it as the background
        if (grid == 0).any():
            return 0
        #step 2: no black in the grid, so get a count of the other colors in the grid
        color_counts = {}
        for row in grid:
            for pixel in row:
                color = int(pixel)
                if color in color_counts:
                    color_counts[color] += 1
                else:
                    color_counts[color] = 1

        # step 3: find color with the biggest color count
        most_color, color_count = None, 0
        for color, count in color_counts.items():
            if count > color_count:
                most_color, color_count = color, count
        return most_color

    ##module 4 - scaling and tiles
    def find_factor_value(self,training_set):
        """output size / input size, but only if every training pair is the same"""
        factors = set()
        #62c24649: train_input.shape = (3, 3), train_output.shape = (6, 6)
        for train_input, train_output in training_set:
            #62c24649: 6 % 3 = 0 and 6 % 3 = 0  so  both 0 (False), so keep going
            #3de23699: input (7, 7), output (3, 3) -> 3 % 7 = 3 (True), return None
            if train_output.shape[0] % train_input.shape[0] or train_output.shape[1] % train_input.shape[1]:
                self.log(f"infer_factor: {train_input.shape} to  {train_output.shape}  is not whole so its not a factor")
                return None

            #62c24649: (6 // 3, 6 // 3) = (2, 2)  ,, factors = {(2, 2)}
            factors.add((train_output.shape[0] // train_input.shape[0], train_output.shape[1] // train_input.shape[1]))

        if len(factors) != 1:
            self.log("infer_factor: pairs disagree:", factors)
            return None

        factor = factors.pop()
        #(1,1) means the output is same size as input
        return None if factor == (1, 1) else factor


    def resize_grid_by_factor(self, grid, rule):
        ####
        #module 4: grab the 'factor' from the training pairs and resize the grid with the factor instead of assuming its 2x2
        #####
        if self.factor == None:
            return None
        rows, cols = self.factor
        self.log(f"resize_by_factor: {rule} {grid.shape} by {self.factor}")
        print(f'rows: {rows}, cols: {cols}')
        if rule == "tile":
            return np.tile(grid, (rows, cols))
        if rule == "scale":
            return np.kron(grid, np.ones((rows, cols), dtype=grid.dtype))
        grid_rows = []
        for x in range(rows):
            pixel_row = []
            for y in range(cols):
                block = grid
                if x % 2 == 1:
                    block = np.flipud(block)
                if y % 2 == 1:
                    block = np.fliplr(block)
                pixel_row.append(block)
            grid_rows.append(np.hstack(pixel_row))
        return np.vstack(grid_rows)


    ##module 6 - grid partitioning
    def split_grids(self, grid):
    ##dividing the grid into multiple independent grids
        print('test')
        background = self.background(grid)
        color_in_each_row, color_in_each_col = {},{}
        num_rows, num_cols = grid.shape[0], grid.shape[1]

        ##
        for row in range(num_rows):
            first = int(grid[row, 0])
            whole_row_same = (grid[row] == first).all()  #every cell in row r equals its first cell????
            if first != background and whole_row_same:
                if first not in color_in_each_row:
                    color_in_each_row[first] = []  #first time we see this color: start a list
                color_in_each_row[first].append(row)

        for col in range(grid.shape[1]):
            first = int(grid[0, col])
            whole_col_same = (grid[:, col] == first).all()
            if first != background and whole_col_same:
                if first not in color_in_each_col:
                    color_in_each_col[first] = []
                color_in_each_col[first].append(col)

        self.log(f"split_panels: grid {grid.shape}, background {background}, divider rows {color_in_each_row}, divider cols {color_in_each_col}")

        row_options = list(color_in_each_row.values()) + [[]]
        col_options = list(color_in_each_col.values()) + [[]]

        for rows in row_options:
            for cols in col_options:
                if not rows and not cols:
                    continue

                #aplit the grid between the dividers
                panels = []
                for top, bottom in self.spans(grid.shape[0], rows):
                    for left, right in self.spans(grid.shape[1], cols):
                        panels.append(grid[top:bottom, left:right])

                # ---- step 3: only accept the split if every panel is the same size ----
                #   cf98881b, cols [3]:    widths 3 and 10 -> different sizes, try next option
                #   cf98881b, cols [4, 9]: three 4x4 panels -> accept
                shapes = set()
                for p in panels:
                    shapes.add(p.shape)
                if len(panels) >= 2 and len(shapes) == 1 and panels[0].size > 0:
                    self.log(f"split_panels: rows {rows} cols {cols} -> {len(panels)} panels of {panels[0].shape}  ACCEPTED")
                    return panels
                self.log(f"split_panels: rows {rows} cols {cols} -> panel sizes {shapes}  rejected")
            self.log("split_panels: no split works")
            return None

    def spans(self, size, dividers):
        """the leftovers from removing diviers in the grid
          like cf98881b columns or 3428a4f5 rows)
          set the edge of the grid as a divider so it knows where to stop (if two dividers hit eachother))
        """
        result = []
        start = 0
        for d in dividers + [size]:
            if d > start:
                result.append((start, d))
            start = d + 1
        return result


    def combine_grids(self, grid, operation):
        print('test')
        background = self.background(grid)
        panels = self.split_grids(grid)
        if panels is None:
            return

        filled = [p != background for p in panels]
        self.log(f"combine_panels: '{operation}' on {len(panels)} panels, filled cells per panel {[int(f.sum()) for f in filled]}")

        

