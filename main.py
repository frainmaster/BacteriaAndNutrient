import random
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

# factors that affect the effectiveness of the test
BACTERIA_DYING_CHANCE = .005
BACTERIA_STEPS = 200
BOARD_SIZE = 200

OUTPUT_PATH = 'output'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
IMG_DUMP = f'{OUTPUT_PATH}/img_dump'
test_cnt = len([_ for _ in os.listdir(OUTPUT_PATH) if _.endswith('avi')])
# OUTPUT_PATH = f'{OUTPUT_PATH}/test{test_cnt+1}'


def get_random_pos(bac_pos=None) -> (int, int):
    """
    Get random position for initialization of bacterias and nutrients
    If the bac_pos is empty, the bacteria is created within radius of given position.
    :param bac_pos: radius of area for bacteria to be created.
    :return:
    """
    multiply_radius = 10
    if bac_pos:
        return random.randint(bac_pos[0] - multiply_radius, bac_pos[0] + multiply_radius), \
               random.randint(bac_pos[1] - multiply_radius, bac_pos[1] + multiply_radius)
    return random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1)


class Nutrient:
    lives = 50

    def __init__(self, _id: int):
        self.pos = get_random_pos()
        self.is_alive = True
        self._id = _id

    def __repr__(self):
        if self.is_alive:
            return f'N{self._id}'
        else:
            return f'N{self._id} (dead)'

    def get_visitor(self):
        self.lives -= 1
        if not self.lives:
            self.kill()

    def kill(self):
        self.is_alive = False
        self.lives = 0


class Bacteria:
    dying_chance = BACTERIA_DYING_CHANCE
    multiplying_chance = .9
    steps = BACTERIA_STEPS
    m_left, m_right, m_down, m_up = (0, -1), (0, 1), (-1, 0), (1, 0)
    moves_list = [m_left, m_right, m_down, m_up]
    move_direction = 0, 0

    def __init__(self, _id: int, bac_pos: (int, int) = None):
        self.pos = get_random_pos(bac_pos)
        self.is_alive = True
        self._id = _id
        self.closest_nutrient = None
        self.nutrient_dist = 0, 0
        self.has_touched_nutrient = False

    def __repr__(self):
        if self.is_alive:
            return f'B{self._id}'
        else:
            return f'B{self._id} (dead)'

    def find_closest_nutrient(self, nutrients_list: list):
        def total_moves(move_x: int, move_y: int) -> int:
            return abs(move_x) + abs(move_y)

        # if all nutrients died
        if not nutrients_list:
            self.closest_nutrient = None
            self.nutrient_dist = 0, 0
            return

        dist = BOARD_SIZE, BOARD_SIZE
        nutrient_id = 0
        for num, nutrient in enumerate(nutrients_list):
            n_dist = nutrient.pos
            # d = self.pos[0] - n_dist[0], self.pos[1] - n_dist[1]
            d = n_dist[0] - self.pos[0], n_dist[1] - self.pos[1]
            if total_moves(d[0], d[1]) < total_moves(dist[0], dist[1]):
                dist = d
                nutrient_id = num
        self.closest_nutrient = nutrients_list[nutrient_id]
        self.nutrient_dist = dist

    def move(self, m_type=None) -> (int, int):
        """
        Bacteria's movement function. The bacteria can decide to move randomly or towards the nutrient.
        :param m_type: Type of movement. 'n' for directed to nutrient, 'r' for random.
        :return: Tuple of the new position of the bacteria.
        """
        def get_direction(n_dist):
            dir_x, dir_y = n_dist
            move_x, move_y = 0, 0
            if dir_x < 0:
                move_x = -1
            elif dir_x == 0:
                move_x = 0
            elif dir_x > 0:
                move_x = 1
            if dir_y < 0:
                move_y = -1
            elif dir_y == 0:
                move_y = 0
            elif dir_y > 0:
                move_y = 1
            # check if both axis of direction is given, to avoid moving diagonally
            if all((move_x, move_y)):
                move_x = 0  # prioritize vertical movement
            return move_x, move_y
        move_survival_chance = random.random()
        if move_survival_chance < self.dying_chance:
            self.kill()
            return -1, -1
        if m_type == "n":
            self.move_direction = get_direction(self.nutrient_dist)
        elif m_type == "r":
            self.move_direction = random.choice(self.moves_list)
        new_pos = self.pos[0] + self.move_direction[0], self.pos[1] + self.move_direction[1]
        return new_pos

    def accept_move(self, new_pos):
        self.pos = new_pos
        self.nutrient_dist = (self.nutrient_dist[0] - self.move_direction[0],
                              self.nutrient_dist[1] - self.move_direction[1])
        self.steps -= 1
        if not self.steps:
            self.kill()

    def multiply(self) -> bool:
        # get a random float from 0 to 1, if it is <.9 then bacteria will multiply
        multiply_chance = random.random()
        return multiply_chance < self.multiplying_chance

    def kill(self):
        self.is_alive = False
        self.steps = 0


class Ecosystem:
    def __init__(self):
        self.nutrients = []
        self.nutrients_pos = []
        self.bacterias = []
        self.bacterias_pos = []
        self.bacteria_cnt = 0
        self.init_nutrients()
        self.init_bacterias()
        # self.bacterias_finds_closest_nutrient()
        self.nutrient_collision = 0
        self.total_steps = 0

    def init_nutrients(self):
        while len(self.nutrients) < 3:
            new_nutrient = Nutrient(len(self.nutrients))
            n_pos = new_nutrient.pos
            if n_pos not in self.nutrients_pos and n_pos not in self.bacterias_pos:
                self.nutrients_pos.append(n_pos)
                self.nutrients.append(new_nutrient)

    def init_bacterias(self):
        while len(self.bacterias) < 5:
            self.create_bacteria()

    def create_bacteria(self, bac_pos=None):
        new_bacteria = Bacteria(self.bacteria_cnt, bac_pos)
        b_pos = new_bacteria.pos
        if b_pos not in self.bacterias_pos and b_pos not in self.nutrients_pos:
            new_bacteria.find_closest_nutrient(self.nutrients)
            self.bacterias_pos.append(b_pos)
            self.bacterias.append(new_bacteria)
            self.bacteria_cnt += 1

    def generate_matplotlib(self, m_count, vid_writer):
        x_pts, y_pts = [], []
        for bacteria in self.bacterias:
            y_pts.append(bacteria.pos[0])
            x_pts.append(bacteria.pos[1])
        plt.scatter(np.array(x_pts), np.array(y_pts), marker='x')

        x_pts, y_pts = [], []
        for nutrient in self.nutrients:
            y_pts.append(nutrient.pos[0])
            x_pts.append(nutrient.pos[1])
        plt.scatter(np.array(x_pts), np.array(y_pts))

        plt.title(f'Move {m_count}')
        plt.xlim([0, BOARD_SIZE])
        plt.ylim([0, BOARD_SIZE])
        img_path = f'{IMG_DUMP}/img{m_count}.png'
        plt.savefig(img_path)
        plt.clf()

        img = cv2.imread(img_path)
        vid_writer.write(img)

    def bacterias_finds_closest_nutrient(self):
        for bacteria in self.bacterias:
            bacteria.find_closest_nutrient(self.nutrients)

    def bacteria_moves(self, bacteria: Bacteria, move_tries=4):
        if not move_tries:
            # print(f'{bacteria} cannot move. All legal moves occupied :(')
            return
        # the bacteria has a 10% chance of wanting to move randomly or towards the nutrient.
        move_chance = random.random()
        move_chance_threshold = .1
        # if the bacteria has touched a nutrient, it has 50% chance of wanting to move randomly
        if bacteria.has_touched_nutrient:
            move_chance_threshold = .5
        move_type = "r" if move_chance < move_chance_threshold else "n"
        # move_type = "n"
        new_pos = bacteria.move(move_type)
        while any(a for a in new_pos if not 0 <= a < BOARD_SIZE) and bacteria.is_alive:
            new_pos = bacteria.move(move_type)
        if new_pos == (-1, -1):
            bacteria.kill()
            return
        if not bacteria.is_alive:
            return

        # if the new position is not occupied in the map
        # if map_new_pos == 0:
        if new_pos not in self.bacterias_pos:
            # # swap position i.e. bacteria moves to empty square
            self.bacterias_pos.remove(bacteria.pos)
            self.bacterias_pos.append(new_pos)
            bacteria.accept_move(new_pos)
            # bacteria.find_closest_nutrient(self.nutrients)
            # if the new position is a nutrient
            if new_pos in self.nutrients_pos:
                bacteria.has_touched_nutrient = True
                # decrement nutrient lives
                visited_nutrient = self.nutrients[self.nutrients_pos.index(new_pos)]
                visited_nutrient.get_visitor()
                self.nutrient_collision += 1
                if bacteria.multiply():
                    # multiplication will create a new bacteria at a random point
                    try:
                        self.create_bacteria(new_pos)
                    except IndexError:
                        print('Bacteria is created outside of ecosystem :(')
        else:
            self.bacteria_moves(bacteria, move_tries=move_tries-1)

    def simulate(self):
        for bacteria in self.bacterias:
            if bacteria.closest_nutrient and not bacteria.closest_nutrient.is_alive:
                bacteria.find_closest_nutrient(self.nutrients)
            self.bacteria_moves(bacteria)
        self.check_bacterias_and_nutrients()
        self.total_steps += 1

    def check_bacterias_and_nutrients(self):
        for b in range(len(self.bacterias)):
            if not self.bacterias[b].is_alive:
                b_pos = self.bacterias[b].pos
                self.bacterias[b] = None
                if b_pos in self.bacterias_pos:
                    self.bacterias_pos.remove(b_pos)
        self.bacterias = [b for b in self.bacterias if b]

        for n in range(len(self.nutrients)):
            if not self.nutrients[n].is_alive:
                n_pos = self.nutrients[n].pos
                self.nutrients[n] = None
                self.nutrients_pos[n] = None
        self.nutrients = [n for n in self.nutrients if n]
        self.nutrients_pos = [n for n in self.nutrients_pos if n]

    def final_product(self):
        with open(f'{OUTPUT_PATH}/overview.txt', 'w') as f:
            print(f'Nutrients survived: {len(self.nutrients)}')
            f.write(f'Nutrients survived: {len(self.nutrients)}\n')
            for n in self.nutrients:
                print(f'- {n}, {n.lives} lives left')
                f.write(f'- {n}, {n.lives} lives left\n')
            print(f'Bacterias survived: {len(self.bacterias)}')
            f.write(f'Bacterias survived: {len(self.bacterias)}\n')
            for b in self.bacterias:
                print(f'- {b}, {b.steps} steps left')
                f.write(f'- {b}, {b.steps} steps left\n')
            print()
            f.write('\n')
            print(f'Nutrient collisions: {self.nutrient_collision}')
            print(f'Total bacterias: {self.bacteria_cnt}')
            print(f'Total steps before all bacterias die: {self.total_steps}')
            f.write(f'Nutrient collisions: {self.nutrient_collision}\n')
            f.write(f'Total bacterias: {self.bacteria_cnt}\n')
            f.write(f'Total steps before all bacterias die: {self.total_steps}\n')


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    # else:
    #     for file in os.listdir(OUTPUT_PATH):
    #         os.remove(f'{OUTPUT_PATH}/{file}')

    if not os.path.exists(IMG_DUMP):
        os.mkdir(IMG_DUMP)
    else:
        for file in os.listdir(IMG_DUMP):
            os.remove(f'{IMG_DUMP}/{file}')

    new_eco = Ecosystem()
    out = cv2.VideoWriter(f'{OUTPUT_PATH}/output_video_{test_cnt}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))
    new_eco.generate_matplotlib(0, out)
    print()
    move_cnt = 1
    # the program runs as long as bacterias exist
    while new_eco.bacterias:
        print(f'move {move_cnt}')
        new_eco.simulate()
        new_eco.generate_matplotlib(move_cnt, out)
        move_cnt += 1

    out.release()

    print()
    new_eco.final_product()
