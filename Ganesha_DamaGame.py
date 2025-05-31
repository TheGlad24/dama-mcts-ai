import os
import copy
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import defaultdict

# Constants
EMPTY = ' '
WHITE = 'W'
RED = 'R'

# Initial board setup
def initial_board():
    board = [[EMPTY]*8 for _ in range(8)]
    for i in range(3):
        for j in range(8):
            if (i + j) % 2 == 1:
                board[i][j] = RED
    for i in range(5, 8):
        for j in range(8):
            if (i + j) % 2 == 1:
                board[i][j] = WHITE
    return board

# Move logic with full capture support
def get_piece_captures(board, player, x, y, path=None, visited=None):
    if path is None:
        path = [(x, y)]
    if visited is None:
        visited = set()
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    opponent = WHITE if player == RED else RED
    moves = []
    for dx, dy in directions:
        mid_x, mid_y = x + dx, y + dy
        dest_x, dest_y = x + 2 * dx, y + 2 * dy
        if 0 <= mid_x < 8 and 0 <= mid_y < 8 and 0 <= dest_x < 8 and 0 <= dest_y < 8:
            if board[mid_x][mid_y] == opponent and board[dest_x][dest_y] == EMPTY:
                if ((mid_x, mid_y), (dest_x, dest_y)) in visited:
                    continue
                new_board = copy.deepcopy(board)
                new_board[dest_x][dest_y] = new_board[x][y]
                new_board[x][y] = EMPTY
                new_board[mid_x][mid_y] = EMPTY
                new_path = path + [(dest_x, dest_y)]
                new_visited = visited | {((mid_x, mid_y), (dest_x, dest_y))}
                next_captures = get_piece_captures(new_board, player, dest_x, dest_y, new_path, new_visited)
                if next_captures:
                    moves.extend(next_captures)
                else:
                    moves.append(new_path)
    return moves

def get_all_captures(board, player):
    captures = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == player:
                captures.extend(get_piece_captures(board, player, i, j))
    return captures

def get_simple_moves(board, player):
    direction = -1 if player == WHITE else 1
    moves = []
    for i in range(8):
        for j in range(8):
            if board[i][j] == player:
                for dx in [-1, 1]:
                    ni, nj = i + direction, j + dx
                    if 0 <= ni < 8 and 0 <= nj < 8 and board[ni][nj] == EMPTY:
                        moves.append([(i, j), (ni, nj)])
    return moves

def get_legal_moves(board, player):
    captures = get_all_captures(board, player)
    return captures if captures else get_simple_moves(board, player)

def make_move(board, move):
    new_board = copy.deepcopy(board)
    start = move[0]
    for i in range(1, len(move)):
        end = move[i]
        new_board[end[0]][end[1]] = new_board[start[0]][start[1]]
        new_board[start[0]][start[1]] = EMPTY
        if abs(start[0] - end[0]) == 2:
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            new_board[mid_x][mid_y] = EMPTY
        start = end
    return new_board

def has_pieces(board, player):
    return any(player in row for row in board)

# MCTS Agent
class MCTSNode:
    def __init__(self, board, player, parent=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = None

    def expand(self):
        legal_moves = get_legal_moves(self.board, self.player)
        opponent = WHITE if self.player == RED else RED
        for move in legal_moves:
            new_board = make_move(self.board, move)
            child = MCTSNode(new_board, opponent, parent=self)
            child.move = move
            self.children.append(child)

    def is_terminal(self):
        return not has_pieces(self.board, WHITE) or not has_pieces(self.board, RED)

    def simulate(self):
        board = copy.deepcopy(self.board)
        player = self.player
        for _ in range(40):
            moves = get_legal_moves(board, player)
            if not moves:
                return 1 if player != self.player else 0
            board = make_move(board, random.choice(moves))
            player = WHITE if player == RED else RED
        return 0.5

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)

    def best_child(self, c=1.4):
        return max(self.children, key=lambda child:
                   child.wins / (child.visits + 1e-4) + c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-4)))

    def run_simulation(self, n=20):
        for _ in range(n):
            node = self
            while node.children:
                node = node.best_child()
            if not node.is_terminal():
                node.expand()
                if node.children:
                    node = random.choice(node.children)
            result = node.simulate()
            node.backpropagate(result)

    def best_move(self):
        if not self.children:
            self.expand()
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.visits).move

# GUI Board Drawing
def draw_board_gui(board, ax):
    ax.clear()
    light_color = "#f0d9b5"
    dark_color = "#b58863"
    for i in range(8):
        for j in range(8):
            color = light_color if (i + j) % 2 == 0 else dark_color
            ax.add_patch(patches.Rectangle((j, 7 - i), 1, 1, facecolor=color))
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece == WHITE:
                ax.add_patch(patches.Circle((j + 0.5, 7 - i + 0.5), 0.3, color='white', ec='black'))
            elif piece == RED:
                ax.add_patch(patches.Circle((j + 0.5, 7 - i + 0.5), 0.3, color='red', ec='black'))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title("Dama Game - MCTS vs MCTS")

# Run game and capture frames
def simulate_game_frames(mcts_sims=30, max_turns=40):
    frames = []
    board = initial_board()
    current_player = WHITE
    for _ in range(max_turns):
        frames.append(copy.deepcopy(board))
        if not has_pieces(board, WHITE) or not has_pieces(board, RED):
            break
        root = MCTSNode(board, current_player)
        root.run_simulation(mcts_sims)
        move = root.best_move()
        if not move:
            break
        board = make_move(board, move)
        current_player = WHITE if current_player == RED else RED
    frames.append(copy.deepcopy(board))
    return frames

# Animate
frames = simulate_game_frames()
fig, ax = plt.subplots(figsize=(6, 6))
ani = FuncAnimation(fig, lambda i: draw_board_gui(frames[i], ax), frames=len(frames), interval=1000)
plt.show()



# Create output directory
output_dir = "dama_outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Draw Board GUI Function ---
def draw_board_gui(board, ax):
    ax.clear()
    light_color = "#f0d9b5"
    dark_color = "#b58863"
    for i in range(8):
        for j in range(8):
            color = light_color if (i + j) % 2 == 0 else dark_color
            ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1, color=color))
            piece = board[i][j]
            if piece == WHITE:
                ax.add_patch(plt.Circle((j + 0.5, 7 - i + 0.5), 0.3, color='white', ec='black'))
            elif piece == RED:
                ax.add_patch(plt.Circle((j + 0.5, 7 - i + 0.5), 0.3, color='red', ec='black'))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

# --- Save 1 Sample Game as GIF ---
def save_game_gif(frames, filename="dama_game.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ani = FuncAnimation(fig, lambda i: draw_board_gui(frames[i], ax), frames=len(frames), interval=700)
    path = os.path.join(output_dir, filename)
    ani.save(path, writer=PillowWriter(fps=1))
    plt.close()
    print(f"GIF saved to {path}")

# --- Play One Game and Capture Frames ---
def simulate_game_frames(mcts_sims=30, max_turns=40):
    frames = []
    board = initial_board()
    current_player = WHITE
    for _ in range(max_turns):
        frames.append(copy.deepcopy(board))
        if not has_pieces(board, WHITE) or not has_pieces(board, RED):
            break
        root = MCTSNode(board, current_player)
        root.run_simulation(mcts_sims)
        move = root.best_move()
        if not move:
            break
        board = make_move(board, move)
        current_player = WHITE if current_player == RED else RED
    frames.append(copy.deepcopy(board))
    return frames

# --- Evaluate Agents for 1000 Games ---
def evaluate_agents(num_games=1000, mcts_sims=30, max_turns=50):
    win_counts = defaultdict(int)
    move_lengths = []

    for i in range(num_games):
        board = initial_board()
        current_player = WHITE
        moves = 0

        for _ in range(max_turns):
            if not has_pieces(board, WHITE):
                win_counts["RED"] += 1
                break
            if not has_pieces(board, RED):
                win_counts["WHITE"] += 1
                break
            root = MCTSNode(board, current_player)
            root.run_simulation(mcts_sims)
            move = root.best_move()
            if not move:
                win_counts["RED" if current_player == WHITE else "WHITE"] += 1
                break
            board = make_move(board, move)
            current_player = WHITE if current_player == RED else RED
            moves += 1
        else:
            win_counts["DRAW"] += 1
        move_lengths.append(moves)

    return win_counts, move_lengths

# --- Plot and Save Graphs ---
def save_metrics(win_counts, move_lengths):
    # Win Rate Bar Chart
    plt.figure(figsize=(6,4))
    plt.bar(win_counts.keys(), win_counts.values(), color=["skyblue", "salmon", "gray"])
    plt.title("Win Counts (1000 Games)")
    plt.ylabel("Wins")
    plt.savefig(os.path.join(output_dir, "win_counts.png"))
    plt.close()

    # Move Count Histogram
    plt.figure(figsize=(6,4))
    plt.hist(move_lengths, bins=20, color='green', edgecolor='black')
    plt.title("Distribution of Game Lengths")
    plt.xlabel("Number of Moves")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "move_lengths.png"))
    plt.close()
    print("Plots saved to:", output_dir)

# --- Run Everything ---
sample_game = simulate_game_frames()
save_game_gif(sample_game, "sample_game.gif")
win_counts, move_lengths = evaluate_agents(num_games=1000, mcts_sims=30)
save_metrics(win_counts, move_lengths)
