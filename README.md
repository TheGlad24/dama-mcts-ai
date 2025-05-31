# 🧠 Dama MCTS AI

An intelligent agent that plays the traditional **Dama (Checkers)** board game using **Monte Carlo Tree Search (MCTS)**. This project simulates full games with proper capture rules, evaluates agent performance over 1000 games, and generates visual output including a gameplay GIF and performance plots.

---

## 🎯 Features

- ✅ Full Dama game engine (8×8 board)
- ✅ **Multi-jump capture logic** (mandatory captures)
- ✅ **MCTS agent** with simulation-based decision making
- ✅ Gameplay **visualized and saved as animated GIF**
- ✅ Performance metrics over 1000 games:
  - Win rate bar chart
  - Move count distribution

---

## 📁 Outputs

Saved inside the `dama_outputs/` folder:
- `sample_game.gif` – Animated gameplay
- `win_counts.png` – Win count across 1000 games
- `move_lengths.png` – Distribution of game lengths

---

## 🛠️ How to Run

### Requirements:
- Python 3.x
- `matplotlib`
- `pillow` (for GIF saving)

### Run the script:

```bash
python Ganesha_DamaGame.py
