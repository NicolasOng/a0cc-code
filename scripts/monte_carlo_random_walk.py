import random

def mc_sim():
    steps = 0
    max_steps = 10000
    position = 0
    goal = 60
    while position < goal and steps < max_steps:
        move = random.choice([-1, 1])
        position += move
        position = max(0, position)  # Ensure position doesn't go below 0
        steps += 1
    return position >= goal, steps

def main():
    n = 1000
    wins = 0
    total_steps = 0
    for _ in range(n):
        win, steps = mc_sim()
        if win:
            wins += 1
        total_steps += steps
    print(f"Win rate: {wins/n:.2%}, Average steps: {total_steps/n:.2f}")

if __name__ == "__main__":
    main()
