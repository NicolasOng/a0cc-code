import math

def num_goal_states(s: int = 16, p: int = 3) -> int:
    g = p
    total = 0
    # for all number of k pieces [1, ..., p],
    for k in range(1, p + 1):
        # calculate number of states with k pieces from one of the players in their goal area
        total += math.comb(g, k) * math.comb(s - g, p - k) * math.comb(s - g - (p - k), k)
    # account for the other player
    total *= 2
    return total

def num_total_states(s: int = 16, p: int = 3) -> int:
    return 2 * math.comb(s, p) * math.comb(s - p, p)

sizes: list[int] = [16, 25, 49, 81]
pieces: list[int] = [3, 6, 6, 6]

for s, p in zip(sizes, pieces):
    print(f"Size: {s}, Pieces: {p}")
    print(f"Total States: {num_total_states(s, p)}")
    print(f"Goal States: {num_goal_states(s, p)}")
    print(f"Fraction: {num_goal_states(s, p) / num_total_states(s, p):.6%}")
    print("------")

