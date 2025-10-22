import math
import matplotlib.pyplot as plt

def num_goal_states(s: int = 16, p: int = 3) -> int:
    g = p
    total = 0
    # for all number of k pieces [1, ..., p],
    for k in range(1, p + 1):
        # calculate number of states with k pieces from one of the players in their goal area
        total += (math.comb(g, k) * math.comb(s - g, p - k) * math.comb(s - g - (p - k), k)) - (math.comb(g, k) * math.comb(g, p-k))
    # account for the other player
    total *= 2
    return total

def num_total_states(s: int = 16, p: int = 3) -> int:
    return 2 * math.comb(s, p) * math.comb(s - p, p)

def plot_line_graph_with_x(x_values: list, y_values: list[float], 
                           title: str = "Line Graph", 
                           xlabel: str = "X", ylabel: str = "Y",
                           labels: list[str] = None):
    """Create a line graph with custom x-axis values and optional point labels."""
    plt.figure(figsize=(16, 9))
    plt.plot(x_values, y_values, marker='o', linewidth=2, markersize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    # Add labels to each point
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        label_text = labels[i] if labels and i < len(labels) else f'({x}, {y})'
        plt.annotate(label_text, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.tight_layout()
    plt.show()

sizes: list[int] = [4, 9, 9, 9, 16, 16, 16, 16, 16, 16, 25, 25, 25, 25, 36, 36, 36, 36, 36, 49, 49, 49, 49, 49, 49, 49, 81, 81, 81, 81, 81, 81, 81, 81, 121]
pieces: list[int] = [1, 1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 3, 6, 10, 1, 3, 4, 6, 10, 1, 2, 3, 4, 5, 6, 10, 1, 3, 4, 5, 6, 7, 8, 10, 10]

#sizes: list[int] = [4, 9, 9, 9, 16, 16, 16, 16, 16, 16, 25, 25, 25, 36, 36, 36, 36, 49, 49, 49, 49, 49, 49, 81, 81, 81, 81, 81]
#pieces: list[int] = [1, 1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 3, 6, 1, 3, 4, 6, 1, 2, 3, 4, 5, 6, 1, 3, 4, 5, 6]

total_states_list: list[int] = []
goal_states_list: list[int] = []
fractions: list[float] = []
labels: list[str] = []
for s, p in zip(sizes, pieces):
    num_states = num_total_states(s, p)
    num_goal = num_goal_states(s, p)
    print(f"Size: {s}, Pieces: {p}")
    print(f"Total States: {num_states}")
    print(f"Goal States: {num_goal}")
    fraction = num_goal / num_states
    print(f"Fraction: {fraction:.6%}")
    total_states_list.append(num_states)
    goal_states_list.append(num_goal)
    fractions.append(fraction)
    #labels.append(f"{s}-{p}:\n{num_states:.1e}\n{num_goal:.1e}\n{fraction:.2%}")
    labels.append(f"{s}-{p}")
    print("------")

sorted_data = sorted(zip(total_states_list, goal_states_list), key=lambda pair: pair[0])
sorted_x, sorted_y = zip(*sorted_data)

plot_line_graph_with_x(sorted_x, sorted_y,
                       title="Number of Goal States vs Total States",
                       xlabel="Total States",
                       ylabel="Goal States",
                       labels=labels)

sorted_data = sorted(zip(total_states_list, fractions), key=lambda pair: pair[0])
sorted_x, sorted_y = zip(*sorted_data)

plot_line_graph_with_x(sorted_x, sorted_y,
                       title="Goal States/Total States vs Total States",
                       xlabel="Total States",
                       ylabel="Goal States/Total States",
                       labels=labels)
