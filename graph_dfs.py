import re
import matplotlib.pyplot as plt
from datetime import datetime

# Read the file
with open('new_text_file.txt', 'r') as file:
    lines = file.readlines()

# Initialize lists
times = []
steps = []
stack = []
visited = []
wins = []
losses = []
draws = []

# Regular expression patterns
timestamp_pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
data_pattern1 = r"Steps: (\d+), Stack: (\d+) Wins: (\d+), Losses: (\d+), Draws: (\d+)"
data_pattern2 = r"Steps: (\d+), Stack: (\d+), Visited: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)"

# Parse each line
for line in lines:
    time_match = re.search(timestamp_pattern, line)
    data_match1 = re.search(data_pattern1, line)
    data_match2 = re.search(data_pattern2, line)
    
    if time_match:
        time_str = time_match.group(1)
        time_obj = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        times.append(time_obj)
    if data_match1:
        steps.append(int(data_match1.group(1)))
        stack.append(int(data_match1.group(2)))
        visited.append(0)
        wins.append(int(data_match1.group(3)))
        losses.append(int(data_match1.group(4)))
        draws.append(int(data_match1.group(5)))
    if data_match2:
        steps.append(int(data_match2.group(1)))
        stack.append(int(data_match2.group(2)))
        visited.append(int(data_match2.group(3)))
        wins.append(int(data_match2.group(4)))
        losses.append(int(data_match2.group(5)))
        draws.append(int(data_match2.group(6)))
    

# Plotting
plt.figure(figsize=(12, 8))

# Stack over Time
plt.subplot(2, 1, 1)
plt.plot(times, stack, marker='o', label='Stack')
plt.plot(times, visited, marker='o', label='Visited')
plt.title('Sets vs Time')
plt.xlabel('Time')
plt.ylabel('Stack')
plt.grid(True)
plt.legend()

# Game Outcomes over Time
plt.subplot(2, 1, 2)
plt.plot(times, wins, marker='o', label='Wins')
plt.plot(times, losses, marker='o', label='Losses')
plt.plot(times, draws, marker='o', label='Draws')
plt.title('Game Outcomes vs Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
