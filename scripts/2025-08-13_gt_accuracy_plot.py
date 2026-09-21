import os
import sys
# moved here from the repo root; keep repo-root imports working
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
from a0.utils.plotting import plot_given

def load_losses(losses_path: str) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    '''
    Loads losses from the given path.
    Returns a tuple of lists: (losses, value_losses, policy_losses, value_accuracies, policy_accuracies).
    If the file does not exist, it will log an error and exit.
    '''
    with open(losses_path, 'rb') as f:
        losses_data = pickle.load(f)
    return (losses_data['losses'], losses_data['value_losses'], 
            losses_data['policy_losses'], losses_data['value_accuracies'], losses_data['policy_accuracies'])

# directory holding the *_gtv_eval.pkl files -- an output/eval/ from a run
if len(sys.argv) < 2:
    sys.exit("usage: python scripts/2025-08-13_gt_accuracy_plot.py <eval_dir>")
base = sys.argv[1]
_, _, _, training_acc, _ = load_losses(f'{base}/training_gtv_eval.pkl')
_, _, _, neighbor_1_acc, _ = load_losses(f'{base}/neighbor_1_gtv_eval.pkl')
_, _, _, neighbor_2_acc, _ = load_losses(f'{base}/neighbor_2_gtv_eval.pkl')
_, _, _, random_acc, _ = load_losses(f'{base}/random_gtv_eval.pkl')

with open(f"{base}/gamedata_acc.pkl", "rb") as f:
    itt_acc = pickle.load(f)

x0 = list(range(len(training_acc)))
x1 = list(range(len(itt_acc)))

plot_given("Model Performance on Ground Truth + Training Data Accuracy",
           [
               ("Seen", x0, training_acc),
               ("Neighbor 1", x0, neighbor_1_acc),
               ("Neighbor 2", x0, neighbor_2_acc),
               ("Random", x0, random_acc),
               ("Training Data Accuracy", x1, itt_acc)
           ],
           "Iterations", "Accuracy", "new_plot")

