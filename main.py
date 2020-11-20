import argparse
import numpy as np

from pop_sim import pop_sim
from utilities import calculate_l, calculate_s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="csv with input reproduction data", required=True)
    parser.add_argument("--angler_rule", help="0: no catch, 1: exact age, 2: lower limit", required=True)
    parser.add_argument("--angler_catch_rate", help="number of fish caught by anglers", required=True)
    args = parser.parse_args()

    file_data = np.genfromtxt(args.input_file, delimiter=',', names=True, case_sensitive=True)

    SEED_TRIALS = 100
    TRIALS = 1000
    POP_START = 1000

    w = file_data.shape[0]
    s = calculate_s(file_data['l'])
    n_start = np.full((w, 1), POP_START)

    n_pop_dict = {}
    total_pop_dict = {}
    for i in range(w):
        no_rule_sim = pop_sim(file_data['f'], s, np.copy(n_start), int(args.angler_catch_rate), 0)
        no_rule_sim.step_forward_by(SEED_TRIALS)

        trial_sim = pop_sim(file_data['f'], s, np.copy(no_rule_sim.n), int(args.angler_catch_rate), i, int(args.angler_rule))
        trial_sim.step_forward_by(TRIALS)

        n_pop_dict[i] = trial_sim.n
        total_pop_dict[i] = np.sum(trial_sim.n)

    print(n_pop_dict)
    print(total_pop_dict)


if __name__ == "__main__":
    main()