import numpy as np


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])


def krum(updates, num_malicious=1):
    n = len(updates)

    if n <= 2 + num_malicious:
        raise ValueError("Krum requires n > 2 + num_malicious")

    flat_updates = [flatten_weights(update) for update in updates]

    scores = []

    for i in range(n):
        distances = []

        for j in range(n):
            if i != j:
                dist = np.linalg.norm(flat_updates[i] - flat_updates[j])
                distances.append(dist)

        distances.sort()

        score = sum(distances[: n - num_malicious - 2])
        scores.append(score)

    selected_index = int(np.argmin(scores))

    return updates[selected_index], selected_index