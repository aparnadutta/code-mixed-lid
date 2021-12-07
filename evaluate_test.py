import numpy as np

all_preds = np.load('./prediction_probabilities.npy')

conf_mat = np.zeros((6, 6))

for i in range(len(all_preds)):
    pad_row = np.array([-1, -1])
    for word in all_preds[i]:
        if not np.array_equal(word, pad_row):
            conf_mat[word[0][word[1]]] += 1

print("confusion matrix:", conf_mat)

