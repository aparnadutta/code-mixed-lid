import numpy as np

all_preds = np.load('./prediction_probabilities.npy')

conf_mat = np.zeros((6, 6))
for i in range(len(all_preds)):
    pad_row = np.array([-1, -1])
    for word in all_preds[i]:
        if not np.array_equal(word, pad_row):
            conf_mat[word[0]][word[1]] += 1

accuracy = conf_mat.trace() / np.sum(conf_mat)
precision = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
recall = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
f1 = (2 * precision * recall) / (precision + recall)
print("accuracy:", accuracy)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

