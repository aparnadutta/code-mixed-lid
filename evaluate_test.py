import numpy as np
# from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report

all_preds = np.load('./prediction_probabilities.npy')

preds = [pair[0] for pair in all_preds if not np.array_equal(pair, np.array([-1, -1]))]
true_labels = [pair[1] for pair in all_preds if not np.array_equal(pair, np.array([-1, -1]))]

print(len(all_preds))
report = classification_report(true_labels, preds)
print(report)

# num_correct, total = 0, 0

# for i in range(len(all_preds)):
#     pad_row = np.array([-1, -1])
#     for word in all_preds[i]:
#         if not np.array_equal(word, pad_row):
#             total += 1
#             if word[0] == word[1]:
#                 num_correct += 1
#
# accuracy = num_correct / total
# print("accuracy:", accuracy)

