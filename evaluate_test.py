import numpy as np


# def eval_pred_prob(filepath: str) -> None:
#     all_preds = np.load(filepath)
#
#     conf_mat = np.zeros((6, 6))
#     for i in range(len(all_preds)):
#         for pred, gold in all_preds[i]:
#             if pred >= 0:
#                 conf_mat[pred][gold] += 1
#
#     accuracy = conf_mat.trace() / np.sum(conf_mat)
#     precision = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
#     recall = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
#     f1 = (2 * precision * recall) / (precision + recall)
#     print("accuracy:", accuracy)
#     print("precision:", precision)
#     print("recall:", recall)
#     print("f1:", f1)
#
#
# if __name__ == "__main__":
#     eval_pred_prob('./prediction_probabilities.npy')
#
