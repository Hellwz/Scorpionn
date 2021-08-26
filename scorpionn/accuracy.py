import numpy as np

def accuracy_classification(pred, truth):
    total_num = len(pred)
    correct_num = int(np.sum(pred == truth))
    accuracy = correct_num * 1.0 / total_num
    info = {"correct_num": correct_num, "total_num": total_num}
    return accuracy, info