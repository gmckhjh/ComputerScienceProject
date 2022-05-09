import numpy as np

"""
Gets average values from an ensemble learning algorithm. 
"""


# Average values of predictions if arrays are equal sizes
def average_outputs(np_output1, np_output2):
    # if __validate_data(np_output1, np_output2):
    if np_output1.shape == np_output2.shape:
        try:
            averages = np.mean([np_output1[:, 1], np_output2[:, 1]], axis=0)
        except IndexError as e:
            print(e)
            averages = None
        return averages
    else:
        print('Arrays must be same shape')
        return None

