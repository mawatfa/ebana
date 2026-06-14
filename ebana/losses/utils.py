import numpy as np

def calculate_prediction(output_node_voltages, fold):
    """
    Calculate the prediction as the difference between the two columns
    of the output_node_voltages array.
    """
    if fold:
        output_node_voltages = output_node_voltages.reshape(-1, 2)
        return output_node_voltages[:, 0] - output_node_voltages[:, 1]
    else:
        return output_node_voltages

def default_loss_update_rule(beta, loss):
    """
    Default rule to update loss values: simply scales the loss by a factor beta.
    """
    return beta * loss

def return_current(current):
    return np.column_stack([-current, current])

def verify_result_using_probability(target, prediction):
    # Check if inputs are 1D; if so, wrap them in a list to handle them as rows
    if prediction.ndim == 1:
        prediction = np.array([prediction])
        target = np.array([target])

    # Find the index of the maximum value for each row
    prediction_max_indices = np.argmax(prediction, axis=1)
    target_max_indices = np.argmax(target, axis=1)

    # Check if the indices match for each row
    match = prediction_max_indices == target_max_indices

    # for idx in range(prediction.shape[0]):
    #     print(f"Prediction: {prediction[idx]}, Target: {target[idx]}, Match: {match.astype(int)[idx]}")

    return match.astype(int)
