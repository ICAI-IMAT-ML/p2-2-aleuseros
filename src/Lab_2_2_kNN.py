import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np  




def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """

    #Apply the formula of the minkowski distance, return its value
    return np.sum(np.abs(a - b) ** p) ** (1 / p)


# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](httrue_positives://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](httrue_positives://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        #We first check that we have valid arguments, in case we dont, we simply send out a message and force the return
        if len(X_train) != len(y_train):
            print ("Training sets do not have the same number of rows")
            return
        elif k<= 0:
            print ("k must be a positive variable")
            return
        elif p<= 0:
            print ("p must be a positive integer")
            return
        
        #If everything goes right we execute the function normally
        
        self.k = k
        self.p = p
        self.x_train = X_train
        self.y_train = y_train
    
    def predict(self, X: np.ndarray)->np.ndarray:
        #We first create the list that stores the predictions
        predictions = []

        #We now iterate through the values of X
        for x in X:
            #Compute the distances
            distances = self.compute_distances(x)
            #Obtain the indexes
            neighbour_indexes = self.get_k_nearest_neighbors(distances)
            #Obtain the labels
            neighbour_labels = self.y_train[neighbour_indexes]
            #Predict labels and add it to the list
            predicted_labels = self.most_common_label(neighbour_labels)
            predictions.append(predicted_labels)

        #Finally, we change it into an array and return it
        return np.array(predictions)
    

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        #We first create a list to store the probabilities
        probabilities = []

        #Now iterate the array and calculate the probabilities
        for x in X:
            distances = self.compute_distances(x)
            neighbour_indexes = self.get_k_nearest_neighbors(distances)
            neighbour_labels = self.y_train[neighbour_indexes]
            #Calculate the number of unique labels
            unique_labels, frequency = np.unique(neighbour_labels, return_counts = True)

            #Convert the datatypes if they are not integer, to 1 and 0, binary rules!
            if unique_labels.dtype == 'object':
                unique_labels = [1 if label=='YES' else 0 for label in unique_labels]
            
            #Create a probabilities vector and initialize it to 0
            probabilidades = np.zeros(len(np.unique(self.y_train)))
            
            #Para cada etiqueta, devolvemos la probabilidad (k/frecuencia)
            probabilidades[unique_labels] = frequency/self.k

            #finally, we add it to the probabilities list
            probabilities.append(probabilidades)

        #Out of the loop, change it to an array and return it
        return np.array(probabilities)
    

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        #Use minjowski distance directly on the array given, through a loop we obtain everything
        return np.array([minkowski_distance(point, x, self.p) for x in self.x_train])


    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        #Use the method that is recommended to sort them
        return np.argsort(distances)[:self.k]

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        #we first obtain the most common labels in the data set
        labels, frequency = np.unique(knn_labels, return_counts=True) #Return counts True so that we get the count

        #We now return the most common label, use the argmax method
        return labels[np.argmax(frequency)]

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)

    # Determine true_positive, FP, false_negative, true_negative
    true_positive = (y == unique_labels[1]) & (preds == unique_labels[1])
    false_positive = (y == unique_labels[0]) & (preds == unique_labels[1])
    false_negative = (y == unique_labels[1]) & (preds == unique_labels[0])
    true_negative = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[true_positive, 0], X[true_positive, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[false_positive, 0], X[false_positive, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[false_negative, 0], X[false_negative, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[true_negative, 0], X[true_negative, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [true_negative, FP, false_negative, true_positive]
        - Accuracy: (true_positive + true_negative) / (true_positive + true_negative + FP + false_negative)
        - Precision: true_positive / (true_positive + FP)
        - Recall (Sensitivity): true_positive / (true_positive + false_negative)
        - Specificity: true_negative / (true_negative + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    true_positive = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
    true_negative = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))
    false_positive = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
    false_negative = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))

    #Use the normal formulas for the classification metrics

    # Accuracy: from 0 to 1, number of right predictions divided by the total
    accuracy = np.sum(y_true_mapped==y_pred_mapped)/ len (y_true_mapped)
    

    # Precision
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

    # Recall (Sensitivity)
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    # Specificity
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Confusion Matrix": [true_negative, false_positive, false_negative, true_positive],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }



def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    #Map the labels to binary values
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    #Divide the space between 1 and 0, using linspace, n_bins +1, then calculate the bin centers
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    #Create the list to store the true_proportions
    true_proportions = []
    #Iter the bins and check probabilities
    for i in range(n_bins):
        bin_mask = (y_probs >= bin_edges[i]) & (y_probs < bin_edges[i+1])
        if np.any(bin_mask):
            prop = np.mean(y_true_mapped[bin_mask])
        else:
            prop = np.nan  # In case we do not have data, include a nan value
        true_proportions.append(prop)

    #Plot it
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, true_proportions, marker='o', linestyle='-', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Predicted probabillity')
    plt.ylabel('Proportion of positives')
    plt.title('Calibration curve')
    plt.legend()
    plt.show()


    #Return de dictionary 
    return {"bin_centers": bin_centers, "true_proportions": true_proportions}



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    #Map the true predictions
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    
    #Calculate the probabilities
    positive_probs = y_probs[y_true_mapped == 1]
    negative_probs = y_probs[y_true_mapped == 0]
    
    #Create the figure
    plt.figure(figsize=(12, 5))
    
    #first subplot, positive histogram
    plt.subplot(1, 2, 1)
    plt.hist(positive_probs, bins=n_bins, range=(0, 1), color='blue', alpha=0.7)
    plt.title('Probabilities histogram - Positive')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(negative_probs, bins=n_bins, range=(0, 1), color='red', alpha=0.7)
    plt.title('Probabilities histogram - Negative')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }



def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (true_positiveR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "true_positiver": Array of True Positive Rates for each threshold.

    """
    # Mapear etiquetas a 0 y 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    
    # Obtener todos los umbrales posibles ordenados
    thresholds = np.sort(np.unique(y_probs))
    fpr = []
    tpr = []
    
    # Calcular true_positiveR y FPR para cada umbral
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        true_positive = np.sum((y_true_mapped == 1) & (y_pred == 1))
        true_negative = np.sum((y_true_mapped == 0) & (y_pred == 0))
        false_positive = np.sum((y_true_mapped == 0) & (y_pred == 1))
        false_negative = np.sum((y_true_mapped == 1) & (y_pred == 0))
        
        true_positiver_val = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        fpr_val = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0
        
        tpr.append(true_positiver_val)
        fpr.append(fpr_val)
    
    # Incluir los extremos 0 y 1 para una curva completa
    fpr = [0] + fpr + [1]
    tpr = [0] + tpr + [1]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='o', label='Curva ROC')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Línea de azar')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (true_positiveR)')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()
    
    
    return {"fpr": np.array(fpr), "true_positiver": np.array(tpr)}
