from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt


def plot_training_history(hist):
    # plot the change in accuracy and loss during training
    import matplotlib.pyplot as plt

    # plot training epochs
    fig_acc, axes_acc = plt.subplots(figsize=(10, 6))
    fig_loss, axes_loss = plt.subplots(figsize=(10, 6))

    axes_acc.plot(hist.history['acc'], label='Training Accuracy')
    axes_loss.plot(hist.history['loss'], label='Training Loss')

    axes_acc.set_title(" Accuracy History", fontsize=18)
    axes_loss.set_title(" Loss History", fontsize=18)

    axes_acc.set_xlabel("Epochs", fontsize=18)
    axes_loss.set_xlabel("Epochs", fontsize=18)

    axes_acc.set_ylabel("Accuracy (%) ", fontsize=18)
    axes_loss.set_ylabel("Loss", fontsize=18)

    axes_acc.legend(fontsize=14);
    axes_loss.legend(fontsize=14);

    plt.show()

def plot_acc_loss(hist):
     # plot the change in accuracy and loss during training
     import matplotlib.pyplot as plt

     # plot training epochs
     fig_acc, axes_acc = plt.subplots(figsize=(10, 6))
     fig_loss, axes_loss = plt.subplots(figsize=(10, 6))

     axes_acc.plot(hist.history['acc'], label='Training Accuracy')
     axes_loss.plot(hist.history['loss'], label='Training Loss')

     axes_acc.plot(hist.history['val_acc'], label='Validation Accuracy')
     axes_loss.plot(hist.history['val_loss'], label='Validatio Loss')

     axes_acc.set_title(" Accuracy History", fontsize=18)
     axes_loss.set_title(" Loss History", fontsize=18)

     axes_acc.set_xlabel("Epochs", fontsize=18)
     axes_loss.set_xlabel("Epochs", fontsize=18)

     axes_acc.set_ylabel("Accuracy (%) ", fontsize=18)
     axes_loss.set_ylabel("Loss", fontsize=18)

     axes_acc.legend(fontsize=14);
     axes_loss.legend(fontsize=14);

     plt.show()


def plot_confusion_matrix(X, y, model, threshold=0.5):
     """
     Plot confusion matrix for a binary classification model.

     Parameters:
     - X: Input data.
     - y: True labels.
     - model: Trained binary classification model.
     - threshold: Threshold for converting continuous predictions to binary (default is 0.5).
     """
     # Make predictions on the input data
     y_pred = model.predict(X)

     # Convert continuous predictions to binary using the specified threshold
     y_pred_classes = (y_pred >= threshold).astype(int)

     # Calculate the confusion matrix
     conf_matrix = confusion_matrix(y, y_pred_classes)

     # Display the confusion matrix using seaborn
     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                 xticklabels=['Class 0', 'Class 1'],
                 yticklabels=['Class 0', 'Class 1'])

     plt.title('Confusion Matrix')
     plt.xlabel('Predicted')
     plt.ylabel('Actual')
     plt.show()


def evaluate_classification_model(model, X_test, y_test, threshold=0.5, normalize_param=None):
    """
    Evaluate the model on the test set.

    Parameters:
    - model: the model trained.
    - X_test: Feature of the test set.
    - y_test: Labels of the test set.
    - threshold: Threshold for converting continuous predictions to binary (default is 0.5).
    - normalize_param: {‘true’, ‘pred’, ‘all’}, default=None
    """

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert predictions to binary classes
    y_pred_classes = (y_pred >= threshold).astype(int)

    # Print the classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred_classes))

    # Print the confusion matrix
    print('\nConfusion Matrix:')

    if normalize_param is None:
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
    else:
        conf_matrix = confusion_matrix(y_test, y_pred_classes, normalize=normalize_param)
        sns.heatmap(conf_matrix, annot=True, cmap='Blues',
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """
    Plot the ROC curve for a classification model.

    Parameters:
    - model: Trained model
    - X_test: Feature of the test set
    - y_test: Labels of the test set

    Returns:
    - None (displays the plot)
    """
    # Calculate predicted probabilities
    y_prob = model.predict(X_test) # returns the probability of an instance to belong to each class

    # For binary classification, consider only the positive class
    if y_prob.shape[1] > 1:
        y_prob = y_prob[:, 1] # associate only to the positive class

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)



    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()




def evaluate_classification_models(models, X_test, y_test, threshold=0.5, normalize_param=None):
    """
    Evaluate multiple classification models on the test set and plot their confusion matrices with two per row.

    Parameters:
    - models: List of trained classification models.
    - X_test: Feature of the test set.
    - y_test: Labels of the test set.
    - threshold: Threshold for converting continuous predictions to binary (default is 0.5).
    - normalize_param: {‘true’, ‘pred’, ‘all’}, default=None
    """

    num_models = len(models)
    rows = (num_models + 1) // 2  # Compute the number of rows needed, ensuring there is space for the last model

    plt.figure(figsize=(12, 6 * rows))

    for i, model in enumerate(models, 1):
        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Convert predictions to binary classes
        y_pred_classes = (y_pred >= threshold).astype(int)

        # Print the classification report
        print(f'\nClassification Report - Model {i}:')

        print(classification_report(y_test, y_pred_classes))

        # Plot the confusion matrix
        plt.subplot(rows, 2, i)

        if normalize_param is None:
            conf_matrix = confusion_matrix(y_test, y_pred_classes)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Class 0', 'Class 1'],
                        yticklabels=['Class 0', 'Class 1'])

        else:
            conf_matrix = confusion_matrix(y_test, y_pred_classes, normalize=normalize_param)
            sns.heatmap(conf_matrix, annot=True, cmap='Blues',
                        xticklabels=['Class 0', 'Class 1'],
                        yticklabels=['Class 0', 'Class 1'])

        plt.title(f'Confusion Matrix - Model {i}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()


def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for multiple classification models on the same graph.

    Parameters:
    - models: List of trained classification models.
    - X_test: Feature of the test set.
    - y_test: Labels of the test set.
    """

    plt.figure(figsize=(10, 8))

    for i, model in enumerate(models):
        # Calculate predicted probabilities
        y_prob = model.predict(X_test)

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve for each model with different colors
        plt.plot(fpr, tpr, lw=2, label=f'Model {i+1} (AUC = {roc_auc:.2f})')

    # Plot the random classifier line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.show()
