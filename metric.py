import torch

def accuracy(target, pred):
    """
    Compute the accuracy.
    Args:
    - target (torch.Tensor): True class labels of shape (N,)
    - pred (torch.Tensor): Predicted class labels of shape (N,)

    Returns:
    - float: Accuracy value
    """ 
    correct = (target == pred).sum().item()
    total = target.size(0)
    return correct / total

def precision(target, pred, average='macro'):
    """
    Compute the precision.
    Args:
    - target (torch.Tensor): True class labels of shape (N,)
    - pred (torch.Tensor): Predicted class labels of shape (N,)
    - average (str): Type of averaging performed ('macro', 'micro', 'weighted')

    Returns:
    - float: Precision value
    """
    num_classes = len(torch.unique(target))
    precision_per_class = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        true_positives = ((pred == cls) & (target == cls)).sum().item()
        predicted_positives = (pred == cls).sum().item()
        if predicted_positives == 0:
            precision_per_class[cls] = 0
        else:
            precision_per_class[cls] = true_positives / predicted_positives
    
    if average == 'macro':
        return precision_per_class.mean().item()
    elif average == 'micro':
        true_positives = (pred == target).sum().item()
        predicted_positives = pred.size(0)
        return true_positives / predicted_positives
    elif average == 'weighted':
        class_counts = torch.bincount(target)
        return (precision_per_class * class_counts / class_counts.sum()).sum().item()
    else:
        raise ValueError("Average must be one of ['macro', 'micro', 'weighted']")

def recall(target, pred, average='macro'):
    """
    Compute the recall.
    Args:
    - target (torch.Tensor): True class labels of shape (N,)
    - pred (torch.Tensor): Predicted class labels of shape (N,)
    - average (str): Type of averaging performed ('macro', 'micro', 'weighted')

    Returns:
    - float: Recall value
    """
    num_classes = len(torch.unique(target))
    recall_per_class = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        true_positives = ((pred == cls) & (target == cls)).sum().item()
        actual_positives = (target == cls).sum().item()
        if actual_positives == 0:
            recall_per_class[cls] = 0
        else:
            recall_per_class[cls] = true_positives / actual_positives
    
    if average == 'macro':
        return recall_per_class.mean().item()
    elif average == 'micro':
        true_positives = (pred == target).sum().item()
        actual_positives = target.size(0)
        return true_positives / actual_positives
    elif average == 'weighted':
        class_counts = torch.bincount(target)
        return (recall_per_class * class_counts / class_counts.sum()).sum().item()
    else:
        raise ValueError("Average must be one of ['macro', 'micro', 'weighted']")

def f1_score(target, pred, average='macro'):
    """
    Compute the F1 score.
    Args:
    - target (torch.Tensor): True class labels of shape (N,)
    - pred (torch.Tensor): Predicted class labels of shape (N,)
    - average (str): Type of averaging performed ('macro', 'micro', 'weighted')

    Returns:
    - float: F1 score value
    """
    prec = precision(target, pred, average)
    rec = recall(target, pred, average)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)
