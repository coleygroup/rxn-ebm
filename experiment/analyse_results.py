"""This module contains functions for evaluating the performance of a trained model by analysing
the distribution of its scores for a given test/train dataset. 
Examples include calculating Expected Calibration Error (ECE)"""

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

Tensor = torch.Tensor

def make_probs_from_scores(raw_scores: Tensor) -> Tensor:
    '''Converts raw scores outputted by model into probabilities normalised
    across each set of 1 positive & K negative reactions 
    
    Parameters
    ----------
    raw_scores : Tensor of shape (N positive rxns in dataset, 1 + K negative rxns per positive rxn)
        scores assigned by model to every reaction input
        
    Returns
    -------
    raw_probs : Tensor of identical shape as raw_scores
        probabilities obtained by softmax across each row (set of 1 positive & K negative reactions)    
    '''
    softmax = torch.nn.Softmax(dim=1)
    raw_probs = softmax(raw_scores)
    return raw_probs

def make_probs_correct_class(probs: Tensor) -> List[float]:
    '''Filter probability of correct class (0-th index, aka positive reaction) 
    from each row in probs Tensor 
    
    Parameters
    ----------
    probs : Tensor of shape (N positive rxns in dataset, 1 + K negative rxns per positive rxn)
        probabilities assigned by model to every reaction input
    
    Returns
    -------
    probs_correct_class : List of length (# positive rxns in dataset)
        probabilities assigned by model to the positive reaction input 
        among each set of 1 positive & K negative reactions
        
    Also see: make_probs_from_scores
    '''
    probs_correct_class = probs[:, 0].tolist()
    return probs_correct_class

def make_probs_tuples(probs: Tensor) -> List[Tuple[float, int]]:
    '''Makes tuples (max probability, corresponding class) from tensor of
    probabilities assigned by model to each reaction input (row) 
    
    Parameters
    ----------
    probs : Tensor of shape (N positive rxns in dataset, 1 + K negative rxns per positive rxn)
        probabilities assigned by model to each reaction input
    
    Returns
    -------
    probs_tuples : List[Tuple[float, int]]
        (max probability, corresponding class) of each row in probs Tensor
    
    Also see: make_probs_from_scores, bin_accuracies_from_probs_tuples
    '''
    probs_max, probs_argmax = torch.max(probs, dim=1)
    probs_max, probs_argmax = probs_max.numpy(), probs_argmax.numpy()
    probs_tuples = tuple(zip(probs_max, probs_argmax))
    return list(probs_tuples)

def bin_accuracies_from_probs_tuples(probs_tuples: Iterable[Tuple[float, int]], 
                                    lower_bounds: Iterable[float] = np.linspace(0., 0.95, 20), 
                                    upper_bounds: Iterable[float] = np.linspace(0.05, 1., 20),
                                    get_confidence_data: bool = True) ->Tuple[List[float], 
                                                                              Optional[List[float]], Optional[List[float]]]:
    '''Puts (max probability, corresponding class) into accuracy bins for calibration plotting
    
    Parameters
    ----------
    probs_tuples : Iterable[Tuple[float, int]]
        iterable of tuples (max probability, corresponding class) returned by make_probs_tuple
    lower_bounds : Iterable[float] (Default = np.linspace(0., 0.95, 20))
        lower bounds on confidence bins 
    upper_bounds : Iterable[float] (Default = np.linspace(0.05, 1., 20))
        upper bounds on confidence bins
    get_confidence_data: bool = True
        whether to return probabilities in each bin & # data points in that bin
    
    Returns
    -------
    accuracies : List[float]
        accuracy of each bin, has same length as lower_bounds
    confidences : Optional[List[float]]
        confidences aka probabilities put into bins, only returned when get_confidence is True
    bin_counts : Optional[List[float]]
        # data points in each bin
    
    Also see: make_probs_tuples
    '''
    # puts (probability, predicted index) tuples into bins defined by lower & upper bounds 
    binned_conf_predclass = [list(filter(lambda x: True if (x[0] > lower and x[0] <= upper) else False, probs_tuples)) 
                             for lower, upper in zip(lower_bounds, upper_bounds)]
    
    # binned_classes tracks whether model predicted correctly for each reaction set (1 pos + K neg rxns) within each bin
    # value of 1 if yes, 0 if otherwise
    binned_classes = []
    for binned in binned_conf_predclass: 
        # pair[1] is the predicted class by the model for that reaction set (argmax)
        # model was correct iff pair[1] == 0 since pos rxn is always and only the 0-th index
        binned_classes.append([np.where(pair[1] == 0)[0].shape[0] for pair in binned])

    # sums up the 1's from binned_classes & divide by total count to calculate accuracies within each bin
    # for plotting purposes, assigns default accuracy of 0 if no members exist in that bin
    accuracies = [sum(binned) / len(binned) if len(binned) > 0 else 0 for binned in binned_classes]
    if get_confidence_data:
        # remove predicted class (argmax) from probs_tuples
        binned_conf = []
        for binned in binned_conf_predclass: 
            binned_conf.append([pair[0] for pair in binned])
        
        bin_counts = [len(confs) for confs in binned_conf]
        confidences = [sum(confs) / len(confs) if len(confs) > 0 else 0 for confs in binned_conf]
        return accuracies, confidences, bin_counts
    else:
        return accuracies

def calc_ECE(accs: List[float], confs: List[float], 
             bin_counts: List[float], total_count: int, 
             decimals: int = 3) -> float:
    '''Calculates Expected Calibration Error (ECE) to evaluate how overconfident our trained model is
    Reference: Nixon, Jeremy, et al. "Measuring Calibration in Deep Learning." CVPR Workshops. 2019. 
    https://arxiv.org/abs/1904.01685
    
    Parameters
    ----------
    accs : List[float]
        accuracy of each bin
    confs : List[float]
        confidence (mean probability) of each bin
    bin_counts : List[float]
        # data points in each bin
    total_count : int
        # data points in total
    decimals : int = 3
        # of decimal places to output
        
    Returns
    -------
    ECE : float
        Expected Calibration Error as %
        
    Also see: bin_accuracies_from_probs_tuples
    '''
    ECE = 0
    for acc, conf, bin_count in zip(accs, confs, bin_counts):
        weight = bin_count / total_count
        ECE += weight * abs(acc - conf)
    
    return round(ECE * 100, decimals)