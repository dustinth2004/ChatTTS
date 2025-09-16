
import torch
import torch.nn.functional as F

    
class CustomRepetitionPenaltyLogitsProcessorRepeat():
    """
    A custom logit processor that applies a repetition penalty to the logits.

    This processor penalizes logits of tokens that have appeared in the past,
    making the model less likely to repeat itself.

    Args:
        penalty (float): The penalty factor. A value greater than 1 encourages
            the model to generate new tokens, while a value less than 1
            encourages it to repeat tokens.
        max_input_ids (int): The maximum input ID to consider for the penalty.
        past_window (int): The size of the past window to consider for repetition.
    """

    def __init__(self, penalty: float, max_input_ids, past_window):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Applies the repetition penalty to the logits.

        Args:
            input_ids (torch.LongTensor): The input IDs of the sequence.
            scores (torch.FloatTensor): The logits for the next token.

        Returns:
            torch.FloatTensor: The modified logits.
        """
        
        input_ids = input_ids[:, -self.past_window:]
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        freq[self.max_input_ids:] = 0
        alpha = self.penalty**freq
        scores = torch.where(scores < 0, scores*alpha, scores/alpha)

        return scores
    
class CustomRepetitionPenaltyLogitsProcessor():
    """
    A custom logit processor that applies a repetition penalty to the logits.

    This processor penalizes logits of tokens that have appeared in the past,
    making the model less likely to repeat itself.

    Args:
        penalty (float): The penalty factor. A value greater than 1 encourages
            the model to generate new tokens, while a value less than 1
            encourages it to repeat tokens.
        max_input_ids (int): The maximum input ID to consider for the penalty.
        past_window (int): The size of the past window to consider for repetition.
    """

    def __init__(self, penalty: float, max_input_ids, past_window):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Applies the repetition penalty to the logits.

        Args:
            input_ids (torch.LongTensor): The input IDs of the sequence.
            scores (torch.FloatTensor): The logits for the next token.

        Returns:
            torch.FloatTensor: The modified logits.
        """
        
        input_ids = input_ids[:, -self.past_window:]
        score = torch.gather(scores, 1, input_ids)
        _score = score.detach().clone()
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        score[input_ids>=self.max_input_ids] = _score[input_ids>=self.max_input_ids]
        scores.scatter_(1, input_ids, score)
        
        return scores