# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/utils.py
# reference: https://github.com/lifeiteng/vall-e
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import functools
NEG_INF = -float("Inf")


@torch.jit.script
def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


@torch.jit.script
def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    #>>> lengths = torch.tensor([1, 3, 2, 5])
    #>>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


@torch.jit.script
def make_pad_mask_left(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    #>>> lengths = torch.tensor([1, 3, 2, 5])
    #>>> make_pad_mask(lengths)
    tensor(
        [
            [True,  True,  False],
            [True, False, False],
            [True,  True,  False],
            ...
        ]
    )
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max().item())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    # Use expand instead of repeat for better memory efficiency when possible
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    expaned_lengths = expaned_lengths - (max_len - lengths).unsqueeze(-1)
    return expaned_lengths < 0


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
        logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # Efficient top-k implementation
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Get top-k values and create a mask
        values, _ = torch.topk(logits, top_k)
        min_value = values[..., -1, None]
        indices_to_remove = logits < min_value

    # Efficient top-p implementation
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Create a mask for tokens to remove
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

        # Shift the indices to the right
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(1, sorted_indices, sorted_indices_to_remove)

    return logits.masked_fill(indices_to_remove, filter_value)


@torch.jit.script
def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    # if temperature != 1.0:
    logits /= max(temperature, 1e-5)
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.jit.script
def multinomial_sample_one_no_sync(probs_sort,):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


# Cache for frequently called function with the same parameters
@functools.lru_cache(maxsize=8)
def get_repetition_penalty_processor(repetition_penalty: float):
    """Returns a penalty processor based on repetition penalty value"""
    if repetition_penalty == 1.0:
        return lambda logits, tokens: logits  # No-op for penalty=1

    @torch.jit.script
    def apply_repetition_penalty(logits: torch.Tensor, previous_tokens: torch.Tensor) -> torch.Tensor:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        return logits.scatter_(dim=1, index=previous_tokens, src=score)

    return apply_repetition_penalty


@torch.jit.script
def logits_to_probs(
        logits,
        previous_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        repetition_penalty: float = 1.0,
):
    # if previous_tokens is not None:
    #     previous_tokens = previous_tokens.squeeze()
    # print(logits.shape,previous_tokens.shape)
    # pdb.set_trace()
    # Apply repetition penalty if needed
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(dim=1, index=previous_tokens, src=score)

    # Apply top-p (nucleus) sampling
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Create mask for values to remove
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False  # Keep at least one option
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, NEG_INF)

    # Apply temperature scaling
    scaled_logits = logits / max(temperature, 1e-5)

    # Apply top-k sampling
    if top_k is not None:
        # More efficient top-k implementation
        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
        pivot = v[:, -1].unsqueeze(-1)
        scaled_logits = torch.where(scaled_logits < pivot, torch.full_like(scaled_logits, NEG_INF), scaled_logits)

    # Convert to probabilities
    return F.softmax(scaled_logits, dim=-1)


def sample(
        logits,
        previous_tokens: Optional[torch.Tensor] = None,
        **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.jit.script
def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


@torch.jit.script
def get_batch_logps(logits_target: torch.FloatTensor, logits_reject: torch.FloatTensor,
                    labels_target: torch.LongTensor, labels_reject: torch.LongTensor,
                    average_log_prob: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # Compute log probabilities more efficiently
    log_probs_target = F.log_softmax(logits_target, dim=-1)
    log_probs_reject = F.log_softmax(logits_reject, dim=-1)

    # Gather the log probs for the target and reject labels
    per_token_logps_target = torch.gather(log_probs_target, dim=2, index=labels_target.unsqueeze(2)).squeeze(2)
    per_token_logps_reject = torch.gather(log_probs_reject, dim=2, index=labels_reject.unsqueeze(2)).squeeze(2)

    # Sum along sequence dimension
    return per_token_logps_target.sum(-1), per_token_logps_reject.sum(-1)


# Optimized implementation with vectorized operations when possible
def make_reject_y(y_o, y_lens):
    bs = len(y_lens)
    reject_y = []
    reject_y_lens = []

    # Pre-compute random indices for all batch items
    process_item_idx = torch.randint(0, 2, size=(bs,), device=y_o.device)

    for b in range(bs):
        # Generate random indices for manipulation
        range_idx = torch.randint(0, y_lens[b].item(), size=(2,), device=y_o.device)
        range_idx, _ = range_idx.sort()

        if process_item_idx[b] == 0:  # repeat_P
            pre = y_o[b, :range_idx[0]]
            shf = y_o[b, range_idx[1]:y_lens[b]]
            range_text = y_o[b, range_idx[0]:range_idx[1]]

            # Concatenate tensors efficiently
            if len(pre) > 0 and len(range_text) > 0 and len(shf) > 0:
                new_y = torch.cat([pre, range_text, range_text, shf])
            elif len(pre) > 0 and len(range_text) > 0:
                new_y = torch.cat([pre, range_text, range_text])
            elif len(range_text) > 0 and len(shf) > 0:
                new_y = torch.cat([range_text, range_text, shf])
            else:
                new_y = y_o[b, :y_lens[b]]  # Fallback to original if indices are problematic

        else:  # lost_P
            pre = y_o[b, :range_idx[0]]
            shf = y_o[b, range_idx[1]:y_lens[b]]

            # Handle edge cases
            if len(pre) > 0 and len(shf) > 0:
                new_y = torch.cat([pre, shf])
            elif len(pre) > 0:
                new_y = pre
            elif len(shf) > 0:
                new_y = shf
            else:
                new_y = y_o[b, :y_lens[b]]  # Fallback to original if indices are problematic

        reject_y.append(new_y)
        reject_y_lens.append(len(new_y))

    # Pad sequences to max length efficiently
    max_length = max(reject_y_lens)
    padded_reject_y = []

    # Efficient padding
    for b in range(bs):
        pad_length = max_length - reject_y_lens[b]
        if pad_length > 0:
            padding = torch.zeros(pad_length, dtype=y_o.dtype, device=y_o.device)
            padded_reject_y.append(torch.cat([reject_y[b], padding]))
        else:
            padded_reject_y.append(reject_y[b])

    return torch.stack(padded_reject_y, dim=0), torch.tensor(reject_y_lens, device=y_lens.device)
