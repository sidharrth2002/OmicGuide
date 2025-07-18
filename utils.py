import torch
import torch.nn.functional as F
import os
from os.path import join
import wandb
import math
import pickle
from typing import Tuple, Callable, Dict, List
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from preprocess import loader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MAX_WORKERS = 8

class ResidualTranscriptomicsFusion(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        input_dim = feat1_dim + feat2_dim
        output_dim = feat1_dim

        print(f"input dim: {input_dim}, output_dim: {output_dim}, hidden_dim: {hidden_dim}")

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, feat_1, feat_2):
        # feat_1: [B, N, D], feat_2: [B, N, F]
        fused_input = torch.cat([feat_1, feat_2], dim=-1)
        delta = self.proj(fused_input)

        # Zero out delta where transcriptomics is all zeros
        valid = (feat_2.abs().sum(dim=-1, keepdim=True) != 0).float()
        delta = delta * valid

        return feat_1 + delta


def positional_encoding(length, dim, device=torch.device('cpu'), k=10000.0):
    """Generate the usual sinusoidal positional encoding"""
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(k) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def positional_encoding_2d(n, m, dim, device=torch.device('cpu'), k=10000.0):
    """
    Generate 2D positional encoding for a grid of size (n, m).
    PE2D(h, w) = PE1D(h) || PE1D(w)
    Return shape: (n x m x dim)
    """
    position1 = torch.arange(n, device=device).unsqueeze(1)
    position2 = torch.arange(m, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim // 2, 2, device=device) * (-math.log(k) / dim))

    pe1 = torch.zeros(n, 1, dim // 2, device=device)
    pe1[:, 0, 0::2] = torch.sin(position1 * div_term)
    pe1[:, 0, 1::2] = torch.cos(position1 * div_term)

    pe2 = torch.zeros(1, m, dim // 2, device=device)
    pe2[0, :, 0::2] = torch.sin(position2 * div_term)
    pe2[0, :, 1::2] = torch.cos(position2 * div_term)

    return torch.cat([pe1.expand(n, m, dim // 2), pe2.expand(n, m, dim // 2)], dim=2)


def positional_encoding_2d_from_pos(xpos, ypos, dim, device=torch.device('cpu'), k=10000.0):
    """
    Generate 2D positional encoding for N points with known x/y positions.
    xpos : n,
    ypos : n,
    PE2D(h, w) = PE1D(h) || PE1D(w)
    Return shape: (n x dim)
    """
    n = xpos.shape[0]
    div_term = torch.exp(torch.arange(0, dim // 2, 2, device=device) * (-math.log(k) / dim))[None]

    xpos = xpos.unsqueeze(-1)
    ypos = ypos.unsqueeze(-1)

    pe = torch.zeros(n, dim, device=device)
    pe[:, 0:dim // 2:2] = torch.sin(xpos * div_term)
    pe[:, 1:dim // 2:2] = torch.cos(xpos * div_term)
    pe[:, dim // 2::2] = torch.sin(ypos * div_term)
    pe[:, (dim // 2)+1::2] = torch.cos(ypos * div_term)

    return pe


def positional_encoding_2d_batched(batch_size, n, m, x_off, y_off, dim, device=torch.device('cpu'), k=10000.0):
    """
    Generate 2D positional encoding for a grid of size (n, m) but with a given offset for every batch item.
    PE2D(h, w) = PE1D(h) || PE1D(w)

    x_off: batch_size x n
    y_off: batch_size x m
    Return shape: batch_size x n x m x dim
    """
    position1 = x_off.unsqueeze(-1) + torch.arange(n, device=device)[None]
    position2 = y_off.unsqueeze(-1) + torch.arange(m, device=device)[None]
    div_term = torch.exp(torch.arange(0, dim // 2, 2, device=device) * (-math.log(k) / dim))[None][None]

    # position : B x N
    # div_term : 1 x 1 x dim/2

    pe1 = torch.zeros(batch_size, n, 1, dim // 2, device=device)
    pe1[:, :, 0, 0::2] = torch.sin(position1.unsqueeze(-1) * div_term)
    pe1[:, :, 0, 1::2] = torch.cos(position1.unsqueeze(-1) * div_term)

    pe2 = torch.zeros(batch_size, 1, m, dim // 2, device=device)
    pe2[:, 0, :, 0::2] = torch.sin(position2.unsqueeze(-1) * div_term)
    pe2[:, 0, :, 1::2] = torch.cos(position2.unsqueeze(-1) * div_term)

    return torch.cat([pe1.expand(batch_size, n, m, dim // 2), pe2.expand(batch_size, n, m, dim // 2)], dim=3)


def padding_mask(xs: torch.Tensor, lengths: torch.LongTensor):
    """
    Given a batch of embedded sequence data of shape (B x S x D) and the lengths (B) of each sequence,
    produces a padding mask of shape (B x S).
    """
    batch_size, max_seq_length, _ = xs.shape
    return torch.arange(max_seq_length, device=lengths.device)[None] >= lengths[:, None]


# def apply_to_non_padded(network: Callable, xs: torch.Tensor, transcriptomics: torch.Tensor, inds: torch.BoolTensor, output_dim: int):
#     """
#     Applies a module to only the non-padded indices in sequence `xs`. Padded locations are populated with zeros.
#     `inds` gives the non-padded indices.
#     `network`'s output must be of dimension `output_dim`.
#     """
#     batch_size, max_seq = xs.shape[:2]
#     network_out = network(xs[inds])
#     out = torch.zeros((batch_size, max_seq, output_dim), device=xs.device, dtype=network_out.dtype)
#     out[inds] = network_out
#     return out

def apply_to_non_padded(network: Callable, xs: torch.Tensor, inds: torch.BoolTensor, output_dim: int):
    """
    Applies a module to only the non-padded indices in sequence `xs`. Padded locations are populated with zeros.
    `inds` gives the non-padded indices.
    `network`'s output must be of dimension `output_dim`.
    """
    batch_size, max_seq = xs.shape[:2]
    out = torch.zeros((batch_size, max_seq, output_dim), device=xs.device)
    out[inds] = network({
        "contextualised_features": xs[inds], 
    })
    return out

# def apply_to_non_padded(network: Callable, xs: torch.Tensor, transcriptomics: torch.Tensor, inds: torch.BoolTensor, output_dim: int):
#     """
#     Applies a module to only the non-padded indices in sequence `xs`. Padded locations are populated with zeros.
#     `inds` gives the non-padded indices.
#     `network`'s output must be of dimension `output_dim`.
#     """
#     batch_size, max_seq = xs.shape[:2]
#     out = torch.zeros((batch_size, max_seq, output_dim), device=xs.device)
#     print(f"xs shape: {xs.shape}")
#     print(f"transcriptomics shape: {transcriptomics.shape}")
#     # print(f"transcriptomics[0] shape: {transcriptomics[0].shape}")
#     # print(f"inds: {inds}")
#     # transcriptomics = transcriptomics[0].to(device=xs.device)
#     out[inds] = network({
#         "contextualised_features": xs[inds], 
#         "transcriptomics": transcriptomics[inds]
#     })
#     return out


def next_multiple(n: int, m: int):
    """Returns lowest multiple of m greater than or equal to n."""
    return m * math.ceil(n / m)


def patchify(ims: torch.Tensor, patch_size: int, channels: int = 3):
    """
    Splits a (N x 3 x H x W) batch of images into patches, returning a tensor of shape (N x M x 3 x P x P)
    where M = (H/P)*(W/P). patch_size must divide the height (H) and width (W) of the image batch.
    """
    n = ims.shape[0]
    patched = ims.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) # N x 3 x H' x W' x P x P
    patched = patched.permute(0, 2, 3, 1, 4, 5)  # N x H' x W' x 3 x P x P
    return patched.contiguous().view(n, -1, channels, patch_size, patch_size).contiguous()  # N x (H'W') x 3 x P x P


def patchify_locs(ims: torch.Tensor, patch_size: int, im_locs: torch.LongTensor):
    """
    Patchifies a batch of images (see `patchify` for details) but also computes the new locations of the patches in
    the slide at this resolution.
    """
    n, c, h, w = ims.shape
    assert n == im_locs.shape[0]
    patches = patchify(ims, patch_size)

    h2, w2 = h // patch_size, w // patch_size

    hmul = torch.arange(h2, device=im_locs.device).repeat_interleave(w2)
    wmul = torch.arange(w2, device=im_locs.device).repeat(h2)
    offsets = torch.cat((hmul[:, None], wmul[:, None]), dim=1) * patch_size

    # offsets   : (HW/P^2) x 2
    # im_locs   : N x 2
    locs = offsets[None] + im_locs[:, None]

    # Patches   : N x (HW/P^2) x 3 x P x P
    # Locs      : N x (HW/P^2) x 2
    return patches, locs


def wandb_get_id(folder: str):
    if os.path.isfile(join(folder, "wandb_id")):
        with open(join(folder, "wandb_id"), "r") as f:
            return f.readline().strip()
    else:
        wid = wandb.util.generate_id()
        with open(join(folder, "wandb_id"), "w") as f:
            f.write(wid)
        return wid


def save_state(root_path: str, model, train_stats):
    """Saves model and train stats to separate files."""
    model_path = join(root_path, "model.pt")
    train_stats_path = join(root_path, "train_stats.pkl")

    print(f"Saving to {root_path}...")
    torch.save(model.state_dict(), model_path)

    with open(train_stats_path, "wb") as file:
        pickle.dump(train_stats, file)


def load_state(root_path: str, model, map_location=device) -> Dict:
    """Loads the model and train stats, returning the train stats"""
    model_path = join(root_path, "model.pt")
    train_stats_path = join(root_path, "train_stats.pkl")

    if not os.path.isfile(model_path):
        print(f"{model_path} not found, not loading model state!")
    else:
        model.load_state_dict(torch.load(model_path, map_location=map_location))

    if not os.path.isfile(train_stats_path):
        print("No train stats found, assuming first run")
        return {"epoch": 1}

    with open(train_stats_path, "rb") as file:
        train_stats = pickle.load(file)

    return train_stats


def inference(model, depth, power, batch, importance_penalty, task: str):
    from data_utils import patch_batch  # circular imports...

    data = patch_batch.from_batch(batch, device)
    out = model(depth, data)

    logits = out["logits"]
    imp = out["importance"]

    if task == "survival":
        labels = batch["survival_bin"].to(device)
        censors = batch["censored"].to(device)

        hazards = torch.sigmoid(logits)

        loss_nll = nll_loss(hazards, labels, censors)

        return hazards, loss_nll

    elif task == "subtype_classification":
        subtypes = batch["subtype"].to(device)
        loss = F.cross_entropy(logits, subtypes)

        return logits, loss


# todo; should probably just move somewhere else to prevent circular imports
def inference_end2end(num_levels, keep_patches, model, base_power, batch, task: str, use_mixed_precision: bool = False,
                      random_rec_baseline: bool = False, magnification_factor: int = 2, transcriptomics_type: str = "none", transcriptomics_model_path: str = None, model_dir: str = None):
    from data_utils import patch_batch  # circular imports...
    from data_utils.slide import PreprocessedSlide
    from data_utils.dataset import collate_fn

    slides = batch["slide"]
    
    # what are the keys in batch?
    # print(f"Batch keys: {list(batch.keys())}")

    batch0 = batch
    power = base_power

    for i in range(num_levels):
        # save the batch to a file
        # with open(f"batch_level_{i}.pkl", "wb") as f:
        #     pickle.dump(batch["leaf_fts_grouped"], f)
        
        print(f"Level {i} / {num_levels}")
        locs_cpu = batch["locs"]

        with autocast(enabled=use_mixed_precision):
            data = patch_batch.from_batch(batch, device, transcriptomics_type=transcriptomics_type, transcriptomics_model_path=transcriptomics_model_path)
            out = model(i, data)

            importance = out["importance"]
            new_ctx_slide = out["ctx_slide"]
            new_ctx_patch = out["ctx_patch"]
            
            # log importance histogram to wandb
            wandb.log({
                f"importances/level_{i}": wandb.Histogram(importance.detach().cpu().numpy()),
            })

        if random_rec_baseline:
            importance = torch.randn_like(importance)

        if i != num_levels - 1:
            new_batch = []
            imp_cpu = importance.cpu().float()

            for j in range(len(slides)):
                # print(f"Processing slide {j} at level {i}")
                slide: PreprocessedSlide = slides[j]

                ind = i if magnification_factor == 2 else 2 * i

                x = slide.iter(ind, data.num_ims[j], locs_cpu[j], data.ctx_slide[j], data.ctx_patch[j], importance[j],
                               new_ctx_slide[j], new_ctx_patch[j], keep_patches[i], imp_cpu[j], 
                               # only compute leaves if the transcriptomics type is highest-magnification
                               # if not, we do inference at every magnification
                               return_leaf=(transcriptomics_type == "highest-magnification"))

                if magnification_factor == 4:
                    new_fts = x["fts"]
                    ctx_patch = x["ctx_patch"]
                    ctx_slide = x["ctx_slide"]
                    locs = x["locs"]
                    num_ims = new_fts.shape[0]
                    x = slide.iter(ind + 1, num_ims, locs, ctx_slide, ctx_patch, None,None, None, -1)

                # print(f"Slide {j} at level {i}")
                # print(f"Power: {power}")
                # for k, v in x.items():
                #     # if of type torch.tensor
                #     if isinstance(v, torch.Tensor):
                #         print(f"{k}: {v.shape}")
                # print(f"x parent_inds: {x['parent_inds'].shape}")
                
                new_batch.append(x)

            batch = collate_fn(new_batch)
            power *= 2

    logits = out["logits"].float()
    
    # pickle dump the out object in model_dir
    if model_dir is not None:
        with open(os.path.join(model_dir, "test_output.pkl"), "wb") as f:
            pickle.dump(out, f)

    if task == "survival":
        labels = batch0["survival_bin"].to(device)
        censors = batch0["censored"].to(device)

        hazards = torch.sigmoid(logits)

        loss_nll = nll_loss(hazards, labels, censors)

        if model_dir is not None:
            with open(os.path.join(model_dir, "test_gt_hazards.pkl"), "wb") as f:
                pickle.dump({
                    "hazards": hazards,
                    "labels": labels,
                    "censors": censors,   
                }, f)

        return hazards, loss_nll

    elif task == "subtype_classification":
        subtypes = batch0["subtype"].to(device)
        loss = F.cross_entropy(logits, subtypes)

        return logits, loss


def inference_baseline(model, batch, task: str, model_type: str):
    from data_utils import patch_batch  # circular imports...

    model_type = model_type.lower()

    data = patch_batch.from_batch(batch, device)

    if model_type == "zoommil":
        assert data.batch_size == 1, "ZoomMIL only supports a batch size of 1"
        slide = batch["slide"][0]
        data = convert_to_zoommil_fts(slide, model.power_levels)
        data = [i[None].to(device) for i in data]  # add unit batch dimension + move to cuda

    logits = model(data)

    if task == "survival":
        labels = batch["survival_bin"].to(device)
        censors = batch["censored"].to(device)

        hazards = torch.sigmoid(logits)

        loss = nll_loss(hazards, labels, censors)

        return hazards, loss

    elif task == "subtype_classification":
        subtypes = batch["subtype"].to(device)
        loss = F.cross_entropy(logits, subtypes)

        return logits, loss


# Cox NLL loss function taken from MCAT
def nll_loss(hazards, y, c, alpha=0.4, eps=1e-7):
    """
    Neural network is hazard probability function, h(t) for t = 0,1,2,...,k-1
    corresponding Y = 0,1, ..., k-1. h(t) represents the probability that patient dies in [0, a_1), [a_1, a_2), ..., [a_(k-1), inf]
    :param hazards: predicted probabilities for [0, a_1), [a_1, a_2), ... [a_(k-1), inf). Each value must be in range [0, 1].
    :param y: ground truth.
    :param c: censorship status.
    :param alpha: a value of 1 ignores censored data, and a value of 0 weights it equally to uncensored data.
    :return: Mean loss (scalar).
    """
    batch_size = hazards.shape[0]

    # Survival is cumulative product of 1 - hazards
    survival = torch.cumprod(1 - hazards, dim=1)
    # Left pad with 1s
    survival_padded = torch.cat([torch.ones((batch_size, 1), dtype=survival.dtype, device=survival.device), survival], dim=1)

    r = torch.arange(batch_size)
    uncensored_loss = -(1 - c) * (torch.log(survival_padded[r, y].clamp(min=eps)) + torch.log(hazards[r, y].clamp(min=eps)))
    censored_loss = -c * torch.log(survival_padded[r, y+1].clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    return loss.mean()


def cumcount(a):
    """
    Adapted from a numpy version on StackOverflow:
    https://stackoverflow.com/questions/40602269/how-to-use-numpy-to-get-the-cumulative-count-by-unique-values-in-linear-time
    """
    kwargs = {"device": a.device, "dtype": a.dtype}
    def dfill(a):
        n = a.shape[0]
        z = torch.zeros((1,), **kwargs)
        nr = torch.zeros((1,), **kwargs) + n
        b = torch.cat((z, torch.where(a[:-1] != a[1:])[0] + 1, nr))
        return torch.arange(n, **kwargs)[b[:-1]].repeat_interleave(torch.diff(b))

    def argunsort(s):
        n = s.shape[0]
        u = torch.zeros((n,), **kwargs)
        u[s] = torch.arange(n, **kwargs)
        return u

    n = a.shape[0]
    s = a.argsort(stable=True)
    i = argunsort(s)
    b = a[s]
    return (torch.arange(n, **kwargs) - dfill(b))[i]


def todevice(x, device):
    """Recursively moves all items of `x` to the given device. Works for nested lists/tuples onlys."""
    if hasattr(x, "to"):
        return x.to(device)
    elif isinstance(x, list) or isinstance(x, tuple):
        return [todevice(i, device) for i in x]
    else:
        return x


# From HEALNet
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, mode='min'):
        """
        Constructor for early stopping.

        Parameters:
        - patience (int): How many epochs to wait before stopping once performance stops improving.
        - verbose (bool): If True, prints out a message for each validation metric improvement.
        - mode (str): One of ['min', 'max']. Minimize (e.g., loss) or maximize (e.g., accuracy) the metric.
        """
        assert mode in ['min', 'max'], "Mode must be 'min' or 'max'"
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        if mode == 'min':
            self.best_metric = float('inf')
            self.operator = torch.lt
        else:
            self.best_metric = float('-inf')
            self.operator = torch.gt

        self.best_model_weights = None
        self.should_stop = False

    def step(self, metric, model):
        """
        Check the early stopping conditions.

        Parameters:
        - metric (float): The latest validation metric (loss, accuracy, etc.).
        - model (torch.nn.Module): The model being trained.

        Returns:
        - bool: True if early stopping conditions met, False otherwise.
        """
        if type(metric) == float: # convert to tensor if necessary
            metric = torch.tensor(metric)

        if self.operator(metric, self.best_metric):
            if self.verbose:
                print(f"Validation metric improved from {self.best_metric:.4f} to {metric:.4f}. Saving model weights.")
            self.best_metric = metric
            self.counter = 0
            self.best_model_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation metric did not improve. Patience: {self.counter}/{self.patience}.")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def load_best_weights(self, model):
        """
        Load the best model weights.

        Parameters:
        - model (torch.nn.Module): The model to which the best weights should be loaded.
        """
        if self.verbose:
            print(f"Loading best model weights with validation metric value: {self.best_metric:.4f}")
        model.load_state_dict(self.best_model_weights)
        return model


def convert_to_zoommil_fts(slide, power_levels: List[float]) -> List[torch.Tensor]:
    """
    Util to convert from our PreprocessedSlide to the specific 1D format zoommil expects.
    Output shape: [N x D, (M^2 N) x D, (M^4 N) x D]
      (in the case power_levels = [x, mx, m^2 x]. Also works for e.g. [x, m x, m^3 x])
    """
    data = []
    for power in power_levels:
        fts = loader.load(slide.preprocessed_root, slide.slide_id, power)  # H x W x D
        data.append(fts)

    # Background filter
    d0 = data[0]
    mask = torch.sum(d0.abs(), dim=-1) > 0  # [H x W] bool
    xs, ys = mask.nonzero(as_tuple=True)

    output = []

    # Get all patches at each level, such that
    for i, (tensor, power) in enumerate(zip(data, power_levels)):
        scale = round(power / power_levels[0])
        h, w, _ = tensor.shape

        # Scale up coordinates. E.g. x -> [2x, 2x+1] on the second iter
        scaled_xs = (xs * scale).view(-1, 1) + torch.arange(scale).view(1, -1)
        scaled_ys = (ys * scale).view(-1, 1) + torch.arange(scale).view(1, -1)

        #  (x, y) -> [(2x, 2y), (2x, 2y+1), (2x+1, 2y), (2x+1, 2y+1)]
        scaled_coords = torch.stack([torch.cartesian_prod(sx, sy) for sx, sy in zip(scaled_xs, scaled_ys)], dim=0)
        patch_coords = scaled_coords.view(-1, 2)

        # Clamp coordinates to be within bounds
        #  (coords are *almost always* within bounds. there's just the occasional off-by-one error
        #   at the edges due to slide dimensions not being perfect powers of two.)
        out_of_bounds = (patch_coords[:, 0] >= tensor.shape[0]) | (patch_coords[:, 1] >= tensor.shape[1])
        patch_coords[out_of_bounds] *= 0  # just query (0, 0), we will zero them anyway after
        gathered_patches = tensor[patch_coords[:, 0], patch_coords[:, 1]]
        gathered_patches[out_of_bounds] *= 0

        output.append(gathered_patches)

    return output
