import copy
import json
import os
import tempfile
from os.path import join
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.patches import Rectangle
import xml.etree.ElementTree as ET
from tiatoolbox.wsicore.wsireader import WSIReader

from model.transcriptomics_engine import get_transcriptomics_data_visualisation, tensor_fingerprint
import utils
from config import Config
from utils import device
from data_utils.patch_batch import from_raw_slide
from data_utils.slide import RawSlide, load_raw_slide
from model.interface import RecursiveModel
from model.image_encoder import from_name
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import yaml

# dotenv
from dotenv import load_dotenv

load_dotenv("/home/sn666/dissertation/.env")


def load_gene_list(st_model_config_file: str):
    st_model_config = yaml.safe_load(open(st_model_config_file, "r"))
    gene_sample_selection = st_model_config["dataloader_config"][
        "sample_gene_selection"
    ]

    st_config = json.load(open(gene_sample_selection, "r"))
    gene_list = st_config["highly_variable_genes"]
    return gene_list


def parse_camelyon17_anno_file(path: str):
    assert os.path.isfile(path), f"Couldn't find annotation file at '{path}'."
    # Parse the XML file
    tree = ET.parse(path)
    root = tree.getroot()

    # Check the group name
    group_name = root.find(".//Group").get("Name")
    if group_name != "Tumor":
        raise ValueError(f"Unexpected group name: {group_name}")

    # Initialize list to hold polygon data
    polygons = []

    # Iterate over annotations and extract coordinates
    for annotation in root.findall(".//Annotation"):
        if annotation.get("Type") != "Polygon":
            raise ValueError(f"Unexpected annotation type: {annotation.get('Type')}")

        # Get color and coordinates
        color = annotation.get("Color")
        coords = [
            (float(coord.get("X")), float(coord.get("Y")))
            for coord in annotation.find("Coordinates")
        ]

        # Append polygon data to list
        polygons.append((coords, color))

    return polygons


# `pix` pixels at depth `depth` to a different depth
def convert_pix(pix, depth, to_depth):
    e = to_depth - depth
    if e <= 0:
        return pix // 2 ** (-e)
    else:
        return pix * 2**e


def to_pix_space(depth, y, x):
    return convert_pix(y, depth, 0), convert_pix(x, depth, 0)


def plot_all_gene_expression_overlay(
    config: Config,
    model: RecursiveModel,
    slide_depths: list,
    full_gene_list: list,
    chosen_genes: list,
    base_img: np.ndarray,
    transform,
    transcriptomics_model_path: str,
    out_path: str = None,
    image_encoder=None,
    pad: int = 128,
    P: int = 256,
    # which level should we compute transcriptomics at
    visualisation_depth: int = 5,
    slide_path: str = None,
    genes_to_visualise: list = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # so the best way to do this is to take the slide at the top of the hierarchy
    # and convert all it's patches to it's highest mag children
    # essentially, we recurse without any importance calculation

    # top level side
    slide = copy.deepcopy(slide_depths[0])
    # let's print out the locs
    # print("Top level slide locs:", slide.locs)

    # now let's recursively iterate to the bottom level
    # without any importance filtering

    for depth in range(visualisation_depth):
        # print(f"Recursing to depth {depth + 1} / {visualisation_depth}...")

        slide = slide.simple_recurse(
            multiplier=config.magnification_factor,
        )
        slide.camelyon = True
        slide.load_patches(process_ctx=False)

        # print(f"Locs at depth {depth + 1} / {visualisation_depth}: {slide.locs.shape[0]}")

    transcriptomics_results = []
    patches = []
    with torch.no_grad():
        for patch_loc in slide.locs:
            # print(patch_loc)
            patch_data = slide.get_patch_data(patch_loc)
            patches.append(transform(patch_data))

    patches = [patch.unsqueeze(0).to(device) for patch in patches]
    patches = torch.cat(patches)

    # print(f"Shape of patches before encoder: {patches.shape}")
    # infer in batches
    encoded_patches = []
    inference_batch_size = 4
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i in range(0, len(patches), inference_batch_size):
            batch_patches = patches[i : i + inference_batch_size]
            # print(f"Shape of batch_patches: {batch_patches.shape}")
            # run the encoder
            batch_patches = image_encoder(batch_patches)
            # write to a file
            torch.save(
                batch_patches,
                os.path.join(
                    tmpdirname,
                    tensor_fingerprint(patches[i : i + inference_batch_size]) + ".pt",
                ),
            )
            del batch_patches
            # encoded_patches.append(batch_patches)
        # load all encoded patches
        for i in range(0, len(patches), inference_batch_size):
            batch_patches = torch.load(
                os.path.join(
                    tmpdirname,
                    tensor_fingerprint(patches[i : i + inference_batch_size]) + ".pt",
                )
            )
            encoded_patches.append(batch_patches)
    # concatenate all encoded patches
    patches = torch.cat(encoded_patches)

    # patches = image_encoder(patches)
    # print(f"Shape of patches after encoder: {patches.shape}")

    print(f"transcriptomics model path visualisation: {transcriptomics_model_path}")
    transcriptomics_results = get_transcriptomics_data_visualisation(
        patches, transcriptomics_model_path=transcriptomics_model_path
    )

    for gene_idx, gene_name in enumerate(full_gene_list):
        if genes_to_visualise is not None:
            if gene_name not in genes_to_visualise:
                continue
        print(f"Visualising gene {gene_name}")
        if gene_name in chosen_genes:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(base_img)
            ax.axis("off")

            expressions = transcriptomics_results[:, gene_idx].cpu()
            vmin, vmax = np.quantile(expressions, [0.05, 0.95])
            norm = plt.Normalize(vmin, vmax)
            cmap = plt.get_cmap("inferno")

            for patch_loc, expr in zip(slide.locs, expressions):
                y_base, x_base = to_pix_space(visualisation_depth, *patch_loc)
                patch_size = convert_pix(P, visualisation_depth, 0)

                rect = Rectangle(
                    (x_base, y_base),
                    patch_size,
                    patch_size,
                    facecolor=cmap(norm(expr)),
                    edgecolor="red",
                    linewidth=0.8,
                    alpha=0.7,
                )
                ax.add_patch(rect)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label(f"{gene_name} Expression", rotation=270, labelpad=15)

            y_coords = [to_pix_space(depth, loc[0], 0)[0] for loc in slide.locs]

            # if y_coords:
            #     ax.set_ylim(max(y_coords) + pad + P, min(y_coords) - pad)

            if out_path is not None:
                safe_gene_name = "".join(c if c.isalnum() else "_" for c in gene_name)

                # gene_path = f"{out_path}_{safe_gene_name}_expression.pdf"
                gene_path = os.path.join(
                    out_path,
                    f"{os.path.split(slide_path)[-1]}_{safe_gene_name}_expression.pdf",
                )
                plt.savefig(gene_path, format="pdf", dpi=300, bbox_inches="tight")
                # print(f"Saved {gene_name} expression plot to {gene_path}")

            plt.close(fig)

    return transcriptomics_results

def plot_selected_patches_gene_expression_overlay(
    config: Config,
    slide_depths: list,
    gene_list: list,
    base_img: np.ndarray,
    transform,
    transcriptomics_model_path_visualisation: str,
    out_path: str = None,
    image_encoder=None,
    pad: int = 128,
    P: int = 256,
):
    """
    Plot the gene expression levels of only the selected patches
    at the absolute highest magnification level.

    This could be useful for visualising the true transcriptomic profiles
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    highest_mag_slide = slide_depths[-1]
    highest_mag_patches = highest_mag_slide.locs
    depth = config.num_levels - 1

    transcriptomics_results = []
    patches = []
    with torch.no_grad():
        for patch_loc in highest_mag_patches:
            # print(patch_loc)
            patch_data = highest_mag_slide.get_patch_data(patch_loc)
            patches.append(transform(patch_data))
            # print(f"Shape of added patch: {patches[-1].shape}")

    patches = [patch.unsqueeze(0).to(device) for patch in patches]
    # patches is a list of tensors, each with shape (1, C, P, P)
    # Concatenate them into a single tensor
    # Shape: (N, C, P, P)
    patches = torch.cat(patches)

    # run the encoder
    # print(f"Shape of patches before encoder: {patches.shape}")
    patches = image_encoder(patches)
    # print(f"Shape of patches after encoder: {patches.shape}")

    transcriptomics_results = get_transcriptomics_data_visualisation(
        patches, transcriptomics_model_path=transcriptomics_model_path_visualisation
    )

    # visualization loop
    for gene_idx, gene_name in enumerate(gene_list):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(base_img)
        ax.axis("off")

        expressions = transcriptomics_results[:, gene_idx].cpu()
        vmin, vmax = np.quantile(expressions, [0.05, 0.95])
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.get_cmap("inferno")

        # print(highest_mag_patches)

        # os._exit(0)

        for patch_loc, expr in zip(highest_mag_patches, expressions):
            y_base, x_base = to_pix_space(depth, *patch_loc)
            patch_size = convert_pix(P, depth, 0)

            # print(f"Patch loc: {patch_loc}, y_base: {y_base}, x_base: {x_base}")
            # print(f"Patch size: {patch_size}")

            rect = Rectangle(
                (x_base, y_base),
                patch_size,
                patch_size,
                facecolor=cmap(norm(expr)),
                edgecolor="red",
                linewidth=0.8,
                alpha=0.7,
            )
            ax.add_patch(rect)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label(f"{gene_name} Expression", rotation=270, labelpad=15)

        y_coords = [to_pix_space(depth, loc[0], 0)[0] for loc in highest_mag_patches]

        if y_coords:
            ax.set_ylim(max(y_coords) + pad + P, min(y_coords) - pad)

        if out_path is not None:
            safe_gene_name = "".join(c if c.isalnum() else "_" for c in gene_name)

            gene_path = f"{out_path}_{safe_gene_name}_expression.pdf"
            plt.savefig(gene_path, format="pdf", dpi=300, bbox_inches="tight")
            # print(f"Saved {gene_name} expression plot to {gene_path}")

        plt.close(fig)

def compute_attention_gene_overlap(
    base_img: np.ndarray,
    slide_depths: list,
    transcriptomics_results: torch.Tensor,
    importance_scores: list,
    chosen_genes: list,
    full_gene_list: list,
    P: int,
    threshold_quantile: float = 0.90,
):
    """
    Compute overlap metrics (IoU, Pearson correlation, Cosine similarity)
    between gene expression and model attention per level.

    Returns:
        A pandas DataFrame: rows = genes, columns = metrics per level.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics.pairwise import cosine_similarity

    h, w = base_img.shape[:2]
    results = []

    for gene_name in chosen_genes:
        gene_idx = full_gene_list.index(gene_name)
        row = {"Gene": gene_name}

        for level, (slide, importance) in enumerate(zip(slide_depths, importance_scores)):
            expr_values = transcriptomics_results[:, gene_idx].cpu().numpy()
            attn_values = importance  # shape: (N,) at this level

            # Normalize values
            expr_norm = (expr_values - np.min(expr_values)) / (np.max(expr_values) - np.min(expr_values) + 1e-8)
            attn_norm = (attn_values - np.min(attn_values)) / (np.max(attn_values) - np.min(attn_values) + 1e-8)

            # Initialize heatmaps``
            expr_heatmap = np.zeros((h, w), dtype=float)
            attn_heatmap = np.zeros((h, w), dtype=float)

            for (y, x), e, a in zip(slide.locs, expr_norm, attn_norm):
                y_pix, x_pix = to_pix_space(level, y, x)
                patch_size = convert_pix(P, level, 0)

                expr_heatmap[y_pix:y_pix+patch_size, x_pix:x_pix+patch_size] += e
                attn_heatmap[y_pix:y_pix+patch_size, x_pix:x_pix+patch_size] += a

            # Flatten for numerical comparison
            expr_flat = expr_heatmap.flatten()
            attn_flat = attn_heatmap.flatten()

            # Pearson correlation
            pearson_corr = np.corrcoef(expr_flat, attn_flat)[0, 1]

            # Cosine similarity
            cosine_sim = cosine_similarity(expr_flat.reshape(1, -1), attn_flat.reshape(1, -1))[0, 0]

            # Binary IoU of top-quantile regions
            expr_mask = expr_heatmap >= np.quantile(expr_heatmap, threshold_quantile)
            attn_mask = attn_heatmap >= np.quantile(attn_heatmap, threshold_quantile)
            intersection = np.logical_and(expr_mask, attn_mask).sum()
            union = np.logical_or(expr_mask, attn_mask).sum()
            
            iou = intersection / (union + 1e-8)
            soft_iou = np.sum(np.minimum(expr_heatmap, attn_heatmap)) / np.sum(np.maximum(expr_heatmap, attn_heatmap))
            
            # Store results
            row[f"Level {level} - Pearson"] = round(pearson_corr, 4)
            row[f"Level {level} - Cosine"] = round(cosine_sim, 4)
            row[f"Level {level} - IoU@{threshold_quantile}"] = round(iou, 4)
            row[f"Level {level} - Soft IoU"] = round(soft_iou, 4)

        results.append(row)

    df = pd.DataFrame(results)
    return df

def heatmap_camelyon17_transcriptomics(
    config: Config,
    model: RecursiveModel,
    image_encoder,
    transform,
    slide_path: str,
    annotation_path: str,
    out_path: str,
    full_gene_list: list,
    chosen_genes: list,
    transcriptomics_visualisation_depth: int,
    transcriptomics_model_path_model: str,
    transcriptomics_model_path_visualisation: str,
    genes_to_visualise: list = None,
):
    # check if the WSI exists
    assert os.path.isfile(slide_path), f"Couldn't find WSI at path '{slide_path}'."

    # create the output directory if necessary
    if out_path is not None:
        directory = os.path.join(*os.path.split(out_path)[:-1])
        if directory != "" and not os.path.isdir(directory):
            print("Creating directory:", directory)
            os.makedirs(directory, exist_ok=True)

    if annotation_path is not None:
        assert os.path.isfile(annotation_path), (
            f"Couldn't find annotation XML file at path '{annotation_path}'. The "
            f"annotation file is optional, but if the argument is passed, "
            f"the file must exist."
        )

    def get_slide_rgb():
        return slide.view_at_power(config.base_power)

    L = config.num_levels
    P = config.model_config.patch_size

    slide = load_raw_slide(
        slide_path,
        config.base_power,
        config.model_config.patch_size,
        model.procs[0].ctx_dim(),
        prepatch=False,
        tissue_threshold=0.025,
    )

    slide.camelyon = True
    slide.load_patches()

    slide_depths = [slide]
    imps = []

    print("Recursing...")
    for depth in range(config.num_levels):
        print(f" Depth {depth + 1} / {config.num_levels}...")
        data = from_raw_slide(
            slide,
            image_encoder,
            transform,
            transcriptomics_model_path=transcriptomics_model_path_model,
        )
        out = model(depth, data)
        ctx_slide = out["ctx_slide"][0]
        ctx_patch = out["ctx_patch"][0]
        importance = out["importance"][0]
        imps.append(importance.detach().cpu().numpy())

        if depth != config.num_levels - 1:
            slide = slide.recurse(
                config.magnification_factor,
                ctx_slide.cpu(),
                ctx_patch.cpu(),
                importance.cpu(),
                config.top_k_patches[depth],
            )
            slide.camelyon = True
            slide.load_patches()
            slide_depths.append(slide)

    # print the number of locs at each depth
    for depth in range(config.num_levels):
        print(
            f"Depth {depth + 1} / {config.num_levels}: {slide_depths[depth].locs.shape[0]}"
        )

    # get the base image for visualization
    bigimg = get_slide_rgb()

    transcriptomics_results = plot_all_gene_expression_overlay(
        config=config,
        model=model,
        slide_depths=slide_depths,
        full_gene_list=full_gene_list,
        chosen_genes=chosen_genes,
        base_img=bigimg,
        transform=transform,
        transcriptomics_model_path=transcriptomics_model_path_visualisation,
        out_path=out_path,
        image_encoder=image_encoder,
        pad=128,
        P=config.model_config.patch_size,
        visualisation_depth=transcriptomics_visualisation_depth,
        slide_path=slide_path,
        genes_to_visualise=genes_to_visualise,
    )

    H, W, C = bigimg.shape
    assert C == 3
    print("Bigimg:", H, "x", W, "x", C)

    ns = [s.locs.shape[0] for s in slide_depths]
    print("Patch counts:", ns)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.4))

    sax = axes[0]
    sax.imshow(bigimg, aspect="equal")
    sax.set_xticks([])
    sax.set_yticks([])

    if annotation_path is not None:
        print("Plotting CAMELYON17 slide annotations")
        try:
            polygons = parse_camelyon17_anno_file(annotation_path)
        except Exception as e:
            raise ValueError(
                f"Failed to parse CAMELYON17 annotation path at '{annotation_path}'."
            ) from e

        multiplier = config.base_power / 40
        for coords, color in polygons:
            x, y = zip(*coords)
            x = [i * multiplier for i in x]
            y = [i * multiplier for i in y]
            axes[0].plot(
                x + [x[0]], y + [y[0]], color="blue", linewidth=2
            )  # Close the loop

    ax = axes[1]
    ax.imshow(bigimg, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])

    shape = (H, W)
    overall_imp = np.zeros((L,) + shape)  # 0 = padding

    s1, s2 = shape

    # DRAW WIREFRAME RECTS
    for depth in range(L):
        locs = slide_depths[depth].locs
        size = convert_pix(P, depth, 0)
        lw = 0.5

        imp = imps[depth]

        for i in range(locs.shape[0]):
            y, x = locs[i].tolist()
            y, x = to_pix_space(depth, y, x)
            rect = Rectangle(
                (x, y), size, size, facecolor="None", edgecolor="black", lw=lw
            )
            ax.add_patch(rect)

            overlap_y = max(y, 0) <= min(y + size, s1)
            overlap_x = max(x, 0) <= min(x + size, s2)
            if overlap_y and overlap_x:
                y1 = max(y, 0)
                y2 = min(y + size, s1)
                x1 = max(x, 0)
                x2 = min(x + size, s2)
                overall_imp[depth, y1:y2, x1:x2] = imp[i] + 1e-4

    # Weight importances by 1/2^depth
    for depth in range(L - 2, -1, -1):
        relevant_mask = overall_imp[depth + 1] != 0
        overall_imp[depth][relevant_mask] = (
            overall_imp[depth][relevant_mask]
            + overall_imp[depth + 1][relevant_mask] * 0.5
        )

    overall_imp = overall_imp[0]

    alpha = np.where(overall_imp > 0, 0.5, 0)
    overall_imp[overall_imp == 0] = np.min(overall_imp[overall_imp > 0])

    print(bigimg.shape, overall_imp.shape)
    hm = ax.imshow(overall_imp, cmap="viridis", alpha=alpha, aspect="equal")

    # Try to choose an appropriate viewport: look at the positions of patches
    y = slide_depths[0].locs[:, 0].tolist()
    h = bigimg.shape[0]

    # exclude top/bottom patches: if present, these are often background removal failures
    thresh = 0.1
    y = [i for i in y if 0.1 < (i + P / 2) / h < 1 - thresh]

    pad = 128
    axes[0].set_ylim(max(y) + pad + P, min(y) - pad)
    axes[1].set_ylim(max(y) + pad + P, min(y) - pad)

    cax = inset_axes(axes[1], width="5%", height="100%", loc="right", borderpad=-1.5)
    fig.colorbar(hm, cax=cax, orientation="vertical")

    fig.tight_layout()
    fig.subplots_adjust(right=0.9)

    if out_path is not None:
        # if not out_path.endswith(".pdf"):
        #     out_path += ".pdf"
        diagram_out_path = os.path.join(
            out_path, f"{os.path.split(slide_path)[-1]}_transcriptomics_{config.model_config.add_transcriptomics}_heatmap.pdf"
        )
        plt.savefig(diagram_out_path, format="pdf", dpi=200)
    plt.show()
    
    overlap_df = compute_attention_gene_overlap(
        base_img=bigimg,
        slide_depths=slide_depths,
        transcriptomics_results=transcriptomics_results,
        importance_scores=imps,
        chosen_genes=chosen_genes,
        full_gene_list=full_gene_list,
        P=config.model_config.patch_size,
    )
    overlap_df.to_csv(
        os.path.join(
            out_path,
            f"{os.path.split(slide_path)[-1]}_transcriptomics_{config.model_config.add_transcriptomics}_gene_attention_overlap.csv",
        ),
        index=False,
    )
    print("Overlap metrics saved to CSV.")

    # import ace_tools as tools; tools.display_dataframe_to_user("Gene-Attention Overlap Metrics", overlap_df)

def load_genes(gene_list_path: str):
    with open(gene_list_path, "r") as f:
        gene_list = [line.strip() for line in f.readlines()]
    return gene_list

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-dir",
        required=True,
        type=str,
        help="Path to model directory. Must contain config.json file.",
    )
    parser.add_argument(
        "-s", "--slide-path", required=True, type=str, help="Path to the WSI."
    )
    parser.add_argument(
        "-a",
        "--annotation-path",
        default=None,
        type=str,
        help="For CAMELYON17 slides, the path to the XML annotation file. Do not pass this argument for non-CAMELYON17 slides, or slides without an annotation file.",
    )
    parser.add_argument(
        "-c",
        "--st-model-config",
        default=None,
        type=str,
        help="Path to the transcriptomics model config file. This is used to load the gene list.",
    )
    parser.add_argument(
        "-t",
        "--tmpm",
        default=None,
        type=str,
        help="Path to the transcriptomics model checkpoint.",
    )
    parser.add_argument(
        "-u",
        "--tmpv",
        default=None,
        type=str,
        help="Path to the transcriptomics model checkpoint for visualisation.",
    )
    parser.add_argument(
        "-g",
        "--gene-list",
        default=None,
        type=str,
        help="Path to the gene list file. This is used to load the gene list.",
    )
    parser.add_argument(
        "-o",
        "--out-path",
        default=None,
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-v",
        "--genes_to_visualise",
        default=None,
        help="Comma-separated list of genes to visualise. If not provided, all genes in the gene list will be visualised.",
    )
    # parser.add_argument(
    #     "-o",
    #     "--out",
    #     default=None,
    #     type=str,
    #     help="Output a PDF of the visualisation to the given path.",
    # )
    args = parser.parse_args()

    model_name = os.path.split(args.model_dir)[-1]

    config = Config.load(
        args.model_dir, test_mode=True
    )  # test_mode stops error when checking existence of data dirs

    print(config)
    
    # outpath is a directory, if it doesn't exist, create it
    if (args.out_path is not None) and (not os.path.isdir(args.out_path)):
        os.makedirs(args.out_path, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = config.get_model()
    model = model.eval().to(device)
    train_stats = utils.load_state(args.model_dir, model, map_location=device)
    print("Loaded from epoch", train_stats["epoch"])

    name = os.path.split(args.model_dir)[-1]

    image_encoder, dimension, transform = from_name("UNI")
    image_encoder.to(device)

    # KRT5: A top diagnostic biomarker for squamous cell carcinoma (LUSC) with AUC >0.97 in LUAD/LUSC differentiation.

    # S100A2: Linked to unfavorable prognosis in LUAD and elevated in LUSC.

    # TRIM29: Associated with tumor progression and poor prognosis in LUAD.

    # MUC1: Overexpressed in LUAD and linked to metastasis.

    # QSOX1: Shows prognostic value in LUAD, with higher expression correlating with better outcomes

    print(f"Model path model: {args.tmpm}")
    print(f"Model path visualisation: {args.tmpv}")
    print(f"Genes to visualise: {args.genes_to_visualise}")

    heatmap_camelyon17_transcriptomics(
        config,
        model,
        image_encoder,
        transform,
        args.slide_path,
        args.annotation_path,
        # out_path=f"/home/sn666/dissertation/benchmarking/PATHS/visualisations/",
        out_path=args.out_path,
        full_gene_list=load_gene_list(args.st_model_config),
        # chosen_genes=["ENG", "A2M", "SOD1", "TCFBR2"],
        chosen_genes=load_genes(args.gene_list),
        transcriptomics_visualisation_depth=3,
        transcriptomics_model_path_model=args.tmpm,
        transcriptomics_model_path_visualisation=args.tmpv,
        genes_to_visualise=args.genes_to_visualise.split(",") if args.genes_to_visualise else None,
    )
