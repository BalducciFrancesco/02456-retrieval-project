import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, BertModel

from src.retrievers.models.encoder import TransformerEncoder


def _heatmap(ax, m, tokens, title):
    # Fixed 0-1 scale so panels can be compared.
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=7)
    ax.set_xlabel("Key token (information read from)", fontsize=8)
    ax.set_ylabel("Query token (being updated)", fontsize=8)
    ax.set_title(title, fontsize=10)
    return im


@torch.no_grad()
def extract_attention(model, input_ids, attention_mask):
    """Attention maps for a batch, one [batch, heads, T, T] tensor per layer."""
    model.eval()  # dropout off
    if isinstance(model, TransformerEncoder):
        _, maps = model(input_ids, attention_mask, output_attentions=True)
    else:  # BERT
        maps = model(input_ids=input_ids, attention_mask=attention_mask,
                     output_attentions=True, return_dict=True).attentions
    # BERT's default (sdpa) attention returns None instead of the maps.
    if not maps or any(m is None for m in maps):
        raise ValueError("No attention maps returned; load pretrained models with attn_implementation='eager'")
    return tuple(m.cpu() for m in maps)


def _maps_to_plot(attentions, tokens, attention_mask, batch_index):
    """Pick one text, the first and last layer, up to 4 heads, and drop padding."""
    # one tensor per layer, shape [batch, heads, T, T]
    n_heads = attentions[0].shape[1]
    # positions that are not padding
    keep = attention_mask[batch_index].cpu().bool().nonzero().flatten()
    layers = sorted({0, len(attentions) - 1})  # first and last layer
    heads = list(range(min(n_heads, 4)))
    # add the position so the same word twice gets two different labels
    labels = [f"{i}: {tokens[i]}" for i in keep.tolist()]
    selected = {}
    for layer in layers:
        for head in heads:
            matrix = attentions[layer][batch_index, head].cpu()
            # remove padding rows and columns
            selected[layer, head] = matrix.index_select(0, keep).index_select(1, keep)
    return selected, labels, layers, heads


def plot_attention(attentions, tokens, attention_mask, *, batch_index=0,
                   title="Encoder attention", output_path=None):
    """Heatmap grid: one row per layer, one column per head."""
    selected, labels, layers, heads = _maps_to_plot(attentions, tokens, attention_mask, batch_index)
    side = max(4.0, len(labels) * 0.28)
    fig, axes = plt.subplots(len(layers), len(heads), squeeze=False,
                             figsize=(side * len(heads), side * len(layers)), layout="constrained")
    for row, layer in enumerate(layers):
        for col, head in enumerate(heads):
            im = _heatmap(axes[row, col], selected[layer, head].numpy(), labels,
                          f"Layer {layer + 1}, head {head + 1}")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Attention probability")
    fig.suptitle(title)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    return fig


MODEL_NAME = "prajjwal1/bert-tiny"
TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH = 64
# Must match config/retriever/dpr.yaml, or --weights won't load.
ENCODER_SETTINGS = dict(n_layers=6, n_heads=4, emb_dim=128, feed_forward_dim=768, pool_type="cls")


def main():
    parser = argparse.ArgumentParser(description="Save attention heatmaps for a few texts.")
    parser.add_argument("--backend", choices=("bert", "custom"), default="bert")
    parser.add_argument("--weights", type=Path, help="Saved HFEncoder or TransformerEncoder state_dict")
    parser.add_argument("--texts", nargs="+", default=[
        "How do interest rates affect bond prices?",
        "When interest rates rise, the market price of existing bonds usually falls.",
    ])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention"))
    args = parser.parse_args()

    # Save to files only, no windows (also works on the cluster).
    plt.switch_backend("Agg")

    torch.manual_seed(0)  # same random custom encoder every run
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    batch = tokenizer(args.texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    if args.backend == "custom":
        model = TransformerEncoder(vocab_size=len(tokenizer), **ENCODER_SETTINGS)
        model_description = "Custom encoder"
    else:
        # eager attention is needed to get the attention maps back.
        model = BertModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
        model_description = MODEL_NAME
    if args.weights:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        if args.backend == "bert":
            # Trainer saves HFEncoder, which wraps BertModel under 'model'.
            state = {key.removeprefix("model."): value for key, value in state.items()}
        model.load_state_dict(state)
        model_description += f" ({args.weights.name})"
    maps = extract_attention(model, batch["input_ids"], batch["attention_mask"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(args.texts):
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][index].tolist())
        fig = plot_attention(maps, tokens, batch["attention_mask"], batch_index=index,
                             title=f"{model_description}: {text}",
                             output_path=args.output_dir / f"example_{index + 1}.png")
        plt.close(fig)
    print(f"Saved {len(args.texts)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
