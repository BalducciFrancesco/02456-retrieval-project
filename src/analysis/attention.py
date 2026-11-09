def _heatmap(ax, m, tokens, title):
    ax.imshow(m, cmap="viridis", vmin=0, vmax=m.max())
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=6)
    ax.set_title(title, fontsize=8)
