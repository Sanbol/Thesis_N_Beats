"""Generate experimental pipeline flowchart for thesis methodology section."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

# Colors
dataset_color = '#4472C4'      # Blue for datasets
train_color = '#548235'        # Green for training phases
eval_color = '#BF8F00'         # Gold for evaluation
model_color = '#7030A0'        # Purple for model state
scenario_bg = {
    'scratch': '#F2F2F2',
    'tl': '#E8F0FE',
    'zs': '#FFF2E8',
}

def draw_box(x, y, w, h, text, color, fontsize=10, fontweight='normal', textcolor='white', alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", 
                          facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=textcolor, wrap=True)

def draw_arrow(x1, y1, x2, y2, color='#555555', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_label(x, y, text, fontsize=9, color='#333333', style='normal'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=color, fontstyle=style)

# Title
ax.text(6, 6.7, 'Experimental Pipeline: Three Evaluation Scenarios', 
        ha='center', va='center', fontsize=14, fontweight='bold', color='#1a1a1a')

# Scenario labels
# Scenario A
ax.text(0.3, 5.55, 'Scenario A', ha='left', va='center', fontsize=11, fontweight='bold', color='#333333')
ax.text(0.3, 5.2, 'From Scratch', ha='left', va='center', fontsize=9, color='#666666', fontstyle='italic')

# Scenario B  
ax.text(0.3, 3.55, 'Scenario B', ha='left', va='center', fontsize=11, fontweight='bold', color='#333333')
ax.text(0.3, 3.2, 'Fine-tuned TL', ha='left', va='center', fontsize=9, color='#666666', fontstyle='italic')

# Scenario C
ax.text(0.3, 1.55, 'Scenario C', ha='left', va='center', fontsize=11, fontweight='bold', color='#333333')
ax.text(0.3, 1.2, 'Zero-shot TL', ha='left', va='center', fontsize=9, color='#666666', fontstyle='italic')

# Scenario A: From Scratch
# Background
bg_a = FancyBboxPatch((2.0, 4.8), 9.5, 1.2, boxstyle="round,pad=0.15",
                       facecolor=scenario_bg['scratch'], edgecolor='#CCCCCC', linewidth=0.8)
ax.add_patch(bg_a)

draw_box(2.3, 5.05, 1.8, 0.7, 'Random\nInit', '#888888', fontsize=9)
draw_arrow(4.1, 5.4, 4.8, 5.4)

draw_box(4.8, 5.05, 2.2, 0.7, 'Train on\nM3 Monthly', train_color, fontsize=9, fontweight='bold')
draw_label(5.9, 4.85, '10 epochs, lr=1e-3', fontsize=7.5, color='#548235', style='italic')
draw_arrow(7.0, 5.4, 7.7, 5.4)

draw_box(7.7, 5.05, 2.0, 0.7, 'Evaluate on\nM3 Test', eval_color, fontsize=9, fontweight='bold')
draw_arrow(9.7, 5.4, 10.3, 5.4)

draw_box(10.3, 5.15, 1.0, 0.5, 'Metrics', '#C00000', fontsize=8, fontweight='bold')

# Scenario B: Fine-tuned TL
bg_b = FancyBboxPatch((2.0, 2.8), 9.5, 1.2, boxstyle="round,pad=0.15",
                       facecolor=scenario_bg['tl'], edgecolor='#CCCCCC', linewidth=0.8)
ax.add_patch(bg_b)

draw_box(2.3, 3.05, 1.8, 0.7, 'Pre-train on\nM4 Monthly', dataset_color, fontsize=9, fontweight='bold')
draw_label(3.2, 2.85, '10 epochs, lr=1e-3', fontsize=7.5, color='#4472C4', style='italic')
draw_arrow(4.1, 3.4, 4.8, 3.4)

draw_box(4.8, 3.05, 2.2, 0.7, 'Fine-tune on\nM3 Monthly', train_color, fontsize=9, fontweight='bold')
draw_label(5.9, 2.85, '15 epochs, lr=1e-5', fontsize=7.5, color='#548235', style='italic')
draw_arrow(7.0, 3.4, 7.7, 3.4)

draw_box(7.7, 3.05, 2.0, 0.7, 'Evaluate on\nM3 Test', eval_color, fontsize=9, fontweight='bold')
draw_arrow(9.7, 3.4, 10.3, 3.4)

draw_box(10.3, 3.15, 1.0, 0.5, 'Metrics', '#C00000', fontsize=8, fontweight='bold')

# Scenario C: Zero-shot TL
bg_c = FancyBboxPatch((2.0, 0.8), 9.5, 1.2, boxstyle="round,pad=0.15",
                       facecolor=scenario_bg['zs'], edgecolor='#CCCCCC', linewidth=0.8)
ax.add_patch(bg_c)

draw_box(2.3, 1.05, 1.8, 0.7, 'Pre-train on\nM4 Monthly', dataset_color, fontsize=9, fontweight='bold')
draw_label(3.2, 0.85, '10 epochs, lr=1e-3', fontsize=7.5, color='#4472C4', style='italic')
draw_arrow(4.1, 1.4, 4.8, 1.4)

draw_box(4.8, 1.05, 2.2, 0.7, 'No M3\nTraining', '#999999', fontsize=9, fontweight='bold', textcolor='white')
draw_label(5.9, 0.85, 'skip fine-tuning', fontsize=7.5, color='#666666', style='italic')
draw_arrow(7.0, 1.4, 7.7, 1.4)

draw_box(7.7, 1.05, 2.0, 0.7, 'Evaluate on\nM3 Test', eval_color, fontsize=9, fontweight='bold')
draw_arrow(9.7, 1.4, 10.3, 1.4)

draw_box(10.3, 1.15, 1.0, 0.5, 'Metrics', '#C00000', fontsize=8, fontweight='bold')

# Legend
ax.text(6, 0.3, 'Each scenario is run for 2 architectures × 2 variants (N-BEATS-S, N-HiTS-S: Standard and Stabilized) × 3 seeds',
        ha='center', va='center', fontsize=8.5, color='#555555', fontstyle='italic')

# Metrics note
ax.text(6, 0.05, 'Metrics: sMAPE, RMSSE (accuracy)  |  sMAPC, RMSSC (stability)',
        ha='center', va='center', fontsize=8, color='#888888')

plt.tight_layout()
out = 'thesis_figures_v2/pipeline_diagram.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved to {out}')
plt.close()
