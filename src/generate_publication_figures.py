import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300

def ensure_plots_dir(output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

def generate_figure1_workflow(output_dir: str = "plots"):
    """
    Figure 1: High-level System Workflow Diagram of AlphaReq.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    # Draw conceptual blocks
    boxes = [
        {"x": 0.05, "y": 0.5, "w": 0.18, "h": 0.35, "title": "1. Specification Intake\n(P-EXP 47 Projects)", "color": "#E3F2FD"},
        {"x": 0.28, "y": 0.5, "w": 0.18, "h": 0.35, "title": "2. Hamiltonian Conflict\nDetection (J_ij Tensor)", "color": "#FFF3E0"},
        {"x": 0.51, "y": 0.5, "w": 0.18, "h": 0.35, "title": "3. Thermal Relaxation &\nPhase Audit (α vs α_c)", "color": "#F3E5F5"},
        {"x": 0.74, "y": 0.5, "w": 0.21, "h": 0.35, "title": "4. Autonomous 3D\nArchitecture Folding\n(Cohesion Q & Low Coupling)", "color": "#E8F5E9"},
    ]

    from matplotlib.patches import FancyBboxPatch

    for b in boxes:
        rect = FancyBboxPatch((b['x'], b['y']), b['w'], b['h'], facecolor=b['color'], edgecolor='#37474F', linewidth=1.8, boxstyle="round,pad=0.02")
        ax.add_patch(rect)
        ax.text(b['x'] + b['w']/2, b['y'] + b['h']/2, b['title'], ha='center', va='center', weight='bold', color='#263238', linespacing=1.3)

    # Arrows between blocks
    arrows = [(0.23, 0.675, 0.28, 0.675), (0.46, 0.675, 0.51, 0.675), (0.69, 0.675, 0.74, 0.675)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#37474F', edgecolor='#37474F', width=2.5, headwidth=9, shrink=0.05))

    ax.set_title("Figure 1. The AlphaReq End-to-End Architectural Folding Pipeline", weight='bold', pad=25)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure1_alphareq_workflow.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_figure2_phase_transition(df_bench: pd.DataFrame, output_dir: str = "plots"):
    """
    Figure 2: The Phase Transition Boundary across 47 projects in P-EXP.
    """
    plt.figure(figsize=(9, 5.5))

    phases = df_bench['Phase_State'].unique()
    palette = {'Paramagnetic (Low Risk)': '#2E7D32', 'Marginal (Moderate Risk)': '#F57C00', 'Glassy/Frustrated (High Risk)': '#C62828'}

    sns.scatterplot(
        data=df_bench,
        x='N_Requirements',
        y='Constraint_Density_Alpha',
        hue='Phase_State',
        palette=palette,
        s=120,
        edgecolor='black',
        alpha=0.9
    )

    # Draw critical boundary line alpha_c = 1.80
    plt.axhline(y=1.80, color='#C62828', linestyle='--', linewidth=2.0, label=r'Critical Phase Boundary ($\alpha_c = 1.80$)')
    plt.axhline(y=0.80, color='#F57C00', linestyle=':', linewidth=1.5, label=r'Marginal Warning ($\alpha = 0.80$)')

    # Annotate Project 8
    p8 = df_bench[df_bench['Project_ID'] == 8].iloc[0]
    plt.annotate('Project 8 (N=92, α=5.51)\nDeep Spin-Glass State',
                 xy=(p8['N_Requirements'], p8['Constraint_Density_Alpha']),
                 xytext=(p8['N_Requirements'] - 30, p8['Constraint_Density_Alpha'] - 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', color='#B71C1C')

    # Annotate Project 46
    p46 = df_bench[df_bench['Project_ID'] == 46].iloc[0]
    plt.annotate('Project 46 (N=77, α=0.73)\nParamagnetic Phase',
                 xy=(p46['N_Requirements'], p46['Constraint_Density_Alpha']),
                 xytext=(p46['N_Requirements'] - 28, p46['Constraint_Density_Alpha'] + 1.2),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', color='#1B5E20')

    plt.title("Figure 2. Empirical Thermodynamic Phase Transition across 47 P-EXP Projects", weight='bold', pad=15)
    plt.xlabel("Total Requirements per Project ($N$)", weight='bold')
    plt.ylabel(r"Constraint Density ($\alpha = M_{\rm frustrated} / N$)", weight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left')
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure2_phase_transition.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_figure3_architecture_metrics(df_bench: pd.DataFrame, output_dir: str = "plots"):
    """
    Figure 3: Comparative Architectural Quality: Cohesion & Coupling.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    models = ['Ezzini et al. (2022)', 'Al-Safadi et al. (2023)', 'AlphaReq (Ours)']
    
    # Cohesion comparison
    cohesion_means = [df_bench['Ezzini_Cohesion'].mean(), df_bench['AlSafadi_Cohesion'].mean(), df_bench['AlphaReq_Cohesion'].mean()]
    colors = ['#B0BEC5', '#90CAF9', '#2E7D32']

    bars1 = ax1.bar(models, cohesion_means, color=colors, edgecolor='black', width=0.55)
    ax1.set_title("A. Architectural Cohesion ($Q$, Modularity)", weight='bold')
    ax1.set_ylabel("Cohesion Score (Higher is Better)", weight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', weight='bold')

    # Coupling comparison (Lower is better)
    coupling_means = [df_bench['Ezzini_Coupling'].mean(), df_bench['AlSafadi_Coupling'].mean(), df_bench['AlphaReq_Coupling'].mean()]
    colors_c = ['#EF9A9A', '#FFE082', '#66BB6A']

    bars2 = ax2.bar(models, coupling_means, color=colors_c, edgecolor='black', width=0.55)
    ax2.set_title(r"B. Inter-Service Coupling Density ($\mathcal{C}$)", weight='bold')
    ax2.set_ylabel("Coupling Density (Lower is Better)", weight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', weight='bold')

    plt.suptitle("Figure 3. Comparative Architectural Quality Metrics (N=47 Software Projects)", weight='bold', y=1.03)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure3_architecture_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_figure4_csr(df_bench: pd.DataFrame, output_dir: str = "plots"):
    """
    Figure 4: Constraint Satisfaction Rate (CSR) under escalating project size.
    """
    plt.figure(figsize=(9, 5))

    df_sorted = df_bench.sort_values(by='N_Requirements').reset_index(drop=True)

    plt.plot(df_sorted['N_Requirements'], df_sorted['AlphaReq_CSR'], label='AlphaReq (Ours)', color='#2E7D32', marker='o', linewidth=2.5)
    plt.plot(df_sorted['N_Requirements'], df_sorted['AlSafadi_CSR'], label='Al-Safadi et al. (2023)', color='#1976D2', marker='s', linestyle='--', linewidth=2.0)
    plt.plot(df_sorted['N_Requirements'], df_sorted['Ezzini_CSR'], label='Ezzini et al. (2022)', color='#D32F2F', marker='^', linestyle=':', linewidth=2.0)

    plt.title("Figure 4. Non-Functional Constraint Satisfaction Rate (CSR) across Project Scales", weight='bold', pad=15)
    plt.xlabel("Project Scale (Number of Requirements $N$)", weight='bold')
    plt.ylabel("Constraint Satisfaction Rate (CSR %)", weight='bold')
    plt.ylim(0.4, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower left')
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure4_csr_evaluation.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_figure5_case_study(output_dir: str = "plots"):
    """
    Figure 5: Project 8 Architecture Topology Case Study Blueprint.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Core Components
    components = [
        {"x": 0.08, "y": 0.65, "w": 0.22, "h": 0.22, "name": "Edge API Gateway\n& OAuth2 Enclave", "sub": "(Satisfies 15 SE Reqs)", "c": "#FFE0B2"},
        {"x": 0.40, "y": 0.65, "w": 0.22, "h": 0.22, "name": "Distributed Redis\nIn-Memory Cache", "sub": "(Satisfies 17 PE Reqs)", "c": "#C8E6C9"},
        {"x": 0.72, "y": 0.65, "w": 0.22, "h": 0.22, "name": "HA Elastic Cluster\nLoad Balancer", "sub": "(Satisfies 4 SC / 5 A Reqs)", "c": "#BBDEFB"},
        
        {"x": 0.15, "y": 0.20, "w": 0.28, "h": 0.25, "name": "Core Service Alpha\n(Event Scheduling & Matrix)", "sub": "(Functional Module 1)", "c": "#EDE7F6"},
        {"x": 0.55, "y": 0.20, "w": 0.28, "h": 0.25, "name": "Core Service Beta\n(Real-Time Graphical Sync)", "sub": "(Functional Module 2)", "c": "#EDE7F6"},
    ]

    from matplotlib.patches import FancyBboxPatch

    for comp in components:
        rect = FancyBboxPatch((comp['x'], comp['y']), comp['w'], comp['h'], facecolor=comp['c'], edgecolor='#263238', linewidth=1.8, boxstyle="round,pad=0.02")
        ax.add_patch(rect)
        ax.text(comp['x'] + comp['w']/2, comp['y'] + comp['h']*0.60, comp['name'], ha='center', va='center', weight='bold', color='#212121')
        ax.text(comp['x'] + comp['w']/2, comp['y'] + comp['h']*0.25, comp['sub'], ha='center', va='center', style='italic', fontsize=9, color='#424242')

    # Draw Inter-Service Communication Buses
    ax.annotate('', xy=(0.51, 0.65), xytext=(0.30, 0.65), arrowprops=dict(facecolor='#455A64', width=2, headwidth=7))
    ax.annotate('', xy=(0.72, 0.65), xytext=(0.62, 0.65), arrowprops=dict(facecolor='#455A64', width=2, headwidth=7))
    ax.annotate('', xy=(0.29, 0.45), xytext=(0.19, 0.65), arrowprops=dict(facecolor='#455A64', width=2, headwidth=7))
    ax.annotate('', xy=(0.69, 0.45), xytext=(0.51, 0.65), arrowprops=dict(facecolor='#455A64', width=2, headwidth=7))
    ax.annotate('', xy=(0.55, 0.32), xytext=(0.43, 0.32), arrowprops=dict(facecolor='#455A64', width=2, headwidth=7))

    ax.set_title("Figure 5. Synthesized Architectural Blueprint for Project 8 (92 Requirements, α = 5.51)", weight='bold', pad=25)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure5_case_study_blueprint.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_figure6_conflict_heatmap(output_dir: str = "plots"):
    """
    Figure 6: 12x12 Inter-Requirement Conflict Tension and Cooperation Heatmap.
    Visualizes pairwise architectural trade-offs (antiferromagnetic penalties)
    and synergistic alignments (ferromagnetic bonuses).
    """
    classes = ['A', 'F', 'FT', 'L', 'LF', 'MN', 'O', 'PE', 'PO', 'SC', 'SE', 'US']
    n = len(classes)
    matrix = np.zeros((n, n))

    # Import default coupling values from ConflictDetector
    try:
        from .conflict_detector import ConflictDetector
    except ImportError:
        from conflict_detector import ConflictDetector

    detector = ConflictDetector()
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                pair = tuple(sorted([classes[i], classes[j]]))
                if pair in detector.penalties:
                    matrix[i, j] = detector.penalties[pair]
                elif pair in detector.bonuses:
                    matrix[i, j] = detector.bonuses[pair]
                else:
                    matrix[i, j] = 0.10 if (classes[i] == 'F' or classes[j] == 'F') else 0.05

    plt.figure(figsize=(9.5, 7.5))
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(
        matrix,
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0.0,
        cbar_kws={'label': 'Coupling Coefficient J (Negative = Conflict, Positive = Synergy)'},
        linewidths=0.8,
        linecolor='white'
    )

    plt.title("Figure 6. Pairwise Requirements Interaction Tensor (J_ij) Across 12 Quality Dimensions", weight='bold', pad=15)
    plt.xlabel("Target Requirement Class", weight='bold')
    plt.ylabel("Source Requirement Class", weight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure6_conflict_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_figure7_cross_validation_distribution(df_cv: pd.DataFrame, output_dir: str = "plots"):
    """
    Figure 7: 10-Fold 30-Randomized-Split (300 Evaluations) Empirical Distribution Boxplots.
    Illustrates empirical stability and tight confidence intervals across folds for AlphaReq vs Baselines.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))

    # Metrics to plot
    metric_configs = [
        ('A. Constraint Satisfaction Rate (CSR %)', ['Ezzini_CSR', 'AlSafadi_CSR', 'AlphaReq_CSR'], (axes[0, 0]), True),
        ('B. Inter-Service Coupling Density (Lower is Better)', ['Ezzini_Coupling', 'AlSafadi_Coupling', 'AlphaReq_Coupling'], (axes[0, 1]), False),
        ('C. Architectural Cohesion (Q, Modularity)', ['Ezzini_Cohesion', 'AlSafadi_Cohesion', 'AlphaReq_Cohesion'], (axes[1, 0]), False),
        ('D. Composite Architecture Fitness Score', ['Ezzini_Fitness', 'AlSafadi_Fitness', 'AlphaReq_Fitness'], (axes[1, 1]), False),
    ]

    model_labels = ['Ezzini (2022)', 'Al-Safadi (2023)', 'AlphaReq (Ours)']
    colors = ['#EF9A9A', '#90CAF9', '#66BB6A']

    for title, cols, ax, as_pct in metric_configs:
        data_to_plot = []
        for c in cols:
            vals = df_cv[c].values
            if as_pct:
                vals = vals * 100.0
            data_to_plot.append(vals)

        bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=model_labels, widths=0.55, showmeans=True,
                        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='gold'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('#263238')
            patch.set_linewidth(1.4)

        ax.set_title(title, weight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.45, axis='y')
        if as_pct:
            ax.set_ylabel("CSR (%)", weight='bold')
        else:
            ax.set_ylabel("Metric Score", weight='bold')

    plt.suptitle("Figure 7. 10-Fold Cross-Validation with 30 Randomized Splits (N=300 Evaluations)", weight='bold', y=0.99)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure7_cross_validation_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")

def generate_all_figures(df_bench: pd.DataFrame, output_dir: str = "plots", df_cv: Optional[pd.DataFrame] = None):
    ensure_plots_dir(output_dir)
    generate_figure1_workflow(output_dir)
    generate_figure2_phase_transition(df_bench, output_dir)
    generate_figure3_architecture_metrics(df_bench, output_dir)
    generate_figure4_csr(df_bench, output_dir)
    generate_figure5_case_study(output_dir)
    generate_figure6_conflict_heatmap(output_dir)
    if df_cv is not None and not df_cv.empty:
        generate_figure7_cross_validation_distribution(df_cv, output_dir)
    print("\n[+] Publication-Grade Figures Successfully Generated!")
