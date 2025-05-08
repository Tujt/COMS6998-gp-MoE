from graphviz import Digraph

dot = Digraph("TinyLlama_MoE_Architecture_Improved", format="png")
dot.attr(rankdir="TB", fontsize="10")

# Color styles
preproc_style = {"style": "filled", "fillcolor": "#d0e6f8"}
model_style = {"style": "filled", "fillcolor": "#d4f8d0"}
moe_style = {"style": "filled", "fillcolor": "#fdf5b4"}
training_style = {"style": "filled", "fillcolor": "#e0e0e0"}
monitor_style = {"style": "filled", "fillcolor": "#f8d0e6"}
loss_style = {"style": "filled", "fillcolor": "#ffd1a9"}

# Input & Preprocessing
dot.node("A1", "AskNews-NER-v0\n(instruction + input + output)", shape="box", **preproc_style)
dot.node("A2", "Tokenizer\n(max_length=1024)", shape="box", **preproc_style)
dot.node("A3", "Token Cache\n(disk-backed)", shape="box", **preproc_style)

# Embedding and Backbone
dot.node("B1", "Embedding Layer\n(position + token)", shape="box", **model_style)
dot.node("B2", "Self-Attention Layers\n(24×, shared)", shape="box", **model_style)
dot.node("B3", "Feedforward Layer:\nMLP or MoE", shape="box", **model_style)

# MoE Detail
with dot.subgraph(name="cluster_moe") as moe:
    moe.attr(label="MoE Feedforward Block", style="dashed")
    moe.node("M1", "Router (Linear)", shape="ellipse", **moe_style)
    moe.node("M2", "Top-1 Routing", shape="diamond", **moe_style)
    moe.node("M3", "Expert 0\n(MLP copy)", shape="box", **moe_style)
    moe.node("M4", "Expert 1\n(MLP copy)", shape="box", **moe_style)
    moe.edge("M1", "M2")
    moe.edge("M2", "M3", label="if selected")
    moe.edge("M2", "M4", label="if selected")

# Output & Loss
dot.node("C1", "LM Head\nLinear → Vocab", shape="box", **loss_style)
dot.node("C2", "CrossEntropy + λ * Aux Loss", shape="box", **loss_style)

# Training Environment
with dot.subgraph(name="cluster_train") as train:
    train.attr(label="Training Environment", style="dotted")
    train.node("T1", "DeepSpeed ZeRO-2\n(bf16, grad_acc=2)", shape="box", **training_style)
    train.node("T2", "Adam Optimizer\nlr=1e-4, betas=[0.9, 0.999]", shape="box", **training_style)
    train.node("T3", "A100 x2 GPUs", shape="box", **training_style)
    train.node("T4", "Checkpointing\nBest-per-epoch", shape="box", **training_style)
    train.edge("T1", "T2")
    train.edge("T2", "T3")
    train.edge("T3", "T4")

# Monitoring & Profiling
with dot.subgraph(name="cluster_monitor") as monitor:
    monitor.attr(label="Monitoring & Profiling", style="dotted")
    monitor.node("L11", "Weights & Biases\n(metric logging)", shape="box", **monitor_style)
    monitor.node("L22", "torch.profiler\n(CUDA+CPU tracing)", shape="box", **monitor_style)
    monitor.edge("L11", "L22")

# Dataset Preprocessing Cluster
with dot.subgraph(name="cluster_data") as data:
    data.attr(label="Dataset Preprocessing", style="dotted")
    data.node("D1", "Concatenate → Tokenize → Cache", shape="box", **preproc_style)

# Main Path
dot.edge("A1", "A2")
dot.edge("A2", "A3")
dot.edge("A3", "B1")
dot.edge("B1", "B2")
dot.edge("B2", "B3")
dot.edge("B3", "C1")
dot.edge("C1", "C2")

# MoE callout link
dot.edge("B3", "M1", style="dashed", label="if MoE enabled")

# # Legend (Vertical)
# with dot.subgraph(name="cluster_legend") as legend:
#     legend.attr(label="Legend", style="dotted", rankdir="TB")
#     legend.node("L_loss", "Output / Loss", style="filled", fillcolor="#ffd1a9", shape="ellipse")
#     legend.node("L_mon", "Monitoring", style="filled", fillcolor="#f8d0e6", shape="ellipse")
#     legend.node("L_trn", "Training Setup", style="filled", fillcolor="#e0e0e0", shape="ellipse")
#     legend.node("L_moe", "MoE Components", style="filled", fillcolor="#fdf5b4", shape="ellipse")
#     legend.node("L_mod", "Model Layers", style="filled", fillcolor="#d4f8d0", shape="ellipse")
#     legend.node("L_pre", "Preprocessing", style="filled", fillcolor="#d0e6f8", shape="ellipse")

# Legend (Vertical)
with dot.subgraph(name="cluster_legend") as legend:
    legend.attr(label="Legend", style="dotted")
    legend.node("L1", "Output / Loss", style="filled", fillcolor="#ffd1a9", shape="ellipse")
    legend.node("L2", "Monitoring", style="filled", fillcolor="#f8d0e6", shape="ellipse")
    legend.node("L3", "Training Setup", style="filled", fillcolor="#e0e0e0", shape="ellipse")
    legend.node("L4", "MoE Components", style="filled", fillcolor="#fdf5b4", shape="ellipse")
    legend.node("L5", "Model Layers", style="filled", fillcolor="#d4f8d0", shape="ellipse")
    legend.node("L6", "Preprocessing", style="filled", fillcolor="#d0e6f8", shape="ellipse")

    # Force vertical stacking by creating invisible edges
    legend.edge("D1", "L11", style="invis")
    legend.edge("L22", "L1", style="invis")
    legend.edge("L1", "L2", style="invis")
    legend.edge("L2", "L3", style="invis")
    legend.edge("L3", "L4", style="invis")
    legend.edge("L4", "L5", style="invis")
    legend.edge("L5", "L6", style="invis")

# Positioning trick: Add invisible edge to force legend below Monitoring
# dot.edge("L2", "L_loss", style="invis")

# Export
dot.render("TinyLlama_MoE_Architecture_Improved", view=False)
print(dot.source)
