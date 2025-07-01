# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader
import torch
import torch.nn as nn

# === GNN imports ===
from models.gnn_extractor import TemporalGCN, build_correlation_graph
from diversify.utils.params import gnn_params

# === SHAP imports ===
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shap_custom.shap_utils import _get_shap_array
from shap_custom import (
    get_shap_explainer,
    compute_shap_values,
    plot_summary,
    plot_force,
    evaluate_shap_impact,
    plot_shap_heatmap,
    get_background_batch,
    compute_jaccard_topk,
    compute_kendall_tau,
    cosine_similarity_shap,
    log_shap_numpy,
    overlay_signal_with_shap,
    compute_flip_rate,
    compute_confidence_change,
    compute_aopc,
    compute_feature_coherence,
    compute_shap_entropy,
    plot_emg_shap_4d,
    compute_shap_channel_variance,
    compute_shap_temporal_entropy,
    compare_top_k_channels,
    compute_mutual_info,
    compute_pca_alignment,
    plot_4d_shap_surface
)
import plotly.io as pio
pio.renderers.default = 'colab'  # Or use 'notebook' if you're not on Colab

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)
    if args.latent_domain_num < 6:
        args.batch_size = 32*args.latent_domain_num
    else:
        args.batch_size = 16*args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    best_valid_acc, target_acc = 0, 0
    
    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'total_loss', 'train_acc', 'valid_acc', 'target_acc', 'total_cost_time']}

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()

    # ===== GNN feature extractor integration =====
    use_gnn = getattr(args, "use_gnn", 0)
    gnn = None
    if use_gnn:
        # Assumes data shape is [batch, channels, timesteps]
        example_batch = next(iter(train_loader))[0] if hasattr(train_loader, '__iter__') else None
        in_channels = example_batch.shape[1] if example_batch is not None else 8
        gnn = TemporalGCN(
            in_channels=in_channels,
            hidden_dim=gnn_params["gcn_hidden_dim"],
            num_layers=gnn_params["gcn_num_layers"],
            lstm_hidden=gnn_params["lstm_hidden"],
            output_dim=gnn_params["feature_output_dim"]
        ).cuda()
        # >>>>> KEY CHANGE: Overwrite featurizer with identity for GNN <<<<<
        algorithm.featurizer = nn.Identity()
        print('[INFO] GNN feature extractor initialized. CNN featurizer is bypassed.')
        # >>>>> NEW: Patch bottleneck(s) for GNN feature size <<<<<
        gnn_out_dim = gnn.out.out_features
        if hasattr(algorithm, "bottleneck"):
            algorithm.bottleneck = nn.Linear(gnn_out_dim, 256).cuda()
            print(f"[INFO] Bottleneck adjusted for GNN: {gnn_out_dim} -> 256")
        if hasattr(algorithm, "abottleneck"):
            algorithm.abottleneck = nn.Linear(gnn_out_dim, 256).cuda()
            print(f"[INFO] Adversarial bottleneck adjusted for GNN: {gnn_out_dim} -> 256")
        if hasattr(algorithm, "dbottleneck"):
            algorithm.dbottleneck = nn.Linear(gnn_out_dim, 256).cuda()
            print(f"[INFO] Domain bottleneck (dbottleneck) adjusted for GNN: {gnn_out_dim} -> 256")
        # === NEW for GNN in set_dlabel (do NOT remove these lines) ===
        algorithm.gnn_extractor = gnn
        algorithm.use_gnn = True

    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')

    for round in range(args.max_epoch):
        print(f'\n========ROUND {round}========')
        print('====Feature update====')
        loss_list = ['class']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                # === GNN: extract features if enabled ===
                if use_gnn and gnn is not None:
                    batch_x = data[0] if isinstance(data, (list, tuple)) else data
                    if len(batch_x.shape) == 4 and batch_x.shape[2] == 1:
                        batch_x = batch_x.squeeze(2)
                    gnn_graphs = build_correlation_graph(batch_x.cuda())
                    from torch_geometric.loader import DataLoader as GeoDataLoader
                    geo_loader = GeoDataLoader(gnn_graphs, batch_size=len(gnn_graphs))
                    for graph_batch in geo_loader:
                        graph_batch = graph_batch.cuda()
                        gnn_features = gnn(graph_batch)
                    # >>>>> Only pass GNN features and label(s) forward <<<<<
                    if isinstance(data, (list, tuple)) and len(data) > 1:
                        data = (gnn_features, *data[1:])
                    else:
                        data = gnn_features
                # === END GNN block ===

                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        print('====Latent domain characterization====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                # === GNN: extract features if enabled ===
                if use_gnn and gnn is not None:
                    batch_x = data[0] if isinstance(data, (list, tuple)) else data
                    if len(batch_x.shape) == 4 and batch_x.shape[2] == 1:
                        batch_x = batch_x.squeeze(2)
                    gnn_graphs = build_correlation_graph(batch_x.cuda())
                    from torch_geometric.loader import DataLoader as GeoDataLoader
                    geo_loader = GeoDataLoader(gnn_graphs, batch_size=len(gnn_graphs))
                    for graph_batch in geo_loader:
                        graph_batch = graph_batch.cuda()
                        gnn_features = gnn(graph_batch)
                    if isinstance(data, (list, tuple)) and len(data) > 1:
                        data = (gnn_features, *data[1:])
                    else:
                        data = gnn_features
                # === END GNN block ===

                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step]+[loss_result_dict[item] for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('====Domain-invariant feature learning====')

        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item+'_loss' for item in loss_list])
        print_key.extend([item+'_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)

        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                # === GNN: extract features if enabled ===
                if use_gnn and gnn is not None:
                    batch_x = data[0] if isinstance(data, (list, tuple)) else data
                    if len(batch_x.shape) == 4 and batch_x.shape[2] == 1:
                        batch_x = batch_x.squeeze(2)
                    gnn_graphs = build_correlation_graph(batch_x.cuda())
                    from torch_geometric.loader import DataLoader as GeoDataLoader
                    geo_loader = GeoDataLoader(gnn_graphs, batch_size=len(gnn_graphs))
                    for graph_batch in geo_loader:
                        graph_batch = graph_batch.cuda()
                        gnn_features = gnn(graph_batch)
                    if isinstance(data, (list, tuple)) and len(data) > 1:
                        data = (gnn_features, *data[1:])
                    else:
                        data = gnn_features
                # === END GNN block ===

                step_vals = algorithm.update(data, opt)

            results = {
                'epoch': step,
            }

            results['train_acc'] = modelopera.accuracy(
                algorithm, train_loader_noshuffle, None)

            acc = modelopera.accuracy(algorithm, valid_loader, None)
            results['valid_acc'] = acc

            acc = modelopera.accuracy(algorithm, target_loader, None)
            results['target_acc'] = acc

            for key in loss_list:
                results[key+'_loss'] = step_vals[key]
            for key in logs:
                logs[key].append(results.get(key, 0))
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            results['total_cost_time'] = time.time()-sss
            print_row([results[key] for key in print_key], colwidth=15)

    print(f'Target acc: {target_acc:.4f}')

    if args.enable_shap:
        print("\n📊 Running SHAP explainability...")
        background = get_background_batch(valid_loader, size=64).to('cuda')
        X_eval = background[:10]
        shap_explainer = get_shap_explainer(algorithm, background)
        shap_vals = compute_shap_values(shap_explainer, X_eval)
        shap_array = _get_shap_array(shap_vals)

        plot_summary(shap_vals, X_eval.cpu().numpy())
        plot_force(shap_explainer, shap_vals, X_eval.cpu().numpy())
        overlay_signal_with_shap(X_eval[0].cpu().numpy(), shap_array[0], output_path="shap_overlay_sample0.png")

        base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals)
        log_shap_numpy(shap_vals)

        print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
        print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
        print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
        print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals, evaluate_shap_impact):.4f}")
        print(f"[SHAP] Entropy: {compute_shap_entropy(shap_array):.4f}")
        print(f"[SHAP] Coherence: {compute_feature_coherence(shap_array):.4f}")

        if len(shap_array) > 1:
            print(f"[SHAP] Jaccard: {compute_jaccard_topk(shap_array[0], shap_array[1]):.4f}")
            print(f"[SHAP] Kendall’s Tau: {compute_kendall_tau(shap_array[0], shap_array[1]):.4f}")
            print(f"[SHAP] Cosine Sim: {cosine_similarity_shap(shap_array[0], shap_array[1]):.4f}")

        # 🔬 4D-specific visualization and metrics
        # ---- Plot EMG SHAP 4D ----
        from shap_custom.shap4D import plot_4d_shap_surface

        try:
            plot_emg_shap_4d(X_eval, shap_array)
        except Exception as e:
            print(f"[WARNING] plot_emg_shap_4d failed to render: {e}")

        plot_4d_shap_surface(shap_vals, output_path="shap_4d_surface.html")
        
        # Reshape SHAP array from (samples, 1, time, aux) to (samples, channels, time)
        shap_array_reshaped = shap_array.reshape(shap_array.shape[0], -1, shap_array.shape[2])

        print(f"[SHAP4D] Channel Variance: {compute_shap_channel_variance(shap_array):.4f}")
        print(f"[SHAP4D] Temporal Entropy: {compute_shap_temporal_entropy(shap_array_reshaped):.4f}")
        signal_sample = X_eval[0].cpu().numpy()                     # shape: (8,1,200)
        shap_sample = shap_array[0].mean(axis=-1)                   # reduce (8,1,200,6) → (8,1,200)
        print(f"[SHAP4D] Mutual Info: {compute_mutual_info(signal_sample, shap_sample):.4f}")
        shap_array_reduced = shap_array.mean(axis=-1)
        print(f"[SHAP4D] PCA Alignment: {compute_pca_alignment(shap_array_reduced):.4f}")
        true_labels, pred_labels = [], []
        for data in valid_loader:
            x, y = data[0].cuda(), data[1]
            preds = algorithm.predict(x).cpu()
            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())

        cm = confusion_matrix(true_labels, pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Validation Set)")
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.show()

        plot_shap_heatmap(shap_array, output_path="shap_temporal_heatmap.png")

        print("\n📊 Training baseline model for SHAP comparison...")
        baseline_model = algorithm_class(args).cuda()
        baseline_model.eval()
        baseline_shap_explainer = get_shap_explainer(baseline_model, background)
        baseline_shap_vals = compute_shap_values(baseline_shap_explainer, X_eval)
        baseline_shap_array = _get_shap_array(baseline_shap_vals)
        plot_shap_heatmap(baseline_shap_array, output_path="shap_heatmap_baseline.png")

        print("\n🔍 Running ablation: shuffling SHAP-important segments...")
        
        X_ablation = X_eval.clone()
        
        # Step 1: Extract SHAP sample
        shap_sample = shap_array[0]  # (1, 200, 6)
        
        # Step 2: Reduce to 1D SHAP importance per time step
        shap_scores = torch.from_numpy(np.abs(shap_sample).mean(axis=(0, 2)))  # shape: (200,)
        
        # Step 3: Safe top-k selection
        topk = min(100, shap_scores.numel())
        if topk == 0:
            print("[SKIP] SHAP ablation: not enough time steps for top-k selection.")
        else:
            shap_mask = shap_scores.topk(topk, largest=True).indices
        
            # Step 4: Apply time-step shuffle for selected top-k
            original = X_ablation[0, :, :, shap_mask].clone()
            perm = shap_mask[torch.randperm(len(shap_mask))]
            X_ablation[0, :, :, shap_mask] = X_ablation[0, :, :, perm]
        
            # Step 5: Evaluate accuracy after ablation
            post_preds = algorithm.predict(X_ablation)
            post_labels = np.argmax(post_preds.detach().cpu().numpy(), axis=1)
            original_labels = np.argmax(base_preds, axis=1)
        
            print(f"[Ablation] Accuracy post SHAP shuffle: {(post_labels == original_labels).mean():.4f}")

            # Compute confidence scores (max softmax probability)
            base_conf = base_preds.max(axis=1)   # shape: (N,)
            post_conf = post_preds.detach().cpu().numpy().max(axis=1)
            
            # 📊 Plot comparison
            plt.figure(figsize=(10, 5))
            plt.plot(base_conf, label='Original Confidence', marker='o')
            plt.plot(post_conf, label='Post-Ablation Confidence', marker='x')
            plt.xlabel("Sample Index")
            plt.ylabel("Confidence (Max Softmax Prob)")
            plt.title("Confidence Before vs After SHAP-Ablation")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("confidence_comparison_ablation.png", dpi=300)
            plt.show()
            # Histogram: Confidence distributions
            plt.figure(figsize=(10, 5))
            plt.hist(base_conf, bins=20, alpha=0.6, label="Original Confidence", color="blue", density=True)
            plt.hist(post_conf, bins=20, alpha=0.6, label="Post-Ablation Confidence", color="red", density=True)
            plt.title("Histogram: Confidence Distribution Before vs After SHAP Ablation")
            plt.xlabel("Confidence (Max Softmax Prob)")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig("histogram_confidence_comparison.png", dpi=300)
            plt.show()
            from scipy.stats import entropy as kl_divergence
            
            # Normalize histograms
            hist_base, bins = np.histogram(base_conf, bins=20, range=(0, 1), density=True)
            hist_post, _ = np.histogram(post_conf, bins=bins, density=True)
            
            # Avoid zero division
            hist_base += 1e-10
            hist_post += 1e-10
            
            kl_score = kl_divergence(hist_base, hist_post)
            print(f"[SHAP Ablation] KL Divergence (Original vs Post-Ablation): {kl_score:.4f}")
            # Boxplot comparison
            plt.figure(figsize=(8, 5))
            plt.boxplot([base_conf, post_conf], labels=["Original", "Post-Ablation"])
            plt.title("Boxplot: Confidence Before vs After SHAP Ablation")
            plt.ylabel("Confidence (Max Softmax Probability)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("boxplot_confidence_comparison.png", dpi=300)
            plt.show()

            from collections import defaultdict
            
            # Convert predictions
            true_labels_np = np.array(true_labels)
            pred_labels_np = np.array(pred_labels)
            
            # Get confidence vectors
            base_conf_vec = base_preds.max(axis=1)
            post_conf_vec = post_preds.detach().cpu().numpy().max(axis=1)
            
            # Group confidence drops by class
            drop_by_class = defaultdict(list)
            for i, label in enumerate(true_labels_np[:len(base_conf_vec)]):  # use true label
                drop = base_conf_vec[i] - post_conf_vec[i]
                drop_by_class[label].append(drop)
            
            # Compute mean drop per class
            class_ids = sorted(drop_by_class.keys())
            mean_drops = [np.mean(drop_by_class[c]) for c in class_ids]
            
            # 📊 Bar Plot
            plt.figure(figsize=(10, 5))
            plt.bar(class_ids, mean_drops, color="purple")
            plt.xlabel("Class Label")
            plt.ylabel("Average Confidence Drop")
            plt.title("Confidence Drop per Class (Post SHAP Ablation)")
            plt.xticks(class_ids)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("confidence_drop_per_class.png", dpi=300)
            plt.show()
            
            from scipy.stats import pearsonr
            
            # Use sample-wise SHAP magnitude & confidence
            shap_strength = shap_array.reshape(len(shap_array), -1).mean(axis=1)
            conf_strength = base_preds.max(axis=1)
            
            # Pearson correlation
            corr, pval = pearsonr(shap_strength, conf_strength)
            print(f"[SHAP vs Confidence] Pearson Correlation: {corr:.4f} (p={pval:.4g})")
            
            # Optional scatter plot
            plt.figure(figsize=(6, 5))
            plt.scatter(shap_strength, conf_strength, c='teal', alpha=0.7)
            plt.xlabel("Mean SHAP Magnitude")
            plt.ylabel("Model Confidence")
            plt.title(f"SHAP vs Confidence (r={corr:.2f})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("shap_vs_confidence_correlation.png", dpi=300)
            plt.show()




        print("\n🛠 Real-world Context: EMG classification can support gesture-based interfaces in prosthetics or rehabilitation systems, and insights from SHAP improve trust in deployed models.")

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(logs['epoch'], logs['class_loss'], label="Class Loss", marker='o')
        plt.plot(logs['epoch'], logs['dis_loss'], label="Dis Loss", marker='x')
        plt.plot(logs['epoch'], logs['total_loss'], label="Total Loss", linestyle='--')
        plt.title("Losses over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(logs['epoch'], logs['train_acc'], label="Train Accuracy", marker='o')
        plt.plot(logs['epoch'], logs['valid_acc'], label="Valid Accuracy", marker='x')
        plt.plot(logs['epoch'], logs['target_acc'], label="Target Accuracy", linestyle='--')
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("training_metrics_plot.png", dpi=300)
        plt.show()


if __name__ == '__main__':
    args = get_args()
    main(args)
