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
# Unified SHAP utilities import
from shap_utils import (
    get_background_batch, safe_compute_shap_values, plot_summary,
    overlay_signal_with_shap, plot_shap_heatmap,
    evaluate_shap_impact, compute_flip_rate, compute_jaccard_topk,
    compute_kendall_tau,
    cosine_similarity_shap, save_shap_numpy, 
    compute_confidence_change, _get_shap_array, 
    compute_aopc, compute_feature_coherence, compute_shap_entropy,
    plot_emg_shap_4d, plot_4d_shap_surface, evaluate_advanced_shap_metrics
)

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
    
    # Training metrics logging
    logs = {k: [] for k in ['epoch', 'class_loss', 'dis_loss', 'ent_loss', 
                           'total_loss', 'train_acc', 'valid_acc', 'target_acc', 
                           'total_cost_time']}

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

            # Log losses
            for key in loss_list:
                results[f"{key}_loss"] = step_vals[key]
                logs[f"{key}_loss"].append(step_vals[key])
            
            # Log metrics
            for metric in ['train_acc', 'valid_acc', 'target_acc']:
                logs[metric].append(results[metric])
                
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
            results['total_cost_time'] = time.time()-sss
            print_row([results[key] for key in print_key], colwidth=15)

    print(f'Target acc: {target_acc:.4f}')

    # SHAP explainability analysis
    if getattr(args, 'enable_shap', False):
        print("\n📊 Running SHAP explainability...")
        try:
            # Prepare background and evaluation data
            background = get_background_batch(valid_loader, size=64).cuda()
            X_eval = background[:10]
            
            # Disable inplace operations in the model
            disable_inplace_relu(algorithm)
            
            # Compute SHAP values safely
            shap_vals = safe_compute_shap_values(algorithm, background, X_eval)
            
            # Convert to numpy safely before visualization
            X_eval_np = X_eval.detach().cpu().numpy()
            
            # Generate core visualizations
            plot_summary(shap_vals, X_eval_np, 
                         output_path=os.path.join(args.output, "shap_summary.png"))
            
            overlay_signal_with_shap(X_eval_np[0], shap_vals, 
                                    output_path=os.path.join(args.output, "shap_overlay.png"))
            
            plot_shap_heatmap(shap_vals, 
                             output_path=os.path.join(args.output, "shap_heatmap.png"))

            # Evaluate SHAP impact
            base_preds, masked_preds, acc_drop = evaluate_shap_impact(algorithm, X_eval, shap_vals)
            
            # Save SHAP values
            save_path = os.path.join(args.output, "shap_values.npy")
            save_shap_numpy(shap_vals, save_path=save_path)
            
            # Compute impact metrics
            print(f"[SHAP] Accuracy Drop: {acc_drop:.4f}")
            print(f"[SHAP] Flip Rate: {compute_flip_rate(base_preds, masked_preds):.4f}")
            print(f"[SHAP] Confidence Δ: {compute_confidence_change(base_preds, masked_preds):.4f}")
            print(f"[SHAP] AOPC: {compute_aopc(algorithm, X_eval, shap_vals):.4f}")

            # Compute advanced metrics
            metrics = evaluate_advanced_shap_metrics(shap_vals, X_eval)
            print(f"[SHAP] Entropy: {metrics.get('shap_entropy', 0):.4f}")
            print(f"[SHAP] Coherence: {metrics.get('feature_coherence', 0):.4f}")
            print(f"[SHAP] Channel Variance: {metrics.get('channel_variance', 0):.4f}")
            print(f"[SHAP] Temporal Entropy: {metrics.get('temporal_entropy', 0):.4f}")
            print(f"[SHAP] Mutual Info: {metrics.get('mutual_info', 0):.4f}")
            print(f"[SHAP] PCA Alignment: {metrics.get('pca_alignment', 0):.4f}")
            
            # Compute similarity metrics between first two samples
            shap_array = _get_shap_array(shap_vals)
            if len(shap_array) >= 2:
                # Extract SHAP values for first two samples
                sample1 = shap_array[0]
                sample2 = shap_array[1]
                
                print(f"[SHAP] Jaccard (top-10): {compute_jaccard_topk(sample1, sample2, k=10):.4f}")
                print(f"[SHAP] Kendall's Tau: {compute_kendall_tau(sample1, sample2):.4f}")
                print(f"[SHAP] Cosine Similarity: {cosine_similarity_shap(sample1, sample2):.4f}")
            else:
                print("[SHAP] Not enough samples for similarity metrics")
            
            # Generate 4D visualizations
            plot_emg_shap_4d(X_eval, shap_vals, 
                             output_path=os.path.join(args.output, "shap_4d_scatter.html"))
            
            plot_4d_shap_surface(shap_vals, 
                                output_path=os.path.join(args.output, "shap_4d_surface.html"))

            # Confusion matrix
            true_labels, pred_labels = [], []
            for data in valid_loader:
                x, y = data[0].cuda(), data[1]
                with torch.no_grad():
                    preds = algorithm.predict(x).cpu()
                true_labels.extend(y.cpu().numpy())
                pred_labels.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())

            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix (Validation Set)")
            plt.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=300)
            plt.close()
            
            print("✅ SHAP analysis completed successfully")
            
        except Exception as e:
            print(f"[ERROR] SHAP analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()

    # Plot training metrics
    try:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        epochs = list(range(len(logs['class_loss'])))
        plt.plot(epochs, logs['class_loss'], label="Class Loss", marker='o')
        plt.plot(epochs, logs['dis_loss'], label="Dis Loss", marker='x')
        plt.plot(epochs, logs['total_loss'], label="Total Loss", linestyle='--')
        plt.title("Losses over Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        epochs = list(range(len(logs['train_acc'])))
        plt.plot(epochs, logs['train_acc'], label="Train Accuracy", marker='o')
        plt.plot(epochs, logs['valid_acc'], label="Valid Accuracy", marker='x')
        plt.plot(epochs, logs['target_acc'], label="Target Accuracy", linestyle='--')
        plt.title("Accuracy over Training Steps")
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "training_metrics.png"), dpi=300)
        plt.close()
        print("✅ Training metrics plot saved")
    except Exception as e:
        print(f"[WARNING] Failed to generate training plots: {str(e)}")

if __name__ == '__main__':
    args = get_args()
    main(args)
