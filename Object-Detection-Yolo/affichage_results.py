import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

class AffichageResults:
    def __init__(self, results_csv, best_model_path):
        self.results_csv = results_csv
        self.best_model_path = best_model_path
        self.results_df = pd.read_csv(results_csv)
        self.model = YOLO(best_model_path)

    def plot_metrics(self):
        metrics = [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)"
        ]

        os.makedirs("outputs", exist_ok=True)

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(self.results_df["epoch"], self.results_df[metric], label=metric)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"{metric} over epochs")
            plt.legend()
            plt.grid()
            plt.savefig(f"outputs/{metric}_curve.png")
            plt.close()

    def plot_precision_recall_curves(self):
        val_metrics = self.model.val()
        val_metrics.plot(save_dir="outputs")

    def plot_confusion_matrix(self):
        confusion_matrix = self.model.val().confusion_matrix

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("outputs/confusion_matrix.png")
        plt.close()

    def generate_label_correlogram(self):
        labels = self.model.val().labels

        plt.figure(figsize=(10, 8))
        sns.heatmap(labels.corr(), annot=True, cmap="coolwarm")
        plt.title("Labels Correlogram")
        plt.savefig("outputs/labels_correlogram.jpg")
        plt.close()

    def display_all(self):
        print("Generating metrics plots...")
        self.plot_metrics()
        print("Generating precision-recall curves...")
        self.plot_precision_recall_curves()
        print("Generating confusion matrix...")
        self.plot_confusion_matrix()
        print("Generating labels correlogram...")
        self.generate_label_correlogram()
        print("All visualizations saved in 'outputs' directory.")

if __name__ == "__main__":
    results_csv_path = "runs/detect/combined_results.csv"
    best_model_path = "runs/detect/train13/weights/epoch15.pt"

    affichage = AffichageResults(results_csv_path, best_model_path)
    affichage.display_all()
