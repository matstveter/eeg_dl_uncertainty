def write_metrics_to_file(metrics_majority_vote, metrics_final_classes, file_path):
    """Write the calculated metrics to a file."""
    with open(file_path, 'w') as f:
        f.write("Majority Vote Metrics:\n")
        for metric, value in metrics_majority_vote.items():
            f.write(f"{metric}: {value}\n")

        f.write("\nFinal Classes Metrics:\n")
        for metric, value in metrics_final_classes.items():
            f.write(f"{metric}: {value}\n")
