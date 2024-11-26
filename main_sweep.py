import yaml
import wandb
import pandas as pd
import random
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
from wandb.integration.ultralytics import add_wandb_callback


if __name__ == "__main__":
    with open('sweep_config.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)
        
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Define the model, n = nano, s = small, m = medium, l = large, x = extra large
    # Larger models are more accurate but are slower and require more GPU memory
    theModel = "yolo11s.pt"
    
    max_runs = 10
    for i in range(max_runs):
        learning_rate = random.choice(sweep_config["parameters"]["learning_rate"]["values"])
        batch_size = random.choice(sweep_config["parameters"]["batch_size"]["values"])
        image_size = random.choice(sweep_config["parameters"]["image_size"]["values"])
        optimizer = random.choice(sweep_config["parameters"]["optimizer"]["values"])
        print(f"Run #{i+1}/{max_runs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {image_size}")
        print(f"Optimizer: {optimizer}")

        # initialize wandb for logging
        wandb.init(
            project = config["wandb_project"], 
            job_type= "training",
            entity= "danieedv",
            name= f"{theModel}_baseline_2",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "image_size": image_size,
                "optimizer": optimizer
                }
            )

        # Pretrained model, small version
        model = YOLO(theModel)  
        add_wandb_callback(model, enable_model_checkpointing = True)  # Add wandb callback

        # Train..
        train_results = model.train(
            project = config["wandb_project"],
            data = "data.yaml", 
            epochs = config["max_epochs"], 
            imgsz = image_size,
            workers = config["num_workers"],
            patience = config["early_stopping_patience"],
            batch = batch_size,
            optimizer = optimizer,
            lr0 = learning_rate
            )

        # Log results.csv to wandb:
        output_folder = train_results.save_dir
        results_csv_file = f"{output_folder}/results.csv"
        df = pd.read_csv(results_csv_file)
        for _, row in df.iterrows():
            metrics = {
                "epoch": row["epoch"],
                "time": row["time"],
                "train/box_loss": row["train/box_loss"],
                "train/cls_loss": row["train/cls_loss"],
                "train/dfl_loss": row["train/dfl_loss"],
                "metrics/precision(B)": row["metrics/precision(B)"],
                "metrics/recall(B)": row["metrics/recall(B)"],
                "metrics/mAP50(B)": row["metrics/mAP50(B)"],
                "metrics/mAP50-95(B)": row["metrics/mAP50-95(B)"],
                "val/box_loss": row["val/box_loss"],
                "val/cls_loss": row["val/cls_loss"],
                "val/dfl_loss": row["val/dfl_loss"],
                "lr/pg0": row["lr/pg0"],
                "lr/pg1": row["lr/pg1"],
                "lr/pg2": row["lr/pg2"],
            }
            wandb.log(metrics)

        # Validation
        # results = model.val(save_json=True)  # save results

        wandb.finish()