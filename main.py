from data_loader import ContinualLearningLoader
from model import ContinualLearningModel
from trainer import ContinualLearningTrainer
import torch
import time
import uuid
import json
import os
import numpy as np
import logging
import sys
from codecarbon import EmissionsTracker
os.environ['CODECARBON_LOG_LEVEL'] = 'WARNING'
logging.getLogger("codecarbon").setLevel(logging.WARNING)


def save_experiment_result(experiment_name, metrics, config, filename='continual_learning_experiments.json'):    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_experiments = json.load(f)
    else:
        all_experiments = {}

    record = {
        "config": config,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

    all_experiments[experiment_name] = record

    with open(filename, 'w') as f:
        json.dump(all_experiments, f, indent=4)
    
    print(f"Experiment '{experiment_name}' saved to {filename}")

def run_continual_learning_experiment(data_loader=None, model=None, trainer=None):
    tracker = EmissionsTracker(
            project_name=f"{model.backbone_name}, {data_loader.dataset_name}, {data_loader.num_tasks} tasks",
            output_file=f"exps/Continual_Learning_Exp.csv"
        )
    os.makedirs("exps", exist_ok=True)

    print("Starting continual learning experiment...")
    t0 = time.time()
    tracker.start()
    metrics = trainer.train_continual()
    tracker.stop()
    t1 = time.time()
    emissions_data = tracker.final_emissions_data
    
    print(f"Continual learning completed in {t1 - t0:.2f} seconds.")

    unique_id = uuid.uuid4().hex[:8]  
    timestamp = time.strftime("%Y%m%d-%H%M%S")  

    experiment_name = (
        f"{model.backbone_name}, {data_loader.dataset_name}, "
        f"tasks={data_loader.num_tasks}, id={unique_id}, time={timestamp}"
    )
    config = {
        "dataset": data_loader.dataset_name,
        "scenario": data_loader.scenario,
        "num_tasks": data_loader.num_tasks,
        "backbone": model.backbone_name,
        "epochs_per_task": trainer.epochs_per_task,
        "buffer_size_per_task": trainer.buffer_size_per_task,
        "learning_rate": trainer.learning_rate,
        "device": str(model.device),
        "total_time_s": round(t1 - t0, 2),
        "id": unique_id,
        "timestamp": timestamp
    }
    metrics_json = {
        "task_accuracies": metrics["task_accuracies"],
        "average_accuracy": metrics["average_accuracy"],
        "training_times": metrics["training_times"],
        "extraction_times": metrics["extraction_times"],
        "final_average_accuracy": metrics["final_average_accuracy"]
    }

    save_experiment_result(experiment_name, metrics_json, config)

    trainer.print_final_report()
    trainer.save_metrics('continual_learning_results.json')

    print("Duration (min):", emissions_data.duration/60)
    print("Energy :", emissions_data.energy_consumed)
    print("Emissions (g):", emissions_data.emissions*1000)
    
    return trainer, metrics, model

def main(config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_cfg = config['experiment']
    model_cfg = config['model']
    train_cfg = config['trainer']

    data_loader = ContinualLearningLoader(
        dataset_name=exp_cfg['dataset_name'],
        scenario=exp_cfg['scenario'], 
        num_tasks=exp_cfg['num_tasks'],
        batch_size=exp_cfg['batch_size'],
        data_size=exp_cfg['data_size']
    )

    # Get one batch (x, y, t) from the first task loader
    sample_loader = data_loader.get_task_loader(task_id=0, train=True)
    sample_batch = next(iter(sample_loader))
    sample_x, _, _ = sample_batch

    model = ContinualLearningModel(
        decoupling=model_cfg['decoupling'],
        backbone=model_cfg['backbone'],  
        num_classes=data_loader.classes_per_task,
        pretrained=model_cfg['pretrained'],
        scenario=exp_cfg['scenario'],
        data_sample=sample_x,
        device=device,
        use_ensemble=model_cfg['use_ensemble'],
    )

    trainer = ContinualLearningTrainer(
        model=model,
        data_loader=data_loader,
        decoupling=model_cfg['decoupling'],
        learning_rate=train_cfg['learning_rate'],
        epochs_per_task=train_cfg['epochs_per_task'],  
        verbose=train_cfg['verbose'],
        batch_size=exp_cfg['batch_size'],
        weight_decay=train_cfg['weight_decay'],
        optimizer_type=train_cfg['optimizer_type'],
        use_ensemble=model_cfg['use_ensemble'],
        task_aware=train_cfg['task_aware'],
        buffer_size_per_task=train_cfg['buffer_size_per_task'],
        K_per_class=train_cfg['K_per_class'],
        device=device
    )

    trainer, metrics, model = run_continual_learning_experiment(data_loader, model, trainer)

    os.makedirs("saved_models", exist_ok=True)

    # Save model weights
    model_path = "saved_models/hydraCL_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save prototypes (.npz)
    np.savez("saved_models/hydraCL_prototypes.npz",
            **{f"task_{tid}": proto_data["centroids"]
                for tid, proto_data in trainer.proto_selector.task_prototypes.items()})
    print("Prototype centroids saved to saved_models/hydraCL_prototypes.npz")

    loader_config = {
        "dataset_name": data_loader.dataset_name,
        "scenario": data_loader.scenario,
        "num_tasks": data_loader.num_tasks,
        "batch_size": data_loader.batch_size,
        "data_size": data_loader.data_size
    }

    os.makedirs("saved_models", exist_ok=True)

    with open("saved_models/hydraCL_loader.json", "w") as f:
        json.dump(loader_config, f, indent=4)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    main(path)
