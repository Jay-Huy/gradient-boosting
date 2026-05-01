import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.data.shakespeare_char import ShakespeareCharDataModule
from src.models.gpt2 import GPT2LanguageModel, GPT2BoostingLanguageModel
from src.orchestrator.checkpointing import load_checkpoint

def find_best_checkpoint(run_dir: Path) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Tìm các thư mục có dạng learner_XXX
    learner_folders = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("learner_")]
    
    if not learner_folders:
        pts = list(checkpoints_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint found in {checkpoints_dir}")
        for pt in pts:
            if pt.name == "best.pt":
                return pt
        return pts[0]
        
    # Sắp xếp để lấy learner cuối cùng
    def get_learner_id(folder_path: Path) -> int:
        try:
            return int(folder_path.name.split("_")[1])
        except (IndexError, ValueError):
            return -1
            
    learner_folders.sort(key=get_learner_id)
    last_learner_dir = learner_folders[-1]
    
    # Ưu tiên lấy best.pt, sau đó đến final.pt hoặc latest.pt
    for pt_name in ["best.pt", "final.pt", "latest.pt"]:
        pt_path = last_learner_dir / pt_name
        if pt_path.exists():
            return pt_path
            
    raise FileNotFoundError(f"No suitable checkpont (.pt) found in {last_learner_dir}")

def main():
    parser = argparse.ArgumentParser(description="Output 500 samples from validation data using a trained checkpoint.")
    parser.add_argument("--config", type=str, default="configs/gpt2_shakespear_boosting_checkpointed.yaml", help="Path to config file")
    parser.add_argument("--run-dir", type=str, required=True, help="Base directory of the run (e.g., results/runs/6weak_1layer)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    
    args = parser.parse_args()

    run_dir_path = Path(args.run_dir)
    target_checkpoint = find_best_checkpoint(run_dir_path)
    output_path = run_dir_path / "samples.json"

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loading data module...")
    data_module = ShakespeareCharDataModule(config["data"]["params"])
    data_module.load_data()
    
    # Initialize the correct model
    print("Initializing model...")
    is_boosting = "weak_learner" in config["model"]["params"]
    if is_boosting:
        model = GPT2BoostingLanguageModel(config["model"]["params"])
    else:
        model = GPT2LanguageModel(config["model"]["params"])
        
    model = model.to(args.device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {target_checkpoint}...")
    load_checkpoint(model, target_checkpoint, map_location=args.device)
    model.eval()

    samples_collected = []
    
    # Calculate how many batches we need
    batch_size = config["data"]["params"].get("batch_size", 64)
    num_batches = (args.num_samples + batch_size - 1) // batch_size
    
    print(f"Running inference for {args.num_samples} samples (~{num_batches} batches)...")
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Inferencing"):
            batch = data_module.get_batch("val")
            batch_inputs = batch["input_ids"].to(args.device)
            batch_targets = batch["targets"].to(args.device)
            
            # Run inference
            infer_batch = {"input_ids": batch_inputs}
            predictions = model.infer(infer_batch)
            
            # Process each sample in the batch
            for idx in range(batch_inputs.size(0)):
                if len(samples_collected) >= args.num_samples:
                    break
                    
                # Decode the sequences back to text
                input_text = data_module.decode(batch_inputs[idx].tolist())
                target_text = data_module.decode(batch_targets[idx].tolist())
                pred_text = data_module.decode(predictions[idx].tolist())
                
                samples_collected.append({
                    "sample_index": len(samples_collected) + 1,
                    "input_snippet": input_text,
                    "target_snippet": target_text,
                    "predicted_snippet": pred_text
                })

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples_collected, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Successfully saved {len(samples_collected)} validation samples to {output_path}")
    print("\n--- Example Sample (First Sample) ---")
    
    first_sample = samples_collected[0]
    print(f"INPUT:\n{first_sample['input_snippet'][:100]}...\n")
    print(f"TARGET:\n{first_sample['target_snippet'][:100]}...\n")
    print(f"PREDICTED:\n{first_sample['predicted_snippet'][:100]}...\n")

if __name__ == "__main__":
    main()
