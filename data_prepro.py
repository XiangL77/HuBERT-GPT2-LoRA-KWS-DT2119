import os
import random
import json
from pathlib import Path
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T

# === CONFIGURATION === #
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR =  BASE_DIR / "speech_commands_v0.02"
OUTPUT_DIR = BASE_DIR / "processed"
SAMPLE_RATE = 16000
TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# === FUNCTIONS === #

def get_all_files(label):
    """List all .wav files for a given keyword label."""
    label_path = DATASET_DIR / label
    return [f for f in label_path.glob("*.wav")]

def preprocess_and_save(file_path, label, out_dir, resampler):
    """Load, resample, normalize, and save an audio file."""
    waveform, sr = torchaudio.load(file_path)

    if sr != SAMPLE_RATE:
        waveform = resampler(waveform)

    # Normalize to -1 to 1
    waveform = waveform / waveform.abs().max()

    relative_path = file_path.relative_to(DATASET_DIR)
    output_path = out_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, SAMPLE_RATE)

    return str(output_path), label

def split_dataset(file_list):
    """Split files into train, val, test."""
    random.shuffle(file_list)
    total = len(file_list)
    val_count = int(total * VAL_RATIO)
    test_count = int(total * TEST_RATIO)
    return (
        file_list[val_count + test_count:],  # train
        file_list[:val_count],               # val
        file_list[val_count:val_count + test_count]  # test
    )

def main():
    print(f"Loading from: {DATASET_DIR}")
    print(f"Resolved path: {DATASET_DIR.resolve()}")
    print("Exists:", DATASET_DIR.exists())
    print("Contents:", list(DATASET_DIR.iterdir()))

    print(f"Saving to: {OUTPUT_DIR}")
    
    random.seed(42)
    resampler = T.Resample(orig_freq=16000, new_freq=SAMPLE_RATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {"train": [], "val": [], "test": []}

    for label in TARGET_WORDS:
        print(f"\nProcessing keyword: {label}")
        files = get_all_files(label)
        train_files, val_files, test_files = split_dataset(files)

        for split_name, split_list in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            for f in tqdm(split_list, desc=f"{label} -> {split_name}"):
                out_path, lbl = preprocess_and_save(f, label, OUTPUT_DIR, resampler)
                all_data[split_name].append({"path": out_path, "label": lbl})

    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nDone. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
