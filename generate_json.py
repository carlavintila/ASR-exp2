import librosa
import soundfile
import torch
import hydra
import json
import os
import re
import subprocess


OUTPUT_FILE = ""

def get_spkr_gender_json(metadata_path, language, output_dir=OUTPUT_FILE):
    """
    Generate JSON files containing speaker IDs and corresponding genders from metadata.

    Parameters:
        metadata_path (str): Path to the metadata directory containing 'metainfo.txt'.
        language (str): Language name used for naming output files.
        output_dir (str): Directory where output JSON files will be stored.

    Outputs:
        - JSON file with sorted speaker IDs.
        - JSON file with gender values (0 for male, 1 for female).
    """
    os.makedirs(output_dir, exist_ok=True)

    speaker_id_path = os.path.join(output_dir, f"mls_test_speakers_{language}.json")
    gender_path = os.path.join(output_dir, f"mls_test_gender_{language}.json")

    with open(f"{metadata_path}/metainfo.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = []

    data = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.strip().split('|')]
        if len(parts) >= 6:
            speaker_id, gender, partition, minutes, book_id, title = parts[:6]
            if partition.lower() != "test":
                continue  # Only include test partition
            try:
                data.append({
                    "speaker_id": str(speaker_id),
                    "gender": gender,
                    "partition": partition,
                    "duration": float(minutes),
                    "book_id": book_id
                })
            except ValueError:
                continue  # skip malformed lines

    speaker_ids = sorted({entry["speaker_id"] for entry in data})
    unique_speakers = {}
    for entry in data:
        sid = entry["speaker_id"]
        if sid not in unique_speakers:
            unique_speakers[sid] = 0 if entry["gender"].upper() == "M" else 1
    speaker_ids = sorted(unique_speakers.keys())
    gender_map = [unique_speakers[sid] for sid in speaker_ids]

    with open(speaker_id_path, "w") as f:
        json.dump(speaker_ids, f, indent=2)

    with open(gender_path, "w") as f:
        json.dump(gender_map, f, indent=2)

    print(f"JSON files saved to: {output_dir}")

def convert_all_wav_to_opus(input_dir, output_dir=None, bitrate="64k"):
    """
    Convert all .wav audio files in a directory to .opus format using ffmpeg.

    Parameters:
        input_dir (str): Directory containing .wav files.
        output_dir (str, optional): Directory to save converted .opus files. Defaults to input_dir.
        bitrate (str): Bitrate for output .opus files. Default is "64k".
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"{input_dir} is not a valid directory.")

    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".opus"
            output_path = os.path.join(output_dir, output_filename)

            command = [
                "ffmpeg",
                "-i", input_path,
                "-c:a", "libopus",
                "-b:a", bitrate,
                output_path
            ]

            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Converted: {filename} to {output_filename}")
            except subprocess.CalledProcessError:
                print(f"Failed to convert: {filename}")

    print("Batch conversion completed.")

def get_preprocess_json(metadata_path, language="", output_dir=OUTPUT_FILE):
    """
    Create a JSON list of audio segment metadata for the 'test' partition.

    Parameters:
        metadata_path (str): Directory containing 'metainfo.txt'.
        language (str): Language name used in file path construction.
        output_dir (str): Directory where the JSON file will be saved.

    Outputs:
        - JSON file with entries in the format [audio_path, start_time, duration].
    """
    os.makedirs(output_dir, exist_ok=True)

    segments_file = f".\\test_{language}\\test\\segments.txt"
    base_audio_path = f".\\test_{language}\\test\\audio"
    preprocess_path = os.path.join(output_dir, f"mls_test_preprocess_{language}.json")

    test_speaker_ids = set()
    with open(os.path.join(metadata_path, "metainfo.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = [p.strip() for p in line.strip().split('|')]
            if len(parts) >= 3:
                speaker_id, _, partition = parts[:3]
                if partition.lower() == "test":
                    test_speaker_ids.add(speaker_id)  

    preprocess_entries = []
    with open(segments_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue

            segment_id, _, start, end = parts
            try:
                speaker_id, book_id, _ = segment_id.split("_")
            except ValueError:
                continue

            if speaker_id not in test_speaker_ids:
                continue  # skip if speaker not in test

            audio_path = os.path.join(base_audio_path, speaker_id, book_id, f"{segment_id}.wav")
            preprocess_entries.append([audio_path, 0.0, round(float(end)-float(start), 2)])

    with open(preprocess_path, "w", encoding="utf-8") as f:
        json.dump(preprocess_entries, f, indent=2)

    print(f"Saved {len(preprocess_entries)} entries to {output_dir}")


def audio_txt(audio_dir = "", output_txt=""):
    """
    List all .wav audio files in a directory and write their paths to a text file.

    Parameters:
        audio_dir (str): Directory containing .wav files.
        output_txt (str): Path to the output text file.
    """
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]

    with open(output_txt, "w", encoding="utf-8") as f:
        for audio_file in audio_files:
            f.write(f"{os.path.join(audio_dir, audio_file)}\n")

def audio_text_opus(audio_dir, output_txt):
    """
    List all .opus audio files in a directory and write their paths to a text file.

    Parameters:
        audio_dir (str): Directory containing .opus files.
        output_txt (str): Path to the output text file.
    """
    opus_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".opus"):
                opus_files.append(os.path.join(root, file))

    # Write the paths to the output text file
    with open(output_txt, "w", encoding="utf-8") as f:
        for opus_file in opus_files:
            f.write(f"{opus_file}\n")

def get_train_json(metadata_path, language, output_dir=OUTPUT_FILE):
    """
    Generate JSON files with speaker IDs and gender labels for the 'train' partition.

    Parameters:
        metadata_path (str): Path to metadata containing 'metainfo.txt'.
        language (str): Language identifier used in file naming.
        output_dir (str): Directory to store output JSON files.

    Outputs:
        - JSON file with sorted speaker IDs.
        - JSON file with gender values (0 for male, 1 for female).
    """
    os.makedirs(output_dir, exist_ok=True)

    speaker_id_path = os.path.join(output_dir, f"mls_train_speakers_{language}.json")
    gender_path = os.path.join(output_dir, f"mls_train_gender_{language}.json")

    with open(f"{metadata_path}/metainfo.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = []

    data = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.strip().split('|')]
        if len(parts) >= 6:
            speaker_id, gender, partition, minutes, book_id, title = parts[:6]
            if partition.lower() != "train":
                continue  # Only include test partition
            try:
                data.append({
                    "speaker_id": str(speaker_id),
                    "gender": gender,
                    "partition": partition,
                    "duration": float(minutes),
                    "book_id": book_id
                })
            except ValueError:
                continue  # skip malformed lines

    speaker_ids = sorted({entry["speaker_id"] for entry in data})
    unique_speakers = {}
    for entry in data:
        sid = entry["speaker_id"]
        if sid not in unique_speakers:
            unique_speakers[sid] = 0 if entry["gender"].upper() == "M" else 1
    speaker_ids = sorted(unique_speakers.keys())
    gender_map = [unique_speakers[sid] for sid in speaker_ids]

    with open(speaker_id_path, "w") as f:
        json.dump(speaker_ids, f, indent=2)

    with open(gender_path, "w") as f:
        json.dump(gender_map, f, indent=2)

    print(f"JSON files saved to: {output_dir}")

