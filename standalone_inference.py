import os
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser(description="End-to-End Inference matching FishAudio Website Quality")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--reference-audio", type=str, required=True, help="Path to reference audio")
    parser.add_argument("--prompt-text", type=str, required=True, help="Transcript of the reference audio")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to output audio")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/s2-pro", help="Path to the model checkpoint")
    args = parser.parse_args()

    # Create temporary directories for processing
    tmp_dir = Path("tmp_processing")
    tmp_dir.mkdir(exist_ok=True)
    
    ref_audio_path = Path(args.reference_audio)
    
    # 1. Vocal Separation (Audio Preprocessing)
    print("\n--- Step 1: Separating Vocals ---")
    sep_dir = tmp_dir / "separated"
    sep_dir.mkdir(exist_ok=True)
    # Using python -m to call fish_audio_preprocess modules
    run_cmd([
        "python", "-m", "fish_audio_preprocess.cli.separate_audio",
        "--input_dir", str(ref_audio_path.parent),
        "--output_dir", str(sep_dir)
    ])
    
    # Assume the separated vocal is saved with a specific suffix or in the output_dir
    # (Adjust this path based on how fish_audio_preprocess saves it, commonly it keeps the filename)
    sep_audio_path = sep_dir / ref_audio_path.name
    if not sep_audio_path.exists():
        print(f"Warning: separated audio not found at {sep_audio_path}, falling back to original.")
        sep_audio_path = ref_audio_path

    # 2. Loudness Normalization (Audio Preprocessing)
    print("\n--- Step 2: Normalizing Loudness ---")
    norm_dir = tmp_dir / "normalized"
    norm_dir.mkdir(exist_ok=True)
    run_cmd([
        "python", "-m", "fish_audio_preprocess.cli.loudness_norm",
        "--input_dir", str(sep_dir),
        "--output_dir", str(norm_dir),
        "--loudness", "-23"  # Standard broadcast loudness
    ])
    
    norm_audio_path = norm_dir / ref_audio_path.name
    if not norm_audio_path.exists():
        norm_audio_path = sep_audio_path

    # 3. Generate Prompt Tokens (fish-speech)
    print("\n--- Step 3: Generating Prompt Tokens ---")
    run_cmd([
        "python", "fish_speech/models/dac/inference.py",
        "-i", str(norm_audio_path),
        "--checkpoint-path", f"{args.checkpoint_path}/codec.pth"
    ])
    
    # The output is saved as fake.npy by default in fish_speech/models/dac/inference.py
    prompt_tokens_path = "fake.npy"

    # 4. Generate Semantic Tokens from Text (fish-speech)
    print("\n--- Step 4: Generating Semantic Tokens ---")
    run_cmd([
        "python", "fish_speech/models/text2semantic/inference.py",
        "--text", args.text,
        "--prompt-text", args.prompt_text,
        "--prompt-tokens", prompt_tokens_path,
        "--checkpoint-path", args.checkpoint_path,
        "--num-samples", "1"
        # "--compile" # Uncomment for speed if needed
    ])
    
    # Output is codes_0.npy
    semantic_tokens_path = "codes_0.npy"

    # 5. Decode to Waveform (Vocoder)
    print("\n--- Step 5: Decoding to Waveform using Vocoder ---")
    run_cmd([
        "python", "fish_speech/models/dac/inference.py",
        "-i", semantic_tokens_path,
        "--checkpoint-path", f"{args.checkpoint_path}/codec.pth"
    ])
    
    # Output is fake.wav
    generated_audio_path = Path("fake.wav")
    if generated_audio_path.exists():
        generated_audio_path.rename(args.output)
        print(f"\n✅ Success! Audio saved to {args.output}")
    else:
        print("\n❌ Error: Output audio not generated.")

if __name__ == "__main__":
    main()
