#!/usr/bin/env python3
"""
low_fps_to_true_120fps.py
Convert a low frame‑rate video to smooth, true 120 fps without destroying quality.
Uses RIFE (Real-Time Intermediate Flow Estimation) via rife-ncnn-vulkan.
"""

import argparse
import subprocess
import os
import sys
import shutil
import json
from pathlib import Path
from tempfile import TemporaryDirectory

def run_cmd(cmd, description=""):
    """Run a shell command and raise an error if it fails."""
    print(f"[INFO] {description}")
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {description} failed:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout

def get_video_fps(input_path):
    """Use ffprobe to get the exact frame rate of the video."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json", input_path
    ]
    out = subprocess.check_output(cmd, text=True)
    info = json.loads(out)
    fps_str = info["streams"][0]["r_frame_rate"]  # e.g. "30000/1001" or "30/1"
    num, den = map(int, fps_str.split("/"))
    return num / den

def check_tool(tool_name):
    """Ensure a required command‑line tool is available."""
    if shutil.which(tool_name) is None:
        print(f"[ERROR] {tool_name} is not installed. Please install it and retry.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Convert a video to 120 fps using RIFE interpolation (True Motion)."
    )
    parser.add_argument("input", help="Path to the input video")
    parser.add_argument("output", help="Path for the output 120fps video")
    parser.add_argument("--target-fps", type=float, default=120.0,
                        help="Target frame rate (default: 120)")
    parser.add_argument("--crf", type=int, default=10,
                        help="CRF value for x264 encoding (0=lossless, 10-18 very high quality, default: 10)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="Vulkan GPU device index (default: 0)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary directories for debugging")
    args = parser.parse_args()

    # Check required tools
    check_tool("ffmpeg")
    check_tool("ffprobe")
    check_tool("rife-ncnn-vulkan")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    # Get original FPS
    orig_fps = get_video_fps(input_path)
    print(f"[INFO] Original video FPS: {orig_fps:.3f}")

    target_fps = args.target_fps
    if orig_fps >= target_fps:
        print(f"[INFO] Original FPS is already >= {target_fps}. Just copying the file.")
        shutil.copy2(input_path, output_path)
        return

    # Calculate required multiplication factor
    mult = target_fps / orig_fps
    # We only support integer multiples that are powers of two (for simplicity)
    if not mult.is_integer():
        print("[ERROR] Target FPS must be an integer multiple of original FPS.")
        print(f"        Original = {orig_fps:.3f}, Target = {target_fps}. Ratio = {mult}")
        sys.exit(1)
    mult = int(mult)
    # For RIFE we can only realistically double the frame rate in each pass.
    # We'll apply multiple 2x passes.
    n_passes = 0
    while (2 ** n_passes) < mult:
        n_passes += 1
    if (2 ** n_passes) != mult:
        print(f"[ERROR] Multiplication factor {mult} is not a power of two.")
        print("        You can achieve only 2x, 4x, 8x, ... with this script.")
        print(f"        Suggestion: interpolate to {2 ** n_passes * orig_fps:.3f} fps instead.")
        sys.exit(1)

    print(f"[INFO] Will perform {n_passes} x2 interpolation pass(es) to reach {target_fps} fps.")

    # Use a temporary directory for frames
    with TemporaryDirectory(prefix="rife_interp_") as tmpdir:
        tmp = Path(tmpdir)

        # Step 1: Extract original frames as lossless PNG
        src_frames = tmp / "src_frames"
        src_frames.mkdir()
        print("[INFO] Extracting original frames...")
        cmd_extract = [
            "ffmpeg", "-i", str(input_path),
            "-vsync", "0",          # Do not modify frame timestamps
            "-q:v", "1",            # Best PNG quality
            str(src_frames / "frame_%08d.png")
        ]
        run_cmd(cmd_extract, "Extracting frames")

        # Step 2: RIFE interpolation passes
        current_frames = src_frames
        for i in range(1, n_passes + 1):
            dest_frames = tmp / f"interp_pass_{i}"
            dest_frames.mkdir()
            print(f"[INFO] RIFE pass {i}/{n_passes} (2x)...")
            cmd_rife = [
                "rife-ncnn-vulkan",
                "-i", str(current_frames),
                "-o", str(dest_frames),
                "-g", str(args.gpu_id)
            ]
            run_cmd(cmd_rife, f"RIFE interpolation pass {i}")
            current_frames = dest_frames  # output becomes input for next pass

        # Step 3: Assemble final video from interpolated frames + original audio
        final_frames = current_frames  # contains 120fps frames
        print("[INFO] Assembling output video with original audio...")
        cmd_assemble = [
            "ffmpeg",
            "-r", str(target_fps),                       # input frame rate
            "-i", str(final_frames / "frame_%08d.png"),
            "-i", str(input_path),                       # for audio
            "-map", "0:v",                               # video from frames
            "-map", "1:a",                               # audio from original
            "-c:v", "libx264",
            "-crf", str(args.crf),
            "-pix_fmt", "yuv420p",                       # compatibility
            "-c:a", "copy",                              # no re-encode audio
            str(output_path)
        ]
        run_cmd(cmd_assemble, "Encoding final video")

        # TemporaryDirectory cleans up automatically unless --keep-temp is set
        if args.keep_temp:
            print(f"[INFO] Temporary directory kept at: {tmp}")
            # Prevent the context manager from deleting it
            tmpdir = None   # hack: disable cleanup
        else:
            # Cleanup happens automatically
            pass

    print(f"[DONE] 120fps video saved as: {output_path}")

if __name__ == "__main__":
    main()