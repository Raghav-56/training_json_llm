"""
Data Preparation Tool for Emotion Recognition Dataset

This script transforms video frame directories into JSON training data formats suitable
for vision language models, including ShareGPT and standard vision-LM format.
"""

import json
import argparse
import logging
import sys
from pathlib import Path
import os

from config.logger_config import logger
from config.defaults import (
    JSON_INDENT,
    TRANSFORMED_DATA_FILENAME,
    VISION_LM_FILENAME,
    SHAREGPT_FILENAME,
    DEFAULT_ARGS,
)


def process_directory(root_dir):
    """
    Process a directory structure containing emotion categories and video files
    to create training data for a Vision-Language Model.

    Args:
        root_dir (str): Path to the root directory containing emotion categories

    Returns:
        dict: Dictionary of processed data with video entries containing all frames
    """
    logger.info("Processing directory structure from: %s", root_dir)
    root_path = Path(root_dir)

    # Check if the root directory exists
    if not root_path.exists():
        logger.error("Root directory %s does not exist.", root_dir)
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    # Dictionary to store all processed data
    all_data = {}
    video_count = 0

    # Iterate through all emotion category folders
    for emotion_dir in root_path.iterdir():
        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name
        logger.info("Processing emotion category: %s", emotion)

        # Iterate through all video files in the emotion category
        for video_file in emotion_dir.glob("*.mp4"):  # Adjust file extension as needed
            video_name = video_file.stem
            logger.info("Processing video: %s", video_name)

            video_entry = {
                "conversation": {
                    "0": {
                        "from": "human",
                        "value": "<image>What is the emotion of the person in the video?",
                    },
                    "1": {
                        "from": "gpt",
                        "value": f"The emotion expressed by the person in this video is: {emotion}",
                    },
                },
                "images": {"0": str(video_file)},  # Store the video file path directly
                "emotion": emotion,
            }

            # Add the video entry to the all_data dictionary using a unique identifier
            video_id = f"{emotion}_{video_name}"
            all_data[video_id] = video_entry

            logger.info("Processed video file: %s", video_name)
            video_count += 1

    logger.info("Completed processing %d video items", video_count)
    return all_data


def save_json_data(data, output_path):
    """
    Save JSON data to a file.

    Args:
        data: The data to save
        output_path (str): Path to save the JSON file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=JSON_INDENT)
        logger.info("Successfully saved data to %s", output_path)
    except Exception as e:
        logger.error("Failed to save JSON to %s: %s", output_path, e)
        raise


def create_vision_lm_format(processed_data):
    """
    Create a data format suitable for vision language models training.
    Creates one entry per video including all frames.

    Args:
        processed_data (dict): Dictionary containing processed videos data

    Returns:
        list: Data formatted for vision language model training
    """
    logger.info("Creating vision-LM format...")
    vision_lm_data = []

    for video_id, video_data in processed_data.items():
        conversation = video_data["conversation"]
        image_paths = list(video_data["images"].values())

        entry = {
            "messages": [
                {
                    "content": conversation["0"]["value"],
                    "role": "user",
                },
                {"content": conversation["1"]["value"], "role": "assistant"},
            ],
            "images": image_paths,
            "emotion": video_data["emotion"],
            "video_id": video_id,
        }
        vision_lm_data.append(entry)
    logger.info("Created %d vision-LM format entries", len(vision_lm_data))
    return vision_lm_data


def create_sharegpt_format(processed_data):
    """
    Create data in ShareGPT format for training.
    Creates one entry per video including all frames.

    Args:
        processed_data (dict): Dictionary containing processed videos data

    Returns:
        list: Data formatted for ShareGPT
    """
    logger.info("Creating ShareGPT format...")
    sharegpt_data = []

    for video_id, video_data in processed_data.items():
        conversation = video_data["conversation"]
        image_paths = list(video_data["images"].values())

        entry = {
            "conversations": [
                {
                    "from": conversation["0"]["from"],
                    "value": conversation["0"]["value"],
                },
                {
                    "from": conversation["1"]["from"],
                    "value": conversation["1"]["value"],
                },
            ],
            "images": image_paths,
            "emotion": video_data["emotion"],
            "video_id": video_id,
        }
        sharegpt_data.append(entry)
    logger.info("Created %d ShareGPT format entries", len(sharegpt_data))
    return sharegpt_data


def process_data(input_dir, output_dir=None):
    """
    Process data from input directory and generate transformed formats.

    Args:
        input_dir (str): Path to the input directory containing emotion categories
        output_dir (str, optional): Directory to save output files. Defaults to same directory as input.

    Returns:
        tuple: Paths to the created files
    """
    input_path = Path(input_dir)
    if not output_dir:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    # Process directory to get structured data
    processed_data = process_directory(input_path)

    # Save the transformed data
    transformed_path = output_dir / TRANSFORMED_DATA_FILENAME
    save_json_data(processed_data, transformed_path)

    # Create and save vision LM format
    vision_lm_data = create_vision_lm_format(processed_data)
    vision_lm_path = output_dir / VISION_LM_FILENAME
    save_json_data(vision_lm_data, vision_lm_path)

    # Create and save ShareGPT format
    sharegpt_data = create_sharegpt_format(processed_data)
    sharegpt_path = output_dir / SHAREGPT_FILENAME
    save_json_data(sharegpt_data, sharegpt_path)

    logger.info("Data processing complete.")
    return transformed_path, vision_lm_path, sharegpt_path


def main():
    """
    Main entry point for the data preparation script.
    """
    parser = argparse.ArgumentParser(
        description="Transform video frames into JSON formats for vision language models"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing emotion categories and video frames",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for generated data (default: same directory as input)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=DEFAULT_ARGS["verbose"],
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        output_files = process_data(args.input, args.output)
        print("\nProcessing Summary:")
        print("------------------")
        print(f"Input directory: {args.input}")
        print("Output files:")
        for path in output_files:
            print(f"  - {path}")
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        logger.error("An error occurred during processing: %s", e, exc_info=True)
        sys.exit(1)
    except OSError as e:
        logger.error("OS error during processing: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
