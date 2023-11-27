#!/bin/bash

# Set the paths for the train and validate folders
train_folder="train"
validate_folder="validate"

# Create the validate folder if it doesn't exist
mkdir -p "$validate_folder"

# Loop through each subfolder in the train folder
for subfolder in "$train_folder"/*; do
    # Get the subfolder name
    subfolder_name=$(basename "$subfolder")

    # Create the corresponding subfolder in the validate folder
    validate_subfolder="$validate_folder/$subfolder_name"
    mkdir -p "$validate_subfolder"

    # Get the number of files in the subfolder
    num_files=$(find "$subfolder" -type f | wc -l)

    # Calculate the number of files to extract (10%)
    num_files_to_extract=$((num_files / 10))

    # Randomly select the files to extract
    files_to_extract=$(find "$subfolder" -type f -print0 | shuf -n "$num_files_to_extract" -z)

    # Copy the selected files to the corresponding subfolder in validate
    cp -t "$validate_subfolder" -- "$files_to_extract"
done

echo "Extraction completed."