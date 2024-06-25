#!/bin/bash

# Function to move files and create directories if they don't exist
move_files() {
    local source_dir=$1
    local dest_dir=$2

    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"

    # Move files
    mv "$source_dir"/* "$dest_dir"/ 2>/dev/null

    # Check if the move was successful
    if [ $? -eq 0 ]; then
        echo "Files moved successfully from $source_dir to $dest_dir"
    else
        echo "No files found in $source_dir or error occurred while moving"
    fi
}

# Move files from Midjourney/test/FAKE/ to test/fake/midjourney_cifake/
move_files "Midjourney/test/FAKE" "test/fake/midjourney_cifake"

# Move files from Midjourney/train/FAKE/ to training/train/fake/midjourney_cifake/
move_files "Midjourney/train/FAKE" "training/train/fake/midjourney_cifake"

# Move files from Midjourney/valid/FAKE/ to training/valid/fake/midjourney_cifake/
move_files "Midjourney/valid/FAKE" "training/valid/fake/midjourney_cifake"

echo "File moving process completed."
