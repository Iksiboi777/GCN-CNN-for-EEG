import os

def remove_history_files(root_dir):
    target_files = {'evolution_history.npy', 'training_history.npy'}
    deleted_count = 0

    print(f"Scanning '{root_dir}' for files: {target_files}...")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in target_files:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

    print(f"Finished. Total files deleted: {deleted_count}")

if __name__ == "__main__":
    # Use the current working directory
    current_directory = os.getcwd()
    remove_history_files(current_directory)