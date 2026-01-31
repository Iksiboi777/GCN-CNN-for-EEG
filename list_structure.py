import os

def separate_dirs_and_files(path="."):
    directories = []
    files = []

    # Get all entries in the directory
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    directories.append(entry.name)
                elif entry.is_file():
                    files.append(entry.name)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return [], []

    return sorted(directories), sorted(files)

if __name__ == "__main__":
    # Uses current directory by default
    current_path = os.getcwd()
    
    dirs, file_list = separate_dirs_and_files(current_path)

    print(f"Scanning: {current_path}\n")

    print(f"--- Directories ({len(dirs)}) ---")
    print(dirs)
    
    print(f"\n--- Files ({len(file_list)}) ---")
    print(file_list)

    # Helper for your Modal exclude list
    print(f"\n[Tip for modal_orchestrator.py]") 
    print("Copy names from the Directories list above into your 'exclude' list if they contain data or large artifacts.")