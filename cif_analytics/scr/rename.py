import os
import hashlib
import json

def main():
    rename_map = {}
    for filename in os.listdir("."):
        if filename.endswith(".cif"):
            file_path = os.path.join(".", filename)
            md5_hash = hashlib.md5(filename.encode()).hexdigest()
            new_filename = md5_hash + ".cif"
            new_file_path = os.path.join(".", new_filename)
            os.rename(file_path, new_file_path)
            rename_map[filename] = new_filename
            print(f"Renamed '{filename}' to '{new_filename}'")
    
    with open(os.path.join(".", 'rename_map.json'), 'w') as json_file:
        json.dump(rename_map, json_file, indent=4)

if __name__ == "__main__":
    main()

