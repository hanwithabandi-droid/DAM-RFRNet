import kagglehub
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("mandybr/fogroad-12m-dataset")
print("\nPath to dataset files:", path)

# List the contents of the root folder so we can parse it
print("\nDataset contents:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    if level < 2:  # Print files for first 2 levels only to avoid huge outputs
        for f in files[:5]: # just print 5 files to get an idea
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... ({len(files)} files)")
