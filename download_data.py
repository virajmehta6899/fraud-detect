import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("arockiaselciaa/creditcardcsv")
print("Downloaded to:", path)

# Copy to local data folder
os.makedirs("data", exist_ok=True)
shutil.copy(os.path.join(path, "creditcard.csv"), "data/creditcard.csv")
print("Copied to data/creditcard.csv")