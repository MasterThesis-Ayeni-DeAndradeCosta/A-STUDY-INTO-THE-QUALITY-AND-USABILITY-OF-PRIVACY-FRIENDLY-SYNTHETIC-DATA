import os
import datetime

def create_output_directory(dataset_name, base_dir="singular"):
    """
    Creates a structured output directory under the specified base directory with a timestamp.
    Returns the path to the created directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", base_dir, f"{dataset_name}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir
