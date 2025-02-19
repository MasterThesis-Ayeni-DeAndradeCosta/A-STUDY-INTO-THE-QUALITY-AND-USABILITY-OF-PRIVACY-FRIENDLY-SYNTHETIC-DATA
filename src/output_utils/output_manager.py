import os
import datetime

def create_output_directory(dataset_name):
    """
    Creates a structured output directory with a timestamp.
    Returns the path to the created directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("outputs", f"{dataset_name}_{timestamp}")
    
    # Ensure necessary subdirectories exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir
