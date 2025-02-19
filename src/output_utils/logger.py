import os
import logging

def setup_logger(output_dir):
    """
    Sets up a logger that writes logs to both the console and a log file.
    """
    log_file = os.path.join(output_dir, "benchmark.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("BenchmarkLogger")
