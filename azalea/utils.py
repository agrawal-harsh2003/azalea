
import importlib
from typing import Any
import sys
import os

def import_and_get(name: str) -> Any:
    """Import module and return value from within.
    """
    if '.' not in name:
        raise ImportError(f'name is not like <module>.<name>: {name}')
    module_name, attr_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr_name):
        raise ImportError(f'name not found in module: {name}')
    return getattr(module, attr_name)

def redirect_all_output(log_file):
    """
    Redirect all standard output and standard error to a log file.
    
    Args:
        log_file (str): Path to the log file where output will be written
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Open the log file in write mode
    try:
        log_handler = open(log_file, 'w')
        
        # Redirect stdout
        sys.stdout = log_handler
        # Redirect stderr
        sys.stderr = log_handler
        
        # Optional: Return the file handler in case you want to close it later
        return log_handler
    
    except IOError as e:
        print(f"Error opening log file {log_file}: {e}")
        return None