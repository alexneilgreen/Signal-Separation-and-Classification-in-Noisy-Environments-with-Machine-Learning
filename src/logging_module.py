import os
import sys
from functools import wraps

class LoggerWriter:
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log_file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.log_file.write(message)
        self.stdout.write(message)
        self.log_file.flush()
        self.stdout.flush()

    def flush(self):
        self.log_file.flush()
        self.stdout.flush()

def setup_logging(results_path='Results'):
    """
    Redirect print statements to both console and log file.
    
    Args:
        results_path (str): Path to results folder. Defaults to 'Results'.
    """
    log_file = os.path.join(results_path, 'log.txt')
    sys.stdout = LoggerWriter(log_file)

def print(*args, **kwargs):
    """
    Overridden print function that writes to both console and log file.
    """
    # Convert all arguments to strings
    message = ' '.join(map(str, args))
    
    # Use the overridden stdout (which is our LoggerWriter)
    sys.__stdout__.write(message + '\n')
    
    # Additional kwargs handling to match built-in print
    if kwargs.get('end'):
        sys.__stdout__.write(kwargs['end'])
    if kwargs.get('flush', False):
        sys.__stdout__.flush()