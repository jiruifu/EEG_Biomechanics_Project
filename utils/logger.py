from datetime import datetime
import os
import logging
import traceback
from typing import List, Tuple, Dict, Union, Optional

class SingletonLogger:
    _instance = None
    _log_path = None
    
    @classmethod
    def get_logger(cls, name="experiment", report_dir="logs"):
        if cls._instance is None or not cls._instance.handlers:  # Check if logger has handlers
            # Create new logger instance
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            if not os.path.exists(report_dir):
                os.makedirs(os.path.join(os.path.dirname(__file__), report_dir))
                print("\n" + "="*15 + " Creating Report Directory " + "="*15)
                print(f"Created report directory: {os.path.join(os.path.dirname(__file__), report_dir)}")
            
            log_filename = f"{name}_{timestamp}.log"
            cls._log_path = os.path.join(os.path.dirname(__file__), report_dir, log_filename)
            
            # Configure logger
            logger = logging.getLogger('experiment_logger')
            logger.setLevel(logging.INFO)
            
            # Prevent logging from propagating to the root logger
            logger.propagate = False
            
            # Clear any existing handlers
            if logger.handlers:
                logger.handlers.clear()
            
            # Create file handler
            file_handler = logging.FileHandler(cls._log_path, mode='a')  # 'a' for append mode
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            logger.txt_path = cls._log_path
            
            cls._instance = logger
        
        return cls._instance

def setup_experiment_logger(name, report_dir):
    """Set up logger for the entire experiment"""
    return SingletonLogger.get_logger(name=name, report_dir=report_dir)

def log_and_print(logger: Optional[logging.Logger]=None, message: Optional[str]=None, error: Optional[str]=None):
    """Helper function to both log and print a message"""
    def print_message(message):
        print(message)
    
    match (message, error):
        case (None, None):
            if logger is not None:
                logger.error("Teminated by Fatal Error: Either Message or Error is needed") # Both message and error are strings
            raise ValueError("Teminated by Fatal Error: Either Message or Error is needed")
        case (str(), str()): 
            if logger is not None:
                logger.error("Teminated by Fatal Error: Cannot provide both message and error") # Both message and error are strings
            raise ValueError("Teminated by Fatal Error: Cannot provide both message and error")
        case (str(), None):  # Only message is provided
            print_message(message)
            if logger is not None:
                logger.info(message)
        case (None, str()):  # Only error is provided
            if logger is not None:
                logger.error(error)


if __name__ == "__main__":
    # Test the logger
    log = setup_experiment_logger("test", "logs")
    try:
        log_and_print(log, message="Hello, World!")
        log_and_print(log, error="This is an error message")
        # log_and_print(log, message="This is a message", error="This is an error message")
        log_and_print(log)
        # log_and_print(log, message="This is a message", error="This is an error message")
    except ValueError as e:
        pass
