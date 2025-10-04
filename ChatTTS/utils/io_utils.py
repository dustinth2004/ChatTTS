
import os
import logging

def get_latest_modified_file(directory):
    """Gets the path of the most recently modified file in a directory.

    Args:
        directory (str): The path to the directory.

    Returns:
        str or None: The path to the most recently modified file, or None if the
            directory is empty.
    """
    logger = logging.getLogger(__name__)
    
    files = [os.path.join(directory, f) for f in os.listdir(directory)] 
    if not files:
        logger.log(logging.WARNING, f'No files found in the directory: {directory}')
        return None
    latest_file = max(files, key=os.path.getmtime)

    return latest_file