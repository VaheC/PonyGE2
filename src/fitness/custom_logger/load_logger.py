import logging
import os

def create_logger(filename='process.log', dir_path='log'):

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    logger = logging.getLogger(__name__)
    logger.setLevel('DEBUG')

    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s: %(message)s",
        style='%',
        datefmt="%Y-%m-%d %H:%M"
    )

    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel('DEBUG')
    terminal_handler.setFormatter(formatter)
    logger.addHandler(terminal_handler)

    file_handler = logging.FileHandler(filename=f"{dir_path}/{filename}", mode='a', encoding='utf-8')
    file_handler.setLevel('DEBUG')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def create_terminal_logger():
    
    logger = logging.getLogger(__name__)
    logger.setLevel('DEBUG')

    formatter = logging.Formatter(
        "%(levelname)s %(asctime)s: %(message)s",
        style='%',
        datefmt="%Y-%m-%d %H:%M"
    )

    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel('DEBUG')
    terminal_handler.setFormatter(formatter)
    logger.addHandler(terminal_handler)

    return logger