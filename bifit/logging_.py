import logging


def create_logger(name: str, level=logging.INFO):
    """
    Creates a logger object.

    Args:
        name (str): Name of the logger.
        level (int): Logging level.

    Returns:
        logging.Logger: The logger object.
    """
    logger_ = logging.getLogger(name)
    logger_.setLevel(level=level)

    format_style = "[%(asctime)s] {%(module)s:%(lineno)d} %(levelname)s - %(message)s"
    date_format = "%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=format_style, datefmt=date_format)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger_.addHandler(stream_handler)

    return logger_


logger = create_logger(__name__)
