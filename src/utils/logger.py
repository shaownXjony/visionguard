import logging

def setup_logger(verbose: bool = False) -> None:
    """
    Configure global logging settings.
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
