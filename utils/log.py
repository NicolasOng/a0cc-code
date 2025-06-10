import logging
import sys

def setup_logging(level: int = 5, process_name: str = "log"):
    root_logger = logging.getLogger()
    # always capture all logs
    root_logger.setLevel(level)

    # Clear existing handlers (helps when logging is reset in notebooks or multiple runs)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Error+Info file handler
    file_handler = logging.FileHandler(process_name + ".err.log")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Output file handler
    main_file_handler = logging.FileHandler(process_name + ".out.log")
    main_file_handler.setFormatter(formatter)
    main_file_handler.addFilter(lambda record: record.levelno == 25)
    root_logger.addHandler(main_file_handler)

    # log that logging was configured
    root_logger.debug("Logging initialized for " + process_name)
