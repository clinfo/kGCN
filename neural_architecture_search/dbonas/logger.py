import logging


def get_logger(logname):
    logger = logging.getLogger(logname)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    fmt = " - ".join(
        [
            "%(asctime)s",
            "%(name)-15s",
            "%(levelname)5s",
            "FILE: %(filename)-25s",
            "FUNC: %(funcName)-20s",
            "LINE: %(lineno)3d",
            "MSG: %(message)s",
        ]
    )
    handler_format = logging.Formatter(fmt)
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    return logger
