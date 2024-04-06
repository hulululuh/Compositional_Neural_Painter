import logging
import colorlog

class Logger:
    def __init__(self, name=__name__, level=logging.DEBUG):
        self.logger = colorlog.getLogger(name)
        self.logger.setLevel(level)
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        ))
        self.logger.addHandler(handler)
        # Prevent the log messages from being duplicated in the python output
        self.logger.propagate = False

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

Log = Logger("log", logging.INFO)