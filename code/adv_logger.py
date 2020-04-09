import sys
from pathlib import Path
from logbook import Logger, RotatingFileHandler, StreamHandler

from consts import LOG_FILE

log_file = "Results/clf_log.log"

class AdvLogger(Logger):
    def __init__(self, log_file):
        super().__init__()
        self.handlers.append(RotatingFileHandler(log_file, bubble=True))
        self.handlers.append(StreamHandler(sys.stdout))

logger = AdvLogger(LOG_FILE)
