import logging
from datetime import datetime


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


LOGGER = logging.getLogger('coarse2fine')
LOGGER.setLevel(10)
fileHandler = logging.FileHandler('./coarse2fine.log')
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
LOGGER.addHandler(fileHandler)
LOGGER.addHandler(streamHandler)
