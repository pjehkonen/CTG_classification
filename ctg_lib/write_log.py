import logging

def write_log(logger, msg_array):
    for element in msg_array:
        logger.info(element)
