import logging

logging.basicConfig()

# Some global vars
log = logging.getLogger(__name__) # A logger for this file
graph = None
start_time = None
dry_run = False

# Set logging level to the logger
log.setLevel(logging.DEBUG)

log.info("Logger initiated.")