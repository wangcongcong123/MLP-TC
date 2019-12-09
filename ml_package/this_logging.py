import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('output.log',mode="w")
fh.setFormatter(formatter)
logger.addHandler(fh)

