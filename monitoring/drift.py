import logging
from scipy.stats import ks_2samp

#set up logging
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - DRIFT MONITOR - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
