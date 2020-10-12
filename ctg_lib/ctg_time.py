import time
from time import strftime


def now_time_string():
    s_time = time.time()
    ss_time: str = strftime('%Y-%m-%d_%H-%M-%S', time.localtime(s_time))
    return ss_time

