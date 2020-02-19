#
# My Toolkits:
#
# @author: Eric Li
# 1. 日志logging：
#

import logging, time
import os
import yaml
import time
import logging
import logging.config

__all__ = ['print_time', 'load_config']

# 增加如下代码， 并配置config.yaml文件，使用_logger.info()等。
# import logging.config
# logging.config.dictConfig(load_config('./conf/train_config.yaml')['train_logging'])

def load_config(data_path):
    f = open(data_path, 'r', encoding='UTF-8')
    conf = yaml.load(f)
    f.close()
    return conf

# configure_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../conf/train_config.yaml')
# config_ = load_config(configure_path)

def print_time(f):
    """ 装饰器：记录函数运行时间
        from print_time import print_time as pt

        @pt
        def work(...):
            print('work is running')

        word()
        # work is running
        # --> RUN TIME: <work> : 2.8371810913085938e-05

    """
    def fi(*args, **kwargs):
        s = time.time()
        res = f(*args, **kwargs)
        print('--> RUN TIME: <%s> : %6.2f' % (f.__name__, time.time() - s))
        return res

    return fi


# 日志记录函数初始化：记录级别 DEBUG < INFO < WARNING < ERROR < CRITICAL
def init_logging(logfile="run.log", loglevel=logging.DEBUG):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "[%Y-%m-%d %H:%M:%S]"

    logging.basicConfig(filename=logfile, level=loglevel, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    return True


def write_log(msg, log=logging.INFO, show=False, *args, **kwargs):
    # initLogging(logfile=logfile, loglevel=loglevel)
    if log == logging.DEBUG:
        logging.debug(msg, *args, **kwargs)
    elif log == logging.CRITICAL:
        logging.critical(msg, *args, **kwargs)
    elif log == logging.WARNING:
        logging.warning(msg, *args, **kwargs)
    elif log == logging.ERROR:
        logging.error(msg, *args, **kwargs)
    else:
        logging.info(msg, *args, **kwargs)

    if show:
        print(msg, *args, **kwargs)

