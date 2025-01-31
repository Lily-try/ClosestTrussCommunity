import logging
'''
用于输出日志
'''

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    if filename:#如果指定了日志文件，则创建fileHandler
        fh = logging.FileHandler(filename, "w",encoding='utf-8') #每次覆盖，a是追加
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    #无论是否指定日志文件，始终输出日志到控制台
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_log_path(logroot,args):
    if args.attack == 'meta':
        return f'{logroot}{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}.log'
    elif args.attack == 'random':
        return f'{logroot}{args.dataset}_{args.attack}_{args.type}_{args.ptb_rate}_{args.method}.log'
    elif args.attack == 'add':
        return f'{logroot}{args.dataset}_{args.attack}_{args.noise_level}_{args.method}.log'
    elif args.attack == 'del':
        return f'{logroot}{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}.log'
    else: #原始图
        return f'{logroot}{args.dataset}_{args.method}.log'
