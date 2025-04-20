import logging
import colorlog

# 创建日志记录器
logger = logging.getLogger('ircot')
logger.setLevel(logging.INFO)  # 设置全局日志级别

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 控制台输出所有级别的日志

# 创建文件处理器
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)  # 文件只记录 INFO 及以上级别的日志

# 设置日志格式
formatter = colorlog.ColoredFormatter(
    '[%(asctime)s] [%(filename)s - %(filename)s:%(lineno)d] [%(log_color)s%(levelname)s%(reset)s] - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    reset=True,
    datefmt='%Y/%m/%d %H:%M:%S'
)

# 将格式化器添加到处理器
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ANSI 转义序列
# 基本颜色
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
PRED = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'

# 明亮颜色
LBLACK = '\033[90m'
LRED = '\033[91m'
LGREEN = '\033[92m'
LYELLOW = '\033[93m'
LBLUE = '\033[94m'
LPRED = '\033[95m'
LCYAN = '\033[96m'
LWHITE = '\033[97m'
RESET = '\033[0m'


if __name__ == "__main__":
    print(f"{BLACK}这是黑色文本{RESET}")
    print(f"{RED}这是红色文本{RESET}")
    print(f"{GREEN}这是绿色文本{RESET}")
    print(f"{YELLOW}这是黄色文本{RESET}")
    print(f"{BLUE}这是蓝色文本{RESET}")
    print(f"{PRED}这是品红色文本{RESET}")
    print(f"{CYAN}这是青色文本{RESET}")
    print(f"{WHITE}这是白色文本{RESET}")

    print(f"{LBLACK}这是亮黑色文本{RESET}")
    print(f"{LRED}这是亮红色文本{RESET}")
    print(f"{LGREEN}这是亮绿色文本{RESET}")
    print(f"{LYELLOW}这是亮黄色文本{RESET}")
    print(f"{LBLUE}这是亮蓝色文本{RESET}")
    print(f"{LPRED}这是亮品红色文本{RESET}")
    print(f"{LCYAN}这是亮青色文本{RESET}")
    print(f"{LWHITE}这是亮白色文本{RESET}")
