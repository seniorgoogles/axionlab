import time
from colorama import Fore

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'\> Function {Fore.BLUE}{func.__name__}{Fore.RESET} took {Fore.BLUE}{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}{Fore.RESET} to execute')
        return result
    return wrapper
