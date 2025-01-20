import time

def timer(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        w, b, cost_history = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")
        return w, b, cost_history

    return wrapper