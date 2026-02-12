import multiprocessing
from functools import wraps
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar, Callable, List, Iterable, Any

# Define TypeVars for generic input (T) and output (R)
T = TypeVar("T")
R = TypeVar("R")

def parallel(chunksize: int = 100) -> Callable[[Callable[[T], R]], Callable[[Iterable[T]], List[R]]]:
    """
    A decorator factory that enables process-safe parallel execution.
    
    If called from the MainProcess, it uses a ProcessPoolExecutor.
    If called within a child process, it executes serially to avoid nested pools.
    Possible improvements:
    - more robust check for main process
    - returning an iterator instead of a list for memory efficiency
    - using a single global pool to avoid overhead of creating multiple pools
    """
    def decorator(func: Callable[[T], R]) -> Callable[[Iterable[T]], List[R]]:
        @wraps(func)
        def wrapper(data_list: Iterable[T]) -> List[R]:
            # Check if we are in the main process to avoid spawning sub-pools
            if multiprocessing.current_process().name == 'MainProcess':
                with ProcessPoolExecutor() as executor:
                    # Map returns an iterator; we cast to list for immediate execution
                    return list(executor.map(func, data_list, chunksize=chunksize))
            
            # Fallback to serial execution if already in a worker process
            return [func(x) for x in data_list]
        
        return wrapper
    return decorator

@parallel(chunksize=500)
def super_heavy_task(x: int) -> int:
    """
    Example task: Performs heavy computation on an input.
    """
    # logic here (e.g., return x * x)
    return x
