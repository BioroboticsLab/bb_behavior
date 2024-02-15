import contextlib
import multiprocessing, threading
import inspect
import queue
import math
import numpy as np
import numba
import sys

def get_progress_bar_fn(what="auto"):
    if what == "auto":
        try:
            if 'ipykernel' in sys.modules:
                what = "tqdm_notebook"
        except:
            pass
        if what == "auto":
            try:
                import tqdm
                what = "tqdm"
            except:
                pass
        if what == "auto":
            what = None

    if what is None:
        return lambda x, **kwargs: x
    if what == "tqdm":
        import tqdm
        return tqdm.tqdm
    if what == "tqdm_notebook":
        import tqdm
        return tqdm.tqdm_notebook
    return what

class ParallelPipeline(object):
    
    jobs = None
    done = 0
    
    def __init__(self, jobs = [], n_thread_map = {}, queue_size_map = {}, thread_context_factory=None, unroll_sequentially=False):
        self.jobs = jobs
        self.n_thread_map = n_thread_map
        self.queue_size_map = queue_size_map
        self.use_threads = True
        self.thread_context_factory = thread_context_factory
        self.unroll_sequentially = unroll_sequentially

    def execute_sequentially(self, *first_args, **first_kwargs):
        def get_step_results(fn, *args, thread_context=None, **kwargs):
            def fn_wrap(*args, **kwargs):
                takes_thread_context = "thread_context" in inspect.signature(fn).parameters
                if takes_thread_context:
                    return fn(*args, thread_context=thread_context, **kwargs)
                else:
                    return fn(*args, **kwargs)
            is_generator = inspect.isgeneratorfunction(fn)
            results = None
            if is_generator:
                results = [r for r in fn_wrap(*args, **kwargs)]
            else:
                results = [fn_wrap(*args, **kwargs)]
            return results

        context = contextlib.nullcontext()
        if self.thread_context_factory is not None:
            context = self.thread_context_factory()
        with context as ctx:
            last_results = []
            for idx, step_fn in enumerate(self.jobs):
                step_results = []
                if idx == 0:
                    step_results = get_step_results(step_fn, *first_args, thread_context=ctx, **first_kwargs)
                else:
                    for arguments in last_results:
                        if arguments is None:
                            continue
                        if type(arguments) is tuple:
                            step_results.extend(get_step_results(step_fn, *arguments, thread_context=ctx))
                        else:
                            step_results.extend(get_step_results(step_fn, arguments, thread_context=ctx))
                last_results = [r for r in step_results if r is not None]

        return last_results

    def __call__(self, *args, **kwargs):
        if self.unroll_sequentially:
            return self.execute_sequentially(*args, **kwargs)

        def wrapper(inqueue, finished_barrier, outqueue, target, thread_context_factory=None):
            thread_context = None
            def _wrapped(thread_context=None):
                takes_thread_context = "thread_context" in inspect.signature(target).parameters
                if takes_thread_context and (thread_context is None) and (thread_context_factory is not None):
                    with thread_context_factory as ctx:
                         _wrapped(thread_context=ctx)
                         return

                is_generator = inspect.isgeneratorfunction(target)
                try:
                    if inqueue is not None:
                        call_scheme = None
                        while True:
                            job = inqueue.get()
                            if job is None:
                                # Queue finished. Put marker back.
                                inqueue.put(None)
                                break
                            if not call_scheme:
                                if not type(job) is tuple:
                                    call_scheme = lambda x: target(x, thread_context=thread_context)
                                else:
                                    call_scheme = lambda x: target(*x, thread_context=thread_context)
                            if not is_generator:
                                results = call_scheme(job)
                                if results is not None:
                                    outqueue.put(results)
                            else:
                                for results in call_scheme(job):
                                    if results is not None:
                                        outqueue.put(results)
                    else:
                        for results in target(*args, **kwargs):
                            if results is not None:
                                outqueue.put(results)
                    
                    thread_index = finished_barrier.wait()
                    if thread_index == 0:
                        outqueue.put(None)
                except Exception as e:
                    print ("Error at job: {}".format(str(target)))
                    print (str(e))
                    if not self.use_threads:
                        for queue in self.queues:
                            queue.close()
                    raise

            return _wrapped
        
        execution_unit = multiprocessing.Process if not self.use_threads else threading.Thread
        
        processes = []
        self.queues = []
        queue_style = multiprocessing.Queue if not self.use_threads else queue.Queue

        for i in range(len(self.jobs)):
            inqueue = self.queues[i - 1] if i >= 1 else None
            cnt = self.n_thread_map[i] if i in self.n_thread_map else 1
            if i == 0:
                cnt = 1
            if i == len(self.jobs) - 1: # last job? allow full output
                outqueue = queue_style()
            else:
                if i in self.queue_size_map:
                    queue_size = self.queue_size_map[i]
                else:
                    queue_size = max(2 * cnt, 16)
                outqueue = queue_style(queue_size)
            self.queues.append(outqueue)

            thread_barrier = threading.Barrier(cnt)
            for _ in range(cnt):
                thread_context = None
                if self.thread_context_factory is not None:
                    thread_context = self.thread_context_factory()
                processes.append(execution_unit(target=\
                            wrapper(inqueue=inqueue, finished_barrier=thread_barrier, outqueue=outqueue, target=self.jobs[i], thread_context_factory=thread_context)))
                processes[-1].start()
    
        for p in processes:
            p.join()
            
        results = []
        while not self.queues[-1].empty():
            val = self.queues[-1].get_nowait()
            if val is not None:
                results.append(val)
        return results

class FunctionCacher():
    """Threadsafe callable that itself takes a function and caches n results.
    Successive calls will retrieve the n results.
    Must be reset for the next n calls with reset().

    Example:
        def expensive_function():
            # Return something not-threadsafe.
            return foo
        cache = FunctionCacher(expensive_function, 4)
        for img in load_images():
            cache.reset()

            def thread_fun():
                cached_foo = cache()
                cached_foo(img)
                
            start_4_threads(thread_fun)

    """
    def __init__(self, fun, n=None, use_threads=True):
        """Takes a function and caches the results in a thread-safe way.

        Arguments:
            fun: callable
                Callable that returns a value to be cached. Should be thread-safe (if not, set 'n' in advance).
            n: int
                If given, 'fun' is called n times and the results are cached in advance.
            use_threads: bool
                Whether to use a threading Queue instead of a multiprocessing Queue.

        """
        self.queue_type = queue.Queue if use_threads else multiprocessing.Queue
        self.fun = fun
        self.available = self.queue_type()
        self.all = []

        if n is not None:
            self.reset(n)

    def reset(self, n=0):
        """Makes the cached results available again.
        (E.g. if you want to re-use 4 cached results, you have to call reset every 4 calls.

        Arguments:
            n: integer
            If given, makes sure that at least n cached results are available.
            (Can be omitted if specified during construction.)
        """
        if len(self.all) < n:
            for _ in range(len(self.all), n):
                self.all.append(self.fun())
        self.available = self.queue_type()
        for f in self.all:
            self.available.put(f)
    def get(self):
        """Returns a cached result. If no results are available, creates a new one.
        """
        if self.available.empty():
            self.available.put(self.fun())
            self.all.append(self.fun())
        return self.available.get()
    
    def __call__(self):
        """Same as get(). Returns a cached result.
        """
        return self.get()

def find_close_points(XY, max_distance, min_distance, distance_func=None, return_distances=False):
    """Takes a numpy array of positions and finds close indices.
    
    Arguments:
        XY: np.array of shape (N, 2) containing one x, y coordinate pair per row.
        max_distance: Only pairs closer than this will be returned.
        min_distance: Only pairs farther away than this will be returned.
        distance_func: callable
            Custom distance function taking two vectors as arguments.
        return_distances: bool
            Whether to return the distance as a second return value.
    Returns:
        numpy.array of shape (M, 2) containing M close pairs as tuples (i, j) where i and j are indices 0 <= i, j < N and i != j.
        If /return_distances/ is true, a second value with the calculated distances is returned.
    """
    pairs = None
    distances = None
    if not distance_func:
        import scipy.spatial
        tree = scipy.spatial.cKDTree(XY)
        close_points = tree.query_pairs(max_distance, output_type="ndarray")
        coords1 = XY[close_points[:, 0], :]
        coords2 = XY[close_points[:, 1], :]
        distance = np.linalg.norm(coords1 - coords2, axis=1)
        assert np.all(distance <= max_distance)
        pairs = close_points[distance >= min_distance, :]
        distances = distance[distance >= min_distance]
    else:
        import scipy.spatial.distance
        # Different approach when using a custom distance function.
        distances = scipy.spatial.distance.pdist(XY, metric=distance_func)
        valid = np.ones(shape=(distances.shape[0],), dtype=bool)
        if max_distance is not None:
            valid = valid & (distances <= max_distance)
        if min_distance is not None:
            valid = valid & (distances >= min_distance)
        # Get the pair indices back from the condensed form.
        # See https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs/14839010#14839010
        d = (1 + math.sqrt(1 + 8*len(distances)))/2 
        def row_col_from_condensed_index(d, i):
            b = 1 - 2 * d 
            x = np.floor((-b - np.sqrt(b**2 - 8 * i))/2)
            y = i + x * (b + x + 2)/2 + 1
            return np.stack((x, y), axis=1).astype(int)
        valid_squashed_indices = np.where(valid)[0]
        pairs = row_col_from_condensed_index(d, valid_squashed_indices)
        distances = distances[valid]

    if return_distances:
        pairs = (pairs, distances)
    return pairs

def prefetch_map(fun, iterable, max_workers=4, n_prefetched_results=None):
    """Behaves similarly to map() but usings multiprocessing to pre-fetch results.
    Unlike a MultiprocessingPool.map, this function limits the number of prefetched results.

    Arguments:
        fun: callable
            Function that is applied to every element in 'iterator'.
        iterable: Iterator
        max_workers: int
            Number of processes to spawn.
        n_prefetched_results: int
            Optional. Max number of elements in the queue.
    Returns:
        map(fun, iterable)
    """
    from collections import deque
    import concurrent.futures

    n_prefetched_results = n_prefetched_results or (2 * max_workers + 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        done = False
        futures = deque()
        try:
            for _ in range(2*n_prefetched_results):
                futures.append(executor.submit(fun, next(iterable)))
        except StopIteration:
            done = True

        while len(futures) > 0:
            next_future_result = futures.popleft().result()
            yield next_future_result

            if not done:
                try:
                    futures.append(executor.submit(fun, next(iterable)))
                except StopIteration:
                    done = True
