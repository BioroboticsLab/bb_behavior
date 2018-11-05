import multiprocessing, threading
import inspect
import queue
import numpy as np
import numba

class ParallelPipeline(object):
    
    jobs = None
    done = 0
    
    def __init__(self, jobs = [], n_thread_map = {}, thread_context_factory=None):
        self.jobs = jobs
        self.n_thread_map = n_thread_map
        self.use_threads = True
        self.thread_context_factory = thread_context_factory
        
    def __call__(self, *args, **kwargs):
        def wrapper(inqueue, finished_barrier, outqueue, target, thread_context_factory=None):
            thread_context = None
            def _wrapped(thread_context=None):
                if thread_context is None and thread_context_factory is not None:
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
        self.queue_type = queue.Queue() if use_threads else multiprocessing.Queue
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

def find_close_points(XY, max_distance, min_distance):
    """Takes a numpy array of positions and finds close indices.
    
    Arguments:
        XY: np.array of shape (N, 2) containing one x, y coordinate pair per row.
        max_distance: Only pairs closer than this will be returned.
        min_distance: Only pairs farther away than this will be returned.
        
    Returns:
        numpy.array of shape (M, 2) containing M close pairs as tuples (i, j) where i and j are indices 0 <= i, j < N and i != j.
    """
    import scipy.spatial
    tree = scipy.spatial.cKDTree(XY)
    close_points = tree.query_pairs(max_distance, output_type="ndarray")
    coords1 = XY[close_points[:, 0], :]
    coords2 = XY[close_points[:, 1], :]
    distance = np.linalg.norm(coords1 - coords2, axis=1)
    assert np.all(distance <= max_distance)
    return close_points[distance >= min_distance, :]
