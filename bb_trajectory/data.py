from . import db
from . import utils
import numpy as np
import pandas
from numba import jit
import copy

@jit
def feature_normalize(trajectories):
    traj0 = trajectories[0] # x, y, orientation, mask; (1, 4, N)
    center_index = traj0.shape[2] // 2
    mid_x, mid_y = traj0[0, 0, center_index], traj0[0, 1, center_index]
    mid_r = traj0[0, 2, center_index]

    c, s = np.cos(-mid_r), np.sin(-mid_r)
    R = np.matrix([[c, -s], [s, c]])

    for traj in trajectories:
        traj[0, 0, :] = (traj[0, 0, :] - mid_x) / 1000.0
        traj[0, 1, :] = (traj[0, 1, :] - mid_y) / 1000.0
        traj[0, :2, :] = R.dot(traj[0, :2, :])

        traj[0, 2, :] = traj[0, 2, :] - mid_r

@jit
def feature_angle_to_geometric(trajectories):
    for idx, traj in enumerate(trajectories):
        cos = np.cos(traj[:, 2:3, :])
        traj[0, 2, :] = np.sin(traj[0, 2, :])
        trajectories[idx] = np.concatenate((traj[:, :3, :], cos, traj[:, 3:4, :]), axis=1)

@jit
def trajectories_to_features(trajectories, feature_transformer):
    for ft in feature_transformer:
        ft(trajectories)
    if len(trajectories[0].shape) < 2:
        raise ValueError("Feature transformer must return an array with at least two dimensions with the first being of size 1 and the second being the features.")
    return np.concatenate(trajectories, axis=1)

class FeatureTransform(object):
    _input = None
    _output = None
    _fun = None

    def __init__(self, fun, input=None, output=None):
        self._fun = fun
        self._input = input
        self._output = output or input

    def validate_features(self, input):
        if input != self._input:
            raise ValueError("{} expects features {}. Got {}.".format(str(self._fun), self._input, input))
        return self._output or input

    @staticmethod
    def Normalizer():
        return FeatureTransform(fun=feature_normalize, input=("x", "y", "r", "mask"))
    @staticmethod
    def Angle2Geometric():
        return FeatureTransform(fun=feature_angle_to_geometric, input=("x", "y", "r", "mask"), output=("x", "y", "r_sin", "r_cos", "mask"))

    def __call__(self, x):
        return self._fun(x)

class DataReader(object):

    _dataframe = None
    _sample_count = None
    _from_timestamp = None
    _to_timestamp = None
    _bee_ids = None

    _frame_margin = None
    _n_threads = None

    _feature_procs = None

    # Internal states.
    _bee_id_columns = None

    _samples = None
    _valid_sample_indices = None

    _dataset = None
    _groups = None

    _has_loaded_features = False
    _X = None
    _Y = None
    _train_X, _train_Y = None, None
    _test_X, _test_Y = None, None
    _train_groups, _test_groups = None, None

    def __init__(self,
                    dataframe=None, sample_count=None,
                    from_timestamp=None, to_timestamp=None, bee_ids=None,
                    frame_margin=13, target_column="target", progress="tqdm_notebook", n_threads=16, feature_procs="auto"):
        self._dataframe = dataframe
        self._sample_count = sample_count

        self._from_timestamp = from_timestamp
        self._to_timestamp = to_timestamp
        self._bee_ids = bee_ids

        if (self._dataframe is not None or self._sample_count is not None) and (self._from_timestamp is not None or self._bee_ids is not None):
            raise ValueError("The dataframe/sample_count arguments are mutually exclusive with the timestamp/bee_ids arguments.")

        self._frame_margin = frame_margin
        self._target_column = target_column
        self._n_threads = n_threads
        self._tqdm = lambda x, **kwargs: x
        self._features = ("x", "y", "r", "mask")
        if progress == "tqdm_notebook":
            import tqdm
            self._tqdm = tqdm.tqdm_notebook
        elif progress == "tqdm":
            import tqdm
            self._tqdm = tqdm.tqdm

        self._feature_procs = feature_procs

        if dataframe is not None:
            self._bee_id_columns = [c for c in self._dataframe.columns if c.startswith("bee_id")]

        if self._feature_procs == "auto":
            self._feature_procs = [FeatureTransform.Normalizer(), FeatureTransform.Angle2Geometric()]

        if self._feature_procs is not None:    
            for ft in self._feature_procs:
                self._features = ft.validate_features(self._features)

            

    def save_dataframe(self, path):
        self._dataframe.to_hdf(path, "df")

    @classmethod
    def load_dataframe(self, path, **kwargs):
        df = pandas.read_hdf(path, key="df")
        return DataReader(dataframe=df, **kwargs)
    
    def save_features(self, path):
        import h5py
        h5f = h5py.File(path, 'w')
        h5f.create_dataset('X', data=self.X)
        h5f.create_dataset('Y', data=self.Y)
        h5f.create_dataset('groups', data=self.groups)
        h5f.create_dataset('features', shape=(len(self._features), 1), dtype="S20", data=[f.encode("ascii", "ignore") for f in self._features])
        h5f.close()

    @classmethod
    def load_features(self, path):
        datareader = DataReader(feature_procs=None)

        import h5py
        h5f = h5py.File(path,'r')
        datareader._X = h5f['X'][:]
        datareader._Y = h5f['Y'][:]
        datareader._groups = h5f["groups"][:]
        datareader._features = tuple(f[0].decode("utf-8") for f in list(h5f["features"][:]))
        datareader._has_loaded_features = True

        return datareader
    
    def has_loaded_features(self):
        return self._has_loaded_features

    @classmethod
    def from_XY(self, X, Y=None, groups=None, features=None):
        datareader = DataReader(dataframe=None)
        datareader._X = X
        datareader._Y = Y
        datareader._groups = groups
        datareader._features = features
        datareader._has_loaded_features = True

        return datareader

    @property
    def samples(self):
        if self._samples is None:
            if self._dataframe is None:
                return None
            if self._sample_count is None:
                self._samples = self._dataframe
            else:
                self._samples = self._dataframe.sample(self._sample_count)
        return self._samples

    @property
    def dataset(self):
        if self._dataset is None:
            self.create_dataset()
        return self._dataset

    @staticmethod
    def bee_id_to_trajectory(bee_id, frame_id=None, n_frames=None, frames=None, thread_context=None):
        assert not np.isnan(bee_id)

        if frame_id is not None:
            frame_id = int(frame_id)
        traj = db.get_interpolated_trajectory(
                                    int(bee_id),
                                    frame_id=frame_id, n_frames=n_frames,
                                    frames=frames,
                                    interpolate=True, verbose=False,
                                    cursor=thread_context, cursor_is_prepared=True)
        if traj is None:
            return None
        traj, mask = traj

        n_frames = n_frames or len(frames)
        if n_frames is not None and np.sum(mask) < n_frames // 2:
            return None
        assert np.sum(np.isnan(traj)) == 0 or np.sum(np.isnan(traj)) == traj.size
        
        return np.hstack((traj, mask[:, np.newaxis]))

    def create_dataset(self):
        self._dataset = []
        self._valid_sample_indices = []

        # iter_samples and fetch_data_from_sample are used when a dataframe with events is provided.
        def iter_samples():
            for i in self._tqdm(range(self.samples.shape[0]), desc="Fetching data"):
                yield (i,
                        [self.samples[bee].iloc[i] for bee in self._bee_id_columns], # bee_ids
                        self.samples.frame_id.iloc[i], # frame_id
                        np.float32(self.samples[self._target_column].iloc[i]) # target
                        )

        def fetch_data_from_sample(index, bee_ids, frame_id, target, thread_context=None):
            args = [(bee_id, frame_id, self._frame_margin) for bee_id in bee_ids]
            data = [DataReader.bee_id_to_trajectory(*a, thread_context=thread_context) for a in args]
            return index, data, target, bee_ids

        # the timespan functions are used when begin & end timestamps are provided.
        def iter_cam_ids():
            for cam_id in range(4):
                yield cam_id

        def fetch_cam_id_timespan_data(cam_id, **kwargs):
            margin_in_seconds = self._frame_margin * 0.33
            frames = db.get_frames(cam_id=cam_id, ts_from=self._from_timestamp - margin_in_seconds, ts_to=self._to_timestamp + margin_in_seconds)
            if len(frames) == 0:
                return None
            return frames

        def split_timespan_frames(frames, **kwargs):
            trajectories = []
            for bee_id in self._bee_ids:
                traj = DataReader.bee_id_to_trajectory(bee_id=bee_id, frames=frames)
                # If at least one of the bees has no data available, we can't continue.
                if traj is None:
                    return None
                trajectories.append(traj)

            for index in self._tqdm(range(self._frame_margin, len(frames) - self._frame_margin - 1), desc="Calculating camera data", leave=False):
                timestamp, frame_id, cam_id = frames[index]
                
                subtrajs = [traj[(index - self._frame_margin):(index + self._frame_margin + 1), :] for traj in trajectories]
                
                incomplete_trajs = np.any([np.sum(traj[:, -1]) < self._frame_margin // 2 for traj in subtrajs])
                if incomplete_trajs:
                    continue

                sample = dict(frame_id=frame_id, timestamp=timestamp)
                for i, bee_id in enumerate(self._bee_ids):
                    sample["bee_id{}".format(i)] = bee_id
                self._dataframe.append(sample)
                yield len(self._dataframe) - 1, subtrajs, np.nan, self._bee_ids

        def validate_data(index, data, target, bee_ids, **kwargs):
            if any([d is None for d in data]) or len(data) < len(bee_ids):
                return None
            assert len(data) == len(bee_ids)
            for i, d in enumerate(data):
                if d.shape[0] != self.timesteps:
                    return None
                data[i] = d.T[np.newaxis, :, :]
            return index, data, target

        def store_data(index, data, target, **kwargs):
            self._dataset.append((data, target))
            self._valid_sample_indices.append(index)

        def make_thread_context():
            return db.DatabaseCursorContext(application_name="DataReader")
        
        data_processing = [validate_data, store_data]
        if self.samples is not None:
            pipeline = utils.ParallelPipeline(jobs=[iter_samples, fetch_data_from_sample] + data_processing,
                                        n_thread_map={1:self._n_threads}, thread_context_factory=make_thread_context)
            pipeline()
        else:
            assert self._from_timestamp is not None
            assert self._to_timestamp is not None
            assert self._bee_ids is not None
            pipeline = utils.ParallelPipeline(jobs=[iter_cam_ids, fetch_cam_id_timespan_data, split_timespan_frames] + data_processing,
                                        n_thread_map={1:4})
            
            self._dataframe = []
            pipeline()
            self._dataframe = pandas.DataFrame(self._dataframe)

        if "group" in self.samples.columns:
            self._groups = self.samples.group.values[self._valid_sample_indices]

    @property
    def X(self):
        if self._X is None:
            self.create_features()
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self.create_features()
        return self._Y
    @property
    def groups(self):
        return self._groups

    def create_features(self):
        if len(self.dataset) == 0:
            raise ValueError("Need a valid dataset.")
        self._X = []
        self._Y = np.zeros(shape=(len(self.dataset), 1), dtype=np.float32)

        for idx, (data, target) in enumerate(self._tqdm(self.dataset, desc="Calculating features")):
            self._X.append(trajectories_to_features(copy.deepcopy(data), self._feature_procs))
            self._Y[idx] = target
        assert len(self._X) == len(self.dataset)
        self._X = np.concatenate(self._X, axis=0)

    @property
    def Yonehot(self):
        import sklearn.preprocessing
        return sklearn.preprocessing.OneHotEncoder(n_values=2, sparse=False).fit_transform(self.Y)
    @property
    def timesteps(self):
        if type(self._X) == np.ndarray:
            return self._X.shape[2]
        return 2 * self._frame_margin + 1
    @property
    def n_features(self):
        if type(self._X) == np.ndarray:
            return self._X.shape[1]
        return len(self._features) * len(self._bee_id_columns)

    def create_train_test_split(self, test_size=0.2):
        import sklearn.model_selection

        idx = None
        if self.groups is not None:
            idx = next(sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=test_size).split(self.X, y=self.Y, groups=self.groups))
        else:
            idx = next(sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(self.X, y=self.Y))
        self._train_X, self._train_Y, self._test_X, self._test_Y = \
            self.X[idx[0]], self.Y[idx[0]],\
            self.X[idx[1]], self.Y[idx[1]]
        if self._groups is not None:
            self._train_groups = self._groups[idx[0]]
            self._test_groups = self._groups[idx[1]]

    @property
    def train_X(self):
        if self._train_X is None:
            self.create_train_test_split()
        return self._train_X

    @property
    def train_Y(self):
        if self._train_Y is None:
            self.create_train_test_split()
        return self._train_Y

    @property
    def train_groups(self):
        if self._train_groups is None:
            self.create_train_test_split()
        return self._train_groups

    @property
    def test_X(self):
        if self._test_X is None:
            self.create_train_test_split()
        return self._test_X

    @property
    def test_Y(self):
        if self._test_Y is None:
            self.create_train_test_split()
        return self._test_Y

    @property
    def test_groups(self):
        if self._test_groups is None:
            self.create_train_test_split()
        return self._test_groups

    def cross_validate(self, make_model_fun, train_model_fun=None, scorer=None, *args, **kwargs):

        if scorer is None:
            import sklearn.metrics
            scorer = sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score, needs_proba=True)
        if train_model_fun is None:
            train_model_fun = lambda m, x, y: m.fit(x, y)

        scores = []
        splits = None
        import sklearn.model_selection
        if self.groups is not None:
            splits = sklearn.model_selection.GroupShuffleSplit(n_splits=3, test_size=0.2).split(self.X, y=self.Y, groups=self.groups)
        else:
            splits = sklearn.model_selection.StratifiedShuffleSplit(n_splits=3, test_size=0.2).split(self.X, y=self.Y)
        for idx in splits:
            _train_X, _train_Y, _test_X, _test_Y = \
                self.X[idx[0]], self.Y[idx[0]],\
                self.X[idx[1]], self.Y[idx[1]]

            model = make_model_fun()
            train_model_fun(model, _train_X, _train_Y)
            
            try:
                score = scorer(model, _test_X, _test_Y)
                scores.append(score)
            except:
                pass
       
        return scores
