from .. import db
from .. import utils
from collections import defaultdict
import numpy as np
import pandas
from numba import jit
import scipy.signal
import copy

def feature_normalize(trajectories, downscale_by=1000.0):
    traj0 = trajectories[0] # x, y, orientation, mask; (1, 4, N)
    center_index = traj0.shape[2] // 2
    mid_x, mid_y = traj0[0, 0, center_index], traj0[0, 1, center_index]
    mid_r = traj0[0, 2, center_index]

    c, s = np.cos(-mid_r), np.sin(-mid_r)
    R = np.matrix([[c, -s], [s, c]])

    for traj in trajectories:
        traj[0, 0, :] = (traj[0, 0, :] - mid_x) / downscale_by
        traj[0, 1, :] = (traj[0, 1, :] - mid_y) / downscale_by
        traj[0, :2, :] = R.dot(traj[0, :2, :])

        traj[0, 2, :] = traj[0, 2, :] - mid_r

def feature_angle_to_geometric(trajectories):
    for idx, traj in enumerate(trajectories):
        cos = np.cos(traj[:, 2:3, :])
        traj[0, 2, :] = np.sin(traj[0, 2, :])
        trajectories[idx] = np.concatenate((traj[:, :3, :], cos, traj[:, 3:4, :]), axis=1)

def feature_egomotion(trajectories):
    for idx, traj in enumerate(trajectories):
        traj[0, :4, :-1] = np.diff(traj[0, :4, :], axis=1)
        traj = traj[:, :, :-1]
        velocity = np.linalg.norm(traj[0, :2, :], axis=0)
        velocity = scipy.signal.medfilt(velocity, kernel_size=3)
        movement_direction = np.arctan2(traj[0, 1, :], traj[0, 0, :])
        forward_direction = np.arctan2(traj[0, 2, :], traj[0, 3, :])
        turn_direction = movement_direction - forward_direction
        traj[0, 0, :] = velocity
        traj[0, 1, :] = np.sin(turn_direction)
        cos = np.cos(turn_direction)[None, None, :]

        trajectories[idx] = np.concatenate((traj[:, :2, :], cos, traj[:, 2:5, :]), axis=1)

def trajectories_to_features(trajectories, feature_transformer):
    for ft in feature_transformer:
        ft(trajectories)
    if len(trajectories[0].shape) < 2:
        raise ValueError("Feature transformer must return an array with at least two dimensions with the first being of size 1 and the second being the features.")
    n_trajectories = len(trajectories)
    trajectories = np.concatenate(trajectories, axis=1)
    
    if feature_transformer:
        features = feature_transformer[-1].get_output()
        if (trajectories.shape[1] != n_trajectories * len(features)):
            raise ValueError("Feature transformers declared {}x{} output features ({}) but produced {}.".format(n_trajectories, len(features), features, trajectories.shape[1]))
    return trajectories

class FeatureTransform(object):
    _input = None
    _output = None
    _fun = None

    def __init__(self, fun, input=None, output=None, **kwargs):
        self._fun = fun
        self._input = input
        self._output = output or input
        self._kwargs = kwargs

    def validate_features(self, input):
        if input != self._input:
            raise ValueError("{} expects features {}. Got {}.".format(str(self._fun), self._input, input))
        return self._output or input

    @staticmethod
    def Normalizer(**kwargs):
        return FeatureTransform(fun=feature_normalize, input=("x", "y", "r", "mask"), **kwargs)
    @staticmethod
    def Angle2Geometric():
        return FeatureTransform(fun=feature_angle_to_geometric, input=("x", "y", "r", "mask"), output=("x", "y", "r_sin", "r_cos", "mask"))
    @staticmethod
    def Egomotion():
        return FeatureTransform(fun=feature_egomotion, input=("x", "y", "r_sin", "r_cos", "mask"), output=("vel", "x", "y", "r_sin", "r_cos", "mask"))
    def get_output(self):
        return self._output
    def __call__(self, x):
        if self._kwargs:
            return self._fun(x, **self._kwargs)
        return self._fun(x)

class DataReader(object):

    def __init__(self,
                    dataframe=None, sample_count=None,
                    from_timestamp=None, to_timestamp=None, bee_ids=None, use_hive_coords=False,
                    frame_margin=13, frame_margin_left=None, frame_margin_right=None, fps=3,
                    target_column="target", progress="tqdm_notebook", n_threads=16, feature_procs="auto",
                    chunk_frame_id_queries=False, verbose=False, Y_dtype=np.float32):
        self._dataframe = dataframe
        self._sample_count = sample_count

        self._from_timestamp = from_timestamp
        self._to_timestamp = to_timestamp
        self._bee_ids = bee_ids

        if (self._dataframe is not None or self._sample_count is not None) and (self._from_timestamp is not None or self._bee_ids is not None):
            raise ValueError("The dataframe/sample_count arguments are mutually exclusive with the timestamp/bee_ids arguments.")

        self._use_hive_coords = use_hive_coords
        self._fps = fps
        if frame_margin_left is None:
            self._frame_margin_left = frame_margin
            self._frame_margin_right = frame_margin
        else:
            self._frame_margin_left = frame_margin_left
            self._frame_margin_right = frame_margin_right
        self._target_column = target_column
        self._n_threads = n_threads
        self._tqdm = lambda x, **kwargs: x
        self._features = ("x", "y", "r", "mask")
        self._chunk_frame_id_queries = chunk_frame_id_queries
        self._verbose = verbose
        self._Y_dtype = Y_dtype

        self._bee_id_columns = None
        self._samples = None
        self._valid_sample_indices = None
        self._dataset = None
        self._groups = None
        self._has_loaded_features = False
        self._X = None
        self._Y = None
        self._train_X, self._train_Y = None, None
        self._test_X, self._test_Y = None, None
        self._train_groups, self._test_groups = None, None

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

    @classmethod
    def load_dataframe(self, path, **kwargs):
        df = pandas.read_hdf(path, key="df")
        return DataReader(dataframe=df, **kwargs)

    def save(self, path):
        import h5py
        h5f = h5py.File(path, 'w')
        h5f.create_dataset('X', data=self.X)
        h5f.create_dataset('Y', data=self.Y)
        if self.groups is not None:
            h5f.create_dataset('groups', data=self.groups)
        h5f.create_dataset('features', shape=(len(self._features), 1), dtype="S20", data=[f.encode("ascii", "ignore") for f in self._features])
        # Save original data along side the features.
        h5f.create_dataset('valid_sample_indices', data=self._valid_sample_indices)
        h5f.close()
        
        self.samples.to_hdf(path, "samples")

    @classmethod
    def load(self, path):
        datareader = DataReader(feature_procs=None)

        import h5py
        h5f = h5py.File(path,'r')
        datareader._X = h5f['X'][:]
        datareader._Y = h5f['Y'][:]
        try:
            datareader._groups = h5f["groups"][:]
        except:
            pass
        try:
            datareader._valid_sample_indices = h5f['valid_sample_indices'][:]
        except:
            pass
        datareader._features = tuple(f[0].decode("utf-8") for f in list(h5f["features"][:]))
        datareader._has_loaded_features = True
        h5f.close()

        try:
            datareader._samples = pandas.read_hdf(path, "samples")
        except Exception as e:
            print("Original samples were not stored. ({})".format(str(e)))

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
    def bee_id_to_trajectory(bee_id, frame_id=None, n_frames=None, n_frames_left=None, n_frames_right=None, frames=None,
        use_hive_coords=False, thread_context=None, detections=None, fps=None):
        if bee_id is not None:
            assert not pandas.isnull(bee_id)
            bee_id = int(bee_id)

        if frame_id is not None:
            frame_id = int(frame_id)
        traj = db.get_interpolated_trajectory(
                                    bee_id,
                                    frame_id=frame_id,
                                    n_frames=n_frames, n_frames_left=n_frames_left, n_frames_right=n_frames_right,
                                    fps=fps,
                                    frames=frames,
                                    interpolate=True, verbose=False,
                                    cursor=thread_context, cursor_is_prepared=True,
                                    use_hive_coords=use_hive_coords,
                                    detections=detections, confidence_threshold=0.5)
        if traj is None:
            return None
        traj, mask = traj
        if traj is None:
            return None

        if n_frames is None:
            if frames is not None:
                n_frames = len(frames)
            else:
                n_frames = len(detections)
        if n_frames is not None and np.sum(mask) < n_frames // 2:
            return None
        if not np.sum(np.isnan(traj)) == 0 or np.sum(np.isnan(traj)) == traj.size:
            print(traj.shape, traj.size, np.sum(np.isnan(traj)), flush=True)
            print(traj)
            assert False
        
        return np.hstack((traj, mask[:, np.newaxis]))

    def create_dataset(self):
        self._dataset = []
        self._valid_sample_indices = []

        # iter_samples and fetch_data_from_sample are used when a dataframe with events is provided.
        def iter_samples():
            for i in self._tqdm(range(self.samples.shape[0]), desc="Fetching data"):
                target = np.float32(self.samples[self._target_column].iloc[i]) if self._target_column is not None else 0.0
                yield (i,
                        [self.samples[bee].iloc[i] for bee in self._bee_id_columns], # bee_ids
                        self.samples.frame_id.iloc[i], # frame_id
                        target
                        )

        def fetch_data_from_sample(index, bee_ids, frame_id, target, thread_context=None):
            args = [(bee_id, frame_id) for bee_id in bee_ids]
            data = [DataReader.bee_id_to_trajectory(*a,
                    n_frames_left=self._frame_margin_left, n_frames_right=self._frame_margin_right,
                    fps=self._fps,
                    use_hive_coords=self._use_hive_coords, thread_context=thread_context
                    ) for a in args]
            return index, data, target, bee_ids
        # Used when chunk_frame_id_queries is true.
        # This is an optimization to reduce the number of calls to the database and is likely only helpful when
        # the data contains many events from the same (or neighboured) frame ids.
        def iter_samples_chunk_frames():
            # Allow some leeway in the frame timestamps - make sure that we have the correct number later.
            margin_in_seconds_left = self._frame_margin_left / self._fps + 1
            margin_in_seconds_right = self._frame_margin_right / self._fps + 1
            all_frame_ids = set(self.samples.frame_id.values)

            with db.DatabaseCursorContext(application_name="Batch frame ids") as cursor:
                frame_metadata = db.get_frame_metadata(frames=all_frame_ids, cursor=cursor, cursor_is_prepared=True)
                # First, fetch all the required neighbouring frames.
                for frame_id, cam_id, timestamp in self._tqdm(frame_metadata[["frame_id", "cam_id", "timestamp"]].itertuples(index=False), total=frame_metadata.shape[0]):
                    ts_from, ts_to = timestamp - margin_in_seconds_left, timestamp + margin_in_seconds_right
                    neighbour_frames = db.get_frames(cam_id, ts_from, ts_to, cursor=cursor, cursor_is_prepared=True)
                    if len(neighbour_frames) < self.timesteps:
                        if self._verbose:
                            print("Not enough neighbour frames for frame {} (found {}) (1)".format(frame_id, len(neighbour_frames)))
                        continue
                    # We have potentially requested a bit more frames than we need. Filter.
                    # First, find out where our target frame actually is.
                    middle_index = None
                    for i in range(len(neighbour_frames)):
                        if neighbour_frames[i][1] == frame_id:
                            middle_index = i
                            break
                    if middle_index is None:
                        if self._verbose:
                            print("Center frame ID not in return value of db.get_frames.")
                        continue
                    # Then cut the frames around our target frame based on index (instead of timestamp).
                    neighbour_frames = neighbour_frames[(middle_index - self._frame_margin_left):(middle_index + self._frame_margin_right + 1)]
                    if len(neighbour_frames) != self.timesteps:
                        if self._verbose:
                            print("Not enough neighbour frames for frame {} (found {}) (2)".format(frame_id, len(neighbour_frames)))
                        continue

                    matching_samples = self.samples.frame_id == frame_id
                    sample_idx = np.where(matching_samples)[0]
                    yield self.samples[matching_samples], sample_idx, neighbour_frames

        def fetch_trajectory_data_for_frame_samples(samples, samples_idx, neighbour_frames, thread_context):    
            cursor = thread_context

            bee_ids = set()
            for col in self._bee_id_columns:
                bee_ids |= set(map(int, samples[col].values))
            query = "get_all_bee_pixel_detections_for_frames"
            if self._use_hive_coords:
                query = "get_all_bee_hive_detections_for_frames"
            frame_ids = [int(f[1]) for f in neighbour_frames]
            cursor.execute("EXECUTE get_all_bee_hive_detections_for_frames(%s, %s)", (frame_ids, list(bee_ids)))
            bee_to_traj = defaultdict(list)
            # Sort into bee ids.
            for row in cursor.fetchall():
                bee_to_traj[int(row[0])].append(row[1:])
            # And, for all the data, generate consistent trajectories.
            for bee_id in bee_to_traj:
                bee_to_traj[bee_id] = db.get_consistent_track_from_detections(neighbour_frames, bee_to_traj[bee_id])

            # Now we have a lookup table for all bees and can go through the samples.
            for idx in range(samples.shape[0]):
                bee_ids = [int(samples[bee].iloc[idx]) for bee in self._bee_id_columns]
                target = np.float32(samples[self._target_column].iloc[idx]) if self._target_column is not None else 0.0
                yield (samples_idx[idx],
                    [DataReader.bee_id_to_trajectory(bee_id=None, detections=bee_to_traj[bee_id]) for bee_id in bee_ids],
                    target, bee_ids)

        # the timespan functions are used when begin & end timestamps are provided.
        def iter_cam_ids():
            for cam_id in range(4):
                yield cam_id

        def fetch_cam_id_timespan_data(cam_id, **kwargs):
            margin_in_seconds_left = self._frame_margin_left / self.fps
            margin_in_seconds_right = self._frame_margin_right / self.fps
            frames = db.get_frames(cam_id=cam_id, ts_from=self._from_timestamp - margin_in_seconds_left, ts_to=self._to_timestamp + margin_in_seconds_right)
            if self._verbose:
                print("{} frames found for cam id {} for the provided timespan.".format(len(frames), cam_id))
            if len(frames) == 0:
                return None
            return frames

        def split_timespan_frames(frames, **kwargs):
            trajectories = []
            for bee_id in self._bee_ids:
                traj = DataReader.bee_id_to_trajectory(bee_id=bee_id, frames=frames, use_hive_coords=self._use_hive_coords)
                # If at least one of the bees has no data available, we can't continue.
                if traj is None:
                    if self._verbose:
                        print("No data found for bee {} in the frames {}".format(bee_id, frames))
                    return None
                trajectories.append(traj)

            for index in self._tqdm(range(self._frame_margin_left, len(frames) - self._frame_margin_right - 1), desc="Calculating camera data", leave=False):
                timestamp, frame_id, cam_id = frames[index]
                
                subtrajs = [traj[(index - self._frame_margin_left):(index + self._frame_margin_right + 1), :] for traj in trajectories]
                
                incomplete_trajs = np.any([np.sum(traj[:, -1]) < (self._frame_margin_left + self._frame_margin_right) // 4 for traj in subtrajs])
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
        execute_sequentially = self._n_threads == 0
        if self.samples is not None:
            if self._verbose:
                print("Generating data for samples of the provided dataframe.")
            data_source = [iter_samples, fetch_data_from_sample]
            if self._chunk_frame_id_queries:
                data_source = [iter_samples_chunk_frames, fetch_trajectory_data_for_frame_samples]
                if self._verbose:
                    print("Using chunked frames optimization.")
            pipeline = utils.processing.ParallelPipeline(jobs=data_source + data_processing,
                                        n_thread_map={1:self._n_threads}, thread_context_factory=make_thread_context,
                                        unroll_sequentially=execute_sequentially)
            pipeline()
        else:
            assert self._from_timestamp is not None
            assert self._to_timestamp is not None
            assert self._bee_ids is not None
            if self._verbose:
                print("Fetching data for the provided bee ids and timespan.")
            pipeline = utils.processing.ParallelPipeline(jobs=[iter_cam_ids, fetch_cam_id_timespan_data, split_timespan_frames] + data_processing,
                                        n_thread_map={1:4}, unroll_sequentially=execute_sequentially)
            
            self._dataframe = []
            pipeline()
            self._dataframe = pandas.DataFrame(self._dataframe)
            if self._verbose:
                print("Generated {} trajectory samples.".format(len(self._dataset)))

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
        if self._feature_procs is None:
            raise ValueError("feature_procs must not be None (hint: use 'auto' or load features from a file).")
        self._X = []
        self._Y = np.zeros(shape=(len(self.dataset), 1), dtype=self._Y_dtype)

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
        return (self._frame_margin_left + self._frame_margin_right) + 1
    @property
    def n_features(self):
        if type(self._X) == np.ndarray:
            return self._X.shape[1]
        return len(self._features) * len(self._bee_id_columns)

    def create_train_test_split(self, test_size=0.2, predefined_train_groups=None):
        import sklearn.model_selection

        idx = None
        if self.groups is not None:
            if predefined_train_groups is not None:
                idx = [None, None]
                idx[0] = np.isin(self.groups, predefined_train_groups)
                idx[1] = ~idx[0]
            else:
                idx = next(sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=test_size).split(self.X, y=self.Y, groups=self.groups))
        else:
            idx = next(sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(self.X, y=self.Y))
        self._train_X, self._train_Y, self._test_X, self._test_Y = \
            self.X[idx[0]], self.Y[idx[0]],\
            self.X[idx[1]], self.Y[idx[1]]

        self._train_indices = np.where(idx[0])[0]
        self._test_indices = np.where(idx[1])[0]

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
