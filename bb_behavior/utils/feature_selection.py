import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import scipy
import pandas
import scipy.stats
import joblib
import copy
import queue

import sklearn.metrics
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.externals.joblib

class StochasticFeatureSelection(object):
	
	def __init__(self, model=None, scoring=None,
				 dropped_fraction=0.5, step=256, verbose=1, cv=10, feature_names=None, iterations=-1,
				 confidence_sigma=1.645, min_n_features=5, min_rows_required_to_drop=32, jupyter=False,
				 retain_experiments=False, n_jobs=-1, cv_n_jobs=-1, iteration_callback=None, reuse_available_information=False,
				 static_train_test_split=False):
		if scoring is None:
			self.scoring = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error)
		else:
			self.scoring = scoring
		self.model = model
		self.dropped_fraction = dropped_fraction
		self.step = step
		self.verbose = verbose
		self.cv = cv
		self.iterations=iterations
		self.min_n_features = min_n_features
		self.min_rows_required_to_drop = min_rows_required_to_drop
		self.confidence_sigma = confidence_sigma
		self.available_feature_mask = None
		self.retain_experiments = retain_experiments
		self.n_jobs = n_jobs
		self.cv_n_jobs = cv_n_jobs
		
		self.iteration_callback = iteration_callback
		self.reuse_available_information = reuse_available_information
		self.static_train_test_split = static_train_test_split
		
		self.fixed_feature_names = feature_names
		self.jupyter = jupyter
		self.reset()
	
	def printwrap(self, text):
		try:
			from tqdm import tqdm
			tqdm.write(text)
		except:
			print(text)
	
	def iterwrap(self, iterator):
		if self.verbose != 0:
			try:
				if self.jupyter:
					from tqdm import tqdm_notebook
					return tqdm_notebook(iterator)
				else:
					from tqdm import tqdm
					return tqdm(iterator)
			except:
				return iterator
		return iterator
	
	def reset(self):
		self.feature_names = self.fixed_feature_names
		self.dropped_features = []
		self.cv_p_value = []
		self.cv_all_median = []
		self.cv_all_mean = []
		self.cv_all_std = []
		self.cv_significance = []
		
		self.stochastic_subset_data = []
		self.cross_validation_data = []
		
		self.all_p_values = []
		
	def fit_partial(self, X, y):
		return self.fit(X, y, reset=False)
	
	def fit(self, X, y, groups=None, reset=True):
		# When doing a fresh start, initialize a few variables.
		if reset and not self.reuse_available_information:
			self.reset()
			self.available_feature_mask = np.ones(X.shape[1], dtype=np.bool)
		else:
			assert self.available_feature_mask is not None
			assert self.available_feature_mask.shape[0] == X.shape[1]
		
		def do_split(_X, _y, _groups, only_one=False):
			if groups is None:
				if only_one:
					return next(sklearn.model_selection.StratifiedShuffleSplit().split(_X, _y))
				else:
					return list(sklearn.model_selection.StratifiedShuffleSplit(n_splits=self.cv).split(_X, _y))
			else:
				if only_one:
					return next(sklearn.model_selection.GroupShuffleSplit().split(_X, _y, groups=_groups))
				else:
					return list(sklearn.model_selection.GroupShuffleSplit(n_splits=self.cv).split(_X, _y, groups=_groups))
			assert False

		# Dummy feature names if none are provided.
		if self.feature_names is None:
			self.feature_names = ["f{:06d}".format(i) for i in range(X.shape[1])]
		# Holds the cv runs with the randomly dropped features - will be filled to self.step before making decisions.
		current_experiment_set = []
		
		# Initial train-validation-split.
		# This constitutes the per-dropout-experiment sets.
		initial_train_test_split = None
		def roll_train_test_split():
			nonlocal initial_train_test_split
			def roll():
				train_idx, test_idx = do_split(X, y, groups, only_one=True)
				return X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
				
			if self.static_train_test_split:
				if initial_train_test_split is None:
					initial_train_test_split = roll()
				return initial_train_test_split
			else:
				return roll()
		if self.static_train_test_split: # Roll here and not lazily because of thread-safety.
			roll_train_test_split()
			assert initial_train_test_split is not None
			
		# These are the splits for the CV runs after one feature was dropped.
		cross_validation_splits = do_split(X, y, groups)
		
		# Setup model zoo to reduce per-iteration overhead.
		model_zoo = None
		number_of_jobs = self.n_jobs
		if number_of_jobs is None:
			number_of_jobs = 1
		elif number_of_jobs == -1:
			import multiprocessing
			number_of_jobs = multiprocessing.cpu_count()
		if number_of_jobs > 1:
			model_zoo = [copy.deepcopy(self.model) for i in range(number_of_jobs + 1)]
		else:
			model_zoo = [self.model]
		model_queue = queue.Queue()
		for model in model_zoo:
			model_queue.put(model)
		
		# Each iteration will drop one feature.
		n_iterations = self.iterations if (self.iterations != -1) else (X.shape[1] - len(self.dropped_features))
		for runs in self.iterwrap(range(n_iterations)):
			feature_count = self.available_feature_mask.sum()
			if feature_count <= 1:
				last_feature_index = np.where(self.available_feature_mask == True)[0]
				assert len(last_feature_index) == 1
				last_feature_name = self.feature_names[last_feature_index[0]]
				self.dropped_features.append(last_feature_name)
				continue
			absolute_dropout_number = int(self.dropped_fraction * float(feature_count))
			if absolute_dropout_number < 1:
				absolute_dropout_number = 1
				
			def run_experiment(iteration):
				selected_features = np.ones(feature_count, dtype=np.bool)
				selected_features[np.random.choice(selected_features.shape[0], absolute_dropout_number, replace=False)] = False
				
				assert self.available_feature_mask.sum() == selected_features.shape[0]
				
				final_mask = np.zeros(X.shape[1], dtype=np.bool)
				final_mask[self.available_feature_mask] = selected_features
				
				model = model_queue.get()
				X_train, X_test, y_train, y_test = roll_train_test_split()
				model.fit(X_train[:,final_mask], y_train)
				metric_results = self.scoring(model, X_test[:,final_mask], y_test)
				model_queue.put(model)
				
				if np.isnan(metric_results):
					if self.verbose != 0:
						fs = [self.feature_names[c] for c in np.where(self.available_feature_mask)[0]]
						self.printwrap("Iteration {:3d}, set {:}: retaining {:} results in NaN".format(runs, iteration, ",".join(fs)))
					metric_results = 0.0
					return
				
				r = dict(zip(self.feature_names, final_mask))
				r["metric"] = metric_results
				return r
			
			new_experiments = joblib.Parallel(n_jobs=number_of_jobs, backend="threading")(\
								(joblib.delayed(run_experiment, check_pickle=False)(i)) for i in range(self.step - len(current_experiment_set)))
			current_experiment_set = current_experiment_set + [new_exp for new_exp in new_experiments if not new_exp is None]
	
				
			df = pandas.DataFrame(current_experiment_set)
			before_median = np.median(df["metric"].values)
			# self.stochastic_subset_data.append(df["metric"].values)
			
			# As a performance optimization, fetch a few results beforehand.
			metric_results = df["metric"].values
			df.drop(["metric"], axis=1, inplace=True)
			features_active = df.values
			column_names = df.columns
			feature_index_mapping = [column_names.get_loc(f_name) for f_name in self.feature_names]
			
			# Drop exactly one feature.
			to_drop = None
			worst_p_value = None
			current_p_values = []
			for f_i in np.where(self.available_feature_mask)[0]:
				f_name = self.feature_names[f_i]
				df_index = feature_index_mapping[f_i]
				feature_active_idx = features_active[:, df_index]
				metric_results_active   = metric_results[feature_active_idx]
				metric_results_inactive = metric_results[~feature_active_idx]
				
				if metric_results_active.shape[0] < self.min_rows_required_to_drop:
					print ("Try increasing the number of iterations per step please!")
					continue
				# Do a simple sign-test on the median.
				other_population_median = np.median(metric_results_inactive)
				signs = metric_results_active < other_population_median
				p_value = scipy.stats.binom_test(signs.sum(), signs.shape[0], p=0.5, alternative="greater")
				current_p_values.append(p_value)
				
				if self.verbose >= 2:
					self.printwrap("\t {:30s}: {:5.4f} -> {:5.4f}?\t p={:2.4f} \t\t({:4d}/{:4d})".format(f_name,
						np.median(metric_results_active), other_population_median, p_value, signs.sum(), signs.shape[0]))

				if (worst_p_value is None) or (p_value < worst_p_value):
					worst_p_value = p_value
					to_drop = f_i
						
			if (to_drop is None):
				self.step += 10
				if self.verbose != 0:
					self.printwrap("Iteration {:3d}: no feature to be dropped - step at {:}".format(runs, self.step))
				continue
			self.all_p_values.append(current_p_values)
			
			to_drop_named = self.feature_names[to_drop]
			self.dropped_features.append(to_drop_named)
			self.available_feature_mask[to_drop] = False
			
			# Keep all experiments where the feature had already been dropped.
			if self.retain_experiments:
				retained_indices = np.where(df[to_drop_named] == False)[0]
				assert (len(retained_indices) == (df[to_drop_named].values == False).sum())
				
				retained_experiments = []
				for idx in retained_indices:
					retained_experiments.append(current_experiment_set[idx])
				current_experiment_set = retained_experiments
			else:
				current_experiment_set = []
			
			# Now do a final cross-validation with the remaining features.
			# Once with all the samples.
			with sklearn.externals.joblib.parallel_backend('threading'):
				cv_results = sklearn.model_selection.cross_val_score(self.model, X[:,self.available_feature_mask], y,
									cv=cross_validation_splits, scoring=self.scoring, n_jobs=self.cv_n_jobs)
			last_best_cv_median_idx = None
			if len(self.cv_all_median) > 0:
				last_best_cv_median_idx = np.argmax(self.cv_all_median)
			cv_median = np.median(cv_results)
			self.cv_all_median.append(cv_median)
			self.cv_all_mean.append(cv_results.mean())
			self.cv_all_std.append(cv_results.std())
			self.cross_validation_data.append(cv_results)
			
			# Now check whether the new subset is significantly better than the last best.
			p_value = np.nan
			significance_index = 0
			significance_marker = "  "
			if last_best_cv_median_idx is not None:
				try:
					stat, p, m, table = scipy.stats.median_test(cv_results, self.cross_validation_data[last_best_cv_median_idx])
					p_value = p
					
					if cv_median > self.cv_all_median[last_best_cv_median_idx]:
						if p <= 0.05:
							significance_marker = "**"
							significance_index = 3
						else:
							significance_marker = "*"
							significance_index = 2
					else:
						if p > 0.1:
							significance_marker = "+"
							significance_index = 1
				except:
					significance_marker = "?"
					significance_index = -1
			self.cv_significance.append(significance_index)
			self.cv_p_value.append(p_value)
			
			if self.verbose >= 1:
				self.printwrap("Iteration {:3d}: CV @ {:5.3f}{:} \t\t[dropped {:}]".format(
						runs, np.median(cv_results), significance_marker, to_drop_named))
				sys.stdout.flush()
			
			if self.iteration_callback is not None:
				self.iteration_callback(self, runs)
			
		return self

	def get_support(self):
		drop_until = np.where(np.array(self.cv_significance) == 2)[0][-1]
		kept_features = set(self.dropped_features[drop_until:])
		support_mask = [(feature in kept_features) for feature in self.feature_names]
		support_mask = np.array(support_mask)
		assert support_mask.shape[0] == len(self.dropped_features)
		return support_mask
				
	def transform(self, X, y=None):
		return X[:, self.get_support()]
		
	# Serialization.
	# Serializing the class with these members allows for the continuation of training,
	# after it has been initialized with the same (not necessarily, though) constructor
	# arguments as in the original run.
	# The saved data should be the minimal subset necessary and thus smaller than pickling
	# the whole model. Also it should prove to be more stable over different versions.
	
	def to_dict(self):
		pars = {
			"feature_names": self.feature_names,
			"dropped_features": self.dropped_features,
			"cv_p_value": self.cv_p_value,
			"cv_all_median": self.cv_all_median,
			"cv_all_mean": self.cv_all_mean,
			"cv_all_std": self.cv_all_std,
			"cv_significance": self.cv_significance,
			"stochastic_subset_data": self.stochastic_subset_data,
			"cross_validation_data": self.cross_validation_data,
			"all_p_values": self.all_p_values,
			"available_feature_mask": self.available_feature_mask
		}
		return pars
		
	def from_dict(self, pars):
		for prop in pars:
			setattr(self, prop, pars[prop])
		self.reuse_available_information = True
		return self
		
	def save(self, path):
		import pickle
		import gzip
		
		with gzip.open(path, 'wb') as f:
			pickle.dump(self.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
		return self
		
	def load(self, path):
		import pickle
		import gzip
		
		with gzip.open(path, 'rb') as f:
			pars = pickle.load(f)
			self.from_dict(pars)
		return self