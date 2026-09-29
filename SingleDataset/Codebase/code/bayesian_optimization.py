from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def set_random_seed(seed):
    # Seed numpy's global RNG so results are reproducible across runs.
    np.random.seed(seed)


class Bayesian_Optimization():
    """
    Minimal Bayesian Optimization engine.

    Fits a Gaussian Process to the (point, target) observations seen so far,
    then uses an acquisition/utility function to suggest the next point to
    evaluate. Points are registered externally via `register`.
    """

    def __init__(self, pbounds, random_state=42,
                 matern_nu=2.5, gp_alpha=1e-6, gp_n_restarts=25):
        # -- GP hyperparameters --
        # matern_nu     : smoothness of the Matern kernel
        # gp_alpha      : noise added to the GP diagonal for stability 
        # gp_n_restarts : restarts when fitting GP hyperparameters 

        # Seed both the global RNG and a local RandomState for this optimizer,
        # so the GP and the random sampling below are reproducible.
        set_random_seed(random_state)
        self._random_state = np.random.RandomState(random_state)

        # Sort parameter names so the ordering of columns in the internal
        # arrays is always deterministic (independent of dict order).
        self._keys = sorted(pbounds)

        # Build the (dim x 2) array of [min, max] bounds, in the same sorted
        # order as self._keys.
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=float
        )
        self.dim = len(pbounds)

        # Preallocate empty arrays for the observed points (X) and targets (Y).
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # Default Gaussian Process surrogate model.
        # Matern(nu=...) is a smooth kernel; alpha adds a small amount of noise
        # for numerical stability; n_restarts_optimizer helps the GP avoid poor
        # local optima when fitting its hyperparameters.
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=matern_nu),
            alpha=gp_alpha,
            normalize_y=True,
            n_restarts_optimizer=gp_n_restarts,
            random_state=self._random_state,
        )

    @property
    def max(self):
        # Return the best observation so far (highest target) and the
        # parameters that produced it. Empty dict if nothing registered yet.
        try:
            res = {
                'target': self._target.max(),
                'params': dict(
                    zip(self._keys, self._params[self._target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    @property
    def res(self):
        """Get all target values found and corresponding parameters."""
        # Zip each stored parameter row back into a {key: value} dict.
        params = [dict(zip(self._keys, p)) for p in self._params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self._target, params)
        ]

    def params_to_array(self, params):
        # Convert a {key: value} dict into an array ordered by self._keys.
        try:
            assert set(params) == set(self._keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self._keys])

    def array_to_params(self, x):
        # Convert an array back into a {key: value} dict using self._keys.
        try:
            assert len(x) == len(self._keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self._keys, x))

    def register(self, params, target):
        """Expect observation with known target"""
        # Accept either a raw array or a {key: value} dict of parameters.
        try:
            x = np.asarray(params, dtype=float)
        except TypeError:
            x = self.params_to_array(params)

        # if x in self._params:
        #     raise KeyError('Data point {} is not unique'.format(x))

        # Append the new point and its target to the observation history.
        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def suggest(self, utility_class, n_warmup=500, n_iter=10):
        """
        Find point to sample from utility function max.
        """
        # Fit the GP to everything observed so far.
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._params, self._target)

        # Maximize the acquisition function to pick the next point to probe.
        suggestion = self.acq_max(
            utility_func=utility_class.utility,
            gp=self._gp,
            y_max=self._target.max(),
            bounds=self._bounds,
            random_state=self._random_state,
            n_warmup=n_warmup,
            n_iter=n_iter
        )

        return self.array_to_params(suggestion)

    def acq_max(self, utility_func, gp, y_max, bounds,
                random_state, n_warmup, n_iter):
        # --- Warmup: coarse random search over the whole space ---
        # Draw n_warmup random points and keep the best one according to
        # the acquisition function. This gives a good starting region.
        x_rand = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                      size=(n_warmup, bounds.shape[0])
                                      )
        y_rand = utility_func(x_rand, gp=gp, y_max=y_max)
        x_max = x_rand[y_rand.argmax()]
        max_acq = y_rand.max()

        # --- Refinement: local optimization from several random seeds ---
        # Use L-BFGS-B to fine-tune the maximum, starting from n_iter seeds.
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(n_iter, bounds.shape[0]))
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            # (minimizing the negative == maximizing the acquisition).
            res = minimize(lambda x: -utility_func(x.reshape(1, -1), gp=gp, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")

            # Skip seeds where the optimizer did not converge.
            if not res.success:
                continue

            # Store it if better than previous minimum(maximum).
            # compatibilidad con version de scipy
            fun_val = res.fun[0] if hasattr(res.fun, '__len__') else res.fun
            if max_acq is None or -fun_val >= max_acq:
                x_max = res.x
                max_acq = -fun_val

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction:
    """
    Acquisition (utility) functions for Bayesian Optimization.

    Wraps one of three strategies for trading off exploration vs
    exploitation when choosing the next point to evaluate:
      - 'ucb' : Upper Confidence Bound
      - 'ei'  : Expected Improvement
      - 'poi' : Probability of Improvement
    """

    def __init__(self, function, hyperparam):
        # hyperparam is kappa (for UCB) or xi (for EI/POI): it controls how
        # much the function favours exploration over exploitation.
        self.hyperparam = hyperparam

        if function not in ['ucb', 'ei', 'poi']:
            err = 'Utility function not implemented.' \
                  'Please choose between ucb, ei or poi'
            raise NotImplementedError(err)
        else:
            self.function = function

    def utility(self, space, gp, y_max):
        # Dispatch to the chosen acquisition function.
        if self.function == 'ucb':
            return self._ucb(space, gp, self.hyperparam)
        if self.function == 'ei':
            return self._ei(space, gp, y_max, self.hyperparam)
        if self.function == 'poi':
            return self._poi(space, gp, y_max, self.hyperparam)

    @staticmethod
    def _ucb(space, gp, kappa):
        # Upper Confidence Bound: mean + kappa * std.
        # Higher kappa -> more exploration (favours uncertain regions).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(space, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(space, gp, y_max, xi):
        # Expected Improvement: expected amount by which a point beats the
        # current best (y_max), integrated over the GP's uncertainty.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(space, return_std=True)

        z = (mean - y_max - xi) / std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(space, gp, y_max, xi):
        # Probability of Improvement: probability that a point beats y_max.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(space, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)
