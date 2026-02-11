from typing import TYPE_CHECKING, Dict, List, Optional

from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.tune.experiment import Trial


def _import_trackio():
    """Try importing trackio.

    Used to check if trackio is installed and, otherwise, pass an informative
    error message.
    """
    try:
        import trackio  # noqa: F401
    except ImportError:
        raise RuntimeError("pip install 'trackio' to use TrackioLoggerCallback")

    return trackio


@PublicAPI(stability="alpha")
class TrackioLoggerCallback(LoggerCallback):
    """TrackioLoggerCallback for logging Tune results to Trackio.

    Trackio (https://github.com/gradio-app/trackio) is a lightweight,
    local-first experiment tracking library with a W&B-compatible API.
    It supports local SQLite storage, Hugging Face Spaces dashboards,
    and Hugging Face Dataset syncing.

    This Ray Tune ``LoggerCallback`` sends metrics and parameters to
    Trackio for tracking.

    In order to use the TrackioLoggerCallback you must first install Trackio
    via ``pip install trackio``

    Args:
        project: Trackio project name.
        name: Run name template. Defaults to the trial name.
        group: Group name for organizing runs. Defaults to the
            experiment name.
        space_id: Hugging Face Space ID for remote dashboard.
        dataset_id: Hugging Face Dataset ID for metric syncing.
        tags: Tags for filtering runs.
        excludes: Metric keys to exclude from logging.
        **trackio_init_kwargs: Additional keyword arguments passed
            through to ``trackio.init()``.

    Example:

    .. code-block:: python

        from ray.air.integrations.trackio import TrackioLoggerCallback
        tuner = tune.Tuner(
            train,
            param_space=config,
            run_config=RunConfig(
                callbacks=[TrackioLoggerCallback(project="my_project")]
            ),
        )
        tuner.fit()

    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        group: Optional[str] = None,
        space_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        **trackio_init_kwargs,
    ):
        _import_trackio()
        self.project = project
        self.name = name
        self.group = group
        self.space_id = space_id
        self.dataset_id = dataset_id
        self.tags = tags
        self.excludes = excludes or []
        self.trackio_init_kwargs = trackio_init_kwargs

        self._trial_runs: Dict["Trial", object] = {}

    def log_trial_start(self, trial: "Trial"):
        trackio = _import_trackio()

        if trial in self._trial_runs:
            return

        name = self.name if self.name else str(trial)

        config = trial.config.copy()
        config.pop("callbacks", None)

        init_kwargs = dict(
            project=self.project,
            name=name,
            config=config,
        )
        if self.group is not None:
            init_kwargs["group"] = self.group
        if self.space_id is not None:
            init_kwargs["space_id"] = self.space_id
        if self.dataset_id is not None:
            init_kwargs["dataset_id"] = self.dataset_id
        if self.tags is not None:
            init_kwargs["tags"] = self.tags
        init_kwargs.update(self.trackio_init_kwargs)

        run = trackio.init(**init_kwargs)
        self._trial_runs[trial] = run

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        if trial not in self._trial_runs:
            self.log_trial_start(trial)

        run = self._trial_runs[trial]
        step = result.get("training_iteration", iteration)

        flat_result = flatten_dict(result, delimiter="/")

        metrics = {}
        for k, v in flat_result.items():
            if k in self.excludes:
                continue
            if isinstance(v, (int, float)):
                metrics[k] = v

        run.log(metrics, step=step)

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        if trial in self._trial_runs:
            run = self._trial_runs.pop(trial)
            run.finish()

    def __del__(self):
        for trial, run in self._trial_runs.items():
            run.finish()
        self._trial_runs = {}
