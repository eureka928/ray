import unittest
from collections import namedtuple
from unittest.mock import MagicMock, patch

from ray.air.integrations.trackio import TrackioLoggerCallback


class MockTrial(
    namedtuple("MockTrial", ["config", "trial_name", "trial_id", "logdir"])
):
    def __hash__(self):
        return hash(self.trial_id)

    def __str__(self):
        return self.trial_name


class TestImportTrackio(unittest.TestCase):
    def test_init_missing_dependency(self):
        """Verify RuntimeError when trackio is not installed."""
        with patch.dict("sys.modules", {"trackio": None}):
            with self.assertRaises(RuntimeError) as ctx:
                TrackioLoggerCallback(project="test")
            self.assertIn("pip install", str(ctx.exception))


@patch("ray.air.integrations.trackio._import_trackio")
class TestLogTrialStart(unittest.TestCase):
    def test_log_trial_start(self, mock_import):
        """Verify trackio.init() called with correct project/name/config."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio

        logger = TrackioLoggerCallback(project="my_project")
        trial = MockTrial({"lr": 0.01}, "trial_1", "1", "/tmp/trial_1")

        logger.log_trial_start(trial)

        mock_trackio.init.assert_called_once_with(
            project="my_project",
            name="trial_1",
            config={"lr": 0.01},
        )

    def test_log_trial_start_with_options(self, mock_import):
        """Verify optional parameters are passed through."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio

        logger = TrackioLoggerCallback(
            project="my_project",
            name="custom_name",
            group="my_group",
            space_id="user/space",
            dataset_id="user/dataset",
            tags=["tag1", "tag2"],
        )
        trial = MockTrial({"lr": 0.01}, "trial_1", "1", "/tmp/trial_1")

        logger.log_trial_start(trial)

        mock_trackio.init.assert_called_once_with(
            project="my_project",
            name="custom_name",
            config={"lr": 0.01},
            group="my_group",
            space_id="user/space",
            dataset_id="user/dataset",
            tags=["tag1", "tag2"],
        )

    def test_log_trial_start_idempotent(self, mock_import):
        """Verify calling log_trial_start twice does not create two runs."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio

        logger = TrackioLoggerCallback(project="my_project")
        trial = MockTrial({"lr": 0.01}, "trial_1", "1", "/tmp/trial_1")

        logger.log_trial_start(trial)
        logger.log_trial_start(trial)

        mock_trackio.init.assert_called_once()

    def test_log_trial_start_removes_callbacks(self, mock_import):
        """Verify callbacks key is removed from config before logging."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio

        logger = TrackioLoggerCallback(project="my_project")
        trial = MockTrial(
            {"lr": 0.01, "callbacks": ["cb1"]}, "trial_1", "1", "/tmp/trial_1"
        )

        logger.log_trial_start(trial)

        call_kwargs = mock_trackio.init.call_args
        self.assertNotIn("callbacks", call_kwargs.kwargs["config"])


@patch("ray.air.integrations.trackio._import_trackio")
class TestLogTrialResult(unittest.TestCase):
    def setUp(self):
        self.trial = MockTrial({"lr": 0.01}, "trial_1", "1", "/tmp/trial_1")
        self.result = {
            "training_iteration": 5,
            "metric1": 0.8,
            "metric2": 1,
            "str_metric": "not_a_number",
            "nested": {"a": 1, "b": 2},
        }

    def test_log_trial_result(self, mock_import):
        """Verify run.log() called with flattened metrics and correct step."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio
        mock_run = MagicMock()
        mock_trackio.init.return_value = mock_run

        logger = TrackioLoggerCallback(project="my_project")
        logger.log_trial_result(1, self.trial, self.result)

        mock_run.log.assert_called_once()
        call_args = mock_run.log.call_args
        logged_metrics = call_args[0][0]
        logged_step = call_args[1]["step"]

        # Should include numeric values only
        self.assertIn("metric1", logged_metrics)
        self.assertIn("metric2", logged_metrics)
        self.assertIn("nested/a", logged_metrics)
        self.assertIn("nested/b", logged_metrics)
        self.assertIn("training_iteration", logged_metrics)
        # String values should be excluded
        self.assertNotIn("str_metric", logged_metrics)
        self.assertEqual(logged_step, 5)

    def test_log_trial_result_excludes(self, mock_import):
        """Verify excluded keys are not logged."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio
        mock_run = MagicMock()
        mock_trackio.init.return_value = mock_run

        logger = TrackioLoggerCallback(
            project="my_project", excludes=["metric1"]
        )
        logger.log_trial_result(1, self.trial, self.result)

        call_args = mock_run.log.call_args
        logged_metrics = call_args[0][0]

        self.assertNotIn("metric1", logged_metrics)
        self.assertIn("metric2", logged_metrics)

    def test_log_trial_result_auto_starts_trial(self, mock_import):
        """Verify log_trial_result starts trial if not already started."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio
        mock_run = MagicMock()
        mock_trackio.init.return_value = mock_run

        logger = TrackioLoggerCallback(project="my_project")
        # Don't call log_trial_start explicitly
        logger.log_trial_result(1, self.trial, self.result)

        mock_trackio.init.assert_called_once()
        mock_run.log.assert_called_once()


@patch("ray.air.integrations.trackio._import_trackio")
class TestLogTrialEnd(unittest.TestCase):
    def test_log_trial_end(self, mock_import):
        """Verify run.finish() called and trial removed."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio
        mock_run = MagicMock()
        mock_trackio.init.return_value = mock_run

        logger = TrackioLoggerCallback(project="my_project")
        trial = MockTrial({"lr": 0.01}, "trial_1", "1", "/tmp/trial_1")

        logger.log_trial_start(trial)
        logger.log_trial_end(trial)

        mock_run.finish.assert_called_once()
        self.assertNotIn(trial, logger._trial_runs)

    def test_log_trial_end_not_started(self, mock_import):
        """Verify log_trial_end is a no-op if trial was never started."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio

        logger = TrackioLoggerCallback(project="my_project")
        trial = MockTrial({"lr": 0.01}, "trial_1", "1", "/tmp/trial_1")

        # Should not raise
        logger.log_trial_end(trial)


@patch("ray.air.integrations.trackio._import_trackio")
class TestMultipleTrials(unittest.TestCase):
    def test_multiple_trials(self, mock_import):
        """Verify independent runs per trial."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio
        mock_run_1 = MagicMock()
        mock_run_2 = MagicMock()
        mock_trackio.init.side_effect = [mock_run_1, mock_run_2]

        logger = TrackioLoggerCallback(project="my_project")
        trial_1 = MockTrial({"lr": 0.01}, "trial_1", "1", "/tmp/trial_1")
        trial_2 = MockTrial({"lr": 0.02}, "trial_2", "2", "/tmp/trial_2")

        logger.log_trial_start(trial_1)
        logger.log_trial_start(trial_2)

        self.assertEqual(mock_trackio.init.call_count, 2)
        self.assertEqual(logger._trial_runs[trial_1], mock_run_1)
        self.assertEqual(logger._trial_runs[trial_2], mock_run_2)

        logger.log_trial_end(trial_1)
        mock_run_1.finish.assert_called_once()
        mock_run_2.finish.assert_not_called()

        logger.log_trial_end(trial_2)
        mock_run_2.finish.assert_called_once()


@patch("ray.air.integrations.trackio._import_trackio")
class TestDelCleanup(unittest.TestCase):
    def test_del_cleanup(self, mock_import):
        """Verify __del__ finishes all runs."""
        mock_trackio = MagicMock()
        mock_import.return_value = mock_trackio
        mock_run_1 = MagicMock()
        mock_run_2 = MagicMock()
        mock_trackio.init.side_effect = [mock_run_1, mock_run_2]

        logger = TrackioLoggerCallback(project="my_project")
        trial_1 = MockTrial({"lr": 0.01}, "trial_1", "1", "/tmp/trial_1")
        trial_2 = MockTrial({"lr": 0.02}, "trial_2", "2", "/tmp/trial_2")

        logger.log_trial_start(trial_1)
        logger.log_trial_start(trial_2)

        mock_run_1.finish.assert_not_called()
        mock_run_2.finish.assert_not_called()

        logger.__del__()

        mock_run_1.finish.assert_called_once()
        mock_run_2.finish.assert_called_once()
        self.assertEqual(len(logger._trial_runs), 0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
