import io
import time
import unittest
from unittest.mock import MagicMock

from switch_trust.eval.core.eval.evaluation import (
    Evaluation,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.eval.core.eval.observer import (
    ConsoleObserver,
    ConsolePollingObserver,
    PollingObserver,
    format_summary,
    print_summary,
)


def _make_summary(
    status: EvaluationStatus = EvaluationStatus.RUNNING,
    total: int = 10,
    finished: int = 5,
    errors: int = 1,
    max_score: float = 10.0,
    achieved: float = 7.0,
    error_messages: list[str] | None = None,
) -> EvaluationSummary:
    return EvaluationSummary(
        status=status,
        total_evaluations=total,
        finished_evaluations=finished,
        error_evaluations=errors,
        max_score=max_score,
        achieved_score=achieved,
        error_messages=error_messages or [],
    )


class TestFormatSummary(unittest.TestCase):
    def test_running(self):
        s = _make_summary()
        result = format_summary(s)
        self.assertIn("[running]", result)
        self.assertIn("6/10 evaluations", result)
        self.assertIn("60%", result)
        self.assertIn("1 errors", result)

    def test_finished_with_score(self):
        s = _make_summary(
            status=EvaluationStatus.FINISHED,
            total=4,
            finished=4,
            errors=0,
            max_score=4.0,
            achieved=3.0,
        )
        result = format_summary(s)
        self.assertIn("[finished]", result)
        self.assertIn("score: 0.75", result)
        self.assertNotIn("errors", result)

    def test_no_score_when_running(self):
        s = _make_summary(status=EvaluationStatus.RUNNING)
        result = format_summary(s)
        self.assertNotIn("score:", result)


class TestPrintSummary(unittest.TestCase):
    def test_basic_print(self):
        buf = io.StringIO()
        s = _make_summary()
        print_summary(s, file=buf)
        output = buf.getvalue()
        self.assertIn("[running]", output)
        self.assertNotIn("Error:", output)

    def test_print_with_errors(self):
        buf = io.StringIO()
        s = _make_summary(error_messages=["something broke"])
        print_summary(s, include_errors=True, file=buf)
        output = buf.getvalue()
        self.assertIn("Error: something broke", output)

    def test_print_without_errors_flag(self):
        buf = io.StringIO()
        s = _make_summary(error_messages=["something broke"])
        print_summary(s, include_errors=False, file=buf)
        output = buf.getvalue()
        self.assertNotIn("Error:", output)


class TestConsoleObserver(unittest.TestCase):
    def test_prints_on_call(self):
        buf = io.StringIO()
        observer = ConsoleObserver(file=buf)

        evaluation = MagicMock(spec=Evaluation)
        evaluation.get_summary.return_value = _make_summary()

        observer(evaluation)
        output = buf.getvalue()
        self.assertIn("[running]", output)
        self.assertTrue(output.startswith("\r"))

    def test_finish_prints_newline(self):
        buf = io.StringIO()
        observer = ConsoleObserver(file=buf)
        observer.finish()
        self.assertEqual(buf.getvalue(), "\n")


class TestPollingObserver(unittest.TestCase):
    def test_polls_and_stops(self):
        summaries = []

        evaluation = MagicMock(spec=Evaluation)
        call_count = 0

        def get_summary():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return _make_summary(
                    status=EvaluationStatus.FINISHED,
                )
            return _make_summary(
                status=EvaluationStatus.RUNNING,
            )

        evaluation.get_summary = get_summary

        poller = PollingObserver(interval=0.05)
        poller.start(evaluation, lambda s: summaries.append(s))

        # Wait for polling to complete
        for _ in range(50):
            if poller._thread is None:
                break
            time.sleep(0.05)

        self.assertGreaterEqual(len(summaries), 2)
        self.assertEqual(
            summaries[-1].status,
            EvaluationStatus.FINISHED,
        )

    def test_stop_halts_polling(self):
        evaluation = MagicMock(spec=Evaluation)
        evaluation.get_summary.return_value = _make_summary(
            status=EvaluationStatus.RUNNING,
        )

        called = []
        poller = PollingObserver(interval=0.05)
        poller.start(evaluation, lambda s: called.append(s))
        time.sleep(0.1)
        poller.stop()

        count_at_stop = len(called)
        time.sleep(0.15)
        self.assertEqual(len(called), count_at_stop)


class TestConsolePollingObserver(unittest.TestCase):
    def test_prints_to_file(self):
        buf = io.StringIO()
        evaluation = MagicMock(spec=Evaluation)
        evaluation.get_summary.return_value = _make_summary(
            status=EvaluationStatus.FINISHED,
        )

        poller = ConsolePollingObserver(interval=0.05, file=buf)
        poller.start(evaluation)
        time.sleep(0.2)
        poller.stop()

        output = buf.getvalue()
        self.assertIn("[finished]", output)


if __name__ == "__main__":
    unittest.main()
