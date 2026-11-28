
import time


POSTURE_LABELS = (
    "neutral",
    "forward_head",
    "slouched",
)


class CollectionSession:
    def __init__(
        self,
        subject_id,
        posture_label,
        duration=120,
        countdown=5,
    ):
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("Subject ID must be a non-empty string.")

        if posture_label not in POSTURE_LABELS:
            raise ValueError("Invalid posture label.")

        if duration <= 0:
            raise ValueError("Duration must be positive.")

        if countdown < 0:
            raise ValueError("Countdown cannot be negative.")

        self.subject_id = subject_id
        self.posture_label = posture_label
        self.duration = duration
        self.countdown = countdown

        self.started_at = None
        self.finished = False

    def start(self):
        if self.started_at is not None:
            raise RuntimeError("Session has already started.")

        self.started_at = time.monotonic()

    def elapsed_time(self):
        if self.started_at is None:
            return 0.0

        return max(0.0, time.monotonic() - self.started_at)

    def remaining_time(self):
        if self.started_at is None:
            return float(self.duration)

        return max(0.0, self.duration - self.elapsed_time())

    def is_complete(self):
        if self.started_at is None:
            return False

        if self.finished:
            return True

        if self.elapsed_time() >= self.duration:
            self.finished = True

        return self.finished

    def finish(self):
        self.finished = True
