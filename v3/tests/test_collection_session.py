
import time

import pytest

from posture.collection_session import CollectionSession


def test_session_initializes_correctly():
    session = CollectionSession(
        subject_id="subject_001",
        posture_label="neutral",
    )

    assert session.subject_id == "subject_001"
    assert session.posture_label == "neutral"
    assert session.duration == 120
    assert session.countdown == 5


def test_session_accepts_all_posture_labels():
    for label in ("neutral", "forward_head", "slouched"):
        session = CollectionSession("subject_001", label)
        assert session.posture_label == label


def test_session_rejects_empty_subject_id():
    with pytest.raises(ValueError):
        CollectionSession("", "neutral")


def test_session_rejects_invalid_posture_label():
    with pytest.raises(ValueError):
        CollectionSession("subject_001", "unknown")


def test_session_rejects_invalid_duration():
    with pytest.raises(ValueError):
        CollectionSession("subject_001", "neutral", duration=0)


def test_session_rejects_negative_countdown():
    with pytest.raises(ValueError):
        CollectionSession("subject_001", "neutral", countdown=-1)


def test_session_is_not_complete_before_start():
    session = CollectionSession("subject_001", "neutral")

    assert session.is_complete() is False


def test_session_starts_and_tracks_elapsed_time():
    session = CollectionSession(
        "subject_001",
        "neutral",
        duration=10,
    )

    session.start()
    time.sleep(0.1)

    assert session.elapsed_time() >= 0.1
    assert session.remaining_time() < 10


def test_session_cannot_start_twice():
    session = CollectionSession("subject_001", "neutral")

    session.start()

    with pytest.raises(RuntimeError):
        session.start()


def test_session_completes_after_duration():
    session = CollectionSession(
        "subject_001",
        "neutral",
        duration=0.1,
    )

    session.start()
    time.sleep(0.15)

    assert session.is_complete() is True
    assert session.remaining_time() == 0


def test_session_can_be_finished_manually():
    session = CollectionSession("subject_001", "neutral")

    session.start()
    session.finish()

    assert session.is_complete() is True
