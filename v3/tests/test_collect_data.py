from collect_data import get_subject_id, get_posture_label


def test_get_subject_id_retries_invalid_input(monkeypatch):
    answers = iter(["bad id", "subject_001"])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(answers)
    )

    assert get_subject_id() == "subject_001"


def test_get_posture_label_neutral(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1")

    assert get_posture_label() == "neutral"


def test_get_posture_label_forward_head(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")

    assert get_posture_label() == "forward_head"


def test_get_posture_label_slouched(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "3")

    assert get_posture_label() == "slouched"


def test_get_posture_label_retries_invalid_input(monkeypatch):
    answers = iter(["4", "neutral", "2"])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(answers)
    )

    assert get_posture_label() == "forward_head"