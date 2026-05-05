import numpy as np

from rl_strategy.discrete.detector import DiscreteOpsDemoDetector


def test_observed_error_is_one_minus_observed_action_probability():
    probs = np.array([0.1, 0.7, 0.2])

    assert np.isclose(DiscreteOpsDemoDetector.observed_error(probs, 1), 0.3)


def test_expected_following_error_matches_paper_formula():
    probs = np.array([0.5, 0.25, 0.25])

    assert np.isclose(
        DiscreteOpsDemoDetector.expected_following_error(probs),
        0.5 * 0.5 + 0.25 * 0.75 + 0.25 * 0.75,
    )


def test_detector_switches_to_lowest_running_error_policy():
    detector = DiscreteOpsDemoDetector(
        ["chase_x", "chase_y"],
        alpha=1.0,
        threshold=0.5,
        initial_policy="chase_x",
    )

    result = detector.update(
        {
            "chase_x": np.array([0.01, 0.99]),
            "chase_y": np.array([0.99, 0.01]),
        },
        observed_action=0,
    )

    assert result.switched
    assert result.assumed_policy == "chase_y"
