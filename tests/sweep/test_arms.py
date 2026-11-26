"""Arm space: compatibility rules (spec section 5.4) and the rung 0 / rung 1 expansion."""

import pytest

from resetsweep.methods import METHODS, P_TASK_SWEEP
from resetsweep.sweep.arms import DENSE_TASKS, SPARSE_TASKS, Arm, arm_space
from resetsweep.train import TASKS

POPULATION_AND_BUFFER = {"buffer_scored", "walk", "ball", "genetic"}
SIDE_MODEL_SCORES = {"success_classifier", "rnd", "random"}
SCORELESS = {"uniform", "point", "visited", "phased"}


@pytest.fixture(scope="module")
def rung0() -> list[Arm]:
    return arm_space("rung0")


def _kw(pairs: tuple[tuple[str, object], ...]) -> dict[str, object]:
    return dict(pairs)


def test_sparse_and_dense_task_sets_match_the_registry() -> None:
    assert DENSE_TASKS == ("pendulum_dense", "arm_dense", "claw")
    assert set(SPARSE_TASKS) == {
        n for n in TASKS.names() if not n.endswith("_dense") and n != "claw"
    }
    assert "maze" in SPARSE_TASKS and "maze_trap" in SPARSE_TASKS


def test_no_scoreless_method_paired_with_score(rung0: list[Arm]) -> None:
    offenders = [
        a.label() for a in rung0 if a.method in SCORELESS and a.score is not None
    ]
    assert offenders == []
    assert {a.method for a in rung0 if a.score is None} >= SCORELESS


def test_every_arm_score_is_accepted_by_its_method(rung0: list[Arm]) -> None:
    for arm in rung0:
        accepted = METHODS.get(arm.method).accepts_scores  # pyright: ignore[reportAttributeAccessIssue] # declared on every method
        assert (arm.score or "none") in accepted, arm.label()


def test_arm_rejects_a_score_its_method_does_not_accept() -> None:
    with pytest.raises(ValueError, match="does not accept score 'rnd'"):
        Arm(
            task="maze",
            task_kw=(),
            method="uniform",
            method_kw=(),
            score="rnd",
            score_kw=(),
        )


def test_steps_to_reward_only_with_own_success(rung0: list[Arm]) -> None:
    methods = {a.method for a in rung0 if a.score == "steps_to_reward"}
    assert methods == {"own_success"}
    own = [a for a in rung0 if a.method == "own_success" and a.task == "maze"]
    assert sorted(str(a.score) for a in own) == ["None", "steps_to_reward"]
    assert {_kw(a.method_kw)["order"] for a in own} == {"uniform"}


def test_visit_count_only_with_visited_count(rung0: list[Arm]) -> None:
    arms = [a for a in rung0 if a.score == "visit_count"]
    assert {a.method for a in arms} == {"visited_count"}
    on_maze = [a for a in arms if a.task == "maze"]
    combos = {
        (_kw(a.score_kw)["post_reward"], _kw(a.method_kw)["count"]) for a in on_maze
    }
    assert combos == {
        (post, rule)
        for post in ("count", "count_x_learnability")
        for rule in ("neighbour", "lineage_depth", "embedding")
    }
    assert len(on_maze) == 6


def test_side_model_scores_only_with_population_and_buffer_methods(
    rung0: list[Arm],
) -> None:
    side = [a for a in rung0 if a.score in SIDE_MODEL_SCORES | {"critic_value"}]
    assert {a.method for a in side} == POPULATION_AND_BUFFER


def test_population_and_buffer_methods_get_twelve_scores_each(rung0: list[Arm]) -> None:
    for method in POPULATION_AND_BUFFER:
        arms = [a for a in rung0 if a.method == method and a.task == "maze"]
        assert len(arms) == 12, method
        assert {a.score for a in arms} == {
            "success_band",
            "slope",
            "critic_loss",
            "critic_value",
            "success_classifier",
            "rnd",
            "random",
        }
    forms = {
        _kw(a.score_kw)["form"]
        for a in rung0
        if a.method == "walk" and a.task == "maze" and a.score == "slope"
    }
    assert forms == {"signed", "abs", "fast_minus_slow"}


def test_critic_value_always_has_paper_temperature(rung0: list[Arm]) -> None:
    arms = [a for a in rung0 if a.score == "critic_value"]
    assert arms
    assert all(_kw(a.score_kw) == {"temperature": 0.01} for a in arms)


def test_score_kw_never_carries_a_seed(rung0: list[Arm]) -> None:
    assert all("seed" not in _kw(a.score_kw) for a in rung0)


def test_step_distance_is_never_a_standalone_score(rung0: list[Arm]) -> None:
    assert "step_distance" not in {a.score for a in rung0}


def test_visited_gets_all_p_task_values_at_rung0(rung0: list[Arm]) -> None:
    visited = [a for a in rung0 if a.method == "visited" and a.task == "maze"]
    assert sorted(_kw(a.method_kw)["p_task"] for a in visited) == sorted(P_TASK_SWEEP)  # pyright: ignore[reportArgumentType,reportCallIssue] # floats
    others = [a for a in rung0 if a.method not in {"visited", "uniform", "point"}]
    assert {_kw(a.method_kw)["p_task"] for a in others} == {0.5}


def test_rung0_has_no_dense_tasks_or_variants(rung0: list[Arm]) -> None:
    assert {a.task for a in rung0} == set(SPARSE_TASKS)
    assert all(a.task_kw == () for a in rung0)
    assert all(a.score != "inverted" for a in rung0)


def test_uniform_and_point_on_every_task(rung0: list[Arm]) -> None:
    for name in ("uniform", "point"):
        assert {a.task for a in rung0 if a.method == name} == set(SPARSE_TASKS)
    # Spec 5.4: dense tasks run only finalists, so rung 1 adds no dense anchors.
    assert arm_space("rung1_expand", survivors=[]) == []
    assert arm_space("final_expand", survivors=[]) == []


def _survivor(rung0: list[Arm], *, task: str, method: str, score: str | None) -> Arm:
    return next(
        a for a in rung0 if (a.task, a.method, a.score) == (task, method, score)
    )


def test_rung1_expand_adds_only_p_task_and_task_variants(rung0: list[Arm]) -> None:
    """Spec 5.4 / final-review ruling C2a: rung 1 adds the p_task variants {0.2, 0.8} and the
    task variants (each with its own uniform/point anchors), nothing else."""
    survivor = _survivor(rung0, task="maze", method="walk", score="rnd")
    expanded = {a.label() for a in arm_space("rung1_expand", survivors=[survivor])}
    assert expanded == {
        "maze/walk[p_task=0.2]/rnd",
        "maze/walk[p_task=0.8]/rnd",
        "maze[lanes=5]/walk[p_task=0.5]/rnd",
        "maze[lanes=5]/uniform/-",
        "maze[lanes=5]/point/-",
    }


def test_rung1_expand_has_no_inverted_dense_or_order_variants(rung0: list[Arm]) -> None:
    survivors = [a for a in rung0 if a.task in {"maze", "pendulum_sparse"}]
    expanded = arm_space("rung1_expand", survivors=survivors)
    assert [a for a in expanded if a.score == "inverted"] == []
    assert [a for a in expanded if a.task in DENSE_TASKS] == []
    own_orders = {
        _kw(a.method_kw)["order"] for a in expanded if a.method == "own_success"
    }
    assert own_orders == {"uniform"}


def test_rung1_expand_p_task_variants(rung0: list[Arm]) -> None:
    walk = _survivor(rung0, task="maze", method="walk", score="rnd")
    expanded = arm_space("rung1_expand", survivors=[walk])
    walk_p = {
        _kw(a.method_kw)["p_task"]
        for a in expanded
        if a.method == "walk"
        and (a.task, a.task_kw) == ("maze", ())
        and a.score == "rnd"
    }
    assert walk_p == {0.2, 0.8}


def test_final_expand_adds_inverted_and_dense_copies(rung0: list[Arm]) -> None:
    survivor = _survivor(rung0, task="maze", method="walk", score="rnd")
    expanded = arm_space("final_expand", survivors=[survivor])
    inverted = [a for a in expanded if a.score == "inverted"]
    assert [(a.task, a.method) for a in inverted] == [("maze", "walk")]
    assert inverted[0].score_kwargs() == {"inner": "rnd", "inner_kw": {}}
    dense = [a for a in expanded if a.task in DENSE_TASKS and a.method == "walk"]
    assert {a.task for a in dense} == set(DENSE_TASKS)
    assert {a.score for a in dense} == {"rnd"}
    anchors = {(a.task, a.method) for a in expanded if a.method in {"uniform", "point"}}
    assert anchors == {(t, m) for t in DENSE_TASKS for m in ("uniform", "point")}
    assert len(expanded) == 1 + 3 + 6


def test_final_expand_takes_rung1_arms_and_clears_task_kw_on_dense(
    rung0: list[Arm],
) -> None:
    base = _survivor(rung0, task="maze", method="walk", score="rnd")
    variant = Arm(**{**base.__dict__, "task_kw": (("lanes", 5),)})
    p_variant = Arm(**{**base.__dict__, "method_kw": (("p_task", 0.2),)})
    expanded = arm_space("final_expand", survivors=[variant, p_variant])
    dense = {
        a.label() for a in expanded if a.task in DENSE_TASKS and a.method == "walk"
    }
    assert dense == {
        f"{t}/walk[p_task={p}]/rnd" for t in DENSE_TASKS for p in (0.5, 0.2)
    }
    inverted = {(a.task, a.task_kw) for a in expanded if a.score == "inverted"}
    assert inverted == {("maze", (("lanes", 5),)), ("maze", ())}


def test_final_expand_dedupes_dense_copies_across_tasks(rung0: list[Arm]) -> None:
    a = _survivor(rung0, task="maze", method="walk", score="rnd")
    b = _survivor(rung0, task="pendulum_sparse", method="walk", score="rnd")
    expanded = arm_space("final_expand", survivors=[a, b])
    assert len(expanded) == len(set(expanded))
    dense_walk = [x for x in expanded if x.task in DENSE_TASKS and x.method == "walk"]
    assert len(dense_walk) == len(DENSE_TASKS)
    assert set(expanded) & {a, b} == set()


def test_final_expand_rejects_dense_survivor(rung0: list[Arm]) -> None:
    survivor = _survivor(rung0, task="maze", method="walk", score="rnd")
    dense = Arm(**{**survivor.__dict__, "task": "pendulum_dense"})
    with pytest.raises(ValueError, match="sparse task"):
        arm_space("final_expand", survivors=[dense])


def test_final_expand_adds_nothing_for_scoreless_controls(rung0: list[Arm]) -> None:
    uniform = _survivor(rung0, task="maze", method="uniform", score=None)
    assert arm_space("final_expand", survivors=[uniform]) == []


def test_rung1_expand_task_variants(rung0: list[Arm]) -> None:
    maze = _survivor(rung0, task="maze", method="walk", score="rnd")
    trap = _survivor(rung0, task="maze_trap", method="walk", score="rnd")
    expanded = arm_space("rung1_expand", survivors=[maze, trap])
    sparse_variants = {
        (a.task, a.task_kw) for a in expanded if a.task_kw and a.task in SPARSE_TASKS
    }
    assert sparse_variants == {
        ("maze", (("lanes", 5),)),
        ("maze_trap", (("trap_scale", 0.5),)),
        ("maze_trap", (("trap_scale", 2.0),)),
    }


def test_rung1_expand_adds_anchors_for_each_task_variant(rung0: list[Arm]) -> None:
    maze = _survivor(rung0, task="maze", method="walk", score="rnd")
    trap = _survivor(rung0, task="maze_trap", method="walk", score="rnd")
    expanded = arm_space("rung1_expand", survivors=[maze, trap])
    anchors = {
        (a.task, a.task_kw, a.method)
        for a in expanded
        if a.task_kw and a.method in {"uniform", "point"}
    }
    assert anchors == {
        (task, kw, method)
        for task, kw in (
            ("maze", (("lanes", 5),)),
            ("maze_trap", (("trap_scale", 0.5),)),
            ("maze_trap", (("trap_scale", 2.0),)),
        )
        for method in ("uniform", "point")
    }


def test_variant_anchors_match_the_plain_anchors(rung0: list[Arm]) -> None:
    maze = _survivor(rung0, task="maze", method="walk", score="rnd")
    plain = {
        a.method: a
        for a in rung0
        if a.task == "maze" and a.method in {"uniform", "point"}
    }
    expanded = arm_space("rung1_expand", survivors=[maze])
    variant = [a for a in expanded if a.task_kw and a.method in {"uniform", "point"}]
    assert variant == [
        Arm(**{**plain[a.method].__dict__, "task_kw": (("lanes", 5),)}) for a in variant
    ]
    assert len(variant) == 2


def test_rung1_expand_rejects_non_rung0_survivor(rung0: list[Arm]) -> None:
    survivor = _survivor(rung0, task="maze", method="walk", score="rnd")
    dense = Arm(**{**survivor.__dict__, "task": "pendulum_dense"})
    with pytest.raises(ValueError, match="rung 0 arm"):
        arm_space("rung1_expand", survivors=[dense])


def test_arm_labels_unique(rung0: list[Arm]) -> None:
    labels = [a.label() for a in rung0]
    assert len(labels) == len(set(labels))
    survivors = [a for a in rung0 if a.task in {"maze", "maze_trap"}]
    r1 = [a.label() for a in arm_space("rung1_expand", survivors=survivors)]
    assert len(r1) == len(set(r1))


def test_arms_are_hashable_and_distinct(rung0: list[Arm]) -> None:
    assert len(set(rung0)) == len(rung0)


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        arm_space("rung9")  # pyright: ignore[reportArgumentType] # deliberately invalid


def test_rung0_arm_count_printed(rung0: list[Arm]) -> None:
    # Per task: uniform 1 + point 1 + visited 5 (P_TASK_SWEEP) + visited_count 6
    # (2 post_reward x 3 count rules) + own_success 2 (None, steps_to_reward) + phased 1
    # + 4 population/buffer methods x 12 scores = 48  ->  64 arms per task.
    # 64 arms x 11 sparse tasks = 704.
    per_task = 1 + 1 + 5 + 6 + 2 + 1 + 4 * 12
    assert per_task == 64
    assert len(SPARSE_TASKS) == 11
    assert len(rung0) == per_task * 11 == 704
    print(f"rung0 arms: {len(rung0)}")


def test_inverted_wraps_the_survivors_score_kw(rung0: list[Arm]) -> None:
    survivor = _survivor(rung0, task="maze", method="walk", score="critic_value")
    inverted = [
        a
        for a in arm_space("final_expand", survivors=[survivor])
        if a.score == "inverted"
    ]
    assert [a.score_kwargs() for a in inverted] == [
        {"inner": "critic_value", "inner_kw": {"temperature": 0.01}}
    ]


def test_rung1_expand_never_repeats_rung0_arms(rung0: list[Arm]) -> None:
    survivors = [a for a in rung0 if a.task in {"maze", "maze_trap"}]
    expanded = arm_space("rung1_expand", survivors=survivors)
    assert set(expanded) & set(rung0) == set()
    assert set(expanded) & set(survivors) == set()
    assert len(expanded) == len(set(expanded))


def test_visited_survivor_adds_no_p_task_variants(rung0: list[Arm]) -> None:
    visited = _survivor_where(rung0, "visited", p_task=0.0)
    expanded = arm_space("rung1_expand", survivors=[visited])
    p_variants = [
        a
        for a in expanded
        if a.method == "visited" and (a.task, a.task_kw) == ("maze", ())
    ]
    assert p_variants == []
    # Beyond the maze variant (task_kw set), rung 1 adds no visited arm.
    plain = {a.task for a in expanded if a.method == "visited" and not a.task_kw}
    assert plain == set()


def _survivor_where(rung0: list[Arm], method: str, *, p_task: float) -> Arm:
    return next(
        a
        for a in rung0
        if a.task == "maze"
        and a.method == method
        and _kw(a.method_kw).get("p_task") == p_task
    )


def test_critic_value_never_runs_on_dense_tasks(rung0: list[Arm]) -> None:
    # arm_dense and claw rewards are mostly negative, so V * (vmax - V) is meaningless there.
    survivors = [a for a in rung0 if a.task == "maze" and a.score == "critic_value"]
    assert survivors
    expanded = arm_space("final_expand", survivors=survivors)
    on_dense = [a for a in expanded if a.task in DENSE_TASKS]
    assert [a for a in on_dense if a.score in {"critic_value", "inverted"}] == []
    assert [a for a in on_dense if a.method == "walk"] == []
    # inverted(critic_value) still runs on the sparse task itself.
    assert {a.score for a in expanded if a.task == "maze"} == {"inverted"}
