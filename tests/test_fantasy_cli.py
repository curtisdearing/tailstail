"""Data-independent smoke tests for the fantasy CLI argument surface.

These assert the command-line contract (subcommands, option defaults, custom
type parsers, and required-argument enforcement) without loading any historical
parquet frame or touching the network, so they run in the reproducible CI list.
"""

from __future__ import annotations

import argparse

import pytest

from nflvalue.fantasy import cli

SUBCOMMANDS = {
    "fetch", "build", "train", "backtest",
    "audit-monte-carlo", "project", "simulate",
}


def test_parser_exposes_every_subcommand():
    parser = cli.build_parser()
    choices = set(parser._subparsers._group_actions[0].choices)
    assert choices == SUBCOMMANDS


def test_season_range_parser_expands_inclusive():
    assert cli._seasons("2023:2025") == [2023, 2024, 2025]


def test_season_comma_parser_is_sorted_and_deduped():
    assert cli._seasons("2021,2019,2021,2020") == [2019, 2020, 2021]


def test_season_range_rejects_reversed_bounds():
    with pytest.raises(argparse.ArgumentTypeError):
        cli._seasons("2025:2023")


def test_project_requires_season_and_week():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["project"])
    args = parser.parse_args(["project", "--season", "2024", "--week", "3"])
    assert (args.season, args.week) == (2024, 3)


def test_command_is_required():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_scoring_choice_is_validated():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "--scoring", "superflex"])
    assert parser.parse_args(["build", "--scoring", "half_ppr"]).scoring == "half_ppr"


def test_role_mixture_flag_defaults_off_and_toggles():
    parser = cli.build_parser()
    base = parser.parse_args(["audit-monte-carlo"])
    assert base.role_mixture is False
    on = parser.parse_args(["audit-monte-carlo", "--role-mixture"])
    assert on.role_mixture is True


def test_defaults_use_ppr_scoring():
    parser = cli.build_parser()
    assert parser.parse_args(["train"]).scoring == "ppr"
