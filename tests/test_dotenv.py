import os

from cosq.dotenv import load_dotenv, parse_env


def test_parses_the_documented_shape():
    parsed = parse_env(
        "# a comment\n"
        "\n"
        "COSQ_API_TOKEN=abc123\n"
        "export COSQ_PROJECT_ID=proj-1\n"
        'QUOTED="with spaces"\n'
        "SINGLE='single'\n"
    )
    assert parsed == {
        "COSQ_API_TOKEN": "abc123",
        "COSQ_PROJECT_ID": "proj-1",
        "QUOTED": "with spaces",
        "SINGLE": "single",
    }


def test_malformed_lines_are_skipped_not_fatal():
    assert parse_env("no equals sign here\n=novalue\nOK=1\n") == {"OK": "1"}


def test_values_may_contain_equals():
    assert parse_env("TOKEN=a=b=c")["TOKEN"] == "a=b=c"


def test_loads_into_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("COSQ_API_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("COSQ_API_TOKEN=from-file\n", encoding="utf-8")
    assert load_dotenv(env_file) == ["COSQ_API_TOKEN"]
    assert os.environ["COSQ_API_TOKEN"] == "from-file"


def test_the_real_environment_wins(tmp_path, monkeypatch):
    """A stale .env must never clobber a token set in the shell or by CI."""
    monkeypatch.setenv("COSQ_API_TOKEN", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("COSQ_API_TOKEN=from-file\n", encoding="utf-8")
    assert load_dotenv(env_file) == []
    assert os.environ["COSQ_API_TOKEN"] == "from-shell"


def test_a_missing_file_is_a_no_op(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_cli_loads_dotenv_before_running(tmp_path, monkeypatch, capsys):
    from cosq.cli import main

    monkeypatch.delenv("COSQ_API_TOKEN", raising=False)
    (tmp_path / ".env").write_text("COSQ_API_TOKEN=cli-token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    main(["list"])
    capsys.readouterr()
    assert os.environ["COSQ_API_TOKEN"] == "cli-token"
