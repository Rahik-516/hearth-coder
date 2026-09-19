"""The index watcher — I2.

The acceptance criterion is "switching git branches reindexes only changed files", which
is really two claims: the burst must arrive as one batch rather than four hundred, and the
files git left byte-identical must not be parsed again.

Most of this tests :func:`plan_batch`, which decides what a batch of events *means*. The
watching loop itself is thin on purpose — a test that races a real filesystem watcher
fails on a slow machine and passes on a fast one, which is worse than no test. The one
end-to-end case drives `apply` with a real indexer and real files, no watcher involved.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hearth.indexing.filters import PathFilter
from hearth.indexing.pipeline import Indexer
from hearth.indexing.watcher import IndexWatcher, WatchBatch, plan_batch
from hearth.storage.db import connect
from hearth.storage.index_repo import IndexRepository
from hearth.storage.migrate import migrate

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "repos"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(FIXTURES / "py_small", root)
    return root


def events(root: Path, *relatives: str) -> list[tuple[object, str]]:
    """Watcher-shaped events. The change kind is ignored by design, so it is arbitrary."""
    return [(2, str(root / relative)) for relative in relatives]


# --------------------------------------------------------------------- planning


def test_an_existing_file_is_planned_for_reindex(workspace: Path) -> None:
    batch = plan_batch(events(workspace, "src/billing/models.py"), root=workspace)

    assert batch.changed == ("src/billing/models.py",)
    assert batch.deleted == ()


def test_a_missing_file_is_planned_for_removal(workspace: Path) -> None:
    batch = plan_batch(events(workspace, "src/billing/gone.py"), root=workspace)

    assert batch.deleted == ("src/billing/gone.py",)
    assert batch.changed == ()


def test_the_event_kind_is_ignored_in_favour_of_the_filesystem(workspace: Path) -> None:
    """An atomic save emits delete-then-add for what the user did once.

    Trusting the kind would remove the file from the index and re-add it — the same
    end state for twice the work, and a window where it is missing.
    """
    target = "src/billing/models.py"
    deleted_kind, added_kind = 3, 1

    batch = plan_batch(
        [(deleted_kind, str(workspace / target)), (added_kind, str(workspace / target))],
        root=workspace,
    )

    assert batch.changed == (target,)
    assert batch.deleted == ()


def test_a_path_seen_twice_is_planned_once(workspace: Path) -> None:
    """A burst repeats the same path; re-indexing it three times costs three parses."""
    batch = plan_batch(events(workspace, "src/billing/models.py") * 3, root=workspace)

    assert batch.total == 1


def test_git_internals_never_trigger_work(workspace: Path) -> None:
    """A branch switch rewrites .git, so watching it makes every checkout self-trigger."""
    batch = plan_batch(
        events(workspace, ".git/HEAD", ".git/index", ".git/refs/heads/main"), root=workspace
    )

    assert batch.empty
    assert batch.ignored == 3


@pytest.mark.parametrize(
    "noisy",
    ["node_modules/pkg/index.js", "__pycache__/models.cpython-312.pyc", ".venv/lib/x.py"],
)
def test_noisy_directories_are_ignored(workspace: Path, noisy: str) -> None:
    assert plan_batch(events(workspace, noisy), root=workspace).empty


def test_events_outside_the_workspace_are_dropped(workspace: Path, tmp_path: Path) -> None:
    """The watcher is rooted at the workspace; anything else is not ours to index."""
    batch = plan_batch([(2, str(tmp_path / "elsewhere.py"))], root=workspace)

    assert batch.empty


def test_the_path_filter_decides_what_counts(workspace: Path) -> None:
    """The watcher must agree with the indexer about what is indexable, or it will
    queue work the indexer then refuses."""
    (workspace / "notes.log").write_text("noise\n", encoding="utf-8")
    path_filter = PathFilter(exclude=["*.log"])

    batch = plan_batch(events(workspace, "notes.log"), root=workspace, path_filter=path_filter)

    assert batch.empty
    assert batch.ignored == 1


# ---------------------------------------------------------------------- applying


@pytest.fixture
def indexed(workspace: Path, tmp_path: Path):
    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    repo = IndexRepository(connection)
    indexer = Indexer(root=workspace, repository=repo)
    indexer.run()
    return indexer, repo


def test_applying_a_batch_reindexes_an_edited_file(workspace: Path, indexed) -> None:
    indexer, repo = indexed
    target = "src/billing/models.py"
    before = repo.get_file(target)
    assert before is not None

    (workspace / target).write_text("NEW = 1\n", encoding="utf-8")
    watcher = IndexWatcher(root=workspace, indexer=indexer)
    watcher.apply(plan_batch(events(workspace, target), root=workspace))

    after = repo.get_file(target)
    assert after is not None
    assert after.content_hash != before.content_hash
    assert watcher.stats.reindexed == 1


def test_applying_a_batch_removes_a_deleted_file(workspace: Path, indexed) -> None:
    indexer, repo = indexed
    target = "src/billing/payments.py"
    assert repo.get_file(target) is not None

    (workspace / target).unlink()
    watcher = IndexWatcher(root=workspace, indexer=indexer)
    watcher.apply(plan_batch(events(workspace, target), root=workspace))

    assert repo.get_file(target) is None
    assert watcher.stats.removed == 1


def test_a_failing_batch_is_accounted_for(workspace: Path) -> None:
    """A file rewritten again mid-read is the usual cause; the next batch covers it.

    Losing the other 399 files of a checkout over one racing write is a poor trade.
    """

    class Flaky:
        def index_paths(self, relatives):
            raise OSError("file vanished mid-read")

    watcher = IndexWatcher(root=workspace, indexer=Flaky())  # type: ignore[arg-type]
    batch = plan_batch(
        events(workspace, "src/billing/models.py", "src/billing/errors.py"), root=workspace
    )
    watcher.apply(batch)

    assert watcher.stats.failed == 2, "the batch is accounted for rather than lost silently"
    assert watcher.stats.reindexed == 0


# -------------------------------------------------- the acceptance criterion


def test_a_branch_switch_reindexes_only_the_changed_files(workspace: Path, tmp_path: Path) -> None:
    """The I2 criterion, driven by a real `git checkout`.

    A branch switch rewrites the working tree wholesale, so the watcher sees a burst
    covering many files. Only the ones git actually changed may be parsed — the rest come
    back byte-identical, and change detection must say so.
    """

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(workspace), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Hearth Test")
    git("add", "-A")
    git("commit", "-qm", "baseline")

    git("checkout", "-qb", "feature")
    (workspace / "src" / "billing" / "models.py").write_text("CHANGED = True\n", encoding="utf-8")
    git("commit", "-qam", "one change")

    connection = connect(tmp_path / "index.db")
    migrate(connection, database="index")
    indexer = Indexer(root=workspace, repository=IndexRepository(connection))
    indexer.run()

    # Switch back: git rewrites models.py and touches nothing else.
    git("checkout", "-q", "main")

    tracked = subprocess.run(
        ["git", "-C", str(workspace), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split()
    burst = events(workspace, *tracked)

    watcher = IndexWatcher(root=workspace, indexer=indexer)
    batch = plan_batch(burst, root=workspace)
    assert len(batch.changed) > 5, "premise: the burst covers the whole tree"

    watcher.apply(batch)

    # The criterion, measured as work rather than as outcome: git rewrote one file, so
    # exactly one may be parsed however many the burst named. Asserting on stored content
    # instead would pass even if every file had been re-parsed.
    assert watcher.stats.parsed == 1, (
        f"parsed {watcher.stats.parsed} of {len(batch.changed)} files in the burst"
    )
    assert watcher.stats.unchanged == len(batch.changed) - 1
    assert watcher.stats.failed == 0

    stored = IndexRepository(connection).get_file("src/billing/models.py")
    assert stored is not None
    assert "CHANGED" not in (workspace / "src" / "billing" / "models.py").read_text(encoding="utf-8")


def test_a_burst_of_unchanged_files_parses_nothing(workspace: Path, indexed) -> None:
    """The cheap case, and the one the first version of this module got wrong.

    Routing every path through `index_one` re-parsed all eight files of a burst in which
    nothing had changed. The tests passed anyway, because they asserted on the index's
    contents — which were correct — rather than on the work done to get them.
    """
    indexer, _repo = indexed
    every = [p.relative_to(workspace).as_posix() for p in workspace.rglob("*.py")]

    watcher = IndexWatcher(root=workspace, indexer=indexer)
    watcher.apply(plan_batch(events(workspace, *every), root=workspace))

    assert watcher.stats.parsed == 0
    assert watcher.stats.unchanged == len(every)


def test_an_empty_batch_is_not_applied() -> None:
    assert WatchBatch().empty
    assert WatchBatch(changed=("a.py",)).empty is False
