"""One book source works for complete context and exact physical retrieval."""

from pathlib import Path

import pytest

from ouroboros.reference_books import compose_book, compose_book_scoped, load_reference_book, overview_book, read_book_range


def sources():
    return {
        "docs/ARCHITECTURE.md": b"# Ouroboros v7.0.0\n\nThe body and its reasons.\n\n## Chapters\n\n- [Runtime](architecture/runtime.md)\n- [Memory][memory]\n\n[memory]: architecture/memory.md\n\n## See also\n\n[Not a member](outside.md)\n",
        "docs/architecture/runtime.md": b"# Runtime\n\nProcesses carry the work; this explains their lifetime.\n\n## Startup\n\nThe full startup mechanism.\n",
        "docs/architecture/memory.md": "# Memory\n\nПамять is continuous; details live in notes.\n\n## Sources\n\nThe full source mechanism.\n".encode(),
    }


def test_exact_source_composition_is_independent_of_checkout_and_reader(tmp_path):
    corpus = sources()
    for path, raw in corpus.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    disk = load_reference_book(tmp_path, "architecture")
    git_snapshot = load_reference_book(Path("/different/checkout"), "architecture", corpus.__getitem__)
    expected = b"\n\n".join(corpus[p] for p in corpus).decode()
    assert compose_book(disk) == compose_book(git_snapshot) == expected
    assert compose_book(disk).count("Processes carry the work") == 1
    assert str(tmp_path) not in compose_book(disk)
    assert disk.entrypoint.sha256 not in compose_book(disk)


def test_overview_is_authored_introduction_and_details_have_physical_addresses():
    book = load_reference_book(Path("unused"), "architecture", sources().__getitem__)
    view = overview_book(book)
    assert "Processes carry the work; this explains their lifetime." in view.text
    assert "Память is continuous; details live in notes." in view.text
    assert "The full startup mechanism" not in view.text
    assert view.text.count('read_file(root="system_repo", path=...)') == 1
    assert "physical Source path" in view.text
    assert "not complete chapters" in view.text
    assert not view.source_complete
    part = read_book_range(book, "docs/architecture/memory.md", 3, 1)
    assert part.text == "Память is continuous; details live in notes.\n"
    assert part.sources[0].path == "docs/architecture/memory.md"
    assert part.sources[0].span.start_line == part.sources[0].span.end_line == 3
    assert part.sources[0].sha256 == book.chapters[1].sha256
    assert not part.source_complete
    assert read_book_range(book, "docs/architecture/memory.md").source_complete
    with pytest.raises(ValueError, match="not a member"):
        read_book_range(book, "docs/outside.md")


@pytest.mark.parametrize("membership", [
    "- [One](architecture/runtime.md)\n- [Duplicate](architecture/runtime.md)",
    "- [Missing](architecture/missing.md)",
    "- [Outside](../outside.md)",
    "- [Remote](https://example.com/a.md)",
    "- [Section](architecture/runtime.md#startup)",
    "- No link",
    "",
])
def test_incomplete_membership_never_returns_a_partial_full_book(membership):
    corpus = sources()
    corpus["docs/ARCHITECTURE.md"] = f"# Book\n\nPurpose.\n\n## Chapters\n\n{membership}\n".encode()
    with pytest.raises((ValueError, KeyError)):
        load_reference_book(Path("unused"), "architecture", corpus.__getitem__)


def test_missing_authored_preamble_is_not_generated_from_body():
    corpus = sources()
    corpus["docs/architecture/runtime.md"] = b"# Runtime\n\n## Startup\n\nBody is not an introduction.\n"
    with pytest.raises(ValueError, match="introductory paragraph"):
        load_reference_book(Path("unused"), "architecture", corpus.__getitem__)


def test_a_historical_monolith_revision_still_reads_as_one_complete_legacy_source():
    """The migration is forward-only; an exact older revision must still compose."""
    monolith = b"# Book\n\nOrientation.\n\n## Runtime\n\nProcesses carry the work.\n"
    book = load_reference_book(Path("unused"), "architecture", lambda _: monolith)
    assert book.legacy and not book.chapters
    assert compose_book(book).encode() == monolith
    view = overview_book(book)
    assert "docs/ARCHITECTURE.md" in view.text
    assert "Runtime" in view.text and not view.source_complete


@pytest.mark.parametrize("book_id,path", [
    ("architecture", "docs/ARCHITECTURE.md"),
    ("development", "docs/DEVELOPMENT.md"),
])
def test_current_production_books_are_chaptered_and_composition_covers_the_closure(book_id, path):
    root = Path(__file__).resolve().parents[1]
    book = load_reference_book(root, book_id)
    assert not book.legacy and book.chapters
    closure = (book.entrypoint, *book.chapters)
    composed = compose_book(book)
    for source in closure:
        # Every declared source, whole, exactly once: a composed book is the
        # complete book or it is a lie about coverage.
        assert composed.count(source.text) == 1, source.source_path
        assert source.source_path == path or source.source_path.startswith(f"docs/{book_id}/")
    assert len(composed) >= sum(len(source.text) for source in closure)
    view = overview_book(book)
    assert view.sources[0].path == path
    for chapter in book.chapters:
        # The compact view orients by authored introduction and addresses the
        # PHYSICAL chapter, never a line of the composed book.
        assert f"Source: `{chapter.source_path}`" in view.text
    assert not view.source_complete


def test_scoped_composition_is_strict_subset_of_whole_book():
    """compose_book_scoped returns entrypoint + named chapters only, strictly shorter than the whole book."""
    repo_root = Path(__file__).resolve().parent.parent
    for relpath, book_id in [("docs/ARCHITECTURE.md", "architecture"), ("docs/DEVELOPMENT.md", "development")]:
        from ouroboros.reference_books import BOOK_ENTRYPOINTS
        book = load_reference_book(repo_root, book_id)
        assert book.chapters, f"book {book_id} has no chapters"
        first = book.chapters[0]
        scoped = compose_book_scoped(book, (first.source_path,))
        whole = compose_book(book)
        assert len(scoped) < len(whole), f"scoped is not shorter than whole for {book_id}"
        assert first.text[:80] in scoped  # selected chapter content is present
        if len(book.chapters) > 1:
            other = book.chapters[1]
            assert other.text[:80] not in scoped  # non-selected chapter content absent


def test_scoped_composition_rejects_unknown_chapter():
    """compose_book_scoped raises ValueError when given a path not declared in the book."""
    repo_root = Path(__file__).resolve().parent.parent
    book = load_reference_book(repo_root, "architecture")
    with pytest.raises(ValueError, match="unknown chapter path"):
        compose_book_scoped(book, ("docs/architecture/nonexistent-chapter.md",))


def test_scoped_book_falls_back_to_whole_book_on_bad_chapter():
    """_load_scoped_book falls back to the whole book (superset) when the scoped selection is invalid."""
    import ouroboros.skill_review_prompt as srp

    repo_root = Path(__file__).resolve().parent.parent
    for relpath in srp._SKILL_REVIEW_BOOK_SCOPE:
        original = srp._SKILL_REVIEW_BOOK_SCOPE[relpath]
        try:
            srp._SKILL_REVIEW_BOOK_SCOPE[relpath] = ("docs/architecture/nonexistent-chapter.md",)
            result = srp._load_scoped_book(repo_root, relpath)
            whole = srp._load_governance_artifact(repo_root, relpath)
            assert result == whole, f"fallback for {relpath} did not return the whole book"
        finally:
            srp._SKILL_REVIEW_BOOK_SCOPE[relpath] = original
