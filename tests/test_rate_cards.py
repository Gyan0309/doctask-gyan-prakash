"""A rate card is one term with many roles, not many terms competing to be one.

Found by running the system against realistically drafted agreements rather than the
seed corpus. A law firm engagement letter carries five timekeepers in one table; a
facilities agreement carries core, out-of-hours and holiday columns. Before this, every
row of such a table was an equal claimant to be *the* vendor's `hourly_rate`, which
produced two failures at once:

* **C(n,2) false conflicts.** Five timekeepers is ten conflict candidates, every one a
  pair of rows from the same table. Measured: 36 of 41 candidates on a three-vendor
  corpus were a rate card compared against itself.
* **An arbitrary governing value.** When facts tie on effective date and precedence the
  winner is whichever sorted last, so the register's headline rate for a vendor was
  decided by input order. Measured: it chose the *paralegal* rate three times out of
  three, with the partner rate sitting two rows above it.

The second is the worse one, and the reason these tests exist. A miss is visible; this
was a confident wrong number carrying a genuine citation that **passed verification** —
the verifier checks the cited passage contains the value, and the paralegal row really
does say $225. No amount of verification catches right-value-wrong-meaning; only
modelling the scope does.
"""

from __future__ import annotations

from datetime import date
from uuid import uuid4

from domain.conflicts import detect
from domain.reconcile import FactView, reconcile


def _rate(value, *, qualifier=None, subject="Brightmoor LLP", kind="msa", day=None):
    """One rate-card row, scoped the way `_scope_for` scopes it."""
    return FactView(
        fact_id=uuid4(),
        predicate="hourly_rate",
        subject=subject,
        value_raw=value,
        value_norm=None,
        unit=None,
        effective_date=day or date(2025, 1, 1),
        document_id=uuid4(),
        document_kind=kind,
        scope=qualifier.casefold() if qualifier else None,
    )


class TestARateCardDoesNotFightItself:
    def test_five_timekeepers_are_five_terms_not_one_contest(self) -> None:
        """The exact shape from the engagement letter: one table, five grades."""
        card = [
            _rate("$840", qualifier="Partner"),
            _rate("$610", qualifier="Senior Associate"),
            _rate("$430", qualifier="Associate"),
            _rate("$300", qualifier="Trainee"),
            _rate("$225", qualifier="Paralegal"),
        ]

        resolutions = reconcile(card)

        assert len(resolutions) == 5, (
            "each grade is its own term; collapsing them makes five rates compete to be one"
        )
        assert {r.governing.value_raw for r in resolutions} == {
            "$840", "$610", "$430", "$300", "$225",
        }
        # Nothing superseded anything: these do not disagree, they describe different rows.
        assert all(not r.superseded for r in resolutions)

    def test_a_rate_card_raises_no_conflicts(self) -> None:
        """C(5,2) = 10 candidates before this, all false."""
        card = [
            _rate("$840", qualifier="Partner"),
            _rate("$610", qualifier="Senior Associate"),
            _rate("$430", qualifier="Associate"),
            _rate("$300", qualifier="Trainee"),
            _rate("$225", qualifier="Paralegal"),
        ]

        assert detect(card) == []

    def test_the_headline_rate_is_not_decided_by_sort_order(self) -> None:
        """The failure that mattered: the paralegal rate reported as the vendor's rate.

        With the standard rate extracted unqualified alongside the card, the agreement's
        own rate is a distinct term and cannot be displaced by a table row.
        """
        facts = [
            _rate("$840", qualifier="Partner"),
            _rate("$225", qualifier="Paralegal"),
            _rate("$430"),  # the stated standard rate, agreement-wide
        ]

        resolutions = reconcile(facts)
        agreement_wide = [r for r in resolutions if r.scope is None]

        assert len(agreement_wide) == 1
        assert agreement_wide[0].governing.value_raw == "$430"

    def test_shift_columns_are_scopes_too(self) -> None:
        """Facilities agreements price the same work by shift: core, out-of-hours,
        holiday. Same shape, different vocabulary."""
        facts = [
            _rate("$68", qualifier="core hours", subject="Ardent"),
            _rate("$102", qualifier="out-of-hours", subject="Ardent"),
            _rate("$136", qualifier="holiday", subject="Ardent"),
        ]

        assert len(reconcile(facts)) == 3
        assert detect(facts) == []

    def test_two_genuinely_competing_rates_still_conflict(self) -> None:
        """The fix must not buy its correctness by going quiet.

        Two agreement-wide rates for one vendor, same date, no scope to tell them
        apart, is a real contradiction and must still be reported.
        """
        facts = [_rate("$195"), _rate("$210")]

        assert len(reconcile(facts)) == 1, "unscoped rates share one term"
        assert detect(facts), "a real disagreement must still be caught"


class TestScopeLabelsAgreeAcrossDocuments:
    """Case-folding alone was not enough, and stage-3 testing proved it.

    The agreement writes `out-of-hours`; the rate notice writes `Out of Hours`. Folded
    only for case, those are two scopes — so the notice could never supersede the rate
    it existed to change, and both sat in the register as live values that then
    conflicted with each other.
    """

    def test_punctuation_and_case_do_not_split_one_column(self) -> None:
        from domain.reconcile import canonical_scope

        assert canonical_scope("out-of-hours") == canonical_scope("Out of Hours")
        assert canonical_scope("All trades — standard") == canonical_scope(
            "all trades standard"
        )

    def test_digits_are_kept_so_grades_stay_apart(self) -> None:
        from domain.reconcile import canonical_scope

        assert canonical_scope("Grade 3") != canonical_scope("Grade 4")

    def test_the_same_column_across_two_documents_supersedes_itself(self) -> None:
        """The consequence, end to end: an agreement rate and a later notice for the
        same column resolve to one term with one governing value."""
        facts = [
            _rate("$102", qualifier="out-of-hours", subject="Ardent", day=date(2024, 1, 1)),
            _rate(
                "$107.10",
                qualifier="Out of Hours",
                subject="Ardent",
                kind="amendment",
                day=date(2026, 1, 1),
            ),
        ]

        resolutions = reconcile(facts)

        assert len(resolutions) == 1, "one column, however each document spells it"
        assert resolutions[0].governing.value_raw == "$107.10"
        assert detect(facts) == []


class TestConflictDetectionFoldsTheSameWay:
    """Reconciliation and detection must agree on identity, or the register is coherent
    while the findings list is not. Detection grouped on raw strings after
    reconciliation had been taught to fold, so a rate card still fought itself across
    documents — 25 false conflicts survived the first fix because of it.
    """

    def test_a_shouted_vendor_name_does_not_conflict_with_itself(self) -> None:
        facts = [
            _rate("$68", subject="Ardent Facilities Management LLC", day=date(2026, 1, 1)),
            _rate("$68", subject="ARDENT FACILITIES MANAGEMENT LLC", day=date(2026, 1, 1)),
        ]

        assert detect(facts) == []

    def test_one_column_spelled_two_ways_does_not_conflict_with_itself(self) -> None:
        facts = [
            _rate("$102", qualifier="out-of-hours", subject="Ardent", day=date(2026, 1, 1)),
            _rate("$102", qualifier="Out of Hours", subject="Ardent", day=date(2026, 1, 1)),
        ]

        assert detect(facts) == []


class TestAGridNeedsBothItsLabels:
    """Stage-3 testing found the first fix was one-dimensional.

    A trades rate table is a grid — four trades down, three shifts across. The extractor
    captured the shift and dropped the trade, so `Out of Hours` in a single document
    held four different values and produced C(4,2) = 6 conflicts. The scope has to name
    the *cell*, not the column.
    """

    def test_a_shift_alone_is_not_a_unique_cell(self) -> None:
        """The failing shape, stated directly: same column, four trades, one scope."""
        grid = [
            _rate("$141.00", qualifier="Out of Hours", subject="Ardent"),
            _rate("$118.50", qualifier="Out of Hours", subject="Ardent"),
            _rate("$102.00", qualifier="Out of Hours", subject="Ardent"),
            _rate("$61.50", qualifier="Out of Hours", subject="Ardent"),
        ]

        assert len(detect(grid)) == 6, (
            "four values in one scope is six pairs — this is what the prompt fix "
            "must stop producing at the source"
        )

    def test_naming_both_axes_makes_every_cell_its_own_term(self) -> None:
        grid = [
            _rate("$141.00", qualifier="Electrician — out of hours", subject="Ardent"),
            _rate("$118.50", qualifier="Plumber — out of hours", subject="Ardent"),
            _rate("$102.00", qualifier="General Operative — out of hours", subject="Ardent"),
            _rate("$61.50", qualifier="Apprentice — out of hours", subject="Ardent"),
        ]

        assert len(reconcile(grid)) == 4
        assert detect(grid) == []


class TestFindingsNameTheVendorAsWritten:
    def test_a_finding_does_not_report_the_folding_key(self) -> None:
        """Folding is an implementation detail. A finding that says "ardent facilities
        management llc" has leaked one into text a person is meant to act on."""
        facts = [
            _rate("$68", subject="Ardent Facilities Management LLC"),
            _rate("$75", subject="Ardent Facilities Management LLC"),
        ]

        found = detect(facts)

        assert found
        assert found[0].subject == "Ardent Facilities Management LLC"


class TestWhereTheScopeComesFrom:
    """The origin of the bug, tested directly.

    `reconcile` and `detect` both honoured scope already. The fault was upstream: nothing
    ever *set* one outside a SOW, so every rate-card row arrived agreement-wide and the
    downstream logic was correct about facts that were wrong. Testing only the
    downstream behaviour would have left the actual defect uncovered.
    """

    class _Fact:
        def __init__(self, qualifier=None, document_id="doc-1"):
            self.qualifier = qualifier
            self.document_id = document_id

    def test_a_rate_card_row_is_scoped_by_its_label(self) -> None:
        from services.graph import _scope_for

        assert _scope_for("msa", self._Fact(qualifier="Partner")) == "partner"

    def test_an_agreement_wide_term_has_no_scope(self) -> None:
        from services.graph import _scope_for

        assert _scope_for("msa", self._Fact()) is None

    class _Document:
        def __init__(self, uri):
            self.uri = uri

    def test_a_sow_rate_card_is_scoped_by_both(self) -> None:
        """A SOW's own rate card needs both dimensions: the engagement it belongs to and
        the grade within it. Either alone would merge rows that should stay apart."""
        from services.graph import _scope_for

        scope = _scope_for(
            "sow",
            self._Fact(qualifier="Senior"),
            self._Document("/data/inbox/talus-sow-eu-west-migration.md"),
        )

        assert scope == "talus sow eu west migration::senior"

    def test_the_engagement_is_named_not_numbered(self) -> None:
        """The scope reaches a human: it is appended to the register row's heading and to
        the published document. It used to be the document's raw UUID, so a reviewer read
        "Payment Terms (days) — 89Bcf37E 4753 4270 B534 3E97Cc29A320" — a label nobody can
        read is a label that gets ignored, which undoes the point of scoping."""
        from services.graph import _scope_for

        scope = _scope_for(
            "sow", self._Fact(), self._Document("/data/inbox/talus-sow-eu-west.md")
        )

        assert scope == "talus sow eu west"

    def test_two_engagements_stay_apart(self) -> None:
        from services.graph import _scope_for

        first = _scope_for("sow", self._Fact(), self._Document("/x/sow-alpha.md"))
        second = _scope_for("sow", self._Fact(), self._Document("/x/sow-beta.md"))

        assert first != second

    def test_a_sow_with_no_document_still_gets_a_scope(self) -> None:
        """It must not silently become agreement-wide, which is the failure the whole
        scoping mechanism exists to prevent."""
        from services.graph import _scope_for

        assert _scope_for("sow", self._Fact(), None) == "engagement"

    def test_scoping_is_case_insensitive(self) -> None:
        """"Partner" and "PARTNER" in two documents are one grade, not two."""
        from services.graph import _scope_for

        assert _scope_for("msa", self._Fact(qualifier="PARTNER")) == _scope_for(
            "msa", self._Fact(qualifier="partner")
        )


class TestOneVendorStaysOneVendor:
    def test_capitalisation_does_not_split_a_vendor_in_two(self) -> None:
        """A rate-adjustment notice shouted the vendor name; every other document used
        title case. Subjects were grouped by exact string, so the register carried both
        as separate vendors asserting different current rates — and the notice could
        never supersede the agreement it existed to amend.
        """
        facts = [
            _rate("$68", subject="Ardent Facilities Management LLC", day=date(2024, 1, 1)),
            _rate(
                "$71.40",
                subject="ARDENT FACILITIES MANAGEMENT LLC",
                kind="amendment",
                day=date(2026, 2, 1),
            ),
        ]

        resolutions = reconcile(facts)

        assert len(resolutions) == 1, "one company, one term"
        assert resolutions[0].governing.value_raw == "$71.40", (
            "the later amendment must supersede the agreement it amends"
        )
        assert len(resolutions[0].superseded) == 1

    def test_the_register_does_not_read_as_a_folding_key(self) -> None:
        """Folding is for grouping only. The register must show a name a person wrote,
        never `ardent facilities management llc`."""
        facts = [
            _rate("$68", subject="Ardent Facilities Management LLC", day=date(2024, 1, 1)),
            _rate(
                "$71.40",
                subject="ARDENT FACILITIES MANAGEMENT LLC",
                kind="amendment",
                day=date(2026, 2, 1),
            ),
        ]

        assert reconcile(facts)[0].subject in {
            "Ardent Facilities Management LLC",
            "ARDENT FACILITIES MANAGEMENT LLC",
        }

    def test_a_shouted_variant_does_not_win_the_display(self) -> None:
        """Chosen deliberately rather than by accident of sort order: a company's name in
        a contract is not shouted, and the SHOUTED spelling is reliably the outlier a
        letterhead or a header row introduced."""
        facts = [
            _rate("$68", subject="Ardent Facilities Management LLC", day=date(2024, 1, 1)),
            _rate(
                "$71.40",
                subject="ARDENT FACILITIES MANAGEMENT LLC",
                kind="amendment",
                day=date(2026, 2, 1),
            ),
        ]

        assert reconcile(facts)[0].subject == "Ardent Facilities Management LLC"

    def test_extra_whitespace_does_not_split_a_vendor(self) -> None:
        facts = [
            _rate("$68", subject="Ardent  Facilities   Management LLC"),
            _rate("$68", subject="Ardent Facilities Management LLC"),
        ]

        assert len(reconcile(facts)) == 1

    def test_one_vendor_reads_the_same_way_on_every_row(self) -> None:
        """The half that folding identity did not fix. Display was decided per resolution
        — "the governing document's spelling wins" — which has no answer for a row with
        nothing governing it, and fell back to whichever fact sorted first. A live register
        had 60 rows reading `Ardent Facilities Management LLC` and one reading
        `ARDENT FACILITIES MANAGEMENT LLC`, because that row was unsupported.
        """
        facts = [
            _rate("$68", subject="Ardent Facilities Management LLC", day=date(2024, 1, 1)),
            _rate(
                "$71.40",
                subject="Ardent Facilities Management LLC",
                qualifier="Supervisor",
                day=date(2024, 1, 1),
            ),
            # An observation with no agreement behind it: the row that used to inherit
            # the shouted spelling.
            _rate(
                "$99",
                subject="ARDENT FACILITIES MANAGEMENT LLC",
                kind="invoice",
                qualifier="Apprentice",
                day=date(2026, 3, 1),
            ),
        ]

        spellings = {r.subject for r in reconcile(facts)}

        assert len(spellings) == 1, f"one company must read one way, got {spellings}"
        assert spellings == {"Ardent Facilities Management LLC"}

    def test_a_vendor_named_only_in_invoices_still_gets_a_name(self) -> None:
        """Governing facts are weighted, not required. Excluding correspondence entirely
        would leave an invoice-only vendor with no display name at all."""
        facts = [
            _rate("$99", subject="Kestrel Cold Chain Ltd", kind="invoice"),
        ]

        assert reconcile(facts)[0].subject == "Kestrel Cold Chain Ltd"

    def test_different_companies_are_not_merged(self) -> None:
        """The fix must not overreach. Merging two companies' obligations silently is
        worse than the split it replaces, so matching is case and whitespace only —
        never fuzzy."""
        facts = [
            _rate("$68", subject="Northwind Analytics LLC"),
            _rate("$72", subject="Northwind Analytics Ltd"),
        ]

        assert len(reconcile(facts)) == 2


class TestASectionKeyIsIdentityNotDisplay:
    """The half of the vendor fix that was missed the first time.

    Folding the *grouping* made supersession work and left the register still showing
    one company twice, because the section key was built from the governing document's
    spelling. A live corpus carried both `Ardent Facilities Management LLC::auto_renew_months`
    and `ARDENT FACILITIES MANAGEMENT LLC::hourly_rate::apprentice`, which sort apart
    and read as two vendors.
    """

    def test_the_key_does_not_carry_the_documents_capitalisation(self) -> None:
        shouted = reconcile([_rate("$68", subject="ARDENT FACILITIES MANAGEMENT LLC")])
        titled = reconcile([_rate("$68", subject="Ardent Facilities Management LLC")])

        assert shouted[0].key() == titled[0].key()

    def test_the_key_folds_while_the_display_stays_written(self) -> None:
        """Identity folds; presentation does not. Both must hold at once, which is why
        they are two fields rather than one."""
        facts = [
            _rate("$68", subject="Ardent Facilities Management LLC", day=date(2024, 1, 1)),
            _rate(
                "$71.40",
                subject="ARDENT FACILITIES MANAGEMENT LLC",
                kind="amendment",
                day=date(2026, 2, 1),
            ),
        ]

        resolution = reconcile(facts)[0]

        assert resolution.subject == "Ardent Facilities Management LLC"
        assert "ARDENT" not in resolution.key()

    def test_a_new_spelling_becoming_governing_does_not_rename_the_section(self) -> None:
        """The second, quieter fault. A renamed key is a removal plus an insertion, so
        an ordinary amendment registered as churn in the incrementality figures and, on
        publish, deleted a section and wrote a new one instead of editing it."""
        before = reconcile(
            [_rate("$68", subject="Ardent Facilities Management LLC", day=date(2024, 1, 1))]
        )
        after = reconcile(
            [
                _rate(
                    "$68", subject="Ardent Facilities Management LLC", day=date(2024, 1, 1)
                ),
                _rate(
                    "$71.40",
                    subject="ARDENT FACILITIES MANAGEMENT LLC",
                    kind="amendment",
                    day=date(2026, 2, 1),
                ),
            ]
        )

        assert before[0].key() == after[0].key(), "an amendment must edit, not replace"

    def test_two_companies_still_get_two_keys(self) -> None:
        a = reconcile([_rate("$68", subject="Northwind Analytics LLC")])
        b = reconcile([_rate("$72", subject="Northwind Analytics Ltd")])

        assert a[0].key() != b[0].key()

    def test_scope_is_part_of_the_identity(self) -> None:
        rows = reconcile(
            [
                _rate("$840", subject="Brightmoor Legal LLP", qualifier="Partner"),
                _rate("$225", subject="Brightmoor Legal LLP", qualifier="Paralegal"),
            ]
        )

        assert len({r.key() for r in rows}) == 2

    def test_a_hand_built_resolution_still_has_a_coherent_identity(self) -> None:
        """Nothing should have to remember to fold. A Resolution constructed without a
        subject_key derives one rather than keying on an empty string."""
        from domain.reconcile import Resolution

        resolution = Resolution(
            subject="ARDENT FACILITIES MANAGEMENT LLC",
            predicate="hourly_rate",
            scope=None,
            governing=None,
        )

        assert resolution.key() == "ardent facilities management llc::hourly_rate"


class TestAnInvoiceIsJudgedGradeByGrade:
    """The live defect, end to end. A rate card is many roles; an invoice bills against
    those roles. The temporal-precedence comparator grouped without scope, so an invoice
    line for one grade was measured against whichever grade's governing rate sorted last.
    Because the grades differ, a correctly-billed invoice manufactured a finding on every
    line, and the direction fix printed an exact, fabricated number for each: "$720
    higher" was $905 (managing partner) minus $185 (litigation support).

    `reconcile` and comparator 1 already scoped their grouping; this comparator did not —
    the same "folds in one place, not its peer" shape the vendor fix had. These tests
    drive `detect()`, so they cover the comparator as it is actually reached.
    """

    def test_an_invoice_billing_every_grade_correctly_raises_nothing(self) -> None:
        """Half the value of a detector is what it declines to report. Every line billed
        at its agreed grade rate, across a five-figure spread of grades, is silence."""
        card = [
            _rate("$905", qualifier="Managing Partner"),
            _rate("$560", qualifier="Senior Associate"),
            _rate("$185", qualifier="Litigation Support"),
        ]
        invoice = [
            _rate("$905", qualifier="Managing Partner", kind="invoice", day=date(2026, 3, 1)),
            _rate("$560", qualifier="Senior Associate", kind="invoice", day=date(2026, 3, 1)),
            _rate("$185", qualifier="Litigation Support", kind="invoice", day=date(2026, 3, 1)),
        ]

        assert detect(card + invoice) == []

    def test_overbilling_one_grade_is_caught_against_that_grade(self) -> None:
        """Real overbilling survives the fix, and is measured against the right grade."""
        card = [
            _rate("$905", qualifier="Managing Partner"),
            _rate("$185", qualifier="Litigation Support"),
        ]
        invoice = [
            _rate("$950", qualifier="Managing Partner", kind="invoice", day=date(2026, 3, 1)),
            _rate("$185", qualifier="Litigation Support", kind="invoice", day=date(2026, 3, 1)),
        ]

        found = detect(card + invoice)

        assert len(found) == 1
        assert "$905" in found[0].detail and "$950" in found[0].detail
        assert "$185" not in found[0].detail, "not measured against a different grade"
