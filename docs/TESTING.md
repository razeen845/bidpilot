# How good are the tests?

Checked 2026-10-18.

- **131 automated tests**, all passing (`pytest`), plus `ruff` lint. CI runs both on every push.
- **93% line coverage** of the product code (optimizer, metrics and simulator 100%, guardrails 98%).
- **Mutation testing: 25 / 25 planted bugs caught.** `scripts/mutation_test.py` copies the project, plants one
  realistic bug at a time in the safety-critical logic and checks that the test suite fails.

The first run caught 22 / 25: three tests were too weak (cautious raises, unverified consultant actions,
reviewed-analysis priority). They were strengthened and the full run repeated. A timing-based test that could
fail on a busy machine was replaced by one that counts overlapping AI calls.

| Planted bug | Result | Caught by |
|---|---|---|
| optimizer picks the WORST bid | caught | `tests/test_engine.py::test_optimizer_moves_toward_profit` |
| +-20%/day safety limit removed | caught | `tests/test_engine.py::test_limit_bid` |
| stock guard cuts 10% instead of 50% | caught | `tests/test_engine.py::test_stock_guard_rules` |
| out-of-stock products not paused | caught | `tests/test_engine.py::test_stock_guard_rules` |
| waste rule ignores statistical evidence | caught | `tests/test_engine.py::test_waste_pause_needs_strong_evidence` |
| big changes no longer need approval | caught | `tests/test_engine.py::test_approval_flags` |
| proven winners not protected | caught | `tests/test_practitioner.py::test_proven_targets_protected` |
| daily budget not enforced | caught | `tests/test_platform.py::test_budget_never_exceeded` |
| paused ads still get clicks | caught | `tests/test_phase13.py::test_experiment_lifecycle_on_live_account` |
| number check accepts any number | caught | `tests/test_consultant.py::test_consultant_report_is_cited_and_checked` |
| explanations may talk about price | caught | `tests/test_llm.py::test_explainer_guards_meaning` |
| citation check accepts unknown sources | caught | `tests/test_research.py::test_registry_rejects[Something happened [D7].-ids1-unknown source]` |
| web quotes not verified against the page | caught | `tests/test_research.py::test_quote_check_rejects_invented_or_long` |
| sandbox allows any import | caught | `tests/test_sandbox.py::test_policy_blocks[import os]` |
| sandbox allows dunder attributes | caught | `tests/test_sandbox.py::test_policy_blocks[x = stats.__class__]` |
| hidden market columns reach the sandbox | caught | `tests/test_sandbox.py::test_run_code_errors_reported` |
| incrementality estimate sign wrong | caught | `tests/test_phase13.py::test_analyze_detects_cannibalization` |
| rollback tests the wrong direction | caught | `tests/test_phase13.py::test_rollback_reverts_a_bad_bid` |
| cautious raises disabled | caught | `tests/test_engine.py::test_cautious_raise_ignores_one_lucky_sale` |
| REJECTED changes get applied | caught | `tests/test_orchestrator.py::test_manual_approval_flow` |
| unverified consultant bid actions applied | caught | `tests/test_consultant.py::test_unverified_bid_action_never_changes_bids` |
| ACOS computed upside down | caught | `tests/test_practitioner.py::test_acos_and_cvr_metrics` |
| restocked products never resume | caught | `tests/test_engine.py::test_stock_guard_rules` |
| junk keywords convert like normal ones | caught | `tests/test_generate.py::test_junk_keywords_near_zero_and_patterned` |
| reviewed analyses bypassed for AI code | caught | `tests/test_sandbox.py::test_known_questions_use_reviewed_code_even_with_llm` |

Real-model checks (Qwen2.5-3B on an AMD Radeon laptop GPU, llama.cpp Vulkan) found further issues that unit
tests with fake LLMs could not (numbers attached to the wrong meaning, "bid" read as "price", code-in-JSON
failures, empty citations, wrong currency); each now has a guard and a regression test.
