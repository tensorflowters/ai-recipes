import pytest
from ambiguity_clarity.openai_costs import enable_cost_tracking, disable_cost_tracking, print_cost_summary


@pytest.fixture(autouse=True)
def _integration_cost_tracker(request):
    """Enable cost tracking automatically for tests marked with `integration`."""
    if request.node.get_closest_marker("integration") is None:
        yield  # not an integration test
        return

    enable_cost_tracking()
    try:
        yield
    finally:
        # Print and persist cost summary
        summary = print_cost_summary()
        import json
        import pathlib

        report_path = pathlib.Path.cwd() / "integration_cost_report.json"
        try:
            with open(report_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"💾 Cost report written to {report_path}")
        except Exception as exc:
            print(f"⚠️ Could not write cost report: {exc}")

        disable_cost_tracking()
