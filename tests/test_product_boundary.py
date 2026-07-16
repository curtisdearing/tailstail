from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tailstail_has_only_fantasy_production_workflow():
    workflow = ROOT / ".github" / "workflows" / "fantasy-weekly.yml"
    text = workflow.read_text()
    assert workflow.exists()
    assert not (ROOT / ".github" / "workflows" / "live-weekly.yml").exists()
    assert "fantasy-model-state" in text
    assert "tailstail-fantasy-production" in text
    assert "ODDS_API_KEY" not in text
    assert "DISCORD_WEBHOOK_URL" not in text
    assert "actions/deploy-pages" in text
    assert "branches: [main]" in text
    assert "- site/**" in text
    assert "github.event_name == 'push'" in text
    assert "site/index.html" in text
    assert (ROOT / "site" / "index.html").exists()


def test_tailstail_readme_does_not_claim_fablesfable_operations():
    text = (ROOT / "README.md").read_text()
    assert "curtisdearing.github.io/fablesfable" not in text
    assert "ODDS_API_KEY" not in text
    assert "fantasy-model-state" in text
