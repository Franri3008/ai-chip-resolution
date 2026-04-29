import json
import os
import re

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json");

with open(data_path, encoding="utf-8") as f:
    data = json.load(f);

# Capture owner/repo and the trailing path so asset URLs can be filtered.
GITHUB_RE = re.compile(
    r'https?://github\.com/([\w.\-]+)/([\w.\-]+?)(?=$|[/?#\s")\']|\.git)([^\s")\']*)',
    re.IGNORECASE,
)

# Trailing-path components that mean "this URL points at a file/asset", not the
# repo home. E.g. `<user>/<repo>/raw/refs/heads/main/videos/foo.mp4` is just an
# asset host — selecting it as main_github sends the classifier chasing noise.
ASSET_PATH_RE = re.compile(
    r'^/(?:raw|blob|tree|commits?|issues?|pull|wiki|releases|discussions|'
    r'sponsors|actions|security|network|insights|projects|labels|milestones|'
    r'compare|graphs|stargazers|watchers|forks|contributors)\b',
    re.IGNORECASE,
)

# Common non-source repo names — skip these even though they have owner/repo.
JUNK_REPOS = {"sponsors", "users", "orgs", "search", "topics", "marketplace", "settings"}


def _normalize_github(text):
    """Extract canonical https://github.com/<owner>/<repo> URLs, dropping asset
    paths (/raw/, /blob/, …)."""
    out = []
    for m in GITHUB_RE.finditer(text):
        owner, repo, tail = m.group(1), m.group(2), m.group(3) or ""
        if owner.lower() in JUNK_REPOS:
            continue
        if ASSET_PATH_RE.match(tail):
            continue
        repo = re.sub(r'\.git$', '', repo, flags=re.IGNORECASE)
        repo = repo.strip(".,;:")
        if not repo:
            continue
        out.append(f"https://github.com/{owner}/{repo}")
    return list(dict.fromkeys(out))


for model in data:
    model["github_links"] = _normalize_github(model.get("modelcard", "") or "")

with open(data_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False);
