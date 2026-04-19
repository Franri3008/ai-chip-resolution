import json
import os
import re

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "database", "modelcards.json");

with open(data_path, encoding="utf-8") as f:
    data = json.load(f);

github_re = re.compile(r'https?://github\.com/[\w.\-]+/[\w.\-]+');

for model in data:
    links = github_re.findall(model["modelcard"])
    model["github_links"] = list(dict.fromkeys(links))

with open(data_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False);
