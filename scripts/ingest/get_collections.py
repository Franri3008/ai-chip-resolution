"""Augment arxiv_links by walking owner-trusted HuggingFace Collections.

Many model cards omit the paper link, but the model still appears in a
Collection (sidebar on the HF model page) owned by the same org, and that
Collection lists the paper. We discover those papers and propagate them
to every queried model that appears in such a Collection.

Trust filter: a Collection is trusted iff its owner equals the model's
namespace (case-insensitive). This avoids third-party "favorites"
Collections attaching unrelated papers.

Direction: we query `list_collections(owner=NS)` for each unique namespace
in the dataset (not `list_collections(item=...)`). The latter is sorted by
trending and buries owner-matched Collections beneath community favorites
for popular models — by the time we'd hit a moonshotai-owned Collection
for a Kimi model, we'd be hundreds of pages deep.

Curation guard: Collections with more than MAX_PAPERS_PER_COLLECTION
papers are skipped — these tend to be a lab's full publication index
rather than the specific paper for the queried model.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _keys import hf_token  # noqa: E402

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import HfApi, login
from huggingface_hub import logging as hf_logging
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm

hf_logging.set_verbosity_error()

DATA_PATH = Path(__file__).parent.parent.parent / "database" / "modelcards.json"

MAX_COLLECTIONS_PER_NAMESPACE = 200
MAX_PAPERS_PER_COLLECTION = 5
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _arxiv_url(item_id: str) -> str:
    return f"https://arxiv.org/abs/{item_id}"


def _namespace(model_id: str) -> str:
    """Extract namespace from a model_id; legacy root models like
    'bert-base-uncased' have no slash and return as-is (matching nothing
    in practice, since there's no owner with that name)."""
    return model_id.split("/", 1)[0]


def _list_namespace_collections(api: HfApi, namespace: str) -> list[str]:
    """Return all collection slugs owned by `namespace`. Returns [] on 404
    or transient errors. Capped at MAX_COLLECTIONS_PER_NAMESPACE."""
    out = []
    try:
        for i, coll in enumerate(api.list_collections(owner=namespace)):
            if i >= MAX_COLLECTIONS_PER_NAMESPACE:
                break
            out.append(coll.slug)
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return []
        return []
    except Exception:
        return []
    return out


def _fetch_collection(api: HfApi, slug: str, attempts: int = 3, base_delay: float = 1.0):
    for k in range(attempts):
        try:
            return api.get_collection(slug)
        except HfHubHTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in RETRYABLE_STATUS and k < attempts - 1:
                time.sleep(base_delay * (2 ** k))
                continue
            return None
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16,
                        help="Concurrent HF API calls (default: 16).")
    args = parser.parse_args()

    token = hf_token()
    login(token=token)
    api = HfApi(token=token)

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    model_ids = [m["id"] for m in data]
    queried_set = set(model_ids)

    # Phase 1: per-namespace -> list of collection slugs.
    # Many models share a namespace, so this concentrates work.
    namespaces = sorted({_namespace(mid) for mid in model_ids})
    ns_to_slugs: dict[str, list[str]] = {}

    def _phase1(ns: str):
        return ns, _list_namespace_collections(api, ns)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_phase1, ns) for ns in namespaces]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="Listing namespace collections"):
            ns, slugs = fut.result()
            ns_to_slugs[ns] = slugs

    # Phase 2: dedup slugs across namespaces and fetch each Collection once.
    all_slugs = {s for slugs in ns_to_slugs.values() for s in slugs}
    slug_cache: dict[str, dict] = {}

    def _phase2(slug: str):
        coll = _fetch_collection(api, slug)
        if coll is None:
            return slug, None
        papers, models = [], []
        for it in (getattr(coll, "items", None) or []):
            t = getattr(it, "item_type", None)
            iid = getattr(it, "item_id", None)
            if not iid:
                continue
            if t == "paper":
                papers.append(iid)
            elif t == "model":
                models.append(iid)
        return slug, {"papers": papers, "models": models}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_phase2, s) for s in all_slugs]
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="Fetching collection items"):
            slug, info = fut.result()
            if info is None:
                continue
            if not info["papers"]:
                continue
            if len(info["papers"]) > MAX_PAPERS_PER_COLLECTION:
                continue
            slug_cache[slug] = info

    # Phase 3: propagate. For every namespace, walk its trusted Collections
    # and attach each Collection's papers to any queried model listed in it.
    propagated: dict[str, set[str]] = {mid: set() for mid in queried_set}
    for ns, slugs in ns_to_slugs.items():
        for slug in slugs:
            info = slug_cache.get(slug)
            if not info:
                continue
            for sib in info["models"]:
                if sib in queried_set:
                    propagated[sib].update(info["papers"])

    # Phase 4: write back. New `collection_arxiv_links` field for provenance,
    # then merge into existing `arxiv_links` (preserve order, append new only).
    n_new_models, n_new_links = 0, 0
    for m in data:
        new_ids = sorted(propagated.get(m["id"], set()))
        new_urls = [_arxiv_url(i) for i in new_ids]
        m["collection_arxiv_links"] = new_urls

        existing = list(m.get("arxiv_links") or [])
        existing_set = set(existing)
        for url in new_urls:
            if url not in existing_set:
                existing.append(url)
                existing_set.add(url)
                n_new_links += 1
        m["arxiv_links"] = existing
        if new_urls:
            n_new_models += 1

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"Collections: {n_new_models}/{len(data)} models gained papers; "
        f"{n_new_links} new arxiv links merged "
        f"({len(slug_cache)} trusted collections, "
        f"{len(namespaces)} namespaces, {len(all_slugs)} total slugs scanned)"
    )


if __name__ == "__main__":
    main()
