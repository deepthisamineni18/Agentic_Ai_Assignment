"""
Generates a mock corpus of ~10,000 pre-crawled URLs + snippets used to simulate
a search API (so the Searcher agent doesn't need live internet access).

Run once: `python -m research_pipeline.data.generate_mock_corpus`
Produces `research_pipeline/data/corpus.json`.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

TOPICS = [
    "artificial intelligence", "climate change", "quantum computing", "renewable energy",
    "gene editing", "space exploration", "cybersecurity", "blockchain", "electric vehicles",
    "machine learning", "neuroscience", "public health policy", "supply chain logistics",
    "urban planning", "ocean conservation", "robotics", "biotechnology", "economics",
    "education technology", "cloud computing", "autonomous vehicles", "5G networks",
    "carbon capture", "vaccine development", "nuclear fusion", "data privacy",
    "remote work trends", "food security", "water scarcity", "semiconductor manufacturing",
]

ADJECTIVES = ["emerging", "recent", "groundbreaking", "controversial", "global", "regional",
              "innovative", "long-term", "short-term", "experimental", "large-scale", "novel"]

NOUNS = ["research", "developments", "policy", "breakthrough", "trends", "analysis",
         "investment", "regulation", "adoption", "impact", "risks", "opportunities",
         "case study", "report", "review", "outlook", "forecast", "debate"]

DOMAINS = ["nature.com", "sciencedirect.com", "reuters.com", "bloomberg.com", "arxiv.org",
           "mit.edu", "stanford.edu", "who.int", "un.org", "techcrunch.com", "wired.com",
           "economist.com", "nytimes.com", "bbc.com", "forbes.com", "ieee.org",
           "sciencemag.org", "pnas.org", "wsj.com", "theguardian.com"]


def make_snippet(topic: str, adj: str, noun: str) -> str:
    templates = [
        f"A {adj} look at {noun} in {topic}, examining key drivers, stakeholders, and open questions.",
        f"New {noun} on {topic} highlights {adj} shifts across the industry and academic community.",
        f"This article covers {adj} {noun} related to {topic}, with data from multiple independent sources.",
        f"Experts discuss the {adj} {noun} surrounding {topic} and what it means going forward.",
        f"An in-depth {noun} of {topic}, focused on {adj} implications for practitioners and policymakers.",
    ]
    return random.choice(templates)


def generate(n: int = 10000) -> list[dict]:
    records = []
    for i in range(n):
        topic = random.choice(TOPICS)
        adj = random.choice(ADJECTIVES)
        noun = random.choice(NOUNS)
        domain = random.choice(DOMAINS)
        slug = "-".join(topic.split()) + f"-{noun.replace(' ', '-')}-{i}"
        record = {
            "source_id": f"src_{i:06d}",
            "url": f"https://www.{domain}/articles/{slug}",
            "title": f"{adj.capitalize()} {noun} in {topic}".strip(),
            "snippet": make_snippet(topic, adj, noun),
            "topic_tags": [topic],
            "domain": domain,
            "published_days_ago": random.randint(0, 900),
        }
        records.append(record)
    return records


def main():
    out_path = Path(__file__).parent / "corpus.json"
    data = generate(10000)
    out_path.write_text(json.dumps(data))
    print(f"Wrote {len(data)} mock records to {out_path}")


if __name__ == "__main__":
    main()
