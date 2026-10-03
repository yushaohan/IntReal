"""Benchmark categories shared by evaluation and reporting."""

CATEGORIES = ("story", "knowledge", "planning", "tutorial")
SOURCE_CATEGORY = {
    "storystream": "story",
    "the_land_before_time": "story",
    "curious_george": "story",
    "rabbids": "story",
    "wikipedia": "knowledge",
    "wikitravel": "planning",
    "wikivoyage": "planning",
    "xiachufang": "tutorial",
}


def category(record):
    return SOURCE_CATEGORY.get(record.get("source"))


def subject_consistency_applicable(record):
    return category(record) in ("story", "tutorial")
