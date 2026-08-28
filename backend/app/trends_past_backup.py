from datetime import datetime, timezone


TREND_CATALOG = [
    {
        "keyword": "oversized streetwear",
        "type": "shirt",
        "colors": ["black", "white", "grey"],
        "seasons": ["summer", "fall"],
        "score": 94,
        "suggestion": "Style with relaxed bottoms and minimal sneakers for a current streetwear look.",
    },
    {
        "keyword": "linen neutrals",
        "type": "shirt",
        "colors": ["white", "beige", "cream", "brown"],
        "seasons": ["summer"],
        "score": 91,
        "suggestion": "Pair with light trousers or a skirt to create a breathable warm-weather outfit.",
    },
    {
        "keyword": "denim on denim",
        "type": "jeans",
        "colors": ["blue", "navy"],
        "seasons": ["spring", "fall"],
        "score": 89,
        "suggestion": "Match with a denim jacket or crisp white top for an easy trend-led outfit.",
    },
    {
        "keyword": "monochrome black",
        "type": "dress",
        "colors": ["black"],
        "seasons": ["winter", "fall"],
        "score": 88,
        "suggestion": "Use metallic accessories or a bright bag to make the outfit feel intentional.",
    },
    {
        "keyword": "soft pastel styling",
        "type": "dress",
        "colors": ["pink", "lavender", "light blue", "yellow"],
        "seasons": ["spring", "summer"],
        "score": 86,
        "suggestion": "Combine with white or nude accessories for a soft seasonal look.",
    },
    {
        "keyword": "athleisure basics",
        "type": "t-shirt",
        "colors": ["black", "white", "grey", "green"],
        "seasons": ["summer", "spring"],
        "score": 84,
        "suggestion": "Layer with joggers, leggings, or a lightweight jacket for a casual outfit.",
    },
    {
        "keyword": "tailored minimalism",
        "type": "trousers",
        "colors": ["black", "grey", "navy", "beige"],
        "seasons": ["winter", "fall", "spring"],
        "score": 82,
        "suggestion": "Add a fitted top or blazer to create a clean smart-casual outfit.",
    },
]


def _normalize(value):
    return str(value or "").strip().lower()


def _matches_type(trend_type, clothing_type):
    trend_type = _normalize(trend_type)
    clothing_type = _normalize(clothing_type)
    return trend_type and (
        trend_type == clothing_type
        or trend_type in clothing_type
        or clothing_type in trend_type
    )


def get_trend_matches(clothing_type, color, season, limit=3):
    color = _normalize(color)
    season = _normalize(season)

    matches = []

    for trend in TREND_CATALOG:
        match_reasons = []
        score = int(trend["score"])

        if _matches_type(trend["type"], clothing_type):
            score += 8
            match_reasons.append("clothing type")

        if color in [_normalize(item) for item in trend["colors"]]:
            score += 6
            match_reasons.append("color")

        if season in [_normalize(item) for item in trend["seasons"]]:
            score += 4
            match_reasons.append("season")

        if match_reasons:
            matches.append({
                "keyword": trend["keyword"],
                "score": min(score, 100),
                "matched_on": match_reasons,
                "suggestion": trend["suggestion"],
            })

    matches.sort(key=lambda item: item["score"], reverse=True)

    return {
        "source": "Local Google Trends keyword catalog",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches": matches[:limit],
    }
