from datetime import datetime, timezone
from statistics import mean

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None


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
        "keyword": "monochrome black outfit",
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

AUDIENCE_KEYWORDS = {
    "women": ["women fashion trends", "women outfit ideas", "women street style"],
    "men": ["men fashion trends", "men outfit ideas", "men street style"],
    "general": ["fashion trends", "outfit ideas", "street style"],
}


def _normalize(value):
    return str(value or "").strip().lower()


def _status(score):
    if score >= 60:
        return "trending"
    if score <= 35:
        return "outdated"
    return "stable"


def _season_for_date(date_text):
    try:
        month = datetime.fromisoformat(str(date_text)).month
    except Exception:
        month = datetime.now().month

    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    if month in [9, 10, 11]:
        return "fall"
    return "winter"


def _matches_type(trend_type, clothing_type):
    trend_type = _normalize(trend_type)
    clothing_type = _normalize(clothing_type)
    return trend_type and (
        trend_type == clothing_type
        or trend_type in clothing_type
        or clothing_type in trend_type
    )


def _build_keywords(clothing_type, color="", season="", gender=""):
    clothing_type = _normalize(clothing_type) or "fashion"
    color = _normalize(color)
    season = _normalize(season)
    gender = _normalize(gender)

    keywords = []
    if color:
        keywords.append(f"{color} {clothing_type} outfit")
    if season:
        keywords.append(f"{season} {clothing_type} fashion")
    if gender and gender not in ["unknown", "unisex"]:
        keywords.append(f"{gender} {clothing_type} style")
    keywords.append(f"{clothing_type} fashion trend")

    return keywords[:5]


def _pytrends_scores(keywords, timeframe="today 3-m", geo=""):
    if TrendReq is None:
        return None

    try:
        pytrends = TrendReq(hl="en-US", tz=330, timeout=(5, 10))
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop="")
        interest = pytrends.interest_over_time()

        if interest.empty:
            return None

        scores = {}
        for keyword in keywords:
            if keyword in interest:
                values = interest[keyword].tail(8).tolist()
                scores[keyword] = round(mean(values), 1) if values else 0

        return scores or None
    except Exception:
        return None


def _local_trend_matches(clothing_type, color, season, limit=3):
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
            final_score = min(score, 100)
            matches.append({
                "keyword": trend["keyword"],
                "score": final_score,
                "status": _status(final_score),
                "matched_on": match_reasons,
                "suggestion": trend["suggestion"],
            })

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:limit]


def get_fashion_trends(audience="general", limit=5):
    audience = _normalize(audience) or "general"
    keywords = AUDIENCE_KEYWORDS.get(audience, AUDIENCE_KEYWORDS["general"])
    scores = _pytrends_scores(keywords)

    if scores:
        matches = [
            {
                "keyword": keyword,
                "score": score,
                "status": _status(score),
                "matched_on": ["Google Trends interest"],
                "suggestion": f"Use '{keyword}' as a styling direction for current outfit planning.",
            }
            for keyword, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]
        source = "Pytrends live Google Trends"
    else:
        matches = [
            {
                "keyword": trend["keyword"],
                "score": trend["score"],
                "status": _status(trend["score"]),
                "matched_on": ["fallback catalog"],
                "suggestion": trend["suggestion"],
            }
            for trend in TREND_CATALOG[:limit]
        ]
        source = "Local fallback catalog"

    return {
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches": matches[:limit],
    }


def get_trend_matches(clothing_type, color="", season="", gender="", limit=3):
    keywords = _build_keywords(clothing_type, color, season, gender)
    scores = _pytrends_scores(keywords)

    if scores:
        matches = []
        for keyword, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            matches.append({
                "keyword": keyword,
                "score": score,
                "status": _status(score),
                "matched_on": ["Google Trends interest"],
                "suggestion": _suggestion_for_score(keyword, score),
            })
        source = "Pytrends live Google Trends"
    else:
        matches = _local_trend_matches(clothing_type, color, season, limit)
        source = "Local fallback catalog"

    return {
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matches": matches[:limit],
    }


def analyze_wardrobe_items(items):
    analyzed_items = []

    for item in items:
        trend_analysis = get_trend_matches(
            item.get("type"),
            item.get("color"),
            item.get("season"),
            item.get("gender"),
            limit=1,
        )
        best_match = trend_analysis["matches"][0] if trend_analysis["matches"] else {
            "keyword": "fashion trend",
            "score": 0,
            "status": "outdated",
            "suggestion": "Refresh this item with more current colors, layers, or accessories.",
        }

        analyzed_items.append({
            "id": item.get("id"),
            "type": item.get("type"),
            "color": item.get("color"),
            "gender": item.get("gender"),
            "season": item.get("season"),
            "processedImageUrl": item.get("processedImageUrl"),
            "score": best_match["score"],
            "status": best_match["status"],
            "keyword": best_match["keyword"],
            "suggestion": best_match["suggestion"],
            "source": trend_analysis["source"],
        })

    status_counts = {"trending": 0, "stable": 0, "outdated": 0}
    for item in analyzed_items:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

    total = len(analyzed_items)
    average_score = round(mean([item["score"] for item in analyzed_items]), 1) if total else 0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_items": total,
            "average_score": average_score,
            "status_counts": status_counts,
        },
        "chart_data": [
            {"label": "Trending", "value": status_counts["trending"], "color": "#22c55e"},
            {"label": "Stable", "value": status_counts["stable"], "color": "#38bdf8"},
            {"label": "Outdated", "value": status_counts["outdated"], "color": "#f97316"},
        ],
        "items": sorted(analyzed_items, key=lambda item: item["score"], reverse=True),
        "suggestions": _wardrobe_suggestions(analyzed_items),
    }


def recommend_for_event(items, event):
    event_date = event.get("eventDate") or event.get("date")
    event_name = _normalize(event.get("eventName") or event.get("name"))
    event_season = _season_for_date(event_date)

    analyzed = analyze_wardrobe_items(items)["items"]
    ranked = []

    for item in analyzed:
        score = item["score"]
        reasons = [f"{item['status']} item"]

        if _normalize(item.get("season")) == event_season:
            score += 12
            reasons.append(f"good for {event_season}")

        if any(word in event_name for word in ["meeting", "office", "formal", "interview"]):
            if _normalize(item.get("type")) in ["shirt", "trousers", "dress", "blazer"]:
                score += 10
                reasons.append("fits formal event")

        if any(word in event_name for word in ["party", "dinner", "wedding"]):
            if _normalize(item.get("type")) in ["dress", "shirt", "jeans"]:
                score += 8
                reasons.append("fits social event")

        ranked.append({**item, "event_score": min(score, 100), "reasons": reasons})

    ranked.sort(key=lambda item: item["event_score"], reverse=True)
    best = ranked[0] if ranked else None

    return {
        "event_date": event_date,
        "event_season": event_season,
        "best_item": best,
        "alternatives": ranked[1:4],
        "suggestion": _event_suggestion(best, event_season),
    }


def _suggestion_for_score(keyword, score):
    if score >= 60:
        return f"'{keyword}' is performing strongly, so this item can be used in a trend-focused outfit."
    if score <= 35:
        return f"'{keyword}' has low recent interest. Modernize it with updated layers, colors, or accessories."
    return f"'{keyword}' has steady interest. Use it as a dependable base and add one current accent."


def _wardrobe_suggestions(items):
    if not items:
        return ["Save wardrobe items first, then run the trend analysis."]

    outdated = [item for item in items if item["status"] == "outdated"]
    trending = [item for item in items if item["status"] == "trending"]

    suggestions = []
    if trending:
        suggestions.append(f"Use your {trending[0]['color']} {trending[0]['type']} for trend-forward outfits.")
    if outdated:
        suggestions.append(f"Refresh your {outdated[0]['color']} {outdated[0]['type']} with newer styling pieces.")
    suggestions.append("For scheduled events, prefer high-score items that match the event season.")

    return suggestions


def _event_suggestion(best, event_season):
    if not best:
        return "Save wardrobe items before requesting an event suggestion."

    return (
        f"Best choice for this date is the {best.get('color')} {best.get('type')} "
        f"because it scores {best.get('event_score')} and matches {', '.join(best.get('reasons', []))}. "
        f"Style it for {event_season} with suitable layers and accessories."
    )
