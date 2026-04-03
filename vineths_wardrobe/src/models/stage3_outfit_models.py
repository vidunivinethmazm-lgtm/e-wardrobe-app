"""
eWardrobeAI — Stage 3: Dual Outfit Recommendation Models

Model 1 — HeuristicRecommender
  Rule-based scoring: occasion match + body type + completeness + colour harmony.
  No training. Deterministic, interpretable.

Model 2 — ContentBasedRecommender
  TF-IDF vectorisation over garment tags + style + category.
  Cosine similarity between user preference vector and garment vectors.
  Trained in seconds from the wardrobe catalogue.

Accuracy Metrics
  Since ground-truth user preferences are unavailable, we use:
  - Coverage:   % of clean catalogue items appearing in top-K results
  - Diversity:  average pairwise dissimilarity within recommended bundles
  - Avg Score:  mean relevance score of top-K recommendations
  - Precision@K: % of recommended items matching target occasion tags
  - Response Time: ms per recommendation call
"""

from __future__ import annotations

import time
import numpy as np
from collections import defaultdict
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise         import cosine_similarity

from src.outfit_recommender import (
    RaveehaOrganisationalDB, NisfaMatchmaking,
    GarmentRecord, OutfitRecommendation, Style,
    GarmentCategory, CleaningStatus,
    _colour_harmony_score,
)


# ── Model 1: Heuristic Recommender (wrapper around existing NisfaMatchmaking) ──

class HeuristicRecommender:
    """
    Wraps the existing NisfaMatchmaking engine.
    Occasion match + body type + bundle completeness + colour theory.
    """
    name = "Heuristic Recommender"

    def __init__(self, db: RaveehaOrganisationalDB):
        self._engine = NisfaMatchmaking(db)
        self._db     = db

    def recommend(self, size: str, body_type: str,
                  styles: list[Style], occasion: str, top_k: int = 5):
        return self._engine.recommend(size, body_type, styles, occasion, top_k)


# ── Model 2: Content-Based TF-IDF Recommender ─────────────────────────────────

class ContentBasedRecommender:
    """
    TF-IDF content-based recommendation using garment metadata.

    Each garment is represented as a text document:
      "<tags> <style> <category> <body_types> <colours>"

    User preference is represented as a query document:
      "<preferred_styles> <occasion> <body_type>"

    Garments are ranked by cosine similarity to the user query.
    Only Clean + owned garments are considered (same as HeuristicRecommender).
    """
    name = "Content-Based TF-IDF Recommender"

    def __init__(self, db: RaveehaOrganisationalDB):
        self._db        = db
        self._vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._garment_ids     : list[str]        = []
        self._garment_matrix  : Optional[np.ndarray] = None
        self._trained = False
        self._fit()

    def _fit(self):
        """Build TF-IDF matrix from all garment documents."""
        garments = list(self._db._records.values())
        docs = []
        for g in garments:
            doc = ' '.join([
                ' '.join(g.tags),
                g.style.value.replace('_', ' '),
                g.category.value,
                ' '.join(g.body_types),
                ' '.join(g.colours),
                g.name.lower(),
            ])
            docs.append(doc)
        self._garment_ids    = [g.garment_id for g in garments]
        self._garment_matrix = self._vectoriser.fit_transform(docs)
        self._trained        = True

    def _user_vector(self, styles: list[Style], occasion: str, body_type: str) -> np.ndarray:
        query = ' '.join([
            ' '.join(s.value.replace('_', ' ') for s in styles),
            occasion.replace('_', ' '),
            body_type.replace('_', ' '),
        ])
        return self._vectoriser.transform([query])

    def recommend(self, size: str, body_type: str,
                  styles: list[Style], occasion: str, top_k: int = 5) -> list[OutfitRecommendation]:
        if not self._trained:
            self._fit()

        user_vec = self._user_vector(styles, occasion, body_type)
        sims     = cosine_similarity(user_vec, self._garment_matrix).flatten()

        # Rank garments by similarity, filter to wearable only
        ranked_ids = [self._garment_ids[i]
                      for i in np.argsort(sims)[::-1]
                      if self._db._records[self._garment_ids[i]].is_wearable]

        # Build bundles from top-ranked items
        import uuid
        bundles: list[OutfitRecommendation] = []
        tops     = [self._db.get(g) for g in ranked_ids
                    if self._db.get(g) and self._db.get(g).category == GarmentCategory.TOP][:4]
        bottoms  = [self._db.get(g) for g in ranked_ids
                    if self._db.get(g) and self._db.get(g).category == GarmentCategory.BOTTOM][:4]
        dresses  = [self._db.get(g) for g in ranked_ids
                    if self._db.get(g) and self._db.get(g).category == GarmentCategory.DRESS][:2]
        suits    = [self._db.get(g) for g in ranked_ids
                    if self._db.get(g) and self._db.get(g).category == GarmentCategory.SUIT][:2]

        for t in tops:
            for b in bottoms:
                if not t or not b: continue
                score = float(sims[self._garment_ids.index(t.garment_id)] * 0.5 +
                              sims[self._garment_ids.index(b.garment_id)] * 0.5)
                colour_bonus = _colour_harmony_score(t.colours + b.colours) * 0.15
                score = min(1.0, score + colour_bonus)
                bundles.append(OutfitRecommendation(
                    outfit_id=f"TFIDF-{uuid.uuid4().hex[:8].upper()}",
                    name=f"TF-IDF Pick — {t.name} + {b.name}",
                    style=styles[0] if styles else Style.CASUAL,
                    items=[t, b], score=round(score, 4), occasion=occasion,
                ))

        for d in dresses:
            if not d: continue
            score = float(sims[self._garment_ids.index(d.garment_id)])
            bundles.append(OutfitRecommendation(
                outfit_id=f"TFIDF-{uuid.uuid4().hex[:8].upper()}",
                name=f"TF-IDF Pick — {d.name}",
                style=styles[0] if styles else Style.CASUAL,
                items=[d], score=round(score, 4), occasion=occasion,
            ))

        for s in suits:
            if not s: continue
            score = float(sims[self._garment_ids.index(s.garment_id)])
            bundles.append(OutfitRecommendation(
                outfit_id=f"TFIDF-{uuid.uuid4().hex[:8].upper()}",
                name=f"TF-IDF Pick — {s.name}",
                style=styles[0] if styles else Style.FORMAL,
                items=[s], score=round(score, 4), occasion=occasion,
            ))

        bundles.sort(key=lambda b: b.score, reverse=True)
        return bundles[:top_k]


# ── Accuracy Checker ──────────────────────────────────────────────────────────

class Stage3AccuracyChecker:
    """
    Evaluates both recommenders across a set of test scenarios.

    Metrics:
      precision_at_k  : % of recommended items with matching occasion tags
      coverage        : % of clean catalogue covered across all test queries
      avg_score       : mean relevance score (model's own scoring)
      diversity       : avg number of unique categories per recommendation set
      response_time_ms: average ms per recommendation call
    """

    TEST_SCENARIOS = [
        {'size': 'M',  'body': 'rectangle',        'styles': [Style.SMART, Style.CASUAL], 'occasion': 'office'},
        {'size': 'S',  'body': 'hourglass',         'styles': [Style.CASUAL],             'occasion': 'casual'},
        {'size': 'L',  'body': 'pear',              'styles': [Style.FORMAL],             'occasion': 'interview'},
        {'size': 'M',  'body': 'inverted_triangle', 'styles': [Style.SMART],              'occasion': 'smart_casual'},
        {'size': 'XL', 'body': 'rectangle',         'styles': [Style.CASUAL],             'occasion': 'weekend'},
        {'size': 'M',  'body': 'hourglass',         'styles': [Style.EVENING, Style.FORMAL], 'occasion': 'formal'},
    ]

    def __init__(self):
        self._db        = RaveehaOrganisationalDB()
        self.heuristic  = HeuristicRecommender(self._db)
        self.contentbased = ContentBasedRecommender(self._db)

    def run(self, top_k: int = 5) -> dict:
        print("[Stage3Accuracy] Evaluating recommendation models…")
        h_results  = self._eval_model(self.heuristic,    top_k)
        cb_results = self._eval_model(self.contentbased, top_k)

        better_prec = ("Heuristic" if h_results['precision_at_k'] >= cb_results['precision_at_k']
                       else "ContentBased TF-IDF")

        report = {
            'stage':      3,
            'task':       'Outfit Recommendation',
            'testScenarios': len(self.TEST_SCENARIOS),
            'topK':       top_k,
            'catalogueSize': len(self._db._records),
            'wearableCount': sum(1 for g in self._db._records.values() if g.is_wearable),
            'models':     [h_results, cb_results],
            'bestPrecision': better_prec,
            'summary': {
                h_results['model']:  h_results,
                cb_results['model']: cb_results,
            }
        }
        print(f"\n── Stage 3 Accuracy Summary ──")
        for m in [h_results, cb_results]:
            print(f"  {m['model']:<35} "
                  f"Prec@{top_k}={m['precision_at_k']:.3f}  "
                  f"Coverage={m['coverage']:.3f}  "
                  f"AvgScore={m['avg_score']:.3f}  "
                  f"Speed={m['response_time_ms']:.1f}ms")
        return report

    def _eval_model(self, model, top_k: int) -> dict:
        precisions, coverages, scores, diversities, times = [], [], [], [], []
        all_recommended_ids: set = set()
        total_clean = {g.garment_id for g in self._db._records.values() if g.is_wearable}

        for sc in self.TEST_SCENARIOS:
            t0   = time.perf_counter()
            recs = model.recommend(sc['size'], sc['body'], sc['styles'], sc['occasion'], top_k)
            elapsed = (time.perf_counter() - t0) * 1000

            if not recs: continue

            # Precision@K: % of items whose tags match the occasion
            all_items = [item for rec in recs for item in rec.items]
            occ = sc['occasion'].lower()
            prec = sum(1 for it in all_items
                       if any(occ in t.lower() for t in it.tags)) / max(len(all_items), 1)
            precisions.append(prec)

            # Coverage
            rec_ids = {item.garment_id for rec in recs for item in rec.items}
            all_recommended_ids |= rec_ids
            coverages.append(len(rec_ids) / max(len(total_clean), 1))

            # Avg score
            scores.append(np.mean([r.score for r in recs]))

            # Diversity: unique categories per recommendation set
            cats = {item.category for rec in recs for item in rec.items}
            diversities.append(len(cats))
            times.append(elapsed)

        return {
            'model':            getattr(model, 'name', type(model).__name__),
            'precision_at_k':   round(float(np.mean(precisions)), 4)  if precisions  else 0,
            'coverage':         round(len(all_recommended_ids) / max(len(total_clean), 1), 4),
            'avg_score':        round(float(np.mean(scores)),     4)  if scores       else 0,
            'avg_diversity':    round(float(np.mean(diversities)),4)  if diversities  else 0,
            'response_time_ms': round(float(np.mean(times)),      3)  if times        else 0,
            'scenarios_tested': len(self.TEST_SCENARIOS),
        }

    def compare_single(self, size: str, body_type: str,
                       styles: list[Style], occasion: str, top_k: int = 3) -> dict:
        """Run both models on one query and return side-by-side results."""
        h  = self.heuristic.recommend(size, body_type, styles, occasion, top_k)
        cb = self.contentbased.recommend(size, body_type, styles, occasion, top_k)
        return {
            'Heuristic': [{'name': r.name, 'score': r.score,
                           'items': [i.name for i in r.items]} for r in h],
            'ContentBased TF-IDF': [{'name': r.name, 'score': r.score,
                                     'items': [i.name for i in r.items]} for r in cb],
        }


if __name__ == '__main__':
    checker = Stage3AccuracyChecker()
    report  = checker.run(top_k=5)
