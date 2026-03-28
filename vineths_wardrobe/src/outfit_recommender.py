"""
eWardrobeAI — Stage 3: Outfit Recommendation & Wardrobe Database
Two-Component System:

  ┌─────────────────────────────────────────────────────────────┐
  │  NisfaMatchmaking Engine                                    │
  │  Rule-based + cosine-similarity outfit recommendation.      │
  │  Matches user SizingProfile and style preferences to        │
  │  curated outfit combinations from the wardrobe catalogue.   │
  └─────────────────────────────────────────────────────────────┘
       │ queries available outfit IDs
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  RaveehaOrganisationalDB                                    │
  │  In-memory wardrobe database (production: replaces with     │
  │  SQLite / PostgreSQL).                                      │
  │  Tracks each garment's:                                     │
  │    - availability (owned, borrowed, wishlist)               │
  │    - cleaning_status: Clean | Dirty | In Laundry            │
  │  Only 'Clean' + 'owned' items are eligible for try-on.      │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import uuid
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────────────────

class CleaningStatus(str, Enum):
    CLEAN      = 'Clean'
    DIRTY      = 'Dirty'
    IN_LAUNDRY = 'In Laundry'


class Availability(str, Enum):
    OWNED    = 'owned'
    BORROWED = 'borrowed'
    WISHLIST = 'wishlist'


class GarmentCategory(str, Enum):
    TOP        = 'top'
    BOTTOM     = 'bottom'
    DRESS      = 'dress'
    OUTERWEAR  = 'outerwear'
    SUIT       = 'suit'
    FOOTWEAR   = 'footwear'
    ACCESSORY  = 'accessory'


class Style(str, Enum):
    CASUAL    = 'casual'
    FORMAL    = 'formal'
    SMART     = 'smart_casual'
    SPORTY    = 'sporty'
    EVENING   = 'evening'


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class GarmentRecord:
    """
    Single clothing item in the RaveehaOrganisationalDB.

    asset_path: relative path to the .glb or .fbx 3D model file
                used by AvatarManager to load the cloth mesh.
    """
    garment_id:      str
    name:            str
    category:        GarmentCategory
    style:           Style
    sizes:           list[str]           # e.g. ['S', 'M', 'L']
    colours:         list[str]
    body_types:      list[str]           # compatible body shapes
    asset_path:      str                 # relative path to .glb / .fbx
    thumbnail_path:  str
    cleaning_status: CleaningStatus = CleaningStatus.CLEAN
    availability:    Availability   = Availability.OWNED
    tags:            list[str]      = field(default_factory=list)

    @property
    def is_wearable(self) -> bool:
        """Item can only be rendered if it is Clean and owned."""
        return (
            self.cleaning_status == CleaningStatus.CLEAN
            and self.availability == Availability.OWNED
        )


@dataclass
class OutfitRecommendation:
    """
    A complete outfit bundle: one or more garments selected by
    NisfaMatchmaking and cleared by RaveehaOrganisationalDB.
    """
    outfit_id:   str
    name:        str
    style:       Style
    items:       list[GarmentRecord]
    score:       float                   # recommendation relevance [0, 1]
    occasion:    str
    asset_paths: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.asset_paths = [item.asset_path for item in self.items]


# ── RaveehaOrganisationalDB ───────────────────────────────────────────────────

class RaveehaOrganisationalDB:
    """
    Simulated wardrobe organisational database.

    In production this class wraps a SQLite / PostgreSQL session.
    For the demo it initialises a hardcoded catalogue of garments
    that maps directly to prebuilt .glb asset files.

    Cleaning-status lifecycle
    -------------------------
    Clean → (worn) → Dirty → (sent to laundry) → In Laundry → (returned) → Clean
    """

    def __init__(self):
        self._records: dict[str, GarmentRecord] = {}
        self._seed_catalogue()

    def _seed_catalogue(self):
        """Populate in-memory catalogue with demo garments."""
        catalogue: list[GarmentRecord] = [
            # ── Tops ─────────────────────────────────────────────────────────
            GarmentRecord(
                garment_id='GAR-001', name='White Oxford Shirt',
                category=GarmentCategory.TOP, style=Style.SMART,
                sizes=['S', 'M', 'L', 'XL'], colours=['white', 'light_blue'],
                body_types=['rectangle', 'inverted_triangle', 'hourglass'],
                asset_path='assets/outfits/white_oxford_shirt.glb',
                thumbnail_path='assets/thumbnails/white_oxford_shirt.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['formal', 'office', 'interview'],
            ),
            GarmentRecord(
                garment_id='GAR-002', name='Navy Polo Shirt',
                category=GarmentCategory.TOP, style=Style.SMART,
                sizes=['XS', 'S', 'M', 'L', 'XL'],
                colours=['navy', 'burgundy', 'forest_green'],
                body_types=['all'],
                asset_path='assets/outfits/navy_polo.glb',
                thumbnail_path='assets/thumbnails/navy_polo.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['smart_casual', 'weekend'],
            ),
            GarmentRecord(
                garment_id='GAR-003', name='Striped T-Shirt',
                category=GarmentCategory.TOP, style=Style.CASUAL,
                sizes=['XS', 'S', 'M'], colours=['blue_white', 'black_white'],
                body_types=['pear', 'rectangle'],
                asset_path='assets/outfits/striped_tshirt.glb',
                thumbnail_path='assets/thumbnails/striped_tshirt.jpg',
                cleaning_status=CleaningStatus.DIRTY,           # NOT wearable
                tags=['casual', 'beach', 'summer'],
            ),
            GarmentRecord(
                garment_id='GAR-004', name='Graphic Hoodie',
                category=GarmentCategory.TOP, style=Style.CASUAL,
                sizes=['S', 'M', 'L', 'XL', 'XXL'],
                colours=['charcoal', 'burgundy'],
                body_types=['all'],
                asset_path='assets/outfits/graphic_hoodie.glb',
                thumbnail_path='assets/thumbnails/graphic_hoodie.jpg',
                cleaning_status=CleaningStatus.IN_LAUNDRY,      # NOT wearable
                tags=['casual', 'lounge'],
            ),

            # ── Bottoms ───────────────────────────────────────────────────────
            GarmentRecord(
                garment_id='GAR-005', name='Slim-Fit Chinos',
                category=GarmentCategory.BOTTOM, style=Style.SMART,
                sizes=['28', '30', '32', '34', '36'],
                colours=['khaki', 'navy', 'olive'],
                body_types=['rectangle', 'inverted_triangle'],
                asset_path='assets/outfits/slim_chinos.glb',
                thumbnail_path='assets/thumbnails/slim_chinos.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['smart_casual', 'office'],
            ),
            GarmentRecord(
                garment_id='GAR-006', name='Dark Wash Jeans',
                category=GarmentCategory.BOTTOM, style=Style.CASUAL,
                sizes=['28', '30', '32', '34'],
                colours=['dark_indigo', 'black'],
                body_types=['all'],
                asset_path='assets/outfits/dark_jeans.glb',
                thumbnail_path='assets/thumbnails/dark_jeans.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['casual', 'versatile'],
            ),
            GarmentRecord(
                garment_id='GAR-007', name='Tailored Dress Trousers',
                category=GarmentCategory.BOTTOM, style=Style.FORMAL,
                sizes=['30', '32', '34', '36'],
                colours=['charcoal', 'black', 'navy'],
                body_types=['all'],
                asset_path='assets/outfits/dress_trousers.glb',
                thumbnail_path='assets/thumbnails/dress_trousers.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['formal', 'interview', 'dinner'],
            ),

            # ── Outerwear ─────────────────────────────────────────────────────
            GarmentRecord(
                garment_id='GAR-008', name='Wool Blazer',
                category=GarmentCategory.OUTERWEAR, style=Style.SMART,
                sizes=['S', 'M', 'L', 'XL'],
                colours=['charcoal', 'navy', 'mid_grey'],
                body_types=['rectangle', 'hourglass', 'inverted_triangle'],
                asset_path='assets/outfits/wool_blazer.glb',
                thumbnail_path='assets/thumbnails/wool_blazer.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['smart', 'layering', 'office'],
            ),
            GarmentRecord(
                garment_id='GAR-009', name='Puffer Jacket',
                category=GarmentCategory.OUTERWEAR, style=Style.CASUAL,
                sizes=['XS', 'S', 'M', 'L', 'XL', 'XXL'],
                colours=['black', 'olive', 'cobalt_blue'],
                body_types=['all'],
                asset_path='assets/outfits/puffer_jacket.glb',
                thumbnail_path='assets/thumbnails/puffer_jacket.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['winter', 'outdoor'],
            ),

            # ── Dresses ───────────────────────────────────────────────────────
            GarmentRecord(
                garment_id='GAR-010', name='Wrap Midi Dress',
                category=GarmentCategory.DRESS, style=Style.SMART,
                sizes=['XS', 'S', 'M', 'L', 'XL'],
                colours=['emerald', 'terracotta', 'navy'],
                body_types=['hourglass', 'pear'],
                asset_path='assets/outfits/wrap_midi_dress.glb',
                thumbnail_path='assets/thumbnails/wrap_midi_dress.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['wedding_guest', 'date_night', 'smart'],
            ),
            GarmentRecord(
                garment_id='GAR-011', name='Evening Gown',
                category=GarmentCategory.DRESS, style=Style.EVENING,
                sizes=['XS', 'S', 'M', 'L'],
                colours=['black', 'deep_red', 'champagne'],
                body_types=['hourglass', 'rectangle'],
                asset_path='assets/outfits/evening_gown.glb',
                thumbnail_path='assets/thumbnails/evening_gown.jpg',
                cleaning_status=CleaningStatus.DIRTY,           # NOT wearable
                tags=['gala', 'formal', 'evening'],
            ),

            # ── Suits ─────────────────────────────────────────────────────────
            GarmentRecord(
                garment_id='GAR-012', name='Classic Two-Piece Suit',
                category=GarmentCategory.SUIT, style=Style.FORMAL,
                sizes=['36R', '38R', '40R', '42R', '44R'],
                colours=['charcoal', 'navy', 'mid_grey'],
                body_types=['rectangle', 'inverted_triangle', 'hourglass'],
                asset_path='assets/outfits/classic_suit.glb',
                thumbnail_path='assets/thumbnails/classic_suit.jpg',
                cleaning_status=CleaningStatus.CLEAN,
                tags=['formal', 'interview', 'wedding'],
            ),
        ]
        for g in catalogue:
            self._records[g.garment_id] = g

        logger.info(f"[RaveehaDB] Seeded {len(self._records)} garment records.")

    # ── CRUD-like API ─────────────────────────────────────────────────────────

    def get(self, garment_id: str) -> Optional[GarmentRecord]:
        return self._records.get(garment_id)

    def query_available(
        self,
        size: Optional[str] = None,
        style: Optional[Style] = None,
        category: Optional[GarmentCategory] = None,
        body_type: Optional[str] = None,
    ) -> list[GarmentRecord]:
        """
        Return Clean + owned garments matching the given filters.
        Items marked Dirty or In Laundry are excluded.
        """
        results = [g for g in self._records.values() if g.is_wearable]

        if size:
            results = [g for g in results if size in g.sizes or 'all' in g.sizes]
        if style:
            results = [g for g in results if g.style == style]
        if category:
            results = [g for g in results if g.category == category]
        if body_type:
            results = [g for g in results if
                       body_type in g.body_types or 'all' in g.body_types]

        return results

    def update_cleaning_status(self, garment_id: str,
                                status: CleaningStatus) -> bool:
        if garment_id in self._records:
            self._records[garment_id].cleaning_status = status
            logger.info(f"[RaveehaDB] {garment_id} → {status.value}")
            return True
        return False

    def wardrobe_summary(self) -> dict:
        """Returns counts by cleaning status for dashboard display."""
        counts = {s.value: 0 for s in CleaningStatus}
        for g in self._records.values():
            counts[g.cleaning_status.value] += 1
        return counts


# ── NisfaMatchmaking Engine ───────────────────────────────────────────────────

class NisfaMatchmaking:
    """
    Outfit recommendation engine.

    Algorithm
    ---------
    1. Query RaveehaDB for Clean + owned items matching user size & body type
    2. Apply style preference filter (occasion-driven)
    3. Build outfit bundles (top + bottom, or full dress, or suit)
    4. Score each bundle using a heuristic relevance function
    5. Return top-K ranked OutfitRecommendation objects

    All recommended garments are verified as wearable (Clean + owned)
    before being included in the output.
    """

    def __init__(self, db: RaveehaOrganisationalDB):
        self._db = db

    def recommend(
        self,
        size: str,
        body_type: str,
        preferred_styles: list[Style],
        occasion: str = 'casual',
        top_k: int = 5,
    ) -> list[OutfitRecommendation]:
        """
        Main recommendation entry point called by the VirtualTryOnPipeline.

        Parameters
        ----------
        size            : Standard size label from BodyCalibrator (e.g. 'M')
        body_type       : Body shape from BodyCalibrator (e.g. 'hourglass')
        preferred_styles: Ordered list of Style preferences
        occasion        : Free-text occasion tag ('casual', 'formal', etc.)
        top_k           : Maximum number of outfit bundles to return

        Returns
        -------
        List of OutfitRecommendation sorted by score (descending)
        """
        recommendations: list[OutfitRecommendation] = []

        for style in preferred_styles:
            bundles = self._build_outfit_bundles(size, body_type, style)
            for bundle in bundles:
                score = self._score_bundle(bundle, occasion, body_type)
                outfit_id = f"OUTFIT-{uuid.uuid4().hex[:8].upper()}"
                rec = OutfitRecommendation(
                    outfit_id=outfit_id,
                    name=self._name_outfit(bundle, style),
                    style=style,
                    items=bundle,
                    score=score,
                    occasion=occasion,
                )
                recommendations.append(rec)

        # Sort by score descending, return top-K
        recommendations.sort(key=lambda r: r.score, reverse=True)
        top = recommendations[:top_k]

        logger.info(
            f"[NisfaMatchmaking] {len(top)} outfit(s) recommended "
            f"for size={size}, body_type={body_type}, styles={preferred_styles}."
        )
        return top

    # ── Bundle Construction ────────────────────────────────────────────────────

    def _build_outfit_bundles(
        self,
        size: str,
        body_type: str,
        style: Style,
    ) -> list[list[GarmentRecord]]:
        """
        Construct valid outfit bundles (list of GarmentRecords) by querying
        the RaveehaDB for available garments and pairing them logically.

        Bundle types attempted:
          (a) Top + Bottom + optional Outerwear
          (b) Dress (standalone)
          (c) Suit (standalone)
        """
        bundles: list[list[GarmentRecord]] = []

        tops = self._db.query_available(
            size=size, style=style,
            category=GarmentCategory.TOP, body_type=body_type
        )
        bottoms = self._db.query_available(
            size=None, style=style,
            category=GarmentCategory.BOTTOM, body_type=body_type
        )
        outerwear = self._db.query_available(
            size=size, style=style,
            category=GarmentCategory.OUTERWEAR, body_type=body_type
        )

        # (a) Top + Bottom combinations
        for top in tops:
            for bottom in bottoms:
                bundle: list[GarmentRecord] = [top, bottom]
                if outerwear and random.random() > 0.5:
                    bundle.append(random.choice(outerwear))
                bundles.append(bundle)

        # (b) Dresses
        dresses = self._db.query_available(
            size=size, style=style,
            category=GarmentCategory.DRESS, body_type=body_type
        )
        for dress in dresses:
            bundles.append([dress])

        # (c) Suits
        suits = self._db.query_available(
            size=None, style=style,
            category=GarmentCategory.SUIT, body_type=body_type
        )
        for suit in suits:
            bundles.append([suit])

        return bundles

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_bundle(
        self,
        bundle: list[GarmentRecord],
        occasion: str,
        body_type: str,
    ) -> float:
        """
        Heuristic relevance score in [0, 1].

        Factors
        -------
        - Occasion tag match          (+0.35)
        - Body type match             (+0.25)
        - Bundle completeness         (+0.20)
        - Colour palette harmony      (+0.20 — colour theory rules)
        """
        score = 0.0

        # Occasion tag match
        all_tags = [t for item in bundle for t in item.tags]
        if any(occasion.lower() in tag.lower() for tag in all_tags):
            score += 0.35

        # Body type compatibility
        for item in bundle:
            if body_type in item.body_types or 'all' in item.body_types:
                score += 0.25 / len(bundle)

        # Bundle completeness
        categories = {item.category for item in bundle}
        if (GarmentCategory.TOP in categories and
                GarmentCategory.BOTTOM in categories):
            score += 0.20
        elif GarmentCategory.DRESS in categories:
            score += 0.20
        elif GarmentCategory.SUIT in categories:
            score += 0.20

        # Colour palette harmony (colour theory)
        all_colours = [c for item in bundle for c in item.colours]
        score += _colour_harmony_score(all_colours) * 0.20

        return round(min(score, 1.0), 4)

    @staticmethod
    def _name_outfit(bundle: list[GarmentRecord], style: Style) -> str:
        if len(bundle) == 1:
            return f"{style.value.title()} Look — {bundle[0].name}"
        names = ' + '.join(item.name for item in bundle[:2])
        return f"{style.value.title()} Ensemble — {names}"


# ── Colour Palette Harmony ────────────────────────────────────────────────────

# Colour families used for harmony classification
_NEUTRALS   = {'white', 'black', 'charcoal', 'mid_grey', 'light_grey',
               'off_white', 'beige', 'ivory', 'champagne'}
_EARTH_TONES = {'khaki', 'olive', 'terracotta', 'camel', 'tan',
                'brown', 'rust', 'burnt_orange'}
_COOL_TONES  = {'navy', 'cobalt_blue', 'light_blue', 'sky_blue',
                'teal', 'mid_grey', 'lavender', 'slate'}
_WARM_TONES  = {'burgundy', 'deep_red', 'forest_green', 'mustard',
                'coral', 'salmon', 'peach'}

# Complementary pairs that look great together (colour wheel opposites)
_COMPLEMENTARY_PAIRS = {
    frozenset({'navy',        'terracotta'}),
    frozenset({'navy',        'white'}),
    frozenset({'charcoal',    'white'}),
    frozenset({'charcoal',    'light_blue'}),
    frozenset({'black',       'white'}),
    frozenset({'black',       'champagne'}),
    frozenset({'olive',       'burgundy'}),
    frozenset({'forest_green','deep_red'}),
    frozenset({'cobalt_blue', 'rust'}),
    frozenset({'khaki',       'navy'}),
    frozenset({'mid_grey',    'cobalt_blue'}),
    frozenset({'emerald',     'champagne'}),
}

# Analogous groups (same colour family — always harmonious)
_ANALOGOUS_GROUPS = [_NEUTRALS, _EARTH_TONES, _COOL_TONES, _WARM_TONES]


def _colour_harmony_score(colours: list[str]) -> float:
    """
    Colour theory-based palette harmony score in [0, 1].

    Rules applied (in order, additive):
      1. Monochromatic / all-neutral palette       → 1.0
      2. Contains a complementary pair             → 0.85
      3. All colours from same analogous family    → 0.75
      4. Neutral + one accent colour               → 0.65
      5. Mixed families with no clash              → 0.50
      6. More than 3 distinct non-neutral families → 0.20 (clash penalty)

    Returns float in [0.20, 1.0].
    """
    if not colours:
        return 0.5

    unique = set(colours)

    # Rule 1: Monochromatic or all-neutral
    if unique.issubset(_NEUTRALS) or len(unique) == 1:
        return 1.0

    # Rule 2: Complementary pair present
    for pair in _COMPLEMENTARY_PAIRS:
        if pair.issubset(unique):
            return 0.85

    # Rule 3: All from same analogous family
    for group in _ANALOGOUS_GROUPS:
        if unique.issubset(group):
            return 0.75

    # Rule 4: Neutrals + exactly one accent
    non_neutral = unique - _NEUTRALS
    if len(non_neutral) == 1:
        return 0.65

    # Rule 5: Two non-neutral families — acceptable
    families_hit = sum(
        1 for g in _ANALOGOUS_GROUPS if unique & g
    )
    if families_hit <= 2:
        return 0.50

    # Rule 6: Clash (3+ colour families)
    return 0.20
