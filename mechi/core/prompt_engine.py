"""Zar mantikli, SQLite sıfır-tekrarlı ve stil-uyumlu prompt harmanlama motoru."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from mechi.config import PROFILES_DIR, PROMPT_DB_PATH

CATEGORY_FILES = {
    "frame": "cerceveler.json",
    "ornament": "suslemeler.json",
    "subject": "objeler.json",
    "typography": "yazilar.json",
    "palette": "paletler.json",
}


@dataclass
class PromptResult:
    roll_number: int
    profile_id: str
    profile_name: str
    signature: str
    summary_tr: str
    full_prompt: str
    picked: Dict[str, dict]
    theme_id: str = "all"
    theme_label_tr: str = ""
    render_mode: str = "auto"


class PromptEngine:
    """Kategorileri esit olasilik + mood uyumu ile harmanlar; SQLite ile tekrari %100 engeller."""

    def __init__(self, profiles_dir: str = PROFILES_DIR, db_path: str = PROMPT_DB_PATH) -> None:
        self.profiles_dir = profiles_dir
        self.db_path = db_path
        self._lock = threading.Lock()
        self._profile_cache: Dict[str, Tuple[dict, Dict[str, List[dict]]]] = {}
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_conn", None) is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _init_db(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                signature TEXT NOT NULL UNIQUE,
                summary_tr TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_picks (
                profile_id TEXT NOT NULL,
                category TEXT NOT NULL,
                item_id TEXT NOT NULL,
                used_at INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompts_prof ON generated_prompts(profile_id);"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_recent_prof_time ON recent_picks(profile_id, used_at DESC);"
        )
        self._conn.commit()

    def list_profiles(self) -> List[dict]:
        profiles: List[dict] = []
        if not os.path.isdir(self.profiles_dir):
            return profiles
        for slug in sorted(os.listdir(self.profiles_dir)):
            pdir = os.path.join(self.profiles_dir, slug)
            meta_path = os.path.join(pdir, "meta.json")
            if not os.path.isdir(pdir) or not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as fp:
                    meta = json.load(fp)
                meta.setdefault("id", slug)
                meta.setdefault("name", slug)
                profiles.append(meta)
            except Exception:
                continue
        return profiles

    def load_profile(self, profile_id: str, force_reload: bool = False) -> Tuple[dict, Dict[str, List[dict]]]:
        if not force_reload and profile_id in self._profile_cache:
            return self._profile_cache[profile_id]
        pdir = os.path.join(self.profiles_dir, profile_id)
        if not os.path.isdir(pdir):
            raise FileNotFoundError(f"Profil klasörü bulunamadı: {profile_id}")
        with open(os.path.join(pdir, "meta.json"), "r", encoding="utf-8") as fp:
            meta = json.load(fp)
        categories: Dict[str, List[dict]] = {}
        for cat, fname in CATEGORY_FILES.items():
            fpath = os.path.join(pdir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, list) and data:
                    for itm in data:
                        if isinstance(itm, dict):
                            itm["_tags_set"] = frozenset(str(t).lower() for t in (itm.get("tags") or []))
                    categories[cat] = data
        if not categories:
            raise ValueError(f"Profil boş veya kategori dosyaları eksik: {profile_id}")
        self._profile_cache[profile_id] = (meta, categories)
        return meta, categories

    def get_profile_themes(self, profile_id: str) -> List[dict]:
        meta, _ = self.load_profile(profile_id)
        return list(meta.get("themes") or [])

    def get_profile_render_modes(self, profile_id: str) -> List[dict]:
        meta, _ = self.load_profile(profile_id)
        return list(meta.get("render_modes") or [])

    @staticmethod
    def _resolve_theme_tags(meta: dict, theme_id: str) -> set:
        if not theme_id or theme_id == "all":
            return set()
        for th in meta.get("themes") or []:
            if th.get("id") == theme_id:
                return {str(t).lower() for t in (th.get("tags") or [])}
        return set()

    @staticmethod
    def _filter_category_for_theme(items: List[dict], theme_tags: set) -> List[dict]:
        if not theme_tags:
            return items
        allowed = theme_tags | {"universal"}
        matched = [
            x
            for x in items
            if ((x.get("_tags_set") or frozenset(str(t).lower() for t in (x.get("tags") or []))) & allowed)
        ]
        return matched if len(matched) >= 3 else items

    def get_total_combinations(self, profile_id: str, theme_id: str = "all") -> int:
        meta, cats = self.load_profile(profile_id)
        theme_tags = self._resolve_theme_tags(meta, theme_id)
        total = 1
        for items in cats.values():
            pool = self._filter_category_for_theme(items, theme_tags)
            total *= max(1, len(pool))
        return total

    def get_roll_count(self, profile_id: Optional[str] = None) -> int:
        with self._lock:
            if profile_id:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM generated_prompts WHERE profile_id = ?", (profile_id,)
                ).fetchone()
            else:
                row = self._conn.execute("SELECT COUNT(*) FROM generated_prompts").fetchone()
            return int(row[0]) if row else 0

    @staticmethod
    def _weighted_pick(
        items: List[dict],
        active_tags: set,
        recent_ids: set,
        rng: random.Random,
    ) -> dict:
        weights: List[float] = []
        for item in items:
            base_w = float(item.get("weight", 1.0))
            tags = item.get("_tags_set") or frozenset(str(t).lower() for t in (item.get("tags") or []))
            overlap = len(tags & active_tags) if active_tags else 0
            # Esit 1/N olasiligi korurken ayni mood etiketine sahip parcalara hafif bonus,
            # yakin zamanda secilenlere cooldown cezasi uygula
            mood_factor = 1.0 + 0.35 * overlap
            cooldown_factor = 0.25 if item.get("id") in recent_ids else 1.0
            weights.append(max(0.05, base_w * mood_factor * cooldown_factor))
        return rng.choices(items, weights=weights, k=1)[0]

    def _assemble_prompt(
        self,
        meta: dict,
        picked: Dict[str, dict],
        custom_text: str,
        bg_mode: str,
        render_mode: str = "auto",
        theme_label_en: str = "",
    ) -> str:
        brand_text = custom_text.strip() if custom_text.strip() else "[YOUR BRAND TEXT]"
        dna = meta.get(
            "style_dna",
            "Refined editorial luxury resort illustration, crisp ink outlines, harmonious flat graphic shading.",
        )

        frame_p = (picked.get("frame") or {}).get("prompt", "Ornate symmetrical geometric border frame.")
        orn_p = (picked.get("ornament") or {}).get("prompt", "Botanical olive and citrus foliage framing the edges.")
        subj_p = (picked.get("subject") or {}).get("prompt", "Classic vintage coastal sports centerpiece.")
        typo_raw = (picked.get("typography") or {}).get(
            "prompt", f'Clean bold editorial serif headline reading "{brand_text}".'
        )
        typo_p = typo_raw.replace("[YOUR BRAND TEXT]", brand_text).replace("{TEXT}", brand_text)
        if brand_text not in typo_p:
            typo_p = f'{typo_p} Display the exact text "{brand_text}".'

        pal = picked.get("palette") or {}
        pal_p = pal.get("prompt", "Balanced luxury resort spot-color palette.")
        hexes = pal.get("colors_hex") or []
        if hexes:
            pal_p = f"{pal_p} Use clean spot colors: {', '.join(hexes)}."

        bg_spec = (
            "Solid flat pure crisp white background (#FFFFFF) for light-garment printing."
            if bg_mode == "acik"
            else "Solid flat deep obsidian black background (#0B0B0B) for dark-garment printing."
        )

        render_directive = ""
        for rm in meta.get("render_modes") or []:
            if rm.get("id") == render_mode:
                render_directive = (rm.get("prompt") or rm.get("directive_en") or "").strip()
                break

        render_section = f"[RENDERING & PRINT TECHNIQUE]\n{render_directive}\n\n" if render_directive else ""
        quality_line = (
            "Present strictly as a complete, uncropped, centered 2D print-ready digital production artwork without distortion."
            if render_directive
            else "Present strictly as a complete, uncropped, centered 2D vector-clean digital illustration ready for screen printing. Crisp uniform edges, clean spot-color separations, no photographic blur."
        )
        theme_line = f"- Capsule Direction: {theme_label_en}\n" if theme_label_en else ""

        return (
            "[MAIN TASK: ORIGINAL 2D PRINT-READY T-SHIRT GRAPHIC ARTWORK]\n"
            "Create a high-resolution, razor-sharp, orthographically flat 2D standalone screen-print graphic illustration.\n\n"
            f"[STYLE DNA]\n{dna}\n\n"
            "[MULTI-LAYER DEPTH & COMPOSITION ARCHITECTURE]\n"
            "- NEVER generate a simplistic, sparse, or single-object flat icon.\n"
            "- Build a rich multi-plane collage combining: (1) the outer architectural portal frame or sculpted altar pedestal, "
            "(2) a foreground botanical/fauna/artifact layer with dynamic frame-breaking overlap projecting across the border edges, "
            "(3) a detailed mid-ground architectural/court/landscape sanctuary, and (4) an atmospheric or cosmic horizon backdrop.\n\n"
            "[COMPOSITIONAL ELEMENTS]\n"
            f"{theme_line}"
            f"- Frame & Architecture: {frame_p}\n"
            f"- Border Ornaments & Frame-Breaking Accents: {orn_p}\n"
            f"- Multi-Layer Central Scene: {subj_p}\n"
            f"- Typography & Placement: {typo_p}\n"
            f"- Color Palette: {pal_p}\n"
            f"- Background: {bg_spec}\n\n"
            f"{render_section}"
            "[PRINT & QUALITY PROTOCOL]\n"
            f"{quality_line}\n"
            "PROHIBITED: NO t-shirt mockup, NO garment, NO collar, NO fabric weave, NO wrinkles, NO folds, NO mannequin, "
            "NO studio shadows, NO third-party logos or watermarks."
        )

    def roll_unique(
        self,
        profile_id: str,
        locked: Optional[Dict[str, str]] = None,
        custom_text: str = "",
        bg_mode: str = "acik",
        seed: Optional[int] = None,
        theme_id: str = "all",
        render_mode: str = "auto",
    ) -> PromptResult:
        meta, categories = self.load_profile(profile_id)
        locked = locked or {}
        theme_id = theme_id or "all"
        render_mode = render_mode or "auto"
        theme_tags = self._resolve_theme_tags(meta, theme_id)
        theme_label_tr = ""
        theme_label_en = ""
        if theme_id != "all":
            for th in meta.get("themes") or []:
                if th.get("id") == theme_id:
                    theme_label_tr = th.get("label_tr", "")
                    theme_label_en = th.get("label_en", "")
                    break

        active_cats = {
            cat: self._filter_category_for_theme(items, theme_tags) for cat, items in categories.items()
        }
        rng = random.Random(seed) if seed is not None else random.SystemRandom()

        with self._lock:
            conn = self._conn
            recent_map: Dict[str, set] = {cat: set() for cat in CATEGORY_FILES}
            rows = conn.execute(
                "SELECT category, item_id FROM recent_picks WHERE profile_id = ? ORDER BY used_at DESC LIMIT 60",
                (profile_id,),
            ).fetchall()
            for cat, iid in rows:
                recent_map.setdefault(cat, set()).add(iid)

            chosen: Optional[Dict[str, dict]] = None
            chosen_sig = ""
            candidate: Dict[str, dict] = {}
            raw_key = ""

            for _attempt in range(250):
                active_tags: set = set(theme_tags)
                candidate = {}
                # Kilitli ogeleri once ekle ki mood etiketleri aktif olsun
                for cat, items in categories.items():
                    lock_id = locked.get(cat)
                    if lock_id:
                        matched = next((x for x in items if x.get("id") == lock_id), None)
                        if matched:
                            candidate[cat] = matched
                            active_tags.update(str(t).lower() for t in (matched.get("tags") or []))

                for cat in ("frame", "subject", "ornament", "typography", "palette"):
                    if cat in candidate or cat not in active_cats:
                        continue
                    pick = self._weighted_pick(
                        active_cats[cat],
                        active_tags=active_tags,
                        recent_ids=recent_map.get(cat, set()),
                        rng=rng,
                    )
                    candidate[cat] = pick
                    active_tags.update(str(t).lower() for t in (pick.get("tags") or []))

                key_parts = [profile_id, bg_mode]
                if theme_id != "all":
                    key_parts.append(f"t:{theme_id}")
                if render_mode != "auto":
                    key_parts.append(f"r:{render_mode}")
                key_parts.extend(
                    f"{c}:{candidate[c]['id']}"
                    for c in ("frame", "ornament", "subject", "typography", "palette")
                    if c in candidate
                )
                raw_key = "|".join(key_parts)
                sig_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
                sig = f"{raw_key}#{sig_hash}"

                exists = conn.execute(
                    "SELECT 1 FROM generated_prompts WHERE signature = ?", (sig,)
                ).fetchone()
                if not exists:
                    chosen = candidate
                    chosen_sig = sig
                    break

            if chosen is None:
                # Tum kombinasyonlar veya kilitli alan tukenmisse benzersiz varyasyon eki ekle
                assert candidate
                chosen = candidate
                nonce = rng.randint(100000, 999999)
                chosen_sig = f"{raw_key}#v{nonce}"

            summary_parts = [
                chosen[c].get("label_tr", chosen[c]["id"])
                for c in ("frame", "ornament", "subject", "typography")
                if c in chosen
            ]
            summary_tr = " / ".join(summary_parts)

            cur = conn.execute(
                "INSERT INTO generated_prompts (profile_id, signature, summary_tr) VALUES (?, ?, ?)",
                (profile_id, chosen_sig, summary_tr),
            )
            roll_no = int(cur.lastrowid or 1)

            conn.executemany(
                "INSERT INTO recent_picks (profile_id, category, item_id, used_at) VALUES (?, ?, ?, ?)",
                [(profile_id, cat, item["id"], roll_no) for cat, item in chosen.items()],
            )
            if roll_no % 50 == 0:
                conn.execute(
                    "DELETE FROM recent_picks WHERE profile_id = ? AND used_at < ?",
                    (profile_id, max(0, roll_no - 30)),
                )
            conn.commit()

        full_prompt = self._assemble_prompt(
            meta,
            chosen,
            custom_text=custom_text,
            bg_mode=bg_mode,
            render_mode=render_mode,
            theme_label_en=theme_label_en,
        )
        return PromptResult(
            roll_number=roll_no,
            profile_id=profile_id,
            profile_name=meta.get("name", profile_id),
            signature=chosen_sig,
            summary_tr=summary_tr,
            full_prompt=full_prompt,
            picked=chosen,
            theme_id=theme_id,
            theme_label_tr=theme_label_tr,
            render_mode=render_mode,
        )
