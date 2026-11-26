"""MeChi Studio cekirdek birim ve entegrasyon testleri."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mechi.config import ASSETS_MASCOT_DIR, PROFILES_DIR, get_env_key, save_env_values, save_to_downloads
from mechi.core.api_bridge import ImageApiBridge
from mechi.core.enhancer import enhance_single_file
from mechi.core.mascot_engine import MascotEngine
from mechi.core.module_registry import BaseToolModule, ModuleRegistry, ModuleSpec
from mechi.core.prompt_engine import PromptEngine
from mechi.core.ps_bridge import PhotoshopBridge
from mechi.core.repair_analyzer import analyze_and_build_repair_prompt
from mechi.modules import register_builtin_modules


class _DummyPluginModule(BaseToolModule):
    module_id = "dummy_test"
    title = "Test Eklenti"


class TestMeChiCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_01_prompt_engine_10k_unique_and_balanced(self) -> None:
        self.assertFalse(os.path.exists(os.path.join(PROFILES_DIR, "riviera_atelier")))
        brand_ids = ["casablanca", "offwhite", "burberry"]

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "test_prompts.db")
            eng = PromptEngine(profiles_dir=PROFILES_DIR, db_path=db_path)
            self.addCleanup(eng.close)
            expected_counts = {
                "casablanca": {"frame": 20, "ornament": 20, "subject": 24, "typography": 24, "palette": 12, "comb": 2_764_800},
                "offwhite": {"frame": 50, "ornament": 50, "subject": 50, "typography": 100, "palette": 12, "comb": 150_000_000},
                "burberry": {"frame": 50, "ornament": 50, "subject": 50, "typography": 100, "palette": 12, "comb": 150_000_000},
            }

            for pid in brand_ids:
                meta, cats = eng.load_profile(pid)
                exp = expected_counts[pid]
                self.assertEqual(len(cats["frame"]), exp["frame"], f"{pid} frame != {exp['frame']}")
                self.assertEqual(len(cats["ornament"]), exp["ornament"], f"{pid} ornament != {exp['ornament']}")
                self.assertEqual(len(cats["subject"]), exp["subject"], f"{pid} subject != {exp['subject']}")
                self.assertEqual(len(cats["typography"]), exp["typography"], f"{pid} typography != {exp['typography']}")
                self.assertEqual(len(cats["palette"]), exp["palette"], f"{pid} palette != {exp['palette']}")
                self.assertEqual(eng.get_total_combinations(pid), exp["comb"])
                if pid == "casablanca":
                    forbidden = ("casablanca", "casa way", "hermes", "hermès")
                    for bad in forbidden:
                        self.assertNotIn(bad, meta.get("style_dna", "").lower())
                    for cname, citems in cats.items():
                        for itm in citems:
                            p_txt = (itm.get("prompt") or "").lower()
                            for bad in forbidden:
                                self.assertNotIn(
                                    bad,
                                    p_txt,
                                    f"Telif sizintisi {cname}/{itm.get('id')}: {bad}",
                                )
                            if cname in ("frame", "subject", "ornament", "typography"):
                                self.assertGreaterEqual(
                                    len(p_txt.split()),
                                    30,
                                    f"Casablanca {cname}/{itm.get('id')} prompt cok kisa/sig!",
                                )

            sigs = set()
            seen_frames = {pid: set() for pid in brand_ids}
            seen_typos = {pid: set() for pid in brand_ids}
            t0 = time.perf_counter()
            for idx in range(10_000):
                pid = brand_ids[idx % len(brand_ids)]
                res = eng.roll_unique(
                    pid,
                    custom_text="ATELIER STUDIO",
                    seed=idx + 12345,
                )
                sig_key = f"{pid}:{res.signature}"
                self.assertNotIn(sig_key, sigs)
                sigs.add(sig_key)
                seen_frames[pid].add(res.picked["frame"]["id"])
                seen_typos[pid].add(res.picked["typography"]["id"])
                if idx < 3:
                    self.assertIn("ATELIER STUDIO", res.full_prompt)
                    self.assertIn("[MULTI-LAYER DEPTH & COMPOSITION ARCHITECTURE]", res.full_prompt)
                    if pid == "casablanca":
                        self.assertGreaterEqual(len(res.full_prompt.split()), 220)
            elapsed = time.perf_counter() - t0

            self.assertEqual(len(sigs), 10_000)
            for pid in brand_ids:
                exp = expected_counts[pid]
                self.assertEqual(len(seen_frames[pid]), exp["frame"], f"{pid} frame kapsama eksik")
                self.assertEqual(len(seen_typos[pid]), exp["typography"], f"{pid} typography kapsama eksik")
            eng.close()
            self.assertLess(elapsed, 20.0, f"10k zar cok yavas: {elapsed:.2f}s")
            print(f"[test_prompt_engine] 3 marka icin 10.000 benzersiz zar {elapsed:.2f} sn icinde uretildi (0 tekrar).")

    def test_02_enhancer_single_file_rgba_and_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 1) Gurultulu RGBA PNG: ortada duz kare, disi seffaf
            rng = np.random.default_rng(42)
            rgba = np.zeros((256, 256, 4), dtype=np.uint8)
            base_patch = np.full((128, 128, 3), [180, 90, 60], dtype=np.int16)
            noise = rng.integers(-22, 23, size=(128, 128, 3), dtype=np.int16)
            rgba[64:192, 64:192, :3] = np.clip(base_patch + noise, 0, 255).astype(np.uint8)
            rgba[64:192, 64:192, 3] = 255

            in_png = os.path.join(tmp, "noisy.png")
            out_png = os.path.join(tmp, "clean.png")
            Image.fromarray(rgba, mode="RGBA").save(in_png)
            with open(in_png, "rb") as fp:
                orig_hash = hashlib.sha256(fp.read()).hexdigest()

            res = enhance_single_file(
                in_png,
                out_png,
                options={"denoise": 55, "sharpen": 40, "deblock": True, "upscale": "2x"},
            )
            self.assertTrue(os.path.exists(out_png))
            with open(in_png, "rb") as fp:
                self.assertEqual(hashlib.sha256(fp.read()).hexdigest(), orig_hash)
            self.assertEqual(res["width"], 512)
            self.assertEqual(res["height"], 512)

            out_arr = np.array(Image.open(out_png).convert("RGBA"))
            # Dis kenar pikselleri tamamen seffaf kalmali
            self.assertEqual(int(out_arr[0:40, 0:40, 3].max()), 0)
            # Ic bolgedeki tek kanal gurultu standart sapmasi belirgin sekilde dusmeli
            in_std = float(rgba[80:176, 80:176, 0].std())
            out_std = float(out_arr[180:330, 180:330, 0].std())
            self.assertLess(out_std, in_std * 0.7)

            # 2) JPEG testi + 4K & 254 DPI + Ince Tuy/Suluboya Detay Koruma dogrulamasi
            # Ince 1-2px siyah murekkep/tuy cizgileri ve beyaz uzerinde acik turkuaz suluboya lekesi ekle
            detail_canvas = rgba[..., :3].copy()
            detail_canvas[10:60, 10:240] = [252, 252, 252]
            detail_canvas[20:50, 30:110] = [175, 232, 236]  # Acik turkuaz suluboya dokusu
            for y_line in range(72, 180, 6):
                detail_canvas[y_line : y_line + 1, 72:184] = [24, 18, 15]  # 1px ince tuy tarama cizgileri

            in_jpg = os.path.join(tmp, "noisy.jpg")
            out_jpg = os.path.join(tmp, "clean_4k.jpg")
            Image.fromarray(detail_canvas, mode="RGB").save(in_jpg, quality=85, dpi=(72, 72))
            in_jpg_arr = np.array(Image.open(in_jpg).convert("RGB"))
            res_4k = enhance_single_file(
                in_jpg,
                out_jpg,
                options={
                    "denoise": 22,
                    "sharpen": 72,
                    "vector_clean": 35,
                    "edge_crisp": 78,
                    "despeckle": True,
                    "deblock": True,
                    "upscale": "4k",
                    "target_dpi": 254,
                },
            )
            self.assertTrue(os.path.exists(out_jpg))
            self.assertEqual(res_4k["width"], 3840)
            self.assertEqual(res_4k["height"], 3840)
            self.assertEqual(res_4k["dpi"], 254)
            self.assertAlmostEqual(res_4k["width_cm"], 38.4, places=1)
            with Image.open(out_jpg) as im_4k:
                saved_dpi = im_4k.info.get("dpi")
                self.assertIsNotNone(saved_dpi)
                self.assertEqual(int(round(saved_dpi[0])), 254)
                self.assertEqual(int(round(saved_dpi[1])), 254)
                arr_4k = np.array(im_4k)
                down_1x = cv2.resize(arr_4k, (256, 256), interpolation=cv2.INTER_AREA)
                # Ince 1px tuy cizgileri pastellesip silinmemeli; girdi JPEG'deki cizgi-zemin kontrasti korunup guclenmeli
                line_rows = [y_line for y_line in range(72, 180, 6)]
                bg_rows = [y_line + 3 for y_line in range(72, 174, 6)]
                in_contrast = float(in_jpg_arr[bg_rows, 80:176, 0].mean() - in_jpg_arr[line_rows, 80:176, 0].mean())
                out_contrast = float(down_1x[bg_rows, 80:176, 0].mean() - down_1x[line_rows, 80:176, 0].mean())
                self.assertGreaterEqual(out_contrast, in_contrast * 0.92)
                # Acik turkuaz suluboya lekesi beyaz arka plana erimemeli (R ve B farki korunmali)
                watercolor_patch = down_1x[25:45, 40:100].mean(axis=(0, 1))
                self.assertGreater(float(watercolor_patch[2] - watercolor_patch[0]), 40.0)
            print(
                f"[test_enhancer] RGBA in_std={in_std:.2f} -> out_std={out_std:.2f}, "
                f"4K(3840px/254DPI) 1px_tuy_kontrast={in_contrast:.1f}->{out_contrast:.1f}, orijinal korundu."
            )

    def test_03_api_bridge_and_env_offline_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = os.path.join(tmp, ".env")
            bridge = ImageApiBridge(env_path=env_file)

            ok_none, _ = bridge.is_configured({"api_provider": "none"})
            self.assertFalse(ok_none)

            ok_missing, msg_missing = bridge.is_configured({"api_provider": "cloudflare"})
            self.assertFalse(ok_missing)
            self.assertIn("Eksik anahtar", msg_missing)

            save_env_values({"CF_ACCOUNT_ID": "acc123", "CF_API_TOKEN": "tok_xyz"}, env_path=env_file)
            self.assertEqual(get_env_key("CF_ACCOUNT_ID", env_path=env_file), "acc123")
            ok_cf, _ = bridge.is_configured({"api_provider": "cloudflare"})
            self.assertTrue(ok_cf)

            # Bos deger verince silinmeli
            save_env_values({"CF_API_TOKEN": ""}, env_path=env_file)
            ok_after, _ = bridge.is_configured({"api_provider": "cloudflare"})
            self.assertFalse(ok_after)

    def test_04_module_registry_extensibility(self) -> None:
        reg = ModuleRegistry()
        register_builtin_modules(reg)
        ids = [s.module_id for s in reg.all_specs()]
        self.assertEqual(ids, ["tasarim", "duzelt", "netlestir", "emprime"])

        reg.register(
            ModuleSpec(
                module_id="dummy_test",
                title="Test Eklenti",
                short_desc="Yeni modul testi",
                badge_label="Test",
                order=40,
                icon_char="*",
                widget_cls=_DummyPluginModule,
            )
        )
        ids2 = [s.module_id for s in reg.all_specs()]
        self.assertEqual(ids2, ["tasarim", "duzelt", "netlestir", "emprime", "dummy_test"])

    def test_05_mascot_engine_auto_sequence(self) -> None:
        eng = MascotEngine(mascot_dir=ASSETS_MASCOT_DIR)
        self.assertTrue(eng.has_clips())
        c1 = eng.pick_clip("opening")
        c2 = eng.pick_clip("home")
        c3 = eng.pick_clip("home")
        c_busy = eng.pick_clip("busy")
        c_ok = eng.pick_clip("success")
        assert c1 and c2 and c3 and c_busy and c_ok
        self.assertEqual(c1.stem, "01_tekir_acilis")
        self.assertEqual(c2.stem, "02_bombay_cam")
        self.assertEqual(c3.stem, "01_tekir_acilis")
        self.assertEqual(c_busy.stem, "04_kuyruk_takip")
        self.assertEqual(c_ok.stem, "03_ikili_selam")

    def test_06_repair_analyzer_local_a1_a2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 1) Beyaz cerceveli izole grafik -> A2 + B1 beklenir
            iso = np.full((320, 320, 3), 252, dtype=np.uint8)
            iso[80:240, 80:240] = [24, 68, 138]
            iso[110:210, 110:210] = [212, 162, 52]
            iso_path = os.path.join(tmp, "isolated.png")
            Image.fromarray(iso, mode="RGB").save(iso_path)

            res_iso = analyze_and_build_repair_prompt(iso_path, api_key="")
            self.assertEqual(res_iso.analysis_mode, "local_python")
            self.assertIn("A2", res_iso.detected_modules)
            self.assertIn("B1", res_iso.detected_modules)
            self.assertIn("[CRITICAL PROHIBITION PROTOCOL - HIGHEST PRIORITY]", res_iso.full_prompt)
            self.assertGreaterEqual(len(res_iso.palette_hexes), 2)

            # 2) Tum tuvali dolduran desen -> A1 beklenir
            yy, xx = np.mgrid[0:320, 0:320]
            checker = ((xx // 32) + (yy // 32)) % 2
            full_arr = np.zeros((320, 320, 3), dtype=np.uint8)
            full_arr[checker == 0] = [165, 42, 48]
            full_arr[checker == 1] = [30, 58, 95]
            full_path = os.path.join(tmp, "fullbleed.png")
            Image.fromarray(full_arr, mode="RGB").save(full_path)

            res_full = analyze_and_build_repair_prompt(full_path, api_key="")
            self.assertEqual(res_full.analysis_mode, "local_python")
            self.assertIn("A1", res_full.detected_modules)
            self.assertGreaterEqual(len(res_full.palette_hexes), 2)
            print(f"[test_repair] A2={res_iso.detected_modules} | A1={res_full.detected_modules}")

    def test_07_repair_recreation_protocols_and_api_sys_inst(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            iso = np.full((320, 320, 3), 250, dtype=np.uint8)
            iso[60:260, 60:260] = [20, 64, 132]
            iso_path = os.path.join(tmp, "worn.png")
            Image.fromarray(iso, mode="RGB").save(iso_path)

            res = analyze_and_build_repair_prompt(
                iso_path,
                overrides={
                    "source_condition": "worn_garment",
                    "render_style": "flat_vector",
                    "custom_text": "CASA WAY",
                    "text_mode": "auto",
                    "strict_single_output": True,
                    "strict_zero_hallucination": True,
                },
                api_key="",
            )
            self.assertIn("C4", res.detected_modules)
            self.assertIn("[SYSTEM DIRECTIVE: ABSOLUTE RECREATION ON A PERFECT SURFACE]", res.full_prompt)
            self.assertIn("Smart UV Unwrap", res.full_prompt)
            self.assertIn("ONE standalone flat 2D graphic image", res.full_prompt)
            self.assertIn("Absolute Recreation on a Perfect Surface", res.system_instruction)

            res_tex = analyze_and_build_repair_prompt(
                iso_path,
                overrides={
                    "source_condition": "coated_textured",
                    "license_safe": True,
                    "custom_text": "LOGO TEXT",
                    "render_style": "flat_vector",
                    "text_mode": "C2",
                },
                api_key="",
            )
            self.assertIn("coated canvas", res_tex.full_prompt.lower())
            self.assertIn("leather", res_tex.full_prompt.lower())
            self.assertIn("make minimal subtle adjustments", res_tex.full_prompt)
            self.assertIn("keeping the same visual layout, pose, drawing style, and color energy", res_tex.full_prompt)
            self.assertIn("recreate this adjusted graphic from scratch", res_tex.full_prompt)
            self.assertIn("orthographically flat 2D standalone print artwork", res_tex.full_prompt)
            self.assertIn("0°", res_tex.full_prompt)
            self.assertIn("90°", res_tex.full_prompt)
            self.assertIn("LOGO TEXT", res_tex.full_prompt)
            self.assertIn("TELİFSİZ", res_tex.detected_modules)
            self.assertIn("2D apparel graphic print specialist", res_tex.system_instruction)

            res_ls_notxt = analyze_and_build_repair_prompt(
                iso_path,
                overrides={"source_condition": "worn_garment", "license_safe": True},
                api_key="",
            )
            self.assertIn("adjusting any brand wording into clean generic lettering", res_ls_notxt.full_prompt)

            res_ls_variant = analyze_and_build_repair_prompt(
                iso_path,
                overrides={
                    "source_condition": "worn_garment",
                    "license_safe": True,
                    "render_style": "painterly_2d",
                    "extra_note": "make outline thicker",
                },
                api_key="",
            )
            self.assertNotEqual(res_ls_notxt.full_prompt, res_ls_variant.full_prompt)
            self.assertIn("make outline thicker", res_ls_variant.full_prompt)
            self.assertIn("2D editorial illustration shading", res_ls_variant.full_prompt)

            forbidden_triggers = (
                "loose inspiration",
                "STREET STUDIO",
                "invent an original",
                "ABSOLUTE RECREATION",
                "CRITICAL PROHIBITION",
                "STRICTLY FORBIDDEN",
                "human body parts",
                "intellectual property",
                "copyrighted",
                "trademarked",
                "stripping",
                "Strip all",
                "[STEP 1",
            )
            for sample_txt in (
                res_tex.full_prompt,
                res_tex.system_instruction,
                res_ls_notxt.full_prompt,
                res_ls_notxt.system_instruction,
                res_ls_variant.full_prompt,
            ):
                for bad in forbidden_triggers:
                    self.assertNotIn(bad.lower(), sample_txt.lower(), f"Tetikleyici ifade bulundu: {bad}")

            # Gemini API payload dogrulamasi (mock POST)
            env_file = os.path.join(tmp, ".env")
            save_env_values({"GEMINI_API_KEY": "dummy_gemini_test_key_0001"}, env_path=env_file)
            bridge = ImageApiBridge(env_path=env_file)

            buf = io.BytesIO()
            Image.new("RGB", (4, 4), (255, 255, 255)).save(buf, format="PNG")
            dummy_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": dummy_b64}}]}}
                ]
            }

            with patch("mechi.core.api_bridge.requests.post", return_value=mock_resp) as m_post:
                ok, out_path = bridge.generate_image(
                    prompt=res.full_prompt,
                    out_dir=tmp,
                    aspect="1:1",
                    settings={"api_provider": "gemini_image", "api_model": "gemini-3.1-flash-image"},
                    reference_image_path=iso_path,
                    system_instruction=res.system_instruction,
                )
                self.assertTrue(ok, out_path)
                self.assertTrue(os.path.exists(out_path))
                sent_json = m_post.call_args.kwargs["json"]
                self.assertIn("systemInstruction", sent_json)
                self.assertIn("Absolute Recreation on a Perfect Surface", sent_json["systemInstruction"]["parts"][0]["text"])
                self.assertEqual(sent_json["generationConfig"]["imageConfig"]["aspectRatio"], "1:1")
                parts = sent_json["contents"][0]["parts"]
                self.assertIn("inlineData", parts[0])
                print("[test_recreation_protocol] C4 + systemInstruction + Smart UV Unwrap + mock API payload OK.")

    def test_08_casablanca_capsule_filters_and_render_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "caps_test.db")
            eng = PromptEngine(profiles_dir=PROFILES_DIR, db_path=db_path)

            themes = eng.get_profile_themes("casablanca")
            modes = eng.get_profile_render_modes("casablanca")
            self.assertEqual(len(themes), 8)
            self.assertEqual([m["id"] for m in modes], ["auto", "scarf_spot", "airbrush_watercolor"])

            for idx, th in enumerate(themes):
                tid = th["id"]
                if tid == "all":
                    self.assertEqual(eng.get_total_combinations("casablanca", theme_id="all"), 2_764_800)
                    continue
                comb = eng.get_total_combinations("casablanca", theme_id=tid)
                self.assertGreaterEqual(comb, 50_000, f"{tid} kombinasyon yetersiz: {comb}")

                res = eng.roll_unique("casablanca", theme_id=tid, custom_text="RIVIERA STUDIO", seed=900 + idx)
                self.assertEqual(res.theme_id, tid)
                ttags = {str(t).lower() for t in th["tags"]} | {"universal"}
                for cat, item in res.picked.items():
                    itags = {str(t).lower() for t in (item.get("tags") or [])}
                    self.assertTrue(bool(itags & ttags), f"{tid} -> {cat}/{item['id']} etiket disi: {itags}")

            r_spot = eng.roll_unique("casablanca", theme_id="zellige_arch", render_mode="scarf_spot", seed=51)
            r_air = eng.roll_unique("casablanca", theme_id="memphis_airbrush", render_mode="airbrush_watercolor", seed=52)
            self.assertIn("Flat Spot Silk-Scarf Execution", r_spot.full_prompt)
            self.assertIn("1980s Retro Airbrush", r_air.full_prompt)
            eng.close()
            print("[test_casablanca_capsules] 7 kapsul + 3 render modu tam uyumlu.")

    def test_09_ps_bridge_process_and_jsx_fallback(self) -> None:
        br = PhotoshopBridge("auto")

        # 1) Photoshop kapaliysa connected=False
        with patch.object(br, "_find_running_photoshop", return_value=None):
            st0 = br.get_status()
            self.assertFalse(st0["connected"])
            self.assertIsNone(st0["active_doc"])

        # 2) PS 2021 acikken COM PS 2026'ya kayitliysa COM yerine dogrudan JSX runner kullanilmali
        fake_proc = {
            "pid": 4242,
            "exe_path": r"C:\Program Files\Adobe\Adobe Photoshop 2021\Photoshop.exe",
            "year": "2021",
            "progid": "Photoshop.Application.150",
        }
        fake_json = '{"version":"22.5.9","doc_count":1,"active_doc":{"name":"Untitled-1","path":"","width_px":1000,"height_px":1000,"layer_count":2,"has_layers":true}}'
        with (
            patch.object(br, "_find_running_photoshop", return_value=fake_proc),
            patch.object(br, "_get_registered_com_exe", return_value=r"c:\program files\adobe\adobe photoshop 2026\photoshop.exe"),
            patch.object(br, "_run_jsx_via_com") as m_com,
            patch.object(br, "_run_jsx_via_exe", return_value=(True, fake_json)) as m_jsx,
            patch.object(br, "_read_fresh_launch_ctx", return_value=(None, False)),
        ):
            st1 = br.get_status(force_jsx_probe=False)
            self.assertTrue(st1["connected"])
            self.assertEqual(st1["year"], "2021")
            self.assertEqual(st1["transport"], "jsx")
            self.assertEqual(st1["active_doc"]["name"], "Untitled-1")
            m_com.assert_not_called()
            self.assertEqual(m_jsx.call_count, 1)

            # Ayni PID icin arka plan yoklamasi tekrar tekrar PS.exe baslatmamali
            st2 = br.get_status(force_jsx_probe=False)
            self.assertTrue(st2["connected"])
            self.assertEqual(st2["active_doc"]["name"], "Untitled-1")
            self.assertEqual(m_jsx.call_count, 1)

            # Kullanici pil'e tikladiginda (force=True) yeniden okumali
            st3 = br.get_status(force_jsx_probe=True)
            self.assertTrue(st3["connected"])
            self.assertEqual(m_jsx.call_count, 2)
            print("[test_ps_bridge] PS 2021/2026 cakisma korumasi + JSX onbellek + force yenileme OK.")

    def test_10_downloads_export_and_direct_ps_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = os.path.join(tmp, "out")
            dl_dir = os.path.join(tmp, "dl")
            os.makedirs(src_dir)
            src = os.path.join(src_dir, "sonuc.png")
            Image.new("RGB", (8, 8), (10, 20, 30)).save(src)

            ok1, dst1 = save_to_downloads(src, downloads_dir=dl_dir, reveal_in_explorer=False)
            self.assertTrue(ok1)
            self.assertEqual(os.path.basename(dst1), "sonuc.png")
            self.assertTrue(os.path.exists(dst1))

            ok2, dst2 = save_to_downloads(src, downloads_dir=dl_dir, reveal_in_explorer=False)
            self.assertTrue(ok2)
            self.assertEqual(os.path.basename(dst2), "sonuc_2.png")

            ok3, dst3 = save_to_downloads(dst1, downloads_dir=dl_dir, reveal_in_explorer=False)
            self.assertTrue(ok3)
            self.assertEqual(os.path.normcase(dst3), os.path.normcase(os.path.abspath(dst1)))
            self.assertEqual(len(os.listdir(dl_dir)), 2)

            ok4, msg4 = save_to_downloads(os.path.join(tmp, "yok.png"), downloads_dir=dl_dir, reveal_in_explorer=False)
            self.assertFalse(ok4)

            br = PhotoshopBridge("auto")
            fake_exe = r"C:\Program Files\Adobe\Adobe Photoshop 2021\Photoshop.exe"
            with (
                patch.object(br, "_find_installed_photoshop_exe", return_value=fake_exe),
                patch("mechi.core.ps_bridge.subprocess.Popen") as m_popen,
                patch.object(br, "_exec_jsx") as m_jsx,
            ):
                ok5, res5 = br.open_file(src)
                self.assertTrue(ok5)
                m_popen.assert_called_once_with([fake_exe, os.path.abspath(src)])
                m_jsx.assert_not_called()
            print("[test_downloads] Indirilenlere kopya + cakisma + dogrudan PS acma OK.")

    def test_11_repair_geometry_detector_and_prerectify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Beyaz zemin uzerine:
            # 1) Ust kenara yapisik/kesik baslik harf bloklari (top clipping + text band)
            # 2) +6 derece egik dikdortgen cerceve (has_rect_frame + tilt_deg)
            # 3) Altta egik metin bandi (has_text_bands + text_band_tilt_deg)
            canvas = np.full((320, 320, 3), 255, dtype=np.uint8)
            for idx_g, x0 in enumerate(range(60, 250, 26)):
                canvas[0:14, x0 : x0 + 16] = [18, 18, 18]  # Ust sinirdan kesik harf bloklari
                y_bot = 276 + int(round((idx_g - 3) * 1.8))
                canvas[y_bot : y_bot + 16, x0 : x0 + 16] = [18, 18, 18]  # Altta egik metin satiri

            # +6 derece egik dikdortgen cerceve ciz
            rect_layer = np.full((320, 320, 3), 255, dtype=np.uint8)
            cv2.rectangle(rect_layer, (68, 56), (252, 240), (20, 20, 20), thickness=4)
            rot_m = cv2.getRotationMatrix2D((160.0, 148.0), -6.0, 1.0)
            tilted_rect = cv2.warpAffine(
                rect_layer,
                rot_m,
                (320, 320),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            mask_dark = tilted_rect < 240
            canvas[mask_dark] = tilted_rect[mask_dark]

            in_path = os.path.join(tmp, "tilted_clipped.png")
            Image.fromarray(canvas, mode="RGB").save(in_path)

            res = analyze_and_build_repair_prompt(
                in_path,
                overrides={"pre_rectify": True},
                rectify_out_dir=tmp,
                api_key="",
            )
            m = res.metrics
            self.assertIn("top", m.clipped_sides)
            self.assertTrue(m.has_rect_frame or abs(m.tilt_deg) >= 2.0)
            self.assertTrue(m.has_text_bands)
            self.assertEqual(m.recommended_aspect, "3:4")
            # custom_text bos olsa bile metin bandi + egim nedeniyle otomatik C4 secilmeli!
            self.assertIn("C4", res.detected_modules)
            self.assertIn("[STRICT 0°/90° ARCHITECTURAL GEOMETRY & ALIGNMENT LOCK]", res.full_prompt)
            self.assertIn("CRITICAL EDGE-CLIPPING ALERT", res.full_prompt)
            self.assertIn("TOP", res.full_prompt)

            self.assertTrue(bool(res.rectified_ref_path and os.path.exists(res.rectified_ref_path)))
            post_m = analyze_and_build_repair_prompt(res.rectified_ref_path, api_key="").metrics
            self.assertLessEqual(abs(post_m.tilt_deg), 1.5, f"Deskew sonrasi hala egik: {post_m.tilt_deg}")
            with Image.open(res.rectified_ref_path) as r_im:
                rw, rh = r_im.size
                self.assertGreater(rw, 320)
                self.assertGreater(rh, 320)
                r_arr = np.array(r_im.convert("RGB"))
                # On-duzeltilmis referansin ust 12 piksel seridi artik tamamen temiz beyaz (%16 margin) olmali
                self.assertGreaterEqual(float(r_arr[:12, :, :].min()), 250.0)
            print(
                f"[test_geometry_detector] tilt={m.tilt_deg:+.1f}°->post_tilt={post_m.tilt_deg:+.1f}°, "
                f"clipped={m.clipped_sides}, text_bands={m.has_text_bands}, mods={res.detected_modules}, "
                f"pre_rectified={rw}x{rh} OK."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
