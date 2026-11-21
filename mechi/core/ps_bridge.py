"""Photoshop Windows canli koprusu (PS 2021 / 2026 uyumlu, COM + JSX hibrit IPC)."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
import winreg
from typing import Dict, List, Optional, Tuple

from mechi.config import LAUNCH_CTX_PATH


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


class PhotoshopBridge:
    """Acik Photoshop 2021 / 2026 ornegine COM veya dogrudan JSX IPC ile baglanir."""

    def __init__(self, preferred_version: str = "auto") -> None:
        self.preferred_version = preferred_version
        self._com_failed_pids: set = set()
        self._cached_pid: Optional[int] = None
        self._cached_status: Optional[dict] = None
        self._cached_ctx_mtime: float = 0.0
        self._jsx_lock = threading.Lock()

    @staticmethod
    def _query_process_exe(pid: int) -> str:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ""
        try:
            buf_len = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(1024)
            ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(buf_len))
            return buf.value if ok else ""
        finally:
            kernel32.CloseHandle(handle)

    def _find_running_photoshop(self) -> Optional[dict]:
        """Win32 Toolhelp ile acik Photoshop.exe sureclerini hizla bulur."""
        if os.name != "nt":
            return None
        kernel32 = ctypes.windll.kernel32
        snap = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snap == wintypes.HANDLE(-1).value:
            return None
        matches: List[dict] = []
        try:
            pe = _PROCESSENTRY32W()
            pe.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
            ok = kernel32.Process32FirstW(snap, ctypes.byref(pe))
            while ok:
                if (pe.szExeFile or "").lower() == "photoshop.exe":
                    pid = int(pe.th32ProcessID)
                    exe_path = self._query_process_exe(pid)
                    m_yr = re.search(r"(20\d{2})", exe_path or "")
                    year = m_yr.group(1) if m_yr else ""
                    if year == "2026":
                        progid = "Photoshop.Application.200"
                    elif year == "2021":
                        progid = "Photoshop.Application.150"
                    else:
                        progid = "Photoshop.Application"
                    matches.append(
                        {
                            "pid": pid,
                            "exe_path": exe_path,
                            "year": year,
                            "progid": progid,
                        }
                    )
                ok = kernel32.Process32NextW(snap, ctypes.byref(pe))
        except Exception:
            return None
        finally:
            kernel32.CloseHandle(snap)

        if not matches:
            return None
        pref = str(self.preferred_version or "").strip()
        if pref and pref != "auto":
            for m in matches:
                if m["year"] == pref:
                    return m
        return matches[0]

    @staticmethod
    def _get_registered_com_exe() -> str:
        """HKCR kaydindaki varsayilan Photoshop COM hedef .exe yolunu dondurur."""
        if os.name != "nt":
            return ""
        try:
            curver = "Photoshop.Application"
            try:
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Photoshop.Application\CurVer") as k:
                    v, _ = winreg.QueryValueEx(k, "")
                    if v:
                        curver = str(v)
            except OSError:
                pass
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, f"{curver}\\CLSID") as k:
                clsid, _ = winreg.QueryValueEx(k, "")
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, f"CLSID\\{clsid}\\LocalServer32") as k:
                cmd, _ = winreg.QueryValueEx(k, "")
            exe = re.sub(r"\s+/Automation.*$", "", str(cmd), flags=re.IGNORECASE).strip().strip('"')
            return os.path.normcase(os.path.abspath(exe)) if exe else ""
        except OSError:
            return ""

    @staticmethod
    def _escape_js_str(val: str) -> str:
        return json.dumps(val.replace("\\", "/"), ensure_ascii=False)

    def _run_jsx_via_com(self, progid: str, jsx_iife: str, timeout_sec: float = 4.0) -> Tuple[bool, str]:
        box: dict = {}

        def _worker() -> None:
            import pythoncom
            import win32com.client.dynamic

            pythoncom.CoInitialize()
            try:
                app = win32com.client.dynamic.Dispatch(progid)
                box["res"] = str(app.DoJavaScript(jsx_iife))
            except Exception as exc:
                box["err"] = str(exc)
            finally:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        th.join(timeout=timeout_sec)
        if th.is_alive():
            return False, "Photoshop COM yanıt vermedi (zaman aşımı)."
        if "err" in box:
            return False, f"COM hatası: {box['err']}"
        return True, box.get("res", "")

    def _run_jsx_via_exe(self, exe_path: str, jsx_iife: str, timeout_sec: float = 12.0) -> Tuple[bool, str]:
        """Acik Photoshop.exe ornegine tek seferlik JSX dosyasi iletip sonuc dosyasini okur."""
        if not exe_path or not os.path.exists(exe_path):
            return False, f"Photoshop exe bulunamadı: {exe_path}"

        ipc_dir = os.path.join(tempfile.gettempdir(), "mechi_ps_ipc")
        os.makedirs(ipc_dir, exist_ok=True)
        uid = uuid.uuid4().hex[:12]
        job_jsx = os.path.join(ipc_dir, f"job_{uid}.jsx")
        tmp_out = os.path.join(ipc_dir, f"res_{uid}.tmp")
        res_out = os.path.join(ipc_dir, f"res_{uid}.txt")

        expr = jsx_iife.strip().rstrip(";")
        wrapper = (
            "(function () {\n"
            '    var __resStr = "";\n'
            "    try {\n"
            f"        __resStr = String({expr});\n"
            "    } catch (e) {\n"
            '        __resStr = "ERR:" + String(e);\n'
            "    }\n"
            "    try {\n"
            f"        var __tmpF = new File({self._escape_js_str(tmp_out)});\n"
            '        __tmpF.encoding = "UTF-8";\n'
            '        if (__tmpF.open("w")) {\n'
            "            __tmpF.write(__resStr);\n"
            "            __tmpF.close();\n"
            "        }\n"
            f"        var __finalF = new File({self._escape_js_str(res_out)});\n"
            "        if (__finalF.exists) { try { __finalF.remove(); } catch (er) {} }\n"
            "        __tmpF.rename(__finalF.name);\n"
            "    } catch (ew) {}\n"
            "})();\n"
        )

        with self._jsx_lock:
            try:
                with open(job_jsx, "w", encoding="utf-8") as fp:
                    fp.write(wrapper)

                subprocess.Popen([exe_path, job_jsx])

                deadline = time.monotonic() + timeout_sec
                while time.monotonic() < deadline:
                    if os.path.exists(res_out):
                        try:
                            with open(res_out, "r", encoding="utf-8-sig") as fp:
                                data = fp.read()
                            return True, data
                        except OSError:
                            pass
                    time.sleep(0.08)

                return False, "Photoshop JSX yanıt vermedi (zaman aşımı)."
            except Exception as exc:
                return False, f"Photoshop JSX yürütme hatası: {exc}"
            finally:
                for p in (job_jsx, tmp_out, res_out):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass

    def _can_use_com(self, proc: dict) -> bool:
        if proc["pid"] in self._com_failed_pids:
            return False
        reg_exe = self._get_registered_com_exe()
        if not reg_exe or not proc.get("exe_path"):
            return False
        return os.path.normcase(os.path.abspath(proc["exe_path"])) == reg_exe

    def _exec_jsx(
        self, jsx_iife: str, launch_if_closed: bool = False, timeout_sec: float = 15.0
    ) -> Tuple[bool, str, str]:
        """Aktif Photoshop uzerinde JSX calistirir; donus: (ok, sonuc, transport)."""
        proc = self._find_running_photoshop()
        if proc is None:
            if not launch_if_closed:
                return False, "Açık Photoshop bulunamadı.", "none"
            ok_c, res_c = self._run_jsx_via_com("Photoshop.Application", jsx_iife, timeout_sec=max(timeout_sec, 20.0))
            return ok_c, res_c, "com"

        if self._can_use_com(proc):
            ok_c, res_c = self._run_jsx_via_com(proc["progid"], jsx_iife, timeout_sec=min(timeout_sec, 6.0))
            if ok_c:
                return True, res_c, "com"
            self._com_failed_pids.add(proc["pid"])

        ok_j, res_j = self._run_jsx_via_exe(proc["exe_path"], jsx_iife, timeout_sec=timeout_sec)
        return ok_j, res_j, "jsx"

    def _read_fresh_launch_ctx(self) -> Tuple[Optional[dict], bool]:
        """mechi_launch_ctx.json dosyasi son okumadan yeniyse belge bilgisini dondurur."""
        if not os.path.exists(LAUNCH_CTX_PATH):
            return None, False
        try:
            mtime = os.path.getmtime(LAUNCH_CTX_PATH)
            is_fresh = mtime > self._cached_ctx_mtime
            is_recent = (time.time() - mtime) <= 5.0
            if not is_fresh and not is_recent:
                return None, False
            self._cached_ctx_mtime = max(self._cached_ctx_mtime, mtime)
            with open(LAUNCH_CTX_PATH, "r", encoding="utf-8-sig") as fp:
                ctx = json.load(fp)
            if not isinstance(ctx, dict):
                return None, False
            name = str(ctx.get("active_doc_name") or "").strip()
            if not name:
                return None, is_recent
            lyr_cnt = int(ctx.get("layer_count") or 1)
            doc_info = {
                "name": name,
                "path": str(ctx.get("active_doc_path") or ""),
                "width_px": int(ctx.get("width_px") or 0),
                "height_px": int(ctx.get("height_px") or 0),
                "layer_count": lyr_cnt,
                "has_layers": lyr_cnt > 1,
            }
            return doc_info, is_recent
        except Exception:
            return None, False

    def get_status(self, force_jsx_probe: bool = False) -> dict:
        proc = self._find_running_photoshop()
        if proc is None:
            self._cached_pid = None
            self._cached_status = None
            return {
                "connected": False,
                "version": "",
                "year": "",
                "doc_count": 0,
                "active_doc": None,
                "transport": "none",
            }

        year = proc.get("year", "")
        can_com = self._can_use_com(proc)
        ctx_doc, ctx_recent = self._read_fresh_launch_ctx()

        if ctx_doc is not None and not force_jsx_probe and not can_com:
            st = {
                "connected": True,
                "version": year,
                "year": year,
                "doc_count": 1,
                "active_doc": ctx_doc,
                "transport": "jsx",
            }
            self._cached_pid = proc["pid"]
            self._cached_status = st
            return dict(st)

        should_probe = (
            can_com
            or force_jsx_probe
            or (proc["pid"] != self._cached_pid and not ctx_recent)
            or (self._cached_status is None and not ctx_recent)
        )

        if not should_probe and self._cached_status is not None:
            cached = dict(self._cached_status)
            cached["connected"] = True
            cached["year"] = year
            return cached

        jsx = """
        (function() {
            function q(s) {
                return '"' + String(s).replace(/\\\\/g, "\\\\\\\\").replace(/"/g, '\\\\"').replace(/\\r|\\n/g, " ") + '"';
            }
            if (app.documents.length === 0) {
                return '{"version":' + q(app.version) + ',"doc_count":0,"active_doc":null}';
            }
            var d = app.activeDocument;
            var prevUnits = app.preferences.rulerUnits;
            app.preferences.rulerUnits = Units.PIXELS;
            var w = 0, h = 0, pathStr = "", lyrCnt = 0;
            try { w = Math.round(d.width.as("px")); } catch (e1) {}
            try { h = Math.round(d.height.as("px")); } catch (e2) {}
            try { pathStr = d.fullName.fsName; } catch (e3) { pathStr = ""; }
            try { lyrCnt = d.layers.length; } catch (e4) { lyrCnt = 0; }
            app.preferences.rulerUnits = prevUnits;
            return '{"version":' + q(app.version) + ',"doc_count":' + app.documents.length + ',"active_doc":{"name":' + q(d.name) + ',"path":' + q(pathStr) + ',"width_px":' + w + ',"height_px":' + h + ',"layer_count":' + lyrCnt + ',"has_layers":' + (lyrCnt > 1 ? 'true' : 'false') + '}}';
        })()
        """
        ok, raw, transport = self._exec_jsx(jsx, launch_if_closed=False, timeout_sec=3.5)
        if ok and raw and not raw.startswith("ERR:"):
            try:
                parsed = json.loads(raw)
                st = {
                    "connected": True,
                    "version": str(parsed.get("version") or year),
                    "year": year,
                    "doc_count": int(parsed.get("doc_count") or 0),
                    "active_doc": parsed.get("active_doc"),
                    "transport": transport,
                }
                self._cached_pid = proc["pid"]
                self._cached_status = st
                return dict(st)
            except Exception:
                pass

        if self._cached_status is not None and proc["pid"] == self._cached_pid:
            cached = dict(self._cached_status)
            cached["connected"] = True
            cached["year"] = year
            return cached

        st_fallback = {
            "connected": True,
            "version": year,
            "year": year,
            "doc_count": 1 if ctx_doc else 0,
            "active_doc": ctx_doc,
            "transport": transport,
        }
        self._cached_pid = proc["pid"]
        self._cached_status = st_fallback
        return dict(st_fallback)

    def _find_installed_photoshop_exe(self) -> str:
        """Calisan veya kurulu en uygun Photoshop.exe yolunu bulur."""
        import glob

        proc = self._find_running_photoshop()
        if proc and proc.get("exe_path") and os.path.exists(proc["exe_path"]):
            return str(proc["exe_path"])

        candidates: List[Tuple[str, str]] = []
        for p in glob.glob(r"C:\Program Files\Adobe\Adobe Photoshop *\Photoshop.exe"):
            if os.path.exists(p):
                m_yr = re.search(r"(20\d{2})", p)
                yr = m_yr.group(1) if m_yr else "0000"
                candidates.append((yr, p))

        pref = str(self.preferred_version or "").strip()
        if pref and pref != "auto":
            for yr, p in candidates:
                if yr == pref:
                    return p

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        reg_exe = self._get_registered_com_exe()
        if reg_exe and os.path.exists(reg_exe):
            return reg_exe
        return ""

    def open_file(self, file_path: str) -> Tuple[bool, str]:
        if not file_path or not os.path.exists(file_path):
            return False, f"Dosya bulunamadı: {file_path}"
        abs_path = os.path.abspath(file_path)
        exe = self._find_installed_photoshop_exe()
        if exe:
            try:
                subprocess.Popen([exe, abs_path])
                return True, abs_path
            except Exception as exc:
                return False, f"Photoshop başlatılamadı: {exc}"
        return False, "Photoshop kurulumu bulunamadı. 'İndirilenlere İndir' ile dosyayı açabilirsiniz."

    def export_active_doc_png(
        self, out_png_path: str, expected_doc_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
        js_out = self._escape_js_str(os.path.abspath(out_png_path))
        js_exp = self._escape_js_str(expected_doc_name or "")
        jsx = f"""
        (function() {{
            if (app.documents.length === 0) return "ERR:Photoshop'ta acik belge yok.";
            var doc = app.activeDocument;
            var expName = {js_exp};
            if (expName && doc.name !== expName) {{
                for (var i = 0; i < app.documents.length; i++) {{
                    if (app.documents[i].name === expName) {{ doc = app.documents[i]; break; }}
                }}
            }}
            var dup = doc.duplicate("_mechi_tmp_export", true);
            try {{
                if (dup.mode !== DocumentMode.RGB) dup.changeMode(ChangeMode.RGB);
                if (dup.bitsPerChannel !== BitsPerChannelType.EIGHT) dup.bitsPerChannel = BitsPerChannelType.EIGHT;
                var opts = new PNGSaveOptions();
                opts.compression = 6;
                opts.interlaced = false;
                var outF = new File({js_out});
                dup.saveAs(outF, opts, true, Extension.LOWERCASE);
            }} finally {{
                try {{ dup.close(SaveOptions.DONOTSAVECHANGES); }} catch (eClose) {{}}
            }}
            return "OK";
        }})()
        """
        ok, res, _ = self._exec_jsx(jsx, launch_if_closed=False, timeout_sec=25.0)
        if ok and res == "OK" and os.path.exists(out_png_path):
            return True, out_png_path
        return False, res

    def export_raster_layers_for_enhance(
        self,
        src_psd_or_none: Optional[str],
        work_dir: str,
        expected_doc_name: Optional[str] = None,
    ) -> Tuple[bool, object]:
        layers_dir = os.path.join(work_dir, "layers")
        os.makedirs(layers_dir, exist_ok=True)
        manifest_path = os.path.join(work_dir, "manifest.json")

        js_src = self._escape_js_str(os.path.abspath(src_psd_or_none)) if src_psd_or_none else '""'
        js_layers_dir = self._escape_js_str(os.path.abspath(layers_dir))
        js_manifest = self._escape_js_str(os.path.abspath(manifest_path))
        js_exp = self._escape_js_str(expected_doc_name or "")

        jsx = f"""
        (function() {{
            function q(s) {{
                return '"' + String(s).replace(/\\\\/g, "\\\\\\\\").replace(/"/g, '\\\\"').replace(/\\r|\\n/g, " ") + '"';
            }}
            var prevDialogs = app.displayDialogs;
            var prevUnits = app.preferences.rulerUnits;
            app.displayDialogs = DialogModes.NO;
            app.preferences.rulerUnits = Units.PIXELS;
            var srcPath = {js_src};
            var expName = {js_exp};
            var openedTemp = null;
            var workDoc = null;
            try {{
                var targetSrc = null;
                if (srcPath) {{
                    var f = new File(srcPath);
                    if (!f.exists) return "ERR:Kaynak dosya bulunamadi.";
                    openedTemp = app.open(f);
                    targetSrc = openedTemp;
                }} else {{
                    if (app.documents.length === 0) return "ERR:Photoshop'ta acik belge yok.";
                    targetSrc = app.activeDocument;
                    if (expName && targetSrc.name !== expName) {{
                        for (var i = 0; i < app.documents.length; i++) {{
                            if (app.documents[i].name === expName) {{ targetSrc = app.documents[i]; break; }}
                        }}
                    }}
                }}

                var baseName = targetSrc.name.replace(/\\.[^\\.]+$/, "");
                workDoc = targetSrc.duplicate(baseName + "_mechi_net", false);
                if (openedTemp) {{
                    try {{ openedTemp.close(SaveOptions.DONOTSAVECHANGES); }} catch (e0) {{}}
                    openedTemp = null;
                }}

                if (workDoc.mode !== DocumentMode.RGB) workDoc.changeMode(ChangeMode.RGB);
                if (workDoc.bitsPerChannel !== BitsPerChannelType.EIGHT) workDoc.bitsPerChannel = BitsPerChannelType.EIGHT;

                // Arka plan katmani varsa gizlenebilmesi icin kilidini ac
                try {{
                    var lastLyr = workDoc.layers[workDoc.layers.length - 1];
                    if (lastLyr.isBackgroundLayer) lastLyr.isBackgroundLayer = false;
                }} catch (eb) {{}}

                var allLeaves = [];
                var rasterItems = [];
                var skippedCount = 0;

                function collect(container, pathArr) {{
                    for (var i = 0; i < container.layers.length; i++) {{
                        var lyr = container.layers[i];
                        var curPath = pathArr.concat([i]);
                        if (lyr.typename === "LayerSet") {{
                            collect(lyr, curPath);
                        }} else {{
                            var origVis = lyr.visible;
                            allLeaves.push({{ layer: lyr, vis: origVis }});
                            if (lyr.kind === LayerKind.NORMAL && origVis) {{
                                var b = lyr.bounds;
                                var bw = b[2].as("px") - b[0].as("px");
                                var bh = b[3].as("px") - b[1].as("px");
                                if (bw > 0 && bh > 0) {{
                                    rasterItems.push({{ layer: lyr, path: curPath, name: lyr.name }});
                                }}
                            }} else if (origVis) {{
                                skippedCount++;
                            }}
                        }}
                    }}
                }}
                collect(workDoc, []);

                // Tum yaprak katmanlari gecici gizle
                for (var k = 0; k < allLeaves.length; k++) {{
                    try {{ allLeaves[k].layer.visible = false; }} catch (ev) {{}}
                }}

                var exportedMeta = [];
                var pngOpts = new PNGSaveOptions();
                pngOpts.compression = 6;
                pngOpts.interlaced = false;

                for (var r = 0; r < rasterItems.length; r++) {{
                    var item = rasterItems[r];
                    var lyr = item.layer;
                    var savedOp = lyr.opacity;
                    var savedFill = lyr.fillOpacity;
                    var savedBlend = lyr.blendMode;
                    var ancestors = [];
                    var pNode = lyr.parent;
                    while (pNode && pNode.typename === "LayerSet") {{
                        ancestors.push({{ node: pNode, vis: pNode.visible }});
                        pNode.visible = true;
                        pNode = pNode.parent;
                    }}
                    lyr.visible = true;
                    try {{ lyr.opacity = 100; lyr.fillOpacity = 100; lyr.blendMode = BlendMode.NORMAL; }} catch (ep) {{}}

                    var pngPath = {js_layers_dir} + "/layer_" + r + ".png";
                    workDoc.saveAs(new File(pngPath), pngOpts, true, Extension.LOWERCASE);

                    try {{ lyr.opacity = savedOp; lyr.fillOpacity = savedFill; lyr.blendMode = savedBlend; }} catch (ep2) {{}}
                    lyr.visible = false;
                    for (var a = 0; a < ancestors.length; a++) {{
                        ancestors[a].node.visible = ancestors[a].vis;
                    }}

                    exportedMeta.push(
                        '{{"index":' + r +
                        ',"path":[' + item.path.join(",") + ']' +
                        ',"name":' + q(item.name) +
                        ',"png_path":' + q(pngPath) + '}}'
                    );
                }}

                // Orijinal gorunurlukleri geri yukle
                for (var k2 = 0; k2 < allLeaves.length; k2++) {{
                    try {{ allLeaves[k2].layer.visible = allLeaves[k2].vis; }} catch (ev2) {{}}
                }}

                var mf = new File({js_manifest});
                mf.encoding = "UTF-8";
                mf.open("w");
                mf.write('{{"doc_name":' + q(workDoc.name) + ',"skipped_vector_count":' + skippedCount + ',"layers":[' + exportedMeta.join(",") + ']}}');
                mf.close();
                return "OK";
            }} finally {{
                if (openedTemp) {{ try {{ openedTemp.close(SaveOptions.DONOTSAVECHANGES); }} catch (e1) {{}} }}
                app.preferences.rulerUnits = prevUnits;
                app.displayDialogs = prevDialogs;
            }}
        }})()
        """
        ok, res, _ = self._exec_jsx(jsx, launch_if_closed=bool(src_psd_or_none), timeout_sec=60.0)
        if not ok or res != "OK" or not os.path.exists(manifest_path):
            return False, res
        with open(manifest_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return True, data

    def reimport_enhanced_layers(
        self,
        work_dir: str,
        out_psd_path: Optional[str],
        keep_open: bool = True,
    ) -> Tuple[bool, str]:
        manifest_path = os.path.join(work_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return False, "Katman manifest.json bulunamadı."
        js_manifest = self._escape_js_str(os.path.abspath(manifest_path))
        js_out_psd = self._escape_js_str(os.path.abspath(out_psd_path)) if out_psd_path else '""'
        keep_js = "true" if keep_open else "false"

        jsx = f"""
        (function() {{
            var prevDialogs = app.displayDialogs;
            var prevUnits = app.preferences.rulerUnits;
            app.displayDialogs = DialogModes.NO;
            app.preferences.rulerUnits = Units.PIXELS;
            try {{
                var mf = new File({js_manifest});
                mf.encoding = "UTF-8";
                mf.open("r");
                var meta = eval("(" + mf.read() + ")");
                mf.close();

                var doc = null;
                for (var d = 0; d < app.documents.length; d++) {{
                    if (app.documents[d].name === meta.doc_name) {{
                        doc = app.documents[d];
                        break;
                    }}
                }}
                if (!doc) return "ERR:Calisma belgesi Photoshop'ta bulunamadi.";
                app.activeDocument = doc;

                function resolveLayer(rootDoc, pathArr) {{
                    var cur = rootDoc;
                    for (var i = 0; i < pathArr.length; i++) {{
                        cur = cur.layers[pathArr[i]];
                    }}
                    return cur;
                }}

                function placePngAsLayer(targetDoc, pngFile) {{
                    var desc = new ActionDescriptor();
                    desc.putPath(charIDToTypeID("null"), pngFile);
                    desc.putEnumerated(charIDToTypeID("FTcs"), charIDToTypeID("QCSt"), charIDToTypeID("Qcsa"));
                    executeAction(charIDToTypeID("Plc "), desc, DialogModes.NO);
                    var placed = targetDoc.activeLayer;
                    try {{ placed.rasterize(RasterizeType.ENTIRELAYER); }} catch (er) {{}}
                    return targetDoc.activeLayer;
                }}

                for (var i = 0; i < meta.layers.length; i++) {{
                    var item = meta.layers[i];
                    var pngFile = new File(item.png_path);
                    if (!pngFile.exists) continue;
                    var oldLyr = resolveLayer(doc, item.path);
                    var oldName = oldLyr.name;
                    var oldOp = oldLyr.opacity;
                    var oldFill = oldLyr.fillOpacity;
                    var oldBlend = oldLyr.blendMode;
                    var oldVis = oldLyr.visible;
                    var oldGrouped = oldLyr.grouped;

                    doc.activeLayer = oldLyr;
                    oldLyr.visible = true;
                    var newLyr = placePngAsLayer(doc, pngFile);
                    newLyr.move(oldLyr, ElementPlacement.PLACEBEFORE);
                    doc.activeLayer = oldLyr;
                    try {{
                        doc.selection.selectAll();
                        doc.selection.clear();
                        doc.selection.deselect();
                    }} catch (ec) {{}}
                    doc.activeLayer = newLyr;
                    var merged = newLyr.merge();
                    merged.name = oldName;
                    merged.opacity = oldOp;
                    merged.fillOpacity = oldFill;
                    merged.blendMode = oldBlend;
                    merged.visible = oldVis;
                    try {{ merged.grouped = oldGrouped; }} catch (eg) {{}}
                }}

                var outPath = {js_out_psd};
                if (outPath) {{
                    var outFile = new File(outPath);
                    if (outFile.parent && !outFile.parent.exists) outFile.parent.create();
                    var psdOpts = new PhotoshopSaveOptions();
                    psdOpts.layers = true;
                    psdOpts.alphaChannels = true;
                    psdOpts.spotColors = true;
                    doc.saveAs(outFile, psdOpts, false, Extension.LOWERCASE);
                }}
                if (!{keep_js}) {{
                    doc.close(SaveOptions.DONOTSAVECHANGES);
                }}
                return outPath ? outPath : doc.name;
            }} finally {{
                app.preferences.rulerUnits = prevUnits;
                app.displayDialogs = prevDialogs;
            }}
        }})()
        """
        ok, res, _ = self._exec_jsx(jsx, launch_if_closed=False, timeout_sec=60.0)
        if not ok or str(res).startswith("ERR:"):
            return False, str(res)
        return True, str(res)
