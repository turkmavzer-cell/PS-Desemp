#target photoshop

/**
 * Kedihane — Hibrit LAP + Dot-on-Dot Emprime Renk Ayirici v2 (5 Adimli Sihirbaz)
 *
 * Ozellikler:
 *   1. 5 Adimli Interaktif Sihirbaz (< Geri / Ileri > / Iptal / Baslat):
 *      - Adim 1: Kaynak (Acik belge veya dosya) & Tisort Zemini (Ikisi / Koyu / Acik)
 *      - Adim 2: Kafa Sayisi & Ayrim Profili (Profesyonel / Ekonomik 6 / Ozel 5-10) + Siyah (CMYK K / Lab)
 *      - Adim 3: Baski Olcusu (cm / PPI) & Dis Zemin Maskesi (Oto / Maskeli / Tumu)
 *      - Adim 4: Tram (45/55 LPI), Aci (15/45/72), Choke, %5 Baski Acma, %85 Degrade, Kros+Etiket
 *      - Adim 5: Cikti Bicimi (Tek PSD / Tek TIF / Yalnizca Acik Belge) & Ozet -> Baslat
 *   2. Sifir Klasor Kalabaligi: Tum gecici islemler %TEMP%\kedihane_emp_<ts> altinda yapilir,
 *      yalnizca tek calisma dosyasi uretilip Photoshop'ta acilir ve %TEMP% temizlenir.
 */

(function () {
    app.displayDialogs = DialogModes.NO;

    function trAscii(s) {
        if (!s) return "";
        return String(s)
            .replace(/\u00e7/g, "c").replace(/\u00c7/g, "C")
            .replace(/\u011f/g, "g").replace(/\u011e/g, "G")
            .replace(/\u0131/g, "i").replace(/\u0130/g, "I")
            .replace(/\u00f6/g, "o").replace(/\u00d6/g, "O")
            .replace(/\u015f/g, "s").replace(/\u015e/g, "S")
            .replace(/\u00fc/g, "u").replace(/\u00dc/g, "U")
            .replace(/"/g, "");
    }

    function deriveShortJobName(rawName) {
        if (!rawName) return "desen";
        var stem = decodeURI(String(rawName)).replace(/\.[^\.]+$/, "").toLowerCase();
        stem = stem.replace(/(calisma|\u00e7al\u0131\u015fma|tasarim|tasar\u0131m|baski|bask\u0131|desen)/g, "");
        stem = trAscii(stem).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
        return stem || "desen";
    }

    function findPythonExe() {
        var candidates = [
            "C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
            "C:\\Users\\dell\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
            "C:\\Python312\\python.exe"
        ];
        for (var i = 0; i < candidates.length; i++) {
            var f = new File(candidates[i]);
            if (f.exists) return f.fsName;
        }
        return "python";
    }

    function findBackendScript() {
        var thisFile = new File($.fileName);
        var dir = thisFile.parent;
        var candidates = [
            dir.fsName + "\\..\\lap_aile_ayir.py",
            dir.fsName + "\\lap_aile_ayir.py",
            "C:\\Users\\dell\\.gemini\\antigravity\\scratch\\PS-Desemp\\tools\\lap_aile_ayir.py"
        ];
        for (var i = 0; i < candidates.length; i++) {
            var f = new File(candidates[i]);
            if (f.exists) return f.fsName;
        }
        return null;
    }

    function readU16BE(s, off) {
        return (s.charCodeAt(off) << 8) | s.charCodeAt(off + 1);
    }

    function readU32BE(s, off) {
        return ((s.charCodeAt(off) << 24) >>> 0)
            + (s.charCodeAt(off + 1) << 16)
            + (s.charCodeAt(off + 2) << 8)
            + s.charCodeAt(off + 3);
    }

    function readU16LE(s, off) {
        return s.charCodeAt(off) | (s.charCodeAt(off + 1) << 8);
    }

    function readU32LE(s, off) {
        return (s.charCodeAt(off))
            + (s.charCodeAt(off + 1) << 8)
            + (s.charCodeAt(off + 2) << 16)
            + ((s.charCodeAt(off + 3) << 24) >>> 0);
    }

    function readImageDimensions(fileObj) {
        if (!fileObj || !fileObj.exists) return null;
        var f = new File(fileObj.fsName);
        f.encoding = "BINARY";
        if (!f.open("r")) return null;
        try {
            var head = f.read(524288);
            if (!head || head.length < 24) return null;

            // PNG: 89 50 4E 47
            if (head.charCodeAt(0) === 0x89 && head.substr(1, 3) === "PNG") {
                return { width: readU32BE(head, 16), height: readU32BE(head, 20) };
            }
            // PSD: 8BPS
            if (head.substr(0, 4) === "8BPS") {
                return { width: readU32BE(head, 18), height: readU32BE(head, 14) };
            }
            // JPEG / JFIF: FF D8
            if (head.charCodeAt(0) === 0xFF && head.charCodeAt(1) === 0xD8) {
                var pos = 2;
                while (pos + 9 < head.length) {
                    if (head.charCodeAt(pos) !== 0xFF) { pos++; continue; }
                    while (pos < head.length && head.charCodeAt(pos) === 0xFF) pos++;
                    if (pos >= head.length) break;
                    var marker = head.charCodeAt(pos);
                    pos++;
                    if (marker === 0xD9 || marker === 0xDA) break;
                    if (marker === 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;
                    if (pos + 2 > head.length) break;
                    var segLen = readU16BE(head, pos);
                    if (marker >= 0xC0 && marker <= 0xCF && marker !== 0xC4 && marker !== 0xC8 && marker !== 0xCC) {
                        if (pos + 7 < head.length) {
                            return { width: readU16BE(head, pos + 5), height: readU16BE(head, pos + 3) };
                        }
                    }
                    pos += segLen;
                }
            }
            // TIFF: II*\0 or MM\0*
            var isLE = (head.substr(0, 2) === "II" && readU16LE(head, 2) === 42);
            var isBE = (head.substr(0, 2) === "MM" && readU16BE(head, 2) === 42);
            if (isLE || isBE) {
                var ifdOff = isLE ? readU32LE(head, 4) : readU32BE(head, 4);
                if (ifdOff + 2 < head.length) {
                    var cnt = isLE ? readU16LE(head, ifdOff) : readU16BE(head, ifdOff);
                    var tw = 0, th = 0;
                    for (var i = 0; i < cnt; i++) {
                        var eOff = ifdOff + 2 + i * 12;
                        if (eOff + 12 > head.length) break;
                        var tag = isLE ? readU16LE(head, eOff) : readU16BE(head, eOff);
                        var typ = isLE ? readU16LE(head, eOff + 2) : readU16BE(head, eOff + 2);
                        var val = (typ === 3)
                            ? (isLE ? readU16LE(head, eOff + 8) : readU16BE(head, eOff + 8))
                            : (isLE ? readU32LE(head, eOff + 8) : readU32BE(head, eOff + 8));
                        if (tag === 256) tw = val;
                        if (tag === 257) th = val;
                    }
                    if (tw > 0 && th > 0) return { width: tw, height: th };
                }
            }
        } catch (e) {
        } finally {
            try { f.close(); } catch (e2) {}
        }
        return null;
    }

    function getOpenDocumentInfo() {
        if (app.documents.length === 0) return null;
        try {
            var doc = app.activeDocument;
            var fullFile = null;
            try { fullFile = doc.fullName; } catch (e0) { fullFile = null; }
            var parentDir = (fullFile && fullFile.parent && fullFile.parent.exists)
                ? fullFile.parent.fsName
                : Folder.desktop.fsName;
            var pxW = 0, pxH = 0;
            try {
                pxW = Math.round(doc.width.as("px"));
                pxH = Math.round(doc.height.as("px"));
            } catch (ed) {}
            return {
                doc: doc,
                name: doc.name,
                fullFile: fullFile,
                parentDir: parentDir,
                jobName: deriveShortJobName(doc.name),
                widthPx: pxW,
                heightPx: pxH
            };
        } catch (e) {
            return null;
        }
    }

    function deleteFolderRecursive(folder) {
        if (!folder || !folder.exists) return;
        try {
            var items = folder.getFiles();
            for (var i = 0; i < items.length; i++) {
                if (items[i] instanceof Folder) {
                    deleteFolderRecursive(items[i]);
                } else {
                    try { items[i].remove(); } catch (e1) {}
                }
            }
            folder.remove();
        } catch (e2) {}
    }

    function exportDocToTempPng(doc, tmpPngFile) {
        var dup = doc.duplicate("_kedihane_tmp_export", true);
        try {
            if (dup.mode !== DocumentMode.RGB) {
                dup.changeMode(ChangeMode.RGB);
            }
            if (dup.bitsPerChannel !== BitsPerChannelType.EIGHT) {
                dup.bitsPerChannel = BitsPerChannelType.EIGHT;
            }
            var pngOpts = new PNGSaveOptions();
            pngOpts.compression = 6;
            pngOpts.interlaced = false;
            dup.saveAs(tmpPngFile, pngOpts, true, Extension.LOWERCASE);
        } finally {
            try { dup.close(SaveOptions.DONOTSAVECHANGES); } catch (e3) {}
        }
        return tmpPngFile.exists;
    }

    function runWizard(openInfo) {
        var win = new Window("dialog", "Kedihane — Emprime Renk Ay\u0131r\u0131c\u0131 v2 (5 Ad\u0131ml\u0131 Sihirbaz)");
        win.orientation = "column";
        win.alignChildren = ["fill", "top"];
        win.spacing = 10;
        win.margins = 16;
        win.preferredSize = [560, -1];

        var hdrGroup = win.add("group");
        hdrGroup.orientation = "column";
        hdrGroup.alignChildren = ["fill", "top"];
        var lblStepTitle = hdrGroup.add("statictext", undefined, "Ad\u0131m 1 / 5 \u2014 Kaynak & Ti\u015f\u00f6rt Zemini");
        try { lblStepTitle.graphics.font = ScriptUI.newFont(lblStepTitle.graphics.font.name, "Bold", 13); } catch (ef) {}
        var sepTop = hdrGroup.add("panel");
        sepTop.preferredSize = [-1, 2];

        var stack = win.add("group");
        stack.orientation = "stack";
        stack.alignChildren = ["fill", "fill"];

        // ==================== ADIM 1: Kaynak & Tisort Zemini ====================
        var p1 = stack.add("group");
        p1.orientation = "column";
        p1.alignChildren = ["fill", "top"];
        p1.spacing = 10;

        var pnlSrc = p1.add("panel", undefined, "Kaynak G\u00f6rsel");
        pnlSrc.orientation = "column";
        pnlSrc.alignChildren = ["fill", "top"];
        pnlSrc.spacing = 6;
        pnlSrc.margins = 12;

        var rbUseOpen = pnlSrc.add("radiobutton", undefined,
            openInfo ? ("Photoshop'ta A\u00e7\u0131k Belgeyi Kullan: " + openInfo.name) : "A\u00e7\u0131k Belge Yok"
        );
        var rbUseFile = pnlSrc.add("radiobutton", undefined, "Diskten Dosya Se\u00e7...");
        if (openInfo) {
            rbUseOpen.value = true;
        } else {
            rbUseOpen.enabled = false;
            rbUseFile.value = true;
        }

        var grpFilePick = pnlSrc.add("group");
        grpFilePick.alignChildren = ["fill", "center"];
        var txtFilePath = grpFilePick.add("edittext", undefined, (openInfo && openInfo.fullFile) ? openInfo.fullFile.fsName : "");
        txtFilePath.characters = 42;
        var btnBrowseFile = grpFilePick.add("button", undefined, "G\u00f6zat...");
        btnBrowseFile.preferredSize = [82, 24];

        var pnlFabric = p1.add("panel", undefined, "Ti\u015f\u00f6rt Zemini");
        pnlFabric.orientation = "column";
        pnlFabric.alignChildren = ["left", "top"];
        pnlFabric.spacing = 6;
        pnlFabric.margins = 12;

        var rbKumasIkisi = pnlFabric.add("radiobutton", undefined, "\u0130kisi Birlikte (Koyu + A\u00e7\u0131k Ti\u015f\u00f6rt \u2014 3 Katman + 2 Kanal Seti)");
        var rbKumasKoyu = pnlFabric.add("radiobutton", undefined, "Koyu / Siyah Ti\u015f\u00f6rt (Beyaz LAP fon + \u00fcst tramlar)");
        var rbKumasAcik = pnlFabric.add("radiobutton", undefined, "A\u00e7\u0131k / Beyaz Ti\u015f\u00f6rt (Fonsuz \u00fcst tramlar)");
        rbKumasIkisi.value = true;

        // ==================== ADIM 2: Kafa Sayisi & Profil ====================
        var p2 = stack.add("group");
        p2.orientation = "column";
        p2.alignChildren = ["fill", "top"];
        p2.spacing = 10;

        var pnlProfil = p2.add("panel", undefined, "Ayr\u0131m Profili & Kafa Say\u0131s\u0131");
        pnlProfil.orientation = "column";
        pnlProfil.alignChildren = ["left", "top"];
        pnlProfil.spacing = 7;
        pnlProfil.margins = 12;

        var rbProfPro = pnlProfil.add("radiobutton", undefined, "Profesyonel Tam Ayr\u0131m (\u00d6nerilen \u2014 A\u00e7\u0131k Baz LAP + %58 Koyu Tram, ~8 Kafa)");
        var rbProfEko = pnlProfil.add("radiobutton", undefined, "Ekonomik 6 Kafa (At\u00f6lye 6 Kafa Standard\u0131)");
        var grpCustomHead = pnlProfil.add("group");
        var rbProfCustom = grpCustomHead.add("radiobutton", undefined, "\u00d6zel Kafa Say\u0131s\u0131 (5\u201310, fon dahil):");
        var txtCustomHead = grpCustomHead.add("edittext", undefined, "8");
        txtCustomHead.characters = 4;
        txtCustomHead.enabled = false;
        rbProfPro.value = true;

        rbProfPro.onClick = function () { txtCustomHead.enabled = false; };
        rbProfEko.onClick = function () { txtCustomHead.enabled = false; };
        rbProfCustom.onClick = function () { txtCustomHead.enabled = true; };

        var pnlSiyah = p2.add("panel", undefined, "Siyah Kanal\u0131 (Siyah_tram) \u00dcretim Y\u00f6ntemi");
        pnlSiyah.orientation = "column";
        pnlSiyah.alignChildren = ["left", "top"];
        pnlSiyah.spacing = 6;
        pnlSiyah.margins = 12;

        var rbSiyahCmyk = pnlSiyah.add("radiobutton", undefined, "CMYK K Kanal\u0131 (Usta \u00d6nerisi \u2014 FOGRA39 ger\u00e7ek siyah tram)");
        var rbSiyahLab = pnlSiyah.add("radiobutton", undefined, "Lab L E\u011frisi (Klasik parlakl\u0131k e\u011frisi)");
        rbSiyahCmyk.value = true;

        // ==================== ADIM 3: Baski Olcusu & Zemin Maskesi ====================
        var p3 = stack.add("group");
        p3.orientation = "column";
        p3.alignChildren = ["fill", "top"];
        p3.spacing = 10;

        var pnlSize = p3.add("panel", undefined, "Bask\u0131 \u00d6l\u00e7\u00fcs\u00fc");
        pnlSize.orientation = "column";
        pnlSize.alignChildren = ["left", "top"];
        pnlSize.spacing = 8;
        pnlSize.margins = 12;

        var grpDimIn = pnlSize.add("group");
        grpDimIn.add("statictext", undefined, "En (cm):");
        var txtWidthCm = grpDimIn.add("edittext", undefined, "28.0");
        txtWidthCm.characters = 6;
        grpDimIn.add("statictext", undefined, "  \u00c7\u00f6z\u00fcn\u00fcrl\u00fck (PPI):");
        var txtPPI = grpDimIn.add("edittext", undefined, "508");
        txtPPI.characters = 5;

        var grpPresets = pnlSize.add("group");
        grpPresets.add("statictext", undefined, "H\u0131zl\u0131 Se\u00e7im:");
        var btnW24 = grpPresets.add("button", undefined, "24 cm");
        var btnW28 = grpPresets.add("button", undefined, "28 cm (Varsay\u0131lan)");
        var btnW32 = grpPresets.add("button", undefined, "32 cm");

        var lblAutoSize1 = pnlSize.add("statictext", undefined, "Boy (otomatik): \u2014");
        lblAutoSize1.preferredSize = [480, 18];
        var lblAutoSize2 = pnlSize.add("statictext", undefined, "");
        lblAutoSize2.preferredSize = [480, 18];

        var cachedDimsPath = "";
        var cachedDims = null;

        function getCurrentSourceDims() {
            if (rbUseOpen.value && openInfo && openInfo.widthPx > 0 && openInfo.heightPx > 0) {
                return { width: openInfo.widthPx, height: openInfo.heightPx };
            }
            var p = txtFilePath.text;
            if (!p) return null;
            if (p !== cachedDimsPath) {
                cachedDimsPath = p;
                cachedDims = readImageDimensions(new File(p));
            }
            return cachedDims;
        }

        function computeSizeMetrics() {
            var enCm = parseFloat(String(txtWidthCm.text).replace(",", "."));
            var ppi = parseInt(txtPPI.text, 10) || 508;
            if (isNaN(enCm) || enCm <= 0 || ppi <= 0) return null;
            var dims = getCurrentSourceDims();
            if (!dims || !dims.width || !dims.height) {
                return { enCm: enCm, boyCm: null, ppi: ppi, targetW: Math.round(enCm / 2.54 * ppi), targetH: null };
            }
            var targetW = Math.round(enCm / 2.54 * ppi);
            var targetH = Math.round(dims.height * (targetW / dims.width));
            var boyCm = enCm * (dims.height / dims.width);
            return { enCm: enCm, boyCm: boyCm, ppi: ppi, targetW: targetW, targetH: targetH };
        }

        function updateSizePreview() {
            var m = computeSizeMetrics();
            if (!m) {
                lblAutoSize1.text = "Boy (otomatik): Ge\u00e7erli bir En (cm) girin";
                lblAutoSize2.text = "";
                return;
            }
            if (m.boyCm === null) {
                lblAutoSize1.text = "Boy (otomatik): G\u00f6rsel oran\u0131na g\u00f6re hesaplanacak | En: " + m.targetW + " px @ " + m.ppi + " PPI";
                lblAutoSize2.text = chkKros.value ? "(+3.0 cm etiket/kros \u015feridi alt k\u0131sma eklenecek)" : "(Etiket/kros kapal\u0131)";
                return;
            }
            lblAutoSize1.text = "Boy (otomatik): " + m.boyCm.toFixed(1) + " cm  |  Piksel: " + m.targetW + " x " + m.targetH + " px @ " + m.ppi + " PPI";
            if (chkKros.value) {
                var stripPx = Math.round(30 * (m.ppi / 25.4));
                var totalBoyCm = m.boyCm + 3.0;
                lblAutoSize2.text = "(+3.0 cm etiket/kros = " + totalBoyCm.toFixed(1) + " cm toplam \u2192 " + m.targetW + " x " + (m.targetH + stripPx) + " px)";
            } else {
                lblAutoSize2.text = "(Etiket/kros kapal\u0131 \u2014 net bask\u0131 tuvali: " + m.enCm.toFixed(1) + " x " + m.boyCm.toFixed(1) + " cm)";
            }
        }

        btnW24.onClick = function () { txtWidthCm.text = "24.0"; updateSizePreview(); };
        btnW28.onClick = function () { txtWidthCm.text = "28.0"; updateSizePreview(); };
        btnW32.onClick = function () { txtWidthCm.text = "32.0"; updateSizePreview(); };
        txtWidthCm.onChanging = updateSizePreview;
        txtWidthCm.onChange = updateSizePreview;
        txtPPI.onChanging = updateSizePreview;
        txtPPI.onChange = updateSizePreview;
        rbUseOpen.onClick = updateSizePreview;
        rbUseFile.onClick = updateSizePreview;

        var pnlBg = p3.add("panel", undefined, "Arka Plan (D\u0131\u015f Zemin) Maskesi");
        pnlBg.orientation = "column";
        pnlBg.alignChildren = ["left", "top"];
        pnlBg.spacing = 6;
        pnlBg.margins = 12;

        var rbBgOto = pnlBg.add("radiobutton", undefined, "Otomatik (Metraj desende t\u00fcm tuval, tekil bask\u0131da d\u0131\u015f zemin maskeli)");
        var rbBgMask = pnlBg.add("radiobutton", undefined, "D\u0131\u015f beyaz zemin bask\u0131 d\u0131\u015f\u0131 (maskele + halat cebi temizle)");
        var rbBgTumu = pnlBg.add("radiobutton", undefined, "T\u00fcm tuval bas\u0131ls\u0131n (maske yok \u2014 metraj / dolu zemin)");
        rbBgOto.value = true;

        // ==================== ADIM 4: Tram, Aci, Baski Acma & Kros ====================
        var p4 = stack.add("group");
        p4.orientation = "column";
        p4.alignChildren = ["fill", "top"];
        p4.spacing = 8;

        var pnlScreen = p4.add("panel", undefined, "Tram Frekans\u0131, A\u00e7\u0131 ve Telafi Ayarlar\u0131");
        pnlScreen.orientation = "column";
        pnlScreen.alignChildren = ["left", "top"];
        pnlScreen.spacing = 6;
        pnlScreen.margins = 10;

        var grpLpi = pnlScreen.add("group");
        grpLpi.add("statictext", undefined, "Bask\u0131 T\u00fcr\u00fc:");
        var rbLpi45 = grpLpi.add("radiobutton", undefined, "Spot Emprime (45 LPI)");
        var rbLpi55 = grpLpi.add("radiobutton", undefined, "CMYK Foto\u011fraf Bask\u0131 (55 LPI)");
        rbLpi45.value = true;

        var grpAng = pnlScreen.add("group");
        grpAng.add("statictext", undefined, "Tram A\u00e7\u0131s\u0131:");
        var rbAng15 = grpAng.add("radiobutton", undefined, "15\u00b0 (At\u00f6lye Standard\u0131)");
        var rbAng45 = grpAng.add("radiobutton", undefined, "45\u00b0");
        var rbAng72 = grpAng.add("radiobutton", undefined, "72\u00b0 (Usta Tercihi)");
        rbAng15.value = true;

        var grpChoke = pnlScreen.add("group");
        grpChoke.add("statictext", undefined, "Fon Choke:");
        var ddChoke = grpChoke.add("dropdownlist", undefined, ["2 px (Genel / \u0130nce)", "3 px (Geni\u015f LAP)"]);
        ddChoke.selection = 0;

        var chkAciklama = pnlScreen.add("checkbox", undefined, "%5 Bask\u0131 A\u00e7ma (Kal\u0131p incelmesi ve boya yay\u0131lma telafisi)");
        chkAciklama.value = true;
        var chkDegrade = pnlScreen.add("checkbox", undefined, "%85 Degrade Alt Bindirme (Kom\u015fu ge\u00e7i\u015flerde beyaz fon parlamas\u0131n\u0131 \u00f6nle)");
        chkDegrade.value = true;

        var pnlKros = p4.add("panel", undefined, "+ Kay\u0131t Krosu ve 4 Alanl\u0131 Film Bilgi Etiketi");
        pnlKros.orientation = "column";
        pnlKros.alignChildren = ["fill", "top"];
        pnlKros.spacing = 6;
        pnlKros.margins = 10;

        var chkKros = pnlKros.add("checkbox", undefined, "+ Kay\u0131t Krosu ve 4 Alanl\u0131 Film Bilgi Etiketi Ekle");
        chkKros.value = true;

        var grpLbl1 = pnlKros.add("group");
        grpLbl1.add("statictext", undefined, "Beden:");
        var txtBeden = grpLbl1.add("edittext", undefined, "s-xl");
        txtBeden.characters = 8;
        grpLbl1.add("statictext", undefined, "Y\u00fcz:");
        var txtYuz = grpLbl1.add("edittext", undefined, "on baski");
        txtYuz.characters = 10;
        grpLbl1.add("statictext", undefined, "\u0130\u015f No:");
        var txtIsNo = grpLbl1.add("edittext", undefined, "703");
        txtIsNo.characters = 6;
        grpLbl1.add("statictext", undefined, "\u0130\u015f Ad\u0131:");
        var txtIsAd = grpLbl1.add("edittext", undefined, openInfo ? openInfo.jobName : "desen");
        txtIsAd.characters = 12;

        chkKros.onClick = function () {
            var en = chkKros.value;
            txtBeden.enabled = en;
            txtYuz.enabled = en;
            txtIsNo.enabled = en;
            txtIsAd.enabled = en;
            updateSizePreview();
        };

        // ==================== ADIM 5: Cikti Bicimi & Ozet ====================
        var p5 = stack.add("group");
        p5.orientation = "column";
        p5.alignChildren = ["fill", "top"];
        p5.spacing = 8;

        var pnlOutFmt = p5.add("panel", undefined, "\u00c7\u0131kt\u0131 Bi\u00e7imi");
        pnlOutFmt.orientation = "column";
        pnlOutFmt.alignChildren = ["left", "top"];
        pnlOutFmt.spacing = 6;
        pnlOutFmt.margins = 10;

        var rbOutPsd = pnlOutFmt.add("radiobutton", undefined, "Tek PSD Dosyas\u0131 (3 Katman + Spot Kanallar \u2014 s\u0131f\u0131r klas\u00f6r)");
        var rbOutTif = pnlOutFmt.add("radiobutton", undefined, "Tek TIF Dosyas\u0131 (Spot Kanall\u0131 tek TIFF \u2014 s\u0131f\u0131r klas\u00f6r)");
        var rbOutMem = pnlOutFmt.add("radiobutton", undefined, "Yaln\u0131zca Photoshop'ta A\u00e7\u0131k \u00c7al\u0131\u015fma Sayfas\u0131 (diske ek dosya b\u0131rakma)");
        rbOutPsd.value = true;

        var grpOutDir = pnlOutFmt.add("group");
        grpOutDir.alignChildren = ["fill", "center"];
        grpOutDir.add("statictext", undefined, "Hedef Klas\u00f6r:");
        var defaultOutDir = openInfo ? openInfo.parentDir : Folder.desktop.fsName;
        var txtOutDir = grpOutDir.add("edittext", undefined, defaultOutDir);
        txtOutDir.characters = 34;
        var btnBrowseOut = grpOutDir.add("button", undefined, "De\u011fi\u015ftir...");
        btnBrowseOut.preferredSize = [82, 24];

        function syncOutDirEnabled() {
            var saveDisk = !rbOutMem.value;
            txtOutDir.enabled = saveDisk;
            btnBrowseOut.enabled = saveDisk;
        }
        rbOutPsd.onClick = syncOutDirEnabled;
        rbOutTif.onClick = syncOutDirEnabled;
        rbOutMem.onClick = syncOutDirEnabled;

        btnBrowseOut.onClick = function () {
            var d = Folder.selectDialog("Tek \u00c7al\u0131\u015fma Dosyas\u0131n\u0131n Kaydedilece\u011fi Klas\u00f6r");
            if (d) txtOutDir.text = d.fsName;
        };

        var pnlSum = p5.add("panel", undefined, "Ayar \u00d6zeti");
        pnlSum.orientation = "column";
        pnlSum.alignChildren = ["fill", "fill"];
        pnlSum.margins = 10;
        var txtSummary = pnlSum.add("edittext", undefined, "", { multiline: true, readonly: true });
        txtSummary.preferredSize = [-1, 130];

        // Dosya secim olaylari
        btnBrowseFile.onClick = function () {
            var f = File.openDialog("Emprime Renk Ayr\u0131m\u0131 Yap\u0131lacak G\u00f6rseli Se\u00e7in", "G\u00f6rseller:*.jpg;*.jpeg;*.jfif;*.png;*.tif;*.tiff;*.psd;*.webp;*.*");
            if (f) {
                rbUseFile.value = true;
                txtFilePath.text = f.fsName;
                txtIsAd.text = deriveShortJobName(f.name);
                if (f.parent && f.parent.exists) txtOutDir.text = f.parent.fsName;
                updateSizePreview();
            }
        };

        // Alt Navigasyon Butonlari
        var sepBot = win.add("panel");
        sepBot.preferredSize = [-1, 2];
        var grpNav = win.add("group");
        grpNav.alignment = ["fill", "bottom"];
        var btnCancel = grpNav.add("button", undefined, "\u0130ptal", { name: "cancel" });
        var spacer = grpNav.add("group");
        spacer.alignment = ["fill", "center"];
        var btnBack = grpNav.add("button", undefined, "< Geri");
        var btnNext = grpNav.add("button", undefined, "\u0130leri >");
        var btnStart = grpNav.add("button", undefined, "\u25b6 Ba\u015flat", { name: "ok" });

        var steps = [p1, p2, p3, p4, p5];
        var stepTitles = [
            "Ad\u0131m 1 / 5 \u2014 Kaynak & Ti\u015f\u00f6rt Zemini",
            "Ad\u0131m 2 / 5 \u2014 Kafa Say\u0131s\u0131 & Ayr\u0131m Profili",
            "Ad\u0131m 3 / 5 \u2014 Bask\u0131 \u00d6l\u00e7\u00fcs\u00fc & D\u0131\u015f Zemin Maskesi",
            "Ad\u0131m 4 / 5 \u2014 Tram, A\u00e7\u0131, Bask\u0131 A\u00e7ma & Kros/Etiket",
            "Ad\u0131m 5 / 5 \u2014 \u00c7\u0131kt\u0131 Bi\u00e7imi & \u00d6zet"
        ];
        var curStep = 0;

        function buildSummaryText() {
            var srcStr = rbUseOpen.value ? ("A\u00e7\u0131k Belge (" + (openInfo ? openInfo.name : "-") + ")") : txtFilePath.text;
            var kumasStr = rbKumasIkisi.value ? "\u0130kisi Birlikte (Koyu + A\u00e7\u0131k)" : (rbKumasKoyu.value ? "Yaln\u0131zca Koyu Ti\u015f\u00f6rt" : "Yaln\u0131zca A\u00e7\u0131k Ti\u015f\u00f6rt");
            var profStr = rbProfPro.value ? "Profesyonel (A\u00e7\u0131k LAP + %58 Koyu Tram)" : (rbProfEko.value ? "Ekonomik 6 Kafa" : ("\u00d6zel " + txtCustomHead.text + " Kafa"));
            var siyahStr = rbSiyahCmyk.value ? "CMYK K Kanal\u0131 (FOGRA39)" : "Lab L E\u011frisi";
            var bgStr = rbBgOto.value ? "Otomatik (Metraj/Tekil)" : (rbBgMask.value ? "D\u0131\u015f Zemin Maskeli" : "T\u00fcm Tuval (Maskesiz)");
            var mSize = computeSizeMetrics();
            var sizeStr = (mSize && mSize.boyCm !== null)
                ? (mSize.enCm.toFixed(1) + " x " + mSize.boyCm.toFixed(1) + " cm (" + mSize.targetW + "x" + mSize.targetH + " px @ " + mSize.ppi + " PPI)")
                : (txtWidthCm.text + " cm @ " + txtPPI.text + " PPI");
            var lpiStr = rbLpi45.value ? "45 LPI (Spot Emprime)" : "55 LPI (CMYK Foto\u011fraf)";
            var angStr = rbAng15.value ? "15\u00b0" : (rbAng45.value ? "45\u00b0" : "72\u00b0");
            var fmtStr = rbOutPsd.value ? "Tek PSD (3 Katman + Spot)" : (rbOutTif.value ? "Tek TIF (Spot)" : "Yaln\u0131zca Photoshop'ta A\u00e7\u0131k");
            var outDest = rbOutMem.value ? "(Diske dosya yaz\u0131lmayacak)" : (txtOutDir.text + "\\" + (deriveShortJobName(txtIsAd.text) || "desen") + "_emprime." + (rbOutTif.value ? "tif" : "psd"));

            return "Kaynak: " + srcStr + "\n"
                + "Ti\u015f\u00f6rt Seti: " + kumasStr + "\n"
                + "Profil & Siyah: " + profStr + "  |  " + siyahStr + "\n"
                + "\u00d6l\u00e7\u00fc & Zemin: " + sizeStr + "  |  " + bgStr + "\n"
                + "Tram & Telafi: " + lpiStr + " @ " + angStr + "  |  Choke: " + (ddChoke.selection.index === 1 ? "3 px" : "2 px")
                + "  |  %5 A\u00e7ma: " + (chkAciklama.value ? "A\u00e7\u0131k" : "Kapal\u0131")
                + "  |  %85 Degrade: " + (chkDegrade.value ? "A\u00e7\u0131k" : "Kapal\u0131") + "\n"
                + "Kros & Etiket: " + (chkKros.value ? ("A\u00e7\u0131k (" + txtBeden.text + " | " + txtYuz.text + " | " + txtIsNo.text + " " + txtIsAd.text + ")") : "Kapal\u0131") + "\n"
                + "\u00c7\u0131kt\u0131: " + fmtStr + "\n"
                + "Hedef: " + outDest;
        }

        function showStep(idx) {
            curStep = idx;
            for (var i = 0; i < steps.length; i++) {
                steps[i].visible = (i === idx);
            }
            lblStepTitle.text = stepTitles[idx];
            btnBack.enabled = (idx > 0);
            btnNext.visible = (idx < steps.length - 1);
            btnStart.visible = (idx === steps.length - 1);
            if (idx === 2) {
                updateSizePreview();
            }
            if (idx === steps.length - 1) {
                txtSummary.text = buildSummaryText();
            }
            win.layout.layout(true);
        }

        btnBack.onClick = function () {
            if (curStep > 0) showStep(curStep - 1);
        };

        btnNext.onClick = function () {
            if (curStep === 0 && rbUseFile.value) {
                var fChk = new File(txtFilePath.text);
                if (!txtFilePath.text || !fChk.exists) {
                    alert("L\u00fctfen ge\u00e7erli bir kaynak g\u00f6rsel se\u00e7in veya a\u00e7\u0131k belgeyi kullan\u0131n.");
                    return;
                }
            }
            if (curStep === 1 && rbProfCustom.value) {
                var hNum = parseInt(txtCustomHead.text, 10);
                if (isNaN(hNum) || hNum < 5 || hNum > 10) {
                    alert("\u00d6zel kafa say\u0131s\u0131 5 ile 10 aras\u0131nda olmal\u0131d\u0131r.");
                    return;
                }
            }
            if (curStep === 2) {
                var wChk = parseFloat(String(txtWidthCm.text).replace(",", "."));
                if (isNaN(wChk) || wChk <= 0) {
                    alert("L\u00fctfen ge\u00e7erli bir En (cm) de\u011feri girin.");
                    return;
                }
            }
            if (curStep < steps.length - 1) showStep(curStep + 1);
        };

        btnStart.onClick = function () {
            if (rbUseFile.value) {
                var fChk = new File(txtFilePath.text);
                if (!fChk.exists) {
                    alert("Kaynak g\u00f6rsel bulunamad\u0131!");
                    return;
                }
            }
            win.close(1);
        };
        btnCancel.onClick = function () { win.close(0); };

        showStep(0);
        if (win.show() !== 1) return null;

        var kumas = rbKumasIkisi.value ? "ikisi" : (rbKumasKoyu.value ? "koyu" : "acik");
        var profil = rbProfEko.value ? "ekonomik" : "profesyonel";
        var renkParam = null;
        if (rbProfEko.value) {
            renkParam = (kumas === "acik") ? 6 : 5;
        } else if (rbProfCustom.value) {
            var totalHeads = Math.max(5, Math.min(10, parseInt(txtCustomHead.text, 10) || 8));
            renkParam = (kumas === "acik") ? totalHeads : Math.max(2, totalHeads - 1);
        }

        var zeminMaske = rbBgOto.value ? "oto" : (rbBgMask.value ? "maskeli" : "tumu");
        var lpi = rbLpi55.value ? 55.0 : 45.0;
        var aci = rbAng15.value ? 15.0 : (rbAng45.value ? 45.0 : 72.0);
        var choke = (ddChoke.selection.index === 1) ? 3 : 2;
        var outMode = rbOutPsd.value ? "psd" : (rbOutTif.value ? "tif" : "memory");
        var isAdClean = deriveShortJobName(txtIsAd.text);

        return {
            useOpenDoc: rbUseOpen.value && !!openInfo,
            openDoc: openInfo ? openInfo.doc : null,
            inputFile: rbUseFile.value ? new File(txtFilePath.text) : null,
            kumas: kumas,
            profil: profil,
            renk: renkParam,
            siyahMod: rbSiyahCmyk.value ? "cmyk" : "lab",
            widthCm: parseFloat(String(txtWidthCm.text).replace(",", ".")) || 28.0,
            ppi: parseInt(txtPPI.text, 10) || 508,
            zeminMaske: zeminMaske,
            lpi: lpi,
            aci: aci,
            choke: choke,
            baskiAciklama: chkAciklama.value ? 0.05 : 0.0,
            degradeBindirme: chkDegrade.value ? 0.85 : 0.0,
            krosEtiket: chkKros.value,
            beden: trAscii(txtBeden.text) || "s-xl",
            yuz: trAscii(txtYuz.text) || "on baski",
            isNo: trAscii(txtIsNo.text) || "703",
            isAd: isAdClean,
            outMode: outMode,
            outFormat: (outMode === "tif") ? "tif" : "psd",
            outDir: txtOutDir.text
        };
    }

    function main() {
        var pyExe = findPythonExe();
        var pyScript = findBackendScript();
        if (!pyScript) {
            alert("HATA: tools/lap_aile_ayir.py bulunamad\u0131!");
            return;
        }

        var openInfo = getOpenDocumentInfo();
        var cfg = runWizard(openInfo);
        if (!cfg) return;

        var ts = (new Date()).getTime();
        var tmpDir = new Folder(Folder.temp.fsName + "\\kedihane_emp_" + ts);
        if (!tmpDir.exists) tmpDir.create();

        try {
            var tmpInputFile = null;
            if (cfg.useOpenDoc && cfg.openDoc) {
                tmpInputFile = new File(tmpDir.fsName + "\\kaynak.png");
                if (!exportDocToTempPng(cfg.openDoc, tmpInputFile)) {
                    alert("HATA: A\u00e7\u0131k belge ge\u00e7ici PNG olarak d\u0131\u015fa aktar\u0131lamad\u0131!");
                    return;
                }
            } else if (cfg.inputFile && cfg.inputFile.exists) {
                var extMatch = cfg.inputFile.name.match(/\.[^\.]+$/);
                var srcExt = extMatch ? extMatch[0].toLowerCase() : ".png";
                tmpInputFile = new File(tmpDir.fsName + "\\kaynak" + srcExt);
                if (!cfg.inputFile.copy(tmpInputFile)) {
                    alert("HATA: Kaynak dosya ge\u00e7ici klas\u00f6re kopyalanamad\u0131!");
                    return;
                }
            } else {
                alert("HATA: Ge\u00e7erli bir kaynak g\u00f6rsel bulunamad\u0131!");
                return;
            }

            var tmpOutPath = tmpDir.fsName + "\\cikti." + cfg.outFormat;
            var tmpLogPath = tmpDir.fsName + "\\run.log";

            var cmd = 'cmd.exe /c "chcp 65001 >nul & "' + pyExe + '" "' + pyScript + '" "' + tmpInputFile.fsName + '" "' + tmpDir.fsName + '"'
                + ' --tek-dosya'
                + ' --format ' + cfg.outFormat
                + ' --cikti-dosya "' + tmpOutPath + '"'
                + ' --kumas ' + cfg.kumas
                + ' --profil ' + cfg.profil
                + (cfg.renk !== null ? (' --renk ' + cfg.renk) : '')
                + ' --siyah-mod ' + cfg.siyahMod
                + ' --genislik-cm ' + cfg.widthCm
                + ' --ppi ' + cfg.ppi
                + ' --zemin-maske ' + cfg.zeminMaske
                + ' --lpi ' + cfg.lpi
                + ' --aci ' + cfg.aci
                + ' --choke ' + cfg.choke
                + ' --baski-aciklama ' + cfg.baskiAciklama
                + ' --degrade-bindirme ' + cfg.degradeBindirme
                + (cfg.krosEtiket ? ' --kros-etiket' : ' --kros-yok')
                + ' --beden "' + cfg.beden + '"'
                + ' --yuz "' + cfg.yuz + '"'
                + ' --is-no "' + cfg.isNo + '"'
                + ' --is-ad "' + cfg.isAd + '"'
                + ' > "' + tmpLogPath + '" 2>&1"';

            app.system(cmd);

            var tmpOutFile = new File(tmpOutPath);
            if (!tmpOutFile.exists) {
                var logTxt = "";
                var logFile = new File(tmpLogPath);
                if (logFile.exists) {
                    logFile.open("r");
                    logTxt = logFile.read();
                    logFile.close();
                    if (logTxt.length > 900) logTxt = logTxt.slice(-900);
                }
                alert("HATA: \u00c7al\u0131\u015fma dosyas\u0131 olu\u015fturulamad\u0131!\n\n" + logTxt);
                return;
            }

            if (cfg.outMode === "memory") {
                var openedMem = app.open(tmpOutFile);
                try {
                    openedMem.duplicate(cfg.isAd + "_emprime", false);
                    openedMem.close(SaveOptions.DONOTSAVECHANGES);
                } catch (eDup) {}
            } else {
                var outFolder = new Folder(cfg.outDir);
                if (!outFolder.exists) outFolder.create();
                var targetFile = new File(outFolder.fsName + "\\" + cfg.isAd + "_emprime." + cfg.outFormat);
                if (targetFile.exists) {
                    try { targetFile.remove(); } catch (eRm) {}
                }
                if (tmpOutFile.copy(targetFile) && targetFile.exists) {
                    app.open(targetFile);
                } else {
                    alert("HATA: \u00c7\u0131kt\u0131 hedef konuma kopyalanamad\u0131:\n" + targetFile.fsName);
                }
            }
        } finally {
            deleteFolderRecursive(tmpDir);
        }
    }

    main();
})();
