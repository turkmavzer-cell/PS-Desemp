#target photoshop

/**
 * Kedihane — Renk Katmanı & Emprime Spot Kanal Ayırıcı v1.1
 * 
 * Photoshop CS6, CC 2018 - 2026 Uyumlu Kurşun Geçirmez Sürüm.
 * - Seçim ve katman hatalarını ("Make command not available") önler.
 * - Güvenli Color Range ve Spot Kanal motoru.
 * - Renk katmanlarına ayırma ve emprime serigrafi spot kanalları.
 */

(function () {
    function cID(s) { return charIDToTypeID(s); }
    function sID(s) { return stringIDToTypeID(s); }

    // Seçim olup olmadığını güvenli kontrol et
    function hasActiveSelection(doc) {
        try {
            var b = doc.selection.bounds;
            var w = b[2].as("px") - b[0].as("px");
            var h = b[3].as("px") - b[1].as("px");
            return (w > 0 && h > 0);
        } catch (e) {
            return false;
        }
    }

    // Seçimi piksel cinsinden içeri daralt (Choke)
    function contractSelection(px) {
        if (px <= 0) return;
        try {
            var desc = new ActionDescriptor();
            desc.putUnitDouble(cID("By  "), cID("#Pxl"), px);
            executeAction(cID("Cntc"), desc, DialogModes.NO);
        } catch (e) {}
    }

    // RGB renk aralığını seç (Color Range)
    function selectColorRangeRGB(r, g, b, fuzziness) {
        try {
            var desc = new ActionDescriptor();
            desc.putInteger(sID("fuzziness"), fuzziness);
            var minColor = new ActionDescriptor();
            minColor.putDouble(sID("red"), r);
            minColor.putDouble(sID("green"), g);
            minColor.putDouble(sID("blue"), b);
            desc.putObject(sID("minimum"), sID("RGBColor"), minColor);
            desc.putObject(sID("maximum"), sID("RGBColor"), minColor);
            executeAction(sID("colorRange"), desc, DialogModes.NO);
            return true;
        } catch (e) {
            return false;
        }
    }

    // Güvenli Spot Kanal Oluşturma (Action Manager)
    function createSpotChannelSafe(doc, name, r, g, b, opacity) {
        try {
            var desc = new ActionDescriptor();
            var ref = new ActionReference();
            ref.putClass(cID("Chnl"));
            desc.putReference(cID("null"), ref);
            
            var chnlDesc = new ActionDescriptor();
            chnlDesc.putString(cID("Nm  "), name);
            
            var colorDesc = new ActionDescriptor();
            colorDesc.putDouble(cID("Rd  "), r);
            colorDesc.putDouble(cID("Grn "), g);
            colorDesc.putDouble(cID("Bl  "), b);
            
            chnlDesc.putObject(cID("Clr "), cID("RGBC"), colorDesc);
            chnlDesc.putInteger(cID("Opct"), opacity || 100);
            
            desc.putObject(cID("Usng"), cID("SCnl"), chnlDesc);
            executeAction(cID("Mk  "), desc, DialogModes.NO);
            return doc.activeChannels[0];
        } catch (e) {
            // Alternatif DOM yöntemi
            try {
                var ch = doc.channels.add();
                ch.name = name;
                ch.kind = ChannelType.SPOTCOLOR;
                var sc = new SolidColor();
                sc.rgb.red = r; sc.rgb.green = g; sc.rgb.blue = b;
                ch.color = sc;
                ch.opacity = opacity || 100;
                return ch;
            } catch (err2) {
                return null;
            }
        }
    }

    // Güvenli Katman Kopyalama (Layer Via Copy)
    function safeLayerViaCopy(doc, layerName) {
        if (!hasActiveSelection(doc)) return null;
        try {
            executeAction(sID("copyToLayer"), new ActionDescriptor(), DialogModes.NO);
            var lyr = doc.activeLayer;
            if (layerName) lyr.name = layerName;
            return lyr;
        } catch (e) {
            // Alternatif: Clipboard ile kopyala/yapıştır
            try {
                doc.selection.copy();
                var newLyr = doc.artLayers.add();
                if (layerName) newLyr.name = layerName;
                doc.paste();
                return newLyr;
            } catch (err2) {
                return null;
            }
        }
    }

    function rgbToHex(r, g, b) {
        var rh = Number(Math.round(r)).toString(16);
        var gh = Number(Math.round(g)).toString(16);
        var bh = Number(Math.round(b)).toString(16);
        if (rh.length < 2) rh = "0" + rh;
        if (gh.length < 2) gh = "0" + gh;
        if (bh.length < 2) bh = "0" + bh;
        return ("#" + rh + gh + bh).toUpperCase();
    }

    // ------------------------------------------------------------------------
    // KULLANICI ARAYÜZÜ (ScriptUI Dialog)
    // ------------------------------------------------------------------------

    function showDialog() {
        var hasActiveDoc = (app.documents.length > 0);
        var activeDocName = hasActiveDoc ? app.activeDocument.name : "Açık belge yok";

        var win = new Window("dialog", "Kedihane — Renk Katmanı & Emprime Ayrıcı v1.1");
        win.orientation = "column";
        win.alignChildren = ["fill", "top"];
        win.spacing = 10;
        win.margins = 15;

        // 1. Belge Kaynağı
        var pnlSource = win.add("panel", undefined, "1. Belge Kaynağı");
        pnlSource.orientation = "column";
        pnlSource.alignChildren = ["fill", "top"];
        pnlSource.spacing = 6;

        var rbCurrentDoc = pnlSource.add("radiobutton", undefined, "Açık olan belgeyi kullan: " + activeDocName);
        rbCurrentDoc.value = hasActiveDoc;
        rbCurrentDoc.enabled = hasActiveDoc;

        var rbOpenFile = pnlSource.add("radiobutton", undefined, "Dosyadan seç (JPEG, PNG, PSD, TIFF)");
        rbOpenFile.value = !hasActiveDoc;

        var grpFilePath = pnlSource.add("group");
        grpFilePath.orientation = "row";
        grpFilePath.alignChildren = ["fill", "center"];
        var txtFilePath = grpFilePath.add("edittext", undefined, "");
        txtFilePath.characters = 30;
        txtFilePath.enabled = !hasActiveDoc;
        var btnBrowse = grpFilePath.add("button", undefined, "Gözat...");
        btnBrowse.enabled = !hasActiveDoc;

        var selectedFile = null;
        btnBrowse.onClick = function () {
            var f = File.openDialog("Görsel veya PSD Dosyası Seçin", "Görseller:*.jpg;*.jpeg;*.png;*.psd;*.tif;*.tiff;*.*");
            if (f) {
                selectedFile = f;
                txtFilePath.text = f.fsName;
                rbOpenFile.value = true;
            }
        };

        rbCurrentDoc.onClick = function () {
            txtFilePath.enabled = false;
            btnBrowse.enabled = false;
        };
        rbOpenFile.onClick = function () {
            txtFilePath.enabled = true;
            btnBrowse.enabled = true;
        };

        // 2. Yapılacak İşlem
        var pnlMode = win.add("panel", undefined, "2. Yapılacak İşlem");
        pnlMode.orientation = "column";
        pnlMode.alignChildren = ["fill", "top"];
        pnlMode.spacing = 6;

        var rbLayers = pnlMode.add("radiobutton", undefined, "Yalnızca Renk Katmanlarına Ayır (Şeffaf Layerlar)");
        var rbChannels = pnlMode.add("radiobutton", undefined, "Emprime Spot Renk Kanallarına Ayır (Serigrafi / Film Hazırlığı)");
        var rbBoth = pnlMode.add("radiobutton", undefined, "Hibrit: Hem Katmanları Hem Spot Kanalları Üret");
        rbChannels.value = true;

        // 3. Renk & Ayrım Ayarları
        var pnlParams = win.add("panel", undefined, "3. Renk & Emprime Ayarları");
        pnlParams.orientation = "column";
        pnlParams.alignChildren = ["fill", "top"];
        pnlParams.spacing = 8;

        // Palet Seçimi
        var grpPal = pnlParams.add("group");
        grpPal.orientation = "row";
        grpPal.add("statictext", undefined, "Renk Ayrım Paleti:");
        var ddPalette = grpPal.add("dropdownlist", undefined, [
            "Temel Renkler (Siyah, Beyaz, Kırmızı, Mavi, Sarı, Yeşil)",
            "Genişletilmiş Renkler (8 Renk: +Turuncu, Mor/Magenta)",
            "CMYK Temelli Spot Ayrımı (C, M, Y, K + Beyaz)",
            "Koyu/Açık Gri Tonlama (Monokrom 4 Renk)"
        ]);
        ddPalette.selection = 0;

        // Renk Toleransı
        var grpFuzz = pnlParams.add("group");
        grpFuzz.orientation = "row";
        grpFuzz.add("statictext", undefined, "Renk Toleransı (Fuzziness):");
        var slFuzz = grpFuzz.add("slider", undefined, 50, 10, 120);
        var lblFuzz = grpFuzz.add("statictext", undefined, "50");
        lblFuzz.characters = 4;
        slFuzz.onChanging = function () {
            lblFuzz.text = Math.round(slFuzz.value);
        };

        // Kumaş Tipi & Alt Taban Beyazı
        var grpZemin = pnlParams.add("group");
        grpZemin.orientation = "row";
        grpZemin.add("statictext", undefined, "Kumaş Zemin Tipi:");
        var ddZemin = grpZemin.add("dropdownlist", undefined, ["Koyu Zemin (Siyah / Koyu Kumaş)", "Açık Zemin (Beyaz Kumaş)"]);
        ddZemin.selection = 0;

        var grpUnderbase = pnlParams.add("group");
        grpUnderbase.orientation = "row";
        var chkUnderbase = grpUnderbase.add("checkbox", undefined, "Alt Baskı Beyazı (Underbase) Üret");
        chkUnderbase.value = true;
        grpUnderbase.add("statictext", undefined, "Choke (Daraltma):");
        var ddChoke = grpUnderbase.add("dropdownlist", undefined, ["1 px (İnce Detay)", "2 px (Standart)", "3 px (Geniş LAP)", "4 px", "5 px"]);
        ddChoke.selection = 1; // 2 px (699-701 atolye standardi)

        ddZemin.onChange = function () {
            var isDark = (ddZemin.selection.index === 0);
            chkUnderbase.enabled = isDark;
            ddChoke.enabled = isDark;
        };

        var chkDuplicate = pnlParams.add("checkbox", undefined, "Orijinal dosyayı koru (Kopya üzerinde çalış)");
        chkDuplicate.value = true;

        // Butonlar
        var grpButtons = win.add("group");
        grpButtons.orientation = "row";
        grpButtons.alignment = "right";
        var btnCancel = grpButtons.add("button", undefined, "İptal", { name: "cancel" });
        var btnOK = grpButtons.add("button", undefined, "İşlemi Başlat", { name: "ok" });

        btnOK.onClick = function () {
            if (rbOpenFile.value && (!selectedFile || !selectedFile.exists)) {
                if (txtFilePath.text !== "") {
                    var manualFile = new File(txtFilePath.text);
                    if (manualFile.exists) selectedFile = manualFile;
                }
            }
            if (rbOpenFile.value && (!selectedFile || !selectedFile.exists)) {
                alert("Lütfen geçerli bir görsel dosyası seçin!");
                return;
            }
            win.close(1);
        };

        btnCancel.onClick = function () { win.close(0); };

        var res = win.show();
        if (res !== 1) return null;

        var chokeVals = [1, 2, 3, 4, 5];

        return {
            useCurrentDoc: rbCurrentDoc.value,
            selectedFile: selectedFile,
            mode: rbLayers.value ? "LAYERS" : (rbChannels.value ? "CHANNELS" : "BOTH"),
            paletteIndex: ddPalette.selection.index,
            fuzziness: Math.round(slFuzz.value),
            isDarkFabric: (ddZemin.selection.index === 0),
            createUnderbase: chkUnderbase.value && (ddZemin.selection.index === 0),
            chokePx: chokeVals[ddChoke.selection.index],
            duplicateDoc: chkDuplicate.value
        };
    }

    // ------------------------------------------------------------------------
    // İŞLEM MOTORU
    // ------------------------------------------------------------------------

    function runSeparation(cfg) {
        var originalRuler = app.preferences.rulerUnits;
        app.preferences.rulerUnits = Units.PIXELS;
        app.displayDialogs = DialogModes.NO;

        var workDoc = null;

        try {
            // 1. Belgeyi Hazırla
            if (cfg.useCurrentDoc) {
                if (cfg.duplicateDoc) {
                    var cleanName = app.activeDocument.name.replace(/\.[^\.]+$/, "");
                    workDoc = app.activeDocument.duplicate(cleanName + "_Ayrilmis");
                } else {
                    workDoc = app.activeDocument;
                }
            } else {
                var openedDoc = app.open(cfg.selectedFile);
                if (cfg.duplicateDoc) {
                    var cleanName2 = openedDoc.name.replace(/\.[^\.]+$/, "");
                    workDoc = openedDoc.duplicate(cleanName2 + "_Ayrilmis");
                    openedDoc.close(SaveOptions.DONOTSAVECHANGES);
                } else {
                    workDoc = openedDoc;
                }
            }

            app.activeDocument = workDoc;

            // RGB Moduna Çevir
            if (workDoc.mode !== DocumentMode.RGB) {
                workDoc.changeMode(ChangeMode.RGB);
            }

            // Katmanları birleştirerek taban oluştur
            try { workDoc.flatten(); } catch (e) {}
            var baseLayer = workDoc.activeLayer;
            baseLayer.name = "Orijinal_Zemin";

            // 2. Palet Listesini Belirle
            var targetColors = [];
            if (cfg.paletteIndex === 0) {
                // Temel 6 Renk
                targetColors = [
                    { name: "Siyah", r: 0, g: 0, b: 0 },
                    { name: "Beyaz", r: 255, g: 255, b: 255 },
                    { name: "Kirmizi", r: 220, g: 20, b: 20 },
                    { name: "Mavi", r: 20, g: 70, b: 220 },
                    { name: "Sari", r: 240, g: 220, b: 20 },
                    { name: "Yesil", r: 20, g: 180, b: 40 }
                ];
            } else if (cfg.paletteIndex === 1) {
                // 8 Renk
                targetColors = [
                    { name: "Siyah", r: 0, g: 0, b: 0 },
                    { name: "Beyaz", r: 255, g: 255, b: 255 },
                    { name: "Kirmizi", r: 220, g: 20, b: 20 },
                    { name: "Turuncu", r: 240, g: 130, b: 20 },
                    { name: "Sari", r: 240, g: 220, b: 20 },
                    { name: "Yesil", r: 20, g: 180, b: 40 },
                    { name: "Mavi", r: 20, g: 70, b: 220 },
                    { name: "Magenta_Mor", r: 180, g: 20, b: 180 }
                ];
            } else if (cfg.paletteIndex === 2) {
                // CMYK Temelli Spot
                targetColors = [
                    { name: "Siyah_K", r: 10, g: 10, b: 10 },
                    { name: "Cyan", r: 0, g: 160, b: 230 },
                    { name: "Magenta", r: 230, g: 0, b: 130 },
                    { name: "Yellow", r: 250, g: 230, b: 0 },
                    { name: "Beyaz", r: 255, g: 255, b: 255 }
                ];
            } else {
                // Monokrom 4 Renk
                targetColors = [
                    { name: "Koyu_Siyah", r: 15, g: 15, b: 15 },
                    { name: "Koyu_Gri", r: 80, g: 80, b: 80 },
                    { name: "Acik_Gri", r: 170, g: 170, b: 170 },
                    { name: "Beyaz", r: 250, g: 250, b: 250 }
                ];
            }

            // 3. Kumaş Kanalı (Koyu Zemin) ve Alt Baskı Beyazı (Underbase) Hazırlığı
            if (cfg.isDarkFabric && (cfg.mode === "CHANNELS" || cfg.mode === "BOTH")) {
                var fabricCh = createSpotChannelSafe(workDoc, "00_Kumas_Zemin", 30, 30, 30, 100);
                if (fabricCh) {
                    workDoc.activeChannels = [fabricCh];
                    workDoc.selection.selectAll();
                    var fBlack = new SolidColor();
                    fBlack.rgb.red = 0; fBlack.rgb.green = 0; fBlack.rgb.blue = 0;
                    workDoc.selection.fill(fBlack);
                    workDoc.selection.deselect();
                }
            }
            if (cfg.createUnderbase && (cfg.mode === "CHANNELS" || cfg.mode === "BOTH")) {
                workDoc.activeLayer = baseLayer;
                // Koyu kumaş: Siyah zemin dışındaki tüm grafiği seç
                selectColorRangeRGB(0, 0, 0, 35);
                try {
                    workDoc.selection.invert();
                    if (hasActiveSelection(workDoc)) {
                        contractSelection(cfg.chokePx); // Choke uygula
                        var ubCh = createSpotChannelSafe(workDoc, "01_Alt_Baski_Beyaz (Underbase)", 255, 255, 255, 100);
                        if (ubCh) {
                            workDoc.activeChannels = [ubCh];
                            var blackSolid = new SolidColor();
                            blackSolid.rgb.red = 0; blackSolid.rgb.green = 0; blackSolid.rgb.blue = 0;
                            workDoc.selection.fill(blackSolid);
                        }
                    }
                } catch (e) {}
                workDoc.selection.deselect();
            }

            // 4. Renk Katmanları Grubu (Varsa)
            var layerGroup = null;
            if (cfg.mode === "LAYERS" || cfg.mode === "BOTH") {
                try {
                    layerGroup = workDoc.layerSets.add();
                    layerGroup.name = "Ayrilmis_Renk_Katmanlari";
                } catch (e) {
                    layerGroup = null;
                }
            }

            var processedCount = 0;
            var channelSeq = cfg.createUnderbase ? 2 : 1;

            // 5. Renkleri Tara ve Ayır
            for (var i = 0; i < targetColors.length; i++) {
                var item = targetColors[i];
                var hex = rgbToHex(item.r, item.g, item.b);
                var seqStr = (channelSeq < 10 ? "0" + channelSeq : "" + channelSeq);
                var label = seqStr + "_" + item.name + "_" + hex;

                workDoc.activeLayer = baseLayer;
                selectColorRangeRGB(item.r, item.g, item.b, cfg.fuzziness);

                // Seçim var mı kontrol et
                if (!hasActiveSelection(workDoc)) {
                    continue; // Görselde bu renkten yoksa atla
                }

                // A) Katman Oluştur
                if (cfg.mode === "LAYERS" || cfg.mode === "BOTH") {
                    workDoc.activeLayer = baseLayer;
                    var newLyr = safeLayerViaCopy(workDoc, label);
                    if (newLyr && layerGroup) {
                        try { newLyr.move(layerGroup, ElementPlacement.INSIDE); } catch (e) {}
                    }
                    // Seçimi tekrar al (aktif katmanı Orijinal_Zemin'e geri döndür!)
                    workDoc.activeLayer = baseLayer;
                    selectColorRangeRGB(item.r, item.g, item.b, cfg.fuzziness);
                }

                // B) Spot Kanal Oluştur (Atölye standardı: üst renkler Solidity %16)
                if (cfg.mode === "CHANNELS" || cfg.mode === "BOTH") {
                    var spCh = createSpotChannelSafe(workDoc, label, item.r, item.g, item.b, 16);
                    if (spCh) {
                        workDoc.activeChannels = [spCh];
                        var fillBlack = new SolidColor();
                        fillBlack.rgb.red = 0; fillBlack.rgb.green = 0; fillBlack.rgb.blue = 0;
                        try {
                            workDoc.selection.fill(fillBlack);
                        } catch (e) {}
                    }
                    workDoc.selection.deselect();
                }

                processedCount++;
                channelSeq++;
            }

            // RGB Kanallarını tekrar aktif yap ve seçimi kaldır
            try {
                workDoc.activeChannels = [workDoc.channels[0], workDoc.channels[1], workDoc.channels[2]];
                workDoc.selection.deselect();
            } catch (e) {}

            // Sonuç Bildirimi
            var msg = "Renk Ayrımı Başarıyla Tamamlandı!\n\n";
            msg += "• Ayrılan Renk Sayısı: " + processedCount + "\n";
            msg += "• İşlem Modu: " + cfg.mode + "\n";
            if (cfg.createUnderbase) {
                msg += "• Alt Baskı Beyazı: Eklendi (" + cfg.chokePx + " px choke)\n";
            }
            msg += "\nKanallar veya Katmanlar panelinden ayrılmış renklerinizi kontrol edebilirsiniz.";
            alert(msg, "Kedihane — Başarılı");

        } catch (err) {
            alert("Hata oluştu (Satır " + err.line + "):\n" + err.toString(), "Hata");
        } finally {
            app.preferences.rulerUnits = originalRuler;
        }
    }

    // ------------------------------------------------------------------------
    // BAŞLAT
    // ------------------------------------------------------------------------
    var cfg = showDialog();
    if (cfg) {
        runSeparation(cfg);
    }
})();
