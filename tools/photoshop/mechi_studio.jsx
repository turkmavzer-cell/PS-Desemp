#target photoshop

/**
 * MeChi Studio — Photoshop Bagimsiz Baslatici (Non-Blocking)
 * Photoshop'u dondurmadan masaustu MeChi Studio arayuzunu acar
 * veya aciksa one getirir.
 */
(function () {
    function q(s) {
        return '"' + String(s)
            .replace(/\\/g, "\\\\")
            .replace(/"/g, '\\"')
            .replace(/\r|\n/g, " ") + '"';
    }

    function writeLaunchContext() {
        try {
            var ctxFile = new File(Folder.temp.fsName + "\\mechi_launch_ctx.json");
            ctxFile.encoding = "UTF-8";
            if (!ctxFile.open("w")) return;
            if (app.documents.length > 0) {
                var d = app.activeDocument;
                var prevUnits = app.preferences.rulerUnits;
                app.preferences.rulerUnits = Units.PIXELS;
                var w = 0, h = 0, pathStr = "", lyrCnt = 1;
                try { w = Math.round(d.width.as("px")); } catch (e1) {}
                try { h = Math.round(d.height.as("px")); } catch (e2) {}
                try { pathStr = d.fullName.fsName; } catch (e3) { pathStr = ""; }
                try { lyrCnt = d.layers.length; } catch (e4) { lyrCnt = 1; }
                app.preferences.rulerUnits = prevUnits;
                ctxFile.write(
                    '{"active_doc_name":' + q(d.name) +
                    ',"active_doc_path":' + q(pathStr) +
                    ',"width_px":' + w +
                    ',"height_px":' + h +
                    ',"layer_count":' + lyrCnt + '}'
                );
            } else {
                ctxFile.write('{"active_doc_name":""}');
            }
            ctxFile.close();
        } catch (e) {}
    }

    function findLauncherVbs() {
        var thisDir = (new File($.fileName)).parent;
        var candidates = [
            thisDir.fsName + "\\..\\..\\MeChi_Baslat.vbs",
            thisDir.fsName + "\\MeChi_Baslat.vbs",
            "C:\\Users\\dell\\.gemini\\antigravity\\scratch\\PS-Desemp\\MeChi_Baslat.vbs"
        ];
        for (var i = 0; i < candidates.length; i++) {
            var f = new File(candidates[i]);
            if (f.exists) return f;
        }
        return null;
    }

    writeLaunchContext();
    var vbs = findLauncherVbs();
    if (!vbs) {
        alert("MeChi_Baslat.vbs bulunamadi!\nLutfen PS-Desemp repo yolunu kontrol edin.");
        return;
    }
    vbs.execute();
})();
