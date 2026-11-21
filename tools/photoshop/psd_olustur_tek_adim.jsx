#target photoshop
// psd_olustur_tek_adim.jsx (v2 — 699-702 Atolye Solidity & Kumas Kanali Standardi)
// Betikle ayni klasordeki kanallar_hibrit.json icindeki HER set (Acik_Tisort, Koyu_Tisort ...) icin
// filmleri tek bir Multichannel belgede spot kanal olarak toplar:
//   - Koyu/renkli sette en basa tam dolu Kumas kanali (Solidity 100%) ekler.
//   - fon / fon_koyusuz: Solidity 100%; tum ust renkler, lap ve ust_beyaz: Solidity 16%.
//   - Hem eski [ad, dosya, rgb] hem yeni {ad, dosya, rgb, rol, solidity} JSON formatini destekler.
app.displayDialogs = DialogModes.NO;
var root = new File($.fileName).parent;
var jf = new File(root.fsName + "/kanallar_hibrit.json");
if (!jf.exists) {
  jf = File.openDialog("kanallar_hibrit.json secin", "*.json");
  if (!jf) { throw new Error("JSON secilmedi."); }
  root = jf.parent;
}
jf.encoding = "UTF-8";
jf.open("r");
var meta = eval("(" + jf.read() + ")");
jf.close();

function normalizeItem(it) {
  if (it instanceof Array) {
    var nm = it[0];
    var low = nm.toLowerCase();
    var isFon = (low.indexOf("fon") >= 0 || low.indexOf("underbase") >= 0 || low.indexOf("beyaz zemin") >= 0);
    return {
      ad: nm,
      dosya: it[1],
      rgb: it[2],
      rol: isFon ? "fon" : "tram",
      solidity: isFon ? 100 : 16
    };
  }
  var rol = it.rol || "tram";
  var defSol = (rol === "fon" || rol === "fon_koyusuz" || rol === "kumas") ? 100 : 16;
  return {
    ad: it.ad || it.dosya,
    dosya: it.dosya,
    rgb: it.rgb || [0, 0, 0],
    rol: rol,
    solidity: (typeof it.solidity === "number") ? it.solidity : defSol
  };
}

function makeSolidColor(rgb) {
  var c = new SolidColor();
  c.rgb.red = rgb[0];
  c.rgb.green = rgb[1];
  c.rgb.blue = rgb[2];
  return c;
}

function openGray(setName, name) {
  var d = app.open(new File(root.fsName + "/" + setName + "/" + name));
  if (d.mode == DocumentMode.BITMAP) d.changeMode(ChangeMode.GRAYSCALE);
  return d;
}

function savePSDorPSB(doc, basePathNoExt) {
  var usePsb = (doc.width.as("px") > 30000 || doc.height.as("px") > 30000);
  if (!usePsb) {
    var out = new File(basePathNoExt + ".psd");
    var o = new PhotoshopSaveOptions();
    o.spotColors = true; o.alphaChannels = true; o.layers = false;
    try {
      doc.saveAs(out, o, true, Extension.LOWERCASE);
      return out;
    } catch (e) {
      usePsb = true; // 2 GB PSD limiti asilirsa PSB'ye dus
    }
  }
  var outPsb = new File(basePathNoExt + ".psb");
  var desc = new ActionDescriptor();
  var fmt = new ActionDescriptor();
  fmt.putBoolean(stringIDToTypeID("maximizeCompatibility"), true);
  desc.putObject(charIDToTypeID("As  "), charIDToTypeID("Pht8"), fmt);
  desc.putPath(charIDToTypeID("In  "), outPsb);
  desc.putBoolean(charIDToTypeID("LwCs"), true);
  executeAction(charIDToTypeID("save"), desc, DialogModes.NO);
  return outPsb;
}

function buildSet(setName, rawList) {
  var list = [];
  for (var i = 0; i < rawList.length; i++) list.push(normalizeItem(rawList[i]));
  if (list.length === 0) return null;

  var isDarkSet = (setName.toLowerCase().indexOf("koyu") >= 0);
  var kumasRgb = meta.kumas_rgb || (isDarkSet ? [30, 30, 30] : [255, 255, 255]);

  var target = openGray(setName, list[0].dosya);
  target.changeMode(ChangeMode.MULTICHANNEL);

  var channelSpecs = [];
  var startIdx = 0;

  if (isDarkSet) {
    // Ilk kanali Kumas kanali yap, list[0]'i yeni kanal olarak ekle
    var kCh = target.channels[0];
    kCh.name = "00_Kumas";
    target.activeChannels = [kCh];
    target.selection.selectAll();
    target.selection.fill(makeSolidColor([0, 0, 0])); // Spot kanalda 0 = %100 dolu
    target.selection.deselect();
    channelSpecs.push({ ad: "00_Kumas", rgb: kumasRgb, solidity: 100 });
    startIdx = 0;
  } else {
    target.channels[0].name = list[0].ad;
    channelSpecs.push(list[0]);
    startIdx = 1;
  }

  for (var i = startIdx; i < list.length; i++) {
    var src = openGray(setName, list[i].dosya);
    src.selection.selectAll();
    src.selection.copy();
    src.close(SaveOptions.DONOTSAVECHANGES);
    app.activeDocument = target;
    var ch = target.channels.add();
    ch.name = list[i].ad;
    target.activeChannels = [ch];
    target.paste();
    target.selection.deselect();
    channelSpecs.push(list[i]);
  }

  var doc = app.activeDocument;
  for (var i = 0; i < doc.channels.length && i < channelSpecs.length; i++) {
    var ch = doc.channels[i];
    var sp = channelSpecs[i];
    try { ch.kind = ChannelType.SPOTCOLOR; } catch (e) {}
    try { ch.color = makeSolidColor(sp.rgb); } catch (e) {}
    try { ch.opacity = sp.solidity; } catch (e) {}
  }
  for (var j = 0; j < doc.channels.length; j++) {
    try { doc.channels[j].visible = true; } catch (e) {}
  }

  var out = savePSDorPSB(doc, root.fsName + "/" + root.name + "_" + setName + "_Kanallar");
  doc.close(SaveOptions.DONOTSAVECHANGES);
  return out;
}

var saved = [];
for (var s in meta.sets) {
  if (meta.sets.hasOwnProperty(s)) {
    var f = buildSet(s, meta.sets[s]);
    if (f) saved.push(f);
  }
}
for (var k = 0; k < saved.length; k++) app.open(saved[k]);
