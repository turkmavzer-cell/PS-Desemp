#target photoshop
// master_psd_olustur.jsx — 2 Katli Atolye Master PSD Kurucu (700.psd / 701.psd Standardi)
//
// kanallar_hibrit.json icinden:
//   - Ust Bolum (Calisma Seti): Kumas_Calisma (Solidity %100) + Surekli_Ton/*_ton.png (8-bit, Solidity %16)
//   - Alt Bolum (Baski Seti):   Kumas_Baski (Solidity %100, ayirici) + 1-bit numarali filmler
//     (fon = %100 Solidity, tum ust renkler / lap / ust_beyaz = %16 Solidity)
// Belge RGB olarak korunur (katmanli kaynak korunabilir); 2 GB / 30000 px ustunde PSB kaydeder.
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

function makeSolidColor(rgb) {
  var c = new SolidColor();
  c.rgb.red = rgb[0];
  c.rgb.green = rgb[1];
  c.rgb.blue = rgb[2];
  return c;
}

function addFullFabricChannel(doc, chName, rgb, visible) {
  var ch = doc.channels.add();
  ch.name = chName;
  ch.kind = ChannelType.SPOTCOLOR;
  ch.color = makeSolidColor(rgb);
  ch.opacity = 100;
  doc.activeChannels = [ch];
  doc.selection.selectAll();
  doc.selection.fill(makeSolidColor([0, 0, 0])); // Spot kanalda 0 = %100 dolu
  doc.selection.deselect();
  ch.visible = !!visible;
  return ch;
}

function pasteImageAsSpot(targetDoc, imgFile, chName, rgb, solidity, visible) {
  if (!imgFile.exists) return null;
  var src = app.open(imgFile);
  if (src.mode == DocumentMode.BITMAP) src.changeMode(ChangeMode.GRAYSCALE);
  else if (src.mode != DocumentMode.GRAYSCALE) src.changeMode(ChangeMode.GRAYSCALE);
  src.selection.selectAll();
  src.selection.copy();
  src.close(SaveOptions.DONOTSAVECHANGES);

  app.activeDocument = targetDoc;
  var ch = targetDoc.channels.add();
  ch.name = chName;
  ch.kind = ChannelType.SPOTCOLOR;
  ch.color = makeSolidColor(rgb);
  ch.opacity = solidity;
  targetDoc.activeChannels = [ch];
  targetDoc.paste();
  targetDoc.selection.deselect();
  ch.visible = !!visible;
  return ch;
}

function normalizeItem(it) {
  if (it instanceof Array) {
    var nm = it[0];
    var low = nm.toLowerCase();
    var isFon = (low.indexOf("fon") >= 0 || low.indexOf("underbase") >= 0);
    return {
      ad: nm, dosya: it[1], surekli_ton_dosya: null,
      rgb: it[2], rol: isFon ? "fon" : "tram", solidity: isFon ? 100 : 16
    };
  }
  var rol = it.rol || "tram";
  var defSol = (rol === "fon" || rol === "fon_koyusuz" || rol === "kumas") ? 100 : 16;
  return {
    ad: it.ad || it.dosya,
    dosya: it.dosya,
    surekli_ton_dosya: it.surekli_ton_dosya || null,
    rgb: it.rgb || [0, 0, 0],
    rol: rol,
    solidity: (typeof it.solidity === "number") ? it.solidity : defSol
  };
}

function savePSDorPSB(doc, basePathNoExt) {
  var usePsb = (doc.width.as("px") > 30000 || doc.height.as("px") > 30000);
  if (!usePsb) {
    var out = new File(basePathNoExt + ".psd");
    var o = new PhotoshopSaveOptions();
    o.spotColors = true; o.alphaChannels = true; o.layers = true;
    try {
      doc.saveAs(out, o, true, Extension.LOWERCASE);
      return out;
    } catch (e) {
      usePsb = true;
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

function buildMasterSet(setName, rawList) {
  var list = [];
  for (var i = 0; i < rawList.length; i++) list.push(normalizeItem(rawList[i]));
  if (list.length === 0) return null;

  var ppi = meta.ppi || 508;
  var kumasRgb = meta.kumas_rgb || [30, 30, 30];
  var contDirName = meta.surekli_ton_klasor || "Surekli_Ton";

  // Ilk filmden boyut ve PPI oku
  var probe = app.open(new File(root.fsName + "/" + setName + "/" + list[0].dosya));
  var W = probe.width.as("px");
  var H = probe.height.as("px");
  probe.close(SaveOptions.DONOTSAVECHANGES);

  var doc = app.documents.add(UnitValue(W, "px"), UnitValue(H, "px"), ppi, root.name + "_" + setName + "_Master", NewDocumentMode.RGB, DocumentFill.WHITE);

  // 1) UST BOLUM: 8-bit Surekli Ton Calisma Seti (varsa)
  var hasCont = false;
  for (var i = 0; i < list.length; i++) {
    if (list[i].surekli_ton_dosya) { hasCont = true; break; }
  }
  if (hasCont) {
    addFullFabricChannel(doc, "Kumas_Calisma", kumasRgb, false);
    for (var i = 0; i < list.length; i++) {
      if (list[i].surekli_ton_dosya) {
        var cf = new File(root.fsName + "/" + contDirName + "/" + list[i].surekli_ton_dosya);
        pasteImageAsSpot(doc, cf, list[i].ad + "_ton", list[i].rgb, 16, false);
      }
    }
  }

  // 2) ALT BOLUM: Kumas Ayirici + Nihai 1-bit Baski Kanallari (gorunur acik!)
  addFullFabricChannel(doc, "Kumas_Baski", kumasRgb, true);
  for (var i = 0; i < list.length; i++) {
    var ff = new File(root.fsName + "/" + setName + "/" + list[i].dosya);
    var chName = (i + 1) + " " + list[i].ad;
    pasteImageAsSpot(doc, ff, chName, list[i].rgb, list[i].solidity, true);
  }

  // RGB kanallarini kapatip yalniz baski spot setini gorunur birak
  try {
    doc.channels[0].visible = false;
    doc.channels[1].visible = false;
    doc.channels[2].visible = false;
  } catch (e) {}

  var out = savePSDorPSB(doc, root.fsName + "/" + root.name + "_" + setName + "_Master");
  doc.close(SaveOptions.DONOTSAVECHANGES);
  return out;
}

var saved = [];
for (var s in meta.sets) {
  if (meta.sets.hasOwnProperty(s)) {
    var f = buildMasterSet(s, meta.sets[s]);
    if (f) saved.push(f);
  }
}
for (var k = 0; k < saved.length; k++) app.open(saved[k]);
