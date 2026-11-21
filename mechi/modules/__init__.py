"""MeChi Studio arac modulleri."""
from __future__ import annotations

from mechi.core.module_registry import ModuleRegistry, ModuleSpec
from mechi.modules.duzelt_module import DuzeltModule
from mechi.modules.emprime_module import EmprimeModule
from mechi.modules.netlestir_module import NetlestirModule
from mechi.modules.tasarim_module import TasarimModule


def register_builtin_modules(registry: ModuleRegistry) -> None:
    """Varsayilan 4 cekirdek modulu sirayla kaydeder."""
    registry.register(
        ModuleSpec(
            module_id="tasarim",
            title="Görsel Üret",
            short_desc="Marka stili seç, zar at ve benzersiz 2D tasarım promptu oluştur.",
            badge_label="Tasarım",
            order=10,
            icon_char="✦",
            widget_cls=TasarimModule,
        )
    )
    registry.register(
        ModuleSpec(
            module_id="duzelt",
            title="Görsel Düzelt",
            short_desc="Bozuk veya giyilmiş görseli düzenlemeden sıfırdan 2D üretim desenine yeniden yarat.",
            badge_label="Yeniden Yaratım",
            order=15,
            icon_char="❖",
            widget_cls=DuzeltModule,
        )
    )
    registry.register(
        ModuleSpec(
            module_id="netlestir",
            title="Netleştir",
            short_desc="JPEG, TIF veya çok katmanlı PSD dosyalarını pürüzsüzleştir.",
            badge_label="Netleştir",
            order=20,
            icon_char="◈",
            widget_cls=NetlestirModule,
        )
    )
    registry.register(
        ModuleSpec(
            module_id="emprime",
            title="Emprime Ayır",
            short_desc="Serigrafi baskı için spot kanallara ve tramlara ayır.",
            badge_label="Serigrafi",
            order=30,
            icon_char="◫",
            widget_cls=EmprimeModule,
        )
    )
