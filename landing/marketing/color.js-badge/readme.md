# color.js-badge

Color.js entegrasyonunu (PR #722, merged) duyurmak için hazırlanan
**rozet (badge) ve banner tasarımları**. README'lere, blog yazılarına ve
sosyal medya kartlarına eklenecek görsel öğeler.

## İçerik

| Dosya | Ne işe yarar |
|-------|--------------|
| `helmlab-colorjs-badges.html` | Tarayıcıda render edilebilir tüm rozet varyantları (canvas) |
| `badges.jsx` | React/Vite üzerinde rozetleri parametrik üreten bileşen |
| `design-canvas.jsx`, `tweaks-panel.jsx` | Rozet tasarım/önizleme arayüzü (renk, boyut, metin ayarı) |
| `tongue-path.txt` | Color.js logosunun "tongue" path verisi (rozet köşesinde kullanılan) |
| `assets/` | Üretilmiş PNG/SVG rozetler |
| `uploads/` | Sosyal medyada paylaşılan üst sürüm görseller |

## Kullanım

1. `helmlab-colorjs-badges.html`'i tarayıcıda aç → istenen rozeti seç.
2. PNG/SVG export et → `helmlab-main-repo/README.md`'ye veya blog yazısına
   ekle (`![Color.js merged](assets/colorjs-merged.svg)`).
3. Yeni bir variant gerektiğinde `badges.jsx`'i çalıştır
   (`npm run dev`), `tweaks-panel`'den parametreleri ayarla.

## İlgili

- **[Color.js PR #722](https://github.com/color-js/color.js/pull/722)**
  (merged 2026-05) — rozetlerin atıfta bulunduğu entegrasyon.
- **[../helmlab-main-repo/README.md](../helmlab-main-repo/README.md)**
  — rozetlerin yer aldığı production README.
