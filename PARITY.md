# Helmlab Python / JS Parity Report

**Güncel: 2026-07-08 — 1.0.0 working tree, ÖLÇÜLMÜŞ parite** (alt bölümler 0.11.8 tarihi kaydıdır)

## 1.0 Parite Kapısı (kalıcı test)

`packages/helmlab-js/tests/parity-1.0.test.ts` + Python'dan üretilen
`tests/reference/reference-1.0.json` (`scripts/generate-reference.py`).
Tam public yüzeyi kapsar: conversions, gradient/mix/palette/scale/hueRing/
harmonies/rotate/vivid/cusp/maxChroma/adaptive-gamut-map/contrast/adapt,
difference/euclidean/ciede2000/jnd/distance/confidence/nearest/info, tokens
(6 CSS formatı + android/ios/swift + scale exportları), round-trip.

### Ölçülen sonuçlar (2026-07-08, 14/14 test)

| Kategori | En kötü Py↔JS farkı |
|---|---|
| Lab / LCh koordinatları | 8.4e-13 |
| difference / distance / jnd / euclidean | 1.1e-12 |
| confidence alanları | 6.6e-13 |
| contrast ratio / info / nearest | 7.3e-13 |
| **TÜM string çıktılar** (hex, color(), oklch, gradient/mix/vivid/harmonies/scale/palette/tokens) | **birebir eşit (0 fark)** |
| cusp L/C · maxChroma | 3.1e-4 / 5e-4 (iteratif arama, iç tolerans 1e-4 — beklenen) |
| adaptive gamut map | 6.8e-5 (50-iter binary search) |
| near-achromatic hue açısı | ~1e-3° (C≈0'da atan2 gürültü büyütmesi — algısal olarak anlamsız) |

### Round-trip hassasiyeti (dönüşüm kusursuzluğu)

| Test | Sonuç |
|---|---|
| Hex → Lab → Hex, 1728-renk grid, **iki uzay, iki dil** | **0 kayıp (bit-exact)** |
| XYZ → Lab → XYZ (gamut içi 576 renk), MetricSpace | max **2.9e-15** (makine hassasiyeti) |
| XYZ → Lab → XYZ, GenSpace | max **5.8e-9** (enrichment Halley; 8-bit kuantumundan ~6 mertebe küçük), medyan 3.9e-16 |

- 1.0: `hl.metric.distance(labA, labB)` İKİ dilde de MetricLab alır — 0.x
  XYZ/Lab asimetrisi facade'dan kalktı (XYZ-girişli varyant yalnızca Python
  raw MetricSpace sınıfında).
- NC LUT: Python scipy PCHIP ↔ JS elle yazılmış PCHIP — ölçülen fark 1e-13
  sınıfında, parite tam.

---

# 0.11.8 tarihî kaydı (2026-04-12)

## Yapılan Değişiklikler

### JS'ye Eklenen Dosyalar
- `packages/helmlab-js/src/export.ts` — **YENİ DOSYA** (TokenExporter modülü)

### JS'de Değiştirilen Dosyalar
- `packages/helmlab-js/src/helmlab.ts` — yeni metodlar eklendi
- `packages/helmlab-js/src/utils/gamut.ts` — `findCusp()` eklendi
- `packages/helmlab-js/src/index.ts` — yeni exportlar eklendi
- `packages/helmlab-js/tests/helmlab.test.ts` — 41 yeni test eklendi

### Python'da Değiştirilen Dosyalar
- `src/helmlab/helmlab.py` — yeni metodlar eklendi
- `tests/test_helmlab.py` — 8 yeni test eklendi

---

## JS'ye Eklenen Metodlar (helmlab.ts)

| Metod | Açıklama |
|-------|----------|
| `genFromSrgb(rgb)` | sRGB [0,1] → Gen Lab (public) |
| `genToSrgb(lab)` | Gen Lab → sRGB [0,1] (eskiden private idi, public oldu) |
| `baseFromSrgb(rgb)` | deprecated alias → genFromSrgb |
| `baseToSrgb(lab)` | deprecated alias → genToSrgb |
| `toHexP3(lab)` | Lab → `color(display-p3 r g b)` CSS string |
| `adaptToMode(hex, from, to)` | dark/light mode renk adaptasyonu |
| `adaptPair(fg, bg, from, to, minRatio)` | çift renk adaptasyonu + kontrast garanti |
| `export()` | TokenExporter instance döndürür |
| `info()` genişletildi | +`srgb`, `xyz`, `luminance` alanları eklendi |

## JS'ye Eklenen: TokenExporter (export.ts)

| Metod | Çıktı Formatı |
|-------|---------------|
| `toCssHex(lab)` | `#rrggbb` |
| `toCssRgb(lab)` | `rgb(r, g, b)` |
| `toCssOklch(lab)` | `oklch(L% C H)` |
| `toCssDisplayP3(lab)` | `color(display-p3 r g b)` |
| `toCssHsl(lab)` | `hsl(H, S%, L%)` |
| `toAndroidArgb(lab)` | `0xFFrrggbb` |
| `toIosP3(lab)` | `{r, g, b}` (UIColor P3) |
| `toSwiftLiteral(lab)` | `Color(.displayP3, red:, green:, blue:)` |
| `exportScale(scale, name, formats)` | Multi-format scale export |
| `exportCssCustomProperties(scale, prefix)` | CSS custom properties |
| `exportTailwind(scale, name)` | Tailwind config dict |
| `exportJson(scales)` | JSON string |

## JS'ye Eklenen: Gamut Utils (gamut.ts)

| Fonksiyon | Açıklama |
|-----------|----------|
| `findCusp(hRad, space, gamut)` | Verilen hue açısında max chroma noktasını bulur |

## Python'a Eklenen Metodlar (helmlab.py)

| Metod | Açıklama |
|-------|----------|
| `from_XYZ(XYZ)` | CIE XYZ → Helmlab Lab (Helmlab sınıfında doğrudan) |
| `to_XYZ(lab)` | Helmlab Lab → CIE XYZ (Helmlab sınıfında doğrudan) |
| `perceptual_distance(lab1, lab2)` | Full Minkowski + compression mesafe (Lab'dan) |

---

## Test Sonuçları

- **JS**: 237 test geçiyor (5 test dosyası)
- **Python**: 317 test geçiyor + 2 skipped (7 test dosyası)
- **Toplam**: 554 test

## Kalan Deferred (uygulanmadı)

- **Feedback module** — araştırma aracı, kullanıcılar kullanmaz
- **Batch utils** — JS'de `array.map()` ile yapılır
- **`setSurround()`** — pipeline seviyesinde iş gerektirir
- **`gamut_map_batch()`** — JS'de `labs.map(l => gamutMap(l, space))` ile yapılır
