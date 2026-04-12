# Helmlab Python / JS Parity Report

**Tarih:** 2026-04-12
**Versiyon:** 0.11.8 (her iki paket)

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
