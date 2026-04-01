/**
 * Helmlab Landing — main.js
 * Mode toggle, gradient demo, code tabs, copy buttons, mobile nav
 */

(function () {
  'use strict';

  // ── Color Math Utilities ──────────────────────────────────
  // Inline Oklab / CIE Lab for gradient comparison (helmlab loaded from CDN)

  function srgbToLinear(c) {
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  function linearToSrgb(c) {
    return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  }
  function clamp01(v) { return Math.max(0, Math.min(1, v)); }

  function srgbToXyz(r, g, b) {
    var lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);
    return [
      0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb,
      0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb,
      0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb
    ];
  }
  function xyzToSrgb(X, Y, Z) {
    var r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
    var g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
    var b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
    return [clamp01(linearToSrgb(r)), clamp01(linearToSrgb(g)), clamp01(linearToSrgb(b))];
  }

  // Oklab
  function srgbToOklab(r, g, b) {
    var lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);
    var l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
    var m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
    var s = 0.0883024619 * lr + 0.2220049494 * lg + 0.6756925887 * lb;
    l = Math.cbrt(l); m = Math.cbrt(m); s = Math.cbrt(s);
    return [
      0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
      1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
      0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    ];
  }
  function oklabToSrgb(L, a, b) {
    var l = L + 0.3963377774 * a + 0.2158037573 * b;
    var m = L - 0.1055613458 * a - 0.0638541728 * b;
    var s = L - 0.0894841775 * a - 1.2914855480 * b;
    l = l * l * l; m = m * m * m; s = s * s * s;
    var r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    var g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    var bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
    return [clamp01(linearToSrgb(r)), clamp01(linearToSrgb(g)), clamp01(linearToSrgb(bb))];
  }

  // CIE Lab
  var D65 = [0.95047, 1.0, 1.08883];
  function labF(t) { return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116; }
  function labFInv(t) { return t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787; }
  function srgbToCielab(r, g, b) {
    var xyz = srgbToXyz(r, g, b);
    var fx = labF(xyz[0] / D65[0]), fy = labF(xyz[1] / D65[1]), fz = labF(xyz[2] / D65[2]);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  }
  function cielabToSrgb(L, a, b) {
    var fy = (L + 16) / 116;
    var fx = a / 500 + fy;
    var fz = fy - b / 200;
    return xyzToSrgb(labFInv(fx) * D65[0], labFInv(fy) * D65[1], labFInv(fz) * D65[2]);
  }

  // CIEDE2000 (simplified for uniformity measurement)
  function ciede2000(lab1, lab2) {
    var dL = lab2[0] - lab1[0];
    var da = lab2[1] - lab1[1];
    var db = lab2[2] - lab1[2];
    var C1 = Math.sqrt(lab1[1] * lab1[1] + lab1[2] * lab1[2]);
    var C2 = Math.sqrt(lab2[1] * lab2[1] + lab2[2] * lab2[2]);
    var dC = C2 - C1;
    var dH2 = da * da + db * db - dC * dC;
    var dH = dH2 > 0 ? Math.sqrt(dH2) : 0;
    var avgL = (lab1[0] + lab2[0]) / 2;
    var avgC = (C1 + C2) / 2;
    var SL = 1 + 0.015 * Math.pow(avgL - 50, 2) / Math.sqrt(20 + Math.pow(avgL - 50, 2));
    var SC = 1 + 0.045 * avgC;
    var SH = 1 + 0.015 * avgC;
    return Math.sqrt(
      Math.pow(dL / SL, 2) + Math.pow(dC / SC, 2) + Math.pow(dH / SH, 2)
    );
  }

  // Hex conversions
  function hexToRgb(hex) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
    return [parseInt(hex.slice(0,2),16)/255, parseInt(hex.slice(2,4),16)/255, parseInt(hex.slice(4,6),16)/255];
  }
  function rgbToHex(r, g, b) {
    function h(v) { var s = Math.round(clamp01(v)*255).toString(16); return s.length<2?'0'+s:s; }
    return '#' + h(r) + h(g) + h(b);
  }

  // ── GenSpace v0.11.0 (depressed cubic + enrichment pipeline) ──────
  var GEN_M1 = [
    [0.81543747356487, 0.360322149126427, -0.124327034179467],
    [0.0329839120754665, 0.92929407882555, 0.0361449466529038],
    [0.0481841136683565, 0.26427748135788, 0.633638827111447]
  ];
  var GEN_M2 = [
    [0.211866680137607, 0.79894400408501, -0.00409937558948928],
    [2.46720188280335, -2.98773480248308, 0.520532919679731],
    [-0.113907878680686, 1.39329828081175, -1.27939040213106]
  ];
  var GEN_M1_INV = invertMatrix3(GEN_M1);
  var GEN_M2_INV = invertMatrix3(GEN_M2);

  var G_ALPHA = 0.02;
  var G_ENR_AMP = 0.055, G_ENR_CENTER = 4.616395871525, G_ENR_SIGMA = 0.7, G_ENR_LLO = 0.37, G_ENR_LHI = 1.0;
  var G_PW_IN = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1];
  var G_PW_OUT = [0,0.00949401352218963,0.0256456983803099,0.0552596616586891,0.105749015312274,0.16055853320726,0.214059648929938,0.267862305088112,0.32204352461045,0.373905209852024,0.430209977809188,0.483546516212887,0.539982467041135,0.595671008133034,0.654216166645048,0.711538021651999,0.770276241271167,0.829331346771284,0.889406386197059,0.946282957347473,1];

  function invertMatrix3(m) {
    var a=m[0][0],b=m[0][1],c=m[0][2],d=m[1][0],e=m[1][1],f=m[1][2],g=m[2][0],h=m[2][1],k=m[2][2];
    var det = a*(e*k-f*h) - b*(d*k-f*g) + c*(d*h-e*g);
    var inv = 1/det;
    return [
      [(e*k-f*h)*inv, (c*h-b*k)*inv, (b*f-c*e)*inv],
      [(f*g-d*k)*inv, (a*k-c*g)*inv, (c*d-a*f)*inv],
      [(d*h-e*g)*inv, (b*g-a*h)*inv, (a*e-b*d)*inv]
    ];
  }

  function matMul3(m, v) {
    return [
      m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
      m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
      m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2]
    ];
  }

  // Depressed cubic: y³ + αy = x (sinh/arcsinh + Halley)
  function gDepcubicFwd(x) {
    var s = Math.sqrt(G_ALPHA / 3), t = x / (2 * s*s*s);
    var y = 2 * s * Math.sinh(Math.asinh(t) / 3);
    var f = y*y*y + G_ALPHA*y - x, fp = 3*y*y + G_ALPHA, fpp = 6*y;
    var d = 2*fp*fp - f*fpp;
    if (Math.abs(d) > 1e-30) y -= 2*f*fp/d;
    return y;
  }
  function gDepcubicInv(y) { return y*y*y + G_ALPHA*y; }

  // Smooth neutral blend
  function gSmoothBlend(c0, c1, c2) {
    var mn = (c0+c1+c2)/3;
    var mx = Math.max(c0,c1,c2), mi = Math.min(c0,c1,c2);
    var spread = (mx-mi) / Math.max(Math.abs(mn), 1e-30);
    var w = Math.exp(-(spread/1e-5)*(spread/1e-5));
    return [c0+w*(mn-c0), c1+w*(mn-c1), c2+w*(mn-c2)];
  }

  // PW L correction
  function gPwFwd(L) { if(L<=0||L>=1)return L; var lo=0,hi=20; while(hi-lo>1){var m=(lo+hi)>>1;if(G_PW_IN[m]<=L)lo=m;else hi=m;} var t=(L-G_PW_IN[lo])/(G_PW_IN[hi]-G_PW_IN[lo]); return G_PW_OUT[lo]+t*(G_PW_OUT[hi]-G_PW_OUT[lo]); }
  function gPwInv(L) { if(L<=0||L>=1)return L; var lo=0,hi=20; while(hi-lo>1){var m=(lo+hi)>>1;if(G_PW_OUT[m]<=L)lo=m;else hi=m;} var t=(L-G_PW_OUT[lo])/(G_PW_OUT[hi]-G_PW_OUT[lo]); return G_PW_IN[lo]+t*(G_PW_IN[hi]-G_PW_IN[lo]); }

  // Enrichment forward
  function gEnrFwd(L, a, b) {
    var C = Math.sqrt(a*a+b*b);
    if (C < 1e-12) return [L,a,b];
    var tg = Math.max(0,Math.min(1,(L-G_ENR_LLO)/(G_ENR_LHI-G_ENR_LLO)));
    var gate = Math.pow(Math.sin(Math.PI*tg),2);
    if (gate < 1e-12) return [L,a,b];
    var h = Math.atan2(b,a), dh = h-G_ENR_CENTER;
    dh = dh - Math.round(dh/(2*Math.PI))*2*Math.PI;
    var g = Math.exp(-0.5*dh*dh/(G_ENR_SIGMA*G_ENR_SIGMA));
    var hn = h + G_ENR_AMP*gate*g;
    return [L, C*Math.cos(hn), C*Math.sin(hn)];
  }

  // Enrichment inverse (Halley)
  function gEnrInv(L, a, b) {
    var C = Math.sqrt(a*a+b*b);
    if (C < 1e-12) return [L,a,b];
    var tg = Math.max(0,Math.min(1,(L-G_ENR_LLO)/(G_ENR_LHI-G_ENR_LLO)));
    var gate = Math.pow(Math.sin(Math.PI*tg),2);
    if (gate < 1e-12) return [L,a,b];
    var ht = Math.atan2(b,a), s2 = G_ENR_SIGMA*G_ENR_SIGMA, h = ht, ag = G_ENR_AMP*gate;
    for (var i=0;i<8;i++) {
      var dh = h-G_ENR_CENTER; dh = dh-Math.round(dh/(2*Math.PI))*2*Math.PI;
      var g = Math.exp(-0.5*dh*dh/s2);
      var F = h+ag*g-ht, Fp = 1+ag*g*(-dh/s2), Fpp = ag*g*(-1/s2+dh*dh/(s2*s2));
      var d = 2*Fp*Fp-F*Fpp;
      if (Math.abs(d)>1e-30) h -= 2*F*Fp/d;
    }
    return [L, C*Math.cos(h), C*Math.sin(h)];
  }

  function srgbToGenlab(r, g, b) {
    var lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);
    var xyz = [
      0.4124564*lr + 0.3575761*lg + 0.1804375*lb,
      0.2126729*lr + 0.7151522*lg + 0.0721750*lb,
      0.0193339*lr + 0.1191920*lg + 0.9503041*lb
    ];
    var lms = matMul3(GEN_M1, xyz);
    lms = [Math.max(lms[0],1e-30), Math.max(lms[1],1e-30), Math.max(lms[2],1e-30)];
    var lms_c = [gDepcubicFwd(lms[0]), gDepcubicFwd(lms[1]), gDepcubicFwd(lms[2])];
    // Smooth neutral blend
    var bl = gSmoothBlend(lms_c[0], lms_c[1], lms_c[2]);
    var lab = matMul3(GEN_M2, bl);
    // PW L correction
    lab[0] = gPwFwd(lab[0]);
    // Enrichment
    var enr = gEnrFwd(lab[0], lab[1], lab[2]);
    return enr;
  }

  function genlabToSrgb(L, a, b) {
    // Inverse enrichment
    var enr = gEnrInv(L, a, b); L = enr[0]; a = enr[1]; b = enr[2];
    // Inverse PW
    L = gPwInv(L);
    var lms_c = matMul3(GEN_M2_INV, [L, a, b]);
    // Smooth neutral blend (inverse)
    var bl = gSmoothBlend(lms_c[0], lms_c[1], lms_c[2]);
    var lms = [gDepcubicInv(bl[0]), gDepcubicInv(bl[1]), gDepcubicInv(bl[2])];
    var xyz = matMul3(GEN_M1_INV, lms);
    return xyzToSrgb(xyz[0], xyz[1], xyz[2]);
  }

  // Interpolation
  function lerp(a, b, t) { return a + (b - a) * t; }
  function interpSrgb(rgb1, rgb2, t) {
    return [lerp(rgb1[0],rgb2[0],t), lerp(rgb1[1],rgb2[1],t), lerp(rgb1[2],rgb2[2],t)];
  }
  function interpOklab(rgb1, rgb2, t) {
    var a = srgbToOklab(rgb1[0],rgb1[1],rgb1[2]);
    var b = srgbToOklab(rgb2[0],rgb2[1],rgb2[2]);
    var lab = [lerp(a[0],b[0],t), lerp(a[1],b[1],t), lerp(a[2],b[2],t)];
    return oklabToSrgb(lab[0], lab[1], lab[2]);
  }
  function interpCielab(rgb1, rgb2, t) {
    var a = srgbToCielab(rgb1[0],rgb1[1],rgb1[2]);
    var b = srgbToCielab(rgb2[0],rgb2[1],rgb2[2]);
    var lab = [lerp(a[0],b[0],t), lerp(a[1],b[1],t), lerp(a[2],b[2],t)];
    return cielabToSrgb(lab[0], lab[1], lab[2]);
  }
  // Arc-length reparameterized GenSpace gradient (CIEDE2000 equal steps)
  function buildGenGradientArclen(rgb1, rgb2, steps) {
    var lab1 = srgbToGenlab(rgb1[0],rgb1[1],rgb1[2]);
    var lab2 = srgbToGenlab(rgb2[0],rgb2[1],rgb2[2]);
    var dL=lab2[0]-lab1[0], da=lab2[1]-lab1[1], db=lab2[2]-lab1[2];

    // Fine-sample and build cumulative CIEDE2000 arc length
    var N = 256;
    var cumDist = [0];
    var prevSrgb = genlabToSrgb(lab1[0], lab1[1], lab1[2]);
    var prevCie = srgbToCielab(prevSrgb[0], prevSrgb[1], prevSrgb[2]);
    for (var i = 1; i <= N; i++) {
      var t = i / N;
      var srgb = genlabToSrgb(lab1[0]+dL*t, lab1[1]+da*t, lab1[2]+db*t);
      var cie = srgbToCielab(srgb[0], srgb[1], srgb[2]);
      cumDist.push(cumDist[i-1] + ciede2000(prevCie, cie));
      prevCie = cie;
    }
    var totalDist = cumDist[N];

    // Binary search for equal-distance t values
    var result = [];
    for (var s = 0; s <= steps; s++) {
      var target = (s / steps) * totalDist;
      var lo = 0, hi = N;
      while (lo < hi - 1) {
        var mid = (lo + hi) >> 1;
        if (cumDist[mid] < target) lo = mid; else hi = mid;
      }
      var frac = (cumDist[hi]-cumDist[lo]) > 1e-12
        ? (target-cumDist[lo])/(cumDist[hi]-cumDist[lo]) : 0;
      var tNew = (lo + frac) / N;
      result.push(genlabToSrgb(lab1[0]+dL*tNew, lab1[1]+da*tNew, lab1[2]+db*tNew));
    }
    return result;
  }

  // ── Gradient Demo ─────────────────────────────────────────
  var STEPS = 32;
  var hl = null; // Helmlab instance, set after CDN loads

  function initHelmlab() {
    if (typeof helmlab !== 'undefined' && helmlab.Helmlab) {
      hl = new helmlab.Helmlab();
      return true;
    }
    return false;
  }

  function buildHelmlabGradient(hex1, hex2) {
    if (!hl) return null;
    var hexColors = hl.gradient(hex1, hex2, STEPS + 1);
    return hexColors.map(function(h) { return hexToRgb(h); });
  }

  function computeCV(colors) {
    // Measure step uniformity using CIEDE2000 on CIELab coords
    var steps = [];
    for (var i = 0; i < colors.length - 1; i++) {
      var lab1 = srgbToCielab(colors[i][0], colors[i][1], colors[i][2]);
      var lab2 = srgbToCielab(colors[i+1][0], colors[i+1][1], colors[i+1][2]);
      steps.push(ciede2000(lab1, lab2));
    }
    var mean = 0;
    for (var j = 0; j < steps.length; j++) mean += steps[j];
    mean /= steps.length;
    if (mean < 1e-10) return { cv: 0, steps: steps };
    var variance = 0;
    for (var k = 0; k < steps.length; k++) variance += Math.pow(steps[k] - mean, 2);
    variance /= steps.length;
    return { cv: Math.sqrt(variance) / mean, steps: steps };
  }

  function renderGradient(hex1, hex2) {
    var rgb1 = hexToRgb(hex1);
    var rgb2 = hexToRgb(hex2);

    // Per-step interpolation spaces
    var interpSpaces = [
      { id: 'oklab', fn: interpOklab },
      { id: 'cielab', fn: interpCielab },
      { id: 'srgb', fn: interpSrgb }
    ];

    var results = [];

    // Helmlab: CDN gradient() when available, inline GenSpace arc-length fallback
    var hlColors = buildHelmlabGradient(hex1, hex2);
    if (!hlColors) {
      hlColors = buildGenGradientArclen(rgb1, rgb2, STEPS);
    }
    var hlCv = computeCV(hlColors);
    results.push({ id: 'helmlab', colors: hlColors, cv: hlCv.cv, steps: hlCv.steps });

    for (var s = 0; s < interpSpaces.length; s++) {
      var colors = [];
      for (var i = 0; i <= STEPS; i++) {
        colors.push(interpSpaces[s].fn(rgb1, rgb2, i / STEPS));
      }
      var cvData = computeCV(colors);
      results.push({ id: interpSpaces[s].id, colors: colors, cv: cvData.cv, steps: cvData.steps });
    }

    // Find best CV
    var bestCV = Infinity;
    for (var r = 0; r < results.length; r++) {
      if (results[r].cv < bestCV) bestCV = results[r].cv;
    }

    // Render strips
    for (var ri = 0; ri < results.length; ri++) {
      var res = results[ri];
      var strip = document.getElementById('strip-' + res.id);
      var cvEl = document.getElementById('cv-' + res.id);
      if (!strip || !cvEl) continue;

      var html = '';
      for (var ci = 0; ci < res.colors.length; ci++) {
        var c = res.colors[ci];
        html += '<div class="g-step" style="background:' + rgbToHex(c[0], c[1], c[2]) + '"></div>';
      }
      strip.innerHTML = html;

      var cvPct = (res.cv * 100).toFixed(0);
      cvEl.textContent = 'CV ' + cvPct + '%';
      cvEl.className = 'cv-badge';
      if (res.cv <= bestCV * 1.05) {
        cvEl.classList.add('best');
      } else if (res.cv <= bestCV * 2) {
        cvEl.classList.add('mid');
      } else {
        cvEl.classList.add('bad');
      }
    }
  }

  function syncColorInputs(prefix, hex) {
    var picker = document.getElementById('color-' + prefix);
    var input = document.getElementById('hex-' + prefix);
    if (picker) picker.value = hex;
    if (input) input.value = hex;
  }

  function getCurrentColors() {
    var s = document.getElementById('hex-start');
    var e = document.getElementById('hex-end');
    return { start: s ? s.value : '#ff6b00', end: e ? e.value : '#00d4ff' };
  }

  function updateGradient() {
    var c = getCurrentColors();
    renderGradient(c.start, c.end);
  }

  function initGradientDemo() {
    // Color picker sync
    var startPicker = document.getElementById('color-start');
    var endPicker = document.getElementById('color-end');
    var startHex = document.getElementById('hex-start');
    var endHex = document.getElementById('hex-end');

    if (startPicker) {
      startPicker.addEventListener('input', function() {
        if (startHex) startHex.value = this.value;
        updateGradient();
      });
    }
    if (endPicker) {
      endPicker.addEventListener('input', function() {
        if (endHex) endHex.value = this.value;
        updateGradient();
      });
    }
    if (startHex) {
      startHex.addEventListener('change', function() {
        if (/^#[0-9a-fA-F]{6}$/.test(this.value)) {
          if (startPicker) startPicker.value = this.value;
          updateGradient();
        }
      });
    }
    if (endHex) {
      endHex.addEventListener('change', function() {
        if (/^#[0-9a-fA-F]{6}$/.test(this.value)) {
          if (endPicker) endPicker.value = this.value;
          updateGradient();
        }
      });
    }

    // Presets
    var presets = document.querySelectorAll('.preset-btn');
    for (var i = 0; i < presets.length; i++) {
      presets[i].addEventListener('click', function() {
        var s = this.getAttribute('data-start');
        var e = this.getAttribute('data-end');
        syncColorInputs('start', s);
        syncColorInputs('end', e);
        updateGradient();
      });
    }

    // Initial render (with retry for CDN load)
    if (initHelmlab()) {
      updateGradient();
      initWhyHelmlabDemos();
    } else {
      // Render without helmlab first, retry when loaded
      updateGradient();
      var retryCount = 0;
      var retryInterval = setInterval(function() {
        retryCount++;
        if (initHelmlab()) {
          updateGradient();
          initWhyHelmlabDemos();
          clearInterval(retryInterval);
        } else if (retryCount > 50) {
          clearInterval(retryInterval);
        }
      }, 100);
    }
  }

  function initWhyHelmlabDemos() {
    initHKDemo();
  }

  // ── Helmholtz-Kohlrausch Demo ────────────────────────────────
  function initHKDemo() {
    if (!hl) return;

    var colorHex = '#ff0000';
    var grayHex = '#808080';

    var swGray = document.getElementById('hk-gray');
    var swColor = document.getElementById('hk-color');
    if (!swGray || !swColor) return;
    swGray.style.background = grayHex;
    swColor.style.background = colorHex;

    // CIE Lab L*
    var clGray = srgbToCielab(0.502, 0.502, 0.502);
    var clColor = srgbToCielab(1, 0, 0);

    // Oklab L
    var okGray = srgbToOklab(0.502, 0.502, 0.502);
    var okColor = srgbToOklab(1, 0, 0);

    // Helmlab L
    var hlGray = hl.fromHex(grayHex);
    var hlColor = hl.fromHex(colorHex);

    // CIE Lab bars (L* is 0-100, normalize to 0-1)
    var barClGray = document.getElementById('hk-bar-cl-gray');
    var barClColor = document.getElementById('hk-bar-cl-color');
    var verdictCl = document.getElementById('hk-cl-verdict');
    if (barClGray) barClGray.style.width = (clGray[0] / 100 * 100).toFixed(1) + '%';
    if (barClColor) barClColor.style.width = (clColor[0] / 100 * 100).toFixed(1) + '%';
    if (verdictCl) {
      var clDiff = clColor[0] - clGray[0];
      verdictCl.textContent = 'L*(red)=' + clColor[0].toFixed(1) + ' vs L*(gray)=' + clGray[0].toFixed(1) +
        ' \u2014 ' + (Math.abs(clDiff) < 5 ? 'No H-K' : 'Small diff');
      verdictCl.className = 'hk-verdict wrong';
    }

    // Oklab bars
    var barOkGray = document.getElementById('hk-bar-ok-gray');
    var barOkColor = document.getElementById('hk-bar-ok-color');
    var verdictOk = document.getElementById('hk-ok-verdict');
    if (barOkGray) barOkGray.style.width = (okGray[0] * 100).toFixed(1) + '%';
    if (barOkColor) barOkColor.style.width = (okColor[0] * 100).toFixed(1) + '%';
    if (verdictOk) {
      var okDiff = okColor[0] - okGray[0];
      verdictOk.textContent = 'L(red)=' + okColor[0].toFixed(3) + ' vs L(gray)=' + okGray[0].toFixed(3) +
        ' \u2014 ' + (Math.abs(okDiff) < 0.05 ? 'No H-K' : 'Small diff');
      verdictOk.className = 'hk-verdict wrong';
    }

    // Helmlab bars
    var barHlGray = document.getElementById('hk-bar-hl-gray');
    var barHlColor = document.getElementById('hk-bar-hl-color');
    var verdictHl = document.getElementById('hk-hl-verdict');
    if (barHlGray) barHlGray.style.width = (hlGray[0] * 100).toFixed(1) + '%';
    if (barHlColor) barHlColor.style.width = (hlColor[0] * 100).toFixed(1) + '%';
    if (verdictHl) {
      var hlDiff = hlColor[0] - hlGray[0];
      verdictHl.textContent = 'L(red)=' + hlColor[0].toFixed(3) + ' vs L(gray)=' + hlGray[0].toFixed(3) +
        ' \u2014 ' + (hlDiff > 0.02 ? 'H-K boost: +' + hlDiff.toFixed(3) : 'Similar');
      verdictHl.className = 'hk-verdict ' + (hlDiff > 0.02 ? 'correct' : '');
    }
  }

  // ── Mode Toggle ───────────────────────────────────────────
  function initModeToggle() {
    var buttons = document.querySelectorAll('.mode-btn');
    var saved = localStorage.getItem('helmlab-mode');
    if (saved === 'developer' || saved === 'designer') {
      setMode(saved);
    }

    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener('click', function() {
        var mode = this.getAttribute('data-mode');
        setMode(mode);
        localStorage.setItem('helmlab-mode', mode);
      });
    }
  }

  function setMode(mode) {
    document.body.className = 'mode-' + mode;
    var buttons = document.querySelectorAll('.mode-btn');
    for (var i = 0; i < buttons.length; i++) {
      var isActive = buttons[i].getAttribute('data-mode') === mode;
      buttons[i].classList.toggle('active', isActive);
      buttons[i].setAttribute('aria-checked', isActive ? 'true' : 'false');
    }
  }

  // ── Code Tabs ───────────────────────────────────────────
  function initCodeTabs() {
    var tabButtons = document.querySelectorAll('.code-tab');
    var tabContents = document.querySelectorAll('.code-tab-content');

    for (var i = 0; i < tabButtons.length; i++) {
      tabButtons[i].addEventListener('click', function () {
        var target = this.getAttribute('data-tab');
        for (var j = 0; j < tabButtons.length; j++) {
          tabButtons[j].classList.remove('active');
        }
        this.classList.add('active');
        for (var k = 0; k < tabContents.length; k++) {
          tabContents[k].classList.toggle(
            'active',
            tabContents[k].getAttribute('data-tab-content') === target
          );
        }
      });
    }
  }

  // ── Copy Buttons ────────────────────────────────────────
  function initCopyButtons() {
    var buttons = document.querySelectorAll('.copy-btn');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener('click', function () {
        var text = this.getAttribute('data-copy');
        var btn = this;
        navigator.clipboard.writeText(text).then(function () {
          btn.classList.add('copied');
          setTimeout(function () {
            btn.classList.remove('copied');
          }, 1500);
        });
      });
    }
  }


  // ── Smooth Scroll for Anchor Links ──────────────────────
  function initSmoothScroll() {
    document.addEventListener('click', function (e) {
      var link = e.target.closest('a[href^="#"]');
      if (!link) return;
      var target = document.querySelector(link.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  }

  // ── Scroll Reveal ───────────────────────────────────────
  function initScrollReveal() {
    var sections = document.querySelectorAll('.section, .hero');
    if (!('IntersectionObserver' in window)) return;

    for (var i = 0; i < sections.length; i++) {
      sections[i].style.opacity = '0';
      sections[i].style.transform = 'translateY(24px)';
      sections[i].style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    }

    var observer = new IntersectionObserver(function(entries) {
      for (var j = 0; j < entries.length; j++) {
        if (entries[j].isIntersecting) {
          entries[j].target.style.opacity = '1';
          entries[j].target.style.transform = 'translateY(0)';
          observer.unobserve(entries[j].target);
        }
      }
    }, { threshold: 0.1 });

    for (var k = 0; k < sections.length; k++) {
      observer.observe(sections[k]);
    }
  }

  // ── FAQ Accordion ─────────────────────────────────────────
  function initFaqAccordion() {
    var items = document.querySelectorAll('.faq-item');
    var summaries = document.querySelectorAll('.faq-q');

    for (var i = 0; i < summaries.length; i++) {
      summaries[i].addEventListener('click', function(e) {
        e.preventDefault();
        var parent = this.parentElement;
        var isOpen = parent.open;

        // Close all others first
        for (var j = 0; j < items.length; j++) {
          if (items[j] !== parent) {
            items[j].open = false;
          }
        }

        // Toggle clicked item
        parent.open = !isOpen;
      });
    }
  }

  // ── Init ────────────────────────────────────────────────
  function init() {
    initModeToggle();
    initCodeTabs();
    initCopyButtons();

    initSmoothScroll();
    initScrollReveal();
    initGradientDemo();
    initFaqAccordion();
    initCommunity();
  }

  /* ── Community section ────────────────────────────────── */
  const COMMUNITY_API = 'https://helmlab.space/api/comments';

  function initCommunity() {
    loadTestimonials();
    const form = document.getElementById('communityForm');
    if (!form) return;
    form.addEventListener('submit', handleCommentSubmit);
  }

  async function loadTestimonials() {
    const track = document.getElementById('testimonialTrack');
    const empty = document.getElementById('testimonialEmpty');
    if (!track) return;

    try {
      const res = await fetch(COMMUNITY_API + '?status=approved');
      if (!res.ok) throw new Error('Failed to load');
      const comments = await res.json();

      if (!comments.length) {
        empty.style.display = 'block';
        return;
      }

      track.innerHTML = comments.map(c => `
        <div class="testimonial-card">
          <q>${escapeHtml(c.message)}</q>
          <div class="testimonial-meta">&mdash; ${escapeHtml(c.name)} &middot; ${formatDate(c.created_at)}</div>
        </div>
      `).join('');
    } catch {
      // API not configured yet — show empty state
      empty.style.display = 'block';
    }
  }

  async function handleCommentSubmit(e) {
    e.preventDefault();
    const form = e.target;
    const btn = document.getElementById('communitySubmit');
    const data = Object.fromEntries(new FormData(form));

    btn.disabled = true;
    btn.textContent = 'Sending...';

    try {
      const res = await fetch(COMMUNITY_API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!res.ok) throw new Error('Submit failed');

      form.style.display = 'none';
      document.getElementById('formSuccess').style.display = 'block';
    } catch {
      btn.disabled = false;
      btn.textContent = 'Submit';
      alert('Something went wrong. Please try again.');
    }
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function formatDate(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
