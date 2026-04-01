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

  // CIEDE2000 (full implementation — matches helmlab package)
  function ciede2000(lab1, lab2) {
    var L1=lab1[0],a1=lab1[1],b1=lab1[2],L2=lab2[0],a2=lab2[1],b2=lab2[2];
    var PI=Math.PI,sqrt=Math.sqrt,pow=Math.pow,abs=Math.abs,sin=Math.sin,cos=Math.cos,atan2=Math.atan2;
    var C1=sqrt(a1*a1+b1*b1),C2=sqrt(a2*a2+b2*b2);
    var Cab=(C1+C2)/2,Cab7=pow(Cab,7),p257=pow(25,7);
    var G=0.5*(1-sqrt(Cab7/(Cab7+p257)));
    var ap1=(1+G)*a1,ap2=(1+G)*a2;
    var Cp1=sqrt(ap1*ap1+b1*b1),Cp2=sqrt(ap2*ap2+b2*b2);
    var hp1=atan2(b1,ap1),hp2=atan2(b2,ap2);
    if(hp1<0)hp1+=2*PI;if(hp2<0)hp2+=2*PI;
    var dLp=L2-L1,dCp=Cp2-Cp1;
    var dhp=0;
    if(Cp1*Cp2!==0){dhp=hp2-hp1;if(dhp>PI)dhp-=2*PI;if(dhp<-PI)dhp+=2*PI;}
    var dHp=2*sqrt(Cp1*Cp2)*sin(dhp/2);
    var Lp=(L1+L2)/2,Cp=(Cp1+Cp2)/2;
    var hp=0;
    if(Cp1*Cp2===0)hp=hp1+hp2;
    else{hp=(hp1+hp2)/2;if(abs(hp1-hp2)>PI){hp+=hp<PI?PI:-PI;}}
    var T=1-0.17*cos(hp-PI/6)+0.24*cos(2*hp)+0.32*cos(3*hp+PI/30)-0.20*cos(4*hp-63*PI/180);
    var Lp50=Lp-50;
    var SL=1+0.015*Lp50*Lp50/sqrt(20+Lp50*Lp50);
    var SC=1+0.045*Cp,SH=1+0.015*Cp*T;
    var Cp7=pow(Cp,7),RC=2*sqrt(Cp7/(Cp7+p257));
    var hpDeg=hp*180/PI;
    var dth=30*Math.exp(-pow((hpDeg-275)/25,2));
    var RT=-sin(2*dth*PI/180)*RC;
    return sqrt(pow(dLp/SL,2)+pow(dCp/SC,2)+pow(dHp/SH,2)+RT*(dCp/SC)*(dHp/SH));
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

  // ── GenSpace v0.11.0 — loaded from CDN (helmlab global) ──────
  var _gs = new helmlab.GenSpace(helmlab.compileGenParams(helmlab.getDefaultGenParams()));
  function srgbToGenlab(r, g, b) {
    var lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b);
    var xyz = [
      0.4124564*lr + 0.3575761*lg + 0.1804375*lb,
      0.2126729*lr + 0.7151522*lg + 0.0721750*lb,
      0.0193339*lr + 0.1191920*lg + 0.9503041*lb
    ];
    return _gs.fromXYZ(xyz);
  }
  function genlabToSrgb(L, a, b) {
    var xyz = _gs.toXYZ([L, a, b]);
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
  console.log('[helmlab] main.js v0.11.0-2 | CDN version:', typeof helmlab !== 'undefined' ? helmlab.VERSION : 'NOT LOADED');

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
