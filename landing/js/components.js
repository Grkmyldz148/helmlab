/**
 * Helmlab — Shared Site Components
 * Injects consistent nav + footer across all pages.
 * Each page includes <div id="site-nav"></div> and <div id="site-footer"></div>.
 */
(function () {
  'use strict';

  var NAV_LINKS = [
    { href: 'docs.html', text: 'Docs', page: 'docs' },
    { href: 'demo.html', text: 'Demo', page: 'demo' },
    { href: 'tools.html', text: 'Tools', page: 'tools' },
    { href: 'palette.html', text: 'Palette', page: 'palette' },
    { href: 'blog.html', text: 'Blog', page: 'blog' },
    { href: 'https://github.com/Grkmyldz148/helmlab', text: 'GitHub', page: null, external: true }
  ];

  function getCurrentPage() {
    var path = window.location.pathname;
    for (var i = 0; i < NAV_LINKS.length; i++) {
      if (NAV_LINKS[i].page && path.indexOf(NAV_LINKS[i].page) !== -1) {
        return NAV_LINKS[i].page;
      }
    }
    return 'home';
  }

  function buildNav() {
    var currentPage = getCurrentPage();
    var linksHtml = '';
    for (var i = 0; i < NAV_LINKS.length; i++) {
      var link = NAV_LINKS[i];
      var cls = (link.page === currentPage) ? ' class="active"' : '';
      var target = link.external ? ' target="_blank" rel="noopener"' : '';
      linksHtml += '<a href="' + link.href + '"' + cls + target + '>' + link.text + '</a>';
    }
    return '<nav class="site-nav"><div class="site-nav-inner">' +
      '<a href="/" class="site-nav-logo" aria-label="Helmlab Home">' +
        '<img src="assets/logo-horizontal.svg" alt="Helmlab" height="24" width="auto">' +
      '</a>' +
      '<div class="site-nav-links">' + linksHtml + '</div>' +
      '<button type="button" class="site-nav-burger" aria-label="Menu" aria-expanded="false">' +
        '<span></span><span></span><span></span>' +
      '</button>' +
    '</div></nav>';
  }

  function buildFooter() {
    return '<footer class="site-footer"><div class="site-footer-inner">' +
      '<div class="site-footer-left">' +
        '<span>MIT License</span>' +
        '<span class="site-footer-sep">&middot;</span>' +
        '<a href="https://gorkemyildiz.com">Gorkem Yildiz</a>' +
      '</div>' +
      '<div class="site-footer-right">' +
        '<a href="https://arxiv.org/abs/2602.23010">arXiv</a>' +
        '<span class="site-footer-sep">&middot;</span>' +
        '<a href="https://github.com/Grkmyldz148/helmlab">GitHub</a>' +
      '</div>' +
    '</div></footer>';
  }

  function initBurger() {
    var btn = document.querySelector('.site-nav-burger');
    var links = document.querySelector('.site-nav-links');
    if (!btn || !links) return;
    btn.addEventListener('click', function () {
      var isOpen = links.classList.toggle('open');
      btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });
    var anchors = links.querySelectorAll('a');
    for (var i = 0; i < anchors.length; i++) {
      anchors[i].addEventListener('click', function () {
        links.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
      });
    }
  }

  function init() {
    var navEl = document.getElementById('site-nav');
    var footerEl = document.getElementById('site-footer');
    if (navEl) navEl.innerHTML = buildNav();
    if (footerEl) footerEl.innerHTML = buildFooter();
    initBurger();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
