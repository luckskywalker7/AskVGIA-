/* askVGIA — mobile nav for the public site pages.
 *
 * Every marketing page ships the same <nav class="nav"> markup, so instead of
 * hand-editing four files this wraps .nav-menu + .nav-right in a drawer and
 * adds the hamburger that opens it. The wrapper carries `display: contents`
 * above 980px (see responsive.css), which takes it back out of the layout
 * entirely — the desktop nav renders exactly as it did before.
 */
/* Wide tables (the API reference on documentation.html) have a min content
 * width that no amount of CSS on the table itself can shrink — the fix is a
 * scrolling wrapper, so add one to every table that does not have one. */
(function () {
  'use strict';
  var tables = document.querySelectorAll('table');
  for (var i = 0; i < tables.length; i++) {
    var t = tables[i];
    if (t.parentElement && t.parentElement.classList.contains('table-scroll')) continue;
    var wrap = document.createElement('div');
    wrap.className = 'table-scroll';
    t.parentNode.insertBefore(wrap, t);
    wrap.appendChild(t);
  }
})();

(function () {
  'use strict';

  var nav = document.querySelector('.nav');
  if (!nav || nav.querySelector('.nav-burger')) return;

  var inner = nav.querySelector('.nav-inner');
  if (!inner) return;

  var menu = inner.querySelector('.nav-menu');
  var right = inner.querySelector('.nav-right');
  if (!menu && !right) return;

  var drawer = document.createElement('div');
  drawer.className = 'nav-drawer';
  drawer.id = 'navDrawer';
  inner.insertBefore(drawer, menu || right);
  if (menu) drawer.appendChild(menu);
  if (right) drawer.appendChild(right);

  var burger = document.createElement('button');
  burger.type = 'button';
  burger.className = 'nav-burger';
  burger.setAttribute('aria-label', 'Menu');
  burger.setAttribute('aria-expanded', 'false');
  burger.setAttribute('aria-controls', 'navDrawer');
  burger.innerHTML = '<span></span><span></span><span></span>';
  inner.appendChild(burger);

  function setOpen(open) {
    nav.classList.toggle('nav-open', open);
    burger.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  burger.addEventListener('click', function (e) {
    e.stopPropagation();
    setOpen(!nav.classList.contains('nav-open'));
  });

  // Following a link (including the in-page #anchors on the landing page)
  // should leave the drawer behind rather than covering the destination.
  drawer.addEventListener('click', function (e) {
    if (e.target.closest('a')) setOpen(false);
  });

  document.addEventListener('click', function (e) {
    if (!nav.contains(e.target)) setOpen(false);
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') setOpen(false);
  });

  // Widening past the breakpoint restores the desktop nav; leaving nav-open
  // set would then be invisible but still armed on the way back down.
  var wide = window.matchMedia('(min-width: 981px)');
  var onChange = function (e) { if (e.matches) setOpen(false); };
  if (wide.addEventListener) wide.addEventListener('change', onChange);
  else if (wide.addListener) wide.addListener(onChange);
})();
