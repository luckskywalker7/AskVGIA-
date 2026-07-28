// ========== ASKVGIA Dashboard JS ==========

// --- Tab switching (Dashboard) ---
const tabs = document.querySelectorAll('.tab');
const tabPanels = document.querySelectorAll('.tab-panel');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    tabPanels.forEach(panel => panel.classList.remove('active'));
    const target = tab.dataset.tab;
    const targetPanel = document.getElementById(target);
    if (targetPanel) targetPanel.classList.add('active');
    // Graph viz is an SVG force layout — it must be (re)drawn once its panel
    // is actually visible, otherwise it measures a 0×0 container.
    if (target === 'graph' && typeof renderGraphVizIfActive === 'function') {
      renderGraphVizIfActive();
    }
  });
});

// --- Suggested question cards click -> fill textarea ---
const questionCards = document.querySelectorAll('.question-card');
const textarea = document.getElementById('ask-textarea');

questionCards.forEach(card => {
  card.addEventListener('click', () => {
    const question = card.dataset.question || card.querySelector('.q-text').innerText.trim();
    if (textarea) {
      textarea.value = question;
      textarea.focus();
    }
  });
});

// Send button and textarea are handled by api.js (uses IDs: ask-send-btn, ask-textarea)

// --- User dropdown toggle ---
const userMenu = document.getElementById('user-menu');
const userDropdown = document.getElementById('user-dropdown');

if (userMenu && userDropdown) {
  userMenu.addEventListener('click', (e) => {
    e.stopPropagation();
    userDropdown.classList.toggle('open');
  });

  document.addEventListener('click', () => {
    userDropdown.classList.remove('open');
  });
}

// --- Dark/Light mode toggle ---
// Shares the `vgia_theme` key with the landing / login / signup pages, so the
// theme you pick out there is the one you land on in here (and vice versa).
// Those pages set html[data-theme]; the dashboard uses body.light-mode — the
// stored value is the same, only the mechanism differs.
const themeToggle = document.getElementById('theme-toggle');
const THEME_KEY = 'vgia_theme';

function applyTheme(theme) {
  const light = theme === 'light';
  document.body.classList.toggle('light-mode', light);
  document.documentElement.dataset.theme = light ? 'light' : 'dark';
  const icon = themeToggle && themeToggle.querySelector('i');
  if (icon) {
    icon.classList.toggle('fa-sun', light);
    icon.classList.toggle('fa-moon', !light);
  }
}

let storedTheme = 'dark';
try { storedTheme = localStorage.getItem(THEME_KEY) || 'dark'; } catch (_) {}
applyTheme(storedTheme);

if (themeToggle) {
  themeToggle.addEventListener('click', () => {
    const next = document.body.classList.contains('light-mode') ? 'dark' : 'light';
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch (_) {}
  });
}

// --- Upload drag & drop ---
const uploadArea = document.getElementById('upload-area');

if (uploadArea) {
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
      uploadArea.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
      uploadArea.classList.remove('dragover');
    });
  });

  uploadArea.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    handleFiles(files);
  });

  uploadArea.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/*';
    input.onchange = (e) => handleFiles(e.target.files);
    input.click();
  });
}

function handleFiles(files) {
  if (!files.length) return;
  const uploadList = document.getElementById('upload-list');
  if (!uploadList) return;

  Array.from(files).forEach(async (file) => {
    const item = document.createElement('div');
    item.className = 'upload-item';
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    item.innerHTML = `
      <div class="upload-thumb"><i class="fa-solid fa-film"></i></div>
      <div class="upload-details">
        <h4>${file.name}</h4>
        <p>${sizeMB} MB</p>
        <div class="upload-progress-bar">
          <div class="upload-progress-fill" style="width: 0%;"></div>
        </div>
      </div>
      <span class="upload-status">0%</span>
    `;
    uploadList.prepend(item);

    const fill = item.querySelector('.upload-progress-fill');
    const status = item.querySelector('.upload-status');

    try {
      const result = await API.uploadFile(file, null, (pct) => {
        const p = Math.round(pct * 90);
        if (fill) fill.style.width = p + '%';
        if (status) status.textContent = p + '%';
      });

      if (fill) fill.style.width = '95%';
      if (status) status.textContent = 'Indexing…';

      const videoId = result.video_id;
      if (videoId) {
        await API.ingest(videoId).catch(() => {});
        if (fill) fill.style.width = '100%';
        if (status) status.textContent = 'Done';
      }
    } catch (e) {
      if (status) status.textContent = 'Error: ' + e.message;
      if (fill) { fill.style.width = '100%'; fill.style.background = '#ef4444'; }
    }
  });
}

// --- Sidebar menu active state on current page ---
const currentPage = window.location.pathname.split('/').pop() || 'index.html';
const sidebarMenuItems = document.querySelectorAll('.sidebar-menu .menu-item');
sidebarMenuItems.forEach(item => {
  const href = item.getAttribute('href');
  if (href === currentPage) {
    sidebarMenuItems.forEach(mi => mi.classList.remove('active'));
    item.classList.add('active');
  }
});

// --- Back to My Videos link ---
const backLink = document.querySelector('.back-link');
if (backLink) {
  backLink.setAttribute('href', 'my-videos.html');
}

// --- Upload New Video button ---
const btnUpload = document.querySelector('.btn-upload');
if (btnUpload) {
  btnUpload.addEventListener('click', () => {
    window.location.href = 'upload.html';
  });
}

// --- Settings page: section navigation ---
const settingsNavItems = document.querySelectorAll('.settings-nav-item');
const settingsSections = document.querySelectorAll('.settings-section');

function showSettingsSection(name) {
  const target = document.getElementById(name);
  if (!target) return false;
  settingsNavItems.forEach(n => n.classList.toggle('active', n.dataset.section === name));
  settingsSections.forEach(s => s.classList.toggle('active', s === target));
  return true;
}

settingsNavItems.forEach(navItem => {
  navItem.addEventListener('click', (e) => {
    e.preventDefault();
    const name = navItem.dataset.section;
    if (showSettingsSection(name)) history.replaceState(null, '', '#' + name);
  });
});

// Deep link: the top-bar user menu points at settings.html#profile
if (settingsSections.length && window.location.hash) {
  showSettingsSection(window.location.hash.slice(1));
}

// --- Settings page: toggle switches (persisted per toggle id) ---
const settingToggles = document.querySelectorAll('.settings-card .toggle-switch');
const TOGGLE_KEY = 'vgia_toggles';

let toggleState = {};
try { toggleState = JSON.parse(localStorage.getItem(TOGGLE_KEY) || '{}'); } catch (_) {}

settingToggles.forEach(toggle => {
  if (toggle.id && toggleState[toggle.id] != null) {
    toggle.classList.toggle('active', !!toggleState[toggle.id]);
  }
  toggle.addEventListener('click', () => {
    toggle.classList.toggle('active');
    if (!toggle.id) return;
    toggleState[toggle.id] = toggle.classList.contains('active');
    try { localStorage.setItem(TOGGLE_KEY, JSON.stringify(toggleState)); } catch (_) {}
  });
});

// --- Sidebar collapse toggle (shared across every app page) ---
// Targets the sidebar by class so it works whether or not the page sets an id.
// The collapsed state is remembered per browser via localStorage.
const sidebarEl = document.querySelector('.sidebar');
const sidebarToggleBtn = document.getElementById('sidebar-toggle');
if (sidebarEl && sidebarToggleBtn) {
  if (localStorage.getItem('sidebarCollapsed') === '1') {
    sidebarEl.classList.add('collapsed');
  }
  sidebarToggleBtn.addEventListener('click', () => {
    const collapsed = sidebarEl.classList.toggle('collapsed');
    localStorage.setItem('sidebarCollapsed', collapsed ? '1' : '0');
  });
}

// --- Mobile sidebar drawer (shared across every app page) ---
// Below 1024px the sidebar is translated off-screen by CSS; this adds the
// hamburger that slides it back in, plus a scrim to tap it away. The button
// and scrim are built here rather than in each page's markup so all five app
// pages get the same behaviour without five copies of the same HTML.
(function setupMobileDrawer() {
  const sidebar = document.querySelector('.sidebar');
  const topbar = document.querySelector('.topbar');
  const titleBlock = topbar && topbar.querySelector('.topbar-title');
  if (!sidebar || !topbar) return;

  const btn = document.createElement('button');
  btn.className = 'nav-mobile-btn';
  btn.type = 'button';
  btn.id = 'nav-mobile-btn';
  btn.setAttribute('aria-label', 'Open menu');
  btn.setAttribute('aria-expanded', 'false');
  btn.setAttribute('aria-controls', sidebar.id || 'sidebar');
  btn.innerHTML = '<i class="fa-solid fa-bars"></i>';
  // Inside .topbar-title, not beside it: .topbar is space-between, so a third
  // direct child would get flung to the middle of the bar.
  if (titleBlock) titleBlock.insertBefore(btn, titleBlock.firstChild);
  else topbar.insertBefore(btn, topbar.firstChild);

  const scrim = document.createElement('div');
  scrim.className = 'sidebar-scrim';
  document.body.appendChild(scrim);

  const setOpen = (open) => {
    sidebar.classList.toggle('open', open);
    scrim.classList.toggle('show', open);
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    btn.querySelector('i').className = open ? 'fa-solid fa-xmark' : 'fa-solid fa-bars';
    // Only lock the page behind the drawer — on desktop the drawer is not a
    // thing and the body must keep scrolling normally.
    document.body.style.overflow = open ? 'hidden' : '';
  };

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    setOpen(!sidebar.classList.contains('open'));
  });
  scrim.addEventListener('click', () => setOpen(false));

  // Tapping a destination should close the drawer, not leave it covering the
  // page it just navigated to (same-page anchors would otherwise strand it).
  sidebar.querySelectorAll('.menu-item').forEach(item => {
    item.addEventListener('click', () => setOpen(false));
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar.classList.contains('open')) setOpen(false);
  });

  // Growing the window past the breakpoint puts the sidebar back in the flow —
  // leaving `open` (and the body scroll lock) set would then be invisible but
  // still active, which is exactly the "resize looks weird" case.
  const wide = window.matchMedia('(min-width: 1025px)');
  const onChange = (e) => { if (e.matches) setOpen(false); };
  if (wide.addEventListener) wide.addEventListener('change', onChange);
  else wide.addListener(onChange);
})();

// --- Ask panel: horizontal resize (index page only) ---
// Drags the grid's second column via the --ask-w custom property, so the
// transcript column reflows on its own. Width is remembered per browser.
const ASK_W_KEY = 'vgia_ask_width';
const ASK_W_DEFAULT = 440;
const ASK_W_MIN = 340;
const ASK_W_MAX = 820;

(function setupAskResize() {
  const panel = document.querySelector('.ask-panel');
  const wrapper = document.querySelector('.content-wrapper');
  if (!panel || !wrapper || wrapper.classList.contains('single-column')) return;

  // Upper bound also keeps the left column usable on smaller screens.
  const maxFor = () => Math.min(ASK_W_MAX, Math.max(ASK_W_MIN, wrapper.clientWidth - 420));
  const applyWidth = (px) => {
    const w = Math.round(Math.min(maxFor(), Math.max(ASK_W_MIN, px)));
    wrapper.style.setProperty('--ask-w', w + 'px');
    return w;
  };

  const saved = parseInt(localStorage.getItem(ASK_W_KEY) || '', 10);
  if (saved) applyWidth(saved);

  const handle = document.createElement('div');
  handle.className = 'ask-resize';
  handle.title = 'Drag to resize · double-click to reset';
  handle.setAttribute('role', 'separator');
  handle.setAttribute('aria-label', 'Resize ask panel');
  panel.appendChild(handle);

  let startX = 0, startW = 0;

  handle.addEventListener('pointerdown', (e) => {
    startX = e.clientX;
    startW = panel.getBoundingClientRect().width;
    handle.classList.add('dragging');
    document.body.classList.add('ask-resizing');
    handle.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  handle.addEventListener('pointermove', (e) => {
    if (!handle.classList.contains('dragging')) return;
    // Handle is on the LEFT edge, so dragging left (negative dx) widens it.
    applyWidth(startW - (e.clientX - startX));
  });

  const endDrag = (e) => {
    if (!handle.classList.contains('dragging')) return;
    handle.classList.remove('dragging');
    document.body.classList.remove('ask-resizing');
    try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
    const w = parseInt(getComputedStyle(wrapper).getPropertyValue('--ask-w'), 10);
    if (w) localStorage.setItem(ASK_W_KEY, String(w));
  };
  handle.addEventListener('pointerup', endDrag);
  handle.addEventListener('pointercancel', endDrag);

  handle.addEventListener('dblclick', () => {
    applyWidth(ASK_W_DEFAULT);
    localStorage.setItem(ASK_W_KEY, String(ASK_W_DEFAULT));
  });

  // Keyboard nudge for accessibility.
  handle.tabIndex = 0;
  handle.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const cur = panel.getBoundingClientRect().width;
    const w = applyWidth(cur + (e.key === 'ArrowLeft' ? 24 : -24));
    localStorage.setItem(ASK_W_KEY, String(w));
  });
})();

// --- Appearance preferences that must apply on every page ---
// Settings writes these; every page reads them on load so the choice sticks
// as you navigate around the app.
const MOTION_KEY = 'vgia_reduced_motion';
window.applyReducedMotion = function (on) {
  document.body.classList.toggle('reduced-motion', !!on);
  try { localStorage.setItem(MOTION_KEY, on ? '1' : '0'); } catch (_) {}
};
try {
  if (localStorage.getItem(MOTION_KEY) === '1') document.body.classList.add('reduced-motion');
} catch (_) {}

// --- Sign-out confirmation (shared across every app page) ---
// Every sidebar "Sign out" link calls this instead of signing out directly.
// Yes -> end the Supabase session and go to the landing page.
// No  -> close and stay exactly where you are.
window.confirmSignOut = function confirmSignOut() {
  if (document.querySelector('.confirm-overlay')) return;   // already open

  const overlay = document.createElement('div');
  overlay.className = 'confirm-overlay';
  overlay.innerHTML =
      '<div class="confirm-card" role="dialog" aria-modal="true" aria-labelledby="confirm-title">'
    +   '<div class="confirm-icon"><i class="fa-solid fa-hand-peace"></i></div>'
    +   '<h3 id="confirm-title">Sign out?</h3>'
    +   '<p>You\'ll need to sign back in to reach your videos and history.</p>'
    +   '<div class="confirm-actions">'
    +     '<button class="confirm-no" type="button">Stay here</button>'
    +     '<button class="confirm-yes" type="button">Yes, sign out</button>'
    +   '</div>'
    + '</div>';
  document.body.appendChild(overlay);

  const close = () => {
    overlay.classList.add('closing');
    overlay.addEventListener('animationend', () => overlay.remove(), { once: true });
    document.removeEventListener('keydown', onKey);
    setTimeout(() => overlay.remove(), 400);   // belt-and-braces
  };
  function onKey(e) {
    if (e.key === 'Escape') close();
  }
  document.addEventListener('keydown', onKey);

  // Clicking the dimmed backdrop is the same as "Stay here".
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  overlay.querySelector('.confirm-no').addEventListener('click', close);

  overlay.querySelector('.confirm-yes').addEventListener('click', async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Signing out…';
    try { await sb.auth.signOut(); } catch (_) {}
    window.location = '/';
  });

  // Focus the safe option so a stray Enter never signs you out.
  setTimeout(() => overlay.querySelector('.confirm-no').focus(), 60);
};

console.log('ASKVGIA Dashboard Loaded');