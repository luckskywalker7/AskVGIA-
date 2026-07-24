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

console.log('ASKVGIA Dashboard Loaded');