// ============================================================
// AskVGIA — dashboard guard (login required)
// Load on every /app page, AFTER supabase-client.js and BEFORE
// profile.js / main.js, e.g.:
//
//   <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
//   <script src="/app-assets/js/supabase-client.js"></script>
//   <script src="/app-assets/js/guard.js"></script>
//   <script src="/app-assets/js/profile.js?v=6"></script>
//   ...
//
// If there is no active session, kick the user to the login page.
// ============================================================
(async () => {
  try {
    const { data: { session } } = await sb.auth.getSession();
    if (!session) {
      window.location.replace('/login.html');
    }
  } catch (_) {
    window.location.replace('/login.html');
  }
})();
