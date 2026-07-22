// ============================================================
// AskVGIA — shared Supabase client + auth helpers
// Load AFTER the supabase-js CDN <script> on every page that
// touches auth (login, signup, reset, and all /app pages).
//
//   <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
//   <script src="/app-assets/js/supabase-client.js"></script>
// ============================================================

// ---- AskVGIA Supabase project ----
const SUPABASE_URL  = 'https://zievhmocgbrrsltnyfkf.supabase.co';
const SUPABASE_ANON = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InppZXZobW9jZ2JycnNsdG55ZmtmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODQ2MTk0NDQsImV4cCI6MjEwMDE5NTQ0NH0.jEYUVVV59A8gholsKRVkG0GJdtZ0qeWmdaxnR-QE5mg';   // anon public key — safe to expose
// -----------------------------------

// The CDN exposes a global `supabase` with createClient.
const sb = supabase.createClient(SUPABASE_URL, SUPABASE_ANON, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,   // needed for email-confirm + password-reset callbacks
  },
});

// Convenience: current access token (for calling your FastAPI backend later).
async function sbToken() {
  const { data: { session } } = await sb.auth.getSession();
  return session?.access_token || null;
}

// Convenience: Authorization header for fetch() calls to your own /ask, /videos, etc.
async function sbAuthHeaders() {
  const t = await sbToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}
