// ========== ASKVGIA — Profile store (Supabase-backed) ==========
// Single source of truth for "who is using the dashboard".
//
// Data now lives in Supabase: auth.users (email) + public.profiles
// (display_name, username, bio, avatar_url). Every read goes through
// Profile.load() and every write through Profile.save(), so callers only
// had to become `await`-ed. applyToChrome() is unchanged.
//
// Requires supabase-client.js (global `sb`) to be loaded first.

const Profile = (() => {
  const DEFAULTS = {
    display_name: 'Guest',
    email: '',
    username: '',
    bio: '',
    avatar_url: '',
  };

  async function load() {
    const { data: { user } } = await sb.auth.getUser();
    if (!user) return { ...DEFAULTS };
    let row = {};
    try {
      const { data } = await sb
        .from('profiles')
        .select('display_name, username, bio, avatar_url')
        .eq('id', user.id)
        .single();
      row = data || {};
    } catch (_) { row = {}; }
    return { ...DEFAULTS, email: user.email || '', ...row };
  }

  // fields may contain: display_name, username, bio, avatar_url
  async function save(fields) {
    const { data: { user } } = await sb.auth.getUser();
    if (!user) return { ...DEFAULTS };

    const patch = { ...(fields || {}), updated_at: new Date().toISOString() };
    delete patch.email;   // email is managed by auth, not the profiles table

    let row = {};
    try {
      const { data, error } = await sb
        .from('profiles')
        .update(patch)
        .eq('id', user.id)
        .select('display_name, username, bio, avatar_url')
        .single();
      if (error) throw error;
      row = data || {};
    } catch (e) {
      console.error('[VGIA] profile save failed', e);
      throw e;
    }

    const merged = { ...DEFAULTS, email: user.email || '', ...row };
    applyToChrome(merged);
    return merged;
  }

  async function clear() {
    await sb.auth.signOut();
    applyToChrome({ ...DEFAULTS });
  }

  function initial(user) {
    const src = (user.display_name || user.username || '?').trim();
    return (src[0] || '?').toUpperCase();
  }

  // Paint name/avatar into the top bar (and anywhere else that shows them).
  function applyToChrome(user) {
    user = user || { ...DEFAULTS };

    document.querySelectorAll('.user-avatar, #user-avatar-initial, .user-avatar-lg').forEach(el => {
      if (user.avatar_url) {
        el.style.backgroundImage = `url(${user.avatar_url})`;
        el.style.backgroundSize = 'cover';
        el.style.backgroundPosition = 'center';
        el.textContent = '';
      } else {
        el.style.backgroundImage = '';
        el.textContent = initial(user);
      }
    });

    document.querySelectorAll('.user-name, #user-name-label').forEach(el => {
      el.textContent = user.display_name || user.username || 'Guest';
    });
  }

  // Upload an avatar file to Supabase Storage, return its public URL.
  async function uploadAvatar(file) {
    const { data: { user } } = await sb.auth.getUser();
    if (!user) throw new Error('not signed in');
    const ext = (file.name.split('.').pop() || 'png').toLowerCase();
    const path = `${user.id}/avatar_${Date.now()}.${ext}`;
    const { error } = await sb.storage.from('avatars').upload(path, file, { upsert: true });
    if (error) throw error;
    const { data } = sb.storage.from('avatars').getPublicUrl(path);
    return data.publicUrl;
  }

  return { load, save, clear, applyToChrome, initial, uploadAvatar, DEFAULTS };
})();

// Paint the chrome once the profile is fetched (async now).
document.addEventListener('DOMContentLoaded', async () => {
  try { Profile.applyToChrome(await Profile.load()); } catch (_) {}
});
