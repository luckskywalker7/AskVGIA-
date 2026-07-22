# Auth Wiring Diagnosis — linking login/signup data to the dashboard

**Date:** 2026-07-21
**Question:** how can the information a user types on the landing site's login / signup pages become the same information the dashboard shows?
**Short answer:** the backend for this is already written and working. The break is entirely in the front end — three specific lines. Nothing needs to be designed from scratch.

---

## 1. What already exists (and is complete)

### 1.1 The user store — `core/auth.py`

`AuthDB` (`core/auth.py:47`) is a finished SQLite-backed store, not a stub.

```
users (core/auth.py:71-81)          sessions (core/auth.py:83-89)
  id                                  token       PRIMARY KEY (32-byte hex)
  username      UNIQUE                 user_id     FK -> users.id
  display_name                         created_at
  email         UNIQUE                 expires_at  (SESSION_TTL_DAYS = 30, core/auth.py:16)
  password_hash / password_salt
  bio           DEFAULT ''
  avatar_file
```

Passwords are salted and hashed (`_hash_password`, `core/auth.py:31`), and `login()` hashes even when the email is unknown so response timing does not leak account existence (`core/auth.py:250-259`).

**The field names already match the dashboard 1:1.** `public_user()` (`core/auth.py:287`) — the exact shape `/auth/me` returns — is:

| `public_user()` field | dashboard `Profile` field | Settings input |
|---|---|---|
| `display_name` | `display_name` | `#profile-display-name` |
| `email` | `email` | `#profile-email` |
| `username` | `username` | `#profile-username` |
| `bio` | `bio` | `#profile-bio` |
| `avatar_url` | `avatar_url` | `#profile-avatar-preview` |

No mapping layer is needed. The names line up because both were built from the same model.

> Note: `_ensure_default_user()` (`core/auth.py:97-105`) seeds a user `hanane` / `hanane@askvgia.local` / display name **"Hanane"**. That is where the name hard-coded in the dashboard top bar came from.

### 1.2 The endpoints — `api/main.py`

All live, all registered:

| Endpoint | Line | Returns |
|---|---|---|
| `POST /auth/register` | `1759` | `{ token, user }` — 409 on duplicate email/username, 400 if password < 6 chars |
| `POST /auth/login` | `1748` | `{ token, user }` |
| `POST /auth/logout` | `1782` | deletes the session row |
| `GET /auth/me` | `1790` | `public_user()` — the current user |
| `PUT /auth/me` | `1796` | updates `display_name` / `email` / `username` / `bio` |
| `PUT /auth/me/password` | `1817` | verifies current password, then replaces it |
| `POST /auth/me/avatar` | `1828` | stores the file, returns `{ avatar_url }` |
| `GET /avatars/{filename}` | `1841` | serves the stored image |
| `GET /auth/stats` | `1855` | user / active-session counts |

Auth is a `Authorization: Bearer <token>` header, read at `api/main.py:1706-1708`. CORS is already `allow_origins=["*"]` (`api/main.py:120`), so a front end served from another origin would work.

### 1.3 The client that used to call all of this

`static/app_backup_20260720_173806/js/auth.js` (202 lines) is a complete client for every endpoint above — `login`, `register`, `updateProfile`, `changePassword`, `uploadAvatar`, plus a `updateUI(user)` that paints the name/avatar into the top bar. It was removed from `static/app/` when login/signup were dropped from the dashboard. **It does not need to be rewritten, only re-adopted.**

---

## 2. Why nothing is connected today — three exact breaks

### Break 1 — the forms submit nowhere

```html
<!-- project/login.html:579  AND  project/signup.html:348 — identical -->
<form onsubmit="event.preventDefault(); window.location='VGIA.html'">
```

That is the entire handler. No `fetch`, no validation against the server, no token. Whatever the user types is discarded on submit.

### Break 2 — the inputs have no handles

The login and signup inputs carry `data-i18n-ph` (translation) and `placeholder`, but **no `id` and no `name`** — except `#pwd` for the password-strength meter. There is currently no way to read the typed values from JS.

```html
<!-- project/signup.html, inside the form -->
<input type="text"  data-i18n-ph="ph_first"    placeholder="Sara"    required>   <!-- no id -->
<input type="text"  data-i18n-ph="ph_last"     placeholder="Hamidi"  required>   <!-- no id -->
<input type="email" data-i18n-ph="ph_email"    placeholder="vous@email.dz" required> <!-- no id -->
<input type="password" id="pwd" ...>                                             <!-- only this one -->
```

Signup collects **first name + last name**; the backend wants a single `display_name` and a `username`. That is a join and a derivation, decided in step 3.1 below — not a blocker.

### Break 3 — "log in" lands on the marketing page, not the app

Both forms redirect to `VGIA.html`, which is the landing page itself (`/` serves `project/VGIA.html`, `api/main.py:1934`). So a successful "login" bounces the user back where they started.

The only real route into the dashboard is the landing's two `href="/app"` buttons ("Try VGIA"), which bypass login entirely. `/app` serves `static/app/index.html` (`api/main.py:1954`) with no session check.

**Net effect:** the auth pages and the working dashboard are two disconnected islands, with a finished backend sitting unused between them.

---

## 3. The fix

### 3.1 Decisions to make first

| Decision | Options | Recommendation |
|---|---|---|
| `display_name` from signup | join `first + " " + last` | join — matches the seeded "Hanane" style |
| `username` from signup | derive (`sara.hamidi`, de-duplicated) or add a field | derive; the backend enforces UNIQUE and returns 409, so retry with a suffix |
| Is `/app` gated? | redirect to `/login.html` when no session, **or** stay open | *open for now* — the dashboard was deliberately made usable without accounts |

That last row is the only real product decision. Everything else is mechanical.

### 3.2 Step 1 — make the forms talk to the backend

Add `id`s to the four signup inputs and two login inputs, then replace the two `onsubmit` attributes:

```js
// project/signup.html — replaces the onsubmit at line 348
async function handleSignup(e) {
  e.preventDefault();
  const display_name = `${first.value.trim()} ${last.value.trim()}`.trim();
  const r = await fetch('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      username: deriveUsername(first.value, last.value),
      display_name,
      email: email.value.trim(),
      password: pwd.value,
    }),
  });
  if (!r.ok) return showError((await r.json()).detail);   // 409 dup, 400 weak password
  const { token, user } = await r.json();
  localStorage.setItem('askvgia_session', JSON.stringify({ token, user }));
  window.location = '/app';                                // <- not VGIA.html
}
```

`project/login.html` is the same shape against `POST /auth/login` with `{ email, password }`.

Both pages should keep writing the session under **one** key. `auth.js` used `askvgia_session`; reuse it.

### 3.3 Step 2 — point the dashboard at the server

This is the part that was designed in advance. `static/app/js/profile.js` is the **only** module in the dashboard that knows where user data lives:

```js
Profile.load()          // -> { display_name, email, username, bio, avatar_url }
Profile.save(fields)    // persists, then repaints the top bar
Profile.applyToChrome() // paints name + avatar into .user-name / .user-avatar
```

Every caller — the settings form, the top-bar chip on all five pages, the avatar preview — goes through those. Today they read and write `localStorage['vgia_profile']`. Switching to accounts means rewriting **only these two function bodies**:

```js
async function load() {
  const r = await fetch('/auth/me', { headers: authHeaders() });
  return r.ok ? await r.json() : DEFAULTS;
}

async function save(fields) {
  const r = await fetch('/auth/me', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(fields),
  });
  const user = await r.json();
  applyToChrome(user);
  return user;
}
```

The response shape of `GET /auth/me` is already exactly what `Profile.load()` returns today, so nothing downstream changes. The two callers become `await`-ed — that is the only ripple.

Two smaller follow-ups once that lands:
- the settings **avatar** switches from a `data:` URI in localStorage to `POST /auth/me/avatar`, which returns a real `/avatars/{file}` URL;
- the settings **password** card stops being inert and calls `PUT /auth/me/password` — `#password-save` currently just explains why it can't.

### 3.4 Step 3 — sign out

The sidebar "Sign out" is currently a plain `<a href="/">` back to the landing. With accounts on it becomes `POST /auth/logout` → clear the stored session → redirect. `auth.js:96-103` already does exactly this.

---

## 4. Effort

| Task | Files | Size |
|---|---|---|
| ids + submit handlers on the auth pages | `project/login.html`, `project/signup.html` | ~60 lines |
| session read/write helper (or re-adopt `auth.js`) | `static/app/js/` | existing file |
| `Profile.load` / `Profile.save` → `/auth/me` | `static/app/js/profile.js` | ~2 functions |
| avatar + password wiring | `static/app/settings.html` | ~30 lines |
| sign out → `/auth/logout` | 5 dashboard pages | 1 line each |

No backend work. No schema change. No new endpoints.

---

## 5. Current state, for reference

Until the above is done, the dashboard is deliberately account-free:

- profile data persists in `localStorage['vgia_profile']` via `profile.js`;
- theme is shared with the landing through `localStorage['vgia_theme']` (landing/login use `html[data-theme]`, the dashboard uses `body.light-mode` — same stored value, different mechanism);
- "Sign out" is a link back to `/`;
- `/auth/*` and `core/auth.py` remain live and reachable but nothing in `static/app/` calls them.
