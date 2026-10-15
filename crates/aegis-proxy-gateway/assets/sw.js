// AegisCPG approver service worker.
//
// Author: Pratham Kapoor <prathamkapoor027@gmail.com>
// https://github.com/PrathamKapoor
//
// WHAT: Caches the approval shell so the PWA opens offline.
// WHY:  The page itself is static and public; the *state* lives behind
//       /fido2/* and /api/*, which are never cached. Caching only the shell
//       means an offline open can still render, but can never show a stale
//       challenge or a stale verdict -- those must always come from the
//       gateway or not at all.
// HOW:  Precache the three shell files; serve them cache-first; everything
//       else goes straight to the network.

const SHELL = ["/mobile", "/mobile/manifest.webmanifest", "/mobile/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open("aegis-mobile-v1").then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== "aegis-mobile-v1").map((key) => caches.delete(key)))
    )
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || !SHELL.includes(url.pathname)) {
    return; // never cache API, FIDO2, or cross-origin responses
  }
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
