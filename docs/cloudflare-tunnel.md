# Cloudflare Tunnel setup

This stack runs `api` on a private docker network. Exposing it on the public
internet is the Cloudflare Tunnel's job: `cloudflared` opens an outbound
connection to Cloudflare's edge, so no inbound port on your host is required,
and no public IP is leaked. Web frontend + MMseqs-compatible routes ride the
tunnel; the internal `admin` / operator endpoints do not.

The repo ships a `cloudflared` container behind a Docker Compose
[profile](https://docs.docker.com/compose/profiles/) so local development is
unchanged. You opt in with `docker compose --profile tunnel up -d`.

## Prereqs

- A Cloudflare account with access to **Zero Trust** (free tier is enough).
- A domain on that account (e.g. `deepfold.org`) with DNS managed by Cloudflare.
- Docker + Docker Compose v2 on the host.

## 1. Create the tunnel in the Cloudflare dashboard

1. Go to <https://one.dash.cloudflare.com>. Pick the team (or create one) for
   the Cloudflare account that owns `deepfold.org`.
2. **Networks → Tunnels → Create a tunnel**.
3. Connector: **Cloudflared**. Click **Next**.
4. Name: something descriptive, e.g. `plmmsa-deepfold`. Click **Save tunnel**.
5. You'll see an **Install and run a connector** screen. You don't need to
   copy the shell commands — we run cloudflared in our own container. Instead
   find the **token** (long string, starts with `eyJ…`) and copy it.
6. Click **Next** to continue to **Public Hostname**.

## 2. Configure the public hostname

On the **Public Hostname** tab, click **Add a public hostname**:

| Field     | Value                           |
| --------- | ------------------------------- |
| Subdomain | `plmmsa` (or whatever you want) |
| Domain    | `deepfold.org`                  |
| Path      | *leave blank*                   |
| Type      | `HTTP`                          |
| URL       | `api:8080`                      |

`api:8080` is the container-name reference inside our `plmmsa_net` bridge.
`cloudflared` joins the same network (see `docker-compose.yml`), so it
resolves that hostname directly — no need to expose `api` on the host.

Click **Save hostname**. The dashboard creates the corresponding CNAME in
your Cloudflare DNS automatically.

> Do **not** route the `admin` / internal routes through the public hostname.
> If you add a second public hostname later, only point it at routes that are
> safe to expose.

## 3. Paste the token into `.env`

```bash
# .env (copied from .env.example)
CLOUDFLARE_TUNNEL_TOKEN=eyJ...paste-from-step-1-here...
CLOUDFLARE_TUNNEL_HOSTNAME=plmmsa.deepfold.org   # documentation only
```

`.env` is gitignored — make sure you don't commit the token.

## 4. Bring the tunnel up

```bash
docker compose --profile tunnel up -d
```

That starts every normal service plus `cloudflared`. Without the profile the
tunnel container stays offline and the rest of the stack works as before.

### Verify

```bash
docker compose logs -f cloudflared
# expect: "Registered tunnel connection ..." × 4
```

From any machine with internet access:

```bash
curl https://plmmsa.deepfold.org/health
# → {"status":"ok","service":"api"}
```

## 5. Bringing it down

```bash
docker compose --profile tunnel down
```

If you want to remove the tunnel entirely, delete it in the Zero Trust
dashboard afterwards so the DNS record and listener are torn down.

## Rotating / replacing the token

If the token leaks:

1. In the dashboard, open the tunnel → **Overview** → **Refresh token**.
2. Update `CLOUDFLARE_TUNNEL_TOKEN` in `.env`.
3. `docker compose --profile tunnel up -d cloudflared` to restart just the
   tunnel container.

## Troubleshooting

- `ERR_TUNNEL_HTTP_CONNECTION_ERROR`: cloudflared can't reach `api:8080`.
  Verify `api` is healthy (`docker compose ps api`) and on the `plmmsa_net`
  network.
- `526 Invalid SSL certificate`: CF tried to speak HTTPS to `api`. Confirm
  the public hostname's **Type** is `HTTP`, not `HTTPS`.
- `502 Bad Gateway` intermittently: `api` is restarting. Check its logs
  (`docker compose logs api`).
- Token warnings in cloudflared logs after a refresh: remove the stale
  `cloudflared` container (`docker compose rm -sf cloudflared`) and re-up.

## Where not to host the admin UI

The plan keeps the operator admin page on the **internal** network only —
never add a public hostname for it. If you need remote admin access, route
it through your VPN or Cloudflare Access (an Access policy in front of a
separate hostname), not anonymous tunnel traffic.

### ⚠ Known limitation: `/admin/*` is path-reachable through the tunnel

When you route `plmmsa.deepfold.org` → `http://api:8080` the tunnel
forwards *every* path, including `/admin/tokens`. The admin routes are
bearer-token gated (see `plmmsa.api.auth.require_admin_token`), so an
unauthenticated request gets `401 E_AUTH_MISSING`. That's a real defense,
but the route is still exposed on the public hostname.

Close this in the CF Zero Trust dashboard. Edit the tunnel's **Public
Hostnames** page and add a second ingress rule with **Path = `/admin/*`**
pointing at a service that returns a 404 (or use the built-in "HTTP
response" origin type to return 404 for matched paths). Order matters —
the 404 rule must come *before* the `/` catch-all rule that targets
`api:8080`. Verify with:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://plmmsa.deepfold.org/admin/tokens
# expect 404, not 401
```

Until this is configured, treat the bootstrap `ADMIN_TOKEN` as
directly-guessable credential material: rotate it often, and mint
per-client tokens so the bootstrap isn't used for day-to-day traffic
(see docs/maintenance.md).
