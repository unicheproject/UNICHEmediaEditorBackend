# Production deployment

Merges to `main` run tests, publish a Docker/OCI image to GitHub Container
Registry (GHCR), and deploy that exact image digest to the Ubuntu VM. Pull
requests run the same tests but cannot publish, deploy, or access production
secrets.

## What is stored where

| Value | Location | Public? |
|---|---|---|
| Application image and source | GHCR / GitHub | Yes |
| VM host/IP, SSH user, deploy key, SSH host key | GitHub `production` environment secrets | No |
| Database and provider credentials, runtime URLs | `/opt/uniche-media-editor/.env` on the VM | No |
| Production Compose and deployed image digest | Repository and VM `.image.env` | Yes / non-sensitive |

The workflow never copies the VM's `.env` back to GitHub and never places
runtime secrets inside an image. `.dockerignore` also excludes local override,
environment, PEM, and key files from the Docker build context.

## 1. Prepare the Hetzner Ubuntu VM

Install Docker Engine with the Compose plugin from Docker's official Ubuntu
instructions, plus `curl`. Create a dedicated deployment user with an SSH key.
The user needs Docker access; membership in the `docker` group is effectively
root access, so do not reuse a personal account.

As root on the VM:

```bash
install -d -m 0750 -o deploy -g deploy /opt/uniche-media-editor
usermod -aG docker deploy
```

Log out and back in after changing group membership. Configure the Hetzner
firewall and SSH daemon separately. The SSH workflow uses port 22. GitHub-hosted
runners do not have a small stable egress range, so restricting port 22 to a
single GitHub IP is not practical with this design.

Create the runtime file on the VM—do not create or commit it in this repository:

```bash
sudo -u deploy cp /path/to/env.production.example /opt/uniche-media-editor/.env
sudo -u deploy chmod 600 /opt/uniche-media-editor/.env
sudo -u deploy editor /opt/uniche-media-editor/.env
```

Use [deploy/env.production.example](deploy/env.production.example) as the
template. Generate independent, long random passwords. If the PostgreSQL
password has reserved URL characters, URL-encode it in `DATABASE_URL`.

The production stack talks to the remote catalogue at
`https://catalogue.uniche-eccch.eu`. It does not join the development-only
external `uniche` Docker network.

## 2. Create the deployment SSH key

On a trusted administrator machine, generate a key dedicated to this repository:

```bash
ssh-keygen -t ed25519 -a 100 -f uniche_media_editor_deploy -C github-actions-deploy
```

Install the `.pub` key in `/home/deploy/.ssh/authorized_keys` on the VM. Store
the private key as the GitHub secret described below. Do not reuse an operator's
personal SSH key.

Obtain the server's host public key through the Hetzner console or another
trusted channel. Build the known-hosts entry using the same host/IP stored in
`DEPLOY_HOST`:

```text
HOST_OR_IP ssh-ed25519 AAAAC3...
```

Do not establish initial trust using an unauthenticated `ssh-keyscan` result.
The workflow uses strict host-key checking to prevent a man-in-the-middle from
receiving deployment access.

## 3. Configure GitHub

In **Settings → Environments**, create an environment named `production` and
restrict its deployment branch to `main`. A required reviewer is recommended
if deployments should require a human approval; omit it for fully automatic
deployment after every merge.

Add these environment secrets:

| Secret | Value |
|---|---|
| `DEPLOY_HOST` | VM IP or SSH hostname |
| `DEPLOY_USER` | `deploy` |
| `DEPLOY_SSH_PRIVATE_KEY` | Complete dedicated private key, including header/footer |
| `DEPLOY_SSH_HOST_KEY` | Complete known-hosts line from the trusted step above |

Do not store application `.env` contents as GitHub secrets. They are needed
only by containers on the VM and remain there.

Protect `main` with a ruleset or branch protection rule that:

- requires pull requests;
- requires the `Test, lint, and type-check` status check;
- blocks force pushes and branch deletion;
- prevents bypass where appropriate for the project.

## 4. Make the GHCR image public

GHCR is GitHub's Docker/OCI registry. The workflow publishes to:

```text
ghcr.io/unicheproject/unichemediaeditorbackend
```

Publishing uses the workflow's short-lived `GITHUB_TOKEN`; no registry PAT is
stored. GitHub initially creates a container package as private. After the first
successful **Publish image** job, open the package settings and change its
visibility to **Public**. Public GHCR images support anonymous pulls, so the VM
does not need registry credentials.

The first deployment can fail at `docker compose pull` before that one-time
visibility change. Once the package is public, re-run the failed workflow. A
public package cannot later be changed back to private.

## 5. Configure Nginx and Cloudflare

The Compose stack publishes the API only on `127.0.0.1:8000`. Start from
[deploy/nginx.conf.example](deploy/nginx.conf.example), replace the example
hostname, and configure TLS/Cloudflare as required. The example disables proxy
buffering for the job-progress Server-Sent Events endpoint and permits the
application's 200 MB upload limit.

Set `CORS_ALLOW_ORIGINS` in the VM `.env` to the real frontend HTTPS origin;
do not leave the production value as `*`.

## Deployment behavior

The workflow:

1. builds the Docker image and runs Ruff, mypy, and pytest inside it;
2. publishes commit and `main` tags to GHCR;
3. addresses the deployed image by its immutable registry digest;
4. copies only `compose.prod.yml` to the VM;
5. validates Compose configuration and pulls images;
6. starts PostgreSQL and Redis, applies Alembic migrations, and rolls out the
   API and worker;
7. waits up to 60 seconds for `GET /health` and fails the deployment otherwise.

Deployments are serialized so two quick merges cannot deploy concurrently.
The VM records the active immutable image reference in
`/opt/uniche-media-editor/.image.env`. For manual Compose inspection, use:

```bash
cd /opt/uniche-media-editor
docker compose --env-file .image.env -f compose.prod.yml ps
```

Database and uploaded-asset backups are deliberately not automated here. Set
up and test Hetzner volume/snapshot backups plus logical PostgreSQL backups
before treating the service as production-ready.
