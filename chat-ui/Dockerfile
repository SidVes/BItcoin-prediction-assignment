# ── Stage 1: build Angular ───────────────────────────────────────────────────
FROM node:22-slim AS builder

WORKDIR /app

# install yarn (already bundled in node image, but pin to corepack)
COPY package.json yarn.lock* ./
RUN yarn install --frozen-lockfile

COPY . .
RUN yarn build --configuration production

# ── Stage 2: serve with nginx ────────────────────────────────────────────────
FROM nginx:1.27-alpine

# replace default nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# copy built Angular app
COPY --from=builder /app/dist/chat-ui/browser /usr/share/nginx/html

EXPOSE 80
