# Frontend development

One Vue application owns recording import, analysis, calibration, review,
results and settings. `src/main.ts` mounts it from `index.html`. Components
render escaped templates; domain rules and network state live in small modules.
There are no inline scripts, HTML string renderers, iframe applications or
compatibility pages.

Use Node 22 and npm:

```sh
cd frontend
npm ci
npm run dev
npm run check
npm test
npm run format
npm run build
npx playwright install chromium
npm run test:browser
```

Run `video2tenhou web --no-browser` alongside Vite. Vite proxies API, evidence,
exports and tile artwork to the backend on port 8765; adjust `vite.config.ts`
if using another port. Open `/` on Vite's address.

`npm run build` writes the single application and bundled dependencies into
`src/video2tenhou/tool/static`. These generated files are ignored by Git.
Build them before running the backend from a checkout, running Python web tests,
or creating distributions with `uv build`. Packaging includes the built assets
so wheels, source releases and starter packages work offline without Node.
The build removes obsolete bundles while preserving tile artwork and licenses.
CI requires Prettier, ESLint, and strict `vue-tsc` checks, including Vue templates,
TypeScript modules, tests, and build/configuration scripts. `npm run check` runs
all three; `npm run build` also requires them to pass. There are no source-file
exclusions, inline lint overrides, or skipped declaration checks. Add new code
in TypeScript and use `<script setup lang="ts">` in Vue components.

CI checks component tests, then builds assets from source before
running browser and Python tests and creating distributions.
Browser tests exercise the built application with controlled API responses.

## Responsibilities

- `src/studio/`: workspace selection, intake, analysis, settings and exports.
- `src/review/`: review decisions, answer editors, evidence, calibration and labels.
- `src/shared/`: HTTP client, action error handling and the studio's single
  job-status poll (`useJob`: `GET /api/job`, every second while a job runs and
  every five seconds otherwise).
- `tests/`: Vitest and Vue Test Utils tests against mounted components and domain
  functions. No extraction or evaluation of scripts from built HTML.

The Python tests in `tests/web/` cover HTTP contracts, project isolation,
persistence, subprocess ownership and packaged frontend resources.

The backend has one listener and router (`tool/server.py`). `tool/http.py`
provides HTTP transport, `tool/review_routes.py` declares the project-scoped
review endpoints as individual Starlette routes, `tool/review_state.py` owns
per-recording evidence, answers and hand freshness, and `tool/workflow.py` owns
projects and the single job slot for preparation, analysis, hand updates and
calibration checks. All are modules of the same backend, launched by
`video2tenhou web`. Job failures reach the browser as the command's own message;
the processing log is fetched only while it is open.
