# Docs Structure

This GitHub Pages site is now modularized for easier maintenance:

## Files

- **index.html** (4.7 KB) — Main page shell with nav, hero, footer, and dynamic content loader
- **styles.css** (28 KB) — All CSS styling extracted from inline style
- **script.js** (44 KB) — All JavaScript interactive features
- **sections/** — HTML fragments for each page section:
  - `problem.html` — The problem statement
  - `datasets.html` — Dataset overview
  - `pipeline.html` — Model architecture
  - `federated.html` — Federated learning approach
  - `results.html` — Results and insights
  - `privacy.html` — Privacy mechanisms
  - `speed.html` — Performance/deployability
- **assets/** — PNG diagrams (pipeline, architecture, etc.)

## How It Works

1. Browser loads `index.html` (lightweight)
2. JavaScript at the bottom (in `index.html`) fetches all 8 section files via XHR
3. Sections are inserted into the `#content` container
4. External `script.js` runs to animate visualizations, handle interactions, etc.
5. External `styles.css` styles everything

## Navigation

- Anchor links in nav (`#problem`, `#pipeline`, etc.) still work
- Single-page experience is preserved
- Smooth scrolling works thanks to `scroll-behavior: smooth` in CSS

## Editing

**To add/edit content:**
- Edit the relevant section in `sections/`
- Or edit `index.html` for nav/hero/footer

**To add/edit styles:**
- Edit `styles.css`

**To add/edit interactivity:**
- Edit `script.js`

## Performance

- **Before:** 100 KB (1 file, 1,785 lines)
- **After:** ~4.7 KB index + 28 KB CSS + 44 KB JS + 23 KB sections = 100 KB total (same)
- **Benefit:** Modular, maintainable structure; easier to locate and edit specific features
