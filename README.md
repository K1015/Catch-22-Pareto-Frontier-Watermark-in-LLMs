# Catch-22 Project Page

Static project page for:

> Catch-22: On the Fundamental Tradeoff Between Detectability and Robustness in LLM Watermarking

The page is based on the `eliahuhorwitz/Academic-project-page-template` structure and is intended for GitHub Pages hosting under the `K1015` account.

## Local Preview

Opening `index.html` directly works for basic inspection. A local server is better for matching GitHub Pages behavior:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Expected GitHub Pages Target

Assumed repository:

```text
https://github.com/K1015/Catch-22-Pareto-Frontier-Watermark-in-LLMs
```

Expected project page URL after GitHub Pages is enabled:

```text
https://k1015.github.io/Catch-22-Pareto-Frontier-Watermark-in-LLMs/
```

If the website uses a different repository name, update these values in `index.html`:

- `og:url`
- `og:image`
- `citation_pdf_url`
- the Code button link
- the reproduction repository link

## Publish From This Folder

If this folder is pushed as the root of the Pages repository:

```bash
git init
git remote add origin https://github.com/K1015/Catch-22-Pareto-Frontier-Watermark-in-LLMs.git
git add -A
git commit -m "Add Catch-22 project page"
git push -u origin main
```

In GitHub, enable Pages from `Settings -> Pages`, using the `main` branch and `/ (root)` as the source.
