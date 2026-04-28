# Beamer Presentation Style Specification

Use this style for all technical presentations. It is a 16:9 Beamer deck with a muted navy/seahorse palette, TikZ system diagrams, and two-column content slides. Favor information density over whitespace, but never at the cost of clarity.

## 1. Document Setup

```latex
\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{Madrid}
\usecolortheme{seahorse}
```

Required packages: `inputenc` (utf8), `graphicx`, `booktabs`, `amsmath`, `tikz` (with libraries `arrows.meta, positioning, shapes.geometric, fit`), `listings`, `xcolor`.

## 2. Color Palette (fixed)

Define these three brand colors plus two code colors. Rename the brand prefix per project (e.g. `mu2eblue` → `projblue`), but keep the RGB values:

| Name | RGB | Use |
|---|---|---|
| `*blue`  | `0,84,147`    | Primary — titles, frame titles, structure, block titles, enhanced/emphasized diagram nodes |
| `*red`   | `178,34,34`   | Accent — query paths, dashed arrows, strings in code |
| `*green` | `0,128,64`    | Accent — "new" components in diagrams, added features |
| `codebg` | `245,245,250` | Code listing background |
| `codegreen` | `0,128,0`  | Code comments |

Color assignments:
```latex
\setbeamercolor{title}{fg=*blue}
\setbeamercolor{frametitle}{fg=*blue}
\setbeamercolor{structure}{fg=*blue}
\setbeamercolor{block title}{bg=*blue, fg=white}
```

## 3. Navigation & Footer

Strip the default Beamer navigation and replace the footer with a clean three-box author | title | date+page layout:

```latex
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{page number in head/foot}[appendixframenumber]
\setbeamertemplate{footline}{%
  \leavevmode%
  \hbox{%
    \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%
      \usebeamerfont{author in head/foot}\insertshortauthor
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
      \usebeamerfont{title in head/foot}\insertshorttitle
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
      \usebeamerfont{date in head/foot}\insertshortdate{}\hspace*{2em}%
      \insertframenumber\hspace*{2ex}%
    \end{beamercolorbox}}%
  \vskip0pt%
}
\setbeamertemplate{enumerate items}[default]  % plain Arabic numerals
```

## 4. Code Listings

```latex
\lstset{
  basicstyle=\ttfamily\scriptsize,
  backgroundcolor=\color{codebg},
  keywordstyle=\color{*blue}\bfseries,
  commentstyle=\color{codegreen},
  stringstyle=\color{*red},
  frame=single,
  rulecolor=\color{gray!30},
  breaklines=true,
  columns=fullflexible,
}
```

For code shown inside a `block`, drop the frame (`frame=none`) and use `\tiny`. Wrap in a `block` with a `\small` title describing what the reader is looking at (e.g. "Structured Text (embedded for search)").

## 5. Slide Archetypes

Every slide should fit one of these five patterns. If a slide doesn't fit, restructure the content.

### 5.1 Title slide
`\titlepage` only. Title on two lines with `\\`, a subtitle, author, institute, date.

### 5.2 Full-width system diagram (Before/After style)
Use when showing architecture. Wrap a TikZ picture in:
```latex
\vfill
\begin{center}
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[...] ... \end{tikzpicture}}
\end{center}
\vspace{0.3em}
\centering\small
\textcolor{gray}{Tagline $\cdot$ with $\cdot$ bullet-dot $\cdot$ separators}
\vfill
```
For "Before/After" pairs, keep layout, node positions, and style names identical across the two slides — only change colors (gray → blue/green) and add new nodes. Include a small legend row at the bottom of the "After" slide with swatches for `New` / `Enhanced` / `Pre-existing`.

### 5.3 Two-column content slide (most common)
```latex
\begin{columns}[T]
\begin{column}{0.48\textwidth}
  ...left...
\end{column}
\begin{column}{0.50\textwidth}
  ...right...
\end{column}
\end{columns}
```
Standard splits: `0.46/0.52`, `0.48/0.50`, `0.50/0.48`. Left column is usually prose and structure ("Problem / Approach / Result"), right column is usually a visual, table, or code block.

### 5.4 Centered concept diagram with explanation below
A single TikZ picture centered, followed by `\small` prose and a tight `itemize` of 2–3 takeaways. Used for single-idea slides like the query router fan-out.

### 5.5 Summary & Outlook
Two columns: left is "What was built" (completed work), right is "Near-term / Longer-term" (future work). End with a centered repository link in `*blue`.

## 6. TikZ Style Library (canonical)

Every system diagram uses these styles verbatim. Do not invent new ones.

```latex
box/.style={rectangle, draw=gray!60, fill=gray!12, rounded corners=3pt,
            minimum height=0.9cm, minimum width=2.6cm, align=center, font=\normalsize},
newbox/.style={rectangle, draw=*green!80, fill=*green!10, rounded corners=3pt,
               minimum height=0.9cm, minimum width=2.6cm, align=center, font=\normalsize, thick},
enhbox/.style={rectangle, draw=*blue!80, fill=*blue!8, rounded corners=3pt,
               minimum height=0.9cm, minimum width=2.6cm, align=center, font=\normalsize, thick},
store/.style={cylinder, draw=gray!60, fill=gray!10, shape border rotate=90,
              minimum height=0.8cm, minimum width=2.0cm, align=center, aspect=0.25},
arr/.style={-{Stealth[length=2.5mm]}, gray!60, thick},
newarr/.style={-{Stealth[length=2.5mm]}, *green!80, thick},
biarr/.style={<->, >=Stealth, gray!60, thick},
qarr/.style={-{Stealth[length=2.5mm]}, *red!70, thick, dashed},
lbl/.style={font=\small\itshape, text=gray!70},
newlbl/.style={font=\small\itshape, text=*green!90},
qlbl/.style={font=\scriptsize\itshape, text=*red!80},
```

**Semantic meaning (non-negotiable):**
- `box` = pre-existing / unchanged component
- `newbox` (green) = brand-new component
- `enhbox` (blue) = existing component that was enhanced
- `store` = database / persistent storage (cylinder)
- `arr` (solid gray) = data flow / indexing pipeline
- `newarr` (green) = new data flow
- `biarr` (double-headed gray) = bidirectional I/O
- `qarr` (dashed red) = query / runtime path (always distinct from indexing)

Labels go next to nodes in italic gray (`lbl`), not inside them. Use `\def\ytop{...}` / `\def\ymid{...}` / `\def\ybot{...}` at the top of large TikZ pictures so rows stay aligned when you rearrange.

## 7. Typography Rules

- Body text: default `\normalsize`. Drop to `\small` inside two-column slides, `\scriptsize` inside tables or tight enumerations, `\tiny` only inside code blocks.
- **Bold** for the opening word of a structural paragraph: `\textbf{Problem:}`, `\textbf{Approach:}`, `\textbf{Result:}`, `\textbf{Impact:}`. This is the main rhetorical pattern — use it aggressively.
- *Italics* only for quoted example text (user queries, descriptive labels in diagrams).
- `\texttt{}` for identifiers, file paths, config keys, tool names.
- Lists: always `\setlength\itemsep{0.05em}` to `0.15em`. Default Beamer spacing is too loose for this style.
- Numeric counts and percentages get bolded in tables when they are the "after" / "improved" column.

## 8. Tables

Always `booktabs`: `\toprule`, `\midrule`, `\bottomrule` — never `\hline`. Wrap in `\begin{center}{\scriptsize ... }\end{center}` for compact comparison tables. Use `\textbf{}` on the result column to highlight the punchline.

## 9. Language & Voice

The writing is as much part of the style as the visuals.

- **Structure every content slide around Problem → Approach → Result** (or Problem → Solution → Impact). Don't bury the lede.
- **Open technical slides with a single italicized framing sentence** before the bullets: *"Different questions need different search strategies."*
- **Use footnote-style asides in `\scriptsize`** at the bottom of columns to qualify or caveat the main claim.
- **Prefer concrete numbers over adjectives.** Not "a lot faster" — "∼10 ms vs ∼1–2 s". Not "many documents" — "2,765 code files".
- **Use the em-dash for reveals and reframing:** "The reranker does not find new documents — it re-orders existing results." Use `$\cdot$` as a bullet separator in taglines.
- **Avoid filler.** No "In this slide we will discuss…". The frame title is the introduction.
- **Code examples and real query strings** carry more weight than prose description. When explaining a system, show the actual JSON/struct/query.

## 10. Putting It Together: Slide Checklist

Before committing a slide, verify:

1. Frame title is a noun phrase, not a sentence (`Cross-Encoder Reranker`, not `How the reranker works`).
2. If there's a diagram, it uses only the canonical TikZ styles from §6.
3. Colors encode meaning: gray=baseline, green=new, blue=enhanced, red=query-time.
4. Two-column slides are `[T]`-aligned and sum to ≤0.98 to leave a gutter.
5. Any claim of improvement has a number next to it.
6. Page number appears bottom-right; no nav symbols.
7. No slide uses more than two font sizes below `\small` simultaneously.
8. The final "Summary & Outlook" slide ends with a repository/contact URL centered in the primary color.

## 11. Iterative Visual Review (mandatory before delivery)

LaTeX warnings are not enough. Overfull boxes, label collisions, columns that
clip into the footer, table rows that wrap awkwardly, dashed strokes that vanish,
TikZ nodes that overlap a cluster boundary — none of these surface as compile
errors. **Every slide deck must go through a compile → rasterize → eyeball →
fix loop until convergence.**

The loop:

1. **Compile** with `pdflatex` twice (cross-refs): `pdflatex -interaction=nonstopmode -halt-on-error presentation.tex`. Bail on errors immediately.
2. **Rasterize every page** to PNG at 150 DPI:
   ```bash
   rm -f /tmp/slides/*.png && mkdir -p /tmp/slides && \
     pdftoppm -r 150 presentation.pdf /tmp/slides/slide -png
   ```
3. **Look at every page** — not just the ones you edited. A line-number shift in one frame can break layout in a later frame.
4. **Catalog issues**: footer collisions, label-on-label, illegible thin/dashed strokes, table row that wraps, axis labels clipped, tikz node bisecting a circle, font-size cliffs, off-brand colors.
5. **Fix all issues** in a single edit pass when possible — re-using the loop is cheap; partial fixes that need re-eyeballing are not.
6. **Repeat** until no further visual issues remain. Stop only when a fresh sweep finds nothing.

Rules of thumb:

- **Tiny `Overfull \vbox` ($<2$ pt) is usually benign**, but always verify by looking — sometimes 1.5 pt clips a single descender.
- **Auto mode is no excuse to skip this loop**. Compile-and-ship without rasterizing has caused real regressions (footer-eaten conclusions, dead dashed edges, bisected nodes).
- **Plot-script changes count too.** If you change `make_slide_plots.py` or any figure source, regenerate the PNGs and re-rasterize the deck — a stale figure will silently drift from the slide caption.
- **Numbers must trace back.** Any cell in a results table either ties to `docs/findings.md` or comes from a one-shot Python computation — record the snippet in the commit body or alongside the script so it can be re-run later.