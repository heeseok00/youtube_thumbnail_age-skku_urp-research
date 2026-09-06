# Title candidates (ICWSM)

Working pick:

Visual and Textual Signals of Audience Age in Korean YouTube Thumbnails and Titles Across Information Categories

## Why Korean belongs in the title

Earlier drafts left the corpus out of the title. The title analysis makes that
untenable. Word origin (native Korean, Sino-Korean, loanword) is a Korean
linguistic category, KLUE-RoBERTa is a Korean model, and the headline lexical
result (life-circumstance and illness vocabulary for 65-and-over, loanwords for
34-and-under) only holds in Korean. The language is a condition of the finding,
not a sampling detail. ICWSM also publishes country- and language-specific
studies without penalty; two are in the precedent list below.

*Content Categories* becomes *Information Categories* to match the body, which
frames the four domains as informational rather than entertainment content.

## Why drop multimodal from the title

ICWSM uses *multimodal* in titles when the paper's claim is fusion, a joint VLM pipeline, or an object that is image-plus-text as one artifact (memes, protest posts, multi-channel datasets).

Kulsum et al. (2026), *Beyond Metadata: Multimodal, Policy-Aware Detection of YouTube Scam Videos*, is the fusion case. The title is earned by an ablation: text-only 76.61 F1, visual 79.61, fused frames plus title plus description 82.96. Multimodal is the method result.

This study does not make that claim. Thumbnails and titles are measured as two layers (visual features; formal and lexical title features). The headline result is which signals recur across EDU, HEALTH, LIFESTYLE, and SOCIETY, and which reverse with category. Putting multimodal in the title invites a fusion ablation this paper does not run.

The closer published parallel is already in the reference list: Xue et al. (2026). The abstract says they conducted multimodal feature analysis, but the title names the measured things, *Audiovisual and Thematic Markers*. Follow that pattern: name visual and textual signals (or features), and name thumbnails and titles.

Keep multimodal out of the title. In the body it can stay only where the sentence means "thumbnail and title together as pre-watch cues," not "a fused model."

## What the title should show

1. Several content categories, not a single genre.
2. A comparison by audience age (not \(\le 34\) / \(\ge 65\) in the title; those belong in Section 2.1).
3. Visual and textual analysis, stated as features, signals, or thumbnails and titles.

## Candidates

| ID | Title | Categories | Age | Visual / textual | Note |
| --- | --- | --- | --- | --- | --- |
| A | Visual and Textual Signals of Audience Age in Korean YouTube Thumbnails and Titles Across Information Categories | yes | yes | signals + objects | Working pick |
| A2 | Age-Associated Visual and Textual Features of Korean YouTube Thumbnails and Titles Across Information Categories | yes | yes | features + objects | Fallback if the group prefers *features* |
| A3 | What Recurs Across Categories: Visual and Textual Signals of Audience Age in Korean YouTube Thumbnails and Titles | finding-led | yes | both | Colon form; use once results are locked |
| A4 | Visual and Textual Signals of Audience Age in Korean YouTube Thumbnails and Titles | implicit | yes | both | Short form; abstract must name the four domains |
| B | Age-Associated Visual and Textual Features of YouTube Thumbnails and Titles Across Information Categories | yes | yes | features + objects | Closest to the advisor wording |
| C | How YouTube Thumbnails and Titles Differ by Audience Age Across Information Categories | yes | yes | objects only | Plain; less methods-coded |
| D | Thumbnails, Titles, and Age: Visual and Textual Features Across YouTube Information Categories | yes | yes | both | Colon form; compact hook |
| E | Before the Click: Visual and Textual Features of Age-Associated YouTube Thumbnails and Titles Across Categories | yes | yes | both | Names the pre-watch setting; long |
| F | What Recurs Across Categories: Visual and Textual Signals of Age in YouTube Thumbnails and Titles | finding-led | yes | both | Stronger once results are locked |
| G | Visual and Textual Features of Korean YouTube Thumbnails and Titles Across Age Groups and Content Categories | yes | yes | both | Honest about the corpus; narrower pitch |
| H | Cross-Category Visual and Textual Features of Younger and Older YouTube Audiences | yes | yes | features only | Short; easy to misread as a user study |
| X | Age-Related Multimodal Style in YouTube Thumbnails and Titles Across Content Categories | yes | yes | multimodal | Reject; same word as Kulsum, different claim |

## Precedent titles (how ICWSM uses multimodal)

### Title uses multimodal for fusion

- [Beyond Metadata: Multimodal, Policy-Aware Detection of YouTube Scam Videos](https://ojs.aaai.org/index.php/ICWSM/article/view/42698) (ICWSM, 2026). Kulsum et al. Detect YouTube scams. Multimodal means a fused model (frames + title + description) that beats text-only and visual-only baselines.

### Title avoids multimodal; abstract still says it

- [Catching Dark Signals in Algorithms: Unveiling Audiovisual and Thematic Markers of Unsafe Content Recommended for Children and Teenagers](https://ojs.aaai.org/index.php/ICWSM/article/view/42768) (ICWSM, 2026). Xue et al. Feature-level and theme-level analysis of short videos recommended to children. The abstract says "multimodal feature analysis"; the title names audiovisual and thematic markers.

### Title uses multimodal for the object or the dataset

- [Large-Scale Multimodal Content Analysis and Annotation with Vision-Language Models](https://ojs.aaai.org/index.php/ICWSM/article/view/42718) (ICWSM, 2026). Nemani and Garimella. WhatsApp political content in India (text, image, video) annotated with VLMs. Multimodal is the content and the joint toolkit.

- [Large Scale Narrative Analysis of Multimodal Memes](https://ojs.aaai.org/index.php/ICWSM/article/view/42722) (ICWSM, 2026). Peh et al. Cluster memes and write corpus-level narratives. Multimodal names the object: image-plus-text memes.

- [The First Mass Protest on Threads: Multimodal Mobilization and AI-Generated Visuals in Taiwan’s Bluebird Movement](https://ojs.aaai.org/index.php/ICWSM/article/view/42759) (ICWSM, 2026). Weener and Chang. Protest posts as text plus image. Multimodal means the two channels of mobilization.

- [MemeMatch: A Large-Scale Dual-Context Multimodal Dataset and Retrieval System for Internet Memes](https://ojs.aaai.org/index.php/ICWSM/article/view/42785) (ICWSM, 2026). Le et al. Dataset of image-with-text memes. Multimodal describes the resource.

- [SemioMeme: A Symbolic–Subsymbolic Knowledge Graph Dataset for Multimodal Meme Analysis](https://ojs.aaai.org/index.php/ICWSM/article/view/42792) (ICWSM, 2026). Sherratt et al. Knowledge graph with vision and text embeddings. Multimodal is the analysis target.

- [MASH: A Multiplatform and Multimodal Annotated Dataset for Societal Impact of Hurricane](https://ojs.aaai.org/index.php/ICWSM/article/view/42795) (ICWSM, 2026). Hurricane posts from Reddit, TikTok, and YouTube, labeled on text and image together. Multimodal is the annotation unit.

### Image plus text, no multimodal in the title

- [Disturbed YouTube for Kids: Characterizing and Detecting Inappropriate Videos Targeting Young Children](https://ojs.aaai.org/index.php/ICWSM/article/view/7320) (ICWSM, 2020). Papadamou et al. Titles, thumbnails, tags, and video content, without multimodal in the title.

## If A is too long

Do not drop *Korean*, *Audience Age*, or *Visual and Textual*. Drop *Across
Information Categories* only if the abstract's first sentence names the four
domains, which gives A4. Use A2 if the group prefers *features* to *signals*,
and A3 if a colon title is wanted for the program.

Rows B through H below predate the decision to name the corpus; they are kept
for the tradeoffs they record, not as live candidates.
