# Age-Differentiated Multimodal Style in YouTube News Thumbnails:
# A Feature-Level Analysis of Visual and Textual Signals

<!-- ※ SUBMISSION: Author section must be left BLANK for anonymous review -->
<!-- Final version (if accepted): -->
<!-- Yang, Heeseok    | Sungkyunkwan University, Republic of Korea -->
<!-- Lee, Yena        | Sungkyunkwan University, Republic of Korea -->
<!-- Lee, Jungbin     | Sungkyunkwan University, Republic of Korea -->
<!-- Lee, Jungeun     | Sungkyunkwan University, Republic of Korea -->
<!-- Lee, Jangwon     | Sungkyunkwan University, Republic of Korea -->

---

## ABSTRACT

YouTube thumbnails and titles constitute the first multimodal signal users encounter before selecting a video. This study asks whether these first-impression signals differ systematically by viewer age group within the same content category. Focusing on the Current Affairs/News/Events subcategory of YouTube's Society category, we compare dominant age groups ~34 (n=3,701) and 65+ (n=3,057) using a three-pipeline framework: (1) ten interpretable visual features extracted via YOLOv8, EasyOCR, and FER; (2) DINOv2 visual embeddings interpreted via Grad-CAM and SHAP; and (3) KLUE-RoBERTa title embeddings and 35+ linguistic features, supplemented by LLaVA-based thumbnail captioning. Multimodal late fusion achieves AUC=0.905. The 65+ group exhibits higher text density, more saturated colors, and sadder facial expressions; the ~34 group favors minimal text, brighter tones, and neutral-to-positive emotion — patterns replicated across feature-level and deep-learning analyses.

**KEYWORDS**: YouTube thumbnails; age groups; multimodal analysis; visual features; content recommendation

---

## INTRODUCTION

YouTube's recommendation algorithm shapes content consumption across all age groups, yet the visual and textual signals embedded in thumbnails and titles — the user's first moment of content encounter — remain understudied from an age-differentiated perspective. Prior work has examined recommendation systems (Covington et al., 2016) and viewing motivations (Sundar & Limperos, 2013), but has not quantitatively characterized how thumbnail and title style differs for the same content category across age groups.

We draw on Uses & Gratifications Theory (Katz et al., 1973) — which holds that audiences actively select media to satisfy age-differentiated psychological needs — and Multimodal Communication Theory (Kress & van Leeuwen, 2001) — which frames thumbnails as composite image–text artifacts whose visual grammar encodes meaning. Four research questions guide the study: (RQ1) Do thumbnails and titles exhibit measurable differences across age groups? (RQ2) Which features most discriminate the groups? (RQ3) Can these features be quantified reliably? (RQ4) What multimodal combination best predicts viewer age group?

---

## DATA

Data were collected via the YouTube Data API from the **Society category**, selected for its relatively uniform age distribution across all viewer groups (verified via viewer-age entropy analysis). To reduce cross-topic noise, a Vision-Language Model reclassified all Society videos into fine-grained subcategories; the **Current Affairs/News/Events** subcategory was retained for its large and age-balanced sample. Viewer age groups were defined using YouTube Analytics dominant-age labels: **~34** (≤34 years, n=3,701) and **65+** (≥65 years, n=3,057), totaling **n=6,758**.

---

## METHOD

Figure 1 illustrates the three-pipeline framework applied across image and text modalities.

**Figure 1. Three-Pipeline Analysis Framework**

```
┌──────────────────────────────────────────────────────────────────┐
│  YouTube: Current Affairs/News/Events  (n = 6,758)               │
│  Dominant age group:  ~34 (n=3,701)  │  65+ (n=3,057)            │
└───────────────────┬──────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────────────┐
        ▼           ▼                   ▼
  ① Visual      ② DINOv2           ③ Title Text
  Features      Embeddings          + Captioning
  (10 feats)    (768-dim)          RoBERTa (768-dim)
  RF→AUC .720   XGB→AUC .819       XGB→AUC .881
  [Grad-CAM]    [SHAP dims]        [Attention Rollout]
        │           │                   │
        └───────────┴───────────────────┘
                        │
                  Late Fusion
                  AUC = 0.905
```

*Figure 1. Overview of the three analytical pipelines and their fusion. Each pipeline produces independent age-group predictions subsequently combined via probability-averaging late fusion.*

### Visual Feature Extraction

**Interpretable features (10):** person_ratio and person_count (YOLOv8s-seg segmentation); text_ratio (EasyOCR bounding-box area); five HSV color features (saturation, brightness, warm_color_ratio, Shannon color entropy, hue circular std); head_pose (MediaPipe Face Landmarker: front/side/none); expression_dominant (FER; 7 emotion classes). Group differences tested via Mann-Whitney U (continuous) and chi-square (categorical); multivariate importance via Random Forest mean decrease impurity (MDI).

**DINOv2 embeddings:** Each thumbnail encoded into a 768-dim vector via facebook/dinov2-base. XGBoost classifier trained on embeddings; SHAP values identified key dimensions (dim-038: text–background contrast / saturation; dim-606: natural vs. graphic background; dim-108: object-level saliency). Grad-CAM heatmaps and deletion/insertion experiments quantified patch-level contribution; Attention Rollout mapped region importance (text, person, background) per age group.

### Title Text Analysis

**Linguistic features (35+):** Five feature categories — basic structure (char_count, word_count, type-token ratio), punctuation/symbols (emoji, ellipsis, bait punctuation), lexical tone (superlatives, curiosity keywords), morphological tags via Kiwi (noun/verb/adverb ratio, sentence type), and multimodal similarity (KoCLIP image–title cosine, E5 title–OCR cosine). SHAP identified discriminative features per age group.

**RoBERTa embeddings:** KLUE-RoBERTa CLS token (768-dim) fine-tuned on the target subcategory. Attention Rollout traced word-level differences between age groups. Thumbnail titles extracted via two-stage OCR pipeline (qwen2.5vl:7b → qwen2.5:7b correction); after quality filtering, 15,579 titles (92.2%) retained and embedded identically to video titles.

### Image Captioning

LLaVA 1.5 generated descriptive captions per thumbnail (emotion, action, background, text-dominant composition). After NLTK preprocessing and within-channel cosine-similarity deduplication (threshold=0.85), TF-IDF vectorization and XGBoost + SHAP identified discriminative semantic tokens per age group.

### Multimodal Fusion

Three modalities — thumbnail image (DINOv2, 768-dim), video title (RoBERTa, 768-dim), thumbnail title (RoBERTa, 768-dim) — were combined via probability-averaging **late fusion**. Optimal text/image blend was determined by grid search over weighting parameter w ∈ [0, 1].

---

## RESULTS

### Classification Performance by Modality

Table 1 reports 5-fold Stratified CV results for each pipeline and their fusion.

| Modality / Method | Classifier | Accuracy | ROC-AUC |
|---|---|:---:|:---:|
| Visual features (10 interpretable) | Random Forest | 0.668 | 0.720 |
| DINOv2 embeddings (768-dim) | XGBoost | 0.757 | 0.819 |
| Title linguistic features (35+) | XGBoost | 0.676 | 0.720 |
| Title RoBERTa embeddings (768-dim) | XGBoost | 0.808 | 0.881 |
| **Late Fusion (all 3 modalities)** | **Ensemble** | **0.833** | **0.905** |

*Table 1. Age-group classification performance by modality. Late Fusion combines thumbnail image, video title, and thumbnail title embeddings.*

### Key Visual Feature Differences

Table 2 summarizes statistically significant visual feature differences between age groups (Mann-Whitney U / chi-square).

| Feature | ~34 group | 65+ group | p-value |
|---|:---:|:---:|:---:|
| text_ratio | 0.234 | 0.345 | 2.02×10⁻¹¹⁸ *** |
| color_saturation | 77.8 | 92.4 | 8.90×10⁻⁵⁹ *** |
| color_brightness | 124.6 | 118.4 | 6.38×10⁻⁸ *** |
| color_entropy | 2.327 | 2.515 | 2.39×10⁻²⁵ *** |
| sad expression ratio | 19.0% | 28.4% | χ²=54.33 *** |

*Table 2. Key visual feature differences between age groups. *** p<0.001.*

### Interpretive Findings

**Random Forest importance:** text_ratio alone accounted for ~39% of age separability (MDI=0.393); four color features together (~0.37); head_pose contributed negligibly (0.003). color_entropy and hue_std were correlated (r=0.58), confirming partial redundancy.

**Grad-CAM:** The ~34 model relies on a few concentrated salient patches (person/face region); the 65+ model integrates cues holistically across text, person, and background — consistent with the higher visual complexity observed in Table 2.

**Figure 2. Grad-CAM Attention Comparison (~34 vs. 65+)**

> **[FIGURE PLACEHOLDER]**
> Insert representative Grad-CAM heatmap pairs:
> ~34 thumbnails → focal attention on person/face region
> 65+ thumbnails → distributed attention across text overlay + full image
> Source: `/home/urp_jwl2/urp_bin/gradcam_correct_34_top50` (Google Drive assets)

*Figure 2. Grad-CAM heatmaps for correctly classified high-confidence samples. The ~34 model attends to localized person regions; the 65+ model attends holistically to text, person, and background.*

**DINOv2 SHAP dimensions:** dim-038 activates on high-contrast text–background regions (dominant in 65+); dim-606 distinguishes natural vs. graphic backgrounds; dim-108 encodes object-level saliency.

**Image captioning (SHAP tokens):** 65+-discriminative tokens: *suit, tie* (authority/formality), *crying, yelling* (emotional intensity); ~34-discriminative tokens: *painting, screen, sitting, mask* (informal/indoor framing).

**Fusion weighting:** Optimal blend at text weight 0.6 / image weight 0.4 (AUC=0.901), confirming that semantic title content carries a stronger age signal than visual content alone.

---

## CONCLUSION

This study demonstrates that YouTube news thumbnails and titles carry quantifiable, statistically significant age-group signals even within a controlled content subcategory. The 65+ group consistently shows higher text density, more vivid and complex color palettes, and emotionally heavier imagery (sad expressions, authority-signaling attire); the ~34 group favors minimal text, brighter tones, and calm or positive framing. These patterns are replicated across interpretable feature analysis, DINOv2 deep embeddings, linguistic features, and image captioning, confirming their robustness. Multimodal late fusion (AUC=0.905) substantially outperforms any single modality, and the text modality's dominant weight (0.6 vs. 0.4) identifies semantic title framing as the strongest single age signal. These findings offer actionable guidance for age-targeted content creators and open a pathway toward automated, age-adaptive style tools for digital information access.

---
<!-- ============================================================ -->
<!-- 이하 섹션은 ASIS&T 페이지 카운트에 미포함 (References 포함) -->
<!-- ============================================================ -->

## GENERATIVE AI USE

We employed the following generative AI tools for the purposes listed below. (1) **LLaVA 1.5** for automated thumbnail image captioning to extract semantic descriptions from visual content; we evaluated the output by applying NLTK-based preprocessing and within-channel cosine-similarity filtering (threshold=0.85) to remove noisy and duplicate captions. (2) An **AI-assisted translation tool** for translation assistance during manuscript preparation; all translated content was reviewed and revised by the authors for accuracy and fidelity. The authors assume all responsibility for the content of this submission.

---

## ACKNOWLEDGMENTS

This work was supported by the Undergraduate Research Program (URP) of Sungkyunkwan University. The authors are affiliated with the **Intelligence and Interactive Systems Lab**, Sungkyunkwan University, Republic of Korea.

---

## REFERENCES

*(APA 7th edition, alphabetical order by first author's last name)*

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 9650–9660.

Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks for YouTube recommendations. *Proceedings of the 10th ACM Conference on Recommender Systems*, 191–198.

Geise, S., & Baden, C. (2015). Putting the image back into the frame: Modeling the linkage between visual communication and frame-processing theory. *Communication Theory*, 25(1), 46–69.

Katz, E., Blumler, J. G., & Gurevitch, M. (1973). Uses and gratifications research. *The Public Opinion Quarterly*, 37(4), 509–523.

Kress, G., & van Leeuwen, T. (2001). *Multimodal discourse: The modes and media of contemporary communication*. Arnold.

Ksiazek, T. B., Peer, L., & Lessard, K. (2016). User engagement with online news: Conceptualizing interactivity and exploring the relationship between online news videos and user comments. *New Media & Society*, 18(3), 502–520.

Li, H., Zhang, Y., Keuper, M., & Yao, A. (2023). LLaVA: Visual instruction tuning. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

Park, S., Moon, H., Kim, J., Lee, J. K., & Kim, J. (2021). KLUE: Korean language understanding evaluation. *Advances in Neural Information Processing Systems (NeurIPS)*, 35.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., … Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 8748–8763.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 618–626.

Sundar, S. S., & Limperos, A. M. (2013). Uses and grats 2.0: New gratifications for new media. *Journal of Broadcasting & Electronic Media*, 57(4), 504–525.
