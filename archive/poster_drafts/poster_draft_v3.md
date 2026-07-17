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

When users scroll through YouTube, their viewing choices are shaped by what they see first: the thumbnail and title. Yet how these first-impression signals differ across age groups has rarely been examined through a quantitative, multimodal lens. We analyze thumbnails and titles in the Current Affairs/News/Events subcategory of YouTube's Society category, comparing dominant viewer groups ~34 (n=3,701) and 65+ (n=3,057) via interpretable visual features, deep image embeddings, and semantic title embeddings with linguistic analysis. Multimodal late fusion achieves AUC=0.905. The 65+ group's thumbnails show denser text overlays, more saturated colors, and a higher prevalence of negative emotion; the ~34 group's thumbnails show lighter tones, minimal text, and positive or neutral emotion. These findings demonstrate that age-differentiated content consumption leaves measurable, quantifiable traces in the visual and textual style of thumbnails and titles.

**KEYWORDS**: YouTube thumbnails; age-differentiated consumption; multimodal analysis; visual embeddings; image captioning

---

## INTRODUCTION

YouTube is the world's largest video-sharing platform, where users across generations make content selection decisions based on what they encounter first: the thumbnail-title pair, a composite visual-textual artifact that encodes meaning and shapes viewing behavior (Geise & Baden, 2015). Research has shown that the visual and textual features of thumbnails and titles are systematically linked to viewer engagement (Cui et al., 2024), and that thumbnail aesthetics differ measurably across cultural and national contexts (Limpijankit & Kender, 2024).

However, these lines of work have focused on content-level popularity predictors or cross-group aesthetic comparison, not on how thumbnail and title style characterizes *age-differentiated* consumption within the same content category. Although prior work has established that viewer demographics are reflected in what content people watch (Ulges et al., 2013), no study has quantitatively characterized the multimodal style differences in thumbnails and titles that attract different age groups. Focusing on Current Affairs/News/Events, a subcategory selected for its high viewer-age entropy indicating balanced consumption across all generations, this study asks: *What measurable multimodal features distinguish thumbnails and titles consumed predominantly by younger versus older viewers, and can these differences be quantitatively characterized?*

---

## DATA

We collaborated with Vling (vling.net), a YouTube analytics platform, to obtain channel lists for the Society category ranked by dominant viewer age group. Using the resulting channel IDs, thumbnail images and video metadata (titles, upload dates, view counts) were retrieved via the YouTube Data API. The Society category was selected because its viewer-age entropy, computed across eight YouTube age brackets, was among the highest across all categories, indicating a relatively balanced cross-age audience and minimizing topic-driven demographic confounds (Ulges et al., 2013). To further reduce within-category topic noise, LLaVA 1.5 (Liu et al., 2023) reclassified all Society videos into fine-grained subcategories; the Current Affairs/News/Events subcategory was retained for its large, age-balanced sample. Viewer age groups were defined using YouTube Analytics dominant-age labels: ~34 (n=3,701) and 65+ (n=3,057), totaling n=6,758.

---

## ANALYSIS FRAMEWORK

Grounded in Uses and Gratifications Theory (Katz et al., 1973) and Multimodal Communication Theory (Kress & van Leeuwen, 2001), we adopt a **deep learning-guided feature validation** approach. First, a DINOv2-base binary classifier (Oquab et al., 2024) is trained to distinguish age groups, and Grad-CAM (Selvaraju et al., 2017) visualization with deletion/insertion experiments and Attention Rollout (Abnar & Zuidema, 2020) are applied to identify which thumbnail regions (text overlays, person areas, or background) most influence the model's decisions. These findings motivate the design of **ten handcrafted visual features**: person_ratio and person_count (YOLOv8s-seg); text_ratio (EasyOCR); five HSV color features (saturation, brightness, warm_color_ratio, color entropy, hue_std); head_pose (MediaPipe); and dominant facial expression via FER (7 classes). Each feature is validated through statistical testing (Mann-Whitney U; chi-square; Random Forest MDI). Complementing the visual analysis, **35+ linguistic features** are extracted from video titles (retrieved via YouTube Data API), and thumbnail-embedded text is separately extracted via a VLM-based OCR pipeline (qwen2.5vl:7b). Both are encoded with KLUE-RoBERTa CLS embeddings (768-dim; Park et al., 2021), fine-tuned on the target subcategory, with Attention Rollout tracing word-level age signals. LLaVA 1.5 (Liu et al., 2023) captions provide semantic grounding for visual patterns identified by Grad-CAM; after preprocessing and deduplication, TF-IDF and SHAP (Lundberg & Lee, 2017) identify the most discriminative tokens per age group. Finally, all modalities are combined via **probability-averaging late fusion**, with the optimal text-to-image weighting determined through grid search.

---

## RESULTS AND DISCUSSION

### Classification Performance and Key Discriminating Signals

Interpretable visual features (Random Forest, Acc=0.668) provide a transparent baseline; deep embeddings substantially outperform them: DINOv2 (XGBoost, Acc=0.748, AUC=0.819) and RoBERTa title embeddings (XGBoost, Acc=0.805, AUC=0.881). Late fusion of all three modalities achieves Acc=0.833, AUC=0.905 (text/image weight=0.6/0.4). Table 1 reports the key discriminating signals identified across all modalities.

| Modality | Feature / Signal | ~34 group | 65+ group | Evidence |
|---|---|:---:|:---:|:---:|
| Visual | text_ratio (MDI=0.393) | 0.234 | 0.345 | MW p=2.02×10⁻¹¹⁸ *** |
| Visual | color_saturation | 77.8 | 92.4 | MW p=8.90×10⁻⁵⁹ *** |
| Visual | color_brightness | 124.6 | 118.4 | MW p=6.38×10⁻⁸ *** |
| Visual | color_entropy | 2.327 | 2.515 | MW p=2.39×10⁻²⁵ *** |
| Visual | expression_dominant | positive/neutral | negative | χ²=66.99 *** |
| Text | Attention keywords | 이유·방법·가장 | 몰아·비밀·마음 | Rollout |
| Caption | SHAP tokens | painting·sitting·mask | suit·crying·yelling | SHAP |

*Table 1. Key discriminating signals across modalities. *** p<0.001. Mann-Whitney U (continuous) / chi-square (categorical). Visual: warm_color_ratio excluded (p=0.106, ns). Text: Title Attention Rollout top words. Caption: LLaVA SHAP top tokens.*

### Interpretive Findings

**Figure 2. Grad-CAM Attention Comparison (~34 vs. 65+)**

> **[FIGURE PLACEHOLDER — insert Grad-CAM heatmap pairs]**
> ~34: concentrated attention on person/face region
> 65+: distributed attention across text overlay, person, and background
> Source: urp_bin/gradcam_correct_34_top50

*Figure 2. Grad-CAM heatmaps for correctly classified high-confidence samples. The ~34 model attends locally to person regions; the 65+ model attends holistically to text, person, and background.*

Grad-CAM deletion/insertion experiments reveal that both age groups rely on a combination of text, person, and background regions, with text emerging as the most consistently discriminative single region across groups (RF MDI=0.393; Deletion AUC=0.852). However, the two groups differ in how they weight these regions. For ~34 thumbnails, the person region carries the strongest signal (Insertion AUC: person 0.964 > text 0.906), indicating that a salient person anchors the classification. For 65+ thumbnails, text removal produces the largest confidence drop, yet text alone restores prediction to only 0.451, confirming that the full holistic composition, not any single region, is required. DINOv2 SHAP corroborates this structure: embedding dimensions sensitive to high-contrast text and text-background contrast dominate 65+ predictions, while dimensions encoding salient objects drive ~34 predictions.

The same contrast emerges in the semantic layers. Caption SHAP tokens show ~34-discriminative terms clustering around informal and creative contexts (*painting*, *screen*, *sitting*, *mask*), while 65+-discriminative terms foreground authority, formality, and emotional intensity (*suit*, *tie*, *crying*, *yelling*). Title Attention Rollout reveals a parallel pattern: ~34 titles weight information-seeking words ("reason", "most", "method"), while 65+ titles foreground emotional-narrative words ("binge", "heart", "secret", "moment"). The optimal late fusion weight (text 0.6 / image 0.4) confirms that title framing carries a marginally stronger age signal than thumbnail appearance alone.

Taken together, these findings demonstrate that age-differentiated content consumption is systematically encoded across the visual and textual dimensions of thumbnails and titles, offering actionable implications for age-targeted content design and information access systems.

---
<!-- ============================================================ -->
<!-- 이하 섹션은 ASIS&T 페이지 카운트에 미포함 -->
<!-- ============================================================ -->

## GENERATIVE AI USE

We employed the following generative AI tools: (1) **LLaVA 1.5** for automated thumbnail image captioning; output quality was evaluated via NLTK preprocessing and cosine-similarity filtering (threshold=0.85). (2) An **AI-assisted writing tool** for manuscript drafting assistance; all content was reviewed and revised by the authors. The authors assume all responsibility for the content of this submission.

---

## ACKNOWLEDGMENTS

This work was supported by the Undergraduate Research Program (URP) of Sungkyunkwan University. The authors are affiliated with the **Intelligence and Interactive Systems Lab**, Sungkyunkwan University, Republic of Korea.

---

## REFERENCES

*(APA 7th edition, alphabetical order)*

Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 4190–4197.

Cui, G., Chung, Y., Peng, L., & Wang, Q. (2024). Clicks for money: Predicting video views through a sentiment analysis of titles and thumbnails. *Journal of Business Research*, 183, 114849.

Geise, S., & Baden, C. (2015). Putting the image back into the frame: Modeling the linkage between visual communication and frame-processing theory. *Communication Theory*, 25(1), 46–69.

Katz, E., Blumler, J. G., & Gurevitch, M. (1973). Uses and gratifications research. *The Public Opinion Quarterly*, 37(4), 509–523.

Kress, G., & van Leeuwen, T. (2001). *Multimodal discourse: The modes and media of contemporary communication*. Arnold.

Limpijankit, M., & Kender, J. (2024). Detecting cultural differences in news video thumbnails via computational aesthetics. *Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)*. https://doi.org/10.36190/2024.61

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.

Newman, N., Fletcher, R., Robertson, C. T., Ross Arguedas, A., & Nielsen, R. K. (2024). *Reuters Institute digital news report 2024*. Reuters Institute for the Study of Journalism.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., & Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*.

Park, S., Moon, J., Kim, S., Cho, W. I., Han, J., Park, J., Chang, M., Lim, H., Oh, S., Park, J., Shin, J., Kim, S., Park, K., Oh, N., Kim, K., Jang, D., Seo, W., Kim, D., Min, S., & Cho, K. (2021). KLUE: Korean language understanding evaluation. *Advances in Neural Information Processing Systems (NeurIPS)*, 35.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 618–626.

Ulges, A., Borth, D., & Koch, M. (2013). Content analysis meets viewers: Linking concept detection with demographics on YouTube. *International Journal of Multimedia Information Retrieval*, 2, 145–157. https://doi.org/10.1007/s13735-012-0029-x
