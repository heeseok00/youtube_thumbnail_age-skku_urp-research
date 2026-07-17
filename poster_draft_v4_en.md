# Age-Differentiated Multimodal Style in YouTube Thumbnails and Titles:
# A Feature-Level Analysis of Visual and Textual Signals

<!-- Author block intentionally left blank for anonymous review. -->

---

## ABSTRACT

This study examines how YouTube thumbnails and titles in the current affairs category differ across two dominant-viewer-age groups, ~34 (n=3,701) and 65+ (n=3,057). We apply Grad-CAM to localize discriminative thumbnail regions and validate them with deletion/insertion experiments, extract and test interpretable visual features, analyze titles through formal and semantic text analysis, and use image captioning to compare the overall meaning thumbnails convey. Multimodal late fusion achieves AUC=0.905. The 65+ group's thumbnails show denser on-image text, more saturated colors, and a higher prevalence of negative emotion, paired with relational and narrative title language; the ~34 group's thumbnails show person-centric visuals and lighter tones, paired with informational, format-marked titles. These findings demonstrate that age-differentiated content consumption leaves measurable, interpretable traces across the visual and textual style of thumbnails and titles.

**KEYWORDS**: YouTube thumbnails; age-differentiated consumption; multimodal analysis; interpretability; content style

---

## INTRODUCTION

YouTube users largely decide what to watch from a single first-impression artifact: the thumbnail–title pair. The features of thumbnails and titles are known to influence viewer engagement (Cui et al., 2024), and computational analysis has further shown that thumbnail aesthetics vary measurably across different audience segments (Limpijankit & Kender, 2024). At the same time, viewer demographics on YouTube have been linked to the characteristics of the content people watch (Ulges et al., 2013). Yet little is known about how the visual and textual style of thumbnails and titles varies across age groups within the same content category.

This study addresses that gap in the current affairs category, comparing ~34- and 65+-leaning audiences. We ask: *What features distinguish the thumbnails and titles of videos each age group prefers, and how do they differ?* Grad-CAM first locates discriminative thumbnail regions (e.g., person and on-image text) and guides extraction of interpretable visual features, which we test for significant age-group differences. In parallel, we analyze titles through formal and semantic text analysis, and use image captioning to examine how the overall meaning conveyed by thumbnails differs across age groups.

---

## DATA

We used Vling (vling.net), a YouTube analytics platform, to obtain channel lists ranked by dominant viewer age group. Using channel IDs, we collected thumbnail images and video metadata (titles, upload dates, view counts) through the YouTube Data API. To reduce topic- and category-driven noise, videos were first organized by Vling's platform labels and then reclassified using Qwen2.5-7B-Instruct; the current affairs category was retained for its sample size and age balance. Following YouTube Analytics dominant-age labels, we define two groups: channels whose dominant viewers fall within the 13–34 age brackets (hereafter ~34; n=3,701) and those within the 65+ bracket (n=3,057), totaling n=6,758. This binary grouping was chosen to balance class sizes and maximize the age contrast for classification.

---

## ANALYSIS FRAMEWORK

We analyze three modalities—thumbnail appearance, video title, and thumbnail semantics—following a shared logic: deep models first locate what differs between age groups, and interpretable features then confirm and describe those differences.

For thumbnails, we train a DINOv2-based binary classifier (Oquab et al., 2024) to distinguish the two age groups, then apply Grad-CAM (Selvaraju et al., 2017) to visualize which thumbnail regions the model attends to when raising each age-group score. Because Grad-CAM computes the gradient of each class score independently, the ~34 and 65+ heatmaps reflect separate visual priorities rather than simple inverses of one another. To confirm that the highlighted regions genuinely carry the age signal, we run deletion and insertion experiments—masking text, person, and background areas detected by EasyOCR and YOLOv8—and measure each region's impact on prediction confidence as AUC. These validated regions then guide the extraction of ten interpretable visual features (e.g., person_ratio, text_ratio, color saturation, brightness, and entropy, head pose, dominant facial expression), each statistically tested for group differences.

For titles, we apply lexical and structural attribution (XGBoost + SHAP; Lundberg & Lee, 2017), morpheme-level attribution via a fine-tuned KLUE-RoBERTa classifier with LIME (Park et al., 2021; Ribeiro et al., 2016), and topic-level embedding clustering. For thumbnail semantics, LLaVA 1.5 (Liu et al., 2023) generates one image description per thumbnail; SHAP-attributed terms are grouped into semantic clusters to reveal what each age group's thumbnails depict.

Finally, predictions from all three modalities are combined via **probability-averaging late fusion**, with the text-to-image weight optimized by grid search.

---

## RESULTS AND DISCUSSION

The analyses reveal a consistent multimodal contrast between the two age groups. Late fusion of all three modalities achieves AUC=0.905 (text:image=0.6:0.4), indicating that the age signal is jointly encoded across thumbnail appearance and title language. Table 1 summarizes the key discriminating signals.

### Table 1. Key Discriminating Signals Across Modalities

| Modality | Feature / Signal | ~34 group | 65+ group | Evidence |
|---|---|:---:|:---:|:---:|
| Visual | text_ratio (MDI=0.393) | 0.234 | 0.345 | MW *p*=2.02e-118 |
| | person_count | 1 person (26.4%) | 3+ persons (39.0%) | chi² *p*=8.26e-7 |
| | color_saturation | 77.8 | 92.4 | MW *p*=8.90e-59 |
| | color_brightness | 124.6 | 118.4 | MW *p*=6.38e-8 |
| | color_entropy | 2.327 | 2.515 | MW *p*=2.39e-25 |
| | expression_dominant | positive / neutral | negative | chi²=66.99 |
| Title | key terms (KLUE-RoBERTa + LIME) | incident, accident, real, actual | shock, twist, touching, tears | KLUE-RoBERTa + LIME |
| | structural signals | emoji use↑, CLIP similarity↑ | word count↑, noun ratio↑, curiosity↑ | XGBoost SHAP |
| Semantics | SHAP tokens | room, door, smiling, joyful | mountain, street, yelling, crying | LLaVA SHAP |

### Figure 1. Grad-CAM heatmaps: ~34 (top) vs. 65+ (bottom).

> *(Place here a two-row Grad-CAM grid: top row = high-confidence ~34-leaning thumbnails; bottom row = high-confidence 65+-leaning thumbnails.)*

*Figure 1. Heatmaps indicate where each age-specific score increases most rapidly.*

Grad-CAM analysis (Figure 1) suggests a clear structural contrast at the visual level: in ~34-leaning thumbnails, contributions concentrate on people and central visual content, while in 65+-leaning thumbnails, they align with large, high-contrast on-image text. ROI experiments further indicate that the ~34 signal appears distributed across the image composition, whereas the 65+ signal appears to depend on the joint presence of text, person, and background—removing any single region noticeably weakens prediction. The interpretable visual features (Table 1) corroborate this: 65+ thumbnails show higher text coverage, color saturation, and color entropy, and a higher proportion of negatively valenced facial expressions relative to ~34 thumbnails.

Title analysis reveals a parallel contrast. Attribution analyses suggest that 65+-leaning titles foreground emotionally charged language (e.g., *shock, twist, touching, tears*), while ~34-leaning titles foreground fact-based terms (e.g., *incident, accident, real, actual*). Structural features indicate that longer, noun-heavy titles with curiosity-evoking expressions lean toward 65+, while emoji use and stronger thumbnail–title visual alignment (Radford et al., 2021) lean toward ~34. Thumbnail semantics reinforce this axis: ~34-leaning thumbnails associate with indoor structures and positive affect (e.g., *room, door, smiling, joyful*), while 65+-leaning thumbnails associate with broader outdoor scenes and negative emotional expressions (e.g., *mountain, street, yelling, crying*).

Together, these patterns suggest that age-differentiated content consumption leaves interpretable, modality-consistent traces in both thumbnail appearance and title language. The contribution of this work is a reproducible, attribution-driven procedure for surfacing such style differences. Findings appear bounded by the current affairs category; dominant-age labels operate at the channel-aggregate level, and attribution methods explain model behavior rather than directly measure viewer cognition.

---

<!-- ============================================================ -->
<!-- The section below does not count toward the 2-page poster limit. -->
<!-- ============================================================ -->

## GENERATIVE AI USE

We employed generative AI tools/services for the following purpose(s): thumbnail image captioning with LLaVA 1.5, fine-grained subcategory classification of video titles with Qwen2.5-7B-Instruct, and writing assistance during manuscript drafting. We evaluated the output by manual inspection and author review and revision. The authors assume all responsibility for the content of this submission.

---

## REFERENCES

Cui, G., Chung, Y., Peng, L., & Wang, Q. (2024). Clicks for money: Predicting video views through a sentiment analysis of titles and thumbnails. *Journal of Business Research*, *183*, 114849.

Limpijankit, M., & Kender, J. (2024). Detecting cultural differences in news video thumbnails via computational aesthetics. *Proceedings of the International AAAI Conference on Web and Social Media*. https://doi.org/10.36190/2024.61

Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. *Advances in Neural Information Processing Systems*, *36*.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, *30*, 4765–4774.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., & Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. *Transactions on Machine Learning Research*.

Park, S., Moon, J., Kim, S., Cho, W. I., Han, J., Park, J., Chang, M., Lim, H., Oh, S., Park, J., Shin, J., Kim, S., Park, K., Oh, N., Kim, K., Jang, D., Seo, W., Kim, D., Min, S., & Cho, K. (2021). KLUE: Korean language understanding evaluation. *Advances in Neural Information Processing Systems*, *35*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*, 8748–8763.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 618–626.

Ulges, A., Borth, D., & Koch, M. (2013). Content analysis meets viewers: Linking concept detection with demographics on YouTube. *International Journal of Multimedia Information Retrieval*, *2*(3), 145–157.
