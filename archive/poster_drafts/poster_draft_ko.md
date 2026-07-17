# Multimodal Feature Differences in YouTube News Content Across Age Groups:
# Focusing on Thumbnail Images and Title Text
<!-- 제목은 영문 제출 기준. 한국어 번역: 연령대에 따른 유튜브 뉴스 콘텐츠의 멀티모달 피처 차이 분석 -->

<!-- ※ 제출 시 저자 정보 제외 (익명 심사) — accept 후 최종본에 추가 -->
<!-- Last name, First name | Affiliation, Country | email -->
<!-- Yang, Heeseok | Sungkyunkwan University, South Korea -->
<!-- Lee, Yena    | Sungkyunkwan University, South Korea -->
<!-- Lee, Jungbin | Sungkyunkwan University, South Korea -->
<!-- Lee, Jungeun | Sungkyunkwan University, South Korea -->
<!-- Lee, Jangwon | Sungkyunkwan University, South Korea -->

---

## ABSTRACT

YouTube thumbnails and video titles constitute the first moment users encounter content, functioning as critical multimodal signals that determine click decisions. This study examines whether thumbnails and titles exhibit distinct visual and textual style patterns depending on the dominant viewer age group within the same content category. Targeting the "Current Affairs/News/Events" subcategory of YouTube's Society category, we compared two groups: viewers aged ~34 (n=3,701) and 65+ (n=3,057). Three analysis pipelines were applied: (1) extraction of 10 interpretable thumbnail visual features using YOLOv8, EasyOCR, and FER; (2) DINOv2-based thumbnail visual embeddings; and (3) KLUE-RoBERTa-based title semantic embeddings and linguistic feature extraction. Results show that title semantic embeddings achieved the highest age-group discriminability (ROC-AUC=0.881), followed by thumbnail visual embeddings (0.819) and interpretable visual features (0.720). The 65+ group showed significantly higher text density, color saturation, and sad facial expressions, while the ~34 group exhibited lower text density, lighter color tones, and happy/neutral expressions.

**KEYWORDS**: YouTube thumbnails, age groups, multimodal analysis, title text, visual features

---

## INTRODUCTION

YouTube is the world's most widely used video platform, where users encounter dozens of thumbnails and titles daily before deciding what to watch. This first-impression moment is composed of multimodal signals — visual images (thumbnails) combined with text (titles) — and its style reflects deliberate design choices by content creators targeting specific audiences.

Even within the same news topic category, thumbnails and titles of videos predominantly watched by younger (~34) versus older (65+) viewers likely exhibit different visual and linguistic patterns. However, existing research has largely focused on content classification and recommendation systems (Covington et al., 2016), with few studies quantitatively comparing age-differentiated multimodal style differences within the same category.

This study addresses the following research questions through a multimodal approach analyzing both thumbnail images and video titles:

- **RQ1**: Do YouTube video thumbnails and titles show significant visual and textual feature differences across age groups?
- **RQ2**: Which features contribute most to distinguishing between age groups?
- **RQ3**: Can each feature be quantitatively measured and compared?
- **RQ4**: Can age groups be predicted from thumbnail and title features, and which analytical approach achieves the highest discriminability?

---

## THEORETICAL BACKGROUND

**Uses & Gratifications Theory (U&G)** (Katz et al., 1973) posits that people actively select media based on their psychological needs and motivations. Since information-seeking styles and visual preferences vary by age (Sundar & Limperos, 2013), content targeting specific age groups is expected to optimize its visual language and title style accordingly.

**Multimodal Communication Theory** (Kress & van Leeuwen, 2001) holds that information is conveyed simultaneously through multiple semiotic modes — images, text, and color. YouTube thumbnails and titles are canonical multimodal artifacts; analyzing each mode separately allows quantification of their respective contributions to age-group discrimination. Our three pipelines follow this theoretical framework by independently analyzing the image domain (pixel-based features, deep embeddings) and the text domain (linguistic features, semantic embeddings).

---

## METHODOLOGY

### Data Scope

Video metadata and thumbnail images were collected via the YouTube Data API. To control for cross-category noise, we first limited the scope to the **Society category**, selected for its relatively balanced age distribution across all viewer groups (measured by viewer-age entropy). A Vision-Language Model (VLM) was then used to reclassify Society videos into fine-grained subcategories. The **Current Affairs/News/Events** subcategory was selected as the final analysis unit due to its large sample size and balanced distribution between the two target groups (n=6,758; ~34: 3,701, 65+: 3,057).

### Age Group Definition

Two groups were defined based on the dominant viewer age derived from YouTube Analytics audience data: ~34 group and 65+ group.

### Analysis Pipelines

| Pipeline | Input | Method | Output |
|----------|-------|--------|--------|
| ① Thumbnail visual features | Thumbnail image | YOLOv8s-seg, EasyOCR, HSV analysis, MediaPipe, FER | 10 interpretable features (person_ratio, text_ratio, 6 color features, head_pose, expression) |
| ② Thumbnail deep embedding | Thumbnail image | DINOv2-base (768-dim) | 768-dim visual embedding vector |
| ③ Title semantic embedding | Title text | KLUE-RoBERTa fine-tuned (768-dim) | 768-dim language embedding vector |
| ③-b Title linguistic features | Title text | Regex + CLIP similarity | 14 linguistic features (char count, TTR, jamo ratio, emoji count, etc.) |

### Statistical Analysis

Univariate significance of continuous features was tested with the Mann-Whitney U test, and categorical features with the Chi-square test. For multivariate classification, Random Forest and XGBoost were evaluated using StratifiedKFold (k=5) cross-validation. Feature importance was quantified using Random Forest MDI and SHAP values.

---

## RESULTS

### Age Classification Performance by Pipeline (Table 1)

| Pipeline | Classifier | Accuracy | ROC-AUC |
|----------|-----------|----------|---------|
| ① Thumbnail visual features (10 features) | Random Forest | 0.668 | 0.720 |
| ② Thumbnail DINOv2 embeddings (768-dim) | XGBoost | 0.757 | 0.819 |
| ③ Title linguistic features (14 features) | XGBoost | 0.676 | 0.720 |
| **③ Title RoBERTa embeddings (768-dim)** | **XGBoost** | **0.808** | **0.881** |

*(All results based on 5-fold Stratified CV)*

### Key Visual Feature Differences (Table 2)

| Feature | ~34 group | 65+ group | p-value |
|---------|-----------|-----------|---------|
| text_ratio (text coverage) | 0.234 | 0.345 | 2.02e-118 *** |
| color_saturation | 77.8 | 92.4 | 8.90e-59 *** |
| color_brightness | 124.6 | 118.4 | 6.38e-08 *** |
| color_entropy (color diversity) | 2.327 | 2.515 | 2.39e-25 *** |
| expression: sad ratio | 19.0% | 28.4% | χ²=54.33 *** |

*(*** p < 0.001)*

### Random Forest Feature Importance

[Figure 1: rf_feature_importance.png]

Among the 10 interpretable thumbnail features, text_ratio ranked first with an importance score of 0.393, accounting for 39% of total discriminability. The four color features combined for approximately 0.366, meaning text and color together explain 76% of the visual age signal.

### Title Linguistic Feature SHAP (Top 3, Current Affairs/News/Events)

65+ contributors: **jamo_ratio** (Korean consonant/vowel fragmentation) > numbers_count > word_count

~34 contributors: **punctuation_ratio** > emoji_count > TTR (type-token ratio)

---

## DISCUSSION

The results support two key interpretations. First, **title semantic embeddings (ROC-AUC=0.881) achieved the highest discriminability** of all single pipelines, indicating that the semantic framing embedded in titles is strongly associated with target viewer age — even for videos covering identical news events. Second, while thumbnail visual features show lower classification performance (0.720), they offer superior **interpretability**. The 65+ group's pattern of high text density, high saturation, and sadder facial expressions reflects a visual framing strategy that emphasizes the gravity of information, consistent with U&G theory's prediction of age-differentiated information-seeking orientations.

From a multimodal perspective, thumbnails and titles carry largely independent age signals. A fusion model combining both modalities is expected to yield further performance gains beyond any single pipeline.

---

## CONCLUSION

This study demonstrates that YouTube news video thumbnails and titles exhibit quantifiable, statistically significant differences across age groups, even within a controlled content category. The 65+ group favors higher text density, stronger colors, and emotionally heavy imagery; the ~34 group prefers minimal text, lighter tones, and neutral/positive expressions. Title semantics emerge as the single most powerful age discriminator, while interpretable visual features provide complementary explanatory value. These findings contribute to a more principled understanding of age-differentiated content design in digital news environments and lay the groundwork for future multimodal fusion models.

Limitations include restriction to a single subcategory and binary comparison of extreme age groups. Future work will extend to multiple categories, implement late fusion and cross-attention multimodal models, and apply SHAP-based sample-level explanations.

---

## GENERATIVE AI USE

Generative AI tools were used to assist in drafting portions of this submission. The authors take full responsibility for all content, analyses, and claims presented herein.

---

## ACKNOWLEDGEMENTS

*(펀딩 정보가 있으면 여기에 추가. 없으면 섹션 삭제 가능)*

---

## REFERENCES

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. *Proceedings of ICCV*, 9650–9660.

Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks for YouTube recommendations. *Proceedings of the 10th ACM Conference on Recommender Systems*, 191–198.

Geise, S., & Baden, C. (2015). Putting the image back into the frame: Modeling the linkage between visual communication and frame-processing theory. *Communication Theory*, 25(1), 46–69.

Joo, J., & Steinert-Threlkeld, Z. (2022). Image as data: Automated content analysis for visual presentations of political actors and events. *Computational Communication Research*, 4(1).

Katz, E., Blumler, J. G., & Gurevitch, M. (1973). Uses and gratifications research. *The Public Opinion Quarterly*, 37(4), 509–523.

Kress, G., & van Leeuwen, T. (2001). *Multimodal discourse: The modes and media of contemporary communication*. Arnold.

Ksiazek, T. B., Peer, L., & Lessard, K. (2016). User engagement with online news: Conceptualizing interactivity and exploring the relationship between online news videos and user comments. *New Media & Society*, 18(3), 502–520.

Park, S., Moon, H., Kim, J., Lee, J. K., & Kim, J. (2021). KLUE: Korean language understanding evaluation. *Proceedings of NeurIPS*, 35.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of ICML*, 8748–8763.

Sundar, S. S., & Limperos, A. M. (2013). Uses and grats 2.0: New gratifications for new media. *Journal of Broadcasting & Electronic Media*, 57(4), 504–525.

Ternoskiy, A., & Schäfer, M. (2021). Age-differentiated news consumption on YouTube. *Digital Journalism*, 9(4), 498–515.

Zhao, Y., Wu, Y., & Wang, H. (2020). Visual feature extraction for social media content classification by age group. *Journal of the Association for Information Science and Technology*, 71(8), 921–934.

Zhou, J., & Slater, M. D. (2021). Thumbnail design and viewer engagement on YouTube news. *Journalism & Mass Communication Quarterly*, 98(2), 445–462.
