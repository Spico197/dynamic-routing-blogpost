---
layout: distill
title: Your MoE Model Does Not Have to Select K Experts All the Time
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text.
  Do not include math/latex or hyperlinks.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Given Family
#     url: "https://example.com"
#     affiliations:
#       name: An, Affiliation
#   - name: Given Family
#     url: "https://example.com"
#     affiliations:
#       name: An, Affiliation

# must be the exact same name as your blogpost
bibliography: 2026-04-27-dynamic-routing.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What is Dynamic Routing and Why is it Important?
  - name: How to Apply Dynamic Routing?
    subsections:
      - name: Routing Probability Thresholding
      - name: Dynamic Proposer
      - name: Zero-Computational Experts
  - name: Applications
  - name: Challenges
  - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## What is Dynamic Routing and Why is it Important?


## How to Apply Dynamic Routing?


### Routing Probability Thresholding

Top-P<d-cite key="huang2024topp"></d-cite>, DynMoE (cosine)<d-cite key="guo2025dynmoe"></d-cite>

### Dynamic Proposer

Ada-K<d-cite key="yue2025adak"></d-cite>

Should we put these into thresholding? ReMoE, BlockFFN

### Zero-Computational Experts

MoE++<d-cite key="jin2025moepp"></d-cite>, AdaMoE<d-cite key="zeng2024adamoe"></d-cite>


## Applications

LongCat-Flash<d-cite key="meituan2025longcat"></d-cite>, UniMoE-2.0<d-cite key="li2025uni-moe-2.0-omni"></d-cite>


## Challenges

- Performance-Efficiency Tradeoff
- Efficient Implementations: infra, kernel, frameworks
- Expert Load Balancing
- Sparsity controlling

## Conclusion


