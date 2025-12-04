---
layout: distill
title: Your MoE Model Does Not Have to Select K Experts All the Time
description: Standard Mixture-of-Experts (MoE) models rely on fixed top-k routing, applying uniform computation across tokens regardless of their complexity.
  This rigidity often leads to suboptimal efficiency and performance.
  Dynamic routing addresses this by adaptively selecting the optimal number of experts for each token.
  This post introduces the principles of dynamic routing and reviews key techniques for flexible expert allocation.
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

Supposing we have 8 cargo vans and we are running a moving company.
The company has a famous regulation: **each time we receive an order, we must assign 2 vans to deliver customers' belongings.**<d-footnote>Similar to Top-2 expert selection out of 8 ones in MoE models, and each order represents a token.</d-footnote>
Ideally, we should assign vans according to the order size.
For example, we can assign 1 van to small orders and more vans to large orders.
However, due to the regulation, we have to assign 2 vans to each order, even if the order is small or large.

{% include figure.liquid path="assets/img/2026-04-27-dynamic-routing/two-van-regulation.jpg" class="img-fluid" %}

In the first case (left) of the above figure, it is a waste of money to use 2 vans for a suitcase and a teddy bear.
In the last case (right), it may consume more time to deliver the order because the vans may need to make more trips to deliver all the belongings.<d-footnote>More trips means more time consumption if no stuff is abandoned due to limited space, similar to disabling token dropping in MoE training. If tokens are dropped, there may induce a performance degradation.<d-cite key="gale2023megablocks"></d-cite></d-footnote>

Normally, customers would like to have their belongings delivered cheaply and quickly. So here comes the questions:
1. **Could we dynamically assign vans according to the order size?**
2. **If we could dynamically assign vans, is it possible to reduce the overall cost and time consumption?**

These are exactly the problems that dynamic routing aims to address in Mixture-of-Experts (MoE) models.
For each token, a conventional router selects top-k experts for computation, where $k$ is a fixed number.
Let's say there are 8 experts in total, the router always selects top-2 experts for computation.

```mermaid
graph TD
    T[Token Input] --> R{Router}
    
    R -.-> E1[Expert 1]
    R -- "Prob: 0.4 (Top-1)" --> E2[Expert 2]
    R -.-> E3[Expert 3]
    R -.-> E4[Expert 4]
    R -.-> E5[Expert 5]
    R -- "Prob: 0.3 (Top-2)" --> E6[Expert 6]
    R -.-> E7[Expert 7]
    R -.-> E8[Expert 8]
    
    E2 --> Out[Output]
    E6 --> Out

    style E2 fill:#ff9999,stroke:#333,stroke-width:2px
    style E6 fill:#ff9999,stroke:#333,stroke-width:2px
```

However, different tokens may have different expert selection preferences.
For example, for complex reasoning tasks, a token may be forwarded to more experts for better performance.
While in simple tasks, a token may be forwarded to fewer experts for better efficiency.

```mermaid
graph LR
    subgraph Simple [Simple Token Activates Few Experts]
        direction LR
        S_E1[Expert 1]:::inactive ~~~ S_E2[Expert 2]:::active ~~~ S_E3[Expert 3]:::inactive ~~~ S_E4[Expert 4]:::inactive ~~~ S_E5[Expert 5]:::inactive ~~~ S_E6[Expert 6]:::inactive ~~~ S_E7[Expert 7]:::inactive ~~~ S_E8[Expert 8]:::inactive
    end

    classDef active fill:#ff9999,stroke:#333,stroke-width:2px;
    classDef inactive fill:#fff,stroke:#999,stroke-dasharray: 3 3;
```

```mermaid
graph LR
    subgraph Complex [Complex Token Activates Many Experts]
        direction LR
        C_E1[Expert 1]:::active ~~~ C_E2[Expert 2]:::active ~~~ C_E3[Expert 3]:::active ~~~ C_E4[Expert 4]:::active ~~~ C_E5[Expert 5]:::inactive ~~~ C_E6[Expert 6]:::inactive ~~~ C_E7[Expert 7]:::inactive ~~~ C_E8[Expert 8]:::inactive
    end

    classDef active fill:#ff9999,stroke:#333,stroke-width:2px;
    classDef inactive fill:#fff,stroke:#999,stroke-dasharray: 3 3;
```

Here, let's transform the above moving company's problems into the MoE context:
1. **Could a router dynamically select the number of experts for each token?**
  - Yes, there are some techniques to dynamically select the number of experts for each token!
2. **Does dynamic routing help improve performance and reduce time consumption?**
  - Yes, dynamic routing could be both effective and efficient!

In this post, we will provide a brief introduction to MoE dynamic routing techniques and their applications in LLMs.
Now let's dive into the details!

## How to Apply Dynamic Routing?

There are three main categories of dynamic routing techniques:
1. **Thresholding**: Instead of selecting top-k experts for each token, the router selects experts based on a threshold.
2. **Dynamic Proposer**: An additional proposer module for predicting the number of activated experts for each token.
3. **Zero-Computational Experts**: Some special experts are regarded as placeholders for routing, but not used for computation.

Here, we will introduce these three categories of dynamic routing techniques in detail.

### Thresholding

Top-P<d-cite key="huang2024topp"></d-cite>, DynMoE (cosine)<d-cite key="guo2025dynmoe"></d-cite>
Should we put these into thresholding? ReMoE<d-cite key="wang2025remoe"></d-cite>, BlockFFN<d-cite key="song2025blockffn"></d-cite>

### Dynamic Proposer

Ada-K<d-cite key="yue2025adak"></d-cite>

### Zero-Computational Experts

MoE++<d-cite key="jin2025moepp"></d-cite>, AdaMoE<d-cite key="zeng2024adamoe"></d-cite>


## Applications

- LongCat-Flash<d-cite key="meituan2025longcat"></d-cite>
- UniMoE-2.0<d-cite key="li2025uni-moe-2.0-omni"></d-cite>
- BlockFFN<d-cite key="song2025blockffn"></d-cite>


## Challenges

- Performance-Efficiency Tradeoff
- Efficient Implementations: infra, kernel, frameworks
- Expert Load Balancing
- Sparsity controlling

## Conclusion


