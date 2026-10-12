# Flagged Fragment Audit

Manual, post-hoc audit of arm-A eligible fragments at t = 0.25.

| Page | Chunk ID | Ratio | Length | Class | Justification |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | doc_self-attention-transformers-2023_chunk_001 | 0.235294 | 74 | truncated prose | classified from the complete fragment text |

```text
We’ll use the notation, wt ∼softmax( f (w1:t−1)), (1) to mean that under a
```

| 1 | doc_self-attention-transformers-2023_chunk_002 | 0.051282 | 303 | diagram/figure labels | classified from the complete fragment text |

```text
Add & Norm Multi-Head Attention Repeat for number of encoder blocks Add & Norm Add & Norm Masked Multi- Head Attention Feed-Forward Block Add & Norm Add Position Embeddings Multi-Head Attention Embeddings Block Decoder Inputs Add Position Embeddings Embeddings Encoder Inputs Transformer Encoder-Decoder
```

| 3 | doc_self-attention-transformers-2023_chunk_005 | 0.148936 | 266 | other | mixed prose and diagram sequence |

```text
(Serial meaning one-after-the- other.) 1 2 3 4 5 0 0 0 0 0 Zuko made his uncle tea As GPUs (and later, other accelerators like Tensor Processing Units (TPUs) became more powerful and researchers wanted to take fuller advantage of them, this dependence in time became
```

| 5 | doc_self-attention-transformers-2023_chunk_009 | 0.000000 | 69 | equation/math | classified from the complete fragment text |

```text
∑ hi = αij vj (weighted average) Self-Attention scalar vector qi =Qxi
```

| 5 | doc_self-attention-transformers-2023_chunk_010 | 0.033333 | 166 | equation/math | classified from the complete fragment text |

```text
∑ hi = αij vj (weighted average) Self-Attention scalar vector qi =Qxi (query) (weights) α qi v1:n =Vx1:n k1:n =Kx1:n (key) (value) with deep learning 5 Tkj -> softmax
```

| 8 | doc_self-attention-transformers-2023_chunk_016 | 0.210526 | 280 | other | prose mixed with equation/header material |

```text
2.5 Summary of a minimal self-attention architecture Our minimal self-attention architecture has (1) the self-attention operation, (2) position representations, (3) elementwise nonlinearities, and (4) future masking (in the context of language modeling.) with deep learning 8 (21)
```

| 9 | doc_self-attention-transformers-2023_chunk_017 | 0.200000 | 179 | truncated prose | classified from the complete fragment text |

```text
What we’ll present now, multi-head self-attention, intuitively applies self-attention multiple times at once, each with different key, query, and value transformations of the same
```

| 9 | doc_self-attention-transformers-2023_chunk_018 | 0.148148 | 346 | diagram/figure labels | classified from the complete fragment text |

```text
Words in each row have words in the future masked out (e.g., “Zuko” can only attend to “Zuko”, while “made” can attend to “Zuko” and “made”.) Probabilities Softmax Linear Add & Norm Feed-Forward Repeat for number of encoder blocks Add & Norm Masked Multi- Head Attention Block Add Position Embeddings Embeddings Decoder Inputs Transformer Decoder
```

| 10 | doc_self-attention-transformers-2023_chunk_019 | 0.093750 | 150 | equation/math | classified from the complete fragment text |

```text
(26) h1:n = softmax(x1:nQK⊤x⊤ Here’s a diagram showing the matrix ops: α softmax n (x1:nK)T n = x1:nQ d n When we perform multi-head self-attention in
```

| 11 | doc_self-attention-transformers-2023_chunk_022 | 0.074074 | 95 | equation/math | classified from the complete fragment text |

```text
, n} as v u u t1 d ∑ j=1 d ∑ j=1 ˆµi = 1 (hij −µi)2, (27) hij ˆσi = d d with deep learning 11 k
```

| 12 | doc_self-attention-transformers-2023_chunk_024 | 0.185185 | 125 | equation/math | classified from the complete fragment text |

```text
So, we normalize the dot products by d to stop this scaling: α = softmax(x1:nQK⊤x⊤ ) ∈Rn×n (32) 1:n √ d with deep learning 12
```

| 13 | doc_self-attention-transformers-2023_chunk_025 | 0.230769 | 71 | truncated prose | classified from the complete fragment text |

```text
Famous examples of this are GPT- 2 [Radford et al., 2019], GPT-3 [Brown
```

| 13 | doc_self-attention-transformers-2023_chunk_026 | 0.103448 | 218 | diagram/figure labels | classified from the complete fragment text |

```text
with deep learning 13 Probabilities Softmax Linear Add & Norm Feed-Forward Repeat for number of encoder blocks Add & Norm Multi-Head Attention Block Add Position Embeddings Embeddings Encoder Inputs Transformer Encoder
```

| 14 | doc_self-attention-transformers-2023_chunk_028 | 0.157895 | 120 | bibliography | classified from the complete fragment text |

```text
D., Dhari- wal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, with deep learning 14
```

| 15 | doc_self-attention-transformers-2023_chunk_029 | 0.069767 | 325 | diagram/figure labels | classified from the complete fragment text |

```text
Add & Norm Multi-Head Attention Repeat for number of encoder blocks Add & Norm Add & Norm Masked Multi- Head Attention Feed-Forward Block Add & Norm Add Position Embeddings Multi-Head Attention Embeddings Block Decoder Inputs Add Position Embeddings Embeddings Encoder Inputs Transformer Encoder-Decoder with deep learning 15
```

| 17 | doc_self-attention-transformers-2023_chunk_034 | 0.000000 | 163 | bibliography | classified from the complete fragment text |

```text
W., Tae, J., Phang, J., Press, O., Li, C., Narayanan, D., Bourfoune, H., Casper, J., Rasley, J., Ryabinin, M., Mishra, M., Zhang, M., Shoeybi, M., Peyrounette, M.,
```

| 17 | doc_self-attention-transformers-2023_chunk_036 | 0.000000 | 82 | bibliography | classified from the complete fragment text |

```text
S., Mishra, S., Kiblawi, S., Ott, S., Sang-aroonsiri, S., Kumar, S., Schweter, S.,
```

| 17 | doc_self-attention-transformers-2023_chunk_037 | 0.047619 | 102 | bibliography | classified from the complete fragment text |

```text
S., Venkatraman, Y., Xu, Y., Xu, Y., Xu, Y., Tan, Z., Xie, Z., Ye, Z., Bras, M., with deep learning 17
```

Class counts: diagram/figure labels=4, equation/math=5, bibliography=4, truncated prose=3, other=2 (18 total).
The earlier equation/math count of 6 was an arithmetic error; no fragment changed class.

Precision: 4/18 = 0.2222

Uncertain fragments:
- doc_self-attention-transformers-2023_chunk_005: mixed prose and diagram sequence
- doc_self-attention-transformers-2023_chunk_016: prose mixed with equation/header material

## Audited Label Share

This section is manual audit, post-hoc. It counts only fragments classified as diagram/figure labels.

| Question | Arm | Fragments | Raw share | Audited label share |
| --- | --- | --- | ---: | ---: |
| q003 | A | doc_self-attention-transformers-2023_chunk_029 | 0.404229 | 0.404229 |
- q003 A: doc_self-attention-transformers-2023_chunk_029, length=325, class=diagram/figure labels, class_basis=manual re-read of this arm's fragment, text=Add & Norm Multi-Head Attention Repeat for number of encoder blocks Add & Norm Add & Norm Masked Multi- Head Attention Feed-Forward Block Add & Norm Add Position Embeddings Multi-Head Attention Embeddings Block Decoder Inputs Add Position Embeddings Embeddings Encoder Inputs Transformer Encoder-Decoder with deep learning 15
| q004 | A | doc_self-attention-transformers-2023_chunk_018 | 0.431960 | 0.431960 |
- q004 A: doc_self-attention-transformers-2023_chunk_018, length=346, class=diagram/figure labels, class_basis=manual re-read of this arm's fragment, text=Words in each row have words in the future masked out (e.g., “Zuko” can only attend to “Zuko”, while “made” can attend to “Zuko” and “made”.) Probabilities Softmax Linear Add & Norm Feed-Forward Repeat for number of encoder blocks Add & Norm Masked Multi- Head Attention Block Add Position Embeddings Embeddings Decoder Inputs Transformer Decoder
| q008 | A | doc_self-attention-transformers-2023_chunk_018 | 0.300870 | 0.300870 |
- q008 A: doc_self-attention-transformers-2023_chunk_018, length=346, class=diagram/figure labels, class_basis=manual re-read of this arm's fragment, text=Words in each row have words in the future masked out (e.g., “Zuko” can only attend to “Zuko”, while “made” can attend to “Zuko” and “made”.) Probabilities Softmax Linear Add & Norm Feed-Forward Repeat for number of encoder blocks Add & Norm Masked Multi- Head Attention Block Add Position Embeddings Embeddings Decoder Inputs Transformer Decoder
| q010 | A | doc_self-attention-transformers-2023_chunk_017 | 0.387446 | 0.000000 |
- q010 A: doc_self-attention-transformers-2023_chunk_017, length=179, class=truncated prose, class_basis=manual re-read of this arm's fragment, text=What we’ll present now, multi-head self-attention, intuitively applies self-attention multiple times at once, each with different key, query, and value transformations of the same
| q003 | B | doc_self-attention-transformers-2023_chunk_029 | 0.387468 | 0.387468 |
- q003 B: doc_self-attention-transformers-2023_chunk_029, length=303, class=diagram/figure labels, class_basis=manual re-read of this arm's fragment, text=Add & Norm Multi-Head Attention Repeat for number of encoder blocks Add & Norm Add & Norm Masked Multi- Head Attention Feed-Forward Block Add & Norm Add Position Embeddings Multi-Head Attention Embeddings Block Decoder Inputs Add Position Embeddings Embeddings Encoder Inputs Transformer Encoder-Decoder
| q004 | B | doc_self-attention-transformers-2023_chunk_018 | 0.431960 | 0.431960 |
- q004 B: doc_self-attention-transformers-2023_chunk_018, length=346, class=diagram/figure labels, class_basis=manual re-read of this arm's fragment, text=Words in each row have words in the future masked out (e.g., “Zuko” can only attend to “Zuko”, while “made” can attend to “Zuko” and “made”.) Probabilities Softmax Linear Add & Norm Feed-Forward Repeat for number of encoder blocks Add & Norm Masked Multi- Head Attention Block Add Position Embeddings Embeddings Decoder Inputs Transformer Decoder
| q008 | B | doc_self-attention-transformers-2023_chunk_018 | 0.300870 | 0.300870 |
- q008 B: doc_self-attention-transformers-2023_chunk_018, length=346, class=diagram/figure labels, class_basis=manual re-read of this arm's fragment, text=Words in each row have words in the future masked out (e.g., “Zuko” can only attend to “Zuko”, while “made” can attend to “Zuko” and “made”.) Probabilities Softmax Linear Add & Norm Feed-Forward Repeat for number of encoder blocks Add & Norm Masked Multi- Head Attention Block Add Position Embeddings Embeddings Decoder Inputs Transformer Decoder
| q010 | B | doc_self-attention-transformers-2023_chunk_017 | 0.387446 | 0.000000 |
- q010 B: doc_self-attention-transformers-2023_chunk_017, length=179, class=truncated prose, class_basis=manual re-read of this arm's fragment, text=What we’ll present now, multi-head self-attention, intuitively applies self-attention multiple times at once, each with different key, query, and value transformations of the same
| q003 | C | doc_self-attention-transformers-2023_chunk_029 | 0.408907 | 0.408907 |
- q003 C: doc_self-attention-transformers-2023_chunk_029, length=303, class=diagram/figure labels, class_basis=manual re-read of this arm's fragment, text=Add & Norm Multi-Head Attention Repeat for number of encoder blocks Add & Norm Add & Norm Masked Multi- Head Attention Feed-Forward Block Add & Norm Add Position Embeddings Multi-Head Attention Embeddings Block Decoder Inputs Add Position Embeddings Embeddings Encoder Inputs Transformer Encoder-Decoder
| q004 | C | - | 0.000000 | 0.000000 |
| q008 | C | - | 0.000000 | 0.000000 |
| q010 | C | doc_self-attention-transformers-2023_chunk_017 | 0.387446 | 0.000000 |
- q010 C: doc_self-attention-transformers-2023_chunk_017, length=179, class=truncated prose, class_basis=manual re-read of this arm's fragment, text=What we’ll present now, multi-head self-attention, intuitively applies self-attention multiple times at once, each with different key, query, and value transformations of the same
