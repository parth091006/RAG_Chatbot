# Adjudication Worksheet

This worksheet is read-only evidence for adjudicating disputed relevant chunks.
It uses ARM A (current) chunk text only and contains no retrieval or reranking results.

## q001

Question: What is self-attention?

Chunks:
### doc_self-attention-transformers-2023_chunk_007
Cited by: retrieval_questions_ground_truth.json

```text
the number of intermediate computations—matrix multiplies and nonlinearities, for example—that separate chef from is scales with the number of words between them. We visualize this in Figure 2. 1 2 3 4 5 0 Zuko made his uncle tea Intuitively, researchers believe there’s an issue with linear inter- action distance because it can be difficult for networks to precisely “recall” the presence of a word when a large number of operations occur after observing that word. This can make it difficult to learn how distant words should impact the representation of the current word. This notion of direct interaction between elements of a sequence might remind you of the attention mechanism [Bahdanau et al., 2014] in machine translation. In that context, while generating a translation, we learned how to look back into the source sequence once per token of the translation. In this note, we’ll present an entire replacement for recurrent neural networks just based on attention. This will solve both the parallelization issues and the linear interaction distance issues with recurrent neural networks. 2 A minimal self-attention architecture Attention, broadly construed, is a method for taking a query, and softly looking up information in a key-value store by picking the value(s) of the key(s) most like the query. By “picking” and “most like,” we mean averaging overall values, putting more weight on those which correspond to the keys more like the query. In self- attention, we mean that we use the same elements to help us define the querys
```

### doc_self-attention-transformers-2023_chunk_008
Cited by: retrieval_questions.json, retrieval_questions_ground_truth.json

```text
the value(s) of the key(s) most like the query. By “picking” and “most like,” we mean averaging overall values, putting more weight on those which correspond to the keys more like the query. In self- attention, we mean that we use the same elements to help us define the querys as we do the keys and values. In this section, we’ll discuss how to develop contextual representa- tions with methods wherein the main mechanism for contextualiza- tion is not recurrence, but attention. 2.1 The key-query-value self-attention mechanism There are many forms of self-attention; the form we’ll discuss here is currently the most popular. It’s called key-query-value self-attention. with deep learning 4 Figure 2: A RNN unrolled in time. The rectangles are intermediate states of the RNN (e.g., the first row is the embedding layer, and the second row is the RNN hidden state at each time step) and the number in the rectangle is roughly the number of operations separating lexical information of the word tea from each intermediate state.
```

### doc_self-attention-transformers-2023_chunk_009
Cited by: retrieval_questions.json, retrieval_questions_ground_truth.json

```text
Consider a token xi in the sequence x1:n. From it, we define a query qi = Qxi, for matrix Q ∈Rd×d. Then, for each token in the sequence xj ∈{x1 . . . , xn}, we define both a key and a value similarly, with two other weight matrices: kj = Kxj, and vj = Vxj for K ∈Rd×d and V ∈Rd×d. Our contextual representation hi of xi is a linear combination (that is, a weighted sum) of the values of the sequence, n ∑ j=1 αijvj, (7) hi = where the weights, these αij control the strength of contribution of each vj. Going back to our key-value store analogy, the αij softly se- lects what data to look up. We define these weights by computing the affinities between the keys and the query, q⊤ i kj, and then computing the softmax over the sequence: αij = exp(q⊤ i kj) i kj′) (8) j′=1 exp(q⊤ ∑n Intuitively, what we’ve done by this operation is take our element xi and look in its own sequence x1:n to figure out what information (in an informal sense,) from what other tokens, should be used in representing xi in context. The use of matrices K, Q, V intuitively allow us to use different views of the xi for the different roles of key, query, and value. We perform this operation to build hi for all i ∈{1, . . . , n}. ∑ hi = αij vj (weighted average) Self-Attention scalar vector qi =Qxi
```

### doc_self-attention-transformers-2023_chunk_010
Cited by: retrieval_questions.json

```text
K, Q, V intuitively allow us to use different views of the xi for the different roles of key, query, and value. We perform this operation to build hi for all i ∈{1, . . . , n}. ∑ hi = αij vj (weighted average) Self-Attention scalar vector qi =Qxi (query) (weights) α qi v1:n =Vx1:n k1:n =Kx1:n (key) (value) with deep learning 5 Tkj -> softmax
```

Decision: ______

## Arm A Chunk Index

Document-order index for all current arm-A chunks. No scores, rankings, retrieval, or answer information.

| ID | Page | First 150 characters |
| --- | ---: | --- |
| doc_self-attention-transformers-2023_chunk_001 | 1 | Summary. This note motivates moving away from recurrent archi- tectures in NLP, introduces self-attention, and builds a minimal self- attention-based |
| doc_self-attention-transformers-2023_chunk_002 | 1 | also overload w1:n to be a matrix of one-hot vectors, w1:n ∈Rn×\|V\|. We’ll use w ∈V to represent an arbitrary vocabulary element, and wi ∈V to pick out |
| doc_self-attention-transformers-2023_chunk_003 | 2 | being normalized over, and it should be interpreted as follows. If A is a tensor of shape Rℓ,d, the softmax is computed as follows: softmax(A)i,j = ex |
| doc_self-attention-transformers-2023_chunk_004 | 2 | representation hi that represents wi but is a function of the entire sequence x1:n (or a prefix x1:i, as in the case of language modeling.). A non-con |
| doc_self-attention-transformers-2023_chunk_005 | 3 | sequence index (often called the dependence on “time”) highlighted in Equation 4. Parallelization issues with dependence on the sequence index. Modern |
| doc_self-attention-transformers-2023_chunk_006 | 3 | of serial dependencies. (Serial meaning one-after-the- other.) 1 2 3 4 5 0 0 0 0 0 Zuko made his uncle tea As GPUs (and later, other accelerators like |
| doc_self-attention-transformers-2023_chunk_007 | 4 | the number of intermediate computations—matrix multiplies and nonlinearities, for example—that separate chef from is scales with the number of words b |
| doc_self-attention-transformers-2023_chunk_008 | 4 | the value(s) of the key(s) most like the query. By “picking” and “most like,” we mean averaging overall values, putting more weight on those which cor |
| doc_self-attention-transformers-2023_chunk_009 | 5 | Consider a token xi in the sequence x1:n. From it, we define a query qi = Qxi, for matrix Q ∈Rd×d. Then, for each token in the sequence xj ∈{x1 . . .  |
| doc_self-attention-transformers-2023_chunk_010 | 5 | K, Q, V intuitively allow us to use different views of the xi for the different roles of key, query, and value. We perform this operation to build hi  |
| doc_self-attention-transformers-2023_chunk_011 | 6 | 2.2 Position representations Consider the sequence the oven cooked the bread so. This is a different sequence than the bread cooked the oven so, as yo |
| doc_self-attention-transformers-2023_chunk_012 | 6 | is, αso,0 = exp(q⊤ sokthe) exp(q⊤ sokthe) + · · · + exp(q⊤ sokbread). (11) So, α ∈R5 are our weights, and we compute the weighted average in Equation  |
| doc_self-attention-transformers-2023_chunk_013 | 7 | We then simply add embedded representation of the position of a word to its word embedding: ˜xi = Pi + xi (12) and perform self-attention as we otherw |
| doc_self-attention-transformers-2023_chunk_014 | 7 | it’s odd that this works; but interesting! 2.3 Elementwise nonlinearity Imagine if we were to stack self-attention layers. Would this be sufficient fo |
| doc_self-attention-transformers-2023_chunk_015 | 8 | In practice, after a layer of self-attention, it’s common to apply feed-forward network independently to each word representation: hFF = W2 ReLU(W1hse |
| doc_self-attention-transformers-2023_chunk_016 | 8 | words wt, . . . , wn.) In a Transformer, there’s nothing explicit in the self-attention weight α that says not to look at indices j > i when represent |
| doc_self-attention-transformers-2023_chunk_017 | 9 | Zuko made his uncle tea −∞ −∞ −∞ −∞ Zuko −∞ −∞ −∞ made −∞ −∞ his −∞ uncle tea Intuitively, these are the biggest components to understand. How- ever,  |
| doc_self-attention-transformers-2023_chunk_018 | 9 | the key-query dot products in order to carefully average two or more things. In Assignment 5, you’ll work through a bit of this intuition more careful |
| doc_self-attention-transformers-2023_chunk_019 | 10 | We then perform self-attention with each head: n ∑ j=1 h(ℓ) α(ℓ) ij v(ℓ) j (22) i = exp(q(ℓ)⊤ i k(ℓ) j ) α(ℓ) ij = j′=1 exp(q(ℓ)⊤ i k(ℓ) ∑n j′ ) Note  |
| doc_self-attention-transformers-2023_chunk_020 | 10 | compute our weights in matrix operations: α = softmax(x1:nQK⊤x⊤ 1:n) ∈Rn×n (25) and then compute the self-attention operation for all x1:n via: 1:n)x1 |
| doc_self-attention-transformers-2023_chunk_021 | 11 | then transpose the matrices to Rk,n,d/k, which intuitively should look like k sequences of length n and dimensionality d/k. This allows us to perform  |
| doc_self-attention-transformers-2023_chunk_022 | 11 | (2) normalizes the activations with respect to those estimates, while (3) optionally learning (as parameters) an elementwise additive bias and multipl |
| doc_self-attention-transformers-2023_chunk_023 | 12 | where (as a reminder), ˆµi and ˆσi are scalars, and we compute the layer norm as LN(hi) = hi −ˆµi , (28) ˆσi where we’ve broadcasted the ˆµi and ˆσi a |
| doc_self-attention-transformers-2023_chunk_024 | 12 | self-attention opera- tion, (this is known as pre-normalization), or like: hpost-norm = LN( f (h) + h), (31) which is known as post-normalization. It  |
| doc_self-attention-transformers-2023_chunk_025 | 13 | 3.5 Transformer Encoder A Transformer Encoder takes a single sequence w1:n, and performs no future masking. It embeds the sequence with E to make x1:n |
| doc_self-attention-transformers-2023_chunk_026 | 13 | simply by using future masking at each application of self-attention. This ensures that the informational constraint (no cheating by looking at the fu |
| doc_self-attention-transformers-2023_chunk_027 | 14 | Cross-Attention. Cross-attention uses one sequence to define the keys and values of self-attention, and another sequence to define the queries. You mi |
| doc_self-attention-transformers-2023_chunk_028 | 14 | While such an architecture has been found to provide better performance than decoder-only models at modest scale [Raffel et al., 2020], it involves sp |
| doc_self-attention-transformers-2023_chunk_029 | 15 | Probabilities Softmax Linear Repeat for number of decoder blocks. Add & Norm Attend only to output of Feed-Forward last Encoder Block. Add & Norm Mult |
| doc_self-attention-transformers-2023_chunk_030 | 16 | A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B |
| doc_self-attention-transformers-2023_chunk_031 | 16 | network model for a mechanism of visual pattern recognition. In Competition and cooperation in neural nets, pages 267–285. Springer. [Lafferty et al., |
| doc_self-attention-transformers-2023_chunk_032 | 16 | A., Lee, K., Narang, S., Matena, M., Zhou, Y., and Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text tr |
| doc_self-attention-transformers-2023_chunk_033 | 17 | [Vaswani et al., 2017] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention  |
| doc_self-attention-transformers-2023_chunk_034 | 17 | Tobing, J., Bhattacharjee, J., Almubarak, K., Chen, K., Lo, K., Von Werra, L., Weber, L., Phan, L., allal, L. B., Tanguy, L., Dey, M., Muñoz, M. R., M |
| doc_self-attention-transformers-2023_chunk_035 | 17 | T., Neeraj, T., Thakker, U., Raunak, V., Tang, X., Yong, Z.-X., Sun, Z., Brody, S., Uri, Y., Tojarieh, H., Roberts, A., Chung, H. W., Tae, J., Phang,  |
| doc_self-attention-transformers-2023_chunk_036 | 17 | Saxena, B., Ferrandis, C. M., Contractor, D., Lansky, D., David, D., Kiela, D., Nguyen, D. A., Tan, E., Bay- lor, E., Ozoani, E., Mirza, F., Ononiwu,  |
| doc_self-attention-transformers-2023_chunk_037 | 17 | Liu, M., Freidank, M., Kang, M., Seelam, N., Dahlberg, N., Broad, N. M., Muellner, N., Fung, P., Haller, P., Chandrasekhar, R., Eisenberg, R., Martin,  |
| doc_self-attention-transformers-2023_chunk_038 | 18 | Belkada, Y., and Wolf, T. (2022). Bloom: A 176b-parameter open-access multilingual language model. [Xiong et al., 2020] Xiong, R., Yang, Y., He, D., Z |

## q004

Question: Why does the Transformer use multiple attention heads?

Chunks:
### doc_self-attention-transformers-2023_chunk_017
Cited by: retrieval_questions.json

```text
Zuko made his uncle tea −∞ −∞ −∞ −∞ Zuko −∞ −∞ −∞ made −∞ −∞ his −∞ uncle tea Intuitively, these are the biggest components to understand. How- ever, as of 2023, by far the most-used architecture in NLP is called the Transformer, introduced by [Vaswani et al., 2017], and it contains a number of components that end up being quite important. So now we’ll get into the details of that architecture. 3 The Transformer The Transformer is an architecture based on self-attention that con- sists of stacked Blocks, each of which contains self-attention and feed- forward layers, and a few other components we’ll discuss. If you’d like to take a peek for intuition, we have a diagram of a Transformer language model architecture in Figure 4. The components we haven’t gone over are multi-head self-attention, layer normalization, resid- ual connections, and attention scaling—and of course, we’ll discuss how these components are combined to form the Transformer. 3.1 Multi-head Self-Attention Intuitively, a single call of self-attention is best at picking out a single value (on average) from the input value set. It does so softly, by averaging over all of the values, but it requires a balancing game in the key-query dot products in order to carefully average two or more things. In Assignment 5, you’ll work through a bit of this intuition more carefully. What we’ll present now, multi-head self-attention, intuitively applies self-attention multiple times at once, each with different key, query, and value transformations of the same
```

### doc_self-attention-transformers-2023_chunk_018
Cited by: retrieval_questions.json, retrieval_questions_ground_truth.json

```text
the key-query dot products in order to carefully average two or more things. In Assignment 5, you’ll work through a bit of this intuition more carefully. What we’ll present now, multi-head self-attention, intuitively applies self-attention multiple times at once, each with different key, query, and value transformations of the same input, and then combines the outputs. For an integer number of heads k, we define matrices K(ℓ), Q(ℓ), V(ℓ) ∈ Rd×d/k for ℓin {1, . . . , k}. (We’ll see why we have the dimensionality reduction to d/k soon.) These our the key, query, and value matrices for each head. Correspondingly, we get keys, queries, and values k(ℓ) 1:n, q(ℓ) 1:n, v(ℓ) 1:n, as in single-head self-attention. with deep learning 9 Figure 3: Diagram of autoregressive future masking in self-attention. Words in each row have words in the future masked out (e.g., “Zuko” can only attend to “Zuko”, while “made” can attend to “Zuko” and “made”.) Probabilities Softmax Linear Add & Norm Feed-Forward Repeat for number of encoder blocks Add & Norm Masked Multi- Head Attention Block Add Position Embeddings Embeddings Decoder Inputs Transformer Decoder Figure 4: Diagram of the Transformer Decoder (without corresponding Encoder, and so no cross-attention.
```

### doc_self-attention-transformers-2023_chunk_019
Cited by: retrieval_questions.json, retrieval_questions_ground_truth.json

```text
We then perform self-attention with each head: n ∑ j=1 h(ℓ) α(ℓ) ij v(ℓ) j (22) i = exp(q(ℓ)⊤ i k(ℓ) j ) α(ℓ) ij = j′=1 exp(q(ℓ)⊤ i k(ℓ) ∑n j′ ) Note that the output h(ℓ) i of each head is in reduced dimension d/k. Finally, we define the output of multi-head self-attention as a linear transformation of the concatenation of the head outputs, letting O ∈Rd×d: i h i ; · · · ; v(k) v(1) , (24) hi = O i where we concatenate the head outputs each of dimensionality d × d/k at their second axis, such that their concatenation has dimension d × d. Sequence-tensor form. To understand why we have the reduced dimension of each head output, it’s instructive to get a bit closer to how multi-head self-attention is implemented in code. In practice, multi-head self-attention is no more expensive than single-head due to the low-rankness of the transformations we apply. For a single head, recall that x1:n is a matrix in Rn×d. Then we can compute our value vectors as a matrix as x1:nV, and likewise our keys and queries x1:nK and x1:nQ, all matrices in Rn×d. To compute self-attention, we can compute our weights in matrix operations: α = softmax(x1:nQK⊤x⊤ 1:n) ∈Rn×n (25) and then compute the self-attention operation for all x1:n via: 1:n)x1:nV ∈Rn×d. (26) h1:n = softmax(x1:nQK⊤x⊤ Here’s a diagram showing the matrix ops: α softmax n (x1:nK)T n = x1:nQ d n When we perform multi-head self-attention in
```

### doc_self-attention-transformers-2023_chunk_020
Cited by: neither file

```text
compute our weights in matrix operations: α = softmax(x1:nQK⊤x⊤ 1:n) ∈Rn×n (25) and then compute the self-attention operation for all x1:n via: 1:n)x1:nV ∈Rn×d. (26) h1:n = softmax(x1:nQK⊤x⊤ Here’s a diagram showing the matrix ops: α softmax n (x1:nK)T n = x1:nQ d n When we perform multi-head self-attention in this matrix form, we first reshape x1:nQ, x1:nK, and x1:nV each into a matrix of shape Rn,k,d/k, splitting the model dimensionality into two axes, for the number of heads and the number of dimensions per head. We can with deep learning 10 (23)
```

### doc_self-attention-transformers-2023_chunk_021
Cited by: retrieval_questions_ground_truth.json

```text
then transpose the matrices to Rk,n,d/k, which intuitively should look like k sequences of length n and dimensionality d/k. This allows us to perform the batched softmax operation in parallel across the heads, using the number of heads kind of like a batch axis (and indeed in practice we’ll also have a separate batch axis.) So, the total computation (except the last linear transformation to combine the heads) is the same, just distributed across the (each lower-rank) heads. Here’s a diagram like the single-head diagram, demonstrating how the multi-head operation ends up much like the single-head operation: reshape(x1:nQ) α reshape((x1:nK)T) n = softmax n n d/k 3.2 Layer Norm One important learning aid in Transformers is layer normalization [Ba et al., 2016]. The intuition of layer norm is to reduce uninforma- tive variation in the activations at a layer, providing a more stable input to the next layer. Further work shows that this may be most useful not in normalizing the forward pass, but actually in improving gradients in the backward pass [Xu et al., 2019]. To do this, layer norm (1) computes statistics across the activations at a layer to estimate the mean and variance of the activations, and (2) normalizes the activations with respect to those estimates, while (3) optionally learning (as parameters) an elementwise additive bias and multiplicative gain by which to sort of de-normalize the activations in a predictable way. The third part seems not to be crucial, and may even be harmful [Xu et al.,
```

Decision: ______

## q008

Question: What is the role of the Transformer decoder?

Chunks:
### doc_self-attention-transformers-2023_chunk_025
Cited by: retrieval_questions_ground_truth.json

```text
3.5 Transformer Encoder A Transformer Encoder takes a single sequence w1:n, and performs no future masking. It embeds the sequence with E to make x1:n, adds the position representation, and then applies a stack of independently parameterized Encoder Blocks, each of which consisting of (1) multi- head attention and Add & Norm, and (2) feed-forward and Add & Norm. So, the output of each Block is the input to the next. Figure 5 presents this. In the case that one wants probabilities out of the tokens of a Transformer Encoder (as in masked language modeling for BERT [Devlin et al., 2019], which we’ll cover later), one applies a linear transformation to the output space followed by a softmax. Uses of the Transformer Encoder. A Transformer Encoder is great in contexts where you aren’t trying to generate text autoregressively (there’s no masking in the encoder so each position index can see the whole sequence,) and want strong representations for the whole sequence (again, possible because even the first token can see the whole future of the sequence when building its representation.) 3.6 Transformer Decoder To build a Transformer autoregressive language model, one uses a Transformer Decoder. These differ from Transformer Encoders simply by using future masking at each application of self-attention. This ensures that the informational constraint (no cheating by looking at the future!) holds throughout the architecture. We show a diagram of this architecture in Figure 4. Famous examples of this are GPT- 2 [Radford et al., 2019], GPT-3 [Brown
```

### doc_self-attention-transformers-2023_chunk_026
Cited by: retrieval_questions.json, retrieval_questions_ground_truth.json

```text
simply by using future masking at each application of self-attention. This ensures that the informational constraint (no cheating by looking at the future!) holds throughout the architecture. We show a diagram of this architecture in Figure 4. Famous examples of this are GPT- 2 [Radford et al., 2019], GPT-3 [Brown et al., 2020] and BLOOM [Workshop et al., 2022]. 3.7 Transformer Encoder-Decoder A Transformer encoder-decoder takes as input two sequences. Fig- ure 6 shows the whole encoder-decoder structure. The first sequence x1:n is passed through a Transformer Encoder to build contextual representations. The second sequence y1:m is encoded through a modified Transformer Decoder architecture in which cross-attention (which we haven’t yet defined!) is applied from the encoded repre- sentation of y1:m to the output of the Encoder. So, let’s take a quick detour to discuss cross-attention; it’s not too different from what we’ve already seen. with deep learning 13 Probabilities Softmax Linear Add & Norm Feed-Forward Repeat for number of encoder blocks Add & Norm Multi-Head Attention Block Add Position Embeddings Embeddings Encoder Inputs Transformer Encoder Figure 5: Diagram of the Transformer Encoder.
```

Decision: ______

## q010

Question: What problem does self-attention solve compared with recurrent architectures?

Chunks:
### doc_self-attention-transformers-2023_chunk_005
Cited by: retrieval_questions_ground_truth.json

```text
sequence index (often called the dependence on “time”) highlighted in Equation 4. Parallelization issues with dependence on the sequence index. Modern graphics processing units (GPUs) are excellent at crunching through a lot of simple operations (like addition) in parallel. For example, when I have a matrix A ∈Rn×k and a matrix B ∈Rk×d, a GPU is just blazing fast at computing AB ∈Rn×d. The constraint of the operations occuring in parallel, however, is crucial – when computing AB the simplest way, I’m performing a bunch of multiplies and then a bunch of sums, most of which don’t depend on the output of each other. However, in a recurrent neural network, when I compute h2 = σ(Wh1 + Ux2), (5) I can’t compute h2 until I know the value of h1, so we can write it out as h2 = σ(Wσ(Wh0 + Ux1) + Ux2). (6) Likewise if I wanted to compute h3, I can’t compute it until I know h2, which I can’t compute until I know h1, etc. Visually, this looks like Figure 1. As the sequence gets longer, there is only so much I can parallelize the computation of the network on a GPU because of the number of serial dependencies. (Serial meaning one-after-the- other.) 1 2 3 4 5 0 0 0 0 0 Zuko made his uncle tea As GPUs (and later, other accelerators like Tensor Processing Units (TPUs) became more powerful and researchers wanted to take fuller advantage of them, this dependence in time became
```

### doc_self-attention-transformers-2023_chunk_006
Cited by: retrieval_questions.json, retrieval_questions_ground_truth.json

```text
of serial dependencies. (Serial meaning one-after-the- other.) 1 2 3 4 5 0 0 0 0 0 Zuko made his uncle tea As GPUs (and later, other accelerators like Tensor Processing Units (TPUs) became more powerful and researchers wanted to take fuller advantage of them, this dependence in time became untenable. Linear interaction distance. A related issue with RNNs is the diffi- culty with which distant tokens in a sequence can interact with each other. By interact, we mean that the presence of one token (already observed in the past) gainfully affects the processing of another token. For example, in the sentence The chef1 who ran out of blackberries and went to the stores is1 with deep learning 3 Figure 1: A RNN unrolled in time. The rectangles are intermediate states of the RNN (e.g., the first row is the embedding layer, and the second row is the RNN hidden state at each time step) and the number in the rectangle is the number of serial operations that need to be performed before this intermediate state can be computed
```

### doc_self-attention-transformers-2023_chunk_007
Cited by: retrieval_questions.json, retrieval_questions_ground_truth.json

```text
the number of intermediate computations—matrix multiplies and nonlinearities, for example—that separate chef from is scales with the number of words between them. We visualize this in Figure 2. 1 2 3 4 5 0 Zuko made his uncle tea Intuitively, researchers believe there’s an issue with linear inter- action distance because it can be difficult for networks to precisely “recall” the presence of a word when a large number of operations occur after observing that word. This can make it difficult to learn how distant words should impact the representation of the current word. This notion of direct interaction between elements of a sequence might remind you of the attention mechanism [Bahdanau et al., 2014] in machine translation. In that context, while generating a translation, we learned how to look back into the source sequence once per token of the translation. In this note, we’ll present an entire replacement for recurrent neural networks just based on attention. This will solve both the parallelization issues and the linear interaction distance issues with recurrent neural networks. 2 A minimal self-attention architecture Attention, broadly construed, is a method for taking a query, and softly looking up information in a key-value store by picking the value(s) of the key(s) most like the query. By “picking” and “most like,” we mean averaging overall values, putting more weight on those which correspond to the keys more like the query. In self- attention, we mean that we use the same elements to help us define the querys
```

### doc_self-attention-transformers-2023_chunk_008
Cited by: retrieval_questions.json

```text
the value(s) of the key(s) most like the query. By “picking” and “most like,” we mean averaging overall values, putting more weight on those which correspond to the keys more like the query. In self- attention, we mean that we use the same elements to help us define the querys as we do the keys and values. In this section, we’ll discuss how to develop contextual representa- tions with methods wherein the main mechanism for contextualiza- tion is not recurrence, but attention. 2.1 The key-query-value self-attention mechanism There are many forms of self-attention; the form we’ll discuss here is currently the most popular. It’s called key-query-value self-attention. with deep learning 4 Figure 2: A RNN unrolled in time. The rectangles are intermediate states of the RNN (e.g., the first row is the embedding layer, and the second row is the RNN hidden state at each time step) and the number in the rectangle is roughly the number of operations separating lexical information of the word tea from each intermediate state.
```

Decision: ______

