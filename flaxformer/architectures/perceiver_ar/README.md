# Perceiver AR

This is the Flaxformer implementation of
[Perceiver AR](https://arxiv.org/abs/2202.07765), which can be used as a drop-in
replacement for Transformer decoders and can efficiently handle long
sequences.

Gin configs for using Perceiver AR with T5X can be found in
`t5x/configs/perceiver_ar`.

An example for training a language model on C4 is here:
`t5x/configs/perceiver_ar/examples/c4_lm.gin`.

Currently, the T5X `DecoderOnlyModel` interface is implemented, which allows
training decoder-only and Prefix LM models. Using Perceiver AR as the decoder
of `EncoderDecoderModel` is not yet implemented, but should be relatively
straightforward.

## Configuration

Perceiver AR adds a few important configuration options in addition to the
ones used in a standard decoder.

### Number of latents

Gin config: `NUM_LATENTS`

Perceiver AR decouples the input sequence length from the width of the
self-attention stack used to process it. The width of the self-attention stack
is configured with `NUM_LATENTS` and is typically set to something like 1024,
even for input lengths >10k. As a result of using a narrower self-attention
stack, more layers can likely be used.

### Training cropping method

Gin config: `PerceiverARModel.train_cropping_method`

During inference, the assigned position of the latents will slide forward as
the sequence length increases. So the model needs to be trained for all these
possible positions within the sequence length. This can be done either during
data preprocessing or on the fly during training.

By default, `FULL_LATENTS` mode is used where cropping happens on the fly and
makes maximum use of compute. This is likely the correct setting for most
situations, but see the documentation in `t5_models.py` for a description of
the other options.

### Fast decoding latent reset fill

Gin config: `PerceiverARModel.decoding_latent_reset_fill`

Because Perceiver AR has a narrower self-attention stack than its inputs, we
have to modify how fast inference caching works. The full details are in the
paper in Appendix E.3.

In short, when the KV activation cache gets full, we have to do a full forward
pass of the model for the next prediction and partially fill the cache. There’s
a tradeoff between filling the cache all the way (more compute for predicting
the next tokens) vs. filling it only a little (more steps before the next full
forward pass).

Right now, the default is to refill the cache to `num_latents - 128`, which
seems to work well. But if it’s too slow or results are not as high a quality as
expected, it may be worth exploring other values.

