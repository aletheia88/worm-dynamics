

import marimo

__generated_with = "0.13.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from attention_predict.attention_model_mini import AttentionModelMini
    from attention_predict.attention_mini import AttentionBlockMini

    return AttentionBlockMini, AttentionModelMini


@app.cell
def _():
    depth = 4
    inc_factor = 2
    num_fmaps = 4
    num_encoders = 5
    return depth, inc_factor, num_encoders, num_fmaps


@app.cell
def _(inc_factor, num_encoders, num_fmaps):
    def ref_encoder_fmaps(level):
        if level == 0:
            fmaps_in = num_encoders
        else:
            fmaps_in = num_encoders * (num_fmaps * inc_factor ** (level - 1))

        fmaps_out = num_encoders * (num_fmaps * inc_factor**level)
        return fmaps_in, fmaps_out

    return (ref_encoder_fmaps,)


@app.cell
def _(ref_encoder_fmaps):
    def ref_decoder_fmaps(level, ref_encoder_fmaps=ref_encoder_fmaps):
        # reduce input to same number of output channels as the encoder (at _same_ level)
        fmaps_out = ref_encoder_fmaps(level)[1]  # NOTE: already scaled by num_encoders
        # Num inputs from skip connection = outputs of encoder on previous/lower level (i.e. level + 1)
        prev_level_out = ref_encoder_fmaps(level + 1)[1]
        fmaps_in = fmaps_out + prev_level_out
        return fmaps_in, fmaps_out

    return (ref_decoder_fmaps,)


@app.cell
def _(depth, ref_decoder_fmaps, ref_encoder_fmaps):
    def print_fmaps(depth):
        for level in range(depth):
            inp, out = ref_encoder_fmaps(level)
            print(
                "Reference encoder at level: {level}, I/O channels: ({inp},{out})".format(
                    level=level, inp=inp, out=out
                )
            )

        print("\n")

        for level in range(depth - 1):
            dec_inp, dec_out = ref_decoder_fmaps(level)
            print(
                "Reference decoder at level: {level}, I/O channels: ({inp},{out})".format(
                    level=level, inp=dec_inp, out=dec_out
                )
            )

    print_fmaps(depth)
    return


@app.cell
def _(AttentionModelMini):
    model = AttentionModelMini()
    print(model)
    return (model,)


@app.cell
def _(model):
    for i, chan in enumerate(model.encoder_features):
        print(
            "Encoder level {lvl} input: {inp}, output: {out}".format(
                lvl=i, inp=chan.input, out=chan.output
            )
        )
    for i, chan in enumerate(model.decoder_features):
        print(
            "Decoder level {lvl} input: {inp}, output: {out}".format(
                lvl=i, inp=chan.input, out=chan.output
            )
        )

    return


@app.cell
def _(AttentionBlockMini):
    att_block = AttentionBlockMini(2048, (5,8), (2,5,512))
    return


if __name__ == "__main__":
    app.run()
