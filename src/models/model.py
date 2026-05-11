import segmentation_models_pytorch as smp


def build_model():

    model = smp.Unet(

        encoder_name="efficientnet-b3",

        encoder_weights="imagenet",

        in_channels=4,

        classes=1,

        activation=None,

        decoder_attention_type="scse"
    )

    return model