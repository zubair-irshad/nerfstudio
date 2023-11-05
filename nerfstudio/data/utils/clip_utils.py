import clip

from nerfstudio.lseg.modules.models.lseg_vit import _make_pretrained_clip_vitl16_384


def make_clip():
    # features=512
    backbone = "clip_vitl16_384"
    use_pretrained = True
    use_readout = "ignore"
    enable_attention_hooks = False
    hooks = {
        "clip_vitl16_384": [5, 11, 17, 23],
        "clipRN50x16_vitl16_384": [5, 11, 17, 23],
        "clip_vitb32_384": [2, 5, 8, 11],
    }

    hooks = hooks[backbone]
    clip_pretrained, pretrained = _make_pretrained_clip_vitl16_384(
        use_pretrained,
        hooks=hooks,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained


def extract_clip_features(clip_pretrained, label_src):
    labels = []

    lines = label_src.split(",")
    for line in lines:
        label = line
        labels.append(label)

    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features
