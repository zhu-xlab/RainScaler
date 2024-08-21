def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain_ppd import ModelPlain as M

    elif model == 'plain3':  # two inputs: L, C
        from models.model_plain3 import ModelPlain as M #if only msrresnet is used

    elif model == 'gan':     # one input: L
        from models.model_gan import ModelGAN as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
