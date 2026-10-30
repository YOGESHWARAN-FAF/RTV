import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'ASAPNet':
        from .ASAPNet_model import ASAPNetModel, InferenceModel
        if opt.isTrain:
            model = ASAPNetModel()
        else:
            model = InferenceModel()
    elif opt.model == 'ASAPNet_RGBA':
        from .ASAPNet_rgba import ASAPNet_RGBA, InferenceModel
        if opt.isTrain:
            model = ASAPNet_RGBA()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_DUAL_RGBA':
        from .pix2pixHD_dual_rgba import Pix2PixHD_DUAL_RGBA, InferenceModel
        if opt.isTrain:
            model = Pix2PixHD_DUAL_RGBA()
        else:
            model = InferenceModel()

    elif opt.model == 'pix2pixHD_RGBA':
        from .pix2pixHD_rgba import Pix2PixHD_RGBA, InferenceModel
        if opt.isTrain:
            model = Pix2PixHD_RGBA()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_RNN_RGBA':
        from .pix2pixHD_rnn_rgba import Pix2PixHD_RNN_RGBA, InferenceModel
        if opt.isTrain:
            model = Pix2PixHD_RNN_RGBA()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_mask':
        from.pix2pixHD_mask import Pix2PixHDModel_Mask, InferenceModel_Mask
        if opt.isTrain:
            model = Pix2PixHDModel_Mask()
        else:
            model = InferenceModel_Mask()
    elif opt.model == 'pix2pixHD_align':
        from.pix2pixHD_align import Pix2PixHDModel_Align, InferenceModel_Align
        if opt.isTrain:
            model = Pix2PixHDModel_Align()
        else:
            model = InferenceModel_Align()
    elif opt.model == 'pix2pixHD_Inpaint':
        from.pix2pixHD_inpaint import Pix2PixHDModel_Inpaint, InferenceModel_Inpaint
        if opt.isTrain:
            model = Pix2PixHDModel_Inpaint()
        else:
            model = InferenceModel_Inpaint()
    elif opt.model == 'UTransformerModel':
        from .utransformer_model import UTransformerModel, InferenceModel
        if opt.isTrain:
            model = UTransformerModel()
        else:
            model = InferenceModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    #print(next(model.parameters()).device)
    return model
