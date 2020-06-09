import sys
import torch
from tool.darknet2pytorch import Darknet


def fransform_to_onnx(cfgfile, weightfile):
    batch_size=1
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)  # .cuda()

    onnx_file_name = "yolov4_{}_3_{}_{}.onnx".format(batch_size, model.height, model.width)

    # Export the model
    print('Export the onnx model ...')
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      export_params=True,
                      input_names=['input'],
                      output_names=[
                        'boxes_1', 'labels_1', 'conf_1', 
                        'boxes_2', 'labels_2', 'conf_2', 
                        'boxes_3', 'labels_3', 'conf_3'
                      ],
                      opset_version=11,
                      do_constant_folding=True,
                      dynamic_axes=None)

    print('Onnx model exporting done')
    return onnx_file_name


if __name__ == '__main__':
    if len(sys.argv) == 3:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        fransform_to_onnx(cfgfile, weightfile)
    else:
        print('Please execute this script this way:\n')
        print('  python darknet2onnx.py <cfgFile> <weightFile>')
