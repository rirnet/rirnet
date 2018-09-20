import torch
import numpy as np
import os
import subprocess
import sys
import collections
import pandas as pd
from torch.autograd import Variable
from importlib import import_module
from glob import glob
from rirnet import rirnet_database

def summary(input_size, model):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = collections.OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    weights = torch.nn.Parameter(module.weight)
                    params += torch.prod(torch.LongTensor(list(weights.size())))
                    if weights.requires_grad:
                        summary[m_key]['trainable'] = 'True'
                    else:
                        summary[m_key]['trainable'] = 'False'
                else:
                    summary[m_key]['trainable'] = '-'
                if hasattr(module, 'bias'):
                    biases = torch.nn.Parameter(module.bias)
                    params +=  torch.prod(torch.LongTensor(list(biases.size())))
                summary[m_key]['nb_params'] = int(torch.tensor(params))
            if not isinstance(module, torch.nn.Sequential) and \
               not isinstance(module, torch.nn.ModuleList) and \
               not (module == model):
                hooks.append(module.register_forward_hook(hook))
        # check if torch.re are multiple inputs to torch. network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(1,*in_size)) for in_size in input_size]
        else:
            x = Variable(torch.rand(1,*input_size))

        # create properties
        summary = collections.OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        model(x)
        # remove torch.se hooks
        for h in hooks:
            h.remove()

        return summary


def generate_tex(model_dir, df):
    with open('network.tex', 'w') as tf:
        tf.write('\n\\documentclass{article}')
        tf.write('\n\\usepackage[utf8]{inputenc}')
        tf.write('\n\\usepackage{booktabs}')
        tf.write('\n\\usepackage{fullpage}')
        tf.write('\n\\usepackage{graphicx}')
        tf.write('\n\\begin{document}\n')
        tf.write('\n\\centering\n')
        tf.write(df.to_latex())
        tf.write('\n\\includegraphics[height=200pt]{'+model_dir+'/loss_over_epochs.png}')
        tf.write('\n\\end{document}')


def compile_pdf(model_dir):
    try:
        subprocess.call(['pdflatex','network.tex'], stdout=open(os.devnull, 'wb'))
        subprocess.call(['mv','network.pdf',model_dir])
        subprocess.call(['rm','network.aux', 'network.log', 'network.tex'])
    except OSError:
        print('LaTeX not found, printing to .tex')
        subprocess.call(['mv','network.tex',model_dir])


def get_training_data_shape(model, args):
    data_transform = model.transform()
    db = rirnet_database(is_training = True, args = args, transform = data_transform)
    (a, _) = db.__getitem__(0)
    return np.shape(a)


def append_total_nb_params(df):
    total_nb_params = df['nb_params'].sum()
    df2 = pd.DataFrame([['', total_nb_params, '', '']], columns=list(df))
    df2 = df2.rename({0: 'TOTAL'}, axis='index')
    return df.append(df2)


def main(model_dir):
    sys.path.append(model_dir)
    net = import_module('net')
    model = net.Net()
    args = model.args()
    input_size = get_training_data_shape(model, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    list_net_path = glob(model_dir +'/*.pth')
    list_net_files = [os.path.split(x)[1] for x in list_net_path]
    highest_epoch = max([int(e.split('.')[0]) for e in list_net_files])
    model.load_state_dict(torch.load(os.path.join(model_dir, '{}.pth'.format(str(highest_epoch)))))
    d = summary(input_size, model)
    df = pd.DataFrame(d).T
    df = append_total_nb_params(df)
    generate_tex(model_dir, df)
    compile_pdf(model_dir)


if __name__ == "__main__":
    main(sys.argv[1])
