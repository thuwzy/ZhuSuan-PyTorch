#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re


api_dir = './api'

# add automodule options for distributions and StochasticTensor
options = [':inherited-members:']
modules = ['zhusuan.distributions',
           'zhusuan.framework',
           'zhusuan.variational',
           'zhusuan.mcmc',
           'zhusuan.invertible']

for module in modules:
    module_path = os.path.join(api_dir, module + '.rst')
    with open(module_path, 'r') as f:
        module_string = f.read()
    target = r'\:members\:(\n|.)*\:undoc-members\:(\n|.)*\:show-inheritance\:'
    indent = '    '
    rep = ':members:\n' + indent + ':undoc-members:\n' + indent + \
        ':show-inheritance:'
    for option in options:
        rep += '\n' + indent + option
    post_module_string = re.sub(target, rep, module_string)
    post_module_string = re.sub("Submodules\n----------", "", post_module_string)
    with open(module_path, 'w') as f:
        f.write(post_module_string)
