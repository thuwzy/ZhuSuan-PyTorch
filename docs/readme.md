# Documentation #
Zhusuan-PyTorch Documentation is supported by [Sphinx](http://www.sphinx-doc.org/en/stable/). 
To build the docs, run from the toplevel directory:

## Installation ##
```
pip install -r requirements.txt
```

## Workflow ##
To change the documentation, update the `*.rst` files.

To auto-gen the api rst files, `make api`

Run `python post_apidoc.py`

To build the html pages, `make html`
