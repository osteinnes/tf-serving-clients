import os
from jinja2 import Environment, PackageLoader


class Writer:
    def __init__(self, path):
        environment = Environment(loader=PackageLoader('tools', 'templates'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('annotation.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'folder': os.path.basename(os.path.dirname(abspath)),
            'objects': []
        }

    def addObject(self, path, num_label_cracks, num_pred_cracks, ious):

        abspath = os.path.abspath(path)

        self.template_parameters['objects'].append({
            'filename': os.path.basename(abspath),
            'num_label_cracks': num_label_cracks,
            'num_pred_cracks': num_pred_cracks,
            'score': ious
        })

        print("ious: ", ious)

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)
