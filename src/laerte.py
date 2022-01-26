from glob import glob


def ctimene():
    dict_label = {'road': 'flat', 'sidewalk': 'flat', 'parking': 'flat', 'rail track': 'flat',
                  'person': 'human', 'rider': 'human',
                  'truck': 'vehicle', 'car': 'vehicle', 'bus': 'vehicle', 'on rails': 'vehicle',
                  'motorcycle': 'vehicle', 'bicycle': 'vehicle', 'caravan': 'vehicle',
                  'trailer': 'vehicle',
                  'building': 'construction', 'wall': 'construction', 'fence': 'construction',
                  'guard rail': 'construction',
                  'bridge': 'construction', 'tunnel': 'construction',
                  'pole': 'object', 'pole group': 'object', 'traffic sign': 'object',
                  'traffic light': 'object',
                  'vegetation': 'nature', 'terrain': 'nature',
                  'ground': 'void', 'dynamic': 'void', 'static': 'void',
                  'flat': 'flat', 'human': 'human', 'vehicle': 'vehicle', 'construction': 'construction',
                  'object': 'object', 'nature': 'nature', 'void': 'void', 'sky': 'sky',
                  'license plate': 'vehicle',
                  'ego vehicle': 'void',
                  'out of roi': 'void',
                  'bicyclegroup': 'vehicle',
                  'cargroup': 'vehicle',
                  'persongroup': 'human',
                  'polegroup': 'object',
                  'rectification border': 'void',
                  'train': 'vehicle',
                  'ridergroup': 'vehicle',
                  'motorcyclegroup': 'vehicle',
                  'truckgroup': 'vehicle'
                  }
    return dict_label


def ulysse():
    return glob('../data/finetuning/gtFine/**/*polygons.json', recursive=True)