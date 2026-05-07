from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.xml_style import XMLDataset


@DATASETS.register_module()
class DiverseWeatherDataset(CocoDataset):
    """Dataset for diverse weather conditions in COCO format."""

    METAINFO = {
        'classes': ('person', 'truck', 'rider', 'bus', 'bicycle', 'motorcycle', 'car'),
        # Custom color palette for visualization
        'palette': [
            (102, 194, 165),  # person - mint green
            (252, 141, 98),   # truck - soft orange
            (141, 160, 203),  # rider - light blue-purple
            (231, 138, 195),  # bus - pink-purple
            (166, 216, 84),   # bicycle - light green
            (255, 217, 47),   # motorcycle - soft yellow
            (229, 196, 148)   # car - beige
        ]
    }


@DATASETS.register_module()
class DiverseWeatherDatasetVOC(XMLDataset):

    METAINFO = {
        'classes':
        ('person', 'truck', 'rider', 'bus', 'bike', 'motor', 'car'),  # Class names in VOC format; slightly different after conversion to COCO
        'palette': [
            (220, 20, 60),  # person
            (0, 0, 142),    # truck
            (0, 0, 230),    # rider
            (106, 0, 228),  # bus
            (0, 60, 100),   # bicycle
            (0, 80, 100),   # motorcycle
            (0, 0, 192)     # car
        ]
    }

@DATASETS.register_module()
class VOCDatasetCOCO(CocoDataset):
    """Dataset for Diverse Weather Night Conditions based on COCO format."""

    METAINFO = {
        'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255),
                    (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1), (120, 166, 157),
                    (0, 182, 199), (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
                    (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]
    }
