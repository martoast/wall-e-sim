# TACO Dataset Integration Guide

> How we integrate the TACO (Trash Annotations in Context) dataset into our simulation for real-world perception training.

---

## What is TACO?

**TACO** is an open image dataset of waste in the wild, containing photos of litter taken in diverse environments - tropical beaches, urban streets, woods, roads.

| Metric | Value |
|--------|-------|
| Images | 1,500+ |
| Annotations | 9,823 objects |
| Categories | 60 litter types |
| Super-categories | 28 groups |
| Format | COCO (industry standard) |
| Annotation type | Bounding boxes + segmentation masks |

**Repository**: https://github.com/pedropro/TACO
**Paper**: "TACO: Trash Annotations in Context for Litter Detection" (arXiv:2003.06975)
**License**: CC BY 4.0 (annotations), various open licenses (images)

---

## Why TACO Matters for Trashly

1. **Real-world ground truth** - Actual trash in actual environments
2. **Segmentation masks** - Enables robotic grasping, not just detection
3. **Hierarchical taxonomy** - Categories match real-world trash types
4. **COCO format** - Works with all major ML frameworks
5. **Active community** - Dataset is growing with contributions

---

## TACO Category Taxonomy

### Top 10 Categories (by annotation count)

| Category | Objects | Images | Notes |
|----------|---------|--------|-------|
| Cigarette | 1,336 | 227 | Most common litter |
| Unlabeled litter | 1,068 | 269 | Ambiguous items |
| Plastic film | 957 | 310 | Wrappers, sheets |
| Other plastic wrapper | 583 | 184 | Generic wrappers |
| Clear plastic bottle | 576 | 225 | Water/soda bottles |
| Other plastic | 563 | 171 | Misc plastic items |
| Drink can | 460 | 151 | Aluminum cans |
| Plastic bottle cap | 419 | 185 | Loose caps |
| Plastic straw | 336 | 110 | Single-use straws |
| Broken glass | 280 | - | Safety hazard |

### 28 Super-Categories (grouped)

**Plastic Items:**
- Plastic bottle (clear, other)
- Plastic bottle cap
- Plastic bag (single, polypropylene, other)
- Plastic container
- Plastic film
- Plastic straw
- Disposable plastic cup
- Plastic lid
- Other plastic wrapper
- Other plastic

**Paper/Cardboard:**
- Paper (normal, tissue)
- Paper cup
- Paper bag
- Cardboard
- Corrugated carton
- Carton (egg, drink, meal, pizza box)

**Metal:**
- Drink can
- Food can
- Metal bottle cap
- Aerosol
- Aluminum foil
- Metal container

**Glass:**
- Glass bottle (clear, other)
- Glass jar
- Broken glass

**Organic/Other:**
- Cigarette
- Food waste
- Rope & strings
- Shoe
- Squeezable tube
- Styrofoam piece
- Battery
- Unlabeled litter

### TACO-10 (Simplified for Training)

For experiments with limited data, TACO provides a 10-class mapping:

```
1. Cigarette
2. Drink can
3. Plastic bottle (clear + other)
4. Plastic bag (all types)
5. Bottle cap (plastic + metal)
6. Paper cup
7. Carton
8. Styrofoam
9. Glass bottle
10. Other litter (everything else)
```

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         TACO DATASET                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Images    │  │ Annotations │  │      Segmentation       │  │
│  │  (1500+)    │  │   (COCO)    │  │        Masks            │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML TRAINING PIPELINE                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Train Object Detection Model (YOLO/Faster-RCNN/etc)    │    │
│  │  - Input: TACO images                                    │    │
│  │  - Output: class + confidence + bbox/mask                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼ Model behavior statistics
┌─────────────────────────────────────────────────────────────────┐
│                      SIMULATION                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  WorldObject (uses TACO categories)                      │    │
│  │  - category: "clear_plastic_bottle"                      │    │
│  │  - super_category: "plastic_bottle"                      │    │
│  │  - visual_features: {shape, color, size, texture}        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Perception System (calibrated to real model)            │    │
│  │  - Confidence curves match TACO-trained model            │    │
│  │  - Confusion matrix mimics real misclassifications       │    │
│  │  - Distance/angle effects modeled                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Behavior Controller                                      │    │
│  │  - Decisions based on uncertain perception               │    │
│  │  - Same logic will run on real robot                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Download TACO Dataset

```bash
# Clone repository
git clone https://github.com/pedropro/TACO.git
cd TACO

# Install dependencies
pip3 install -r requirements.txt

# Download images (creates data/images folder)
python3 download.py

# For community annotations (more data, less verified)
python3 download.py --dataset_path ./data/annotations_unofficial.json
```

### Step 2: Explore the Data

```python
# Run the demo notebook
jupyter notebook demo.ipynb

# Or manually load annotations
import json

with open('data/annotations.json') as f:
    taco = json.load(f)

# Structure (COCO format):
# taco['images']      - list of image metadata
# taco['annotations'] - list of object annotations
# taco['categories']  - list of category definitions
# taco['scene_annotations'] - background scene tags

# Example: print all categories
for cat in taco['categories']:
    print(f"{cat['id']}: {cat['name']} (super: {cat['supercategory']})")
```

### Step 3: Create Simulation Object Categories

Map TACO categories to our simulation WorldObject types:

```python
# entities/world_object.py

TACO_CATEGORIES = {
    # Super-category: [list of sub-categories]
    'plastic_bottle': ['clear_plastic_bottle', 'other_plastic_bottle'],
    'plastic_bag': ['single_use_carrier_bag', 'polypropylene_bag', 'other_plastic_bag'],
    'drink_can': ['drink_can'],
    'cigarette': ['cigarette'],
    'paper': ['normal_paper', 'tissues'],
    'carton': ['drink_carton', 'egg_carton', 'meal_carton'],
    'glass_bottle': ['clear_glass_bottle', 'other_glass_bottle'],
    'broken_glass': ['broken_glass'],
    'food_can': ['food_can'],
    'styrofoam': ['styrofoam_piece'],
    # ... etc
}

# Non-trash categories (for false positive testing)
NON_TRASH_CATEGORIES = {
    'natural': ['leaf', 'rock', 'pinecone', 'stick', 'flower'],
    'valuable': ['phone', 'wallet', 'keys', 'toy', 'jewelry'],
    'ambiguous': ['worn_paper', 'fabric_scrap', 'unknown_object'],
}
```

### Step 4: Train Detection Model on TACO

```python
# Using detectron2 (Facebook's detection library)
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

# Register TACO dataset
register_coco_instances(
    "taco_train",
    {},
    "TACO/data/annotations.json",
    "TACO/data/images"
)

# Configure Mask R-CNN
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.DATASETS.TRAIN = ("taco_train",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60  # TACO categories

# Train
trainer = DefaultTrainer(cfg)
trainer.train()
```

### Step 5: Extract Model Behavior for Simulation

After training, analyze the model to calibrate simulation perception:

```python
# Evaluate model and extract confusion matrix
from sklearn.metrics import confusion_matrix

# Run inference on validation set
predictions = []
ground_truths = []

for image in val_dataset:
    pred = model(image)
    predictions.append(pred['class'])
    ground_truths.append(image['label'])

# Build confusion matrix
cm = confusion_matrix(ground_truths, predictions)

# Extract per-class metrics
for i, category in enumerate(categories):
    precision = cm[i,i] / cm[:,i].sum()
    recall = cm[i,i] / cm[i,:].sum()
    print(f"{category}: precision={precision:.2f}, recall={recall:.2f}")

# Extract confidence distribution per class
# This tells us how confident the model is for each category
confidence_stats = {}
for category in categories:
    confs = [p['confidence'] for p in predictions if p['class'] == category]
    confidence_stats[category] = {
        'mean': np.mean(confs),
        'std': np.std(confs),
        'min': np.min(confs),
        'max': np.max(confs)
    }
```

### Step 6: Calibrate Simulation Perception

Use the extracted metrics to make simulation match reality:

```python
# systems/perception.py

class TACOCalibratedPerception:
    def __init__(self, model_stats_path):
        # Load stats from trained TACO model
        with open(model_stats_path) as f:
            self.model_stats = json.load(f)

    def classify(self, obj, distance):
        """
        Simulate perception that matches TACO-trained model behavior.
        """
        category = obj.category

        # Base confidence from model stats
        base_conf = self.model_stats[category]['mean_confidence']
        conf_std = self.model_stats[category]['std_confidence']

        # Add distance-based degradation
        distance_factor = max(0.3, 1.0 - distance / 500)

        # Add noise matching real model variance
        noise = np.random.normal(0, conf_std)

        confidence = base_conf * distance_factor + noise
        confidence = np.clip(confidence, 0.0, 1.0)

        # Simulate misclassification based on confusion matrix
        if np.random.random() > self.model_stats[category]['precision']:
            # Misclassify based on confusion probabilities
            wrong_class = self._sample_confusion(category)
            return wrong_class, confidence * 0.7

        return category, confidence
```

---

## Data Augmentation Techniques (from TACO paper)

The authors recommend these augmentation strategies:

1. **Standard augmentations:**
   - Gaussian blur and AWG noise
   - Exposure and contrast changes
   - Rotation [-45°, 45°]
   - Cropping around objects

2. **Litter transplantation:**
   - Copy-paste TACO segmentations onto new backgrounds
   - Use soft masking via distance transforms
   - Blend naturally into new scenes

3. **For simulation:**
   - Use TACO segmentation masks as object textures
   - Randomize background environments
   - Vary lighting and weather conditions

---

## File Structure After Integration

```
wall_e_sim/
├── tasks/
│   ├── ceo.md
│   ├── todo.md
│   └── taco_integration.md (this file)
├── data/
│   └── taco/
│       ├── annotations.json
│       ├── images/
│       ├── model_stats.json (extracted from training)
│       └── confusion_matrix.json
├── entities/
│   ├── world_object.py (uses TACO categories)
│   └── object_spawner.py (spawns TACO-typed objects)
├── systems/
│   ├── perception.py (TACO-calibrated)
│   └── classifier.py (mimics TACO model)
└── ml/
    ├── train_taco.py
    ├── evaluate_model.py
    └── extract_stats.py
```

---

## Validation Checklist

Before considering TACO integration complete:

- [ ] Downloaded and explored TACO dataset
- [ ] Trained object detection model on TACO
- [ ] Achieved reasonable mAP on validation set
- [ ] Extracted confusion matrix and confidence stats
- [ ] Created WorldObject categories matching TACO taxonomy
- [ ] Calibrated simulation perception to match model behavior
- [ ] Verified sim classifier produces similar error patterns
- [ ] Tested with edge cases (small objects, occlusion, distance)

---

## Resources

- **TACO GitHub**: https://github.com/pedropro/TACO
- **TACO Paper**: https://arxiv.org/abs/2003.06975
- **TACO Stats**: https://datasetninja.com/taco
- **COCO Format**: https://cocodataset.org/#format-data
- **Detectron2**: https://github.com/facebookresearch/detectron2

---

## Next Steps

1. **Immediate**: Create WorldObject class with TACO categories
2. **Short-term**: Build perception system with configurable uncertainty
3. **Medium-term**: Train real model on TACO, extract behavior stats
4. **Long-term**: Deploy TACO-trained model on real robot hardware
