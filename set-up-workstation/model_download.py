import torch
import timm

models_to_save = [
    # 🌍 --- GENERAL FEATURE EXTRACTORS (CNN KLASIK & EFISIEN) ---
    "resnet18",                # CNN klasik (general feature extractor, cepat, ringan)
    "resnet50",                # Lebih dalam, kuat buat domain natural/general
    "mobilenetv3_large_100",   # Super ringan, cocok buat mobile/edge deployment
    "efficientnet_b0",         # Efisien & kuat untuk dataset kecil–menengah
    "efficientnet_b3",         # Lebih akurat, cocok buat dataset fine-grained dengan resolusi lebih tinggi
    "densenet121",             # Rich feature connectivity, bagus buat transfer learning & domain non-natural
    "regnety_008",             # Balanced antara efisiensi & akurasi (modern CNN by Meta)
    
    # ⚙️ --- MODERN CNN BACKBONES (General + Fine-grained) ---
    "convnext_tiny",           # CNN modern berbasis transformer principle, bagus buat general/fine-grained
    "convnext_small",          # Versi sedang, lebih kuat untuk high-res dataset
    "convnext_base",           # Lebih besar, kuat untuk representasi kaya
    "efficientnetv2_s",        # CNN generasi baru (Google), efisien untuk training fine-grained
    
    # 🧩 --- TRANSFORMER-BASED MODELS (SEMANTIC + FINE-GRAINED) ---
    "vit_base_patch16_224",    # Vision Transformer (ViT) klasik, bagus untuk analisis semantik & fine-grained
    "deit_base_patch16_224",   # Data-efficient ViT, bagus untuk dataset kecil (general purpose transformer)
    "swin_tiny_patch4_window7_224", # Hierarchical Transformer, bagus untuk struktur spasial kompleks (scene understanding)
    "beit_base_patch16_224",   # Self-supervised ViT (BEiT), unggul dalam feature generalization & fine-grained detail
    
    # 🧠 --- ADVANCED / VISION-LANGUAGE / SELF-SUPERVISED ---
    "vit_base_patch32_clip_224",  # Vision-language (CLIP, LAION-400M), bagus untuk semantic embedding / auto-labeling
    "vit_small_patch16_224_dino", # Self-supervised (DINO), unggul untuk feature clustering & unsupervised labeling
]

for name in models_to_save:
    print(f"🔹 Downloading & saving {name}...")
    model = timm.create_model(name, pretrained=True)
    torch.save(model.state_dict(), f"model/{name}_pretrained.pth")

print("All Pretrained models saved locally!")
