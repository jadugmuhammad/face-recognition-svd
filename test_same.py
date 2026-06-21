from core.pipeline import compare
from data.loaders import att_loader

images, ids = att_loader.load(split="train")
s01_images = [img for img, sid in zip(images, ids) if sid == "s01"]

# Same person
result = compare(s01_images[0], s01_images[1])
print(f"Same person: confidence={result.confidence:.4f}, is_same={result.is_same}")

# Different person
s02_images = [img for img, sid in zip(images, ids) if sid == "s02"]
result2 = compare(s01_images[0], s02_images[0])
print(f"Diff person: confidence={result2.confidence:.4f}, is_same={result2.is_same}")
