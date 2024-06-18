import numpy as np

# создаем случайную картинку
h, w = 10, 10
img = np.random.randint(0, 256, size=(h, w)).astype(np.uint8)


unique_colors = np.unique(img)

print("Количество уникальных цветов:", len(unique_colors))