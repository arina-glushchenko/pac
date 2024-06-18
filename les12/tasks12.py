import numpy as np
import torch
import torch.nn.functional as F

def im2col(image, kernel_size):
    image_rows, image_cols, image_channels = image.shape
    kernel_rows, kernel_cols = kernel_size

    cols = (image_rows - kernel_rows + 1) * (image_cols - kernel_cols + 1)

    col_matrix = np.zeros((kernel_rows * kernel_cols * image_channels, cols))

    for col in range(cols):
        start_row = col // (image_cols - kernel_cols + 1)
        start_col = col % (image_cols - kernel_cols + 1)
        end_row = start_row + kernel_rows
        end_col = start_col + kernel_cols
        col_matrix[:, col] = image[start_row:end_row, start_col:end_col, :].reshape(-1)

    return col_matrix

def convolution(image, kernel, stride=1):
    image_rows, image_cols, image_channels = image.shape
    kernel_rows, kernel_cols, _ = kernel.shape

    col_matrix = im2col(image, (kernel_rows, kernel_cols))

    output = np.dot(kernel.reshape(-1), col_matrix)

    out_rows = (image_rows - kernel_rows) // stride + 1
    out_cols = (image_cols - kernel_cols) // stride + 1

    return output.reshape(out_rows, out_cols, -1)

image = np.random.rand(4, 4, 3)
kernel = np.random.rand(2, 2, 3)
result_conv = convolution(image, kernel)
print("Custom conv")
print(result_conv)
print()

input_tensor = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float()
kernel_tensor = torch.tensor(kernel.transpose(2, 0, 1)).float().unsqueeze(0)

result_torch = F.conv2d(input_tensor, kernel_tensor)

result_torch = result_torch.squeeze(0).detach().numpy().transpose(1, 2, 0)

print("Usual conv")
print(result_torch)