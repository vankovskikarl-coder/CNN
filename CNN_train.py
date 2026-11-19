import numpy as np
import struct
from random import randint

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

with open("mnistrain", "rb") as f:
    magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
    image_data = np.fromfile(f, dtype=np.uint8, count=num_images * rows * cols)
    images = image_data.reshape(num_images, rows, cols)

with open("mnilabel", "rb") as f:
    magic, label = struct.unpack('>II', f.read(8))
    labels = np.fromfile(f, dtype=np.uint8)

with open("10k_test_images", "rb") as f:
    magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
    image_data = np.fromfile(f, dtype=np.uint8, count=num_images * rows * cols)
    images_test = image_data.reshape(num_images, rows * cols)

with open("10k_test_labels", "rb") as f:
    magic, label = struct.unpack('>II', f.read(8))
    test_labels = np.fromfile(f, dtype=np.uint8)

images = images/255.0
images_test = images_test/255.0
pib = np.zeros((60000, 10))
pib[np.arange(60000), labels] = 1
labels = pib

def ReLU(x):
    return np.maximum(x, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def conv(x, core):
    x = np.pad(x, ((0, 0), (1,1), (1, 1), (0, 0)), mode='constant')
    batch, l, w, c = x.shape
    core_t = core
    short = core.shape[1] * core.shape[2] * core.shape[3]
    z = ReLU((np.lib.stride_tricks.sliding_window_view(x, (1,3, 3, x.shape[-1])).reshape(batch, (l-2)*(w-2), short) @ core_t.transpose(2, 3, 1, 0).reshape(short, core.shape[0])))
    return z.reshape(z.shape[0],l-2,w-2, z.shape[2])

def new_conv(x, core):
    x = np.pad(x, ((0, 0), (1,1), (1, 1), (0, 0)), mode='constant')
    core_t = core[:, :, ::-1, ::-1]
    z = (np.lib.stride_tricks.sliding_window_view(x, (1, 3, 3, 64)).reshape(32, 14, 14, 576) @ core_t.transpose(2, 3, 0, 1).reshape(576, 32))
    return z


def back_conv_c2(x, core):
    x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
    batch, l, w, c = x.shape
    z = (np.lib.stride_tricks.sliding_window_view(x, (1, 14, 14, x.shape[-1])).reshape((32,3,3,14,14,32)).transpose(5, 1, 2, 0, 3, 4).reshape((32,3,3,6272)) @ core.reshape(6272, 64))
    return z.transpose(3, 0, 1, 2)

def back_conv_c1(x, core):
    x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')
    batch, l, w, c = x.shape
    z = (np.lib.stride_tricks.sliding_window_view(x, (1, 28, 28, x.shape[-1])).reshape(32, 3, 3, 28, 28, 1).transpose(5, 1, 2, 0, 3, 4).reshape(1, 3, 3, 25088) @ core.reshape(25088, 32))
    return z.transpose(3, 0, 1, 2)

def back_max_pool_p(img_origin, indi):
    batch, x, y, channel= img_origin.shape
    v = np.zeros_like(img_origin, float)
    v = v.reshape(batch, x//2, 2, y//2, 2, channel).swapaxes(2, 5).swapaxes(4, 5)
    v = v.reshape((-1, 4))
    v[np.arange(len(indi)), indi] = 1
    v = v.reshape(batch, x // 2, channel,  y // 2, 2, 2).swapaxes(4, 5).swapaxes(2, 5).reshape(batch, x, y, channel)
    return v

def max_pool_p(img):
    batch, x, y, channel = img.shape
    img = (img.reshape(batch, x//2,  2, y//2, 2, channel))
    rav = img.swapaxes(2, 5).swapaxes(4, 5).reshape(-1, 4)
    rav = np.argmax(rav, axis=1)
    return [img.max(axis=(2, 4)), rav]


INNER = 3136
HIDDEN_1 = 512
HIDDEN_2 = 512
OUT = 10
ETA = 0.0133
burst_counter = 0
cores_1 = np.array([np.random.uniform(-0.142, 0.142, size=(1, 3, 3)) for i in range(32)])
cores_2 = np.array([np.random.uniform(-0.08, 0.08, size=(32, 3, 3)) for i in range(64)])
B_h1 = np.random.uniform(-0.1, 0.1, size=(HIDDEN_1))
B_h2 = np.random.uniform(-0.1, 0.1, size=(HIDDEN_2))
B_o = np.random.uniform(-0.1, 0.1, size=(OUT))
W_o = np.random.uniform(-0.1, 0.1, size=(HIDDEN_2, OUT))
W_h1 = np.random.uniform(-0.04, 0.04, size=(INNER, HIDDEN_1))
W_h2 = np.random.uniform(-0.0765, 0.0765, size=(HIDDEN_1, HIDDEN_2))
velocity_c1 = np.zeros_like(cores_1)
velocity_c2 = np.zeros_like(cores_2)

for epoch in range(30000):
    k = randint(0, len(images) - 32)
    x, y_true = images[k:k + 32], labels[k:k + 32]
    x = x.reshape((32, 28, 28, 1))
    label_1 = conv(x, cores_1)  # (32, 28, 28, 32)
    label_2, indi_2 = max_pool_p(label_1)  # (32, 14, 14, 32)
    label_3 = conv(label_2, cores_2)  # (32, 14, 14, 64)
    label_4, indi_4 = max_pool_p(label_3)  # (32, 7, 7, 64)
    label_5 = label_4.reshape(32, INNER)
    logits_1 = (label_5 @ W_h1) + B_h1
    hidden_1 = ReLU(logits_1)
    logits_2 = (hidden_1 @ W_h2) + B_h2
    hidden_2 = ReLU(logits_2)
    predict = softmax((hidden_2 @ W_o) + B_o)
    dza = (predict - y_true)
    dzu = (logits_2 > 0).astype(float)
    dzf = dza @ W_o.T
    nabla_o = ETA * (hidden_2.T @ dza)
    nabla_bo = ETA * dza
    nabla_h2 = ETA * (hidden_1.T @ ((dzf) * dzu))
    nabla_bh2 = ETA * ((dzf) * dzu)
    nabla_bh1 = ETA * ((((dzf) * dzu) @ W_h2.T) * (logits_1 > 0).astype(float))
    nabla_h1 = label_5.T @ nabla_bh1
    dL_dlab3 = np.repeat(np.repeat((((((dzf) * dzu) @ W_h2.T) * (logits_1 > 0).astype(float)) @ W_h1.T).reshape(32, 7, 7, 64), 2,axis=1), 2, axis=2) * back_max_pool_p(label_3, indi_4) * (label_3>0)
    dL_dlab2 = np.repeat(np.repeat(new_conv(dL_dlab3, cores_2), 2, axis=1), 2, axis=2) * back_max_pool_p(label_1, indi_2) * (label_1>0)
    dl_dc1 = ETA * back_conv_c1(x, dL_dlab2)
    # velocity_c1 = 0.9 * velocity_c1 + (dl_dc1/32)
    # velocity_c2 = 0.9 * velocity_c2 + (ETA * back_conv_c2(label_2, dL_dlab3)/32)
    cores_2 -= (ETA * back_conv_c2(label_2, dL_dlab3)/32)
    cores_1 -=  (dl_dc1/32)
    W_o -= nabla_o / 32
    B_o -= np.mean(nabla_bo, axis=0)
    W_h2 -= nabla_h2 / 32
    B_h2 -= np.mean(nabla_bh2, axis=0)
    W_h1 -= nabla_h1 / 32
    B_h1 -= np.mean(nabla_bh1, axis=0)


counter = 0
for x, y_true in zip(images_test, test_labels):
    x = x.reshape((1, 28, 28, 1))
    label_1 = conv(x, cores_1)  # (1, 28, 28, 32)
    label_2, indi_2 = max_pool_p(label_1)  # (1, 14, 14, 32)
    label_3 = conv(label_2, cores_2)  # (1, 14, 14, 64)
    label_4, indi_4 = max_pool_p(label_3)  # (1, 7, 7, 64)
    label_5 = label_4.reshape(1, INNER)
    logits_1 = (label_5 @ W_h1) + B_h1
    hidden_1 = ReLU(logits_1)
    logits_2 = (hidden_1 @ W_h2) + B_h2
    hidden_2 = ReLU(logits_2)
    predict = softmax((hidden_2 @ W_o) + B_o).reshape(10)
    if np.argmax(predict) != y_true:
        counter += 1
print(counter)

np.save("B_h1.npy", B_h1)
np.save("B_h2.npy", B_h2)
np.save("W_h1.npy", W_h1)
np.save("W_h2.npy", W_h2)
np.save("W_o", W_o)
np.save("B_o", B_o)
np.save("cores_1", cores_1)
np.save("cores_2", cores_2)











