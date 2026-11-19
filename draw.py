import pygame
import numpy as np

E = 2.718281828045
np.set_printoptions(linewidth=100000)

with open("W_h1.npy", "rb") as f:
    W_h1 = np.load(f)
with open("W_h2.npy", "rb") as f:
    W_h2 = np.load(f)
with open("W_o.npy", "rb") as f:
    W_o = np.load(f)
with open("B_h1.npy", "rb") as f:
    B_h1 = np.load(f)
with open("B_h2.npy", "rb") as f:
    B_h2 = np.load(f)
with open("B_o.npy", "rb") as f:
    B_o = np.load(f)
with open("cores_1.npy", "rb") as f:
    cores_1 = np.load(f)
with open("cores_2.npy", "rb") as f:
    cores_2 = np.load(f)
 
grid = np.full((28,56), -0.1)
pygame.init()
screen = pygame.display.set_mode((1230, 420))
font = pygame.font.Font(None, 36)
predfont = pygame.font.Font(None, 100)
text = font.render("Предположение: ", True, (0, 0, 0))
text1 = font.render("Не ввели число!", True, (0, 0, 0))

def getcoor_1(x, y):
    if ((x - (x % 15)) // 15)>55 or ((y - (y % 15)) // 15)>27:
        return False
    grid[(y - (y % 15)) // 15][(x - (x % 15)) // 15] = 0.75
    return (x - (x % 15)), (y - (y % 15)), 15, 15

def getcoor_2(x, y):
    if ((x - (x % 15)) // 15)>55 or ((y - (y % 15)) // 15)>27:
        return False
    grid[(y - (y % 15)) // 15][(x - (x % 15)) // 15] = 1
    return (x - (x % 15)), (y - (y % 15)), 15, 15

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    log = np.exp(x)
    return np.array([i/np.sum(log, axis=1) for i in log])

def conv(x, core):
    b, w, h, ch = x.shape
    x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
    z = ReLU(np.lib.stride_tricks.sliding_window_view(x, (1, 3, 3, ch)).reshape(b, w*h, core.shape[1]*core.shape[2]*core.shape[3]) @ core.transpose(2, 3, 1, 0).reshape(core.shape[1]*core.shape[2]*core.shape[3], core.shape[0]))
    return z.reshape(z.shape[0],h,w, z.shape[2])

def max_pool_p(img):
    batch, x, y, channel = img.shape
    img = (img.reshape(batch, x//2,  2, y//2, 2, channel))
    return img.max(axis=(2, 4))

def centerd_1(grid):
    mean = np.argmax((grid), axis=0)
    up = min(mean[mean!=0]) if len(mean[mean!=0])>0 else 0
    mean = np.argmax((np.flip(grid)), axis=0)
    down = grid.shape[0] - min(mean[mean!=0]) if len(mean[mean!=0])>0 else 0
    mean = np.argmax((grid), axis=1)
    left = min(mean[mean!=0]) if len(mean[mean!=0])>0 else 0
    mean = np.argmax((np.flip(grid)), axis=1)
    right = grid.shape[1] - min(mean[mean!=0]) if len(mean[mean!=0])>0 else 0
    grid_t = grid[up:down, left:right].copy()
    print(up, down, left, right, grid_t.shape)
    grid_r = np.pad(grid_t, (((28 - grid_t.shape[0]) // 2, (28 - grid_t.shape[0]) // 2),((28 - grid_t.shape[1]) // 2, (28 - grid_t.shape[1]) // 2)), mode="constant",constant_values=-0.1)
    return grid_r

def centerd(grid):
    mean = np.argmax((grid), axis=0)
    up = min(mean[mean!=0])-1 if len(mean[mean!=0])>0 else 0
    mean = np.argmax((np.flip(grid)), axis=0)
    down = grid.shape[0] - min(mean[mean!=0]) + 1 if len(mean[mean!=0])>0 else 0
    mean = np.argmax((grid), axis=1)
    left = min(mean[mean!=0]) if len(mean[mean!=0])>0 else 0
    mean = np.argmax((np.flip(grid)), axis=1)
    right = grid.shape[1] - min(mean[mean!=0]) if len(mean[mean!=0])>0 else 0
    grid_t = grid[up:down, left:right].copy()
    grid_r = np.pad(grid_t, (((28 - grid_t.shape[0])  // 2, (28 - grid_t.shape[0] +1)// 2),((28 - grid_t.shape[1] +1) // 2, (28 - grid_t.shape[1] ) // 2)), mode="constant",constant_values=-0.1)
    return grid_r

grid_1 = np.full((28,28), -0.1)
grid_2 = np.full((28,28), -0.1)
grid_solo = np.full((28,28), -0.1)
drawing = False
last_pos = None
p1, p2 = None, None
spl = []

while True:
    if len(spl) < 1 and (grid_solo.sum(axis = (0, 1)))<0:
        pred = font.render("Не ввели число", True, (0, 0, 0))
    elif p1!=None and p2!=None:
        pred = font.render(f"{p1*10 + p2}", True, (0, 0, 0))
    else:
        pred = font.render("Не ввели число", True, (0, 0, 0))
    pygame.display.update()
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(840, 0, 390, 420))
    screen.blit(text, (940, 100))
    screen.blit(pred, (955, 150))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                drawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            if drawing and  getcoor_2(x, y) and getcoor_1(x, y):
                x, y = event.pos
                try:
                    pygame.draw.rect(screen, (255, 255, 255), getcoor_2(x, y))
                    pygame.draw.rect(screen, (255, 255, 255), getcoor_2(x+15, y-15))
                    pygame.draw.rect(screen, (217, 217, 217), getcoor_1(x, y-15)) if screen.get_at((x, y-15)) == (0, 0, 0) else None
                    pygame.draw.rect(screen, (217, 217, 217), getcoor_1(x+15, y)) if screen.get_at((x+15, y)) == (0, 0, 0) else None
                except: IndexError
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                screen.fill((0, 0, 0))
                grid_1 = np.full((28,28), -0.1)
                grid_2 = np.full((28,28), -0.1)
                grid = np.zeros((28,56))
                grid_solo = np.full((28,28), -0.1)
                spl = []

        a = np.pad(np.any(grid>0, axis=0), 1, mode="constant", constant_values=False)
        diff = a[1:].astype(int) - a[:-1].astype(int)
        spl = np.where(diff < 0)[0]
        if len(spl)==0:
            continue
        if len(spl) == 1:
            grid_solo = centerd(grid)
            label_1 = conv(grid_solo.reshape(1, 28, 28, 1), cores_1)
            label_2 = max_pool_p(label_1)
            label_3 = conv(label_2, cores_2)
            label_4 = max_pool_p(label_3)
            label_5 = label_4.reshape(1, 3136)
            logit_1 = (label_5 @ W_h1) + B_h1
            hidden1 = ReLU(logit_1)
            logit_2 = (hidden1 @ W_h2) + B_h2
            hidden2 = ReLU(logit_2)
            logit_3 = (hidden2 @ W_o) + B_o
            predict = softmax(logit_3)
            predict = [round(float(z), 5) for z in predict[0]]
            p2 = predict.index(max(predict))
            p1 = 0
        if len(spl)==2:
            grid_1 = centerd(grid[:, :spl[0]+1])
            grid_2 = centerd(grid[:, spl[0]:])
            label_1 = conv(grid_1.reshape(1, 28, 28, 1), cores_1)
            label_2 = max_pool_p(label_1)
            label_3 = conv(label_2, cores_2)
            label_4 = max_pool_p(label_3)
            label_5 = label_4.reshape(1, 3136)
            logit_1 = (label_5 @ W_h1) + B_h1
            hidden1 = ReLU(logit_1)
            logit_2 = (hidden1 @ W_h2) + B_h2
            hidden2 = ReLU(logit_2)
            logit_3 = (hidden2 @ W_o) + B_o
            predict = softmax(logit_3)
            predict = [round(float(z), 5) for z in predict[0]]
            p1 = predict.index(max(predict))

            label_1 = conv(grid_2.reshape(1, 28, 28, 1), cores_1)
            label_2 = max_pool_p(label_1)
            label_3 = conv(label_2, cores_2)
            label_4 = max_pool_p(label_3)
            label_5 = label_4.reshape(1, 3136)
            logit_1 = (label_5 @ W_h1) + B_h1
            hidden1 = ReLU(logit_1)
            logit_2 = (hidden1 @ W_h2) + B_h2
            hidden2 = ReLU(logit_2)
            logit_3 = (hidden2 @ W_o) + B_o
            predict = softmax(logit_3)
            predict = [round(float(z), 5) for z in predict[0]]
            p2 = predict.index(max(predict))

    pygame.display.flip()


