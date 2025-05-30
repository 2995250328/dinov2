import torch
 
import torchvision.transforms as T
 
import matplotlib.pyplot as plt
 
from PIL import Image
 
from sklearn.decomposition import PCA
from dinov2.models import build_model
from dinov2.layers.attention import Attention, MemEffAttention

patch_h = 40
patch_w = 40
feat_dim = 384 # vits14
 
transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14, patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
 
dinov2_vitb14 = torch.hub.load('', 'dinov2_vits14',source='local').cuda()
# 2. 找到最后一个 Block 的 Attention 模块
last_block = dinov2_vitb14.blocks[-1]
last_attention_module = last_block.attn
if isinstance(last_attention_module, Attention) and not isinstance(last_attention_module, MemEffAttention):
    print(f"最后一个块的注意力模块是：{last_attention_module.__class__.__name__} (Attention Class)")
 
 
# extract features
features = torch.zeros(4, patch_h * patch_w, feat_dim)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).cuda()
 
 
img_path = f'overview.jpg'#输入图片路径
img = Image.open(img_path).convert('RGB')
imgs_tensor[0] = transform(img)[:3]
with torch.no_grad():
    features_dict = dinov2_vitb14.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']
 
print(features.shape)
 
features = features.reshape(4 * patch_h * patch_w, feat_dim).cpu()

last_block_attention_map = last_attention_module.attn_weights

if last_block_attention_map is not None:
    print("\n成功获取最后一个块的注意力矩阵！")
    print(f"注意力矩阵的形状: {last_block_attention_map.shape}")
    # 注意力矩阵的形状通常是 (Batch_size, Num_Heads, Sequence_Length, Sequence_Length)
    # Sequence_Length = 1 (cls_token) + Num_Patches (例如 14x14=196) + Num_Register_Tokens (如果存在)
    # 例如：(1, 12, 197, 197) 对于 ViT-Base 模型，无 register tokens

    # 你现在可以使用 last_block_attention_map 进行可视化或分析
else:
    print("\n未能获取注意力矩阵。可能是因为最后一个块不是 Attention 类，或者计算未成功。")
aggregated_attention_map = last_block_attention_map.mean(dim=1)
print(aggregated_attention_map.shape)
# 你现在可以像之前那样，对聚合后的矩阵进行可视化或数值检查。
# 例如，选择第一个批次进行可视化：
import seaborn as sns

selected_batch_idx = 0
attention_for_plot = aggregated_attention_map[selected_batch_idx, :, :].cpu().numpy()

plot_size = 500
attention_subset = attention_for_plot[400:plot_size, 400:plot_size]

plt.figure(figsize=(8, 7))
sns.heatmap(attention_subset, cmap='viridis', cbar=True, square=True)
plt.title(f'Batch {selected_batch_idx} Aggregated Attention Map (Subset {plot_size}x{plot_size})')
plt.xlabel('Key Sequence Index')
plt.ylabel('Query Sequence Index')
plt.show()
 
# pca = PCA(n_components=3)
# pca.fit(features)
# pca_features = pca.transform(features)
 
# # visualize PCA components for finding a proper threshold
# plt.subplot(1, 3, 1)
# plt.hist(pca_features[:, 0])
# plt.subplot(1, 3, 2)
# plt.hist(pca_features[:, 1])
# plt.subplot(1, 3, 3)
# plt.hist(pca_features[:, 2])
# plt.show()
# plt.close()
 
# # uncomment below to plot the first pca component
# # pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())
# # for i in range(4):
# #     plt.subplot(2, 2, i+1)
# #     plt.imshow(pca_features[i * patch_h * patch_w: (i+1) * patch_h * patch_w, 0].reshape(patch_h, patch_w))
# # plt.show()
# # plt.close()
 
# # segment using the first component
# pca_features_bg = pca_features[:, 0] < 10
# pca_features_fg = ~pca_features_bg
 
# # plot the pca_features_bg
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(pca_features_bg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w))
# plt.show()
 
# # PCA for only foreground patches
# pca_features_rem = pca.transform(features[pca_features_fg])
# for i in range(3):
#     # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
#     # transform using mean and std, I personally found this transformation gives a better visualization
#     pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5
 
# pca_features_rgb = pca_features.copy()
# pca_features_rgb[pca_features_bg] = 0
# pca_features_rgb[pca_features_fg] = pca_features_rem
 
# pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(pca_features_rgb[i][..., ::-1])
# plt.savefig('features.png')#保存结果图片
# plt.show()
# plt.close()