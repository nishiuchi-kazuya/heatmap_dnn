import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# YAMLファイルの読み込み
with open("./test/annotation.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)



# 最初のブロックを取得
block = data["blocks"][0]

# 凹凸頂点の取得
convex_vertices = np.array(block["convex_vertex_pos"]).reshape(-1, 3)
concave_vertices = np.array(block["concave_vertex_pos"]).reshape(-1, 3)

# 3Dプロット設定
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 凸頂点 (赤) と 凹頂点 (青) をプロット
ax.scatter(convex_vertices[:, 0], convex_vertices[:, 1], convex_vertices[:, 2], color='red', label="Convex")
ax.scatter(concave_vertices[:, 0], concave_vertices[:, 1], concave_vertices[:, 2], color='blue', label="Concave")

# 凸・凹頂点をつなぐラインを作成
edges = [(convex_vertices[i], convex_vertices[i+1]) for i in range(len(convex_vertices)-1)]
edges += [(concave_vertices[i], concave_vertices[i+1]) for i in range(len(concave_vertices)-1)]
edge_lines = [[list(p1), list(p2)] for p1, p2 in edges]

# ラインをプロット
ax.add_collection3d(Line3DCollection(edge_lines, colors='gray', linewidths=1))

# 軸ラベル
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 表示
plt.legend()




plt.savefig("output.png")

