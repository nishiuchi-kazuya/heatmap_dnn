import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# YAMLデータを読み込む
yaml_data = """
name: yellow_block
concave_vertex_pos:
  - -0.015
  - -0.015
  - 0.015
  - -0.015
  - 0.015
  - 0.015
  - 0.015
  - 0.015
  - 0.015
  - 0.015
  - -0.015
  - 0.015
convex_vertex_pos:
  - -0.015
  - -0.015
  - -0.015
  - -0.015
  - 0.015
  - -0.015
  - 0.015
  - 0.015
  - -0.015
  - 0.015
  - -0.015
  - -0.015
  - -0.015
  - -0.045
  - 0.015
  - 0.015
  - -0.045
  - 0.015
  - -0.015
  - -0.045
  - 0.045
  - 0.015
  - -0.045
  - 0.045
  - -0.015
  - 0.045
  - 0.015
  - 0.015
  - 0.045
  - 0.015
  - -0.015
  - 0.045
  - 0.045
  - 0.015
  - 0.045
  - 0.045
"""

# YAMLをパース
data = yaml.safe_load(yaml_data)
concave_vertices = np.array(data["concave_vertex_pos"]).reshape(-1, 3)
convex_vertices = np.array(data["convex_vertex_pos"]).reshape(-1, 3)

# 3Dプロット作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 凹座標（青）
ax.scatter(concave_vertices[:, 0], concave_vertices[:, 1], concave_vertices[:, 2], color='blue', label="Concave Vertices")

# 凸座標（赤）
ax.scatter(convex_vertices[:, 0], convex_vertices[:, 1], convex_vertices[:, 2], color='red', label="Convex Vertices")

# 軸ラベル
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Visualization of yellow_block")
ax.legend()
plt.show()
