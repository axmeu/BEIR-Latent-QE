import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


def apply_rocchio(query_vec, doc_vectors, alpha=1.0, beta=0.75, gamma=0.25, expansion_nb=30):
    sim_scores = cosine_similarity(query_vec, doc_vectors)[0]

    top_indices_pos = np.argsort(sim_scores)[::-1][:expansion_nb]
    mean_pos = doc_vectors[top_indices_pos].mean(axis=0).reshape(1, -1)

    bottom_indices_neg = np.argsort(sim_scores)[:expansion_nb]
    mean_neg = doc_vectors[bottom_indices_neg].mean(axis=0).reshape(1, -1)

    return alpha * query_vec + beta * mean_pos - gamma * mean_neg


def compute_explained_variance(tfidf_matrix, max_components=1000):
    n_comp = min(max_components, tfidf_matrix.shape[1]-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=1)
    svd.fit(tfidf_matrix)
    explained_cumsum = np.cumsum(svd.explained_variance_ratio_)
    return explained_cumsum

def select_k_by_variance(explained_cumsum, tau=0.9, min_k=100, max_k=700):
    k = np.searchsorted(explained_cumsum, tau) + 1
    k = min(max(k, min_k), max_k)
    return k, explained_cumsum


def select_k_by_elbow(explained_cumsum):
    x1, y1 = 0, explained_cumsum[0]
    x2, y2 = len(explained_cumsum)-1, explained_cumsum[-1]

    distances = []
    for i, y in enumerate(explained_cumsum):
        d = np.abs((y2 - y1)*i - (x2 - x1)*(y - y1)) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(d)

    k_elbow = np.argmax(distances) + 1
    return k_elbow, explained_cumsum


def plot_explained_variance(explained_cumsum, k=None, method_name=""):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(explained_cumsum)+1), explained_cumsum, marker='o', markersize=3, linewidth=1.2)
    if k is not None:
        plt.axvline(x=k, color='red', linestyle='--', linewidth=1)
        plt.scatter(k, explained_cumsum[k-1], color='red', marker='x', s=80, label=f"k = {k}")
        plt.legend()
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulative")
    title = "Sélection de k"
    if method_name:
        title += f" ({method_name})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def animate_elbow(explained, k_elbow, save_path="elbow.gif"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(explained)+1)
    y = explained

    curve_line, = ax.plot([], [], marker='o', linewidth=1.2, markersize=3, color='blue', label="Cumulative variance")
    ref_line, = ax.plot([], [], linestyle='--', color='green', linewidth=1, label="Reference line")
    moving_point, = ax.plot([], [], 'ro', markersize=5)
    info_text = ax.text(0.95, 0.02, "", transform=ax.transAxes,
                        fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    ax.set_xlim(0, len(x)+1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Elbow method animation")
    ax.grid(True, alpha=0.3)

    x1, y1 = 0, y[0]
    x2, y2 = len(x)-1, y[-1]

    idx_max = k_elbow - 1

    frames_per_step = 40  # 2 sec / étape à 20 fps
    total_frames = frames_per_step * 4

    def update(frame):
        step = frame // frames_per_step
        step_frame = frame % frames_per_step

        if step == 0:  # Courbe
            n = int((step_frame+1)/frames_per_step*len(x))
            curve_line.set_data(x[:n], y[:n])
            info_text.set_text("Cumulative variance")
        elif step == 1:  # Ligne de référence
            t = (step_frame+1)/frames_per_step
            ref_line.set_data([x1, x1 + t*(x2 - x1)], [y1, y1 + t*(y2 - y1)])
            info_text.set_text("Reference line")
        elif step == 2:  # Point qui se déplace
            idx = int((step_frame+1)/frames_per_step*len(x)) - 1
            moving_point.set_data([x[idx]], [y[idx]])
            info_text.set_text("Computing orthogonal distances")
        else:  # Résultat final
            moving_point.set_data([], [])
            curve_line.set_data(x, y)
            ref_line.set_data([x1, x2], [y1, y2])
            ax.scatter([x[idx_max]], [y[idx_max]], color='red', marker='x', s=100, zorder=5, label=f"k = {idx_max+1}")
            info_text.set_text(f"Elbow found: k = {idx_max+1}")

        return curve_line, ref_line, moving_point, info_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True, repeat=False)  # 50ms → 20fps
    ani.save(save_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"[INFO] Animation saved: {save_path}")





