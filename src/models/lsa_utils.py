import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

def apply_rocchio(
    query_vec,        # shape (1, k)
    doc_vectors,      # shape (N, k)
    window=100,       # local window
    posneg=5,         # number of pseudo positives and negatives
    alpha=1.0,        # original query weight
    beta=0.8,         # positives
    gamma=0.1):       # negatives  

    q = query_vec.ravel()

    #cosinus similarity
    scores = doc_vectors @ q

    #top index
    window = min(window, scores.shape[0])
    topwindow_idx = np.argpartition(scores, -window)[-window:]
    topwindow_idx = topwindow_idx[np.argsort(scores[topwindow_idx])[::-1]]

    #get positive and negative feedback
    posneg = min(posneg, len(topwindow_idx) // 2)
    if posneg == 0:
        return query_vec  #not enough doc

    pos_pool = topwindow_idx[:posneg]
    neg_pool = topwindow_idx[posneg:]
    if len(neg_pool) < posneg:
        p_neg = len(neg_pool)
    else:
        p_neg = posneg
    neg_idx = neg_pool[:p_neg] if p_neg > 0 else None

    #normalized centroids
    mean_pos = doc_vectors[pos_pool].mean(axis=0, keepdims=True)
    mean_pos = normalize(mean_pos, norm="l2")

    if neg_idx is not None and p_neg > 0:
        mean_neg = doc_vectors[neg_idx].mean(axis=0, keepdims=True)
        mean_neg = normalize(mean_neg, norm="l2")
    else:
        mean_neg = np.zeros_like(mean_pos)

    
    q_prime = alpha * query_vec + beta * mean_pos - gamma * mean_neg
    q_prime = normalize(q_prime, norm="l2")
    return q_prime


def compute_explained_variance(tfidf_matrix, max_components=1000):
    n_comp = min(max_components, tfidf_matrix.shape[1]-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=1)
    svd.fit(tfidf_matrix)
    explained_cumsum = np.cumsum(svd.explained_variance_ratio_)
    return explained_cumsum


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
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    title = "k selection by elbow method"
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
    ax.set_title("Components selection by 'elbow'")
    ax.grid(True, alpha=0.3)

    x1, y1 = 0, y[0]
    x2, y2 = len(x)-1, y[-1]

    idx_max = k_elbow - 1

    frames_per_step = 40
    total_frames = frames_per_step * 4

    def update(frame):
        step = frame // frames_per_step
        step_frame = frame % frames_per_step

        if step == 0:  # Curve
            n = int((step_frame+1)/frames_per_step*len(x))
            curve_line.set_data(x[:n], y[:n])
            info_text.set_text("Cumulative variance")
        elif step == 1:  # Reference line
            t = (step_frame+1)/frames_per_step
            ref_line.set_data([x1, x1 + t*(x2 - x1)], [y1, y1 + t*(y2 - y1)])
            info_text.set_text("Reference line")
        elif step == 2:  # Orthogonal distance
            idx = int((step_frame+1)/frames_per_step*len(x)) - 1
            moving_point.set_data([x[idx]], [y[idx]])
            info_text.set_text("Computing orthogonal distances")
        else:  # Final result
            moving_point.set_data([], [])
            curve_line.set_data(x, y)
            ref_line.set_data([x1, x2], [y1, y2])
            ax.scatter([x[idx_max]], [y[idx_max]], color='red', marker='x', s=100, zorder=5, label=f"k = {idx_max+1}")
            info_text.set_text(f"k value: {idx_max+1}")

        return curve_line, ref_line, moving_point, info_text

    ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True, repeat=False)  # 50ms → 20fps
    ani.save(save_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"GIF saved: {save_path}")





