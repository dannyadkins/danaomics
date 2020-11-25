import matplotlib.pylab as plt
import random


def visualize_input(raw_train, pred_cols):
    fig, axs = plt.subplots(5, 1,
                            figsize=(10, 6),
                            sharex=True)
    axs = axs.flatten()
    for i, col in enumerate(pred_cols):
        raw_train['mean_' + col] = raw_train[col].apply(lambda x: np.mean(x))
        raw_train['mean_' + col] \
            .plot(kind='hist',
                  bins=50,
                  title='Distribution of average ' + col,
                  color=(random.random(), random.random(), random.random()),
                  ax=axs[i])
    plt.tight_layout()
    plt.show()
