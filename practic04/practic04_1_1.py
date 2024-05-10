import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

# Parameters for the Gaussian distributions
mu0 = [0, 0] #indicating that the mean of the first variable is 0 and the mean of the second variable is also 0.
cov0 = [[1, 0], [0, 1]] #variance of 1 and there is no covariance between them.

mu1 = [4, 1]
cov1 = [[1, 0], [0, 1]]

mu2 = [2, 2]
cov2 = [[0.5, 0], [0, 0.5]]


def create_data(n_samples):
    #data for each distribution
    data_mu0 = np.random.multivariate_normal(mu0, cov0, n_samples)
    data_mu1 = np.random.multivariate_normal(mu1, cov1, n_samples)
    data_mu2 = np.random.multivariate_normal(mu2, cov2, n_samples)

    return data_mu0, data_mu1, data_mu2


def plot_data(data, title):
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8)
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def main(args):
    # Create data
    train_mu0, train_mu1, _ = create_data(args.train_samples)
    eval_mu0, eval_mu1, eval_mu2 = create_data(args.eval_samples)

    # Create numpy matrices
    data = {}
    data['X'] = np.vstack((train_mu0, train_mu1)) #stack the two training data distributions.
    data['Y_0_1'] = np.vstack((eval_mu0, eval_mu1)) #stack the mu_0 and mu_1 distributions from the evaluation set.
    data['Y_0_1_2'] = np.vstack((eval_mu0, eval_mu1, eval_mu2)) #stack the mu_0, mu_1, and mu_2 distributions from the evaluation set.

    # Create labels
    #store zeros for mu_0 and ones for mu_1 from your training dataset(data[’X ’])
    data['LX'] = np.hstack((np.zeros(args.train_samples), np.ones(args.train_samples)))

    #store zeros for mu_0 and ones for mu_1 from your first evaluation set(data[’ Y_0_1 ’])
    data['LY_0_1'] = np.hstack((np.zeros(args.eval_samples), np.ones(args.eval_samples)))

    #store zeros for mu_0 , ones for mu_1 , and twos for mu_2 from your second evaluation set ( data[’ Y_0_1_2 ’]).
    data['LY_0_1_2'] = np.hstack(
        (np.zeros(args.eval_samples), np.ones(args.eval_samples), 2 * np.ones(args.eval_samples)))

    # Saves the generated data using pickle with the specified file name. Useful tool for generating synthetic datasets for testing clustering algorithms.
    with open(args.data_name, 'wb') as fid:
        pickle.dump(data, fid)

    # Plot all data in one figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Plot training data
    axs[0].scatter(train_mu0[:, 0], train_mu0[:, 1], alpha=0.8, label='mu0')
    axs[0].scatter(train_mu1[:, 0], train_mu1[:, 1], alpha=0.8, label='mu1')
    axs[0].set_title('Training Data')
    axs[0].set_xlabel('X1')
    axs[0].set_ylabel('X2')
    axs[0].legend()

    # Plot evaluation data
    axs[1].scatter(eval_mu0[:, 0], eval_mu0[:, 1], alpha=0.8, label='mu0')
    axs[1].scatter(eval_mu1[:, 0], eval_mu1[:, 1], alpha=0.8, label='mu1')
    axs[1].scatter(eval_mu2[:, 0], eval_mu2[:, 1], alpha=0.8, label='mu2')
    axs[1].set_title('Evaluation Data')
    axs[1].set_xlabel('X1')
    axs[1].set_ylabel('X2')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save multivariate Gaussian data")
    parser.add_argument("--train_samples", type=int, default=1000, help="Number of samples for training data")
    parser.add_argument("--eval_samples", type=int, default=500, help="Number of samples for evaluation data")
    parser.add_argument("--data_name", type=str, default="data.pkl", help="Name of the pickle file to save data")
    args = parser.parse_args()

    main(args)
