import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

# Define parameters for the Gaussian distributions
mu0 = [0, 0]
cov0 = [[1, 0], [0, 1]]

mu1 = [4, 1]
cov1 = [[1, 0], [0, 1]]

mu2 = [2, 2]
cov2 = [[0.5, 0], [0, 0.5]]


def create_data(n_samples):
    # Generate data for each distribution
    data_mu0 = np.random.multivariate_normal(mu0, cov0, n_samples)
    data_mu1 = np.random.multivariate_normal(mu1, cov1, n_samples)
    data_mu2 = np.random.multivariate_normal(mu2, cov2, n_samples)

    return data_mu0, data_mu1, data_mu2


def plot_data(train_data, eval_data, title):
    plt.scatter(train_data[:, 0], train_data[:, 1], c='blue', label='Training Data', alpha=0.5)
    plt.scatter(eval_data[:, 0], eval_data[:, 1], c='red', label='Evaluation Data', alpha=0.5)
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


def main(args):
    # Create data
    train_mu0, train_mu1, _ = create_data(args.train_samples)
    eval_mu0, eval_mu1, eval_mu2 = create_data(args.eval_samples)

    # Plot training and evaluation data
    plot_data(np.vstack((train_mu0, train_mu1)), np.vstack((eval_mu0, eval_mu1, eval_mu2)), 'Multivariate Gaussian Data')

    # Create numpy matrices
    data = {}
    data['X'] = np.vstack((train_mu0, train_mu1))
    data['Y_0_1'] = np.vstack((eval_mu0, eval_mu1))
    data['Y_0_1_2'] = np.vstack((eval_mu0, eval_mu1, eval_mu2))

    # Create labels
    data['LX'] = np.hstack((np.zeros(args.train_samples), np.ones(args.train_samples)))
    data['LY_0_1'] = np.hstack((np.zeros(args.eval_samples), np.ones(args.eval_samples)))
    data['LY_0_1_2'] = np.hstack(
        (np.zeros(args.eval_samples), np.ones(args.eval_samples), 2 * np.ones(args.eval_samples)))

    # Save data using pickle
    with open(args.data_name, 'wb') as fid:
        pickle.dump(data, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save multivariate Gaussian data")
    parser.add_argument("--train_samples", type=int, default=1000, help="Number of samples for training data")
    parser.add_argument("--eval_samples", type=int, default=500, help="Number of samples for evaluation data")
    parser.add_argument("--data_name", type=str, default="data.pkl", help="Name of the pickle file to save data")
    args = parser.parse_args()

    main(args)
