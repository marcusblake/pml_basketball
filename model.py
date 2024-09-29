import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer.svi import SVI
from pyro.infer import Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
import pyro.distributions.constraints as constraints
import argparse
from typing import Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import time


DEFENSE_VECTOR_DIMENSIONS = 8
OFFENSE_VECTOR_DIMENSIONS = 9
N_CONFERENCES = 33


def get_team_ids(df):
    return df['HTeamID'].unique()


class DeepMarkovModel(nn.Module):
    """
    This is a deep markov model inspired by "Structured Inference Networks for Nonlinear State Space Models" Rahul G. Krishnan et. al. https://arxiv.org/abs/1609.09869
    """

    def __init__(self, alpha_input_size: int, beta_input_size: int, rnn_dim: int, hidden_dim: int, conferences: torch.tensor):
        super(DeepMarkovModel, self).__init__()
        self.combiner = CombinerNetwork(rnn_dim, hidden_dim)
        self.transition = TransitionFunction(rnn_dim, hidden_dim)
        self.emitter = EmitterNetwork(hidden_dim)
        self.alpha_rnn = nn.GRU(
            input_size=alpha_input_size, hidden_size=rnn_dim, batch_first=True)
        self.beta_rnn = nn.GRU(
            input_size=beta_input_size, hidden_size=rnn_dim, batch_first=True)
        self.alpha_0 = nn.Parameter(torch.zeros(hidden_dim))
        self.beta_0 = nn.Parameter(torch.zeros(hidden_dim))
        self.alpha_q_0 = nn.Parameter(torch.zeros(hidden_dim))
        self.beta_q_0 = nn.Parameter(torch.zeros(hidden_dim))
        self.alpha_h_0 = nn.Parameter(torch.zeros(rnn_dim))
        self.beta_h_0 = nn.Parameter(torch.zeros(rnn_dim))
        self.home_court_advantage = nn.Parameter(
            torch.tensor(1, dtype=torch.float))
        self.conferences = conferences

    def sample_new_game_score(self, alpha_t_1: torch.tensor, beta_t_1: torch.tensor, home_team_id: int, away_team_id: int):
        z_h_t = torch.concat(
            alpha_t_1[home_team_id], beta_t_1[away_team_id], dim=-1)
        y_h_t = pyro.sample("y_h", dist.Poisson(self.emitter(z_h_t)))

        z_a_t = torch.concat(
            alpha_t_1[away_team_id], beta_t_1[home_team_id], dim=-1)
        y_a_t = pyro.sample("y_a", dist.Poisson(self.emitter(z_a_t)))
        return y_h_t, y_a_t

    # The model is p(x_{1:T}, z_{1:T}) = p(x_{1:T} | z_{t:T}) * p(z_{1:T})
    def model(self, season_scores, offense_game_sequence_data, defense_game_sequence_data):
        """
        Defines the probablisitc model for predicting basketball scores.


        Inputs:
            season_scores - A T x |G| x 4 matrix which contains sequence of games held at a particular time step within a season. First column is score of the home team, and second column is score of the away team.
        """
        T = max(season_scores['DayNum'].unique())
        confs_for_season = self.conferences[str(
            season_scores['Season'].unique()[0])]
        pyro.module("dmm", self)
        # Start each conference strength score as 1, indicating that no conference has any advantage over another and
        # let the model learn from the data to set these conference strengths appropriately.
        conference_strength_scores = pyro.param(
            "conference_strengths", torch.ones(N_CONFERENCES), constraint=constraints.positive)
        # Start off as 1, which indicates that home court does not provide any advatange.
        home_court_advantage = pyro.param(
            "home_court_advantage", torch.tensor(1, dtype=torch.float), constraint=constraints.positive)

        n_teams = offense_game_sequence_data.size(0)

        prev_alpha = self.alpha_0.expand(n_teams, self.alpha_0.size(0))
        prev_beta = self.beta_0.expand(n_teams, self.beta_0.size(0))

        for t in pyro.markov(range(1, T)):
            games_on_day = season_scores[season_scores['DayNum'] == t]
            teams_playing_on_day = list(
                set(games_on_day['HTeamID'].unique()).union(games_on_day['ATeamID'].unique()))
            team_ids = torch.Tensor(teams_playing_on_day).type(torch.int64)
            mask = torch.zeros(n_teams).scatter_(-1, team_ids, 1.0).bool()
            # Only sample for teams that are playing on the given day.
            with pyro.plate("T"):
                with pyro.poutine.mask(mask=mask):
                    # Team's offense and defense capabilities
                    mean_alpha, loc_alpha, mean_beta, loc_beta = self.transition(
                        prev_alpha, prev_beta)
                    alpha_t = pyro.sample(
                        f"offense_{t}", dist.Normal(mean_alpha, loc_alpha).to_event(1))
                    beta_t = pyro.sample(
                        f"defense_{t}", dist.Normal(mean_beta, loc_beta).to_event(1))

            with pyro.plate("G"):
                home_ids, away_ids = torch.Tensor(games_on_day['HTeamID'].to_numpy()).int(
                ), torch.Tensor(games_on_day['ATeamID'].to_numpy()).int()
                # Concatenate the offense for the home team with the defense vector for the away team.
                z_h_t = torch.concat(
                    (alpha_t[home_ids], beta_t[away_ids]), dim=-1)
                home_conference_advantage = conference_strength_scores[confs_for_season[home_ids].int()
                                                                       ] / conference_strength_scores[confs_for_season[away_ids].int()]

                # If the game is neutral, the set the home court advantange to 1.

                # hcourt_adv = torch.tensor(n_games, home_court_advantage)
                # hcourt_adv[games_on_day['loc'] == 'N'] = 1

                pyro.sample(f"y_home_{t}", dist.Poisson(
                    home_court_advantage * home_conference_advantage * self.emitter(z_h_t)).to_event(1), obs=torch.Tensor(games_on_day['HScore'].to_numpy()))
                # Concatenate the offense for the away team with the defense vector for the home team.
                z_a_t = torch.concat(
                    (alpha_t[away_ids], beta_t[home_ids]), dim=-1)
                pyro.sample(f"y_away_{t}", dist.Poisson(
                    (1 / home_conference_advantage) * self.emitter(z_a_t)).to_event(1), obs=torch.Tensor(games_on_day['AScore'].to_numpy()))

            prev_alpha, prev_beta = alpha_t, beta_t

    # The guide is the variational distribution q(z_{1:T} | x_{1:T})
    # This is *independent* of the other team. Only the outcome of the game is dependent on the team.

    def guide(self, season_scores, offense_game_sequence_data, defense_game_sequence_data):
        T = max(season_scores['DayNum'].unique())
        pyro.module("dmm", self)

        n_teams = offense_game_sequence_data.size(0)
        h_alpha_0 = self.alpha_h_0.expand(1, n_teams, self.alpha_h_0.size(0))
        h_beta_0 = self.beta_h_0.expand(1, n_teams, self.beta_h_0.size(0))
        alpha_hidden_states, _ = self.alpha_rnn(
            offense_game_sequence_data, h_alpha_0)
        beta_hidden_states, _ = self.beta_rnn(
            defense_game_sequence_data, h_beta_0)

        prev_alpha = self.alpha_q_0.expand(n_teams, self.alpha_0.size(0))
        prev_beta = self.beta_q_0.expand(n_teams, self.beta_0.size(0))

        game_seq_count_per_team = torch.zeros(n_teams).int()
        # Feed the data into recurrent neural network to get evolution of team offensive and defensive stats in the rnn hidden states h_t
        for t in pyro.markov(range(1, T)):
            games_on_day = season_scores[season_scores['DayNum'] == t]
            teams_playing_on_day = list(
                set(games_on_day['HTeamID'].unique()).union(games_on_day['ATeamID'].unique()))
            team_ids = torch.Tensor(teams_playing_on_day).type(torch.int64)
            mask = torch.zeros(n_teams).scatter_(-1,
                                                 team_ids, 1.0).bool()
            offense_hidden = torch.stack([alpha_hidden_states[team_id, time_step]
                                          for team_id, time_step in enumerate(game_seq_count_per_team)]).float()
            defense_hidden = torch.stack([beta_hidden_states[team_id, time_step]
                                          for team_id, time_step in enumerate(game_seq_count_per_team)]).float()
            # Get all hidden states for each team, at time t-1.
            mean_alpha, loc_alpha, mean_beta, loc_beta = self.combiner(
                prev_alpha, offense_hidden, prev_beta, defense_hidden)

            # Only sample from the teams that actually have games on the current day.
            with pyro.plate("T"):
                with pyro.poutine.mask(mask=mask):
                    alpha_t = pyro.sample(
                        f"offense_{t}", dist.Normal(mean_alpha, loc_alpha).to_event(1))
                    beta_t = pyro.sample(
                        f"defense_{t}", dist.Normal(mean_beta, loc_beta).to_event(1))

            prev_alpha = alpha_t
            prev_beta = beta_t
            # These teams have played 1 game, so increment game count.
            game_seq_count_per_team[team_ids] += 1
            game_seq_count_per_team = torch.clip(
                game_seq_count_per_team, 0, offense_game_sequence_data.size(1)-1)


class TransitionFunction(nn.Module):
    """
    Parameterizes the distribution P(z_t | z_t_1) which is assumed to be Gaussian with
    a mean and a diagonal covariance. This is used in the probability model.
    """

    def __init__(self, rnn_dim: int, hidden_dim: int):
        super(TransitionFunction, self).__init__()
        # Offense latent variable
        self.gru_alpha = nn.GRUCell(hidden_dim, rnn_dim)
        self.hidden_to_mean_alpha = nn.Linear(rnn_dim, hidden_dim)
        self.hidden_to_loc_alpha = nn.Linear(rnn_dim, hidden_dim)

        # Defense latent variable
        self.gru_beta = nn.GRUCell(hidden_dim, rnn_dim)
        self.hidden_to_mean_beta = nn.Linear(rnn_dim, hidden_dim)
        self.hidden_to_loc_beta = nn.Linear(rnn_dim, hidden_dim)
        self.softplus = nn.Softplus()

    def forward(self, alpha_t_1: torch.tensor, beta_t_1: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        h_t = self.gru_alpha(alpha_t_1)
        mean_alpha = self.hidden_to_mean_alpha(h_t)
        loc_alpha = self.softplus(self.hidden_to_loc_alpha(h_t))

        h_t = self.gru_beta(beta_t_1)
        mean_beta = self.hidden_to_mean_beta(h_t)
        loc_beta = self.softplus(self.hidden_to_loc_beta(h_t))
        return mean_alpha, loc_alpha, mean_beta, loc_beta


class EmitterNetwork(nn.Module):
    """
    Neural network which is a function mapping from a latent space representation to a real number
    that parameterizes a Poisson distribution. In other words, if f: R^d -> R is our emitter function,
    P(y_t | z_t) ~ Poisson(f(z_t)).
    """

    def __init__(self, hidden_dim: int):
        super(EmitterNetwork, self).__init__()
        # Takes in offense and defense vectors concatenated.
        hidden_dim *= 2
        self.regression_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            # Softplus at the end since we want a positive value.
            nn.Softplus()
        )

    def forward(self, z_t: torch.tensor) -> torch.tensor:
        return self.regression_network(z_t).float()


class CombinerNetwork(nn.Module):
    """
    Neural network which takes in a latent vector at time step t-1 and produces a mean + co-variance matrix
    which parameterazies a normal distribution for the latent vector at time step t. In other words,
    P(z_t| z_{t-1}, x_{t}, ..., x_{1}) ~ Normal(). This is used for approximating the variational family.
    """

    def __init__(self, rnn_dim: int, hidden_dim: int):
        super(CombinerNetwork, self).__init__()
        self.alpha_rnn_to_hidden = nn.Linear(hidden_dim, rnn_dim)
        self.alpha_hidden_to_mean = nn.Linear(rnn_dim, hidden_dim)
        self.alpha_hidden_to_loc = nn.Linear(rnn_dim, hidden_dim)

        self.beta_rnn_to_hidden = nn.Linear(hidden_dim, rnn_dim)
        self.beta_hidden_to_mean = nn.Linear(rnn_dim, hidden_dim)
        self.beta_hidden_to_loc = nn.Linear(rnn_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, alpha_t_1: torch.tensor, h_alpha_t_1: torch.tensor, beta_t_1: torch.tensor, h_beta_t_1: torch.tensor) -> torch.tensor:
        z_alpha_combined = 0.5 * \
            (self.tanh(self.alpha_rnn_to_hidden(alpha_t_1)) + h_alpha_t_1)
        z_alpha_mean = self.beta_hidden_to_mean(z_alpha_combined)
        z_alpha_loc = self.softplus(self.beta_hidden_to_loc(z_alpha_combined))

        z_beta_combined = 0.5 * \
            (self.tanh(self.beta_rnn_to_hidden(beta_t_1)) + h_beta_t_1)
        z_beta_mean = self.beta_hidden_to_mean(z_beta_combined)
        z_beta_loc = self.softplus(self.beta_hidden_to_loc(z_beta_combined))
        return z_alpha_mean, z_alpha_loc, z_beta_mean, z_beta_loc


class Plotter:
    def __init__(self):
        plt.ion()
        self.iterations = []
        self.elbo = []
        # # creating subplot and figure
        self.fig, self.axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

    def update_plot(self, elbo: float):
        self.iterations.append(len(self.iterations))
        self.elbo.append(elbo)
        self._clear_axes()
        self.axes.set_xlabel('Iterations')
        self.axes.set_ylabel('ELBO Loss')
        self.axes.set_title('ELBO Loss vs Iteration')
        self.axes.plot(self.iterations, self.elbo)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def final_plot(self):
        self._clear_axes()
        self.axes.set_xlabel('Iterations')
        self.axes.set_ylabel('ELBO Loss')
        self.axes.set_title('ELBO Loss vs Iteration')
        self.axes.plot(self.iterations, self.elbo)

    def _clear_axes(self):
        self.axes.clear()


def train(args):
    conferences = torch.load("./conferences.pt")
    dmm = DeepMarkovModel(OFFENSE_VECTOR_DIMENSIONS, DEFENSE_VECTOR_DIMENSIONS,
                          args.rnn_dim, args.hidden_dim, conferences)
    optimizer = ClippedAdam({'lr': args.learning_rate})
    svi = SVI(dmm.model, dmm.guide, optimizer, loss=Trace_ELBO())

    train_scores, train_offense_sequences, train_defense_sequences = pd.read_csv(
        "./team_scores_train.csv"), torch.load("train_offense_sequence_data.pt"), torch.load("train_defense_sequence_data.pt")
    test_scores, test_offense_sequences, test_defense_sequences = pd.read_csv(
        "./team_scores_test.csv"), torch.load("test_offense_sequence_data.pt"), torch.load("test_defense_sequence_data.pt")

    train_seasons = train_scores['Season'].unique()
    test_seasons = test_scores['Season'].unique()
    plotter = Plotter()
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        dmm.train()
        train_loss = 0.
        print(f'Epoch {epoch}:')
        # Batch by season
        for season in train_seasons:
            print('season ', season)
            offense_sequence = train_offense_sequences[str(
                season)]
            defense_sequence = train_defense_sequences[str(season)]
            train_loss += svi.step(train_scores[train_scores['Season'] == season].reset_index(
                drop=True), offense_sequence, defense_sequence)
        print('Train loss: {:.2f}'.format(train_loss))

        if epoch % args.eval_frequency == 0:
            dmm.eval()
            test_loss = 0.
            for season in test_seasons:
                offense_sequence = test_offense_sequences[str(
                    season)]
                defense_sequence = test_defense_sequences[str(season)]
                test_loss += svi.evaluate_loss(test_scores[test_scores['Season'] == season],
                                               offense_sequence, defense_sequence)
            print('Home court coefficient',
                  pyro.get_param_store().get_param("home_court_advantage"))
            print('Conference strengths',
                  pyro.get_param_store().get_param("conference_strengths"))
            print('Test loss {:.2f}'.format(test_loss))

        if epoch % args.save_frequency == 0:
            file_suffix = f"""lr={args.learning_rate}_rnnsize={
                args.rnn_dim}_hidden={args.hidden_dim}_epoch={epoch}"""
            torch.save(dmm.state_dict(), f"dmm_{file_suffix}.pt")
            optimizer.save(f"optimizer_state_{file_suffix}.pt")

        plotter.update_plot(train_loss)
        print(f'Epoch time {time.time() - start_time}')

    print('Training complete')
    print('Saving final model...')
    file_suffix = f"""lr={args.learning_rate}_rnnsize={
        args.rnn_dim}_hidden={args.hidden_dim}_done"""
    torch.save(dmm.state_dict(), f"dmm_{file_suffix}.pt")
    optimizer.save(f"optimizer_state_{file_suffix}.pt")
    print('Model saved.')
    plotter.final_plot()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train', help='Whether to train the model', action='store_true')
    parser.add_argument(
        '--infer', help='Whether to run inference', action='store_true')
    parser.add_argument(
        '--render_model', help='Whether to just render the model', action='store_true')
    parser.add_argument(
        '--rnn_dim', help='Defines the number of dimensions for the rnn hidden state.', type=int, default=50)
    parser.add_argument(
        '--hidden_dim', help='Defines the number of hidden dimensions for the model.', type=int, default=50)
    parser.add_argument(
        '--epochs', help='Number of epochs to run the model for.', type=int, default=20)
    parser.add_argument(
        '--learning_rate', help='Learning rate to use while training', type=float, default=0.1)
    parser.add_argument(
        '--eval_frequency', help='Frequency at which to evaluate the model', type=int, default=10
    )
    parser.add_argument(
        '--save_frequency', help='Frequency at which to save the model', type=int, default=10)
    parser.add_argument('--saved_model', help='Filepath to saved model')

    args = parser.parse_args()
    if args.render_model:
        game_scores = pd.read_csv("./processed_regular_season_stats.csv")
        conferences = torch.load("./conferences.pt")
        print(conferences)
        model = DeepMarkovModel(OFFENSE_VECTOR_DIMENSIONS, DEFENSE_VECTOR_DIMENSIONS,
                                args.rnn_dim, args.hidden_dim, conferences)
        pyro.render_model(model.model, model_args=(
            game_scores[game_scores['Season'] == 2019], torch.load("train_offense_sequence_data.pt")["2019"], None), render_distributions=True, filename="model.png")
    elif args.train:
        train(args)
    else:
        print('Exiting.')


if __name__ == '__main__':
    main()
